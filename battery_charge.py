"""battery_charge.py — 模組 3：電池充電與 SOC 動態（模組 3、4 共用的 SOC 唯一來源）。

職責（對應 `03_電池與充放電模型`）：
- `BatteryState` 與 SOC 離散動態方程（見 `03` §1）。
- `charge_step`：對電池充電一個時步，受 SOC 上限、最大充電功率、充電效率夾限（見 `03` §2、§4）。
- 充電來源優先序：太陽能餘電 > 離峰市電（見 `03` §3）→ `charge_prioritized`。
- 放電的 SOC 數學輔助（`max_dischargeable_kw`、`apply_discharge`），供模組 4 的 `discharge_step` 呼叫，
  讓 SOC 動態維持「單一來源」。
- 電池衰減成本（吞吐量法，見 `03` §5）。

相依方向：本檔 import 同層/更低層 `electrical_base` 與 `config`，不 import 更高層（無循環 import）。
所有可調數值來自 `config.py`；單位標註於變數名。

模型假設（清楚標明，便於日後修改）：
- η_ch（BATT_CHARGE_EFF）= 「外部供給功率 → 存入電芯能量」的充電效率。
- η_dis（BATT_DISCHARGE_EFF）= 「電芯放出 → 對外可用功率」的放電效率（見 `03` §1）。
- 耦合：DC 耦合不額外乘；AC 耦合「充電路徑」再乘 batt_inverter_eff（交流經電池逆變器整流，見 `02` §3）。
  放電端依 `03` 慣例僅以 η_dis 表示（如需 AC 放電逆變器損耗，於此處調整即可）。
- 充電曲線：初版採「定功率充電」（受最大充電功率與剩餘容量夾限）。
  進階 CC-CV 尾段降功率的擴充點見 `max_chargeable_kw` 內標註。
"""

from __future__ import annotations

from dataclasses import dataclass

import electrical_base as eb
from config import AppConfig, load_config

# 合法充電來源（見 `03` §3）。
VALID_SOURCES: frozenset[str] = frozenset({"pv", "grid"})

_MINUTES_PER_HOUR: float = 60.0

__all__ = [
    "BatteryState",
    "charge_step",
    "charge_prioritized",
    "max_chargeable_kw",
    "max_dischargeable_kw",
    "apply_discharge",
    "degradation_cost",
]


@dataclass
class BatteryState:
    """電池目前狀態。

    屬性:
        soc: 目前 State of Charge（0~1，無單位比例）。
    """

    soc: float


# --------------------------------------------------------------------------- #
# 路徑效率（含耦合架構）
# --------------------------------------------------------------------------- #
def _charge_path_eff(cfg: AppConfig) -> float:
    """充電路徑總效率 = η_ch ×（耦合額外因子）。

    參數:
        cfg: 設定物件。
    回傳:
        充電總效率（0~1）：DC=η_ch；AC=η_ch×batt_inverter_eff。
    """
    return cfg.batt_charge_eff * eb.coupling_extra_eff(cfg)


def _discharge_path_eff(cfg: AppConfig) -> float:
    """放電路徑總效率（依 `03` 慣例為 η_dis）。

    參數:
        cfg: 設定物件。
    回傳:
        放電總效率（0~1）。
    """
    return cfg.batt_discharge_eff


def _dt_hours(minutes: float) -> float:
    """時步分鐘換算小時。"""
    return minutes / _MINUTES_PER_HOUR


# --------------------------------------------------------------------------- #
# 充電
# --------------------------------------------------------------------------- #
def max_chargeable_kw(state: BatteryState, cfg: AppConfig, minutes: float) -> float:
    """此時步可吸收的最大「外部充電功率」（kW），取以下兩者較小值：

        1. 最大充電功率上限 BATT_MAX_CHARGE_KW。
        2. 受「離 SOC_max 剩餘空間」限制，能在本時步填滿的外部功率
           = (SOC_max − SOC) × E_cap /（充電效率 × Δt）。

    參數:
        state: 目前電池狀態。
        cfg: 設定物件。
        minutes: 時步長度（分鐘）。
    回傳:
        最大可充外部功率（kW，≥0）。

    進階擴充點（CC-CV 尾段）：若要近似定電壓尾段，可在此依 SOC 門檻
        （如 > 0.9）對回傳值乘上線性遞減係數。初版採定功率，不降載。
    """
    dt_h = _dt_hours(minutes)
    eff = _charge_path_eff(cfg)
    headroom_kwh = max(0.0, cfg.batt_soc_max - state.soc) * cfg.batt_capacity_kwh
    by_headroom_kw = headroom_kwh / (eff * dt_h) if (eff > 0 and dt_h > 0) else 0.0
    return max(0.0, min(cfg.batt_max_charge_kw, by_headroom_kw))


def charge_step(
    state: BatteryState,
    available_kw: float,
    source: str,
    cfg: AppConfig,
    minutes: float,
) -> tuple[float, BatteryState]:
    """對電池充電一個時步（定功率），更新 SOC（見 `03` §1、§2）。

    SOC 動態（僅充電項）:
        SOC(t+1) = SOC(t) + 充電效率 × P_ch × Δt / E_cap
    其中 P_ch 為「實際吸收的外部功率」，受 `max_chargeable_kw` 夾限。

    參數:
        state: 目前電池狀態。
        available_kw: 外部可供充電功率（kW，<0 視為 0）。
        source: 充電來源，須 ∈ {"pv","grid"}（供上層記帳；物理效率相同）。
        cfg: 設定物件。
        minutes: 時步長度（分鐘）。
    回傳:
        (實際吸收功率 kW, 新的 BatteryState)。
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"source 須為 {sorted(VALID_SOURCES)} 之一，收到：{source!r}")

    dt_h = _dt_hours(minutes)
    eff = _charge_path_eff(cfg)
    actual_kw = min(max(0.0, available_kw), max_chargeable_kw(state, cfg, minutes))

    delta_soc = eff * actual_kw * dt_h / cfg.batt_capacity_kwh
    new_soc = min(cfg.batt_soc_max, state.soc + delta_soc)
    return actual_kw, BatteryState(soc=new_soc)


def charge_prioritized(
    state: BatteryState,
    pv_surplus_kw: float,
    grid_kw: float,
    cfg: AppConfig,
    minutes: float,
) -> dict[str, float | BatteryState]:
    """依充電來源優先序充電：先吃太陽能餘電，再用市電補（見 `03` §3）。

    太陽能餘電邊際成本≈0，最優先；市電充電則由上層（模組 4）決定是否提供
    （僅離峰且預測太陽能不足以撐過尖峰時）。本函式只負責「給定兩來源後的優先充電」。

    參數:
        state: 目前電池狀態。
        pv_surplus_kw: 可用的太陽能餘電功率（kW）。
        grid_kw: 願意自市電提供的充電功率（kW；上層未決定充電時給 0）。
        cfg: 設定物件。
        minutes: 時步長度（分鐘）。
    回傳:
        dict：
          pv_to_batt_kw、grid_to_batt_kw、batt_ch_kw（總充電功率）、state（新狀態）。
    """
    pv_actual_kw, state_after_pv = charge_step(state, pv_surplus_kw, "pv", cfg, minutes)
    # 最大充電功率為「總和」上限：市電只能用 PV 用剩的功率額度，且再受剩餘 SOC 空間限制。
    remaining_power_kw = max(0.0, cfg.batt_max_charge_kw - pv_actual_kw)
    grid_offer_kw = min(max(0.0, grid_kw), remaining_power_kw)
    grid_actual_kw, state_after_grid = charge_step(state_after_pv, grid_offer_kw, "grid", cfg, minutes)
    return {
        "pv_to_batt_kw": pv_actual_kw,
        "grid_to_batt_kw": grid_actual_kw,
        "batt_ch_kw": pv_actual_kw + grid_actual_kw,
        "state": state_after_grid,
    }


# --------------------------------------------------------------------------- #
# 放電的 SOC 輔助（供模組 4 的 discharge_step 使用；SOC 數學集中於此）
# --------------------------------------------------------------------------- #
def max_dischargeable_kw(
    state: BatteryState,
    cfg: AppConfig,
    minutes: float,
    floor_soc: float | None = None,
) -> float:
    """此時步可「對外輸出」的最大放電功率（kW），取以下兩者較小值：

        1. 最大放電功率上限 BATT_MAX_DISCHARGE_KW。
        2. 受「離下限可放電量」限制：
           對外可用功率 = (SOC − floor) × E_cap × 放電效率 / Δt。

    參數:
        state: 目前電池狀態。
        cfg: 設定物件。
        minutes: 時步長度（分鐘）。
        floor_soc: 放電下限 SOC；None → 用 BATT_SOC_MIN。
                   智慧放電可傳入較高的下限以「為尖峰保留電量」。
    回傳:
        最大對外放電功率（kW，≥0）。
    """
    dt_h = _dt_hours(minutes)
    eff = _discharge_path_eff(cfg)
    floor = cfg.batt_soc_min if floor_soc is None else max(cfg.batt_soc_min, floor_soc)
    avail_cell_kwh = max(0.0, state.soc - floor) * cfg.batt_capacity_kwh
    by_soc_kw = avail_cell_kwh * eff / dt_h if dt_h > 0 else 0.0
    return max(0.0, min(cfg.batt_max_discharge_kw, by_soc_kw))


def apply_discharge(
    state: BatteryState,
    output_kw: float,
    cfg: AppConfig,
    minutes: float,
) -> BatteryState:
    """套用一次放電對 SOC 的影響（見 `03` §1 的放電項）。

    SOC 動態（僅放電項）:
        SOC(t+1) = SOC(t) − (P_dis / 放電效率) × Δt / E_cap
    其中 P_dis 為「對外輸出功率」（呼叫端應先以 `max_dischargeable_kw` 夾限）。

    參數:
        state: 目前電池狀態。
        output_kw: 對外輸出功率（kW，<0 視為 0）。
        cfg: 設定物件。
        minutes: 時步長度（分鐘）。
    回傳:
        新的 BatteryState（SOC 不會低於 SOC_min）。
    """
    dt_h = _dt_hours(minutes)
    eff = _discharge_path_eff(cfg)
    drawn_cell_kwh = (max(0.0, output_kw) / eff) * dt_h if eff > 0 else 0.0
    delta_soc = drawn_cell_kwh / cfg.batt_capacity_kwh
    new_soc = max(cfg.batt_soc_min, state.soc - delta_soc)
    return BatteryState(soc=new_soc)


# --------------------------------------------------------------------------- #
# 衰減成本（吞吐量法，見 `03` §5）
# --------------------------------------------------------------------------- #
def degradation_cost(throughput_kw: float, cfg: AppConfig, minutes: float) -> float:
    """電池衰減成本（吞吐量法）：BATT_DEGRADATION_COST ×（充+放功率）× Δt。

    用途：併入模組 5 目標函數，或在規則式策略中做「是否值得放電」門檻判斷，
    避免模型無腦狂充放電而高估效益（見 `03` §5）。

    參數:
        throughput_kw: 本時步的吞吐功率（= P_ch + P_dis，kW）。
        cfg: 設定物件。
        minutes: 時步長度（分鐘）。
    回傳:
        本時步衰減成本（元）。
    """
    dt_h = _dt_hours(minutes)
    return cfg.batt_deg_cost * max(0.0, throughput_kw) * dt_h


if __name__ == "__main__":
    print("=== battery_charge 示範（10 kWh 電池，15 分鐘時步）===")
    cfg = load_config()
    dt_min = float(cfg.timestep_min)
    print(f"  容量 {cfg.batt_capacity_kwh} kWh, SOC {cfg.batt_soc_min}~{cfg.batt_soc_max}, "
          f"充/放效率 {cfg.batt_charge_eff}/{cfg.batt_discharge_eff}, 耦合 {cfg.coupling.upper()}")

    # 從 SOC_min 開始，連續用太陽能餘電 3 kW + 市電 2 kW 充電數步。
    state = BatteryState(soc=cfg.batt_soc_min)
    print("\n  [充電] 太陽能餘電 3 kW + 市電 2 kW（優先序：先太陽能）")
    total_deg = 0.0
    for step in range(1, 6):
        res = charge_prioritized(state, pv_surplus_kw=3.0, grid_kw=2.0, cfg=cfg, minutes=dt_min)
        state = res["state"]  # type: ignore[assignment]
        deg = degradation_cost(res["batt_ch_kw"], cfg, dt_min)  # type: ignore[arg-type]
        total_deg += deg
        print(f"   step{step}: PV={res['pv_to_batt_kw']:.2f} 市電={res['grid_to_batt_kw']:.2f} "
              f"總充={res['batt_ch_kw']:.2f} kW → SOC={state.soc:.3f}")

    # 放電：對外輸出 4 kW 數步。
    print("\n  [放電] 對外輸出需求 4 kW")
    for step in range(1, 6):
        out_kw = min(4.0, max_dischargeable_kw(state, cfg, dt_min))
        state = apply_discharge(state, out_kw, cfg, dt_min)
        total_deg += degradation_cost(out_kw, cfg, dt_min)
        print(f"   step{step}: 實際放電={out_kw:.2f} kW → SOC={state.soc:.3f}")

    print(f"\n  累計衰減成本 ≈ {total_deg:.2f} 元（吞吐量法）")
    print("  ⚠️ 衰減成本須併入最佳化，否則會高估效益（見 03 §5）。")
