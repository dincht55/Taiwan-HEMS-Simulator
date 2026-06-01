"""power_dispatch.py — 模組 4：用電調度（自用優先）與計價查詢。

職責：
- 計價（依 `04_台電電價與時段`）：`classify_period` 判斷尖峰/半尖峰/離峰，`get_price` 回單價。
- `discharge_step`：對電池放電一個時步，受 SOC 下限與最大放電功率夾限（SOC 數學在模組 3）。
- `dispatch_step`：規則式「自用優先決策階梯」（見 `05` §1），回傳該時步所有功率分配欄位
  （對應 `01` §3 資料契約）。

相依方向：本檔 import 同層/更低層 `electrical_base`、`battery_charge` 與 `config`，
不 import 更高層（無循環 import）。pv_kw、load_kw 由上層（模組 5）算好後傳入。

時段邊界（小時）為 `04` §2 規格，於本檔以常數表呈現；費率「數值」住在 `config.py`（單一來源原則）。
假日處理：依共識「週六日全時段以離峰近似，平日套尖離峰」（`04` §2 註記，初版起點）。
⚠️ 費率約每年 4 月、10 月調整，時段亦可能調整；正式分析請以台電當期公告為準。
"""

from __future__ import annotations

import pandas as pd

import battery_charge as bc
import electrical_base as eb
from battery_charge import BatteryState
from config import AppConfig, load_config

# =============================================================================
# 時段定義（小時邊界；依 `04` §2，為實作起點，正式以台電公告為準）
# 夏月：6/1–9/30（低壓住宅）。
# =============================================================================
_SUMMER_MONTHS: frozenset[int] = frozenset({6, 7, 8, 9})

# 三段式（平日）。每段以 (起, 迄) 小時區間清單表示，半開區間 [起, 迄)。
_THREE_STAGE_SUMMER: dict[str, list[tuple[float, float]]] = {
    "peak":    [(16.0, 22.0)],
    "mid":     [(9.0, 16.0), (22.0, 24.0)],
    "offpeak": [(0.0, 9.0)],
}
_THREE_STAGE_NONSUMMER: dict[str, list[tuple[float, float]]] = {
    # 非夏月無尖峰
    "mid":     [(6.0, 11.0), (14.0, 24.0)],
    "offpeak": [(0.0, 6.0), (11.0, 14.0)],
}
# 二段式（平日）。非夏月時段沿用同結構（`04` 僅列夏月，已參數化便於調整）。
_TWO_STAGE: dict[str, list[tuple[float, float]]] = {
    "peak":    [(9.0, 24.0)],
    "offpeak": [(0.0, 9.0)],
}

__all__ = [
    "classify_period",
    "get_price",
    "discharge_step",
    "dispatch_step",
    "progressive_monthly_bill",
]


# =============================================================================
# 一般式（累進電價）分段表（元/度）— ⚠️ 近似參考值，須以台電當期公告為準。
# 結構：list[(累進上限度數, 該段單價)]；最後一段上限為 None（無上限）。
# 累進與時間無關，依「當月用電量」分段累加（見 `04` §1、§3）。
# =============================================================================
_PROGRESSIVE_TIERS_SUMMER: list[tuple[float | None, float]] = [
    (120.0, 1.68), (330.0, 2.45), (500.0, 3.70),
    (700.0, 5.04), (1000.0, 6.24), (None, 7.69),
]
_PROGRESSIVE_TIERS_NONSUMMER: list[tuple[float | None, float]] = [
    (120.0, 1.68), (330.0, 2.16), (500.0, 3.03),
    (700.0, 4.14), (1000.0, 5.07), (None, 6.63),
]


def progressive_monthly_bill(monthly_kwh: float, summer: bool) -> float:
    """一般式（累進電價）某月的流動電費（元），依分段累加。

    參數:
        monthly_kwh: 當月用電度數（度）。
        summer: 是否為夏月（6/1–9/30）。
    回傳:
        當月流動電費（元，不含基本費）。
    """
    tiers = _PROGRESSIVE_TIERS_SUMMER if summer else _PROGRESSIVE_TIERS_NONSUMMER
    remaining = max(0.0, monthly_kwh)
    cost = 0.0
    prev_cap = 0.0
    for cap, rate in tiers:
        seg = remaining if cap is None else min(remaining, cap - prev_cap)
        seg = max(0.0, seg)
        cost += seg * rate
        remaining -= seg
        prev_cap = cap if cap is not None else prev_cap
        if remaining <= 0:
            break
    return cost


# --------------------------------------------------------------------------- #
# 計價（對應 `04` §4）
# --------------------------------------------------------------------------- #
def _is_summer(ts: pd.Timestamp) -> bool:
    """是否為夏月（6/1–9/30）。"""
    return ts.month in _SUMMER_MONTHS


def _is_weekend(ts: pd.Timestamp) -> bool:
    """是否為週六日（初版假日近似用）。"""
    return ts.weekday() >= 5  # 5=六, 6=日


def _hour_in(ranges: list[tuple[float, float]], hour: float) -> bool:
    """小時是否落在任一 [起, 迄) 區間。"""
    return any(lo <= hour < hi for lo, hi in ranges)


def classify_period(ts: pd.Timestamp, cfg: AppConfig) -> str:
    """依 `04` §2 時段表回傳 'peak' / 'mid' / 'offpeak'（或一般式 'flat'）。

    規則:
        - 一般式（progressive）：不分時段，回 'flat'。
        - 週六日：全時段以離峰近似（回 'offpeak'）。
        - 平日：依電費類型與夏月/非夏月，用時段表判斷。

    參數:
        ts: 時間戳（建議帶在地時區）。
        cfg: 設定物件（提供 tariff_scheme）。
    回傳:
        時段字串：'peak' / 'mid' / 'offpeak' / 'flat'。
    """
    if cfg.tariff_scheme == "progressive":
        return "flat"

    if _is_weekend(ts):
        return "offpeak"

    hour = ts.hour + ts.minute / 60.0
    summer = _is_summer(ts)

    if cfg.tariff_scheme == "two_stage":
        table = _TWO_STAGE
        if _hour_in(table["peak"], hour):
            return "peak"
        return "offpeak"

    # 預設三段式
    table = _THREE_STAGE_SUMMER if summer else _THREE_STAGE_NONSUMMER
    if summer and _hour_in(table["peak"], hour):
        return "peak"
    if _hour_in(table["mid"], hour):
        return "mid"
    return "offpeak"


def get_price(ts: pd.Timestamp, cfg: AppConfig) -> float:
    """回傳該時刻購電單價（元/度），由 classify_period 對應 config 的 TARIFF_* 數值。

    參數:
        ts: 時間戳。
        cfg: 設定物件（提供各時段費率）。
    回傳:
        購電單價（元/度）。

    註：一般式（progressive）實為依「用電量」累進、與時間無關；此處以半尖峰價作
        代表單價近似（本專案核心在時間電價的削峰填谷，progressive 僅為對照）。
    """
    period = classify_period(ts, cfg)
    if period == "peak":
        return cfg.tariff_peak
    if period == "mid":
        return cfg.tariff_mid
    if period == "offpeak":
        return cfg.tariff_offpeak
    return cfg.tariff_mid  # 'flat'（progressive）代表單價


# --------------------------------------------------------------------------- #
# 放電（介面在模組 4，SOC 數學用模組 3）
# --------------------------------------------------------------------------- #
def _period_price(period: str, cfg: AppConfig) -> float:
    """由時段字串對應購電單價（不需時間戳，供 dispatch 內部門檻判斷）。"""
    return {"peak": cfg.tariff_peak, "mid": cfg.tariff_mid,
            "offpeak": cfg.tariff_offpeak}.get(period, cfg.tariff_mid)


def discharge_step(
    state: BatteryState,
    demand_kw: float,
    cfg: AppConfig,
    minutes: float,
    floor_soc: float | None = None,
) -> tuple[float, BatteryState]:
    """對電池放電一個時步以滿足需求；回傳 (實際對外放電 kW, 新狀態)。

    受 SOC 下限（或傳入的 floor_soc）與最大放電功率夾限、放電效率計入（皆由模組 3 處理）。

    參數:
        state: 目前電池狀態。
        demand_kw: 期望由電池供應的功率（kW，<0 視為 0）。
        cfg: 設定物件。
        minutes: 時步長度（分鐘）。
        floor_soc: 放電下限 SOC；None → BATT_SOC_MIN。智慧放電可傳較高值保留給尖峰。
    回傳:
        (實際對外放電功率 kW, 新的 BatteryState)。
    """
    actual_kw = min(max(0.0, demand_kw), bc.max_dischargeable_kw(state, cfg, minutes, floor_soc))
    new_state = bc.apply_discharge(state, actual_kw, cfg, minutes)
    return actual_kw, new_state


# --------------------------------------------------------------------------- #
# 離峰用市電補電池到目標 SOC（非預測式填谷；預設關閉）
# --------------------------------------------------------------------------- #
def _grid_charge_toward_target(
    state: BatteryState,
    used_charge_power_kw: float,
    cfg: AppConfig,
    minutes: float,
) -> tuple[float, BatteryState]:
    """離峰時用市電把電池補到 `offpeak_charge_target_soc`（受功率/容量夾限）。

    參數:
        state: 目前電池狀態。
        used_charge_power_kw: 本時步已用掉的充電功率（扣除後才是可用功率額度）。
        cfg: 設定物件。
        minutes: 時步長度（分鐘）。
    回傳:
        (實際自市電充電功率 kW, 新狀態)。
    """
    target = cfg.offpeak_charge_target_soc
    if not (target > state.soc and target > cfg.batt_soc_min):
        return 0.0, state  # 未啟用或已達標

    dt_h = minutes / 60.0
    charge_eff = cfg.batt_charge_eff * eb.coupling_extra_eff(cfg)
    # 補到目標所需的外部功率（kW）。
    need_cell_kwh = (target - state.soc) * cfg.batt_capacity_kwh
    ext_kw_to_target = need_cell_kwh / (charge_eff * dt_h) if (charge_eff > 0 and dt_h > 0) else 0.0
    remaining_power_kw = max(0.0, cfg.batt_max_charge_kw - used_charge_power_kw)
    offer_kw = min(ext_kw_to_target, remaining_power_kw)
    return bc.charge_step(state, offer_kw, "grid", cfg, minutes)


# --------------------------------------------------------------------------- #
# 規則式自用優先調度（見 `05` §1）
# --------------------------------------------------------------------------- #
def dispatch_step(
    state: BatteryState,
    pv_kw: float,
    load_kw: float,
    price_period: str,
    cfg: AppConfig,
    minutes: float,
) -> dict[str, float | BatteryState | str]:
    """單一時步「自用優先」調度（規則式），回傳所有功率分配（對應 `01` §3 欄位）。

    決策階梯（見 `05` §1）:
        1. 太陽能優先供當下負載（pv_to_load = min(pv, load)）。
        2. 若太陽能 > 負載（餘電）：餘電充電池；電池滿了才饋電（自用型）或限發。
        3. 若太陽能 < 負載（缺口）：
           a. 貴時段（peak/mid）：電池放電補缺口，電池見底才用市電。
           b. 離峰/一般式（offpeak/flat）：直接用市電供負載；
              若啟用離峰目標 SOC，再用市電把電池補到目標（非預測式填谷）。

    參數:
        state: 目前電池狀態。
        pv_kw: 本時步太陽能可發功率（kW）。
        load_kw: 本時步家庭負載（kW）。
        price_period: classify_period 的輸出（'peak'/'mid'/'offpeak'/'flat'）。
        cfg: 設定物件。
        minutes: 時步長度（分鐘）。
    回傳:
        dict，含（單位皆 kW，soc 為 0~1）：
          pv_to_load_kw, pv_to_batt_kw, pv_curtail_kw,
          batt_ch_kw, batt_dis_kw, grid_in_kw, grid_out_kw,
          soc（時步結束）, state（新狀態）, price_period。
    """
    pv_kw = max(0.0, pv_kw)
    load_kw = max(0.0, load_kw)

    # 1) 太陽能優先供負載。
    pv_to_load_kw = min(pv_kw, load_kw)
    pv_surplus_kw = pv_kw - pv_to_load_kw       # ≥0
    load_remaining_kw = load_kw - pv_to_load_kw  # ≥0

    pv_to_batt_kw = 0.0
    grid_to_batt_kw = 0.0
    pv_curtail_kw = 0.0
    batt_dis_kw = 0.0
    grid_in_kw = 0.0
    grid_out_kw = 0.0

    if pv_surplus_kw > 0.0:
        # 2) 有餘電：先充電池（只用太陽能，市電給 0）。
        res = bc.charge_prioritized(state, pv_surplus_kw, 0.0, cfg, minutes)
        pv_to_batt_kw = float(res["pv_to_batt_kw"])  # type: ignore[arg-type]
        state = res["state"]                          # type: ignore[assignment]
        leftover_pv_kw = pv_surplus_kw - pv_to_batt_kw
        # 電池滿了之後的餘電：自用型不賣電→限發；若有售電價→饋電。
        if cfg.tariff_sell_price > 0.0:
            grid_out_kw = leftover_pv_kw
        else:
            pv_curtail_kw = leftover_pv_kw

    elif load_remaining_kw > 0.0:
        # 3) 有缺口。
        if price_period in ("peak", "mid", "flat"):
            # 3a 非離峰（含一般式 flat）：電池放電補缺口，見底才用市電。
            if cfg.smart_discharge:
                # 智慧放電門檻：
                #  (i) 衰減成本閘：唯有「該時段單價 > 放電邊際衰減成本」才放電（否則不划算）。
                #  (ii) 尖峰保留：半尖峰時只放到 mid_discharge_reserve_soc 以上，把電量留給尖峰。
                price_now = _period_price(price_period, cfg)
                if price_now > cfg.batt_deg_cost:
                    floor = (max(cfg.batt_soc_min, cfg.mid_discharge_reserve_soc)
                             if price_period == "mid" else cfg.batt_soc_min)
                    batt_dis_kw, state = discharge_step(
                        state, load_remaining_kw, cfg, minutes, floor_soc=floor
                    )
            else:
                batt_dis_kw, state = discharge_step(state, load_remaining_kw, cfg, minutes)
            grid_in_kw = load_remaining_kw - batt_dis_kw
        else:
            # 3b 離峰/一般式：用市電供負載；選用：補電池到目標 SOC。
            grid_in_kw = load_remaining_kw
            if price_period == "offpeak":
                grid_to_batt_kw, state = _grid_charge_toward_target(
                    state, used_charge_power_kw=0.0, cfg=cfg, minutes=minutes
                )
                grid_in_kw += grid_to_batt_kw

    batt_ch_kw = pv_to_batt_kw + grid_to_batt_kw

    return {
        "pv_to_load_kw": pv_to_load_kw,
        "pv_to_batt_kw": pv_to_batt_kw,
        "grid_to_batt_kw": grid_to_batt_kw,
        "pv_curtail_kw": pv_curtail_kw,
        "batt_ch_kw": batt_ch_kw,
        "batt_dis_kw": batt_dis_kw,
        "grid_in_kw": grid_in_kw,
        "grid_out_kw": grid_out_kw,
        "soc": state.soc,
        "state": state,
        "price_period": price_period,
    }


if __name__ == "__main__":
    print("=== power_dispatch 示範（高雄、三段式、夏日典型時刻）===")
    cfg = load_config()
    dt = float(cfg.timestep_min)

    # 幾個代表性時刻 + 情境（pv, load）。
    samples = [
        ("2025-07-15 03:00", 0.0, 0.5, "離峰深夜：太陽能 0、低負載"),
        ("2025-07-15 12:00", 4.5, 1.0, "正午：太陽能多、負載低（餘電充電池）"),
        ("2025-07-15 18:30", 0.3, 3.0, "傍晚尖峰：太陽能少、負載高（電池放電）"),
        ("2025-07-19 18:30", 0.3, 3.0, "週六傍晚：以離峰近似（用市電）"),
    ]
    state = BatteryState(soc=0.6)
    for ts_str, pv, load, desc in samples:
        ts = pd.Timestamp(ts_str, tz=cfg.timezone)
        period = classify_period(ts, cfg)
        price = get_price(ts, cfg)
        r = dispatch_step(state, pv, load, period, cfg, dt)
        state = r["state"]  # type: ignore[assignment]
        # 功率平衡殘差檢查（不變式）。
        resid = eb.power_balance_residual(
            pv, r["grid_in_kw"], r["batt_dis_kw"], load,
            r["grid_out_kw"], r["batt_ch_kw"], r["pv_curtail_kw"],
        )
        print(f"\n  {ts_str}（{period}, {price} 元/度）— {desc}")
        print(f"    PV→負載 {r['pv_to_load_kw']:.2f} | PV→電池 {r['pv_to_batt_kw']:.2f} | "
              f"限發 {r['pv_curtail_kw']:.2f}")
        print(f"    電池放電 {r['batt_dis_kw']:.2f} | 購電 {r['grid_in_kw']:.2f} | "
              f"饋電 {r['grid_out_kw']:.2f} → SOC {r['soc']:.3f}")
        print(f"    功率平衡殘差 = {resid:.2e}（應≈0）")
    print("\n  ⚠️ 時段/費率以台電當期公告為準；結果為決策參考，非投資建議。")
