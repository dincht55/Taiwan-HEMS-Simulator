"""electrical_base.py — 模組 1：基本電學與功率平衡（最底層）。

職責（對應使用者規格與 `02_電學與太陽能模型`）：
- 功率↔能量換算（kW ↔ kWh，時步以分鐘計）。
- 單級／多級轉換效率套用（DC-DC、DC-AC、AC-DC，見 `02` §2）。
- 依耦合架構（DC/AC，見 `02` §3）計算太陽能→電池的額外效率因子。
- 節點功率平衡殘差（對應 `01` §3 不變式）。

相依方向：本檔僅 import 同層的 `config`，不 import 任何更高層模組（無循環 import）。
所有可調數值來自 `config.py`／傳入的 `AppConfig`，本檔不含硬編碼魔術數字。
單位一律標註於變數名（`*_kw`、`*_kwh`、`*_min`）。
"""

from __future__ import annotations

import numpy as np

from config import AppConfig, load_config  # 同層；re-export load_config 供上層使用

# 純量或 numpy 陣列皆可（向量化優先：同一函式可一次處理整條時間序列）。
Numeric = float | np.ndarray

# 每分鐘對小時的換算分母（避免散落魔術數字 60）。
_MINUTES_PER_HOUR: float = 60.0

__all__ = [
    "kwh",
    "apply_efficiency",
    "apply_efficiency_chain",
    "round_trip_efficiency",
    "coupling_extra_eff",
    "pv_to_battery_extra_eff",
    "power_balance_residual",
    "load_config",
]


def kwh(power_kw: Numeric, minutes: float) -> Numeric:
    """由「平均功率」與「時間長度」換算能量。

    公式: 能量(kWh) = 功率(kW) × 時間(分鐘) / 60。

    參數:
        power_kw: 該時段平均功率（kW）；可為純量或 numpy 陣列。
        minutes: 時段長度（分鐘）。
    回傳:
        能量（kWh），型別與 power_kw 對應。
    """
    return power_kw * minutes / _MINUTES_PER_HOUR


def apply_efficiency(power_kw: Numeric, eff: float) -> Numeric:
    """套用單一級電力電子轉換效率（見 `02` §2：P_out = P_in × η）。

    參數:
        power_kw: 轉換前功率（kW）。
        eff: 該級效率 η，須 0 < η ≤ 1。
    回傳:
        轉換後功率（kW）。
    """
    if not 0.0 < eff <= 1.0:
        raise ValueError(f"效率 eff 須介於 (0, 1]，收到：{eff}")
    return power_kw * eff


def apply_efficiency_chain(power_kw: Numeric, *effs: float) -> Numeric:
    """連續套用多級轉換效率（例如 AC 耦合的 DC-AC→AC-DC 鏈）。

    參數:
        power_kw: 轉換前功率（kW）。
        *effs: 依序套用的各級效率（每個皆須 0 < η ≤ 1）。
    回傳:
        全部套用後的功率（kW）；未給效率時原樣回傳。
    """
    result: Numeric = power_kw
    for eff in effs:
        result = apply_efficiency(result, eff)
    return result


def round_trip_efficiency(charge_eff: float, discharge_eff: float) -> float:
    """電池往返效率 = 充電效率 × 放電效率（見 `03` §1）。

    參數:
        charge_eff: 充電效率 η_ch（0~1）。
        discharge_eff: 放電效率 η_dis（0~1）。
    回傳:
        往返效率（0~1）。
    """
    return charge_eff * discharge_eff


def coupling_extra_eff(cfg: AppConfig) -> float:
    """耦合架構帶來的「額外」電力電子轉換效率因子（見 `02` §3）。

    說明:
        模組 2 的 `pv_kw` 與市電皆為交流端電力。
        - DC 耦合：電池在 DC 匯流排，太陽能經 DC-DC 充電，無額外交直流轉換，因子 = 1.0。
        - AC 耦合：交流電力需再經電池逆變器整流(AC-DC) 才能充電，
          故額外乘上 `batt_inverter_eff`（同一逆變器，充放電皆適用）。

    參數:
        cfg: 設定物件（提供 coupling 與 batt_inverter_eff）。
    回傳:
        額外效率因子（0~1）：DC=1.0，AC=batt_inverter_eff。
    """
    if cfg.coupling == "dc":
        return 1.0
    if cfg.coupling == "ac":
        return cfg.batt_inverter_eff
    raise ValueError(f"未知的 coupling：{cfg.coupling!r}（應為 'dc' 或 'ac'）")


def pv_to_battery_extra_eff(cfg: AppConfig) -> float:
    """太陽能→電池路徑的額外效率因子（見 `02` §3）。

    為 `coupling_extra_eff` 的語意別名：DC=1.0、AC=batt_inverter_eff；
    實際充電效率 η_ch 由模組 3 在充電時套用。

    參數:
        cfg: 設定物件。
    回傳:
        額外效率因子（0~1）。
    """
    return coupling_extra_eff(cfg)


def power_balance_residual(
    pv_kw: Numeric,
    grid_in_kw: Numeric,
    batt_dis_kw: Numeric,
    load_kw: Numeric,
    grid_out_kw: Numeric,
    batt_ch_kw: Numeric,
    curtail_kw: Numeric,
) -> Numeric:
    """節點功率平衡殘差（對應 `01` §3 不變式）。

    平衡式（任一時步都須成立）:
        pv + grid_in + batt_dis == load + grid_out + batt_ch + curtail
    殘差 = (輸入端總和) − (輸出端總和)，理想為 0。

    參數（單位皆為 kW，皆為正值或 0；可為純量或 numpy 陣列）:
        pv_kw: 太陽能可發功率。
        grid_in_kw: 自市電購入功率。
        batt_dis_kw: 電池放電（對外可用）功率。
        load_kw: 家庭負載。
        grid_out_kw: 餘電饋入市電功率。
        batt_ch_kw: 電池充電功率。
        curtail_kw: 限發/棄電功率。
    回傳:
        殘差（kW）；上層可用 `abs(residual) < tol` 做不變式檢查。
    """
    inflow = pv_kw + grid_in_kw + batt_dis_kw
    outflow = load_kw + grid_out_kw + batt_ch_kw + curtail_kw
    return inflow - outflow


def _self_test() -> None:
    """__main__ 用的最小自我測試；殘差須 ≈ 0、各換算須正確。"""
    cfg = load_config()

    # 1) 能量換算：1 kW 持續 15 分鐘 = 0.25 kWh。
    assert abs(kwh(1.0, 15.0) - 0.25) < 1e-9, "kwh 換算錯誤"

    # 2) 效率套用：5 kW 過 0.95 = 4.75 kW。
    assert abs(apply_efficiency(5.0, 0.95) - 4.75) < 1e-9, "apply_efficiency 錯誤"

    # 3) 效率鏈：2 級 0.96、0.95。
    expected = 10.0 * 0.96 * 0.95
    assert abs(apply_efficiency_chain(10.0, 0.96, 0.95) - expected) < 1e-9, "效率鏈錯誤"

    # 4) 耦合：DC 因子=1.0；AC 因子=batt_inverter_eff。
    assert pv_to_battery_extra_eff(cfg) == 1.0, "DC 耦合額外因子應為 1.0"

    # 5) 功率平衡：建構一個自洽的時步，殘差須 0。
    #    情境：PV 4 + 市電 1 + 放電 0 = 負載 3 + 饋電 0 + 充電 2 + 限發 0
    res = power_balance_residual(
        pv_kw=4.0, grid_in_kw=1.0, batt_dis_kw=0.0,
        load_kw=3.0, grid_out_kw=0.0, batt_ch_kw=2.0, curtail_kw=0.0,
    )
    assert abs(res) < 1e-9, f"功率平衡殘差應為 0，得到 {res}"

    # 6) 向量化：整條序列一次算，殘差皆 0。
    pv = np.array([4.0, 0.0, 6.0])
    gin = np.array([0.0, 3.0, 0.0])
    bdis = np.array([0.0, 0.0, 0.0])
    ld = np.array([3.0, 3.0, 3.0])
    gout = np.array([0.0, 0.0, 0.0])
    bch = np.array([1.0, 0.0, 3.0])
    cur = np.array([0.0, 0.0, 0.0])
    res_vec = power_balance_residual(pv, gin, bdis, ld, gout, bch, cur)
    assert np.allclose(res_vec, 0.0), f"向量化殘差應全為 0，得到 {res_vec}"

    print("  ✓ kwh 換算 / 效率套用 / 效率鏈 / 耦合因子 / 功率平衡（純量+向量）全數通過")


if __name__ == "__main__":
    print("=== electrical_base 自我測試 ===")
    _self_test()
    # 順帶示範往返效率。
    cfg = load_config()
    rte = round_trip_efficiency(cfg.batt_charge_eff, cfg.batt_discharge_eff)
    print(f"  電池往返效率 η_ch×η_dis = {rte:.4f}")
    print("  完成。")
