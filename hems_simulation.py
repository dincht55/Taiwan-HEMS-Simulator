"""hems_simulation.py — 模組 5：整合 1–4，全年逐時模擬、策略切換、指標與輸出。

職責（對應 `01` §5 全域流程、`05` §3 指標、`00` §4 模組 5 驗收）：
- 載入設定 → 建時間索引 → 取負載與太陽能 → 逐時跑控制策略 → 算指標 → 存 CSV/圖 → 印摘要。
- 控制策略（`SIM_CONTROL_STRATEGY`）：
    * "rule_based"          ：純自用優先（模組 4），離峰是否充電由靜態設定決定。
    * "rule_based_forecast" ：本檔新增。**查「明天天氣預報」**估隔日太陽能，
                              據此配置夜間（離峰）用市電把電池補到的目標 SOC（只補太陽能補不到的缺口）。
    * "milp_mpc"            ：以 pulp 解每日 MILP（最佳化基準，見 `05` §2）。
- `compute_metrics`：SCR、SSR、年省電費、回本年限（`05` §3）。

相依方向：本檔為唯一協調者，import 全部更低層模組（electrical_base / solar_pv /
battery_charge / power_dispatch / config），符合單向相依、無循環。
數值來自 `config.py`；單位標註於變數名。

⚠️ 經濟試算之費率與單價會變動（台電約每年 4 月、10 月調整），結果為決策參考、非投資建議。
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
import pandas as pd

import battery_charge as bc
import electrical_base as eb
import fast_core
import power_dispatch as pdm
import solar_pv as sp
from battery_charge import BatteryState
from config import AppConfig, load_config

# pulp 為 milp_mpc 策略所需（選用）。
try:
    import pulp  # type: ignore

    _HAS_PULP = True
except ImportError:  # pragma: no cover
    _HAS_PULP = False

# 主模擬 DataFrame 欄位（對應 `01` §3 資料契約）。
_DF_COLUMNS = [
    "pv_kw", "load_kw", "price", "soc",
    "batt_ch_kw", "batt_dis_kw", "grid_in_kw", "grid_out_kw",
    "pv_to_load_kw", "pv_to_batt_kw", "grid_to_batt_kw", "pv_curtail_kw",
]


# =============================================================================
# 1. 負載資料
# =============================================================================
def build_load_profile(cfg: AppConfig, index: pd.DatetimeIndex) -> pd.Series:
    """取得每時步家庭負載（kW）。

    優先讀 `LOAD_PROFILE_CSV`（單欄 kW，長度需與 index 對齊）；
    讀不到則用內建「台灣住宅合成負載」：早晚雙峰、夏季空調加重、週末白天略高，
    加入小幅日間隨機波動以近似真實（可重現，seed 固定）。

    參數:
        cfg: 設定物件。
        index: 時間索引。
    回傳:
        pd.Series：每時步負載（kW）。
    """
    # (b) 嘗試讀實測 CSV。
    if cfg.load_profile_csv and os.path.exists(cfg.load_profile_csv):
        raw = pd.read_csv(cfg.load_profile_csv)
        col = raw.columns[-1]  # 取最後一欄當 kW
        series = pd.Series(raw[col].to_numpy()[: len(index)], index=index, name="load_kw")
        return series.clip(lower=0.0)

    # (a) 內建合成負載。
    hours = index.hour + index.minute / 60.0
    months = index.month
    weekday = index.dayofweek  # 0~6

    def bell(center: float, width: float, amp: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((hours - center) / width) ** 2)

    base_kw = 0.3
    morning = bell(7.5, 1.2, 1.3)     # 早高峰（起床、早餐）
    evening = bell(19.5, 2.0, 2.4)    # 晚高峰（煮飯、空調、照明）
    midday = bell(12.5, 2.5, 0.5)     # 午間小峰
    load = base_kw + morning + evening + midday

    # 夏季（6~9 月）空調加重。
    summer = np.isin(months, [6, 7, 8, 9])
    load = load * np.where(summer, 1.45, 1.0)

    # 週末白天略高（在家時間長）。
    weekend = weekday >= 5
    load = load + np.where(weekend & (hours > 9) & (hours < 22), 0.4, 0.0)

    # 小幅每日波動（可重現）。
    rng = np.random.default_rng(42)
    day_factor = rng.normal(1.0, 0.08, size=len(np.unique(index.date)))
    day_index = {d: i for i, d in enumerate(np.unique(index.date))}
    factors = np.array([day_factor[day_index[d]] for d in index.date])
    load = load * factors

    # 依負載等級縮放：high≈每月約 950 度（重用戶）；medium≈每月約 600 度。
    level_scale = {"high": 1.0, "medium": 0.62}.get(cfg.load_level, 1.0)
    load = load * level_scale

    return pd.Series(np.clip(load, 0.0, None), index=index, name="load_kw").round(3)


# =============================================================================
# 2. 「查明天天氣預報」與夜間充電額度配置
# =============================================================================
def query_tomorrow_solar(
    pv_series: pd.Series,
    target_date,
    forecast_error_std: float,
    rng: np.random.Generator,
) -> pd.Series:
    """模擬「查詢明天天氣預報」：回傳 target_date 當日的預測太陽能逐時功率（kW）。

    實作說明:
        - 真實系統會呼叫中央氣象署開放資料 API（鄉鎮天氣/日射預報）取得隔日預報。
          本模擬環境無外網對外氣象 API，故以模擬器自身的太陽能序列當「真值」，
          再依 forecast_error_std 疊加雜訊以模擬預報誤差（0 = 完美預報）。
        - 介面預留：未來可把此函式換成真實 API 呼叫，回傳同樣格式的逐時預測即可。

    參數:
        pv_series: 全期太陽能真值序列（kW）。
        target_date: 欲預報的日期（date）。
        forecast_error_std: 預報誤差標準差（相對比例，如 0.15 = ±15%）。
        rng: 隨機產生器（可重現）。
    回傳:
        pd.Series：target_date 當日的預測太陽能（kW）。
    """
    day_mask = pv_series.index.date == target_date
    true_pv = pv_series[day_mask]
    if forecast_error_std <= 0:
        return true_pv
    # 整日一個雲量偏差係數（晴/陰整天相關），再做非負裁切。
    factor = max(0.0, 1.0 + rng.normal(0.0, forecast_error_std))
    return (true_pv * factor).clip(lower=0.0)


def plan_mid_discharge_reserve_soc(
    forecast_pv_day: pd.Series,
    forecast_load_day: pd.Series,
    cfg: AppConfig,
    periods: Optional[np.ndarray] = None,
) -> float:
    """依明日預報，算「半尖峰時應為尖峰保留的最低 SOC」（放電端預測）。

    邏輯:
        預期尖峰缺口（對外可用電量）→ 換成電芯電量（÷η_dis）→ 目標保留 SOC。
        半尖峰放電時不得低於此 SOC，把電量留到最貴的尖峰用。
        若 discharge_only_peak=True 且當日有尖峰 → 直接設為 SOC_max（半尖峰完全不放，全留尖峰）。

    參數:
        forecast_pv_day: 明日預測太陽能逐時（kW）。
        forecast_load_day: 明日預測負載逐時（kW）。
        cfg: 設定物件。
        periods: 預先算好的時段陣列（對齊 forecast_pv_day.index）；None 則自行分類。
    回傳:
        半尖峰放電保留 SOC（0~1）。無尖峰（非夏月）時回 SOC_min（不保留）。
    """
    if len(forecast_pv_day) == 0:
        return cfg.batt_soc_min
    dt_h = cfg.timestep_min / 60.0
    idx = forecast_pv_day.index
    if periods is None:
        periods = np.array([pdm.classify_period(ts, cfg) for ts in idx])
    is_peak = periods == "peak"
    if not is_peak.any():
        return cfg.batt_soc_min  # 非夏月無尖峰，不需保留
    # 「只在最貴時段放電」：有尖峰的日子，半尖峰完全不放電。
    if cfg.discharge_only_peak:
        return cfg.batt_soc_max

    pv = forecast_pv_day.to_numpy()
    load = forecast_load_day.reindex(idx).to_numpy()
    usable_kwh = (cfg.batt_soc_max - cfg.batt_soc_min) * cfg.batt_capacity_kwh
    peak_deficit_kwh = min(
        float(np.maximum(load[is_peak] - pv[is_peak], 0.0).sum() * dt_h), usable_kwh
    )
    cell_kwh = peak_deficit_kwh / cfg.batt_discharge_eff if cfg.batt_discharge_eff > 0 else peak_deficit_kwh
    reserve = cfg.batt_soc_min + cell_kwh / cfg.batt_capacity_kwh
    return float(np.clip(reserve, cfg.batt_soc_min, cfg.batt_soc_max))


def plan_offpeak_target_soc(
    forecast_pv_day: pd.Series,
    forecast_load_day: pd.Series,
    cfg: AppConfig,
) -> float:
    """依「明日預報」算出夜間（離峰）用市電充電的目標 SOC（只補太陽能補不到的缺口）。

    邏輯（對應 `05` §1 第 3b 與你提出的策略）:
        1. 預期尖峰缺口 = Σ_尖峰時段 max(load − pv, 0) × Δt，受可用容量上限。
        2. 預期日間太陽能餘電（可充進電池）= Σ_日間 max(pv − load, 0) × Δt，受可用容量上限。
        3. 夜間需補的「對外可用」電量 gap = max(0, 尖峰缺口 − 太陽能餘電)。
        4. 換成目標 SOC：target = SOC_min + (gap / η_dis) / E_cap，夾在 [SOC_min, SOC_max]。
           （gap 為對外可用電量，需除以放電效率還原成電芯電量）

    參數:
        forecast_pv_day: 明日預測太陽能逐時（kW）。
        forecast_load_day: 明日預測負載逐時（kW）。
        cfg: 設定物件。
    回傳:
        夜間充電目標 SOC（0~1）。若 gap=0（晴天）→ 回 SOC_min（等同不充）。
    """
    if len(forecast_pv_day) == 0:
        return cfg.batt_soc_min

    dt_h = cfg.timestep_min / 60.0
    idx = forecast_pv_day.index
    hours = idx.hour + idx.minute / 60.0
    # 尖峰時段判定（用 classify_period 較嚴謹，但這裡以 16~22 近似日間→尖峰邊界）。
    periods = np.array([pdm.classify_period(ts, cfg) for ts in idx])
    is_peak = periods == "peak"
    is_daytime = (hours >= 8) & (hours < 16)  # 尖峰前的日間充電窗

    pv = forecast_pv_day.to_numpy()
    load = forecast_load_day.reindex(idx).to_numpy()

    usable_kwh = (cfg.batt_soc_max - cfg.batt_soc_min) * cfg.batt_capacity_kwh
    peak_deficit_kwh = float(np.maximum(load[is_peak] - pv[is_peak], 0.0).sum() * dt_h)
    peak_deficit_kwh = min(peak_deficit_kwh, usable_kwh)
    solar_surplus_kwh = float(np.maximum(pv[is_daytime] - load[is_daytime], 0.0).sum() * dt_h)
    solar_surplus_kwh = min(solar_surplus_kwh, usable_kwh)

    gap_kwh = max(0.0, peak_deficit_kwh - solar_surplus_kwh)
    cell_kwh = gap_kwh / cfg.batt_discharge_eff if cfg.batt_discharge_eff > 0 else gap_kwh
    target = cfg.batt_soc_min + cell_kwh / cfg.batt_capacity_kwh
    return float(np.clip(target, cfg.batt_soc_min, cfg.batt_soc_max))


# =============================================================================
# 3. 各策略的逐時模擬
# =============================================================================
def _empty_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=index, columns=_DF_COLUMNS)


def _simulate_rule(
    cfg: AppConfig,
    index: pd.DatetimeIndex,
    pv: pd.Series,
    load: pd.Series,
    daily_overrides: Optional[dict] = None,
) -> pd.DataFrame:
    """規則式逐時模擬（rule_based 與 rule_based_forecast 共用）。

    參數:
        cfg: 設定物件。
        index, pv, load: 時間索引與兩條輸入序列（kW）。
        daily_overrides: {date: {欄位名: 值}}，逐日覆寫設定（如夜充目標、尖峰保留、
                         智慧放電開關）。None → 全程用 cfg（rule_based）。
    回傳:
        pd.DataFrame（`01` §3 欄位）。
    """
    n = len(index)
    pv_arr = pv.to_numpy().astype(np.float64)
    load_arr = load.to_numpy().astype(np.float64)
    dates = index.date

    # 時段與單價只跟 tariff_scheme/時間有關（不受每日覆寫影響）→ 一次算好。
    periods_str = [pdm.classify_period(ts, cfg) for ts in index]
    period_code = np.array([fast_core.PERIOD_CODE[p] for p in periods_str], dtype=np.int64)
    prices = np.array([pdm._period_price(p, cfg) for p in periods_str], dtype=np.float64)

    # 逐步的「離峰目標 / 半尖峰保留 / 智慧放電開關」（由每日覆寫展開；無覆寫則用 cfg 值）。
    offpeak_target = np.full(n, cfg.offpeak_charge_target_soc, dtype=np.float64)
    mid_reserve = np.full(n, cfg.mid_discharge_reserve_soc, dtype=np.float64)
    smart_flag = np.full(n, 1 if cfg.smart_discharge else 0, dtype=np.int64)
    if daily_overrides is not None:
        for i in range(n):
            ov = daily_overrides.get(dates[i])
            if ov is None:
                continue
            if "offpeak_charge_target_soc" in ov:
                offpeak_target[i] = ov["offpeak_charge_target_soc"]
            if "mid_discharge_reserve_soc" in ov:
                mid_reserve[i] = ov["mid_discharge_reserve_soc"]
            if "smart_discharge" in ov:
                smart_flag[i] = 1 if ov["smart_discharge"] else 0

    eta_ch = cfg.batt_charge_eff * eb.coupling_extra_eff(cfg)
    cp_code = 1 if cfg.charge_priority == "battery_first" else 0
    ch, dis, gin, gout, p2l, p2b, g2b, cur, soc = fast_core.rule_loop(
        pv_arr, load_arr, period_code, prices, offpeak_target, mid_reserve, smart_flag,
        float(cfg.batt_capacity_kwh), float(cfg.batt_soc_min), float(cfg.batt_soc_max),
        float(cfg.batt_max_charge_kw), float(cfg.batt_max_discharge_kw),
        float(eta_ch), float(cfg.batt_discharge_eff), float(cfg.batt_deg_cost),
        float(cfg.timestep_min) / 60.0, float(cfg.tariff_sell_price), cp_code,
    )

    data = {
        "pv_kw": pv_arr, "load_kw": load_arr, "price": prices, "soc": soc,
        "batt_ch_kw": ch, "batt_dis_kw": dis, "grid_in_kw": gin, "grid_out_kw": gout,
        "pv_to_load_kw": p2l, "pv_to_batt_kw": p2b, "grid_to_batt_kw": g2b,
        "pv_curtail_kw": cur,
    }
    return pd.DataFrame({c: data[c] for c in _DF_COLUMNS}, index=index)


def _solve_day_milp(
    cfg: AppConfig,
    day_index: pd.DatetimeIndex,
    pv_day: np.ndarray,
    load_day: np.ndarray,
    price_day: np.ndarray,
    soc_init: float,
) -> dict:
    """以 pulp 解單日 MILP（最佳化基準，見 `05` §2）。回傳各時步功率與末端 SOC。

    參數:
        cfg: 設定物件。
        day_index: 當日時間索引。
        pv_day, load_day, price_day: 當日太陽能/負載/購電價陣列。
        soc_init: 當日起始 SOC。
    回傳:
        dict：含各 kW 陣列與 soc 陣列、末端 soc。
    """
    n = len(day_index)
    dt_h = cfg.timestep_min / 60.0
    eta_ch = cfg.batt_charge_eff * eb.coupling_extra_eff(cfg)
    eta_dis = cfg.batt_discharge_eff
    cap = cfg.batt_capacity_kwh
    sell = cfg.tariff_sell_price

    prob = pulp.LpProblem("hems_day", pulp.LpMinimize)
    Pch = [pulp.LpVariable(f"ch{t}", 0, cfg.batt_max_charge_kw) for t in range(n)]
    Pdis = [pulp.LpVariable(f"dis{t}", 0, cfg.batt_max_discharge_kw) for t in range(n)]
    Gin = [pulp.LpVariable(f"gin{t}", 0) for t in range(n)]
    Gout = [pulp.LpVariable(f"gout{t}", 0, (None if sell > 0 else 0)) for t in range(n)]
    Cur = [pulp.LpVariable(f"cur{t}", 0) for t in range(n)]
    Soc = [pulp.LpVariable(f"soc{t}", cfg.batt_soc_min, cfg.batt_soc_max) for t in range(n)]
    U = [pulp.LpVariable(f"u{t}", cat="Binary") for t in range(n)]  # 1=可充, 0=可放

    # 目標：最小化淨成本（含衰減）。
    prob += pulp.lpSum([
        (price_day[t] * Gin[t] - sell * Gout[t] + cfg.batt_deg_cost * (Pch[t] + Pdis[t])) * dt_h
        for t in range(n)
    ])

    for t in range(n):
        # 功率平衡。
        prob += pv_day[t] + Gin[t] + Pdis[t] == load_day[t] + Gout[t] + Pch[t] + Cur[t]
        # 餘電躉售只收「太陽能餘電」：饋網量不得超過當下太陽能發電，
        # 否則 sell>購電價時，MILP 會找出「離峰買電→直接饋網賣」的無限套利（不符 FiT 規則）。
        if sell > 0:
            prob += Gout[t] <= pv_day[t]
        # 充放電與二元變數（禁止同時充放）。
        prob += Pch[t] <= cfg.batt_max_charge_kw * U[t]
        prob += Pdis[t] <= cfg.batt_max_discharge_kw * (1 - U[t])
        # SOC 動態。
        prev = soc_init if t == 0 else Soc[t - 1]
        prob += Soc[t] == prev + (eta_ch * Pch[t] - Pdis[t] / eta_dis) * dt_h / cap

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    val = lambda v: max(0.0, float(pulp.value(v) or 0.0))
    return {
        "batt_ch_kw": np.array([val(Pch[t]) for t in range(n)]),
        "batt_dis_kw": np.array([val(Pdis[t]) for t in range(n)]),
        "grid_in_kw": np.array([val(Gin[t]) for t in range(n)]),
        "grid_out_kw": np.array([val(Gout[t]) for t in range(n)]),
        "pv_curtail_kw": np.array([val(Cur[t]) for t in range(n)]),
        "soc": np.array([float(pulp.value(Soc[t])) for t in range(n)]),
        "soc_end": float(pulp.value(Soc[n - 1])),
    }


def _simulate_milp(
    cfg: AppConfig,
    index: pd.DatetimeIndex,
    pv: pd.Series,
    load: pd.Series,
) -> pd.DataFrame:
    """逐日求解 MILP 並串接（SOC 跨日延續）。"""
    if not _HAS_PULP:
        raise RuntimeError("milp_mpc 需要 pulp 套件：pip install pulp")

    df = _empty_df(index)
    price = pd.Series([pdm.get_price(ts, cfg) for ts in index], index=index)
    pv_arr, load_arr, price_arr = pv.to_numpy(), load.to_numpy(), price.to_numpy()
    soc = cfg.batt_soc_min
    for day in np.unique(index.date):
        mask = index.date == day
        positions = np.where(mask)[0]
        d_index = index[mask]
        m = _solve_day_milp(
            cfg, d_index, pv_arr[mask], load_arr[mask], price_arr[mask], soc,
        )
        soc = m["soc_end"]
        for t, i in enumerate(positions):
            df.iat[i, df.columns.get_loc("pv_kw")] = pv_arr[i]
            df.iat[i, df.columns.get_loc("load_kw")] = load_arr[i]
            df.iat[i, df.columns.get_loc("price")] = price_arr[i]
            for k in ("batt_ch_kw", "batt_dis_kw", "grid_in_kw", "grid_out_kw",
                      "pv_curtail_kw", "soc"):
                df.iat[i, df.columns.get_loc(k)] = m[k][t]
        # 分解 pv_to_load / pv_to_batt / grid_to_batt（事後拆，便於指標）。
    # 事後拆解太陽能流向（不影響功率平衡）。
    pv_to_load = np.minimum(df["pv_kw"], df["load_kw"])
    df["pv_to_load_kw"] = pv_to_load
    pv_surplus = (df["pv_kw"] - pv_to_load).clip(lower=0)
    df["pv_to_batt_kw"] = np.minimum(pv_surplus, df["batt_ch_kw"])
    df["grid_to_batt_kw"] = (df["batt_ch_kw"] - df["pv_to_batt_kw"]).clip(lower=0)
    return df


# =============================================================================
# 4. 主模擬入口
# =============================================================================
def run_simulation(cfg: AppConfig, forecast_error_std: float = 0.0) -> pd.DataFrame:
    """跑完整段模擬，回傳 `01` §3 的 DataFrame。

    參數:
        cfg: 設定物件（control_strategy 決定策略）。
        forecast_error_std: 預測式策略的天氣預報誤差（0=完美預報）。
    回傳:
        pd.DataFrame：每時步功率分配與 SOC。
    """
    periods = int(cfg.sim_days * 24 * 60 / cfg.timestep_min)
    index = pd.date_range(cfg.start_date, periods=periods,
                          freq=f"{cfg.timestep_min}min", tz=cfg.timezone)
    pv = sp.simulate_pv(cfg, index)
    load = build_load_profile(cfg, index)

    strategy = cfg.control_strategy
    if strategy == "rule_based":
        return _simulate_rule(cfg, index, pv, load, daily_overrides=None)

    if strategy == "rule_based_forecast":
        # 對每一天「查明天天氣預報」→ 配置夜間充電目標 + 尖峰保留 + 啟用智慧放電。
        # 向量化：時段與分日位置只算一次，逐日用 numpy 計算（供大量掃描時夠快）。
        rng = np.random.default_rng(2025)
        pv_arr = pv.to_numpy()
        load_arr = load.to_numpy()
        hours = index.hour + index.minute / 60.0
        is_peak_all = np.array([pdm.classify_period(ts, cfg) == "peak" for ts in index])
        dt_h = cfg.timestep_min / 60.0
        usable_kwh = (cfg.batt_soc_max - cfg.batt_soc_min) * cfg.batt_capacity_kwh
        dates_all = index.date

        pos_by_day: dict = {}
        for i, d in enumerate(dates_all):
            pos_by_day.setdefault(d, []).append(i)

        overrides: dict = {}
        for day, pos in pos_by_day.items():
            pos = np.array(pos)
            factor = 1.0 if forecast_error_std <= 0 else max(0.0, 1.0 + rng.normal(0.0, forecast_error_std))
            f_pv = pv_arr[pos] * factor
            f_load = load_arr[pos]
            peak_mask = is_peak_all[pos]
            day_mask = (hours[pos] >= 8) & (hours[pos] < 16)

            peak_deficit = min(float(np.maximum(f_load[peak_mask] - f_pv[peak_mask], 0.0).sum() * dt_h),
                               usable_kwh) if peak_mask.any() else 0.0
            solar_surplus = min(float(np.maximum(f_pv[day_mask] - f_load[day_mask], 0.0).sum() * dt_h),
                                usable_kwh)
            # 夜間離峰充電目標：補太陽能補不到的尖峰缺口。
            gap = max(0.0, peak_deficit - solar_surplus)
            cell = gap / cfg.batt_discharge_eff if cfg.batt_discharge_eff > 0 else gap
            target = float(np.clip(cfg.batt_soc_min + cell / cfg.batt_capacity_kwh,
                                   cfg.batt_soc_min, cfg.batt_soc_max))
            # 半尖峰保留：只在最貴時段放電→有尖峰的日子半尖峰全保留；否則保留尖峰缺口量。
            if not peak_mask.any():
                reserve = cfg.batt_soc_min
            elif cfg.discharge_only_peak:
                reserve = cfg.batt_soc_max
            else:
                cell_r = peak_deficit / cfg.batt_discharge_eff if cfg.batt_discharge_eff > 0 else peak_deficit
                reserve = float(np.clip(cfg.batt_soc_min + cell_r / cfg.batt_capacity_kwh,
                                        cfg.batt_soc_min, cfg.batt_soc_max))
            overrides[day] = {
                "offpeak_charge_target_soc": target,
                "mid_discharge_reserve_soc": reserve,
                "smart_discharge": True,
            }
        return _simulate_rule(cfg, index, pv, load, daily_overrides=overrides)

    if strategy == "milp_mpc":
        return _simulate_milp(cfg, index, pv, load)

    raise ValueError(f"未知策略：{strategy!r}")


# =============================================================================
# 5. 指標（`05` §3）
# =============================================================================
def _purchase_cost(kw: pd.Series, cfg: AppConfig) -> float:
    """由「向市電購電功率序列」計算購電成本（元），自動處理一般式累進與時間電價。

    參數:
        kw: 購電功率序列（kW），index 為 DatetimeIndex。
        cfg: 設定物件。
    回傳:
        購電成本（元，不含基本費）。
    """
    dt_h = cfg.timestep_min / 60.0
    if cfg.tariff_scheme == "progressive":
        # 累進：依「每月用電度數」分段累加。
        kwh = kw * dt_h
        cost = 0.0
        for (yr, mo), g in kwh.groupby([kwh.index.year, kwh.index.month]):
            cost += pdm.progressive_monthly_bill(float(g.sum()), summer=mo in (6, 7, 8, 9))
        return cost
    # 時間電價：逐時單價 × 能量。
    price = np.array([pdm.get_price(ts, cfg) for ts in kw.index])
    return float((kw.to_numpy() * dt_h * price).sum())


def compute_metrics(df: pd.DataFrame, cfg: AppConfig) -> dict:
    """計算 SCR / SSR / 年省電費 / 回本年限等指標（見 `05` §3）。

    參數:
        df: run_simulation 的輸出。
        cfg: 設定物件。
    回傳:
        dict：各項指標（含基準對照）。
    """
    dt_h = cfg.timestep_min / 60.0
    E = lambda col: float((df[col] * dt_h).sum())  # 能量 kWh

    pv_total = E("pv_kw")
    load_total = E("load_kw")
    grid_in = E("grid_in_kw")
    grid_out = E("grid_out_kw")
    grid_to_batt = E("grid_to_batt_kw")
    curtail = E("pv_curtail_kw")

    # 自用率：被自用的太陽能 / 太陽能總發電。
    self_used_solar = pv_total - curtail - grid_out
    scr = self_used_solar / pv_total if pv_total > 0 else 0.0
    # 自給率：自家供應的負載 / 總負載（市電供負載 = 購電 − 市電充電池）。
    grid_to_load = grid_in - grid_to_batt
    ssr = (load_total - grid_to_load) / load_total if load_total > 0 else 0.0

    # 電費（年）：購電成本 − 售電收益 + 基本電費×月數。
    months = round(cfg.sim_days / 30.4)
    sell_rev = grid_out * cfg.tariff_sell_price
    basic = cfg.tariff_basic_fee * months
    bill = _purchase_cost(df["grid_in_kw"], cfg) - sell_rev + basic

    # 基準一：純太陽能、無電池（用來算「電池增益」）。
    deficit = (df["load_kw"] - df["pv_kw"]).clip(lower=0.0)
    surplus = (df["pv_kw"] - df["load_kw"]).clip(lower=0.0)
    pv_only_sell = float((surplus * dt_h).sum()) * cfg.tariff_sell_price
    pv_only_bill = _purchase_cost(deficit, cfg) - pv_only_sell + basic

    # 基準二：無太陽能、無電池（全部向市電買電；用來算「全系統」回本）。
    no_system_bill = _purchase_cost(df["load_kw"], cfg) + basic

    degradation = cfg.batt_deg_cost * (E("batt_ch_kw") + E("batt_dis_kw"))

    # 年省電費（兩種對照）。
    savings_total = no_system_bill - bill                 # 全系統（PV+電池）vs 無系統
    savings_battery = pv_only_bill - bill                 # 電池增益 vs 純太陽能
    net_battery_savings = savings_battery - degradation   # 扣電池衰減後的電池淨增益

    # 回本年限（基準與成本一致配對）。
    pv_cost = cfg.pv_capacity_kwp * cfg.pv_unit_cost_per_kwp
    batt_cost = cfg.batt_capacity_kwh * cfg.batt_unit_cost_per_kwh
    system_cost = pv_cost + batt_cost - cfg.subsidy_total
    payback_system = system_cost / savings_total if savings_total > 0 else float("inf")
    payback_battery = batt_cost / net_battery_savings if net_battery_savings > 0 else float("inf")

    return {
        "strategy": cfg.control_strategy,
        "tariff_scheme": cfg.tariff_scheme,
        "pv_capacity_kwp": cfg.pv_capacity_kwp,
        "batt_capacity_kwh": cfg.batt_capacity_kwh,
        "pv_total_kwh": pv_total,
        "load_total_kwh": load_total,
        "grid_in_kwh": grid_in,
        "curtail_kwh": curtail,
        "scr": scr,
        "ssr": ssr,
        "bill": bill,
        "pv_only_bill": pv_only_bill,
        "no_system_bill": no_system_bill,
        "savings_total": savings_total,
        "savings_battery": savings_battery,
        "degradation_cost": degradation,
        "net_battery_savings": net_battery_savings,
        "system_cost": system_cost,
        "battery_cost": batt_cost,
        "payback_system_years": payback_system,
        "payback_battery_years": payback_battery,
    }


def _eval_config(cfg: AppConfig) -> dict:
    """單一組設定的全年模擬 + 指標（供平行掃描的 worker；須為頂層函式以利 pickle）。"""
    return compute_metrics(run_simulation(cfg), cfg)


def sweep_configurations(
    base_cfg: AppConfig,
    pv_list: list[float],
    batt_list: list[float],
    strategy: str = "rule_based_forecast",
    processes: Optional[int] = None,
) -> pd.DataFrame:
    """對 PV 容量 × 電池容量做網格掃描，平行回測找最佳配置（見 `05` §4）。

    參數:
        base_cfg: 基底設定。
        pv_list: 要掃描的 PV 容量清單（kWp）。
        batt_list: 要掃描的電池容量清單（kWh）。
        strategy: 使用的控制策略（預設預測式規則）。
        processes: 平行程序數（None → 取 CPU 數與任務數較小者）。
    回傳:
        pd.DataFrame：每組配置一列，含 SCR/SSR/年省/回本等。
    """
    import multiprocessing as mp

    tasks = [
        dataclasses.replace(base_cfg, pv_capacity_kwp=p, batt_capacity_kwh=b,
                            control_strategy=strategy)
        for p in pv_list for b in batt_list
    ]
    n_proc = processes or min(mp.cpu_count(), len(tasks))
    with mp.Pool(processes=n_proc) as pool:
        results = pool.map(_eval_config, tasks)
    cols = ["pv_capacity_kwp", "batt_capacity_kwh", "scr", "ssr", "bill",
            "savings_total", "payback_system_years", "net_battery_savings",
            "payback_battery_years", "curtail_kwh"]
    return pd.DataFrame([{c: r[c] for c in cols} for r in results])


def compare_tariffs(
    base_cfg: AppConfig,
    schemes: tuple[str, ...] = ("progressive", "two_stage", "three_stage"),
    strategy: str = "rule_based_forecast",
) -> pd.DataFrame:
    """對同一系統配置比較不同電費方案的全年電費與省電（見 `04`、`05`）。

    參數:
        base_cfg: 系統配置（PV/電池等）。
        schemes: 要比較的電費類型。
        strategy: 控制策略。
    回傳:
        pd.DataFrame：每方案一列。
    """
    rows = []
    for scheme in schemes:
        cfg = dataclasses.replace(base_cfg, tariff_scheme=scheme, control_strategy=strategy)
        m = compute_metrics(run_simulation(cfg), cfg)
        rows.append({c: m[c] for c in ("tariff_scheme", "scr", "ssr", "bill",
                                       "no_system_bill", "savings_total",
                                       "payback_system_years")})
    return pd.DataFrame(rows)


def _eval_factorial(cfg: AppConfig) -> dict:
    """全因子單組 worker（頂層函式以利 pickle）：回傳該設定的指標。"""
    return compute_metrics(run_simulation(cfg), cfg)


def run_factorial(
    base_cfg: AppConfig,
    tariffs: list[str],
    loads: list[str],
    pv_list: list[float],
    batt_list: list[float],
    strategies: list[tuple[str, dict]],
    processes: Optional[int] = None,
) -> pd.DataFrame:
    """大型因子實驗：電費 × 負載 × PV × 電池 × 策略，全組合多核並行回測。

    參數:
        base_cfg: 基底設定。
        tariffs: 電費類型清單（progressive/two_stage/three_stage）。
        loads: 負載等級清單（medium/high）。
        pv_list: PV 容量清單（kWp）。
        batt_list: 電池容量清單（kWh）。
        strategies: [(策略名, 設定覆寫 dict)]；覆寫含 control_strategy/charge_priority/
                    discharge_only_peak 等。
        processes: 並行程序數（None → CPU 數）。
    回傳:
        pd.DataFrame：每組合一列，含 SCR/SSR/年電費/年省/回本等。
    """
    import multiprocessing as mp

    tasks: list[AppConfig] = []
    labels: list[tuple] = []
    for t in tariffs:
        for lv in loads:
            for p in pv_list:
                for b in batt_list:
                    for sname, sover in strategies:
                        cfg = dataclasses.replace(
                            base_cfg, tariff_scheme=t, load_level=lv,
                            pv_capacity_kwp=p, batt_capacity_kwh=b, **sover,
                        )
                        tasks.append(cfg)
                        labels.append((t, lv, p, b, sname))

    n_proc = processes or min(mp.cpu_count(), len(tasks))
    with mp.Pool(processes=n_proc) as pool:
        results = pool.map(_eval_factorial, tasks)

    rows = []
    for (t, lv, p, b, sname), m in zip(labels, results):
        rows.append({
            "tariff": t, "load": lv, "pv_kwp": p, "batt_kwh": b, "strategy": sname,
            "scr": m["scr"], "ssr": m["ssr"], "bill": m["bill"],
            "savings_total": m["savings_total"],
            "payback_system_years": m["payback_system_years"],
            "net_battery_savings": m["net_battery_savings"],
            "payback_battery_years": m["payback_battery_years"],
        })
    return pd.DataFrame(rows)


def _save_outputs(df: pd.DataFrame, metrics: dict, cfg: AppConfig, out_dir: str) -> tuple[str, str]:
    """存出 CSV 與一張代表日的調度圖。回傳 (csv_path, png_path)。"""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"hems_result_{cfg.control_strategy}.csv")
    df.to_csv(csv_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 取一個夏日（7/15）畫日內調度。
    day = df[(df.index.month == 7) & (df.index.day == 15)]
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(day.index, day["pv_kw"], color="#EF9F27", label="PV")
    ax1.plot(day.index, day["load_kw"], color="#D85A30", ls="--", label="Load")
    ax1.plot(day.index, day["grid_in_kw"], color="#378ADD", label="Grid in")
    ax1.set_ylabel("Power (kW)")
    ax1.set_xlabel("Hour (Jul 15)")
    ax2 = ax1.twinx()
    ax2.plot(day.index, day["soc"] * 100, color="#1D9E75", lw=2, label="SOC")
    ax2.set_ylabel("SOC (%)"); ax2.set_ylim(0, 100)
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax1.set_title(f"Kaohsiung HEMS dispatch ({cfg.control_strategy}) - summer day")
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"hems_day_{cfg.control_strategy}.png")
    fig.savefig(png_path, dpi=110); plt.close(fig)
    return csv_path, png_path


def main() -> None:
    """讀設定 → 跑模擬 → 存 output/ → 印摘要。"""
    cfg = load_config()
    print(f"=== HEMS 全年模擬（高雄前鎮區，策略={cfg.control_strategy}）===")
    df = run_simulation(cfg)
    m = compute_metrics(df, cfg)
    csv_path, png_path = _save_outputs(df, m, cfg, "output")

    print(f"  太陽能總發電 {m['pv_total_kwh']:,.0f} kWh | 總負載 {m['load_total_kwh']:,.0f} kWh")
    print(f"  自用率 SCR = {m['scr']*100:.1f}%  自給率 SSR = {m['ssr']*100:.1f}%")
    print(f"  年電費（PV+電池）{m['bill']:,.0f} 元")
    print(f"  全系統年省（vs 無太陽能無電池）{m['savings_total']:,.0f} 元 → 全系統回本 ≈ {m['payback_system_years']:.1f} 年")
    print(f"  電池增益（vs 純太陽能）{m['savings_battery']:,.0f} 元；扣衰減 {m['degradation_cost']:,.0f} 後淨增益 {m['net_battery_savings']:,.0f} 元 → 電池回本 ≈ {m['payback_battery_years']:.1f} 年")
    print(f"  已輸出：{csv_path}、{png_path}")
    print("  ⚠️ 費率與單價會變動，結果為決策參考、非投資建議。")


if __name__ == "__main__":
    main()
