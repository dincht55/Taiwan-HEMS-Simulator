"""solar_pv.py — 模組 2：太陽能發電模擬（依地點與時間）。

職責（對應 `02_電學與太陽能模型`）：
- 首選 pvlib 流程：太陽位置 → 晴空輻照 → 傾斜面輻照(POA) → 直流發電 → 交流端。
- 把「高雄市前鎮區」實際氣候平年值套進來：
  以各月日照時數（Ångström–Prescott 關係）將樂觀的晴空輻照「縮放」到高雄真實多雲程度，
  並以各月氣溫做 PV 溫度降額（高溫減發電）。
- 無 pvlib / 離線時，退回 `02` §4.2 的簡化鐘形模型（精度低，會標明）。

回傳：每時步 `pv_kw`（交流端、已扣系統損耗與逆變器效率），index 與輸入對齊。

相依方向：本檔 import 同層 `config` 與更低層 `electrical_base`，不 import 更高層。
所有可調數值來自 `config.py`；模型常數（如 A-P 係數）以具名常數呈現並註明出處。

⚠️ 高雄氣候平年值資料來源：中央氣象署 1991–2020 氣候平均（「高雄」測站，前鎮區），
   輔以公開氣候彙整。各來源日照時數略有差異，此處為「合理代表值」，
   正式經濟分析建議改用台電/氣象署官方 TMY 或實測逐時輻照 CSV（見 simulate_pv 的 csv 介面）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import electrical_base as eb  # 更低層：效率套用等
from config import AppConfig, load_config

# pvlib 為首選但選用相依；缺少時自動退回簡化模型。
try:
    import pvlib  # type: ignore

    _HAS_PVLIB = True
except ImportError:  # pragma: no cover
    _HAS_PVLIB = False


# =============================================================================
# 高雄市前鎮區 月氣候平年值（中央氣象署 1991–2020「高雄」測站，前鎮區一帶）
#   - temp_c：月平均氣溫（°C），用於 PV 溫度降額。
#   - sun_h：月平均「每日」日照時數（小時/日），用於估算多雲縮放（晴空→實際）。
# 月份鍵 1~12。數值為合理代表值（多來源彼此略有差異），詳見檔頭 ⚠️ 說明。
# =============================================================================
KAOHSIUNG_MONTHLY_NORMALS: dict[int, dict[str, float]] = {
    1:  {"temp_c": 19.3, "sun_h": 6.0},
    2:  {"temp_c": 20.4, "sun_h": 5.5},
    3:  {"temp_c": 22.6, "sun_h": 6.0},
    4:  {"temp_c": 25.4, "sun_h": 6.0},
    5:  {"temp_c": 27.7, "sun_h": 6.2},
    6:  {"temp_c": 28.9, "sun_h": 6.3},
    7:  {"temp_c": 29.5, "sun_h": 7.4},
    8:  {"temp_c": 29.2, "sun_h": 6.5},
    9:  {"temp_c": 28.5, "sun_h": 6.7},
    10: {"temp_c": 26.7, "sun_h": 6.9},
    11: {"temp_c": 24.1, "sun_h": 6.2},
    12: {"temp_c": 20.6, "sun_h": 5.4},
}

# 在地氣候檔（由 setup_config.py 依使用者地區線上查詢後產生）。
# 格式：CSV，欄位 month(1~12), temp_c(月均氣溫°C), sun_h(月均每日日照時數,小時/日)。
_CLIMATE_NORMALS_CSV: str = "data/climate_normals.csv"


def load_monthly_normals(path: str = _CLIMATE_NORMALS_CSV) -> dict[int, dict[str, float]]:
    """載入月氣候平年值：有在地氣候檔就用它，否則退回內建高雄預設。

    參數:
        path: 在地氣候檔路徑（CSV，欄位 month/temp_c/sun_h）。
    回傳:
        dict[int, dict]：{月份(1~12): {'temp_c': 氣溫°C, 'sun_h': 日照時數}}。
    """
    import csv
    import os

    if not os.path.exists(path):
        return KAOHSIUNG_MONTHLY_NORMALS
    normals: dict[int, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = int(float(row["month"]))
            normals[m] = {"temp_c": float(row["temp_c"]), "sun_h": float(row["sun_h"])}
    # 確保 12 個月齊全，缺月以高雄預設補（避免 KeyError）。
    for m in range(1, 13):
        normals.setdefault(m, KAOHSIUNG_MONTHLY_NORMALS[m])
    return normals


# 模組載入時決定使用哪份平年值（程式啟動後唯讀；換地區重跑即可生效）。
MONTHLY_NORMALS: dict[int, dict[str, float]] = load_monthly_normals()

# Ångström–Prescott 係數：晴空指數 Kt ≈ A + B×(n/N)，n=實際日照時數、N=可照時數。
# A、B 為地區經驗常數（典型 A≈0.25、B≈0.50），此處採通用值；可日後依在地實測校正。
ANGSTROM_A: float = 0.25
ANGSTROM_B: float = 0.50

# 晴空指數上限（避免縮放因子超過晴空本身）。
_KT_MAX: float = 0.85

# 簡化模型用：把晴空輻照轉成 1m² 標準測試輻照(STC)的參考值（W/m²）。
_STC_IRRADIANCE_WM2: float = 1000.0

__all__ = [
    "simulate_pv",
    "monthly_energy_summary",
    "load_monthly_normals",
    "MONTHLY_NORMALS",
    "KAOHSIUNG_MONTHLY_NORMALS",
]


# --------------------------------------------------------------------------- #
# 工具：時間索引處理
# --------------------------------------------------------------------------- #
def _localize(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """確保索引為帶時區（tz-aware）；pvlib 計算需要在地時間。

    參數:
        index: 輸入時間索引（可能無時區）。
        tz: IANA 時區字串（如 'Asia/Taipei'）。
    回傳:
        帶時區的 DatetimeIndex。
    """
    if index.tz is None:
        return index.tz_localize(tz)
    return index.tz_convert(tz)


def _timestep_hours(index: pd.DatetimeIndex) -> float:
    """由索引推得時步長度（小時）。

    參數:
        index: 等間隔時間索引。
    回傳:
        時步（小時）；無法判定時退回 1.0。
    """
    if len(index) < 2:
        return 1.0
    delta_min = (index[1] - index[0]).total_seconds() / 60.0
    return delta_min / 60.0


# --------------------------------------------------------------------------- #
# 高雄氣候套用：多雲縮放因子與逐時氣溫
# --------------------------------------------------------------------------- #
def _monthly_temp_series(index: pd.DatetimeIndex) -> pd.Series:
    """依高雄月平均氣溫，展開成逐時的環境氣溫序列（°C）。

    參數:
        index: 時間索引。
    回傳:
        pd.Series：每時步的環境氣溫（°C）。
    """
    temps = np.array([MONTHLY_NORMALS[m]["temp_c"] for m in index.month])
    return pd.Series(temps, index=index, name="temp_air_c")


def _monthly_cloud_scale(
    clearsky_ghi: pd.Series,
    ghi_extra: pd.Series,
    index: pd.DatetimeIndex,
    dt_h: float,
) -> pd.Series:
    """計算各月「晴空→高雄實際」的多雲縮放因子，並展開成逐時序列。

    方法:
        1. 由高雄各月日照時數 n 與該月平均可照時數 N，算目標晴空指數
           Kt_target = clip(A + B·(n/N), 0, _KT_MAX)。
        2. 由 pvlib 晴空輻照得該月晴空指數 Kt_cs = ΣGHI_cs / ΣGHI_extra。
        3. 縮放因子 scale = clip(Kt_target / Kt_cs, 0, 1)，乘到該月每個時步的天空輻照。

    參數:
        clearsky_ghi: pvlib 晴空 GHI（W/m²）。
        ghi_extra: 大氣外水平輻照 GHI_extra（W/m²）。
        index: 時間索引（tz-aware）。
        dt_h: 時步（小時）。
    回傳:
        pd.Series：每時步的縮放因子（0~1）。
    """
    months = index.month
    is_day = ghi_extra.to_numpy() > 0.0  # 白天（有大氣外輻照）

    scale_by_month: dict[int, float] = {}
    for m in range(1, 13):
        m_mask = (months == m)
        if not m_mask.any():
            continue
        day_mask = m_mask & is_day

        # 該月平均每日可照時數 N（小時/日）。
        n_days = max(1, len(np.unique(index[m_mask].date)))
        possible_sun_h = is_day[m_mask].sum() * dt_h / n_days
        n_over_N = 0.0 if possible_sun_h <= 0 else (
            MONTHLY_NORMALS[m]["sun_h"] / possible_sun_h
        )
        kt_target = float(np.clip(ANGSTROM_A + ANGSTROM_B * n_over_N, 0.0, _KT_MAX))

        # 該月晴空指數 Kt_cs。
        sum_cs = clearsky_ghi.to_numpy()[day_mask].sum()
        sum_ex = ghi_extra.to_numpy()[day_mask].sum()
        kt_cs = (sum_cs / sum_ex) if sum_ex > 0 else 1.0

        scale_by_month[m] = float(np.clip(kt_target / kt_cs, 0.0, 1.0)) if kt_cs > 0 else 1.0

    scale = np.array([scale_by_month.get(m, 1.0) for m in months])
    return pd.Series(scale, index=index, name="cloud_scale")


# --------------------------------------------------------------------------- #
# pvlib 主流程
# --------------------------------------------------------------------------- #
def _simulate_pv_pvlib(cfg: AppConfig, index: pd.DatetimeIndex) -> pd.Series:
    """pvlib 流程：晴空輻照 → 套高雄多雲縮放與氣溫 → POA → 直流 → 交流。

    參數:
        cfg: 設定物件。
        index: 時間索引（tz-aware）。
    回傳:
        pd.Series：每時步 pv_kw（交流端、已扣損耗）。
    """
    location = pvlib.location.Location(
        latitude=cfg.latitude, longitude=cfg.longitude,
        tz=cfg.timezone, altitude=cfg.altitude_m,
    )
    solpos = location.get_solarposition(index)
    clearsky = location.get_clearsky(index, model="ineichen")  # ghi/dni/dhi

    # 大氣外水平輻照（供晴空指數計算）。
    dni_extra = pvlib.irradiance.get_extra_radiation(index)
    cos_zen = np.cos(np.radians(solpos["apparent_zenith"])).clip(lower=0.0)
    ghi_extra = pd.Series(dni_extra.to_numpy() * cos_zen.to_numpy(),
                          index=index, name="ghi_extra")

    # 1) 高雄多雲縮放（把晴空輻照壓到實際水準）。
    dt_h = _timestep_hours(index)
    scale = _monthly_cloud_scale(clearsky["ghi"], ghi_extra, index, dt_h)
    ghi = clearsky["ghi"] * scale
    dni = clearsky["dni"] * scale
    dhi = clearsky["dhi"] * scale

    # 2) 傾斜面輻照 POA。
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=cfg.pv_tilt_deg,
        surface_azimuth=cfg.pv_azimuth_deg,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=dni, ghi=ghi, dhi=dhi,
        dni_extra=dni_extra, model="haydavies",
    )
    poa_global = poa["poa_global"].fillna(0.0).clip(lower=0.0)

    # 3) 電池溫度（NOCT 簡化模型）：T_cell = T_air + (NOCT-20)/800 × POA。
    temp_air = _monthly_temp_series(index)
    temp_cell = temp_air + (cfg.pv_noct_c - 20.0) / 800.0 * poa_global

    # 4) 直流發電（PVWatts）：pdc0 用裝置容量（W）。
    pdc0_w = cfg.pv_capacity_kwp * 1000.0
    # 以位置引數呼叫，相容 pvlib 新舊版（參數曾由 g_poa_effective 更名為 effective_irradiance）。
    dc_w = pvlib.pvsystem.pvwatts_dc(
        poa_global, temp_cell, pdc0_w, cfg.pv_gamma_pdc,
    ).clip(lower=0.0)

    # 5) 交流端：乘逆變器效率，再乘 (1 - 系統損耗)。
    pv_ac_kw = eb.apply_efficiency(dc_w / 1000.0, cfg.pv_inverter_eff) * (1.0 - cfg.pv_system_loss)
    return pd.Series(pv_ac_kw, index=index, name="pv_kw").clip(lower=0.0)


# --------------------------------------------------------------------------- #
# 簡化退回模型（無 pvlib）
# --------------------------------------------------------------------------- #
def _day_length_hours(latitude_deg: float, day_of_year: int) -> float:
    """以天文公式估算某日的可照時數（小時）。

    參數:
        latitude_deg: 緯度（度）。
        day_of_year: 一年中的第幾天（1~365/366）。
    回傳:
        日長（小時）。
    """
    lat = np.radians(latitude_deg)
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + day_of_year) / 365.0)
    cos_ws = -np.tan(lat) * np.tan(decl)
    cos_ws = float(np.clip(cos_ws, -1.0, 1.0))
    ws = np.arccos(cos_ws)             # 日落時角（弧度）
    return float(24.0 / np.pi * ws)


def _simulate_pv_simple(cfg: AppConfig, index: pd.DatetimeIndex) -> pd.Series:
    """簡化鐘形模型（無 pvlib 時退回，見 `02` §4.2；精度低，已標明）。

    以日出~日落間的 sin 鐘形近似日照，並乘上高雄各月晴空指數 Kt 反映多雲，
    再乘裝置容量、系統損耗與逆變器效率。

    參數:
        cfg: 設定物件。
        index: 時間索引（tz-aware）。
    回傳:
        pd.Series：每時步 pv_kw（交流端、近似值）。
    """
    out = np.zeros(len(index), dtype=float)
    hours_float = index.hour + index.minute / 60.0
    for i, ts in enumerate(index):
        doy = ts.dayofyear
        day_len = _day_length_hours(cfg.latitude, doy)
        sunrise = 12.0 - day_len / 2.0
        sunset = 12.0 + day_len / 2.0
        t = hours_float[i]
        if sunrise < t < sunset:
            shape = max(0.0, np.sin(np.pi * (t - sunrise) / (sunset - sunrise)))
            # 該月晴空指數當作整體多雲折減（近似）。
            n = MONTHLY_NORMALS[ts.month]["sun_h"]
            kt = float(np.clip(ANGSTROM_A + ANGSTROM_B * (n / day_len), 0.0, _KT_MAX))
            dc_kw = cfg.pv_capacity_kwp * shape * kt
            out[i] = eb.apply_efficiency(dc_kw, cfg.pv_inverter_eff) * (1.0 - cfg.pv_system_loss)
    return pd.Series(out, index=index, name="pv_kw").clip(lower=0.0)


# --------------------------------------------------------------------------- #
# 對外主函式
# --------------------------------------------------------------------------- #
def simulate_pv(
    cfg: AppConfig,
    index: pd.DatetimeIndex,
    irradiance_csv: str | None = None,
) -> pd.Series:
    """模擬每時步太陽能發電（交流端、已扣損耗）。

    來源優先序:
        1. 若提供 irradiance_csv（含 ghi/dni/dhi 逐時實測或 TMY）：未來可接此路徑（保留介面）。
        2. 有 pvlib：晴空輻照 + 高雄氣候縮放 + 溫度降額。
        3. 無 pvlib：簡化鐘形模型（精度低）。

    參數:
        cfg: 設定物件（提供位置、傾角方位、容量、效率、溫度係數）。
        index: 時間索引；若無時區會自動以 cfg.timezone 在地化。
        irradiance_csv: 選用，實測/TMY 輻照檔路徑（保留介面，目前未使用）。
    回傳:
        pd.Series：每時步 pv_kw（kW，交流端）。
    """
    idx = _localize(index, cfg.timezone)

    if irradiance_csv is not None:
        # 介面預留：之後可讀官方 TMY/實測 CSV，將更精確。此版尚未實作。
        raise NotImplementedError(
            "irradiance_csv 讀檔尚未實作；目前以氣候平年值縮放晴空模型。"
        )

    if _HAS_PVLIB:
        pv = _simulate_pv_pvlib(cfg, idx)
    else:
        pv = _simulate_pv_simple(cfg, idx)

    pv.index = index  # 還原成呼叫端原本的索引（時區一致即可）
    return pv


def monthly_energy_summary(pv_kw: pd.Series, cfg: AppConfig) -> pd.DataFrame:
    """彙整每月發電量與每 kWp 的單位發電（specific yield），供回報與驗證。

    參數:
        pv_kw: simulate_pv 的輸出（kW）。
        cfg: 設定物件（取裝置容量）。
    回傳:
        pd.DataFrame：index=月份(1~12)，欄位 energy_kwh、specific_kwh_per_kwp。
    """
    dt_h = _timestep_hours(pv_kw.index)
    energy_kwh = pv_kw * dt_h                      # 每時步能量
    by_month = energy_kwh.groupby(pv_kw.index.month).sum()
    df = pd.DataFrame({"energy_kwh": by_month})
    df["specific_kwh_per_kwp"] = df["energy_kwh"] / cfg.pv_capacity_kwp
    df.index.name = "month"
    return df


if __name__ == "__main__":
    print("=== solar_pv 示範（高雄市前鎮區，套用氣象署氣候平年值）===")
    cfg = load_config()
    print(f"  引擎：{'pvlib' if _HAS_PVLIB else '簡化鐘形模型（無 pvlib）'}")
    print(f"  位置：lat={cfg.latitude}, lon={cfg.longitude}, alt={cfg.altitude_m} m")
    print(f"  系統：{cfg.pv_capacity_kwp} kWp, 傾角 {cfg.pv_tilt_deg}°, 方位 {cfg.pv_azimuth_deg}°")

    # 建一整年、15 分鐘解析度的索引。
    periods = int(365 * 24 * 60 / cfg.timestep_min)
    index = pd.date_range(cfg.start_date, periods=periods,
                          freq=f"{cfg.timestep_min}min", tz=cfg.timezone)

    pv = simulate_pv(cfg, index)
    summary = monthly_energy_summary(pv, cfg)

    total_kwh = float(summary["energy_kwh"].sum())
    specific = total_kwh / cfg.pv_capacity_kwp
    print("\n  月份  發電量(kWh)   單位發電(kWh/kWp)")
    for m, row in summary.iterrows():
        print(f"   {m:>2}    {row['energy_kwh']:>8.1f}      {row['specific_kwh_per_kwp']:>7.1f}")
    print(f"\n  全年總發電 ≈ {total_kwh:,.0f} kWh")
    print(f"  單位發電(年) ≈ {specific:,.0f} kWh/kWp（高雄典型約 1,250~1,450）")
    print("\n  ⚠️ 日照採氣象署 1991–2020 高雄氣候平年值縮放晴空模型；")
    print("     正式分析建議改用官方 TMY/實測逐時輻照，結果為決策參考非投資建議。")
