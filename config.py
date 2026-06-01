"""config.py — HEMS 模擬器的「單一設定來源」。

依 `01_系統架構` §2 定義設定物件 `AppConfig`，並提供 `load_config()`：
- 內含一組「台灣常見住宅情境」的合理預設值（繁中註解、單位標註）。
- 若同目錄存在 `.env`，則以 `.env` 內的同名鍵覆寫預設值（本機執行用）。

設計原則（單一設定來源原則）：
- 所有可調數值集中在此，程式其他地方不得散落硬編碼魔術數字。
- 欄位名稱、單位與 `01_系統架構` 的 `AppConfig` 一致；本檔不重抄領域公式。

⚠️ 電價費率僅為「初始參考值」：台電約每年 4 月、10 月調整，
   正式分析請向台電官網（電價表）確認當期數字，並視需要由 `.env` 覆寫。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Callable

# python-dotenv 為選用相依；缺少時不影響「純預設值」流程。
try:
    from dotenv import load_dotenv  # type: ignore

    _HAS_DOTENV = True
except ImportError:  # pragma: no cover - 環境無 dotenv 時退回純預設
    _HAS_DOTENV = False


# =============================================================================
# 電費類型（tariff_scheme）合法值與中文標籤
# 一般式＝累進電價（不分尖離峰）；二段式／三段式＝時間電價（分尖離峰，見 `04`）。
# 注意：唯有時間電價（two_stage/three_stage）才有「削峰填谷／尖峰套利」空間。
# =============================================================================
TARIFF_SCHEME_LABELS: dict[str, str] = {
    "progressive": "一般式（累進電價，不分尖離峰）",
    "two_stage": "二段式（時間電價）",
    "three_stage": "三段式（時間電價）",
}
VALID_TARIFF_SCHEMES: frozenset[str] = frozenset(TARIFF_SCHEME_LABELS)


# =============================================================================
# 設定物件：AppConfig
# 欄位對應 `01_系統架構` §2；註解中的大寫名稱即為 `.env` 可覆寫的鍵名。
# 凍結（frozen）以符合「載入後唯讀往下傳遞」原則。
# =============================================================================
@dataclass(frozen=True)
class AppConfig:
    """HEMS 模擬器全域設定（載入後唯讀）。

    單位一律標註於欄位名或註解：功率 kW、能量 kWh、時間 分鐘、角度 度、效率 0~1。
    """

    # --- 位置（pvlib Location 用）---
    latitude: float          # HEMS_LATITUDE   緯度（度）
    longitude: float         # HEMS_LONGITUDE  經度（度）
    timezone: str            # HEMS_TIMEZONE   IANA 時區字串
    altitude_m: float        # HEMS_ALTITUDE   海拔（公尺 m）

    # --- 太陽能（模組 2）---
    pv_capacity_kwp: float   # PV_CAPACITY_KWP  裝置容量（kWp）
    pv_tilt_deg: float       # PV_TILT_DEG      傾角（度）
    pv_azimuth_deg: float    # PV_AZIMUTH_DEG   方位角（度，180=朝南）
    pv_system_loss: float    # PV_SYSTEM_LOSS   系統損耗比例（0~1）
    pv_inverter_eff: float   # PV_INVERTER_EFF  PV 逆變器效率（0~1）
    pv_gamma_pdc: float      # PV_GAMMA_PDC     功率溫度係數（每 °C，通常為負，如 -0.004）
    pv_noct_c: float         # PV_NOCT_C        標稱電池工作溫度 NOCT（°C）

    # --- 耦合架構（決定太陽能→電池要乘幾次效率，見 `02` §3）---
    coupling: str            # PV_COUPLING      "dc"（預設）或 "ac"
    batt_inverter_eff: float # BATT_INVERTER_EFF 電池逆變器/整流(AC-DC)效率（僅 AC 耦合用，0~1）

    # --- 電池（模組 3、4 共用，見 `03`）---
    batt_capacity_kwh: float      # BATT_CAPACITY_KWH      可用容量（kWh）
    batt_soc_min: float           # BATT_SOC_MIN           SOC 下限（0~1）
    batt_soc_max: float           # BATT_SOC_MAX           SOC 上限（0~1）
    batt_max_charge_kw: float     # BATT_MAX_CHARGE_KW     最大充電功率（kW）
    batt_max_discharge_kw: float  # BATT_MAX_DISCHARGE_KW  最大放電功率（kW）
    batt_charge_eff: float        # BATT_CHARGE_EFF        充電效率 η_ch（0~1）
    batt_discharge_eff: float     # BATT_DISCHARGE_EFF     放電效率 η_dis（0~1）
    batt_deg_cost: float          # BATT_DEGRADATION_COST  衰減成本（元/kWh 吞吐）

    # --- 電價（數值；時段定義見 `04`）---
    tariff_scheme: str       # TARIFF_SCHEME  電費類型：progressive(一般式)/two_stage(二段式)/three_stage(三段式)
    tariff_basic_fee: float  # TARIFF_BASIC_FEE   基本電費（元/月）
    tariff_sell_price: float # TARIFF_SELL_PRICE  本次模擬實際售電價（元/度）；自用型未登記躉售=0
    tariff_feed_in_price: float # TARIFF_FEED_IN_PRICE 躉購(FiT)參考價（元/度）；feedin 模式評估「若登記躉售」用
    tariff_peak: float       # TARIFF_PEAK        尖峰單價（元/度）
    tariff_mid: float        # TARIFF_MID         半尖峰單價（元/度）
    tariff_offpeak: float    # TARIFF_OFFPEAK     離峰單價（元/度）

    # --- 模擬 ---
    timestep_min: int        # SIM_TIMESTEP_MIN     時步（分鐘）
    start_date: str          # SIM_START_DATE       起始日期（YYYY-MM-DD）
    sim_days: int            # SIM_DAYS             模擬天數（天）
    control_strategy: str    # SIM_CONTROL_STRATEGY "rule_based" 或 "milp_mpc"
    load_profile_csv: str    # LOAD_PROFILE_CSV     負載資料 CSV 路徑
    load_level: str          # LOAD_LEVEL           合成負載等級 "medium"/"high"（CSV 存在時忽略）
    charge_priority: str     # CHARGE_PRIORITY      "load_first"(PV先供電) / "battery_first"(PV先充滿再供電)
    offpeak_charge_target_soc: float  # OFFPEAK_CHARGE_TARGET_SOC 離峰用市電補電池的目標SOC（≤SOC_min=關閉）
    smart_discharge: bool             # SMART_DISCHARGE  是否啟用智慧放電門檻（衰減成本閘＋尖峰保留）
    mid_discharge_reserve_soc: float  # MID_DISCHARGE_RESERVE_SOC 半尖峰時為尖峰保留的最低SOC（放電端預測設定）
    discharge_only_peak: bool         # DISCHARGE_ONLY_PEAK  只在最貴時段(尖峰)放電（半尖峰完全保留）
    # 經濟試算（用於回本年限；單價會變動，僅為參考）
    pv_unit_cost_per_kwp: float    # PV_UNIT_COST_PER_KWP   太陽能建置單價（元/kWp）
    batt_unit_cost_per_kwh: float  # BATT_UNIT_COST_PER_KWH 電池建置單價（元/kWh）
    subsidy_total: float           # SUBSIDY_TOTAL          補助總額（元，扣減建置成本）


# =============================================================================
# 預設值（台灣常見住宅情境）
# 每個鍵對應 AppConfig 的一個欄位，並標注對應 `.env` 鍵名。
# 電價沿用台電「住商型簡易時間電價（三段式）」2025/10/01 起參考值。
# =============================================================================
_DEFAULTS: dict[str, Any] = {
    # 位置（高雄市前鎮區，中央氣象署「高雄」測站 467441 一帶）
    "latitude": 22.5655,         # HEMS_LATITUDE  (度)
    "longitude": 120.3157,       # HEMS_LONGITUDE (度)
    "timezone": "Asia/Taipei",   # HEMS_TIMEZONE
    "altitude_m": 2.3,           # HEMS_ALTITUDE  (m)
    # 太陽能（約 5 kWp 家用系統）
    "pv_capacity_kwp": 5.0,      # PV_CAPACITY_KWP (kWp)
    "pv_tilt_deg": 23.5,         # PV_TILT_DEG     (度，約等於在地緯度)
    "pv_azimuth_deg": 180.0,     # PV_AZIMUTH_DEG  (度，朝南)
    "pv_system_loss": 0.14,      # PV_SYSTEM_LOSS  (0~1，PVWatts 典型值)
    "pv_inverter_eff": 0.96,     # PV_INVERTER_EFF (0~1)
    "pv_gamma_pdc": -0.004,      # PV_GAMMA_PDC    (每 °C；矽晶典型 -0.4%/°C)
    "pv_noct_c": 45.0,           # PV_NOCT_C       (°C；標稱工作溫度)
    # 耦合（預設 DC，可切 AC）
    "coupling": "dc",            # PV_COUPLING  "dc"/"ac"
    "batt_inverter_eff": 0.96,   # BATT_INVERTER_EFF (0~1，AC 耦合才生效)
    # 電池（約 10 kWh 家用電池）
    "batt_capacity_kwh": 10.0,       # BATT_CAPACITY_KWH      (kWh)
    "batt_soc_min": 0.10,            # BATT_SOC_MIN           (0~1)
    "batt_soc_max": 0.95,            # BATT_SOC_MAX           (0~1)
    "batt_max_charge_kw": 5.0,       # BATT_MAX_CHARGE_KW     (kW)
    "batt_max_discharge_kw": 5.0,    # BATT_MAX_DISCHARGE_KW  (kW)
    "batt_charge_eff": 0.95,         # BATT_CHARGE_EFF        (0~1)
    "batt_discharge_eff": 0.95,      # BATT_DISCHARGE_EFF     (0~1)
    "batt_deg_cost": 1.5,            # BATT_DEGRADATION_COST  (元/kWh 吞吐)
    # 電價（⚠️ 參考值，需向台電確認當期數字）
    "tariff_scheme": "three_stage",  # TARIFF_SCHEME
    "tariff_basic_fee": 75.0,        # TARIFF_BASIC_FEE   (元/月)
    "tariff_sell_price": 0.0,        # TARIFF_SELL_PRICE  (元/度，自用型未登記躉售=0)
    "tariff_feed_in_price": 5.6279,  # TARIFF_FEED_IN_PRICE 躉購FiT(元/度)：115年度屋頂型≤10kWp；逐年公告
    "tariff_peak": 7.13,             # TARIFF_PEAK        (元/度)
    "tariff_mid": 4.69,              # TARIFF_MID         (元/度)
    "tariff_offpeak": 2.06,          # TARIFF_OFFPEAK     (元/度)
    # 模擬（預設起始於夏月，以利測試夏月尖峰邏輯）
    "timestep_min": 15,                  # SIM_TIMESTEP_MIN     (分鐘)
    "start_date": "2025-01-01",          # SIM_START_DATE       (YYYY-MM-DD)
    "sim_days": 365,                     # SIM_DAYS             (天)
    "control_strategy": "rule_based",    # SIM_CONTROL_STRATEGY
    "load_profile_csv": "data/load_profile.csv",  # LOAD_PROFILE_CSV
    "load_level": "high",                # LOAD_LEVEL  "medium"/"high"
    "charge_priority": "load_first",     # CHARGE_PRIORITY  "load_first"/"battery_first"
    # 離峰用市電補電池的目標 SOC：預設 0.0（≤SOC_min 視為關閉，純反應式自用優先）。
    # 這是「非預測式填谷」；真正依預測的離峰充電由模組 5 的 MILP 處理。
    "offpeak_charge_target_soc": 0.0,    # OFFPEAK_CHARGE_TARGET_SOC
    # 智慧放電（預設關閉；rule_based_forecast 策略會自動啟用並逐日設定保留 SOC）：
    "smart_discharge": False,            # SMART_DISCHARGE
    "mid_discharge_reserve_soc": 0.0,    # MID_DISCHARGE_RESERVE_SOC（≤SOC_min=不保留）
    "discharge_only_peak": False,        # DISCHARGE_ONLY_PEAK
    # 經濟試算（⚠️ 參考單價，會變動）：太陽能約 4~6 萬/kWp、電池約 1.5~2.5 萬/kWh。
    "pv_unit_cost_per_kwp": 50000.0,     # PV_UNIT_COST_PER_KWP   (元/kWp)
    "batt_unit_cost_per_kwh": 18000.0,   # BATT_UNIT_COST_PER_KWH (元/kWh)
    "subsidy_total": 0.0,                # SUBSIDY_TOTAL          (元)
}

# 欄位名 → 對應的 .env 鍵名（大寫）。
_ENV_KEYS: dict[str, str] = {
    "latitude": "HEMS_LATITUDE",
    "longitude": "HEMS_LONGITUDE",
    "timezone": "HEMS_TIMEZONE",
    "altitude_m": "HEMS_ALTITUDE",
    "pv_capacity_kwp": "PV_CAPACITY_KWP",
    "pv_tilt_deg": "PV_TILT_DEG",
    "pv_azimuth_deg": "PV_AZIMUTH_DEG",
    "pv_system_loss": "PV_SYSTEM_LOSS",
    "pv_inverter_eff": "PV_INVERTER_EFF",
    "pv_gamma_pdc": "PV_GAMMA_PDC",
    "pv_noct_c": "PV_NOCT_C",
    "coupling": "PV_COUPLING",
    "batt_inverter_eff": "BATT_INVERTER_EFF",
    "batt_capacity_kwh": "BATT_CAPACITY_KWH",
    "batt_soc_min": "BATT_SOC_MIN",
    "batt_soc_max": "BATT_SOC_MAX",
    "batt_max_charge_kw": "BATT_MAX_CHARGE_KW",
    "batt_max_discharge_kw": "BATT_MAX_DISCHARGE_KW",
    "batt_charge_eff": "BATT_CHARGE_EFF",
    "batt_discharge_eff": "BATT_DISCHARGE_EFF",
    "batt_deg_cost": "BATT_DEGRADATION_COST",
    "tariff_scheme": "TARIFF_SCHEME",
    "tariff_basic_fee": "TARIFF_BASIC_FEE",
    "tariff_sell_price": "TARIFF_SELL_PRICE",
    "tariff_feed_in_price": "TARIFF_FEED_IN_PRICE",
    "tariff_peak": "TARIFF_PEAK",
    "tariff_mid": "TARIFF_MID",
    "tariff_offpeak": "TARIFF_OFFPEAK",
    "timestep_min": "SIM_TIMESTEP_MIN",
    "start_date": "SIM_START_DATE",
    "sim_days": "SIM_DAYS",
    "control_strategy": "SIM_CONTROL_STRATEGY",
    "load_profile_csv": "LOAD_PROFILE_CSV",
    "load_level": "LOAD_LEVEL",
    "charge_priority": "CHARGE_PRIORITY",
    "offpeak_charge_target_soc": "OFFPEAK_CHARGE_TARGET_SOC",
    "smart_discharge": "SMART_DISCHARGE",
    "mid_discharge_reserve_soc": "MID_DISCHARGE_RESERVE_SOC",
    "discharge_only_peak": "DISCHARGE_ONLY_PEAK",
    "pv_unit_cost_per_kwp": "PV_UNIT_COST_PER_KWP",
    "batt_unit_cost_per_kwh": "BATT_UNIT_COST_PER_KWH",
    "subsidy_total": "SUBSIDY_TOTAL",
}


def _coerce(raw: str, target_type: type) -> Any:
    """把 .env 讀到的字串轉成目標型別。

    參數:
        raw: .env 取得的原始字串值。
        target_type: 目標型別（int / float / str）。
    回傳:
        轉型後的值（型別 = target_type）。
    """
    if target_type is bool:
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    if target_type is int:
        return int(float(raw))  # 容許 "15.0" 這類寫法
    if target_type is float:
        return float(raw)
    return raw  # str 直接回傳


def _validate(cfg: AppConfig) -> None:
    """對載入後的設定做基本合理性檢查，提早攔截錯誤。

    參數:
        cfg: 待驗證的 AppConfig。
    回傳:
        無；不合理時丟出 ValueError。
    """
    if cfg.coupling not in {"dc", "ac"}:
        raise ValueError(f"coupling 必須為 'dc' 或 'ac'，收到：{cfg.coupling!r}")
    if cfg.load_level not in {"medium", "high"}:
        raise ValueError(f"load_level 必須為 'medium' 或 'high'，收到：{cfg.load_level!r}")
    if cfg.charge_priority not in {"load_first", "battery_first"}:
        raise ValueError(f"charge_priority 必須為 'load_first' 或 'battery_first'，收到：{cfg.charge_priority!r}")
    if cfg.tariff_scheme not in VALID_TARIFF_SCHEMES:
        raise ValueError(
            f"tariff_scheme 必須為 {sorted(VALID_TARIFF_SCHEMES)} 之一"
            f"（一般式/二段式/三段式），收到：{cfg.tariff_scheme!r}"
        )
    for name in ("pv_system_loss", "pv_inverter_eff", "batt_inverter_eff",
                 "batt_soc_min", "batt_soc_max",
                 "batt_charge_eff", "batt_discharge_eff",
                 "offpeak_charge_target_soc", "mid_discharge_reserve_soc"):
        val = getattr(cfg, name)
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} 應介於 0~1，收到：{val}")
    if cfg.batt_soc_min >= cfg.batt_soc_max:
        raise ValueError("batt_soc_min 必須小於 batt_soc_max")
    if cfg.timestep_min <= 0:
        raise ValueError("timestep_min 必須為正整數（分鐘）")


def _read_ini(ini_path: str) -> dict[str, str]:
    """讀 config.ini，把所有區段的鍵值攤平成 {大寫鍵: 字串值}。

    參數:
        ini_path: config.ini 路徑。
    回傳:
        dict[str, str]：以大寫鍵為索引的設定值（找不到檔案回空 dict）。
    """
    import configparser

    if not os.path.exists(ini_path):
        return {}
    parser = configparser.ConfigParser()
    parser.optionxform = str  # 保留鍵名大小寫
    parser.read(ini_path, encoding="utf-8")
    flat: dict[str, str] = {}
    for section in parser.sections():
        for key, val in parser.items(section):
            flat[key.upper()] = val
    return flat


def load_config(ini_path: str | None = "config.ini", use_dotenv: bool = False) -> AppConfig:
    """讀取設定，產生唯讀的 AppConfig。

    來源優先序（後者覆寫前者）:
        1. `_DEFAULTS`（台灣住宅情境內建值）。
        2. `config.ini`（若存在；本專案主要設定來源）。
        3. 環境變數 / `.env`（若 use_dotenv，供臨時覆寫）。

    參數:
        ini_path: config.ini 路徑；None 或檔案不存在則跳過（純用預設）。
        use_dotenv: 是否再以環境變數/.env 覆寫（預設 False）。
    回傳:
        AppConfig：載入完成的唯讀設定物件。
    """
    type_map: dict[str, Any] = {f.name: f.type for f in fields(AppConfig)}
    _resolve: dict[str, type] = {"int": int, "float": float, "str": str, "bool": bool}

    def _target(name: str) -> type:
        t = type_map.get(name, "str")
        return _resolve.get(t if isinstance(t, str) else t.__name__, str)  # type: ignore[arg-type]

    ini_flat = _read_ini(ini_path) if ini_path else {}
    if use_dotenv and _HAS_DOTENV:
        load_dotenv()

    values: dict[str, Any] = {}
    for name, default in _DEFAULTS.items():
        values[name] = default
        key = _ENV_KEYS[name]
        if key in ini_flat and ini_flat[key] != "":           # 2) config.ini
            values[name] = _coerce(ini_flat[key], _target(name))
        if use_dotenv:                                          # 3) 環境變數
            raw = os.environ.get(key)
            if raw is not None and raw != "":
                values[name] = _coerce(raw, _target(name))

    cfg = AppConfig(**values)
    _validate(cfg)
    return cfg


def summary_lines(cfg: AppConfig) -> list[str]:
    """產生人類可讀的設定摘要（供 __main__ 與上層印出）。

    參數:
        cfg: 要摘要的設定。
    回傳:
        list[str]：每列一段描述。
    """
    scheme_label = TARIFF_SCHEME_LABELS.get(cfg.tariff_scheme, cfg.tariff_scheme)
    if cfg.tariff_scheme == "progressive":
        tariff_desc = f"電價        ：{scheme_label}（套利效益有限，建議用時間電價）"
    else:
        tariff_desc = (
            f"電價({scheme_label})：尖 {cfg.tariff_peak} / 半 {cfg.tariff_mid} / "
            f"離 {cfg.tariff_offpeak} 元/度, 基本 {cfg.tariff_basic_fee} 元, "
            f"售電 {cfg.tariff_sell_price} 元/度"
        )
    return [
        f"位置        ：lat={cfg.latitude}, lon={cfg.longitude}, tz={cfg.timezone}, alt={cfg.altitude_m} m",
        f"太陽能      ：{cfg.pv_capacity_kwp} kWp, 傾角 {cfg.pv_tilt_deg}°, 方位 {cfg.pv_azimuth_deg}°",
        f"耦合架構    ：{cfg.coupling.upper()}（DC=充電乘1次效率；AC=額外乘逆變器/整流）",
        f"電池        ：{cfg.batt_capacity_kwh} kWh, SOC {cfg.batt_soc_min}~{cfg.batt_soc_max}, "
        f"充/放 {cfg.batt_charge_eff}/{cfg.batt_discharge_eff}",
        f"衰減成本    ：{cfg.batt_deg_cost} 元/kWh 吞吐",
        tariff_desc,
        f"模擬        ：{cfg.start_date} 起 {cfg.sim_days} 天, 時步 {cfg.timestep_min} 分, "
        f"策略 {cfg.control_strategy}",
    ]


if __name__ == "__main__":
    # 最小示範：載入預設設定並印出摘要。
    config = load_config()
    print("=== HEMS 設定摘要（預設值，可由 .env 覆寫）===")
    for line in summary_lines(config):
        print("  " + line)
    print()
    print("⚠️ 提醒：電價費率約每年 4 月、10 月由台電調整；")
    print("   以上為參考值，正式分析請向台電官網確認當期數字。")
    print("   本結果為決策參考，非投資建議。")
