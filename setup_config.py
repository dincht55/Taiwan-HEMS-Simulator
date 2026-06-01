"""setup_config.py — 互動式設定精靈（連網）。

功能:
    1. 使用者輸入地區（城市/鄉鎮名）。
    2. 線上查詢（Open-Meteo 免費 API，免金鑰）：
       - 地理編碼 → 緯度、經度、海拔、IANA 時區。
       - 歷史氣候 → 逐月平均氣溫(°C) 與每日日照時數(小時)。
    3. 把座標/時區/海拔寫回 config.ini 的 [location]；
       把逐月氣候平年值寫到 data/climate_normals.csv（solar_pv 會自動讀取，取代內建高雄預設）。
    4. （選用）快速設定 PV 裝置容量與電費類型。

設計:
    - 只用 Python 標準庫（urllib/json/csv），不新增相依。
    - 網路失敗時保留原設定並提示；座標/氣候查詢分開，任一失敗不影響另一。
    - config.ini 以「逐行替換指定鍵」方式更新，保留註解與排版（不用 configparser 以免吃掉註解）。

資料來源:
    - 地理編碼 API：https://geocoding-api.open-meteo.com/v1/search
    - 歷史氣候 API：https://archive-api.open-meteo.com/v1/archive
    ⚠️ 氣候為歷史統計平均、非預報；正式分析建議改用官方 TMY/實測逐時輻照。

用法:
    python setup_config.py            # 互動式
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from typing import Optional

# Open-Meteo 端點（免金鑰）。
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# 氣候統計期間（近年完整年份；多年平均較穩定）。
_CLIMATE_START = "2021-01-01"
_CLIMATE_END = "2023-12-31"

_INI_PATH = "config.ini"
_CLIMATE_CSV = "data/climate_normals.csv"
_HTTP_TIMEOUT_S = 20


# =============================================================================
# 網路查詢
# =============================================================================
def _http_get_json(url: str, params: dict) -> dict:
    """對指定 URL 帶 query 參數發 GET，回傳解析後的 JSON dict。

    參數:
        url: 端點。
        params: query 參數 dict。
    回傳:
        dict：解析後的 JSON。
    例外:
        urllib.error.URLError / ValueError：連線或解析失敗時往上拋。
    """
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": "hems-simulator/1.0"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def geocode(place: str, count: int = 5) -> list[dict]:
    """以地名查詢候選地點（座標/海拔/時區）。

    參數:
        place: 地名（中英文皆可，如「高雄」「Taipei」）。
        count: 回傳候選數上限。
    回傳:
        list[dict]：每筆含 name/country/admin1/latitude/longitude/elevation/timezone。
    """
    data = _http_get_json(_GEOCODE_URL, {
        "name": place, "count": count, "language": "zh", "format": "json",
    })
    return data.get("results", []) or []


def fetch_climate_normals(lat: float, lon: float) -> dict[int, dict[str, float]]:
    """查詢該座標的逐月氣候平年值（平均氣溫與每日日照時數）。

    參數:
        lat: 緯度（度）。
        lon: 經度（度）。
    回傳:
        dict[int, dict]：{月份(1~12): {'temp_c': 月均氣溫°C, 'sun_h': 月均每日日照時數h}}。
    """
    data = _http_get_json(_ARCHIVE_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": _CLIMATE_START, "end_date": _CLIMATE_END,
        "daily": "temperature_2m_mean,sunshine_duration",
        "timezone": "auto",
    })
    return parse_climate_daily(data)


def parse_climate_daily(data: dict) -> dict[int, dict[str, float]]:
    """把 Open-Meteo archive 的「逐日」資料彙整成逐月平年值（純函式，便於離線測試）。

    參數:
        data: archive API 回傳的 JSON（含 daily.time / temperature_2m_mean / sunshine_duration）。
    回傳:
        dict[int, dict]：逐月 temp_c 與 sun_h（sunshine_duration 由秒換算成小時/日）。
    """
    daily = data.get("daily", {})
    times = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    sun_s = daily.get("sunshine_duration", [])

    acc: dict[int, dict[str, list]] = {m: {"t": [], "s": []} for m in range(1, 13)}
    for i, t in enumerate(times):
        month = int(t[5:7])  # 'YYYY-MM-DD'
        if i < len(temps) and temps[i] is not None:
            acc[month]["t"].append(float(temps[i]))
        if i < len(sun_s) and sun_s[i] is not None:
            acc[month]["s"].append(float(sun_s[i]) / 3600.0)  # 秒 → 小時/日

    normals: dict[int, dict[str, float]] = {}
    for m in range(1, 13):
        t_list, s_list = acc[m]["t"], acc[m]["s"]
        if t_list and s_list:
            normals[m] = {
                "temp_c": round(sum(t_list) / len(t_list), 1),
                "sun_h": round(sum(s_list) / len(s_list), 2),
            }
    return normals


# =============================================================================
# 寫入設定
# =============================================================================
def update_ini_value(path: str, key: str, value: str) -> bool:
    """逐行替換 config.ini 中指定鍵的值，保留註解與排版。

    參數:
        path: config.ini 路徑。
        key: 設定鍵名（大寫，如 HEMS_LATITUDE）。
        value: 新值（字串）。
    回傳:
        bool：有成功替換回 True，找不到該鍵回 False。
    """
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$")
    changed = False
    for i, line in enumerate(lines):
        if pat.match(line):
            lines[i] = pat.sub(rf"\g<1>{value}", line.rstrip("\n")) + "\n"
            changed = True
            break
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return changed


def write_climate_csv(normals: dict[int, dict[str, float]], path: str = _CLIMATE_CSV) -> None:
    """把逐月氣候平年值寫成 solar_pv 可讀的 CSV（month/temp_c/sun_h）。

    參數:
        normals: 逐月平年值 dict。
        path: 輸出 CSV 路徑。
    回傳:
        無。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["month", "temp_c", "sun_h"])
        for m in range(1, 13):
            if m in normals:
                w.writerow([m, normals[m]["temp_c"], normals[m]["sun_h"]])


# =============================================================================
# 互動流程
# =============================================================================
def _ask(prompt: str, default: str = "") -> str:
    """顯示提示並讀取一行輸入；空輸入回傳預設值。"""
    suffix = f"（預設 {default}）" if default else ""
    ans = input(f"{prompt}{suffix}：").strip()
    return ans or default


def _choose_location() -> Optional[dict]:
    """互動查詢並選擇地點。回傳選定的地理編碼結果 dict，或 None（取消/失敗）。"""
    place = _ask("請輸入地區（城市/鄉鎮，如 高雄、台北、Taichung）")
    if not place:
        print("  未輸入地區，略過位置設定。")
        return None
    try:
        results = geocode(place)
    except Exception as e:  # 網路或解析失敗
        print(f"  ⚠️ 地理編碼查詢失敗（{e}）；請確認網路連線。略過位置設定。")
        return None
    if not results:
        print(f"  查無「{place}」的座標；略過位置設定。")
        return None

    print(f"\n  找到 {len(results)} 個候選：")
    for i, r in enumerate(results, 1):
        loc = f"{r.get('name','?')}, {r.get('admin1','')} {r.get('country','')}".strip()
        print(f"   [{i}] {loc}  (lat {r['latitude']:.4f}, lon {r['longitude']:.4f}, "
              f"alt {r.get('elevation','?')} m, tz {r.get('timezone','?')})")
    sel = _ask("選擇編號", "1")
    try:
        idx = int(sel) - 1
        return results[idx] if 0 <= idx < len(results) else results[0]
    except ValueError:
        return results[0]


def main() -> None:
    """互動式設定精靈主流程。"""
    print("=== HEMS 設定精靈（連網查詢座標與氣候）===")
    if not os.path.exists(_INI_PATH):
        print(f"  找不到 {_INI_PATH}，請在專案根目錄執行。")
        sys.exit(1)

    # 1) 位置
    loc = _choose_location()
    if loc:
        ok_lat = update_ini_value(_INI_PATH, "HEMS_LATITUDE", f"{loc['latitude']:.4f}")
        update_ini_value(_INI_PATH, "HEMS_LONGITUDE", f"{loc['longitude']:.4f}")
        if loc.get("timezone"):
            update_ini_value(_INI_PATH, "HEMS_TIMEZONE", str(loc["timezone"]))
        if loc.get("elevation") is not None:
            update_ini_value(_INI_PATH, "HEMS_ALTITUDE", f"{float(loc['elevation']):.1f}")
        print(f"  ✓ 已寫入座標/時區/海拔到 {_INI_PATH}" if ok_lat else "  ⚠️ 座標寫入失敗")

        # 2) 氣候平年值
        try:
            print("  查詢逐月氣候平年值中…")
            normals = fetch_climate_normals(loc["latitude"], loc["longitude"])
            if len(normals) == 12:
                write_climate_csv(normals)
                print(f"  ✓ 已寫入 {_CLIMATE_CSV}（solar_pv 將自動採用此地區氣候）")
                print("    月  氣溫°C  日照h")
                for m in range(1, 13):
                    print(f"    {m:>2}   {normals[m]['temp_c']:>5.1f}  {normals[m]['sun_h']:>5.2f}")
            else:
                print("  ⚠️ 氣候資料不完整（未滿 12 個月），保留內建高雄預設。")
        except Exception as e:
            print(f"  ⚠️ 氣候查詢失敗（{e}）；保留內建高雄預設。")

    # 3) 選用：PV 容量與電費類型
    print("\n  （選用）快速調整常用參數，直接 Enter 可略過：")
    pv = _ask("PV 裝置容量 kWp")
    if pv:
        update_ini_value(_INI_PATH, "PV_CAPACITY_KWP", pv)
    scheme = _ask("電費類型 progressive/two_stage/three_stage")
    if scheme in {"progressive", "two_stage", "three_stage"}:
        update_ini_value(_INI_PATH, "TARIFF_SCHEME", scheme)

    print("\n  完成。可執行：python main.py --mode single")
    print("  ⚠️ 氣候為歷史統計、費率與單價會變動；結果為決策參考、非投資建議。")


if __name__ == "__main__":
    main()
