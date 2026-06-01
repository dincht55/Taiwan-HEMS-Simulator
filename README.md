# HEMS 家庭能源管理系統模擬器

模擬「太陽能板 + 電池 + 市電」在台電分時電價下的能源調度，估算節電與經濟效益。
以 pvlib 計算在地日照、規則式/MILP 策略逐時調度，並輸出自用率、自給率、年電費與回本年限。
預設地點為高雄市前鎮區，可用內建的設定精靈一鍵切換到任何地區。

> ⚠️ 台電費率、躉購費率與設備單價會逐年調整；本專案內建數值僅為**參考值**。
> 結果為**決策參考、非投資建議**，正式評估請向台電/能源署確認當期數字並做現勘。

## 功能特色
- 在地化太陽能：pvlib 晴空模型 + 氣候平年值縮放（多雲/溫度降額）；無 pvlib 時自動退回簡化模型。
- 三種控制策略：純規則式自用優先、預測式（查明日天氣配置夜充）、MILP 最佳化基準。
- 餘電去向可切換：自用限發 / 自發自用餘電躉售（賣台電）。
- 多種分析模式：單次模擬、策略比較、容量掃描、電費比較、餘電躉售對照、面積分階段設計、大型因子實驗。
- 效能：numba JIT 調度核心（全年 35040 步毫秒級）+ multiprocessing 多核掃描。
- 單一設定來源 `config.ini`；附 67 項 pytest。

## 安裝
需 Python 3.10+。
```bash
pip install -r requirements.txt
```

## 快速開始
```bash
# （選用）用設定精靈依地區自動查座標與氣候，寫回 config.ini
python setup_config.py

# 依 config.ini 執行單次全年模擬，輸出 CSV+圖、印指標
python main.py --mode single
```

## 設定精靈（連網）
`setup_config.py` 會：輸入地區 → 線上查詢座標/海拔/時區（Open-Meteo 地理編碼）與逐月氣候
（歷史平均氣溫與日照）→ 寫回 `config.ini` 的 `[location]`，並把氣候平年值存到
`data/climate_normals.csv`（`solar_pv` 會自動採用，取代內建高雄預設）。僅用標準庫、免 API 金鑰。
> 氣候為歷史統計平均（非預報）；最高精度請改用官方 TMY 或實測逐時輻照。

## 執行模式（`config.ini` 的 `[run] MODE`，或 `--mode` 覆寫）
| 模式 | 說明 |
|------|------|
| `single`    | 單次全年模擬，輸出 CSV+圖、印指標 |
| `compare`   | 同配置比較三種控制策略 |
| `sweep`     | PV×電池容量網格掃描（多核） |
| `tariff`    | 比較三種電費方案（一般/二段/三段） |
| `feedin`    | 餘電躉售對照（不加電池）：不躉售 vs 餘電躉售 vs PV+電池自用 |
| `area`      | 無電池餘電躉售系統，依可安裝面積分階段配置元件與最佳設計 |
| `factorial` | 大型因子實驗：電費×負載×PV×電池×策略 全組合（多核） |

```bash
python main.py --mode sweep
python main.py --ini my.ini --mode factorial
```

## 單一設定來源
所有可調數值集中於 **`config.ini`**，由 `config.py` 讀成唯讀的 `AppConfig`。改參數不必動程式碼。
其中售電相關兩個鍵：
- `TARIFF_SELL_PRICE`：本次模擬「實際」售電價（自用型未登記躉售＝0）。
- `TARIFF_FEED_IN_PRICE`：躉購(FiT)參考價，供 `feedin`/`area` 模式評估「若登記躉售」。

實測逐時用電可放 `data/load_profile.csv`（單欄 kW），比合成負載更準。

## 專案結構
```
config.ini            單一設定來源
config.py             讀 config.ini → AppConfig
setup_config.py       設定精靈（連網查座標/氣候）
electrical_base.py    模組1：單位換算、效率、功率平衡
solar_pv.py           模組2：pvlib 太陽能（可載入在地氣候）
battery_charge.py     模組3：SOC 動態、充放電
power_dispatch.py     模組4：計價、規則式調度、累進電價、MILP 輔助
fast_core.py          numba JIT 調度核心
hems_simulation.py    模組5：整合、策略、指標、掃描、因子實驗
main.py               進入點：讀 config.ini → 依模式執行
docs/                 設計知識檔（架構/電學/電池/電價/策略/結論）
tests/test_hems.py    pytest（模組 1–5）
data/                 climate_normals.csv（精靈產生）、load_profile.csv（實測，選用）
```

## 測試
```bash
pytest -q
```

## 授權
MIT License，詳見 [LICENSE](LICENSE)。

## 致謝與資料來源
- 太陽位置與輻照：[pvlib](https://pvlib-python.readthedocs.io/)
- 座標與歷史氣候：[Open-Meteo](https://open-meteo.com/)（免費、免金鑰）
- 電價與躉購費率：台灣電力公司、經濟部能源署（請以官網當期公告為準）
