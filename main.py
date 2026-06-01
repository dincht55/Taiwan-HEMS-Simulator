"""main.py — HEMS 模擬器統一進入點。

用法:
    python main.py                # 依 config.ini 的 [run] MODE 執行
    python main.py --mode sweep   # 命令列覆寫模式
    python main.py --ini my.ini   # 指定不同設定檔

模式（config.ini [run] MODE 或 --mode）:
    single  : 依設定跑單次全年模擬，輸出 CSV+圖，印指標摘要。
    compare : 同一配置下比較三種控制策略（rule_based / rule_based_forecast / milp_mpc）。
    sweep   : PV×電池容量網格掃描（多核並行），找回本最短配置。
    tariff  : 同一配置下比較三種電費方案（一般/二段/三段）。
    feedin  : 餘電躉售情境（不加電池）：掃描 PV，對照「不躉售(限發)」vs「餘電躉售(賣FiT)」。
    area    : 無電池餘電躉售系統，依可安裝面積分階段配置元件與最佳設計試算。

加速：規則式核心以 numba JIT 編譯（見 fast_core.py）；容量掃描以多進程吃滿多核。
⚠️ 費率與單價為參考值、會變動；結果為決策參考，非投資建議。
"""

from __future__ import annotations

import argparse
import configparser
import dataclasses
import os

import config as cfgmod
import fast_core
import hems_simulation as h


def _read_run_section(ini_path: str) -> dict:
    """讀 config.ini 的 [run] 區段（執行設定，非物理模型參數）。

    參數:
        ini_path: 設定檔路徑。
    回傳:
        dict：含 mode / n_jobs / use_jit / forecast_error_std / output_dir /
              sweep_pv / sweep_batt。
    """
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(ini_path, encoding="utf-8")
    run = {k.upper(): v for k, v in (parser.items("run") if parser.has_section("run") else [])}

    def _floats(key: str, fallback: list[float]) -> list[float]:
        raw = run.get(key)
        return [float(x) for x in raw.split(",")] if raw else fallback

    def _strs(key: str, fallback: list[str]) -> list[str]:
        raw = run.get(key)
        return [x.strip() for x in raw.split(",") if x.strip()] if raw else fallback

    n_jobs_raw = run.get("N_JOBS", "auto").strip().lower()
    n_jobs = (os.cpu_count() or 1) if n_jobs_raw in ("auto", "0", "") else int(n_jobs_raw)

    return {
        "mode": run.get("MODE", "single").strip().lower(),
        "n_jobs": n_jobs,
        "use_jit": run.get("USE_JIT", "true").strip().lower() in ("1", "true", "yes", "on"),
        "forecast_error_std": float(run.get("FORECAST_ERROR_STD", "0.0")),
        "output_dir": run.get("OUTPUT_DIR", "output").strip(),
        "sweep_pv": _floats("SWEEP_PV_KWP", [3.0, 5.0, 8.0]),
        "sweep_batt": _floats("SWEEP_BATT_KWH", [0.001, 5.0, 10.0, 15.0]),
        # 大型因子實驗
        "fac_tariff": _strs("FACTORIAL_TARIFF", ["three_stage", "two_stage", "progressive"]),
        "fac_load": _strs("FACTORIAL_LOAD", ["medium", "high"]),
        "fac_pv": _floats("FACTORIAL_PV_KWP", [0.5, 1.0, 2.0, 3.0]),
        "fac_batt": _floats("FACTORIAL_BATT_KWH", [1.0, 3.0, 5.0, 8.0]),
        "fac_strategies": _strs("FACTORIAL_STRATEGIES", ["battery_first_peak", "load_first_peak"]),
    }


def _print_metrics(m: dict) -> None:
    """印出單次模擬的指標摘要。"""
    print(f"  太陽能總發電 {m['pv_total_kwh']:,.0f} kWh | 總負載 {m['load_total_kwh']:,.0f} kWh")
    print(f"  自用率 SCR = {m['scr']*100:.1f}%  自給率 SSR = {m['ssr']*100:.1f}%")
    print(f"  年電費（PV+電池）{m['bill']:,.0f} 元")
    print(f"  全系統年省（vs 無太陽能無電池）{m['savings_total']:,.0f} 元 → 回本 ≈ {m['payback_system_years']:.1f} 年")
    print(f"  電池增益（vs 純太陽能）{m['savings_battery']:,.0f} 元；扣衰減 {m['degradation_cost']:,.0f} "
          f"後淨增益 {m['net_battery_savings']:,.0f} 元 → 電池回本 ≈ {m['payback_battery_years']:.1f} 年")


def run_single(cfg, run: dict) -> None:
    """單次全年模擬 → 輸出 CSV+圖 → 印摘要。"""
    print(f"=== 單次模擬（策略={cfg.control_strategy}）===")
    df = h.run_simulation(cfg, forecast_error_std=run["forecast_error_std"])
    m = h.compute_metrics(df, cfg)
    csv_path, png_path = h._save_outputs(df, m, cfg, run["output_dir"])
    _print_metrics(m)
    print(f"  已輸出：{csv_path}、{png_path}")


def run_compare(cfg, run: dict) -> None:
    """同配置比較三種控制策略。"""
    print("=== 控制策略比較（同一配置）===")
    print(f"{'策略':22}{'SCR':>7}{'SSR':>7}{'年電費':>10}{'全系統回本':>11}{'電池淨增益':>11}{'電池回本':>10}")
    for s in ("rule_based", "rule_based_forecast", "milp_mpc"):
        c = dataclasses.replace(cfg, control_strategy=s)
        m = h.compute_metrics(h.run_simulation(c, forecast_error_std=run["forecast_error_std"]), c)
        print(f"{s:22}{m['scr']*100:6.1f}%{m['ssr']*100:6.1f}%{m['bill']:10,.0f}"
              f"{m['payback_system_years']:9.1f}年{m['net_battery_savings']:11,.0f}"
              f"{m['payback_battery_years']:8.1f}年")


def run_sweep(cfg, run: dict) -> None:
    """PV×電池容量網格掃描（多核並行）。"""
    print(f"=== 容量掃描（{run['n_jobs']} 並行）PV={run['sweep_pv']} × 電池={run['sweep_batt']} ===")
    sweep = h.sweep_configurations(
        cfg, pv_list=run["sweep_pv"], batt_list=run["sweep_batt"],
        strategy=cfg.control_strategy, processes=run["n_jobs"],
    ).sort_values("payback_system_years").reset_index(drop=True)
    os.makedirs(run["output_dir"], exist_ok=True)
    out = os.path.join(run["output_dir"], "sweep_result.csv")
    sweep.to_csv(out, index=False)
    for _, r in sweep.iterrows():
        b = 0.0 if r["batt_capacity_kwh"] < 0.01 else r["batt_capacity_kwh"]
        print(f"  PV {r['pv_capacity_kwp']:>4.1f}kWp 電池 {b:>5.1f}kWh | SSR {r['ssr']*100:4.1f}% | "
              f"全系統年省 {r['savings_total']:7,.0f} | 全系統回本 {r['payback_system_years']:5.1f}年")
    best = sweep.iloc[0]
    print(f"\n  → 回本最短：PV {best['pv_capacity_kwp']}kWp + 電池 {best['batt_capacity_kwh']}kWh，"
          f"{best['payback_system_years']:.1f} 年；明細已存 {out}")


def run_tariff(cfg, run: dict) -> None:
    """同配置比較三種電費方案。"""
    print("=== 電費方案比較（同一配置，年電費越低越好）===")
    cmp = h.compare_tariffs(cfg, strategy=cfg.control_strategy)
    labels = {"progressive": "一般式", "two_stage": "二段式", "three_stage": "三段式"}
    for _, r in cmp.iterrows():
        print(f"  {labels.get(r['tariff_scheme'], r['tariff_scheme'])} | 年電費 {r['bill']:8,.0f} | "
              f"全系統年省 {r['savings_total']:8,.0f} | 回本 {r['payback_system_years']:5.1f}年")


def run_feedin(cfg, run: dict) -> None:
    """餘電去向三方對照：同一 PV 容量下比較三種做法，回答「不裝電池餘電躉售 vs 裝電池自用」。

    三個世界（其餘條件相同，皆用 cfg 的電費/負載/策略/單價）:
        A 不躉售無電池：售電價=0、電池≈0。餘電限發丟掉（自用型現況）。
        B 餘電躉售無電池：售電價=FiT、電池≈0。餘電以 FiT 賣台電（本次主題）。
        C PV+電池自用：售電價=0、電池=cfg.batt_capacity_kwh。餘電先存電池供晚間，
                       存不下才限發；用來檢驗「花錢買電池自用」是否勝過「直接賣餘電」。

    參數:
        cfg: 基底設定（FiT 取 cfg.tariff_feed_in_price；電池容量取 cfg.batt_capacity_kwh）。
        run: 執行設定（sweep_pv 當 PV 掃描清單、output_dir）。
    回傳:
        無；印出對照表並存 feedin_result.csv。
    """
    import math

    import pandas as pd

    fit_price = cfg.tariff_feed_in_price          # 躉購(FiT)參考價，元/度（單一來源 config.ini）
    batt_kwh = cfg.batt_capacity_kwh              # C 世界的電池容量（單一來源 config.ini）
    pv_list = run["sweep_pv"]
    no_batt_kwh = 0.001                           # 代表「不裝電池」（避免容量為 0 除以零；與 sweep 慣例一致）

    label = {"progressive": "一般式", "two_stage": "二段式", "three_stage": "三段式"}.get(
        cfg.tariff_scheme, cfg.tariff_scheme)
    print(f"=== 餘電去向三方對照（電費={label}，負載={cfg.load_level}）===")
    if fit_price <= 0:
        print("  ⚠️ config.ini 的 TARIFF_FEED_IN_PRICE=0，等於不躉售；請設為躉購費率（如 5.6279）再跑此模式。")
    print(f"  躉購(FiT)價 = {fit_price:.4f} 元/度；C 世界電池容量 = {batt_kwh:.1f} kWh（皆來自 config.ini）")

    rows = []
    base_no_system = None
    for p in pv_list:
        # A 不躉售無電池；B 餘電躉售無電池；C PV+電池自用。
        cfg_a = dataclasses.replace(cfg, pv_capacity_kwp=p, batt_capacity_kwh=no_batt_kwh, tariff_sell_price=0.0)
        cfg_b = dataclasses.replace(cfg, pv_capacity_kwp=p, batt_capacity_kwh=no_batt_kwh, tariff_sell_price=fit_price)
        cfg_c = dataclasses.replace(cfg, pv_capacity_kwp=p, batt_capacity_kwh=batt_kwh, tariff_sell_price=0.0)
        m_a = h.compute_metrics(h.run_simulation(cfg_a, forecast_error_std=run["forecast_error_std"]), cfg_a)
        m_b = h.compute_metrics(h.run_simulation(cfg_b, forecast_error_std=run["forecast_error_std"]), cfg_b)
        m_c = h.compute_metrics(h.run_simulation(cfg_c, forecast_error_std=run["forecast_error_std"]), cfg_c)
        base_no_system = m_a["no_system_bill"]    # 無系統基準（三世界相同）

        surplus_kwh = m_a["curtail_kwh"]          # 無電池時的餘電（=B 世界的饋網量）
        rows.append({
            "pv_kwp": p,
            "surplus_kwh": surplus_kwh,
            "scr_a": m_a["scr"], "scr_c": m_c["scr"],
            "bill_a": m_a["bill"], "bill_b": m_b["bill"], "bill_c": m_c["bill"],
            "sell_revenue_b": surplus_kwh * fit_price,
            "payback_a": m_a["payback_system_years"],
            "payback_b": m_b["payback_system_years"],
            "payback_c": m_c["payback_system_years"],
            "batt_net_payback_c": m_c["payback_battery_years"],  # 電池本身(相對純PV自用)淨回本
        })

    df = pd.DataFrame(rows)
    os.makedirs(run["output_dir"], exist_ok=True)
    out = os.path.join(run["output_dir"], "feedin_result.csv")
    df.to_csv(out, index=False)

    def yr(x: float) -> str:
        """回本年限格式化：無限大（不回本）以 ' —' 表示。"""
        return "   —" if math.isinf(x) else f"{x:>4.1f}年"

    print(f"  無系統(不裝)年電費基準 = {base_no_system:,.0f} 元（回本＝建置淨成本 ÷ 對基準的年省）")
    print(f"  {'PV':>4} {'餘電':>6} ┃ {'A不躉售無電池':^16} ┃ {'B餘電躉售無電池':^16} ┃ {'C PV+電池自用':^20}")
    print(f"  {'kWp':>4} {'kWh':>6} ┃ {'年電費':>8}{'回本':>7} ┃ {'年電費':>8}{'回本':>7} ┃ {'年電費':>8}{'回本':>7}{'電池淨回本':>9}")
    for _, r in df.iterrows():
        print(f"  {r['pv_kwp']:>3.1f}k {r['surplus_kwh']:>6,.0f} ┃ "
              f"{r['bill_a']:>8,.0f} {yr(r['payback_a'])} ┃ "
              f"{r['bill_b']:>8,.0f} {yr(r['payback_b'])} ┃ "
              f"{r['bill_c']:>8,.0f} {yr(r['payback_c'])} {yr(r['batt_net_payback_c'])}")
    print(f"\n  讀法：A 是現況(餘電丟掉)；B 只是把餘電拿去賣(無電池成本)；C 花錢買 {batt_kwh:.0f}kWh 電池自用。")
    print(f"  「電池淨回本」= 電池成本 ÷ (電池相對純PV自用多省的錢 − 衰減)；'—' 代表不回本。")
    print(f"  明細已存：{out}")


_MODES = {"single": run_single, "compare": run_compare, "sweep": run_sweep,
          "tariff": run_tariff, "feedin": run_feedin}

# 具名策略：對應 charge_priority / discharge_only_peak 等設定覆寫（皆用預測式 rule_based_forecast）。
_STRATEGIES: dict[str, dict] = {
    # A：日照時 PV 先充滿再供電，只在最貴時段放電（蓄滿＋尖峰套利）
    "battery_first_peak": {"control_strategy": "rule_based_forecast",
                           "charge_priority": "battery_first", "discharge_only_peak": True},
    # B：PV 先供電、餘電才充電，只在最貴時段放電（自用優先＋預測）
    "load_first_peak": {"control_strategy": "rule_based_forecast",
                        "charge_priority": "load_first", "discharge_only_peak": True},
    # 參考：自用優先＋預測，半尖峰也可放電（不限最貴時段）
    "load_first_forecast": {"control_strategy": "rule_based_forecast",
                            "charge_priority": "load_first", "discharge_only_peak": False},
}


def run_factorial(cfg, run: dict) -> None:
    """大型因子實驗：電費 × 負載 × PV × 電池 × 策略，全組合多核並行。"""
    tariffs = run["fac_tariff"]
    loads = run["fac_load"]
    pv_list = run["fac_pv"]
    batt_list = run["fac_batt"]
    strat_names = run["fac_strategies"]
    strategies = [(s, _STRATEGIES[s]) for s in strat_names if s in _STRATEGIES]
    total = len(tariffs) * len(loads) * len(pv_list) * len(batt_list) * len(strategies)
    print(f"=== 大型因子實驗（{run['n_jobs']} 並行，共 {total} 組）===")
    print(f"  電費 {tariffs} × 負載 {loads} × PV {pv_list} × 電池 {batt_list} × 策略 {strat_names}")
    df = h.run_factorial(cfg, tariffs, loads, pv_list, batt_list, strategies,
                         processes=run["n_jobs"])
    os.makedirs(run["output_dir"], exist_ok=True)
    out = os.path.join(run["output_dir"], "factorial_result.csv")
    df.to_csv(out, index=False)
    print(f"  完整對照表已存：{out}（{len(df)} 列）")
    print("\n  各情境最省配置（年電費最低）：")
    for (t, lv), g in df.groupby(["tariff", "load"]):
        best = g.loc[g["bill"].idxmin()]
        print(f"   {t:12} {lv:6} → PV {best['pv_kwp']}kWp 電池 {best['batt_kwh']}kWh "
              f"[{best['strategy']}] 年電費 {best['bill']:,.0f} 元、回本 {best['payback_system_years']:.1f} 年")


_MODES["factorial"] = run_factorial


# =============================================================================
# area 模式：無電池「餘電躉售」系統，依可安裝面積分階段最佳設計
# 以下為「分析端假設常數」（非物理模型參數），集中於此並標明單位與出處，便於修改。
# 物理/經濟模型值（FiT、PV 單價）仍來自 config.ini（單一來源）。
# =============================================================================
_AREA_PER_KWP_M2 = 6.5          # 實務可裝密度（m²/kWp，含走道/退縮/間距；物理極限約 5；業界常用 ≈2坪/kWp）
_PANEL_WATT_W = 450.0           # 單片模組功率（W）；常見 430~460W
_PYEONG_M2 = 3.3058             # 1 坪 = 3.3058 m²
_INVERTER_DCAC_RATIO = 1.0      # 逆變器/PV 容量比（餘電躉售取 1.0，盡量不削峰）
# 每 kWp 建置淨成本（元/kWp），(容量上限kWp, 單價)；規模越小固定成本攤不掉越貴（2025–26 行情 5~8 萬）。
_PV_COST_TIERS = [(2.0, 70000.0), (4.0, 62000.0), (6.0, 56000.0), (10.0, 50000.0)]
_AREA_STAGES_M2 = [10.0, 20.0, 30.0, 40.0, 50.0, 65.0]  # 分階段可安裝面積（m²），可自行調整


def _cost_per_kwp(pv_kwp: float) -> float:
    """依容量規模回傳每 kWp 建置淨成本（元/kWp）；超過最大級距沿用最後一檔。"""
    for cap, price in _PV_COST_TIERS:
        if pv_kwp <= cap:
            return price
    return _PV_COST_TIERS[-1][1]


def _size_components(area_m2: float) -> dict:
    """由可安裝面積推算合理元件規格（無電池、餘電躉售）。

    參數:
        area_m2: 可安裝面積（m²）。
    回傳:
        dict：area_m2/pyeong/pv_kwp/n_panels/inverter_kw/unit_cost_per_kwp/system_cost。
    """
    pv_kwp = area_m2 / _AREA_PER_KWP_M2
    n_panels = max(1, round(pv_kwp * 1000.0 / _PANEL_WATT_W))
    pv_kwp = n_panels * _PANEL_WATT_W / 1000.0                # 以整數片數回推實際容量
    unit_cost = _cost_per_kwp(pv_kwp)
    return {
        "area_m2": area_m2, "pyeong": area_m2 / _PYEONG_M2,
        "pv_kwp": round(pv_kwp, 2), "n_panels": n_panels,
        "inverter_kw": round(pv_kwp * _INVERTER_DCAC_RATIO, 1),
        "unit_cost_per_kwp": unit_cost, "system_cost": pv_kwp * unit_cost,
    }


def run_area(cfg, run: dict) -> None:
    """area 模式：無電池餘電躉售系統，依可安裝面積分階段配置元件並試算。

    參數:
        cfg: 基底設定（FiT 取 tariff_feed_in_price；電費/負載/位置沿用）。
        run: 執行設定（output_dir）。
    回傳:
        無；印出分階段最佳設計表並存 area_stage_result.csv。
    """
    import math

    import pandas as pd

    fit_price = cfg.tariff_feed_in_price
    label = {"progressive": "一般式", "two_stage": "二段式", "three_stage": "三段式"}.get(
        cfg.tariff_scheme, cfg.tariff_scheme)
    print("=== 無電池『餘電躉售』系統：依可安裝面積分階段最佳設計 ===")
    if fit_price <= 0:
        print("  ⚠️ TARIFF_FEED_IN_PRICE=0，等於不躉售；請設為躉購費率（如 5.6279）。")
    print(f"  電費 {label}｜負載 {cfg.load_level}｜FiT {fit_price:.4f} 元/度｜"
          f"密度 {_AREA_PER_KWP_M2} m²/kWp｜模組 {_PANEL_WATT_W:.0f}W")

    rows = []
    for area in _AREA_STAGES_M2:
        spec = _size_components(area)
        c = dataclasses.replace(cfg, pv_capacity_kwp=spec["pv_kwp"], batt_capacity_kwh=0.001,
                                tariff_sell_price=fit_price,
                                pv_unit_cost_per_kwp=spec["unit_cost_per_kwp"])
        m = h.compute_metrics(h.run_simulation(c, forecast_error_std=run["forecast_error_std"]), c)
        gen = m["pv_total_kwh"]
        self_use = m["scr"] * gen
        surplus = max(0.0, gen - self_use - m["curtail_kwh"])  # 售電時餘電走饋網，由自用率回推
        spec.update({
            "scr": m["scr"], "gen_kwh": gen, "surplus_kwh": surplus,
            "sell_revenue": surplus * fit_price, "annual_benefit": m["savings_total"],
            "payback_years": m["payback_system_years"],
            "cum_20yr": m["savings_total"] * 20.0 - spec["system_cost"],
        })
        rows.append(spec)

    df = pd.DataFrame(rows)
    os.makedirs(run["output_dir"], exist_ok=True)
    out = os.path.join(run["output_dir"], "area_stage_result.csv")
    df.to_csv(out, index=False)

    def yr(x: float) -> str:
        return "  —" if math.isinf(x) else f"{x:>4.1f}年"

    print(f"\n  {'面積':>5}{'坪':>4}{'PV':>7}{'片':>4}{'逆變器':>7}{'系統成本':>9}"
          f"{'年發電':>7}{'自用率':>6}{'餘電':>6}{'躉售收入':>8}{'年淨效益':>8}{'回本':>6}{'20年淨現金':>10}")
    for r in rows:
        print(f"  {r['area_m2']:>4.0f}㎡{r['pyeong']:>4.0f}{r['pv_kwp']:>6.2f}k{r['n_panels']:>4d}"
              f"{r['inverter_kw']:>5.1f}kW{r['system_cost']:>9,.0f}{r['gen_kwh']:>7,.0f}"
              f"{r['scr']*100:>5.1f}%{r['surplus_kwh']:>6,.0f}{r['sell_revenue']:>8,.0f}"
              f"{r['annual_benefit']:>8,.0f}{yr(r['payback_years']):>7}{r['cum_20yr']:>10,.0f}")
    print(f"\n  ⚠️ ≥10kWp 將跌出最高 FiT 級距（10–20 瓩費率較低）；單表為 ≤10kWp 同一 FiT。")
    print(f"  20年淨現金為未折現簡化值。明細已存：{out}")


_MODES["area"] = run_area


def main() -> None:
    """解析參數、載入設定、依模式執行。"""
    ap = argparse.ArgumentParser(description="HEMS 家庭能源管理系統模擬器")
    ap.add_argument("--ini", default="config.ini", help="設定檔路徑（預設 config.ini）")
    ap.add_argument("--mode", default=None, choices=list(_MODES),
                    help="執行模式（覆寫 config.ini）")
    args = ap.parse_args()

    cfg = cfgmod.load_config(ini_path=args.ini)
    run = _read_run_section(args.ini)
    mode = args.mode or run["mode"]

    print("=== HEMS 模擬器 ===")
    for line in cfgmod.summary_lines(cfg):
        print("  " + line)
    print(f"  加速：numba JIT {'啟用' if (run['use_jit'] and fast_core.HAS_NUMBA) else '未啟用（純 Python）'}"
          f" | 多核並行數 {run['n_jobs']}")
    print()

    if mode not in _MODES:
        raise SystemExit(f"未知模式：{mode!r}（可用：{list(_MODES)}）")
    _MODES[mode](cfg, run)

    print("\n⚠️ 費率與單價為參考值、會變動；結果為決策參考，非投資建議。")


if __name__ == "__main__":
    main()
