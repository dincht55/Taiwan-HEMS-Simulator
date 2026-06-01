"""test_hems.py — HEMS 模擬器整合測試（合併模組 1–5 的 pytest）。

由原 test_electrical_base / test_solar_pv / test_battery_charge /
test_power_dispatch / test_hems_simulation 合併而成，涵蓋 `00_協作流程` §4 各模組驗收：
- 模組 1：config 載入/覆寫/驗證、kWh 換算、效率、耦合、功率平衡殘差。
- 模組 2：太陽能輸出形狀/非負/夜間零/年單位發電合理/簡化模型退回。
- 模組 3：SOC 動態、夾限、充電來源優先序、放電效率、衰減成本、AC/DC 耦合。
- 模組 4：時段判定/計價、放電夾限、自用優先決策階梯、功率平衡不變式。
- 模組 5：策略切換、逐時功率平衡、指標、預測式夜充規劃。

執行（於專案根目錄）：`pytest -q`
共用 `cfg` fixture 以 sim_days=2 加速模組 5 模擬；模組 2 年發電測試另建 365 天索引，不受影響。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

import battery_charge as bc
import electrical_base as eb
import hems_simulation as h
import power_dispatch as pd_mod
import solar_pv as sp
from battery_charge import BatteryState
from config import AppConfig, load_config


# =============================================================================
# 共用 fixtures 與 helpers
# =============================================================================
@pytest.fixture
def cfg() -> AppConfig:
    """共用設定（sim_days=2 以加速逐時模擬測試）。

    注意：模組 1 的 config 測試直接呼叫 load_config（不用此 fixture），以驗證真實預設值。
    """
    return dataclasses.replace(load_config(use_dotenv=False), sim_days=2)


@pytest.fixture
def one_year_index(cfg: AppConfig) -> pd.DatetimeIndex:
    """整年 15 分鐘索引（與 sim_days 無關，供太陽能年發電測試）。"""
    periods = int(365 * 24 * 60 / cfg.timestep_min)
    return pd.date_range(cfg.start_date, periods=periods,
                         freq=f"{cfg.timestep_min}min", tz=cfg.timezone)


def _with(cfg: AppConfig, **changes) -> AppConfig:
    """產生改了某些欄位的設定副本（frozen dataclass 用 replace）。"""
    return dataclasses.replace(cfg, **changes)


def _residual(pv, load, r) -> float:
    """用模組 1 的功率平衡殘差檢查單時步 dispatch 結果。"""
    return eb.power_balance_residual(
        pv, r["grid_in_kw"], r["batt_dis_kw"], load,
        r["grid_out_kw"], r["batt_ch_kw"], r["pv_curtail_kw"],
    )


def _balance_ok(df: pd.DataFrame) -> bool:
    """整段 DataFrame 每時步功率平衡殘差皆 ≈0。"""
    resid = eb.power_balance_residual(
        df["pv_kw"], df["grid_in_kw"], df["batt_dis_kw"], df["load_kw"],
        df["grid_out_kw"], df["batt_ch_kw"], df["pv_curtail_kw"],
    )
    return bool(np.abs(resid).max() < 1e-6)


# =============================================================================
# 模組 1：config / electrical_base
# =============================================================================
def test_load_config_returns_appconfig() -> None:
    """預設值應能產生合法 AppConfig。"""
    c = load_config(use_dotenv=False)
    assert isinstance(c, AppConfig)
    assert c.coupling in {"dc", "ac"}
    assert c.tariff_sell_price == 0.0  # 預設自用型（未登記躉售）不賣電


def test_load_config_is_frozen() -> None:
    """AppConfig 應為唯讀（frozen）。"""
    c = load_config(use_dotenv=False)
    with pytest.raises(Exception):
        c.batt_capacity_kwh = 999.0  # type: ignore[misc]


def test_env_override(monkeypatch: "pytest.MonkeyPatch") -> None:
    """.env/環境變數應能覆寫預設值，且正確轉型。"""
    monkeypatch.setenv("BATT_CAPACITY_KWH", "20")
    monkeypatch.setenv("PV_COUPLING", "ac")
    monkeypatch.setenv("SIM_TIMESTEP_MIN", "30")
    c = load_config(use_dotenv=True)
    assert c.batt_capacity_kwh == 20.0
    assert isinstance(c.batt_capacity_kwh, float)
    assert c.coupling == "ac"
    assert c.timestep_min == 30 and isinstance(c.timestep_min, int)


def test_validate_rejects_bad_soc(monkeypatch: "pytest.MonkeyPatch") -> None:
    """SOC 下限 ≥ 上限應被擋下。"""
    monkeypatch.setenv("BATT_SOC_MIN", "0.9")
    monkeypatch.setenv("BATT_SOC_MAX", "0.5")
    with pytest.raises(ValueError):
        load_config(use_dotenv=True)


@pytest.mark.parametrize("scheme", ["progressive", "two_stage", "three_stage"])
def test_tariff_scheme_valid(monkeypatch: "pytest.MonkeyPatch", scheme: str) -> None:
    """三種合法電費類型皆可載入。"""
    monkeypatch.setenv("TARIFF_SCHEME", scheme)
    c = load_config(use_dotenv=True)
    assert c.tariff_scheme == scheme


def test_tariff_scheme_invalid(monkeypatch: "pytest.MonkeyPatch") -> None:
    """未知電費類型應被擋下。"""
    monkeypatch.setenv("TARIFF_SCHEME", "flat_rate")
    with pytest.raises(ValueError):
        load_config(use_dotenv=True)


@pytest.mark.parametrize(
    "power_kw, minutes, expected_kwh",
    [(1.0, 60.0, 1.0), (1.0, 15.0, 0.25), (4.0, 30.0, 2.0), (0.0, 15.0, 0.0)],
)
def test_kwh_scalar(power_kw: float, minutes: float, expected_kwh: float) -> None:
    assert eb.kwh(power_kw, minutes) == pytest.approx(expected_kwh)


def test_kwh_vectorized() -> None:
    out = eb.kwh(np.array([1.0, 2.0, 4.0]), 15.0)
    assert np.allclose(out, np.array([0.25, 0.5, 1.0]))


def test_apply_efficiency() -> None:
    assert eb.apply_efficiency(5.0, 0.95) == pytest.approx(4.75)


def test_apply_efficiency_rejects_out_of_range() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            eb.apply_efficiency(5.0, bad)


def test_apply_efficiency_chain() -> None:
    assert eb.apply_efficiency_chain(10.0, 0.96, 0.95) == pytest.approx(10.0 * 0.96 * 0.95)
    assert eb.apply_efficiency_chain(10.0) == pytest.approx(10.0)


def test_round_trip_efficiency() -> None:
    assert eb.round_trip_efficiency(0.95, 0.95) == pytest.approx(0.9025)


def test_coupling_dc(monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("PV_COUPLING", "dc")
    c = load_config(use_dotenv=True)
    assert eb.pv_to_battery_extra_eff(c) == 1.0


def test_coupling_ac(monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("PV_COUPLING", "ac")
    monkeypatch.setenv("BATT_INVERTER_EFF", "0.9")
    c = load_config(use_dotenv=True)
    assert eb.pv_to_battery_extra_eff(c) == pytest.approx(0.9)


def test_power_balance_zero_scalar() -> None:
    res = eb.power_balance_residual(
        pv_kw=4.0, grid_in_kw=1.0, batt_dis_kw=0.0,
        load_kw=3.0, grid_out_kw=0.0, batt_ch_kw=2.0, curtail_kw=0.0,
    )
    assert res == pytest.approx(0.0)


def test_power_balance_nonzero_detects_error() -> None:
    res = eb.power_balance_residual(
        pv_kw=5.0, grid_in_kw=0.0, batt_dis_kw=0.0,
        load_kw=3.0, grid_out_kw=0.0, batt_ch_kw=0.0, curtail_kw=0.0,
    )
    assert res == pytest.approx(2.0)


def test_power_balance_vectorized() -> None:
    pv = np.array([4.0, 0.0, 6.0]); gin = np.array([0.0, 3.0, 0.0])
    bdis = np.array([0.0, 0.0, 0.0]); ld = np.array([3.0, 3.0, 3.0])
    gout = np.array([0.0, 0.0, 0.0]); bch = np.array([1.0, 0.0, 3.0])
    cur = np.array([0.0, 0.0, 0.0])
    res = eb.power_balance_residual(pv, gin, bdis, ld, gout, bch, cur)
    assert np.allclose(res, 0.0)


# =============================================================================
# 模組 2：solar_pv
# =============================================================================
def test_output_shape_and_index(cfg: AppConfig, one_year_index: pd.DatetimeIndex) -> None:
    pv = sp.simulate_pv(cfg, one_year_index)
    assert isinstance(pv, pd.Series)
    assert len(pv) == len(one_year_index)
    assert (pv.index == one_year_index).all()


def test_non_negative(cfg: AppConfig, one_year_index: pd.DatetimeIndex) -> None:
    assert (sp.simulate_pv(cfg, one_year_index) >= 0).all()


def test_night_is_zero(cfg: AppConfig) -> None:
    idx = pd.date_range("2025-06-21 02:00", periods=4, freq="15min", tz=cfg.timezone)
    assert np.allclose(sp.simulate_pv(cfg, idx).to_numpy(), 0.0)


def test_noon_positive(cfg: AppConfig) -> None:
    idx = pd.date_range("2025-06-21 12:00", periods=1, freq="15min", tz=cfg.timezone)
    pv = sp.simulate_pv(cfg, idx)
    assert pv.iloc[0] > 0.0
    assert pv.iloc[0] <= cfg.pv_capacity_kwp * 1.05


def test_annual_specific_yield_reasonable(cfg: AppConfig, one_year_index: pd.DatetimeIndex) -> None:
    pv = sp.simulate_pv(cfg, one_year_index)
    summary = sp.monthly_energy_summary(pv, cfg)
    specific = summary["energy_kwh"].sum() / cfg.pv_capacity_kwp
    assert 1000.0 < specific < 1600.0, f"單位發電 {specific:.0f} 超出合理範圍"


def test_monthly_summary_has_12_months(cfg: AppConfig, one_year_index: pd.DatetimeIndex) -> None:
    summary = sp.monthly_energy_summary(sp.simulate_pv(cfg, one_year_index), cfg)
    assert list(summary.index) == list(range(1, 13))


def test_simple_model_fallback(cfg: AppConfig) -> None:
    idx = pd.date_range("2025-06-21 00:00", periods=96, freq="15min", tz=cfg.timezone)
    pv = sp._simulate_pv_simple(cfg, idx)
    assert (pv >= 0).all()
    assert pv.max() > 0.0
    assert pv.between_time("11:00", "13:00").mean() > pv.between_time("00:00", "04:00").mean()


def test_climate_normals_fallback_to_kaohsiung() -> None:
    """無在地氣候檔時，應退回內建高雄平年值（12 個月齊全）。"""
    normals = sp.load_monthly_normals(path="__nonexistent_climate__.csv")
    assert normals is sp.KAOHSIUNG_MONTHLY_NORMALS or set(normals) == set(range(1, 13))


# =============================================================================
# 模組 3：battery_charge
# =============================================================================
def test_charge_increases_soc_with_efficiency(cfg: AppConfig) -> None:
    c = _with(cfg, coupling="dc", batt_capacity_kwh=10.0, batt_charge_eff=0.95,
              batt_max_charge_kw=5.0, batt_soc_min=0.1, batt_soc_max=0.95)
    actual, new = bc.charge_step(bc.BatteryState(soc=0.5), 4.0, "pv", c, 60.0)
    assert actual == pytest.approx(4.0)
    assert new.soc == pytest.approx(0.5 + 0.95 * 4.0 * 1.0 / 10.0)


def test_charge_clamped_by_max_power(cfg: AppConfig) -> None:
    c = _with(cfg, batt_max_charge_kw=5.0, batt_soc_min=0.1, batt_soc_max=0.95)
    actual, _ = bc.charge_step(bc.BatteryState(soc=0.3), 99.0, "grid", c, 60.0)
    assert actual == pytest.approx(5.0)


def test_charge_clamped_by_headroom(cfg: AppConfig) -> None:
    c = _with(cfg, batt_capacity_kwh=10.0, batt_charge_eff=1.0, coupling="dc",
              batt_max_charge_kw=10.0, batt_soc_max=0.95)
    actual, new = bc.charge_step(bc.BatteryState(soc=0.90), 10.0, "pv", c, 60.0)
    assert actual == pytest.approx(0.5)
    assert new.soc == pytest.approx(0.95)


def test_charge_full_returns_zero(cfg: AppConfig) -> None:
    c = _with(cfg, batt_soc_max=0.95)
    actual, new = bc.charge_step(bc.BatteryState(soc=0.95), 5.0, "pv", c, 60.0)
    assert actual == pytest.approx(0.0)
    assert new.soc == pytest.approx(0.95)


def test_charge_rejects_bad_source(cfg: AppConfig) -> None:
    with pytest.raises(ValueError):
        bc.charge_step(bc.BatteryState(0.5), 1.0, "wind", cfg, 60.0)


def test_negative_available_treated_as_zero(cfg: AppConfig) -> None:
    actual, new = bc.charge_step(bc.BatteryState(0.5), -3.0, "pv", cfg, 60.0)
    assert actual == pytest.approx(0.0)
    assert new.soc == pytest.approx(0.5)


def test_priority_pv_before_grid(cfg: AppConfig) -> None:
    c = _with(cfg, batt_max_charge_kw=5.0, batt_capacity_kwh=10.0, batt_soc_min=0.1,
              batt_soc_max=0.95, coupling="dc", batt_charge_eff=0.95)
    res = bc.charge_prioritized(bc.BatteryState(0.5), 3.0, 4.0, c, 15.0)
    assert res["pv_to_batt_kw"] == pytest.approx(3.0)
    assert res["grid_to_batt_kw"] == pytest.approx(2.0)
    assert res["batt_ch_kw"] == pytest.approx(5.0)


def test_priority_total_within_max_charge_power(cfg: AppConfig) -> None:
    c = _with(cfg, batt_max_charge_kw=5.0, batt_capacity_kwh=20.0, batt_soc_min=0.1,
              batt_soc_max=0.95, coupling="dc", batt_charge_eff=0.95)
    res = bc.charge_prioritized(bc.BatteryState(0.5), 10.0, 10.0, c, 15.0)
    assert res["batt_ch_kw"] <= c.batt_max_charge_kw + 1e-9
    assert res["pv_to_batt_kw"] == pytest.approx(5.0)
    assert res["grid_to_batt_kw"] == pytest.approx(0.0)


def test_priority_respects_headroom(cfg: AppConfig) -> None:
    c = _with(cfg, batt_max_charge_kw=10.0, batt_capacity_kwh=10.0, batt_charge_eff=1.0,
              coupling="dc", batt_soc_max=0.95)
    res = bc.charge_prioritized(bc.BatteryState(0.90), 10.0, 10.0, c, 60.0)
    assert res["batt_ch_kw"] == pytest.approx(0.5)
    assert res["state"].soc == pytest.approx(0.95)  # type: ignore[union-attr]


def test_discharge_decreases_soc_with_efficiency(cfg: AppConfig) -> None:
    c = _with(cfg, batt_capacity_kwh=10.0, batt_discharge_eff=0.95,
              batt_max_discharge_kw=5.0, batt_soc_min=0.1)
    new = bc.apply_discharge(bc.BatteryState(soc=0.6), 4.0, c, 60.0)
    assert new.soc == pytest.approx(0.6 - (4.0 / 0.95) * 1.0 / 10.0)


def test_max_dischargeable_clamped_by_soc(cfg: AppConfig) -> None:
    c = _with(cfg, batt_capacity_kwh=10.0, batt_discharge_eff=1.0,
              batt_max_discharge_kw=10.0, batt_soc_min=0.1)
    assert bc.max_dischargeable_kw(bc.BatteryState(soc=0.15), c, 60.0) == pytest.approx(0.5)


def test_discharge_not_below_min(cfg: AppConfig) -> None:
    c = _with(cfg, batt_soc_min=0.1, batt_capacity_kwh=10.0, batt_discharge_eff=1.0)
    new = bc.apply_discharge(bc.BatteryState(soc=0.12), 99.0, c, 60.0)
    assert new.soc >= c.batt_soc_min - 1e-9


def test_degradation_cost(cfg: AppConfig) -> None:
    c = _with(cfg, batt_deg_cost=1.5)
    assert bc.degradation_cost(4.0, c, 15.0) == pytest.approx(1.5 * 4.0 * 0.25)


def test_ac_coupling_lower_charge_efficiency(cfg: AppConfig) -> None:
    base = dict(batt_capacity_kwh=10.0, batt_charge_eff=0.95, batt_inverter_eff=0.9,
                batt_max_charge_kw=10.0, batt_soc_min=0.1, batt_soc_max=0.99)
    _, dc_new = bc.charge_step(bc.BatteryState(0.5), 4.0, "pv", _with(cfg, coupling="dc", **base), 60.0)
    _, ac_new = bc.charge_step(bc.BatteryState(0.5), 4.0, "pv", _with(cfg, coupling="ac", **base), 60.0)
    assert ac_new.soc < dc_new.soc


# =============================================================================
# 模組 4：power_dispatch
# =============================================================================
def test_three_stage_summer_weekday(cfg: AppConfig) -> None:
    c = _with(cfg, tariff_scheme="three_stage"); tz = c.timezone
    assert pd_mod.classify_period(pd.Timestamp("2025-07-15 03:00", tz=tz), c) == "offpeak"
    assert pd_mod.classify_period(pd.Timestamp("2025-07-15 10:00", tz=tz), c) == "mid"
    assert pd_mod.classify_period(pd.Timestamp("2025-07-15 18:00", tz=tz), c) == "peak"
    assert pd_mod.classify_period(pd.Timestamp("2025-07-15 23:00", tz=tz), c) == "mid"


def test_three_stage_nonsummer_has_no_peak(cfg: AppConfig) -> None:
    c = _with(cfg, tariff_scheme="three_stage"); tz = c.timezone
    assert pd_mod.classify_period(pd.Timestamp("2025-01-15 18:00", tz=tz), c) == "mid"
    assert pd_mod.classify_period(pd.Timestamp("2025-01-15 12:00", tz=tz), c) == "offpeak"
    assert pd_mod.classify_period(pd.Timestamp("2025-01-15 03:00", tz=tz), c) == "offpeak"


def test_weekend_is_offpeak(cfg: AppConfig) -> None:
    c = _with(cfg, tariff_scheme="three_stage"); tz = c.timezone
    assert pd_mod.classify_period(pd.Timestamp("2025-07-19 18:00", tz=tz), c) == "offpeak"
    assert pd_mod.classify_period(pd.Timestamp("2025-07-20 18:00", tz=tz), c) == "offpeak"


def test_progressive_is_flat(cfg: AppConfig) -> None:
    c = _with(cfg, tariff_scheme="progressive"); tz = c.timezone
    assert pd_mod.classify_period(pd.Timestamp("2025-07-15 18:00", tz=tz), c) == "flat"


def test_get_price_maps_periods(cfg: AppConfig) -> None:
    c = _with(cfg, tariff_scheme="three_stage", tariff_peak=7.0, tariff_mid=4.0, tariff_offpeak=2.0)
    tz = c.timezone
    assert pd_mod.get_price(pd.Timestamp("2025-07-15 18:00", tz=tz), c) == 7.0
    assert pd_mod.get_price(pd.Timestamp("2025-07-15 10:00", tz=tz), c) == 4.0
    assert pd_mod.get_price(pd.Timestamp("2025-07-15 03:00", tz=tz), c) == 2.0


def test_discharge_clamped_by_max_power(cfg: AppConfig) -> None:
    c = _with(cfg, batt_max_discharge_kw=5.0, batt_soc_min=0.1, batt_capacity_kwh=10.0)
    actual, _ = pd_mod.discharge_step(BatteryState(0.8), 99.0, c, 60.0)
    assert actual == pytest.approx(5.0)


def test_discharge_empty_returns_zero(cfg: AppConfig) -> None:
    c = _with(cfg, batt_soc_min=0.1)
    actual, new = pd_mod.discharge_step(BatteryState(0.1), 3.0, c, 15.0)
    assert actual == pytest.approx(0.0)
    assert new.soc == pytest.approx(0.1)


def test_surplus_charges_battery_then_curtail(cfg: AppConfig) -> None:
    c = _with(cfg, tariff_sell_price=0.0, batt_max_charge_kw=2.0, batt_capacity_kwh=10.0,
              batt_soc_max=0.95, batt_charge_eff=0.95, coupling="dc")
    pv, load = 5.0, 1.0
    r = pd_mod.dispatch_step(BatteryState(0.5), pv, load, "mid", c, 15.0)
    assert r["pv_to_load_kw"] == pytest.approx(1.0)
    assert r["pv_to_batt_kw"] == pytest.approx(2.0)
    assert r["pv_curtail_kw"] == pytest.approx(2.0)
    assert r["grid_out_kw"] == pytest.approx(0.0)
    assert _residual(pv, load, r) == pytest.approx(0.0)


def test_surplus_feed_in_when_sell_price(cfg: AppConfig) -> None:
    c = _with(cfg, tariff_sell_price=1.0, batt_max_charge_kw=2.0, coupling="dc")
    pv, load = 5.0, 1.0
    r = pd_mod.dispatch_step(BatteryState(0.5), pv, load, "mid", c, 15.0)
    assert r["grid_out_kw"] == pytest.approx(2.0)
    assert r["pv_curtail_kw"] == pytest.approx(0.0)
    assert _residual(pv, load, r) == pytest.approx(0.0)


def test_deficit_peak_discharges_battery(cfg: AppConfig) -> None:
    c = _with(cfg, batt_max_discharge_kw=2.0, batt_soc_min=0.1, batt_capacity_kwh=10.0,
              batt_discharge_eff=0.95)
    pv, load = 0.5, 3.0
    r = pd_mod.dispatch_step(BatteryState(0.8), pv, load, "peak", c, 15.0)
    assert r["pv_to_load_kw"] == pytest.approx(0.5)
    assert r["batt_dis_kw"] == pytest.approx(2.0)
    assert r["grid_in_kw"] == pytest.approx(0.5)
    assert _residual(pv, load, r) == pytest.approx(0.0)


def test_deficit_offpeak_uses_grid(cfg: AppConfig) -> None:
    c = _with(cfg, offpeak_charge_target_soc=0.0)
    pv, load = 0.0, 2.0
    r = pd_mod.dispatch_step(BatteryState(0.5), pv, load, "offpeak", c, 15.0)
    assert r["batt_dis_kw"] == pytest.approx(0.0)
    assert r["grid_in_kw"] == pytest.approx(2.0)
    assert _residual(pv, load, r) == pytest.approx(0.0)


def test_offpeak_grid_charge_to_target(cfg: AppConfig) -> None:
    c = _with(cfg, offpeak_charge_target_soc=0.8, batt_soc_min=0.1, batt_soc_max=0.95,
              batt_max_charge_kw=5.0, batt_capacity_kwh=10.0, batt_charge_eff=0.95, coupling="dc")
    pv, load = 0.0, 1.0
    r = pd_mod.dispatch_step(BatteryState(0.5), pv, load, "offpeak", c, 15.0)
    assert r["batt_ch_kw"] > 0.0
    assert r["grid_in_kw"] > load
    assert r["soc"] > 0.5
    assert r["soc"] <= 0.8 + 1e-9
    assert _residual(pv, load, r) == pytest.approx(0.0)


def test_balanced_when_pv_equals_load(cfg: AppConfig) -> None:
    pv, load = 2.0, 2.0
    r = pd_mod.dispatch_step(BatteryState(0.5), pv, load, "peak", cfg, 15.0)
    assert r["pv_to_load_kw"] == pytest.approx(2.0)
    assert r["batt_ch_kw"] == pytest.approx(0.0)
    assert r["batt_dis_kw"] == pytest.approx(0.0)
    assert r["grid_in_kw"] == pytest.approx(0.0)
    assert _residual(pv, load, r) == pytest.approx(0.0)


def test_returns_all_contract_fields(cfg: AppConfig) -> None:
    r = pd_mod.dispatch_step(BatteryState(0.5), 1.0, 2.0, "peak", cfg, 15.0)
    for key in ("pv_to_load_kw", "pv_to_batt_kw", "pv_curtail_kw", "batt_ch_kw",
                "batt_dis_kw", "grid_in_kw", "grid_out_kw", "soc"):
        assert key in r


# =============================================================================
# 模組 5：hems_simulation
# =============================================================================
def test_load_profile_shape_positive(cfg: AppConfig) -> None:
    idx = pd.date_range(cfg.start_date, periods=96, freq="15min", tz=cfg.timezone)
    load = h.build_load_profile(cfg, idx)
    assert len(load) == 96
    assert (load >= 0).all()
    assert load.sum() > 0


def test_plan_target_sunny_day_no_charge(cfg: AppConfig) -> None:
    idx = pd.date_range("2025-07-15 00:00", periods=96, freq="15min", tz=cfg.timezone)
    hours = idx.hour + idx.minute / 60.0
    pv = pd.Series(np.where((hours >= 8) & (hours < 16), 5.0, 0.0), index=idx)
    load = pd.Series(2.0, index=idx)
    assert h.plan_offpeak_target_soc(pv, load, cfg) == pytest.approx(cfg.batt_soc_min)


def test_plan_target_cloudy_day_charges(cfg: AppConfig) -> None:
    idx = pd.date_range("2025-07-15 00:00", periods=96, freq="15min", tz=cfg.timezone)
    pv = pd.Series(0.0, index=idx)
    load = pd.Series(2.0, index=idx)
    assert h.plan_offpeak_target_soc(pv, load, cfg) > cfg.batt_soc_min


def test_query_tomorrow_solar_perfect(cfg: AppConfig) -> None:
    idx = pd.date_range("2025-07-15 00:00", periods=96, freq="15min", tz=cfg.timezone)
    pv = pd.Series(np.linspace(0, 3, 96), index=idx)
    f = h.query_tomorrow_solar(pv, idx[0].date(), 0.0, np.random.default_rng(0))
    assert np.allclose(f.to_numpy(), pv.to_numpy())


@pytest.mark.parametrize("strategy", ["rule_based", "rule_based_forecast", "milp_mpc"])
def test_run_simulation_balanced(cfg: AppConfig, strategy: str) -> None:
    c = dataclasses.replace(cfg, control_strategy=strategy)
    df = h.run_simulation(c)
    assert len(df) == int(c.sim_days * 24 * 60 / c.timestep_min)
    for col in ("pv_kw", "load_kw", "soc", "grid_in_kw", "batt_ch_kw", "batt_dis_kw"):
        assert col in df.columns
    assert _balance_ok(df)
    assert df["soc"].min() >= c.batt_soc_min - 1e-6
    assert df["soc"].max() <= c.batt_soc_max + 1e-6


def test_milp_feedin_no_grid_arbitrage(cfg: AppConfig) -> None:
    """售電價>0 的 MILP：饋網量不得超過太陽能（杜絕買電轉賣的無限套利），且功率平衡成立。"""
    c = dataclasses.replace(cfg, control_strategy="milp_mpc", tariff_sell_price=5.6279)
    df = h.run_simulation(c)
    assert _balance_ok(df)
    dt_h = c.timestep_min / 60.0
    assert (df["grid_out_kw"] * dt_h).sum() <= (df["pv_kw"] * dt_h).sum() + 1e-6


def test_unknown_strategy_raises(cfg: AppConfig) -> None:
    with pytest.raises(ValueError):
        h.run_simulation(dataclasses.replace(cfg, control_strategy="foobar"))


def test_metrics_sane(cfg: AppConfig) -> None:
    c = dataclasses.replace(cfg, control_strategy="rule_based")
    m = h.compute_metrics(h.run_simulation(c), c)
    assert 0.0 <= m["scr"] <= 1.0
    assert 0.0 <= m["ssr"] <= 1.0
    for key in ("scr", "ssr", "savings_total", "payback_system_years",
                "net_battery_savings", "payback_battery_years"):
        assert key in m
    assert m["savings_total"] > 0


def test_forecast_not_worse_than_baseline(cfg: AppConfig) -> None:
    c_rb = dataclasses.replace(cfg, control_strategy="rule_based")
    c_fc = dataclasses.replace(cfg, control_strategy="rule_based_forecast")
    m_rb = h.compute_metrics(h.run_simulation(c_rb), c_rb)
    m_fc = h.compute_metrics(h.run_simulation(c_fc), c_fc)
    assert m_fc["bill"] <= m_rb["bill"] + 1e-6
