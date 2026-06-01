"""fast_core.py — 規則式調度的 JIT 加速核心。

把 `power_dispatch.dispatch_step` ＋ `battery_charge` 的 SOC 數學，改寫成一支
「純數值、無物件」的時間迴圈，讓 numba 能以 nopython 模式編譯成機器碼，全年
35040 步可在毫秒級跑完，供大量參數掃描使用。

設計重點：
- 行為必須與物件版 `dispatch_step` 完全一致（有單元測試交叉驗證殘差與數值）。
- 時段以整數碼表示：0=offpeak, 1=mid, 2=peak, 3=flat（一般式）。
- 每步可帶不同的「離峰充電目標 SOC / 半尖峰保留 SOC / 智慧放電開關」，
  以支援預測式策略（rule_based_forecast）的逐日設定。
- numba 缺席時自動退回純 Python（較慢但結果相同）。
"""

from __future__ import annotations

import numpy as np

# numba 為選用相依；缺少時 njit 退化為「直接回傳原函式」（純 Python 執行）。
try:
    from numba import njit  # type: ignore

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        if args and callable(args[0]):
            return args[0]

        def _deco(fn):
            return fn

        return _deco


def _rule_loop_impl(
    pv, load, period, price, offpeak_target, mid_reserve, smart_flag,
    cap, soc_min, soc_max, max_ch, max_dis, eta_ch, eta_dis, deg_cost, dt_h, sell,
    charge_priority,
):
    """規則式自用優先調度的逐步迴圈（數值核心）。

    參數（陣列長度皆為 n；單位：功率 kW、能量 kWh、SOC 0~1、時間 dt_h 小時）:
        pv, load: 每步太陽能與負載功率。
        period: 每步時段碼（0 離峰 / 1 半尖峰 / 2 尖峰 / 3 一般式）。
        price: 每步購電單價（元/度），供智慧放電衰減閘判斷。
        offpeak_target: 每步「離峰用市電補電池」的目標 SOC（≤soc_min 視為關閉）。
        mid_reserve: 每步「半尖峰為尖峰保留」的最低 SOC。
        smart_flag: 每步是否啟用智慧放電（1/0）。
        cap, soc_min, soc_max, max_ch, max_dis: 電池容量與限制。
        eta_ch, eta_dis: 充電路徑總效率（含耦合）、放電效率。
        deg_cost: 衰減成本（元/kWh 吞吐），智慧放電門檻用。
        dt_h: 時步（小時）。
        sell: 售電價（>0 才饋電，否則餘電限發）。
        charge_priority: 0=load_first（PV 先供電、餘電充電，標準自用優先）；
                         1=battery_first（日照時 PV 先把電池充滿、再供電，缺口轉市電）。
    回傳:
        tuple of 9 個 np.ndarray：
        (batt_ch, batt_dis, grid_in, grid_out, pv_to_load, pv_to_batt,
         grid_to_batt, pv_curtail, soc)。
    """
    n = pv.shape[0]
    ch = np.zeros(n)
    dis = np.zeros(n)
    gin = np.zeros(n)
    gout = np.zeros(n)
    p2l = np.zeros(n)
    p2b = np.zeros(n)
    g2b = np.zeros(n)
    cur = np.zeros(n)
    socout = np.zeros(n)

    soc = soc_min
    for i in range(n):
        pvi = pv[i]
        ldi = load[i]
        if pvi < 0.0:
            pvi = 0.0
        if ldi < 0.0:
            ldi = 0.0

        pb = 0.0
        gb = 0.0
        cu = 0.0
        ds = 0.0
        gi = 0.0
        go = 0.0
        pl = 0.0
        loadrem = 0.0

        if charge_priority == 1:
            # battery_first：PV 先充電池（充滿優先），再供負載，缺口轉放電/市電。
            headroom = (soc_max - soc) * cap
            if headroom < 0.0:
                headroom = 0.0
            maxchpv = headroom / (eta_ch * dt_h) if (eta_ch > 0.0 and dt_h > 0.0) else 0.0
            if maxchpv > max_ch:
                maxchpv = max_ch
            if maxchpv < 0.0:
                maxchpv = 0.0
            pb = pvi if pvi < maxchpv else maxchpv
            soc = soc + eta_ch * pb * dt_h / cap
            if soc > soc_max:
                soc = soc_max
            pv_rem = pvi - pb
            pl = pv_rem if pv_rem < ldi else ldi
            pv_left = pv_rem - pl
            loadrem = ldi - pl
            if pv_left > 0.0:
                if sell > 0.0:
                    go = pv_left
                else:
                    cu = pv_left
        else:
            # load_first（標準自用優先）：PV 先供負載，餘電才充電。
            pl = pvi if pvi < ldi else ldi
            surplus = pvi - pl
            loadrem = ldi - pl
            if surplus > 0.0:
                headroom = (soc_max - soc) * cap
                if headroom < 0.0:
                    headroom = 0.0
                maxchpv = headroom / (eta_ch * dt_h) if (eta_ch > 0.0 and dt_h > 0.0) else 0.0
                if maxchpv > max_ch:
                    maxchpv = max_ch
                if maxchpv < 0.0:
                    maxchpv = 0.0
                pb = surplus if surplus < maxchpv else maxchpv
                soc = soc + eta_ch * pb * dt_h / cap
                if soc > soc_max:
                    soc = soc_max
                leftover = surplus - pb
                if sell > 0.0:
                    go = leftover
                else:
                    cu = leftover
                loadrem = 0.0  # 有餘電即無缺口

        # 統一缺口處理（兩種充電哲學共用）。
        if loadrem > 0.0:
            pc = period[i]
            if pc == 1 or pc == 2 or pc == 3:
                do_dis = True
                floor = soc_min
                if smart_flag[i] == 1:
                    if price[i] > deg_cost:
                        if pc == 1:  # 半尖峰：為尖峰保留
                            floor = mid_reserve[i]
                            if floor < soc_min:
                                floor = soc_min
                        else:
                            floor = soc_min
                    else:
                        do_dis = False  # 衰減成本閘
                if do_dis:
                    avail = (soc - floor) * cap
                    if avail < 0.0:
                        avail = 0.0
                    maxdis = avail * eta_dis / dt_h if dt_h > 0.0 else 0.0
                    if maxdis > max_dis:
                        maxdis = max_dis
                    if maxdis < 0.0:
                        maxdis = 0.0
                    ds = loadrem if loadrem < maxdis else maxdis
                    soc = soc - (ds / eta_dis) * dt_h / cap if eta_dis > 0.0 else soc
                    if soc < soc_min:
                        soc = soc_min
                gi = loadrem - ds
            else:
                # 離峰：市電供負載；選用：補電池到目標 SOC。
                gi = loadrem
                target = offpeak_target[i]
                if target > soc_min and soc < target:
                    need = (target - soc) * cap
                    extkw = need / (eta_ch * dt_h) if (eta_ch > 0.0 and dt_h > 0.0) else 0.0
                    offer = extkw if extkw < max_ch else max_ch
                    headroom = (soc_max - soc) * cap
                    if headroom < 0.0:
                        headroom = 0.0
                    maxchg = headroom / (eta_ch * dt_h) if (eta_ch > 0.0 and dt_h > 0.0) else 0.0
                    if maxchg > max_ch:
                        maxchg = max_ch
                    if maxchg < 0.0:
                        maxchg = 0.0
                    gb = offer if offer < maxchg else maxchg
                    if gb < 0.0:
                        gb = 0.0
                    soc = soc + eta_ch * gb * dt_h / cap
                    if soc > soc_max:
                        soc = soc_max
                    gi = gi + gb

        ch[i] = pb + gb
        dis[i] = ds
        gin[i] = gi
        gout[i] = go
        p2l[i] = pl
        p2b[i] = pb
        g2b[i] = gb
        cur[i] = cu
        socout[i] = soc

    return ch, dis, gin, gout, p2l, p2b, g2b, cur, socout


# JIT 編譯（numba 缺席時即為純 Python 版本）。
rule_loop = njit(cache=True)(_rule_loop_impl)

# 時段字串 → 整數碼（與核心一致）。
PERIOD_CODE = {"offpeak": 0, "mid": 1, "peak": 2, "flat": 3}
