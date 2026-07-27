#!/usr/bin/env python3
"""research/90: EOD skew-guard test on the Monday arm (Arun's overnight-gap worry).

Base = NSR-W: Monday entry (EOD close prices), next-week expiry, premium-target
strikes, intraday stop 2.0x (pessimistic open/high fills), PT 50, one roll-away,
survivor rides, time exit DTE<=1.
NEW AXIS: EOD guard Y in {None, 1.3, 1.5, 1.75} - at each day's CLOSE, any live
leg whose close >= Y x its credit is exited AT THE CLOSE (roll rules apply),
instead of carrying the skewed leg overnight against a resting 2x GTT.
Question: does closing the heavy leg before the gap improve tail without
killing the mean? 2019-2026, 379 Mondays."""
import csv
import datetime as dt
import os
import sqlite3
import time
from collections import defaultdict

BASE = "/home/arun/quantifyd"
RESDIR = os.path.join(BASE, "research/90_nifty_strangle_rules/results")
DB = os.path.join(BASE, "backtest_data/market_data.db")
START = "2019-01-01"
MIN_CONTRACTS = 50
MIN_LEG_PTS = 1.5
COST_PCT = 0.005
COST_FLAT = 0.15
T_GRID = {"20": 20.0, "30": 30.0}   # rupee premium targets
STOP = 2.0
PT = 0.5
ROLLMODE_GRID = ['fixed', 'match', 'half', 'restrangle']
GUARD = 1.5
DTE_LO, DTE_HI = 6, 12
TIME_EXIT_DTE = 1


def log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def load_chain():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute(
        "SELECT trade_date, expiry_date, option_type, strike, open, high, close, "
        "settle_price, contracts FROM nse_options_bhav "
        "WHERE symbol='NIFTY' AND trade_date>='2018-12-01'")
    data, days_set, exp_set = {}, set(), set()
    for td, ed, ot, k, o, h, cl, sp, c in cur:
        p = cl if (cl or 0) > 0 else (sp or 0)
        if p <= 0:
            continue
        data[(td, ed, ot, float(k))] = (o or p, max(h or p, p), p, c or 0)
        days_set.add(td)
        exp_set.add(ed)
    con.close()
    return data, sorted(days_set), sorted(exp_set)


def build_day_idx(data):
    idx = defaultdict(lambda: defaultdict(lambda: {"CE": {}, "PE": {}}))
    for (td, ed, ot, k), v in data.items():
        idx[td][ed][ot][k] = v
    return idx


def calc_spot(day_idx, day, expiries):
    for e in expiries:
        if e >= day:
            ch = day_idx.get(day, {}).get(e)
            if not ch or not ch["CE"] or not ch["PE"]:
                continue
            common = set(ch["CE"]) & set(ch["PE"])
            if not common:
                continue
            k = min(common, key=lambda s: abs(ch["CE"][s][2] - ch["PE"][s][2]))
            return k + ch["CE"][k][2] - ch["PE"][k][2]
    return None


def pick_by_premium(day_idx, day, exp, ot, target, spot):
    ch = day_idx.get(day, {}).get(exp, {}).get(ot, {})
    best, bd = None, None
    for k, v in ch.items():
        if (ot == "PE" and k >= spot) or (ot == "CE" and k <= spot):
            continue
        if v[3] < MIN_CONTRACTS or v[2] < MIN_LEG_PTS:
            continue
        d = abs(v[2] - target)
        if bd is None or d < bd:
            best, bd = k, d
    return best if (best is not None and bd is not None and bd <= target) else None


def build_series(day_idx, days, entry, exp, pe_k, ce_k):
    i0 = days.index(entry)
    ed = dt.date.fromisoformat(exp)
    out, prev = [], None
    for d in days[i0:]:
        if d > exp:
            break
        ch = day_idx.get(d, {}).get(exp, {})
        pe = ch.get("PE", {}).get(pe_k)
        ce = ch.get("CE", {}).get(ce_k)
        if prev:
            pe = pe or (prev[1][2], prev[1][2], prev[1][2], 0)
            ce = ce or (prev[2][2], prev[2][2], prev[2][2], 0)
        if pe is None or ce is None:
            return None
        prev = (d, pe, ce)
        out.append((d, pe, ce, (ed - dt.date.fromisoformat(d)).days))
    return out


def stop_fill(bar, level):
    o, h, c, _ = bar
    if o >= level:
        return o
    if h >= level:
        return level
    return None


def _patch(series, from_j, day_idx, exp, side, nk):
    out = list(series)
    prev_px = None
    for idx in range(from_j, len(out)):
        d, pe, ce, cdte = out[idx]
        bar = day_idx.get(d, {}).get(exp, {}).get(side, {}).get(nk)
        if bar is None:
            px = prev_px if prev_px is not None else (pe[2] if side == "PE" else ce[2])
            bar = (px, px, px, 0)
        prev_px = bar[2]
        out[idx] = (d, bar, ce, cdte) if side == "PE" else (d, pe, bar, cdte)
    return out


def sell_fresh(day_idx, d, exp, spots, target):
    spot_d = spots.get(d)
    if not spot_d:
        return None
    pe_k = pick_by_premium(day_idx, d, exp, "PE", target, spot_d)
    ce_k = pick_by_premium(day_idx, d, exp, "CE", target, spot_d)
    if not pe_k or not ce_k:
        return None
    return (pe_k, day_idx[d][exp]["PE"][pe_k][2]), (ce_k, day_idx[d][exp]["CE"][ce_k][2])


def sim(series, pe0, ce0, rollmode, day_idx, exp, spots, target):
    action = 'rc1'
    credit = pe0 + ce0
    sl = {"PE": STOP * pe0, "CE": STOP * ce0}
    cred = {"PE": pe0, "CE": ce0}
    live = dict(cred)
    realized, rolled = 0.0, False
    entry_prem, exit_prem = credit, 0.0
    rc_budget = {"rc1": 1, "rcU": 4}.get(action, 0)
    j = 0

    def close_leg(side, fill, d, jj):
        nonlocal realized, exit_prem, rolled, sl, live, entry_prem
        realized += live[side] - fill
        exit_prem += fill
        del live[side]
        if live and not rolled:
            rolled = True
            spot_d = spots.get(d)
            if spot_d:
                other = 'CE' if side == 'PE' else 'PE'
                obar = series[jj][1] if other == 'PE' else series[jj][2]
                tgt = target if rollmode == 'fixed' else (max(2.0, obar[2]) if rollmode == 'match' else target / 2)
                nk = pick_by_premium(day_idx, d, exp, side, tgt, spot_d)
                if nk:
                    bar = day_idx[d][exp][side][nk]
                    live[side] = bar[2]
                    sl[side] = STOP * bar[2]
                    entry_prem += bar[2]
                    return _patch(series, jj, day_idx, exp, side, nk)
        return None

    while j < len(series) - 1:
        j += 1
        d, pe, ce, cdte = series[j]
        bars = {"PE": pe, "CE": ce}
        hit = next((s for s in list(live) if stop_fill(bars[s], sl[s]) is not None), None)
        if hit:
            if rollmode == 'restrangle' and not rolled:
                rolled = True
                fill = stop_fill(bars[hit], sl[hit])
                realized += live[hit] - fill
                exit_prem += fill
                del live[hit]
                for s2 in list(live):
                    realized += live[s2] - bars[s2][2]
                    exit_prem += bars[s2][2]
                    del live[s2]
                fresh = sell_fresh(day_idx, d, exp, spots, target)
                if not fresh:
                    return realized - cost(entry_prem, exit_prem), 'RS_FLAT'
                (npe, npep), (nce, ncep) = fresh
                live = {'PE': npep, 'CE': ncep}
                cred = dict(live)
                sl = {'PE': STOP * npep, 'CE': STOP * ncep}
                entry_prem += npep + ncep
                series = _patch(series, j, day_idx, exp, 'PE', npe)
                series = _patch(series, j, day_idx, exp, 'CE', nce)
                continue
            ns = close_leg(hit, stop_fill(bars[hit], sl[hit]), d, j)
            if ns is not None:
                series = ns
            if not live:
                return realized - cost(entry_prem, exit_prem), "STOP2"
            continue
        comb = sum(bars[s][2] for s in live)
        profit = sum(live[s] for s in live) - comb + realized
        if profit >= PT * credit:
            return profit - cost(entry_prem, exit_prem + comb), "PT"
        if cdte <= TIME_EXIT_DTE:
            return profit - cost(entry_prem, exit_prem + comb), "TIME"
        # EOD skew action at GUARD x
        if action != "none":
            heavy = next((s for s in list(live) if bars[s][2] >= GUARD * cred_of(cred, s)), None)
            if heavy:
                if action == "exit" or rc_budget <= 0:
                    ns = close_leg(heavy, bars[heavy][2], d, j)
                    if ns is not None:
                        series = ns
                    if not live:
                        return realized - cost(entry_prem, exit_prem), "GUARD2"
                else:
                    rc_budget -= 1
                    for s in list(live):
                        realized += live[s] - bars[s][2]
                        exit_prem += bars[s][2]
                        del live[s]
                    fresh = sell_fresh(day_idx, d, exp, spots, target)
                    if not fresh:
                        return realized - cost(entry_prem, exit_prem), "RC_FLAT"
                    (npe, npep), (nce, ncep) = fresh
                    live = {"PE": npep, "CE": ncep}
                    cred = dict(live)
                    sl = {"PE": STOP * npep, "CE": STOP * ncep}
                    entry_prem += npep + ncep
                    series = _patch(series, j, day_idx, exp, "PE", npe)
                    series = _patch(series, j, day_idx, exp, "CE", nce)
    d, pe, ce, _ = series[-1]
    bars = {"PE": pe, "CE": ce}
    comb = sum(bars[s][2] for s in live)
    return sum(live[s] for s in live) - comb + realized - cost(entry_prem, exit_prem + comb), "EXPIRY"


def cred_of(cred, s):
    return cred[s]


def cost(ep, xp):
    return COST_PCT * (ep + xp) + COST_FLAT


def main():
    t0 = time.time()
    data, days, expiries = load_chain()
    day_idx = build_day_idx(data)
    spots = {}
    for d in days:
        if d >= START:
            s = calc_spot(day_idx, d, expiries)
            if s:
                spots[d] = s
    log(f"spots {len(spots)}")
    mondays = []
    prev_wk = None
    for d in days:
        if d < START or d not in spots:
            continue
        wk = dt.date.fromisoformat(d).isocalendar()[:2]
        if wk != prev_wk:
            mondays.append(d)
            prev_wk = wk
    agg = defaultdict(list)
    reasons = defaultdict(lambda: defaultdict(int))
    for entry in mondays:
        d0 = dt.date.fromisoformat(entry)
        exp = None
        for e in expiries:
            if e > entry:
                cdte = (dt.date.fromisoformat(e) - d0).days
                if DTE_LO <= cdte <= DTE_HI:
                    exp = e
                    break
                if cdte > DTE_HI:
                    break
        if not exp:
            continue
        spot = spots[entry]
        for tn, T in T_GRID.items():
            pe_k = pick_by_premium(day_idx, entry, exp, "PE", T, spot)
            ce_k = pick_by_premium(day_idx, entry, exp, "CE", T, spot)
            if not pe_k or not ce_k:
                continue
            series = build_series(day_idx, days, entry, exp, pe_k, ce_k)
            if not series or len(series) < 2:
                continue
            pe0, ce0 = series[0][1][2], series[0][2][2]
            for rollmode in ROLLMODE_GRID:
                net, reason = sim(list(series), pe0, ce0, rollmode, day_idx, exp, spots, T)
                agg[(tn, rollmode)].append((net, entry[:4]))
                reasons[(tn, rollmode)][reason] += 1
    import statistics as st
    log("=== ROLL TARGET: fixed-T vs match-survivor vs half-T (v1.2 base) (Monday arm; 2019-2026) ===")
    for tn in T_GRID:
        for guard in ROLLMODE_GRID:
            s = [x[0] for x in agg[(tn, guard)]]
            if not s:
                continue
            n = len(s)
            mu = st.mean(s)
            sd = st.stdev(s)
            post = [x[0] for x in agg[(tn, guard)] if x[1] >= "2022"]
            log(f"T{tn} guard={guard or 'none'}: n={n} mean={mu:6.2f} t={mu/(sd/n**0.5):5.2f} "
                f"win={100*sum(1 for x in s if x>0)/n:4.1f}% worst={min(s):7.1f} "
                f"p5={sorted(s)[n//20]:7.1f} post22={st.mean(post):6.2f} "
                f"exits={dict(reasons[(tn,guard)])}")
    log(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
