#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/83 — the giveback: can ANY exit/re-entry policy keep it? (chain-based, honest stats)

THE PROBLEM (measured on the real live book, 36 days): the 09:16 book peaks at ~+Rs7,708 and closes
at ~+Rs1,401 — it gives back Rs6,308/day, 82% of the peak. That is real and expensive.

WHAT ALREADY FAILED:
  * "close at +Rs8k" (user's idea)  -> +1,275 vs +1,401 hold. Dead.
  * "trail arm+3k / 25% retrace" (my idea) -> mean +1,118 BUT median -114, hurts 18/36 days,
    edge vanishes without the 2 best days, bootstrap CI [-1663,+4041] includes ZERO. Noise.
  * research/76 (n=14, idealised straddle): "book-early NOT supported"; decay is BACK-loaded;
    the erosion driver is move-stop CHURN. Its winners were EXIT1500 (+962/lot) and LULLHOLD (+768).

WHAT IS STILL UNTESTED — and the reason for this study:
  **RE-ENTRY.** Every test so far closes and stays FLAT. The user's actual proposal was
  "close all AND RE-ENTER LATER". That cannot be answered from a P&L curve (a curve cannot price a
  NEW straddle) — it needs the chain. This does that.

DISCIPLINE (the thing that killed the trail — applied from the start, not after):
  * report MEDIAN and OUTLIER-STRIPPED mean, not win-rate (win-rate is how clipping policies lie)
  * bootstrap 95% CI of the mean -> if it includes 0, it is noise, full stop
  * split by DTE from the start
  * costs modelled: 0.15%/leg per transaction (a re-enter = 4 leg-txns)
Per 1 lot (65) so it is scale-free. Read-only.
"""
import sqlite3, sys, time
from collections import defaultdict
from datetime import datetime, date, timedelta
import numpy as np

ROOT = "/home/arun/quantifyd"
ODB = f"{ROOT}/backtest_data/options_data.db"
LOT = 65
COST = 0.0015          # 0.15% per leg per transaction
T_ENTRY = 9 * 3600 + 16 * 60
T_EOD = 15 * 3600 + 15 * 60
STEP = 50


def tsec(s):
    return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def nifty_exp(d):
    t = 1 if d >= date(2025, 9, 1) else 3
    a = (t - d.weekday()) % 7
    return d if a == 0 else d + timedelta(days=a)


c = sqlite3.connect(f"file:{ODB}?mode=ro", uri=True)
c.row_factory = sqlite3.Row

# calendar-driven (no scans) — the chain table is ~20M rows
first, last = date(2026, 4, 20), date.today()
days = []
cur = first
while cur <= last:
    if cur.weekday() < 5:
        days.append(cur)
    cur += timedelta(days=1)

SIMS = []
t0 = time.time()
for d0 in days:
    day = d0.isoformat()
    E = nifty_exp(d0).isoformat()
    lo, hi = day + "T00:00", day + "T23:59"
    rows = c.execute("SELECT snapshot_time,strike,instrument_type,ltp,underlying_spot FROM option_chain "
                     "WHERE symbol='NIFTY' AND snapshot_time>=? AND snapshot_time<=? AND expiry_date=? "
                     "AND ltp>0 ORDER BY snapshot_time", (lo, hi, E)).fetchall()
    if len(rows) < 500:
        continue
    chain, spot = defaultdict(dict), {}
    for r in rows:
        t = tsec(r["snapshot_time"])
        chain[t][(int(r["strike"]), r["instrument_type"])] = r["ltp"]
        if r["underlying_spot"]:
            spot[t] = float(r["underlying_spot"])
    times = sorted(chain)
    if not times:
        continue
    dte = (nifty_exp(d0) - d0).days
    SIMS.append((day, d0.strftime("%a"), dte, chain, spot, times))
    print("  loaded %s DTE%d (%d snaps)" % (day, dte, len(times))); sys.stdout.flush()
c.close()
print("days loaded: %d in %.0fs" % (len(SIMS), time.time() - t0))


def px(chain, times, t, K, ot, fb=None):
    best = fb
    for x in times:
        if x > t:
            break
        if (K, ot) in chain[x]:
            best = chain[x][(K, ot)]
    return best


def straddle_path(chain, spot, times, t_from):
    """Enter ATM at t_from; return (K, ce0, pe0, path[(t, pnl_per_lot_gross)])."""
    s0 = next((spot[t] for t in times if t >= t_from and t in spot), None)
    if not s0:
        return None
    K = round(s0 / STEP) * STEP
    ce0 = px(chain, times, t_from + 120, K, "CE")
    pe0 = px(chain, times, t_from + 120, K, "PE")
    if not ce0 or not pe0:
        return None
    path = []
    lce, lpe = ce0, pe0
    for t in times:
        if t < t_from or t > T_EOD:
            continue
        lce = chain[t].get((K, "CE"), lce)
        lpe = chain[t].get((K, "PE"), lpe)
        path.append((t, ((ce0 - lce) + (pe0 - lpe)) * LOT))
    return (K, ce0, pe0, path)


def entry_cost(ce0, pe0):
    return (ce0 + pe0) * COST * LOT      # 2 legs opened


def exit_cost(chain, times, K, t):
    ce = px(chain, times, t, K, "CE") or 0
    pe = px(chain, times, t, K, "PE") or 0
    return (ce + pe) * COST * LOT        # 2 legs closed


def run_policy(sim, kind, p1=None, p2=None, reenter_after=None):
    day, wd, dte, chain, spot, times = sim
    st = straddle_path(chain, spot, times, T_ENTRY)
    if not st:
        return None
    K, ce0, pe0, path = st
    if not path:
        return None
    cost = entry_cost(ce0, pe0)
    exit_t, exit_pnl = None, None

    if kind == "hold":
        exit_t, exit_pnl = path[-1]
    elif kind == "exit1500":
        for t, v in path:
            if t >= 15 * 3600:
                exit_t, exit_pnl = t, v; break
        if exit_t is None:
            exit_t, exit_pnl = path[-1]
    elif kind == "target":
        for t, v in path:
            if v >= p1:
                exit_t, exit_pnl = t, v; break
        if exit_t is None:
            exit_t, exit_pnl = path[-1]
    elif kind == "trail":
        pk = -1e9; armed = False
        for t, v in path:
            pk = max(pk, v)
            if pk >= p1:
                armed = True
            if armed and v <= pk * (1 - p2):
                exit_t, exit_pnl = t, v; break
        if exit_t is None:
            exit_t, exit_pnl = path[-1]

    total = exit_pnl - cost - exit_cost(chain, times, K, exit_t)

    # ---- optional RE-ENTRY: fresh ATM straddle `reenter_after` minutes later, held to 15:15 ----
    if reenter_after is not None and exit_t < T_EOD - 600:
        t2 = exit_t + reenter_after * 60
        st2 = straddle_path(chain, spot, times, t2)
        if st2:
            K2, ce2, pe2, path2 = st2
            if path2:
                t_end, v_end = path2[-1]
                total += v_end - entry_cost(ce2, pe2) - exit_cost(chain, times, K2, t_end)
    return dict(day=day, wd=wd, dte=dte, pnl=total)


def stats(rows, lab):
    if not rows:
        return
    a = np.array([r["pnl"] for r in rows], float)
    rng = np.random.default_rng(11)
    bs = [rng.choice(a, len(a), replace=True).mean() for _ in range(3000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    strip = np.sort(a)[:-2].mean() if len(a) > 4 else float("nan")   # drop the 2 best
    print("%-30s n=%-3d mean %+7.0f  MEDIAN %+7.0f  strip2 %+7.0f  CI[%+6.0f,%+6.0f] %s" % (
        lab, len(a), a.mean(), np.median(a), strip, lo, hi,
        "REAL" if lo > 0 else "noise"))


POLICIES = [
    ("HOLD (baseline)", dict(kind="hold")),
    ("EXIT 15:00", dict(kind="exit1500")),
    ("TARGET +1000/lot", dict(kind="target", p1=1000)),
    ("TARGET +1500/lot", dict(kind="target", p1=1500)),
    ("TARGET +2000/lot", dict(kind="target", p1=2000)),
    ("TRAIL arm1000 ret25%", dict(kind="trail", p1=1000, p2=0.25)),
    ("TRAIL arm1500 ret30%", dict(kind="trail", p1=1500, p2=0.30)),
    # ---- the untested half: close AND RE-ENTER ----
    ("TGT+1500 -> REENTER 30m", dict(kind="target", p1=1500, reenter_after=30)),
    ("TGT+1500 -> REENTER 60m", dict(kind="target", p1=1500, reenter_after=60)),
    ("TGT+2000 -> REENTER 30m", dict(kind="target", p1=2000, reenter_after=30)),
    ("TRAIL1500/30 -> REENTER 30m", dict(kind="trail", p1=1500, p2=0.30, reenter_after=30)),
]

print("\n=== research/83: exit + RE-ENTRY policies, per 1 lot, net of 0.15%/leg ===")
print("    (win-rate deliberately NOT reported — it is how clipping policies lie)\n")
for lab, kw in POLICIES:
    rows = [r for r in (run_policy(s, **kw) for s in SIMS) if r]
    stats(rows, lab)

print("\n--- by DTE (the split that matters) ---")
for lab, kw in POLICIES:
    rows = [r for r in (run_policy(s, **kw) for s in SIMS) if r]
    a01 = [r for r in rows if r["dte"] <= 1]
    afar = [r for r in rows if r["dte"] >= 2]
    if a01 and afar:
        m1 = np.array([r["pnl"] for r in a01]); m2 = np.array([r["pnl"] for r in afar])
        print("%-30s DTE0/1 mean %+7.0f med %+7.0f | DTE2+ mean %+7.0f med %+7.0f" % (
            lab, m1.mean(), np.median(m1), m2.mean(), np.median(m2)))
print("\nDONE")
