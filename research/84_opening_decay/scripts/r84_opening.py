#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which weekday gives the best OPENING-HOURS premium decay? (real NIFTY chain)

NOTE: for NIFTY, weekday IS dte (Tuesday expiry) — Mon=DTE1, Tue=DTE0, Wed=DTE6, Thu=DTE5, Fri=DTE4.
So "which day" and "which DTE" are one question, not two.

Two different things are measured, because they answer different questions:
  A) WHAT YOU ACTUALLY CAPTURE — the 09:16 ATM straddle's P&L at +30/+60/+90 min.
     This includes the directional move (delta), because that is what really happens to you.
  B) PURE DECAY — the premium drop measured ONLY across 15-min windows where the index barely
     moved (<0.05%). With delta removed, what is left is theta+IV. This explains the WHY.

Reported per 1 lot (65), MEDIAN first (means are outlier-noise on n~11/day), gross of cost.
Read-only.
"""
import sqlite3, sys
from collections import defaultdict
from datetime import date, timedelta, datetime
import numpy as np

ROOT = "/home/arun/quantifyd"
LOT, STEP = 65, 50
T_ENTRY = 9 * 3600 + 16 * 60
CALM = 0.0005          # |move| < 0.05% in the window => that window is pure decay


def tsec(s):
    return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def nifty_exp(d):
    t = 1 if d >= date(2025, 9, 1) else 3
    a = (t - d.weekday()) % 7
    return d if a == 0 else d + timedelta(days=a)


c = sqlite3.connect(f"file:{ROOT}/backtest_data/options_data.db?mode=ro", uri=True)
c.row_factory = sqlite3.Row

rows_out = []
calm_rows = []
d0 = date(2026, 4, 20)
while d0 <= date.today():
    if d0.weekday() < 5:
        day = d0.isoformat(); E = nifty_exp(d0).isoformat()
        rs = c.execute("SELECT snapshot_time,strike,instrument_type,ltp,underlying_spot FROM option_chain "
                       "WHERE symbol='NIFTY' AND snapshot_time>=? AND snapshot_time<=? AND expiry_date=? "
                       "AND ltp>0 ORDER BY snapshot_time", (day + "T00:00", day + "T23:59", E)).fetchall()
        if len(rs) > 500:
            chain, spot = defaultdict(dict), {}
            for r in rs:
                t = tsec(r["snapshot_time"])
                chain[t][(int(r["strike"]), r["instrument_type"])] = r["ltp"]
                if r["underlying_spot"]:
                    spot[t] = float(r["underlying_spot"])
            times = sorted(chain)
            s0 = next((spot[t] for t in times if t >= T_ENTRY and t in spot), None)
            if s0:
                K = round(s0 / STEP) * STEP
                def px(t, ot, fb=None):
                    best = fb
                    for x in times:
                        if x > t: break
                        if (K, ot) in chain[x]: best = chain[x][(K, ot)]
                    return best
                ce0, pe0 = px(T_ENTRY + 120, "CE"), px(T_ENTRY + 120, "PE")
                if ce0 and pe0:
                    dte = (nifty_exp(d0) - d0).days
                    wd = d0.strftime("%a")
                    rec = dict(day=day, wd=wd, dte=dte, credit=ce0 + pe0)
                    for mins, lab in ((30, "p30"), (60, "p60"), (90, "p90"), (120, "p120")):
                        t = T_ENTRY + mins * 60
                        cx, pxx = px(t, "CE", ce0), px(t, "PE", pe0)
                        rec[lab] = ((ce0 - cx) + (pe0 - pxx)) * LOT
                    rows_out.append(rec)
                    # ---- B) pure decay on CALM 15-min windows, opening hours only (09:16-11:16)
                    buck = {}
                    for t in times:
                        if T_ENTRY <= t <= T_ENTRY + 120 * 60 and (K, "CE") in chain[t] and (K, "PE") in chain[t] and t in spot:
                            b = (t // 900) * 900
                            buck[b] = (chain[t][(K, "CE")] + chain[t][(K, "PE")], spot[t])
                    ks = sorted(buck)
                    for i in range(1, len(ks)):
                        if ks[i] - ks[i - 1] != 900: continue
                        p0, sp0 = buck[ks[i - 1]]; p1, sp1 = buck[ks[i]]
                        if abs(sp1 - sp0) / sp0 < CALM:
                            calm_rows.append(dict(wd=wd, dte=dte, dp=(p1 - p0) * LOT))
    d0 += timedelta(days=1)
c.close()

print("=== A) WHAT YOU ACTUALLY CAPTURE — 09:16 ATM straddle, per lot, MEDIAN (n per day) ===")
print("    (weekday IS dte for NIFTY: Mon=1 Tue=0 Wed=6 Thu=5 Fri=4)\n")
print(f"{'day':>4s} {'dte':>3s} {'n':>3s} {'credit':>7s} {'+30min':>8s} {'+60min':>8s} {'+90min':>8s} {'+120min':>8s}")
order = [("Mon", 1), ("Tue", 0), ("Wed", 6), ("Thu", 5), ("Fri", 4)]
for wd, dte in order:
    sub = [r for r in rows_out if r["wd"] == wd]
    if not sub: continue
    print(f"{wd:>4s} {dte:>3d} {len(sub):>3d} {np.median([r['credit'] for r in sub]):>7.0f} "
          f"{np.median([r['p30'] for r in sub]):>+8.0f} {np.median([r['p60'] for r in sub]):>+8.0f} "
          f"{np.median([r['p90'] for r in sub]):>+8.0f} {np.median([r['p120'] for r in sub]):>+8.0f}")

print("\n\n=== B) PURE DECAY — only 15-min windows where the index barely moved (<0.05%) ===")
print("    delta removed -> what is left is theta+IV. Rs/lot per 15-min window, MEDIAN.\n")
print(f"{'day':>4s} {'dte':>3s} {'windows':>8s} {'median decay/15min':>19s} {'implied/hour':>13s}")
for wd, dte in order:
    sub = [r["dp"] for r in calm_rows if r["wd"] == wd]
    if len(sub) < 5: continue
    m = np.median(sub)
    print(f"{wd:>4s} {dte:>3d} {len(sub):>8d} {-m:>+19.0f} {-m*4:>+13.0f}")

print("\n(n is ~11-12 days per weekday — thin. Medians shown; means are outlier noise at this size.)")
print("DONE")
