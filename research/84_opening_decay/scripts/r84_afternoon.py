#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is there a late-FRIDAY premium decay to harvest? (real NIFTY chain)

USER'S HYPOTHESIS: sell the Friday straddle ~14:00 and hold to ~15:02 — "the weekend is approaching,
people close positions, premium decays".

COUNTER-HYPOTHESIS worth testing at the same time: traders who are SHORT gamma and do not want
weekend risk must BUY BACK, which pushes premium UP on Friday afternoon. Weekend hedging demand can
make late Friday the WORST time to be short, not the best. Whichever is true, the data decides.

Measures, per weekday, for the AFTERNOON (12:00 -> 15:20):
  A) pure decay per 15-min window on CALM windows only (|move| < 0.05%) -> theta+IV, delta removed.
     A NEGATIVE number here means premium was RISING (bad for a seller).
  B) an entry-time sweep: sell the ATM straddle at 13:00/13:30/14:00/14:30/15:00, exit 15:20,
     actual P&L per lot including the move.
Medians (n ~11/weekday is thin). Read-only.
"""
import sqlite3
from collections import defaultdict
from datetime import date, timedelta
import numpy as np

ROOT = "/home/arun/quantifyd"
LOT, STEP = 65, 50
CALM = 0.0005
EXIT_T = 15 * 3600 + 20 * 60


def tsec(s):
    return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def nifty_exp(d):
    t = 1 if d >= date(2025, 9, 1) else 3
    a = (t - d.weekday()) % 7
    return d if a == 0 else d + timedelta(days=a)


c = sqlite3.connect(f"file:{ROOT}/backtest_data/options_data.db?mode=ro", uri=True)
c.row_factory = sqlite3.Row

calm = []          # (wd, bucket_start, d_premium_per_lot)
sweeps = []        # (wd, entry_label, pnl_per_lot)
ENTRIES = [(13 * 3600, "13:00"), (13 * 3600 + 1800, "13:30"), (14 * 3600, "14:00"),
           (14 * 3600 + 1800, "14:30"), (15 * 3600, "15:00")]

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
            wd = d0.strftime("%a")
            # ATM fixed off the 12:00 spot so the afternoon is measured on ONE strike
            s_noon = next((spot[t] for t in times if t >= 12 * 3600 and t in spot), None)
            if s_noon:
                K = round(s_noon / STEP) * STEP

                def px(t, ot, fb=None):
                    best = fb
                    for x in times:
                        if x > t: break
                        if (K, ot) in chain[x]: best = chain[x][(K, ot)]
                    return best

                # ---- A) calm-window decay through the afternoon ----
                buck = {}
                for t in times:
                    if 12 * 3600 <= t <= EXIT_T and (K, "CE") in chain[t] and (K, "PE") in chain[t] and t in spot:
                        b = (t // 900) * 900
                        buck[b] = (chain[t][(K, "CE")] + chain[t][(K, "PE")], spot[t])
                ks = sorted(buck)
                for i in range(1, len(ks)):
                    if ks[i] - ks[i - 1] != 900: continue
                    p0, sp0 = buck[ks[i - 1]]; p1, sp1 = buck[ks[i]]
                    if abs(sp1 - sp0) / sp0 < CALM:
                        calm.append((wd, ks[i], (p0 - p1) * LOT))   # +ve = decay earned
                # ---- B) entry-time sweep ----
                for t_in, lab in ENTRIES:
                    ce0, pe0 = px(t_in + 120, "CE"), px(t_in + 120, "PE")
                    if not ce0 or not pe0: continue
                    cx, pxx = px(EXIT_T, "CE", ce0), px(EXIT_T, "PE", pe0)
                    sweeps.append((wd, lab, ((ce0 - cx) + (pe0 - pxx)) * LOT, ce0 + pe0))
    d0 += timedelta(days=1)
c.close()

order = [("Mon", 1), ("Tue", 0), ("Wed", 6), ("Thu", 5), ("Fri", 4)]

print("=== A) AFTERNOON pure decay (calm windows only, delta removed) — Rs/lot per 15min ===")
print("    POSITIVE = premium decaying (good to be short). NEGATIVE = premium RISING (bad).\n")
print(f"{'day':>4s} {'dte':>3s} {'windows':>8s} {'median':>8s} {'implied/hour':>13s}")
for wd, dte in order:
    v = [x[2] for x in calm if x[0] == wd]
    if len(v) < 5: continue
    m = np.median(v)
    print(f"{wd:>4s} {dte:>3d} {len(v):>8d} {m:>+8.0f} {m*4:>+13.0f}")

print("\n\n=== FRIDAY hour by hour (is the last hour special?) ===\n")
print(f"{'window':>13s} {'n':>4s} {'median Rs/lot/15min':>21s}")
for lo, hi, lab in [(12 * 3600, 13 * 3600, "12:00-13:00"), (13 * 3600, 14 * 3600, "13:00-14:00"),
                    (14 * 3600, 15 * 3600, "14:00-15:00"), (15 * 3600, EXIT_T, "15:00-15:20")]:
    v = [x[2] for x in calm if x[0] == "Fri" and lo <= x[1] < hi]
    if len(v) < 3: continue
    print(f"{lab:>13s} {len(v):>4d} {np.median(v):>+21.0f}")

print("\n\n=== B) SELL the ATM straddle at T, exit 15:20 — median Rs/lot (includes the move) ===\n")
print(f"{'entry':>7s} " + " ".join(f"{w:>9s}" for w, _ in order))
for t_in, lab in ENTRIES:
    line = f"{lab:>7s} "
    for wd, _ in order:
        v = [x[2] for x in sweeps if x[0] == wd and x[1] == lab]
        line += f"{np.median(v):>+9.0f} " if len(v) >= 3 else f"{'-':>9s} "
    print(line)

print("\n--- Friday only: n and median credit at each entry ---")
for t_in, lab in ENTRIES:
    v = [x[2] for x in sweeps if x[0] == "Fri" and x[1] == lab]
    cr = [x[3] for x in sweeps if x[0] == "Fri" and x[1] == lab]
    if v:
        print("  %s n=%-3d median %+7.0f  mean %+7.0f  credit %.0f" % (
            lab, len(v), np.median(v), np.mean(v), np.median(cr)))
print("\n(n ~11 Fridays — THIN. Medians shown.)")
print("DONE")
