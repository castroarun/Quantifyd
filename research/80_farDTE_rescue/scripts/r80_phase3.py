#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G3 — after the 0.4% band is hit, does price CONTINUE or REVERT?

The idea under test (user): the band is hit on ~77% of days, so instead of fading the move with a
neutral strangle, trade WITH it -- once price breaks 0.4% from the 09:20 spot, SELL the option on
the far side (break up -> sell a put below; break down -> sell a call above). You collect premium
while the move walks away from your strike.

That works if, and ONLY if, price tends to CONTINUE after the break. If it reverts through the
entry, you are selling straight into a whipsaw -- and the whipsaw is precisely what makes the
strangle lose.

So, on every day in 2015-02 -> 2026-03 (NIFTY 5-min), for days where the band is hit:
  * which way did it break, and WHEN
  * did it CONTINUE (further move in the break direction) or REVERT
  * how far back does it come -- MAE against a short on the far side
  * how often does it whipsaw ALL the way to the OPPOSITE band (the strangle's killer)
  * conditioned on VIX / opening range / CPR width / time-of-break

No options needed for the verdict -- if price does not continue, no strike or premium can save it.
Reads only.
"""
import sqlite3
from collections import defaultdict
import datetime as dt

import numpy as np

ROOT = "/home/arun/quantifyd"
BAND = 0.004
ENTRY_MIN, LAST_MIN, EOD_MIN = 9 * 60 + 20, 14 * 60 + 45, 15 * 60 + 15

c = sqlite3.connect(f"file:{ROOT}/backtest_data/market_data.db?mode=ro", uri=True)
bars_raw = c.execute(
    "SELECT date, open, high, low, close FROM market_data_unified "
    "WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date").fetchall()
vix = {r[0][:10]: r[1] for r in c.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
c.close()

days = defaultdict(list)
for d, o, h, l, cl in bars_raw:
    days[str(d)[:10]].append((str(d)[11:16], o, h, l, cl))

def mins(x):
    return int(x[:2]) * 60 + int(x[3:5])

recs = []
prev = None
for day in sorted(days):
    b = sorted(days[day])
    if len(b) < 40:
        continue
    dh, dl, dc = max(x[2] for x in b), min(x[3] for x in b), b[-1][4]
    ent = next((x for x in b if mins(x[0]) >= ENTRY_MIN), None)
    if not ent:
        prev = (dh, dl, dc); continue
    s0 = ent[4]
    win = [x for x in b if ENTRY_MIN <= mins(x[0]) <= EOD_MIN]
    if len(win) < 20 or not s0:
        prev = (dh, dl, dc); continue

    # --- first band break ---
    brk = None
    for x in win:
        if (x[2] - s0) / s0 >= BAND:
            brk = (x, +1, s0 * (1 + BAND)); break
        if (s0 - x[3]) / s0 >= BAND:
            brk = (x, -1, s0 * (1 - BAND)); break
    if not brk:
        prev = (dh, dl, dc); continue
    bar, side, lvl = brk
    t_brk = mins(bar[0])
    after = [x for x in win if mins(x[0]) > t_brk]
    if len(after) < 3:
        prev = (dh, dl, dc); continue

    end = after[-1][4]
    hi = max(x[2] for x in after)
    lo = min(x[3] for x in after)

    # continuation = further move in the break direction, from the break LEVEL
    cont = (hi - lvl) / lvl if side > 0 else (lvl - lo) / lvl
    # revert = how far it comes BACK against the break, from the break level
    rev = (lvl - lo) / lvl if side > 0 else (hi - lvl) / lvl
    closed_beyond = (end > lvl) if side > 0 else (end < lvl)
    back_to_entry = (lo <= s0) if side > 0 else (hi >= s0)
    opp = s0 * (1 - BAND) if side > 0 else s0 * (1 + BAND)
    whip = (lo <= opp) if side > 0 else (hi >= opp)     # broke the OPPOSITE band too

    if prev:
        ph, pl, pc = prev
        piv = (ph + pl + pc) / 3.0
        bc = (ph + pl) / 2.0
        tc = 2 * piv - bc
        cpr_w = abs(tc - bc) / piv
    else:
        cpr_w = np.nan
    orb = [x for x in b if mins(x[0]) < ENTRY_MIN]
    or_rng = (max(x[2] for x in orb) - min(x[3] for x in orb)) / s0 if orb else np.nan

    recs.append(dict(day=day, year=day[:4], side=side, t_brk=t_brk, cont=cont, rev=rev,
                     closed_beyond=closed_beyond, back_to_entry=back_to_entry, whip=whip,
                     vix=vix.get(day, np.nan), or_rng=or_rng, cpr_w=cpr_w))
    prev = (dh, dl, dc)

n = len(recs)
C = np.array([r["cont"] for r in recs])
R = np.array([r["rev"] for r in recs])
print(f"=== {n} days where the 0.4% band was HIT (2015-02 -> 2026-03) ===\n")
print(f"  closed BEYOND the break level : {100*np.mean([r['closed_beyond'] for r in recs]):.1f}%   <- continuation")
print(f"  came back to the ENTRY spot   : {100*np.mean([r['back_to_entry'] for r in recs]):.1f}%   <- gave it all back")
print(f"  ALSO broke the OPPOSITE band  : {100*np.mean([r['whip'] for r in recs]):.1f}%   <- full whipsaw")
print()
print(f"  median continuation beyond the break : {100*np.median(C):.2f}%   (mean {100*C.mean():.2f}%)")
print(f"  median REVERSAL back through it      : {100*np.median(R):.2f}%   (mean {100*R.mean():.2f}%)")
print(f"\n  -> a short on the far side is safe only if REVERSAL stays small.")

print("\n--- PER YEAR (does continuation hold up?) ---")
print(f"{'year':>5s} {'n':>5s} {'closed beyond':>14s} {'back to entry':>14s} {'full whipsaw':>13s} {'med cont':>9s} {'med rev':>8s}")
for y in sorted({r["year"] for r in recs}):
    s = [r for r in recs if r["year"] == y]
    print(f"{y:>5s} {len(s):>5d} {100*np.mean([x['closed_beyond'] for x in s]):>13.0f}% "
          f"{100*np.mean([x['back_to_entry'] for x in s]):>13.0f}% "
          f"{100*np.mean([x['whip'] for x in s]):>12.0f}% "
          f"{100*np.median([x['cont'] for x in s]):>8.2f}% {100*np.median([x['rev'] for x in s]):>7.2f}%")

def bucket(key, edges, lab):
    print(f"\n--- by {lab} ---")
    print(f"  {'bucket':<14s} {'n':>5s} {'closed beyond':>14s} {'back to entry':>14s} {'whipsaw':>9s} {'med cont':>9s} {'med rev':>8s}")
    v = np.array([r[key] for r in recs], float)
    ok = ~np.isnan(v)
    for i in range(len(edges) + 1):
        lo = -np.inf if i == 0 else edges[i - 1]
        hi = np.inf if i == len(edges) else edges[i]
        m = ok & (v >= lo) & (v < hi)
        if m.sum() < 30:
            continue
        sub = [r for r, k in zip(recs, m) if k]
        name = (f"< {edges[0]:g}" if i == 0 else f">= {edges[-1]:g}" if i == len(edges) else f"{lo:g}-{hi:g}")
        print(f"  {name:<14s} {m.sum():>5d} {100*np.mean([x['closed_beyond'] for x in sub]):>13.0f}% "
              f"{100*np.mean([x['back_to_entry'] for x in sub]):>13.0f}% "
              f"{100*np.mean([x['whip'] for x in sub]):>8.0f}% "
              f"{100*np.median([x['cont'] for x in sub]):>8.2f}% {100*np.median([x['rev'] for x in sub]):>7.2f}%")

bucket("t_brk", [600, 660, 720], "time of the break (minutes: <10:00, 10:00-11:00, 11:00-12:00, >12:00)")
bucket("vix", [12, 14, 17], "India VIX")
bucket("or_rng", [0.002, 0.004], "opening range")
bucket("cpr_w", [0.001, 0.002], "CPR width")
print("\nDONE")
