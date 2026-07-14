#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G4 — DIRECTIONAL SHORT after the band breaks (the user's idea), on 11 years.

THESIS (earned, not assumed):
  G3 showed that after the 0.4% band is hit, direction is a COIN FLIP (53.8% close beyond) -- so
  BUYING direction is dead. But a SHORT on the far side does not need continuation; it needs the
  move NOT TO COME BACK. And that IS predictable: after a break past 12:00 the price returns to
  the entry spot only 18% of the time and whipsaws to the opposite band just 3% (vs 61% / 31% for
  a pre-10:00 break). Monotone across 4 buckets, 2,133 days, stable per year.

  It also side-steps the curse that killed the far-DTE strangle: a directional short is paid by
  DELTA (the option going out-of-the-money as price walks away), not by theta -- so the fact that
  a 6-DTE option has almost no intraday theta does not matter.

THE SYSTEM:
  wait for |spot - spot_0920| >= BREAK%. On the break, SELL the option on the FAR side,
  OTM_PTS beyond the current spot. Hold to 15:15. Optional stop if the premium doubles.

SWEEPS: break threshold x OTM distance x earliest-break-time filter x VIX filter x DTE.
Every cell reported PER YEAR -- an 11-year mean that comes from 2 good years is not an edge.

Prices from the calibrated+validated engine (results/engine_calib.json).
Costs: Rs20/leg brokerage + SLIPPAGE swept at 0 / 1 / 2% of premium.
Reads only.
"""
import json
import math
import sqlite3
from collections import defaultdict
from datetime import date

import numpy as np

ROOT = "/home/arun/quantifyd"
CAL = json.load(open(f"{ROOT}/research/80_farDTE_rescue/results/engine_calib.json"))["iv_mult_by_dte"]
QTY = 130          # 2 lots
BROK = 20          # single leg, round trip

ENTRY_MIN, EOD_MIN = 9 * 60 + 20, 15 * 60 + 15


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs(S, K, T, iv, kind):
    if T <= 0 or iv <= 0:
        return max(0.0, (S - K) if kind == "CE" else (K - S))
    d1 = (math.log(S / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if kind == "CE":
        return S * norm_cdf(d1) - K * norm_cdf(d2)
    return K * norm_cdf(-d2) - S * norm_cdf(-d1)


def iv_mult(dte):
    if str(dte) in CAL:
        return CAL[str(dte)]
    ks = sorted(int(k) for k in CAL)
    return CAL[str(min(ks, key=lambda k: abs(k - dte)))]


# NIFTY weekly expiry: THURSDAY historically, TUESDAY from 2025-09 (chain confirms Tue in 2026).
def next_expiry(d):
    target = 1 if d >= date(2025, 9, 1) else 3          # Tue=1, Thu=3
    ahead = (target - d.weekday()) % 7
    return d if ahead == 0 else d + __import__("datetime").timedelta(days=ahead)


c = sqlite3.connect(f"file:{ROOT}/backtest_data/market_data.db?mode=ro", uri=True)
bars = c.execute("SELECT date, open, high, low, close FROM market_data_unified "
                 "WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date").fetchall()
vix_d = {r[0][:10]: r[1] for r in c.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
c.close()

days = defaultdict(list)
for d, o, h, l, cl in bars:
    days[str(d)[:10]].append((str(d)[11:16], o, h, l, cl))

def mins(x):
    return int(x[:2]) * 60 + int(x[3:5])

BREAKS = [0.003, 0.004, 0.005]
OTMS = [50, 100, 150, 200]
TMINS = [0, 600, 660, 720]           # earliest break time allowed
SLIPS = [0.0, 0.01, 0.02]

trades = defaultdict(list)
for day in sorted(days):
    b = sorted(days[day])
    if len(b) < 40:
        continue
    v = vix_d.get(day)
    if not v:
        continue
    ent = next((x for x in b if mins(x[0]) >= ENTRY_MIN), None)
    if not ent:
        continue
    s0 = ent[4]
    win = [x for x in b if ENTRY_MIN <= mins(x[0]) <= EOD_MIN]
    if len(win) < 20 or not s0:
        continue
    y, m, dd = map(int, day.split("-"))
    dd_ = date(y, m, dd)
    dte = (next_expiry(dd_) - dd_).days
    iv = (v / 100.0) * iv_mult(dte)
    T0 = max((dte + 6.0 / 24.0) / 365.0, 1e-6)
    close = win[-1][4]

    for BR in BREAKS:
        brk = None
        for x in win:
            if (x[2] - s0) / s0 >= BR:
                brk = (x, +1); break
            if (s0 - x[3]) / s0 >= BR:
                brk = (x, -1); break
        if not brk:
            continue
        bar, side = brk
        tb = mins(bar[0])
        spot_b = bar[4]
        hrs_left = max((EOD_MIN - tb) / 60.0, 0.1)
        T_b = max((dte + hrs_left / 24.0) / 365.0, 1e-6)
        for OT in OTMS:
            # SELL the option on the FAR side: break UP -> sell a PUT below; break DOWN -> sell a CALL above
            kind = "PE" if side > 0 else "CE"
            K = round((spot_b - OT if side > 0 else spot_b + OT) / 50) * 50
            entry_px = bs(spot_b, K, T_b, iv, kind)
            if entry_px < 1.0:
                continue
            exit_px = bs(close, K, T0 - (dte + 0.0) / 365.0 + 1e-6, iv, kind) if False else \
                      bs(close, K, max((dte + 0.25 / 24.0) / 365.0, 1e-6), iv, kind)
            for TM in TMINS:
                if tb < TM:
                    continue
                for sl in SLIPS:
                    pnl = (entry_px * (1 - sl) - exit_px * (1 + sl)) * QTY - 2 * BROK
                    trades[(BR, OT, TM, sl)].append(
                        dict(day=day, year=day[:4], dte=dte, vix=v, pnl=pnl, side=side, tb=tb))

print("=== DIRECTIONAL SHORT after the band breaks — sell the FAR side, hold to 15:15 ===")
print("    (2015-02 → 2026-03, 2 lots = 130 qty, Rs20/leg brokerage)\n")

rows = []
for (BR, OT, TM, sl), tl in trades.items():
    if sl != 0.01 or len(tl) < 100:
        continue
    v = np.array([t["pnl"] for t in tl], float)
    yrs = sorted({t["year"] for t in tl})
    ym = [np.mean([t["pnl"] for t in tl if t["year"] == y]) for y in yrs]
    pos_years = sum(1 for x in ym if x > 0)
    rows.append((v.mean(), BR, OT, TM, len(tl), v.mean(), np.median(v), 100 * (v > 0).mean(),
                 v.min(), pos_years, len(yrs)))

rows.sort(reverse=True)
print(f"{'break':>6s} {'OTM':>5s} {'after':>6s} {'n':>5s} {'mean':>8s} {'median':>8s} {'win%':>6s} "
      f"{'worst':>9s} {'+years':>8s}   <- slippage 1%")
for _, BR, OT, TM, n, mn, md, wr, wo, py, ny in rows[:18]:
    tlab = "any" if TM == 0 else f">{TM//60}:{TM%60:02d}"
    print(f"{100*BR:>5.1f}% {OT:>5d} {tlab:>6s} {n:>5d} {mn:>+8.0f} {md:>+8.0f} {wr:>5.0f}% "
          f"{wo:>+9.0f} {py:>3d}/{ny:<4d}")

print("\n\n--- the best cell, YEAR BY YEAR (an 11-year mean from 2 good years is not an edge) ---")
best = rows[0]
_, BR, OT, TM = best[1], best[2], best[3], best[3]
key = (best[1], best[2], best[3], 0.01)
tl = trades[key]
print(f"break {100*best[1]:.1f}% | sell {best[2]}pt OTM on the far side | break after "
      f"{'any' if best[3]==0 else str(best[3]//60)+':'+format(best[3]%60,'02d')}\n")
print(f"{'year':>5s} {'n':>4s} {'mean':>8s} {'total':>9s} {'win%':>6s} {'worst':>9s}")
for y in sorted({t["year"] for t in tl}):
    s = [t["pnl"] for t in tl if t["year"] == y]
    a = np.array(s, float)
    print(f"{y:>5s} {len(s):>4d} {a.mean():>+8.0f} {a.sum():>+9.0f} {100*(a>0).mean():>5.0f}% {a.min():>+9.0f}")

print("\n--- cost sensitivity on that cell ---")
for sl in SLIPS:
    t2 = trades[(best[1], best[2], best[3], sl)]
    a = np.array([t["pnl"] for t in t2], float)
    print(f"  slippage {100*sl:>3.0f}%  mean {a.mean():>+8.0f}  total {a.sum():>+10.0f}")

print("\n--- that cell by DTE (does it work on the far-DTE days we set out to rescue?) ---")
for d in sorted({t["dte"] for t in tl}):
    s = [t["pnl"] for t in tl if t["dte"] == d]
    if len(s) < 30:
        continue
    a = np.array(s, float)
    print(f"  DTE{d:<2d} n={len(s):>4d}  mean {a.mean():>+8.0f}  win {100*(a>0).mean():>3.0f}%")

print("\n--- that cell by VIX ---")
for lo, hi in [(0, 12), (12, 14), (14, 17), (17, 99)]:
    s = [t["pnl"] for t in tl if lo <= t["vix"] < hi]
    if len(s) < 30:
        continue
    a = np.array(s, float)
    print(f"  VIX {lo:>2d}-{hi:<2d} n={len(s):>4d}  mean {a.mean():>+8.0f}  win {100*(a>0).mean():>3.0f}%")
print("\nDONE")
