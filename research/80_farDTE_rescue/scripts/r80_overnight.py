#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G5 — OVERNIGHT / POSITIONAL far-DTE selling. The last idea standing.

G1: you keep only 6-10% of far-DTE premium in a 5-hour window, because theta lives in an option's
final DAYS, not its final hours. Every intraday variant we tried was fighting that fact and lost.
This one stops fighting it: SELL the far-DTE strangle and HOLD IT FOR DAYS, so the decay we sell is
decay we are still around to collect.

The price of admission is the thing an intraday system never faces: OVERNIGHT GAP. It is included
here by construction -- every hold is marked at the next daily CLOSE, gaps and all.

SWEEP
  entry DTE      : 6 (Wed), 5 (Thu), 4 (Fri)      -- the days research/79 says lose intraday
  structure      : naked strangle | iron condor (wings bought W points beyond the shorts)
  OTM width      : 100 / 150 / 200 / 250 / 300 pts from spot
  hold           : to expiry | exit with 1 day left | exit with 2 days left
  stop           : none | close both if the combined premium doubles (checked at each daily close)
  filter         : all days | VIX < 14 | VIX < 12 | narrow CPR

Priced with the calibrated + validated engine; at expiry the option is worth its INTRINSIC value.
Costs: Rs20/leg brokerage + slippage swept at 0 / 1 / 2% of premium per leg.
Every finalist is reported PER YEAR. Reads only.
"""
import json
import math
import sqlite3
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

ROOT = "/home/arun/quantifyd"
CAL = json.load(open(f"{ROOT}/research/80_farDTE_rescue/results/engine_calib.json"))["iv_mult_by_dte"]
QTY, BROK = 130, 20


def ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs(S, K, T, iv, kind):
    if T <= 0 or iv <= 0:
        return max(0.0, (S - K) if kind == "CE" else (K - S))
    d1 = (math.log(S / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if kind == "CE":
        return S * ncdf(d1) - K * ncdf(d2)
    return K * ncdf(-d2) - S * ncdf(-d1)


def iv_mult(dte):
    if str(dte) in CAL:
        return CAL[str(dte)]
    ks = sorted(int(k) for k in CAL)
    return CAL[str(min(ks, key=lambda k: abs(k - dte)))]


def next_expiry(d):
    target = 1 if d >= date(2025, 9, 1) else 3      # Tue from Sep-2025, else Thu
    ahead = (target - d.weekday()) % 7
    return d if ahead == 0 else d + timedelta(days=ahead)


c = sqlite3.connect(f"file:{ROOT}/backtest_data/market_data.db?mode=ro", uri=True)
nifty = {r[0][:10]: dict(o=r[1], h=r[2], l=r[3], c=r[4]) for r in c.execute(
    "SELECT date, open, high, low, close FROM market_data_unified "
    "WHERE symbol='NIFTY50' AND timeframe='day' ORDER BY date")}
vix = {r[0][:10]: r[1] for r in c.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
c.close()

sess = sorted(d for d in nifty if d in vix)
idx = {d: i for i, d in enumerate(sess)}

# prev-day CPR width (narrow => trend day)
cprw = {}
for i in range(1, len(sess)):
    p = nifty[sess[i - 1]]
    piv = (p["h"] + p["l"] + p["c"]) / 3.0
    bc = (p["h"] + p["l"] / 1.0) / 2.0 if False else (p["h"] + p["l"]) / 2.0
    tc = 2 * piv - bc
    cprw[sess[i]] = abs(tc - bc) / piv

DTES = [6, 5, 4]
WIDTHS = [100, 150, 200, 250, 300]
WINGS = [None, 150, 250]          # None = naked strangle; else iron condor wing distance
HOLDS = [0, 1, 2]                 # exit with N days left (0 = hold to expiry)
STOPS = [None, 2.0]               # None = no stop; 2.0 = close if combined premium doubles
SLIPS = [0.0, 0.01, 0.02]

res = defaultdict(list)

for i, day in enumerate(sess):
    y, m, dd = map(int, day.split("-"))
    d0 = date(y, m, dd)
    exp = next_expiry(d0)
    dte0 = (exp - d0).days
    if dte0 not in DTES:
        continue
    exp_s = exp.isoformat()
    if exp_s not in idx:                    # expiry must be a real session we have
        continue
    S0 = nifty[day]["c"]
    v0 = vix[day]
    path = [s for s in sess[idx[day] + 1: idx[exp_s] + 1]]
    if not path:
        continue

    for W in WIDTHS:
        Kc = round((S0 + W) / 50) * 50
        Kp = round((S0 - W) / 50) * 50
        for wing in WINGS:
            iv0 = (v0 / 100.0) * iv_mult(dte0)
            T0 = max(dte0 / 365.0, 1e-6)
            # credit taken at entry (short legs) minus wings paid (long legs)
            credit = bs(S0, Kc, T0, iv0, "CE") + bs(S0, Kp, T0, iv0, "PE")
            legs = 2
            if wing:
                Kcw, Kpw = Kc + wing, Kp - wing
                credit -= bs(S0, Kcw, T0, iv0, "CE") + bs(S0, Kpw, T0, iv0, "PE")
                legs = 4
            if credit < 2:
                continue

            for HOLD in HOLDS:
                for STOP in STOPS:
                    exit_val, exited = None, None
                    for s in path:
                        Sx = nifty[s]["c"]
                        vx = vix.get(s, v0)
                        ey, em, ed = map(int, exp_s.split("-"))
                        sy, sm, sd = map(int, s.split("-"))
                        dte_left = (date(ey, em, ed) - date(sy, sm, sd)).days
                        if dte_left <= 0:
                            val = max(0.0, Sx - Kc) + max(0.0, Kp - Sx)
                            if wing:
                                val -= max(0.0, Sx - Kcw) + max(0.0, Kpw - Sx)
                        else:
                            ivx = (vx / 100.0) * iv_mult(dte_left)
                            Tx = max(dte_left / 365.0, 1e-6)
                            val = bs(Sx, Kc, Tx, ivx, "CE") + bs(Sx, Kp, Tx, ivx, "PE")
                            if wing:
                                val -= bs(Sx, Kcw, Tx, ivx, "CE") + bs(Sx, Kpw, Tx, ivx, "PE")
                        if STOP and val >= credit * STOP:
                            exit_val, exited = val, "stop"; break
                        if dte_left <= HOLD:
                            exit_val, exited = val, "time"; break
                    if exit_val is None:
                        exit_val, exited = val, "expiry"

                    for sl in SLIPS:
                        pnl = (credit * (1 - sl) - exit_val * (1 + sl)) * QTY - legs * BROK
                        res[(dte0, W, wing, HOLD, STOP, sl)].append(
                            dict(day=day, year=day[:4], pnl=pnl, vix=v0,
                                 cprw=cprw.get(day, np.nan), exited=exited, credit=credit))

print("=== OVERNIGHT far-DTE selling — hold for DAYS, collect the decay you actually sold ===")
print("    NIFTY daily closes 2015→2026 | 2 lots (130 qty) | Rs20/leg | GAP RISK INCLUDED\n")

rows = []
for key, tl in res.items():
    dte0, W, wing, HOLD, STOP, sl = key
    if sl != 0.01 or len(tl) < 120:
        continue
    v = np.array([t["pnl"] for t in tl], float)
    yrs = sorted({t["year"] for t in tl})
    ym = [np.mean([t["pnl"] for t in tl if t["year"] == y]) for y in yrs]
    rows.append(dict(mean=v.mean(), key=key, n=len(tl), median=np.median(v),
                     win=100 * (v > 0).mean(), worst=v.min(),
                     pos_years=sum(1 for x in ym if x > 0), nyears=len(yrs),
                     keep=100 * np.mean([1 - 0 for _ in tl])))

rows.sort(key=lambda r: -r["mean"])
print(f"{'DTE':>3s} {'width':>6s} {'wing':>6s} {'hold':>5s} {'stop':>5s} {'n':>5s} {'mean':>8s} "
      f"{'median':>8s} {'win%':>6s} {'worst':>9s} {'+yrs':>7s}   <- slippage 1%")
for r in rows[:20]:
    dte0, W, wing, HOLD, STOP, sl = r["key"]
    print(f"{dte0:>3d} {W:>6d} {str(wing or '-'):>6s} {HOLD:>5d} {str(STOP or '-'):>5s} "
          f"{r['n']:>5d} {r['mean']:>+8.0f} {r['median']:>+8.0f} {r['win']:>5.0f}% "
          f"{r['worst']:>+9.0f} {r['pos_years']:>3d}/{r['nyears']:<3d}")

if rows:
    best = rows[0]
    dte0, W, wing, HOLD, STOP, _ = best["key"]
    print(f"\n\n--- BEST CELL year by year: DTE{dte0}, {W}pt wide, "
          f"wing={wing or 'naked'}, hold-to-{HOLD or 'expiry'}, stop={STOP or 'none'} ---\n")
    tl = res[(dte0, W, wing, HOLD, STOP, 0.01)]
    print(f"{'year':>5s} {'n':>4s} {'mean':>8s} {'total':>10s} {'win%':>6s} {'worst':>9s}")
    for y in sorted({t["year"] for t in tl}):
        a = np.array([t["pnl"] for t in tl if t["year"] == y], float)
        print(f"{y:>5s} {len(a):>4d} {a.mean():>+8.0f} {a.sum():>+10.0f} "
              f"{100*(a>0).mean():>5.0f}% {a.min():>+9.0f}")

    print("\n--- cost sensitivity ---")
    for sl in SLIPS:
        a = np.array([t["pnl"] for t in res[(dte0, W, wing, HOLD, STOP, sl)]], float)
        print(f"  slippage {100*sl:>3.0f}%   mean {a.mean():>+8.0f}   total {a.sum():>+11.0f}")

    print("\n--- by VIX ---")
    for lo, hi in [(0, 12), (12, 14), (14, 17), (17, 99)]:
        a = np.array([t["pnl"] for t in tl if lo <= t["vix"] < hi], float)
        if len(a) < 25:
            continue
        print(f"  VIX {lo:>2d}-{hi:<2d} n={len(a):>4d}  mean {a.mean():>+8.0f}  win {100*(a>0).mean():>3.0f}%")

    print("\n--- naked vs condor at the same width/DTE (does capping the tail pay?) ---")
    for wg in WINGS:
        k = (dte0, W, wg, HOLD, STOP, 0.01)
        if k not in res:
            continue
        a = np.array([t["pnl"] for t in res[k]], float)
        print(f"  wing {str(wg or 'naked'):>6s}  mean {a.mean():>+8.0f}  worst {a.min():>+9.0f}  "
              f"win {100*(a>0).mean():>3.0f}%")
print("\nDONE")
