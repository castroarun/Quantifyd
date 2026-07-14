#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G5b — the RISK side of overnight far-DTE selling.

The sweep says +3,549/trade, 71% win, 11/12 positive years. That is the variance risk premium, and
it is real -- but a mean without a drawdown is a sales pitch. Selling naked weekly strangles is the
textbook "pennies in front of a steamroller": the worst single trade is -88,330, twenty-five times
the mean, and 2020 was a losing year.

So: equity curve, max drawdown, Calmar, tail stats -- for the NAKED winner and for the CONDOR that
caps the tail. Plus the margin reality: what is the return ON the capital this actually ties up?

Reads only.
"""
import json
import math
import sqlite3
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

ROOT = "/home/arun/quantifyd"
CAL = json.load(open(f"{ROOT}/research/80_farDTE_rescue/results/engine_calib.json"))["iv_mult_by_dte"]
QTY, BROK, SLIP = 130, 20, 0.01


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


def iv_mult(d):
    if str(d) in CAL:
        return CAL[str(d)]
    ks = sorted(int(k) for k in CAL)
    return CAL[str(min(ks, key=lambda k: abs(k - d)))]


def next_expiry(d):
    t = 1 if d >= date(2025, 9, 1) else 3
    a = (t - d.weekday()) % 7
    return d if a == 0 else d + timedelta(days=a)


c = sqlite3.connect(f"file:{ROOT}/backtest_data/market_data.db?mode=ro", uri=True)
nifty = {r[0][:10]: r[1] for r in c.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' ORDER BY date")}
vix = {r[0][:10]: r[1] for r in c.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
c.close()
sess = sorted(d for d in nifty if d in vix)
idx = {d: i for i, d in enumerate(sess)}


def run(width, wing, stop=2.0):
    tr = []
    for day in sess:
        y, m, dd = map(int, day.split("-"))
        d0 = date(y, m, dd)
        exp = next_expiry(d0)
        if (exp - d0).days != 6:
            continue
        es = exp.isoformat()
        if es not in idx:
            continue
        S0, v0 = nifty[day], vix[day]
        dte0 = 6
        iv0 = (v0 / 100.0) * iv_mult(dte0)
        T0 = dte0 / 365.0
        Kc, Kp = round((S0 + width) / 50) * 50, round((S0 - width) / 50) * 50
        cred = bs(S0, Kc, T0, iv0, "CE") + bs(S0, Kp, T0, iv0, "PE")
        legs = 2
        if wing:
            Kcw, Kpw = Kc + wing, Kp - wing
            cred -= bs(S0, Kcw, T0, iv0, "CE") + bs(S0, Kpw, T0, iv0, "PE")
            legs = 4
        if cred < 2:
            continue
        val = cred
        for s in sess[idx[day] + 1: idx[es] + 1]:
            Sx, vx = nifty[s], vix.get(s, v0)
            sy, sm, sd = map(int, s.split("-"))
            left = (exp - date(sy, sm, sd)).days
            if left <= 0:
                val = max(0.0, Sx - Kc) + max(0.0, Kp - Sx)
                if wing:
                    val -= max(0.0, Sx - Kcw) + max(0.0, Kpw - Sx)
                break
            ivx = (vx / 100.0) * iv_mult(left)
            Tx = left / 365.0
            val = bs(Sx, Kc, Tx, ivx, "CE") + bs(Sx, Kp, Tx, ivx, "PE")
            if wing:
                val -= bs(Sx, Kcw, Tx, ivx, "CE") + bs(Sx, Kpw, Tx, ivx, "PE")
            if stop and val >= cred * stop:
                break
        pnl = (cred * (1 - SLIP) - val * (1 + SLIP)) * QTY - legs * BROK
        # margin: naked short strangle ~ 12% of notional per lot (SPAN+exposure, conservative);
        # a condor is capped, so margin ~ the max loss + a buffer.
        if wing:
            margin = (wing * QTY) + 20000
        else:
            margin = 0.12 * S0 * QTY
        tr.append(dict(day=day, year=day[:4], pnl=pnl, margin=margin))
    return tr


def stats(tr, lab):
    v = np.array([t["pnl"] for t in tr], float)
    eq = np.cumsum(v)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    mdd = dd.min()
    yrs = len({t["year"] for t in tr})
    ann = v.sum() / yrs
    calmar = ann / abs(mdd) if mdd < 0 else float("inf")
    marg = np.mean([t["margin"] for t in tr])
    print(f"\n=== {lab} ===")
    print(f"  trades {len(v)}   mean {v.mean():+,.0f}   total {v.sum():+,.0f}   win {100*(v>0).mean():.0f}%")
    print(f"  worst trade   {v.min():+,.0f}   ({abs(v.min())/v.mean():.0f}x the mean)")
    print(f"  MAX DRAWDOWN  {mdd:+,.0f}")
    print(f"  annual P&L    {ann:+,.0f}   CALMAR {calmar:.2f}")
    print(f"  avg margin    {marg:,.0f}   -> return on margin {100*ann/marg:.0f}% p.a.")
    p1, p5 = np.percentile(v, 1), np.percentile(v, 5)
    print(f"  tail: p1 {p1:+,.0f}   p5 {p5:+,.0f}   "
          f"losses > 3x mean: {int((v < -3*v.mean()).sum())} ({100*(v < -3*v.mean()).mean():.1f}%)")
    return mdd, calmar


print("OVERNIGHT far-DTE (enter Wed, DTE6, hold to Tue expiry) — RISK-ADJUSTED")
print("2 lots = 130 qty | 1% slippage | stop = premium doubles (checked at daily close)\n")
a = run(100, None)
stats(a, "NAKED strangle, 100pt wide  (the sweep's winner)")
b = run(150, None)
stats(b, "NAKED strangle, 150pt wide")
c1 = run(100, 250)
stats(c1, "IRON CONDOR, 100pt short / 250pt wings")
c2 = run(150, 250)
stats(c2, "IRON CONDOR, 150pt short / 250pt wings")

print("\n\n--- the 10 worst trades of the NAKED winner (what actually goes wrong) ---")
w = sorted(a, key=lambda t: t["pnl"])[:10]
for t in w:
    print(f"  {t['day']}  {t['pnl']:+,.0f}")
print("\nDONE")
