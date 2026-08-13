# -*- coding: utf-8 -*-
"""PHASE 2 · G3 blend — does the (weak) dual-ST trend sleeve LIFT the naked-straddle book's Calmar?
Combine monthly P&L: straddle (naked DTE-3, ₹) + w lots of the best dual-ST futures (₹75/pt). Sweep w."""
import json, datetime as dt
from collections import defaultdict
LOT = 75
trend_d = json.load(open("/tmp/dualst_best.json"))
tmon = defaultdict(float)
for d, p in trend_d: tmon[d[:7]] += p * LOT          # ₹ per 1 lot per month
strad = defaultdict(float)
for t in json.load(open("/tmp/realistic_trades.json"))["dte3"]: strad[t["exit"][:7]] += t["naked"]
months = sorted(set(tmon) & set(strad))
def calmar(series):
    p = [series[m] for m in months]; n = len(p); tot = sum(p)
    cum = pk = mdd = 0
    for x in p: cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
    yrs = n / 12
    return tot, (tot / yrs) / abs(mdd) if mdd else 0, mdd
# straddle alone
st_tot, st_cal, st_mdd = calmar({m: strad[m] for m in months})
# correlation
xs = [tmon[m] for m in months]; ys = [strad[m] for m in months]; n = len(xs)
mx = sum(xs) / n; my = sum(ys) / n
cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5; sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
corr = cov / (sx * sy) if sx and sy else 0
print(f"Monthly overlap n={n} | corr(trend, straddle) = {corr:+.2f}")
print(f"STRADDLE alone: total ₹{st_tot:+,.0f}  Calmar {st_cal:.2f}  maxDD ₹{st_mdd:+,.0f}\n")
print(f"{'trend lots':>10} {'combined ₹':>13} {'Calmar':>7} {'maxDD ₹':>13}")
best = (st_cal, 0)
for w in [0, 1, 2, 3, 4, 5, 7, 10]:
    comb = {m: strad[m] + w * tmon[m] for m in months}
    tot, cal, mdd = calmar(comb)
    print(f"{w:>10} {tot:>+13,.0f} {cal:>7.2f} {mdd:>+13,.0f}")
    if cal > best[0]: best = (cal, w)
print(f"\nBEST blend: {best[1]} trend lots per straddle book → Calmar {best[0]:.2f} (vs {st_cal:.2f} straddle-alone)")
