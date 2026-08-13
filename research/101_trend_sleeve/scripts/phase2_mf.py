# -*- coding: utf-8 -*-
"""Diversified managed-futures search: weekly Donchian-8 long-only (the gold-validated config) across
all downloaded assets; build an equal-weight sleeve of the strong long-history legs; correlation matrix."""
import sqlite3, math, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
SLIP = 0.001
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=20000")
def weekly(sym):
    b = c.execute("SELECT date,high,low,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0 ORDER BY date", (sym,)).fetchall()
    wk = {}
    for d, h, l, cl in b:
        y, w, _ = dt.date.fromisoformat(d).isocalendar(); k = (y, w)
        if k not in wk: wk[k] = [d, h, l, cl]
        else: wk[k][1] = max(wk[k][1], h); wk[k][2] = min(wk[k][2], l); wk[k][3] = cl
    return [wk[k] for k in sorted(wk)]
def donch(b, N=8):
    out = []; pos = 0
    for i in range(N, len(b) - 1):
        hh = max(x[1] for x in b[i-N:i]); ll = min(x[2] for x in b[i-N:i]); cl = b[i][3]
        if cl > hh: newpos = 1
        elif cl < ll: newpos = 0
        else: newpos = pos
        out.append((b[i+1][0], pos * (b[i+1][3] / b[i][3] - 1) - abs(newpos - pos) * SLIP)); pos = newpos
    return out
def stat(rows):
    r = [x[1] for x in rows]; n = len(r); mu = sum(r) / n; sd = (sum((x-mu)**2 for x in r)/n)**0.5
    eq = 1.0; pk = 1.0; mdd = 0
    for x in r: eq *= (1+x); pk = max(pk, eq); mdd = min(mdd, eq/pk - 1)
    yrs = n/52
    return mu/sd*math.sqrt(52) if sd else 0, eq**(1/yrs)-1, mdd, yrs
assets = ["GOLDBEES","MON100","MOM100","HNGSNGBEES","MAFANG","MASPTOP50","MAHKTECH","ITBEES","PSUBNKBEES","CPSEETF","NIFTYBEES","SILVERBEES"]
print("weekly Donchian-8 long-only, net of 0.1%/side:\n")
print(f"{'asset':>12}  {'Sharpe':>7} {'CAGR':>7} {'maxDD':>7} {'years':>6}")
series = {}
for s in assets:
    rows = donch(weekly(s)); series[s] = dict(rows)
    sh, cagr, mdd, yrs = stat(rows)
    print(f"{s:>12}  {sh:>+7.2f} {cagr*100:>6.1f}% {mdd*100:>6.0f}% {yrs:>6.1f}")
# sleeve of strong long-history legs
legs = ["GOLDBEES", "MON100", "HNGSNGBEES"]
allw = sorted(set().union(*[set(series[s]) for s in legs]))
sleeve = []
for w in allw:
    vals = [series[s][w] for s in legs if w in series[s]]
    if vals: sleeve.append((w, sum(vals)/len(vals)))
sh, cagr, mdd, yrs = stat(sleeve)
print(f"\nSLEEVE (equal-wt {'+'.join(legs)}): Sharpe {sh:+.2f} CAGR {cagr*100:.1f}% maxDD {mdd*100:.0f}% ({yrs:.1f}y)")
# pairwise correlation of weekly leg returns (long-history legs)
print("\ncorrelation (weekly trend returns):")
common = sorted(set(series[legs[0]]) & set(series[legs[1]]) & set(series[legs[2]]))
def corr(a, b):
    xs = [series[a][w] for w in common]; ys = [series[b][w] for w in common]; n = len(xs)
    mx = sum(xs)/n; my = sum(ys)/n
    cov = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/n
    sx = (sum((x-mx)**2 for x in xs)/n)**0.5; sy = (sum((y-my)**2 for y in ys)/n)**0.5
    return cov/(sx*sy) if sx and sy else 0
for i in range(len(legs)):
    for j in range(i+1, len(legs)):
        print(f"  {legs[i]} vs {legs[j]}: {corr(legs[i], legs[j]):+.2f}")
c.close()
