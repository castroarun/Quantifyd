# -*- coding: utf-8 -*-
"""Optimal managed-futures TREND sleeve: weekly Donchian-8 long-only on the strong, low-correlation
legs, equal-weighted. Core (11y): GOLD + Nasdaq(MOM100) + CPSEETF. Full (recent): + S&P + FANG + silver.
Net of 0.1%/side. Reports sleeve Sharpe/CAGR/Calmar/maxDD, per-year, corr matrix, vs gold-alone, OOS."""
import sqlite3, math, datetime as dt, json
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
SLIP = 0.001
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=20000")
def weekly(sym, frm="2015-01-01"):
    b = c.execute("SELECT date,high,low,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0 AND date>=? ORDER BY date", (sym, frm)).fetchall()
    wk = {}
    for d, h, l, cl in b:
        y, w, _ = dt.date.fromisoformat(d).isocalendar(); k = (y, w)
        if k not in wk: wk[k] = [d, h, l, cl]
        else: wk[k][1] = max(wk[k][1], h); wk[k][2] = min(wk[k][2], l); wk[k][3] = cl
    return [wk[k] for k in sorted(wk)]
def donch(sym, N=8):
    b = weekly(sym); out = {}; pos = 0
    for i in range(N, len(b) - 1):
        hh = max(x[1] for x in b[i-N:i]); ll = min(x[2] for x in b[i-N:i]); cl = b[i][3]
        newpos = 1 if cl > hh else (0 if cl < ll else pos)
        out[b[i+1][0]] = pos * (b[i+1][3]/b[i][3]-1) - abs(newpos-pos)*SLIP; pos = newpos
    return out
def stat(series):  # series: sorted list of (date, ret)
    r = [x[1] for x in series]; n = len(r); mu = sum(r)/n; sd = (sum((x-mu)**2 for x in r)/n)**0.5
    eq = 1.0; pk = 1.0; mdd = 0
    for x in r: eq *= (1+x); pk = max(pk, eq); mdd = min(mdd, eq/pk-1)
    yrs = n/52
    return dict(sharpe=mu/sd*math.sqrt(52) if sd else 0, cagr=eq**(1/yrs)-1, mdd=mdd, calmar=(eq**(1/yrs)-1)/abs(mdd) if mdd else 0, yrs=yrs, n=n)
def sleeve(legs):
    S = {s: donch(s) for s in legs}
    allw = sorted(set().union(*[set(S[s]) for s in legs]))
    out = []
    for w in allw:
        vals = [S[s][w] for s in legs if w in S[s]]
        if vals: out.append((w, sum(vals)/len(vals)))   # equal-weight available legs
    return out, S
core = ["GOLDBEES", "MOM100", "CPSEETF"]
full = ["GOLDBEES", "MOM100", "CPSEETF", "MASPTOP50", "MAFANG", "SILVERBEES"]
gold = sorted(donch("GOLDBEES").items())
sc, Sc = sleeve(core); sf, Sf = sleeve(full)
print("== MANAGED-FUTURES TREND SLEEVE (weekly Donchian-8 long-only, net) ==")
for lbl, s in [("GOLD alone", stat(gold)), ("CORE sleeve (Gold+Nasdaq+CPSE, 11y)", stat(sc)), ("FULL sleeve (+S&P+FANG+silver)", stat(sf))]:
    print(f"  {lbl:>36}: Sharpe {s['sharpe']:+.2f} CAGR {s['cagr']*100:5.1f}% Calmar {s['calmar']:.2f} maxDD {s['mdd']*100:.0f}% ({s['yrs']:.1f}y)")
# per-year core sleeve
yr = defaultdict(float)
for d, r in sc: yr[d[:4]] += r
print("\n  CORE sleeve per-year:", " ".join(f"{y}:{yr[y]*100:+.0f}%" for y in sorted(yr)))
# correlation matrix (core legs)
common = sorted(set.intersection(*[set(Sc[s]) for s in core]))
def corr(a, b):
    xs = [Sc[a][w] for w in common]; ys = [Sc[b][w] for w in common]; n = len(xs)
    mx = sum(xs)/n; my = sum(ys)/n; cov = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/n
    sx = (sum((x-mx)**2 for x in xs)/n)**0.5; sy = (sum((y-my)**2 for y in ys)/n)**0.5
    return cov/(sx*sy) if sx and sy else 0
print("  core-leg correlations:", ", ".join(f"{core[i][:4]}/{core[j][:4]}:{corr(core[i],core[j]):+.2f}" for i in range(3) for j in range(i+1,3)))
# walk-forward on core sleeve (train<2021H2 vs OOS)
oos = [x for x in sc if x[0] >= "2021-07-01"]
print(f"  CORE OOS (2021H2-26): Sharpe {stat(oos)['sharpe']:+.2f} CAGR {stat(oos)['cagr']*100:.1f}% maxDD {stat(oos)['mdd']*100:.0f}%")
json.dump([[d, round(r, 5)] for d, r in sc], open("/tmp/mf_sleeve_core.json", "w"))
c.close()
