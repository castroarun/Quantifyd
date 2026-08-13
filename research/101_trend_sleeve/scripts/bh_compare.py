# -*- coding: utf-8 -*-
"""Does the TREND rule add value, or is it buy-and-hold in disguise? Compare weekly Donchian-8
long-only vs BUY-AND-HOLD (always long), per leg and for the equal-weight basket, 2015-2026."""
import sqlite3, math, datetime as dt
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
SLIP = 0.001
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=20000")
def weekly(sym):
    b = c.execute("SELECT date,high,low,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0 AND date>='2015-01-01' ORDER BY date", (sym,)).fetchall()
    wk = {}
    for d, h, l, cl in b:
        y, w, _ = dt.date.fromisoformat(d).isocalendar(); k = (y, w)
        if k not in wk: wk[k] = [d, h, l, cl]
        else: wk[k][1] = max(wk[k][1], h); wk[k][2] = min(wk[k][2], l); wk[k][3] = cl
    return [wk[k] for k in sorted(wk)]
def series(sym, trend):
    b = weekly(sym); out = {}; pos = 0
    for i in range(8, len(b) - 1):
        if trend:
            hh = max(x[1] for x in b[i-8:i]); ll = min(x[2] for x in b[i-8:i]); cl = b[i][3]
            newpos = 1 if cl > hh else (0 if cl < ll else pos)
        else:
            newpos = 1                                   # buy-and-hold: always long
        out[b[i+1][0]] = pos * (b[i+1][3]/b[i][3]-1) - abs(newpos-pos)*SLIP; pos = newpos
    return out
def stat(rr):
    n = len(rr); mu = sum(rr)/n; sd = (sum((x-mu)**2 for x in rr)/n)**0.5
    eq = 1.0; pk = 1.0; mdd = 0; pos_wk = 0
    for x in rr: eq *= (1+x); pk = max(pk, eq); mdd = min(mdd, eq/pk-1)
    yrs = n/52
    return mu/sd*math.sqrt(52) if sd else 0, eq**(1/yrs)-1, mdd
legs = ["GOLDBEES", "MOM100", "CPSEETF"]
print(f"{'asset':>10}  {'BUY&HOLD Sharpe/CAGR/DD':>30}   {'TREND Sharpe/CAGR/DD':>26}  {'%time in mkt':>12}")
for s in legs:
    bh = series(s, False); tr = series(s, True)
    shb, cb, mb = stat(list(bh.values())); sht, ct, mt = stat(list(tr.values()))
    intime = 100 * sum(1 for v in tr.values() if v != 0) / len(tr)
    print(f"{s:>10}  {shb:+.2f} / {cb*100:5.1f}% / {mb*100:4.0f}%          {sht:+.2f} / {ct*100:5.1f}% / {mt*100:4.0f}%      {intime:5.0f}%")
# basket
def basket(trend):
    S = {s: series(s, trend) for s in legs}
    allw = sorted(set().union(*[set(S[s]) for s in legs]))
    return [sum(S[s][w] for s in legs if w in S[s]) / len([1 for s in legs if w in S[s]]) for w in allw]
shb, cb, mb = stat(basket(False)); sht, ct, mt = stat(basket(True))
print(f"\n{'BASKET (equal-wt)':>18}: buy&hold {shb:+.2f}/{cb*100:.1f}%/{mb*100:.0f}%   vs   TREND {sht:+.2f}/{ct*100:.1f}%/{mt*100:.0f}%")
print(f"  → trend edge = {'+' if sht>shb else ''}{sht-shb:.2f} Sharpe, {mb*100-mt*100:+.0f}pts LESS drawdown, for {(ct-cb)*100:+.1f}% CAGR")
c.close()
