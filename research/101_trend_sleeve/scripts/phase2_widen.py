# -*- coding: utf-8 -*-
"""Widen the managed-futures search: weekly Donchian trend on every clean-history asset in the DB,
+ a gold+silver commodity sleeve. (Crude/USDINR not in DB — would need download.)"""
import sqlite3, math, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=20000")
def weekly(sym, frm="2016-01-01"):
    b = c.execute("SELECT date,high,low,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0 AND date>=? ORDER BY date", (sym, frm)).fetchall()
    wk = {}
    for d, h, l, cl in b:
        y, w, _ = dt.date.fromisoformat(d).isocalendar(); k = (y, w)
        if k not in wk: wk[k] = [d, h, l, cl]
        else: wk[k][1] = max(wk[k][1], h); wk[k][2] = min(wk[k][2], l); wk[k][3] = cl
    return [wk[k] for k in sorted(wk)]
def donch(b, N=10):
    r = []; pos = 0
    for i in range(N, len(b) - 1):
        hh = max(x[1] for x in b[i-N:i]); ll = min(x[2] for x in b[i-N:i]); cl = b[i][3]
        if cl > hh: pos = 1
        elif cl < ll: pos = -1
        r.append((b[i+1][0], pos * (b[i+1][3] / b[i][3] - 1)))
    return r
def stat(r):
    v = [x[1] for x in r]; n = len(v); mu = sum(v) / n; sd = (sum((x-mu)**2 for x in v)/n)**0.5
    cum=pk=mdd=0
    for x in v: cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    return mu/sd*math.sqrt(52) if sd else 0, sum(v), mdd, n
print("weekly Donchian-10 trend, per asset:")
for s in ["GOLDBEES", "SILVERBEES", "JUNIORBEES", "NIFTYBEES"]:
    sh, tot, mdd, n = stat(donch(weekly(s)))
    print(f"  {s:>10}: Sharpe {sh:+.2f} | total {tot*100:+.0f}% | maxDD {mdd*100:+.0f}% | ann {tot/(n/52)*100:+.1f}% ({n/52:.1f}y)")
# gold+silver commodity sleeve (equal-weight weekly), 2022+ overlap
gs = defaultdict(list)
for s in ["GOLDBEES", "SILVERBEES"]:
    for d, ret in donch(weekly(s, "2022-01-01")): gs[d].append(ret)
sleeve = [(d, sum(v)/len(v)) for d, v in sorted(gs.items()) if len(v) == 2]
sh, tot, mdd, n = stat(sleeve)
print(f"\ngold+silver sleeve (2022+, {n/52:.1f}y): Sharpe {sh:+.2f} | ann {tot/(n/52)*100:+.1f}% | maxDD {mdd*100:+.0f}%")
# gold-only same window for comparison
sh2, tot2, mdd2, n2 = stat([x for x in donch(weekly("GOLDBEES", "2022-01-01"))])
print(f"gold-only    (2022+, {n2/52:.1f}y): Sharpe {sh2:+.2f} | ann {tot2/(n2/52)*100:+.1f}% | maxDD {mdd2*100:+.0f}%")
c.close()
