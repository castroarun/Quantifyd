# -*- coding: utf-8 -*-
"""NIFTY DTE sweep on the SAME 2024-01->now window as SENSEX, same methodology, for a fair head-to-head.
nse_options_bhav, spot=NIFTY50 daily close, entry OPEN at DTE-N, exit DTE-1 CLOSE, net 0.3% slip + Rs160."""
import sqlite3, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
SYM, STEP, COST, SLIP, QTY = "NIFTY", 50, 160, 0.003, 650   # NIFTY lot 65 x 10
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
FROM = "2024-01-01"
spot = {d: cl for d, cl in c.execute("SELECT date, close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 AND date>=?", (FROM,))}
vix = {d: cl for d, cl in c.execute("SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
exps = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol=? AND expiry_date>=?", (SYM, FROM))})
tdates = sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol=? AND trade_date>=?", (SYM, FROM))})
tset = set(tdates)
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def leg(d, E, K, ot):
    r = c.execute("SELECT open, close, open_interest FROM nse_options_bhav WHERE symbol=? AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (SYM, d, E, K, ot)).fetchone()
    if not r: return None
    o, cl, oi = r
    return (o if o and o > 0 else cl, cl, oi or 0)
def run(N):
    out = []
    for E in exps:
        target = dt.date.fromisoformat(E) - dt.timedelta(days=N)
        d = None
        for k in range(4):
            cd = (target - dt.timedelta(days=k)).isoformat()
            if cd in tset: d = cd; break
        if not d or dte(E, d) < 1 or d not in spot: continue
        K = round(spot[d]/STEP)*STEP
        ce = leg(d, E, K, "CE"); pe = leg(d, E, K, "PE")
        if not ce or not pe or ce[2] < 200 or pe[2] < 200: continue
        hold = [x for x in tdates if d <= x <= E and dte(E, x) >= 1]; xd = hold[-1] if hold else d
        cf = leg(xd, E, K, "CE"); pf = leg(xd, E, K, "PE")
        if not cf or not pf: continue
        out.append((xd, round((ce[0]+pe[0]-(cf[1]+pf[1]))*QTY - COST - SLIP*(ce[0]+pe[0]+cf[1]+pf[1])*QTY)))
    return out
def st(pl):
    p = [x[1] for x in pl]; n = len(p)
    if not n: return "n=0"
    m = sum(p)/n; sd = (sum((x-m)**2 for x in p)/n)**0.5
    cum = pk = mdd = 0
    for x in p: cum += x; pk = max(pk, cum); mdd = min(mdd, cum-pk)
    return f"n={n:>3} total={round(sum(p)):>+10,} mean={round(m):>+7,} win={round(100*sum(1 for x in p if x>0)/n):>3}% M/SD={m/sd if sd else 0:>5.2f} worst={round(min(p)):>+9,} maxDD={round(mdd):>+9,}"
print(f"NIFTY same-period 2024-01->now, QTY {QTY} (10 lots), days {tdates[0]}..{tdates[-1]}")
for N in [1, 2, 3, 4, 5]:
    print(f" DTE-{N} naked: {st(run(N))}")
yr = defaultdict(list)
for xd, v in run(3): yr[xd[:4]].append(v)
print("DTE-3 per year:")
for y in sorted(yr): print(f"  {y}: {st([(0, v) for v in yr[y]])}")
c.close()
