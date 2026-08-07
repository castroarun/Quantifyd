# -*- coding: utf-8 -*-
"""PREMIUM-BASED short strangle optimization (unbiased, from scratch). Select strikes by TARGET PREMIUM
(not %OTM): sell the CE & PE nearest to Rs P each. Weekly cadence (enter ~4 trading days before each
NIFTY expiry, 2019-2026). EOD-close management. Sweep premium level x management x VIX gate. 10 lots
(qty 750). Reports Calmar (calendar basis), total, win, maxDD."""
import sqlite3, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP, QTY = 160, 0.003, 750
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
VIX = {d: cl for d, cl in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
exps = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2019-01-01'")})
tdays = sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2019-01-01'")}); tset = set(tdays)
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def nearest_strike(d, E, ot, P):
    r = c.execute("SELECT strike,close FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date=? AND expiry_date=? AND option_type=? AND close>0 AND open_interest>0 ORDER BY ABS(close-?) LIMIT 1", (d, E, ot, P)).fetchone()
    return r  # (strike, premium)
def cser(E, K, ot, d):
    return {td: cl for td, cl in c.execute("SELECT trade_date,close FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND strike=? AND option_type=? AND trade_date>=? AND trade_date<=? AND close>0", (E, K, ot, d, E))}
# preload trades: for each expiry, entry ~4 trading days before, sell strangle at premium P
def build(P):
    trades = []
    for E in exps:
        before = [x for x in tdays if x < E]
        if len(before) < 4: continue
        d = before[-4]                                   # ~4 trading days before expiry (the "Monday")
        if dte(E, d) < 1 or dte(E, d) > 12: continue     # weekly-ish window
        ce = nearest_strike(d, E, "CE", P); pe = nearest_strike(d, E, "PE", P)
        if not ce or not pe: continue
        credit = ce[1] + pe[1]
        if credit < P: continue                          # need a real strangle near 2P
        ecl = cser(E, ce[0], "CE", d); pcl = cser(E, pe[0], "PE", d)
        path = []
        for x in [t for t in tdays if d <= t <= E]:
            if x in ecl and x in pcl: path.append((dte(E, x), ecl[x]+pcl[x]))
        if len(path) < 2: continue
        trades.append({"E": E, "d": d, "credit": credit, "vix": VIX.get(d), "path": path})
    return trades
def run(trades, PT, PREMx, VIXMIN):
    pl = []
    for t in trades:
        if VIXMIN and (t["vix"] is None or t["vix"] < VIXMIN): continue
        cr = t["credit"]; exitmark = t["path"][-1][1]
        for dx, mark in t["path"]:
            if (PT and mark <= (1-PT)*cr) or (PREMx and mark >= PREMx*cr):
                exitmark = mark; break
        pl.append(round((cr-exitmark)*QTY - COST*10 - SLIP*(cr+exitmark)*QTY))
    n = len(pl); tot = sum(pl); cum=pk=mdd=0
    for x in pl: cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    yrs = 7.4  # 2019-2026 calendar
    return n, tot, (tot/yrs)/abs(mdd) if mdd else 0, mdd, 100*sum(1 for x in pl if x>0)/n if n else 0
print("PREMIUM-BASED weekly NIFTY strangle (10 lots=qty 750, 2019-2026). Sell CE&PE nearest Rs P each.\n")
grid = []
for P in [10, 15, 20, 30, 40]:
    tr = build(P)
    for PT in [None, 0.5, 0.6]:
        for PREMx in [None, 2.0, 2.5, 3.0]:
            for VIXMIN in [None, 13, 15]:
                n, tot, cal, mdd, win = run(tr, PT, PREMx, VIXMIN)
                if n: grid.append((cal, P, PT, PREMx, VIXMIN, n, tot, mdd, win))
grid.sort(key=lambda t: t[0], reverse=True)
print(f"{'premium':>7} {'PT':>4} {'premSL':>6} {'VIX':>5} | {'n':>3} {'total':>11} {'Calmar':>7} {'maxDD':>11} {'win':>4}")
for cal, P, PT, PREMx, VIXMIN, n, tot, mdd, win in grid[:18]:
    print(f"  Rs{P:>3}  {('' if PT is None else str(int(PT*100))+'%'):>4} {('-' if PREMx is None else str(PREMx)+'x'):>6} {('-' if VIXMIN is None else '>='+str(VIXMIN)):>5} | {n:>3} {tot:>+11,} {cal:>7.2f} {mdd:>+11,} {win:>3.0f}%")
c.close()
