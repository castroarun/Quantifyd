# -*- coding: utf-8 -*-
"""PHASE 2 · C2 — Dual-SuperTrend master/child FUTURES trend engine on NIFTY daily (7yr).
Master ST sets direction (long/short 1 lot); each with-trend CHILD flip pyramids +1 lot (cap 3);
exit/reverse on master flip. EOD-close actions. Sweeps ST params, ranks by Calmar. (Option hedge on
counter-trend child flips = stage 2.) Standalone first; straddle blend next."""
import sqlite3, sys, math
import pandas as pd
sys.path.insert(0, "/home/arun/quantifyd")
from services.maruthi_strategy import compute_dual_supertrend
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
LOT, PT_COST, MAXLOTS = 75, 3.0, 3           # NIFTY fut lot 75; ~3 pts round-trip cost/lot; pyramid cap
c = sqlite3.connect(DB)
rows = c.execute("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 AND date>='2018-01-01' ORDER BY date").fetchall()
c.close()
base = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close'])

def backtest(mp, mm, cp, cm):
    df = compute_dual_supertrend(base.copy(), master_period=mp, master_mult=mm, child_period=cp, child_mult=cm)
    md = df['master_dir'].values; cfb = df['child_flip_bull'].values; cfe = df['child_flip_bear'].values
    cl = df['close'].values; dates = df['date'].values
    n = len(df); warm = max(mp, cp) + 2
    pos = 0; prev = 0; daily = []
    for t in range(warm, n - 1):
        d = int(md[t])
        if d != prev:                                    # master flip → reverse to 1 lot
            cost = (abs(pos) + 1) * PT_COST
            pos = d
        else:                                            # same regime: pyramid on with-trend child flip
            cost = 0.0
            if abs(pos) < MAXLOTS and ((d == 1 and cfb[t]) or (d == -1 and cfe[t])):
                pos += d; cost = PT_COST
        pnl = pos * (cl[t + 1] - cl[t]) - cost           # mark-to-market next day, minus event cost
        daily.append((dates[t + 1], pnl))
        prev = d
    return daily

def stats(daily):
    p = [x[1] for x in daily]; n = len(p); tot = sum(p)
    mu = tot / n; sd = (sum((x - mu) ** 2 for x in p) / n) ** 0.5
    sharpe = mu / sd * math.sqrt(252) if sd else 0
    cum = pk = mdd = 0
    for x in p:
        cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
    yrs = n / 252
    calmar = (tot / yrs) / abs(mdd) if mdd else 0
    return tot, sharpe, mdd, calmar, yrs

print("Dual-SuperTrend futures core — NIFTY daily 2018-2026, points (×₹75/lot for ₹)\n")
print(f"{'master':>8} {'child':>8}  {'tot pts':>9} {'ann.pts':>8} {'Sharpe':>7} {'maxDD':>8} {'Calmar':>7}")
results = []
for mp in [7, 10, 14]:
    for mm in [3.0, 4.0, 5.0]:
        for cp in [7, 10]:
            for cm in [1.5, 2.0, 2.5]:
                d = backtest(mp, mm, cp, cm)
                tot, sh, mdd, cal, yrs = stats(d)
                results.append((cal, sh, tot, mdd, mp, mm, cp, cm, d))
results.sort(reverse=True)
for cal, sh, tot, mdd, mp, mm, cp, cm, d in results[:12]:
    print(f"ST({mp},{mm:.0f}) ST({cp},{cm:.1f})  {tot:>+9.0f} {tot/(len(d)/252):>+8.0f} {sh:>7.2f} {mdd:>+8.0f} {cal:>7.2f}")
# save the best config's daily series for the straddle blend
best = results[0]
import json
json.dump([[str(x[0]), round(x[1], 2)] for x in best[8]], open("/tmp/dualst_best.json", "w"))
print(f"\nBEST: ST({best[4]},{best[5]:.0f}) master / ST({best[6]},{best[7]:.1f}) child — Calmar {best[0]:.2f}, Sharpe {best[1]:.2f}, +{best[2]:.0f} pts (~₹{best[2]*LOT:,.0f}/lot over {best[8].__len__()/252:.1f}y)")
print("saved best daily series -> /tmp/dualst_best.json")
