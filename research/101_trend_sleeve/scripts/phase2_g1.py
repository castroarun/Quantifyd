# -*- coding: utf-8 -*-
"""PHASE 2 · G1 PROBE — does a canonical trend signal (a) have a standalone edge and (b) fire in the
short-straddle's WORST months (the cushion test)? Donchian-20 long/short on NIFTY + BankNifty daily
(index = futures proxy). Correlate its monthly P&L with the naked DTE-3 straddle flow. Cheap gate."""
import sqlite3, json, datetime as dt, math
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=20000")
def bars(sym):
    return [(d, o, h, l, cl) for d, o, h, l, cl in c.execute(
        "SELECT date,open,high,low,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0 AND date>='2018-01-01' ORDER BY date", (sym,))]
def donchian(sym, N=20):
    """long/short Donchian-N; returns {month: sum daily strat return}."""
    b = bars(sym); mret = defaultdict(float); pos = 0
    for i in range(N, len(b) - 1):
        hh = max(x[2] for x in b[i-N:i]); ll = min(x[3] for x in b[i-N:i])
        cl = b[i][4]
        if cl > hh: pos = 1
        elif cl < ll: pos = -1
        r = b[i+1][4] / b[i][4] - 1.0          # next-day return (causal, 1-day lag)
        mret[b[i+1][0][:7]] += pos * r
    return mret, b
# trend book = equal-weight NIFTY + BankNifty Donchian-20
tn, _ = donchian("NIFTY50"); tb, _ = donchian("BANKNIFTY")
months = sorted(set(tn) | set(tb))
trend = {m: (tn.get(m, 0) + tb.get(m, 0)) / 2 for m in months}
# straddle flow = naked DTE-3, monthly ₹
sd = json.load(open("/tmp/realistic_trades.json"))
strad = defaultdict(float)
for t in sd["dte3"]: strad[t["exit"][:7]] += t["naked"]
# --- standalone trend stats (daily) ---
def daily_series(sym, N=20):
    b = bars(sym); out = []; pos = 0
    for i in range(N, len(b) - 1):
        hh = max(x[2] for x in b[i-N:i]); ll = min(x[3] for x in b[i-N:i]); cl = b[i][4]
        if cl > hh: pos = 1
        elif cl < ll: pos = -1
        out.append(pos * (b[i+1][4] / b[i][4] - 1.0))
    return out
dn = daily_series("NIFTY50"); db = daily_series("BANKNIFTY")
L = min(len(dn), len(db)); comb = [(dn[i] + db[i]) / 2 for i in range(L)]
mu = sum(comb) / len(comb); sd_ = (sum((x-mu)**2 for x in comb)/len(comb))**0.5
sharpe = mu / sd_ * math.sqrt(252)
cum = pk = mdd = 0
for x in comb: cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
tot_ret = sum(comb)
print(f"TREND (Donchian-20 L/S, NIFTY+BNF equal-wt, 2018-2026):")
print(f"  standalone: total +{tot_ret*100:.0f}% pts | ann.Sharpe={sharpe:.2f} | maxDD={mdd*100:.0f}% pts | ann.ret~{tot_ret/ ((L)/252)*100:.1f}%")
# --- correlation of monthly P&L ---
common = sorted(set(trend) & set(strad))
xs = [trend[m] for m in common]; ys = [strad[m] for m in common]
n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
cov = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/n
sx = (sum((x-mx)**2 for x in xs)/n)**0.5; sy = (sum((y-my)**2 for y in ys)/n)**0.5
corr = cov/(sx*sy) if sx and sy else 0
print(f"\nCORRELATION (monthly, n={n} months): trend vs naked-straddle P&L = {corr:+.2f}")
print("  (negative = complementary: trend earns when the straddle bleeds)")
# --- the cushion test: trend's return in the straddle's 5 WORST months ---
worst = sorted(common, key=lambda m: strad[m])[:6]
print("\nCUSHION TEST — straddle's worst months vs trend that month:")
for m in worst:
    print(f"  {m}: straddle ₹{strad[m]:>+9,.0f} | trend {trend[m]*100:>+6.1f}% pts  {'CUSHION ✓' if trend[m]>0 else 'no help'}")
c.close()
