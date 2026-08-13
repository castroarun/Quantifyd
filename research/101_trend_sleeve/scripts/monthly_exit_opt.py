# -*- coding: utf-8 -*-
"""EXIT/ENTRY optimization on the 10-lot DTE-28 monthly NIFTY straddle (2011-2026, 15y). Levers, all
checked at DAILY CLOSE: profit-target (capture X% of premium), underlying-move stop, early-close at
N-DTE, and VIX entry gate. Preload each trade's daily path once, then sweep all combos."""
import sqlite3, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP, QTY = 160, 0.003, 500
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
NF = {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0")}
VIX = {d: cl for d, cl in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
def mexp():
    ex = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2011-01-01'")})
    bm = defaultdict(list)
    for e in ex: bm[e[:7]].append(e)
    return sorted(max(v) for v in bm.values())
tset = {r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY'")}
tdays = sorted(tset)
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def openleg(d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (d, E, K, ot)).fetchone()
    return (r[0] if r and r[0] else (r[1] if r else None), r[2] or 0 if r else 0) if r else None
def closeseries(E, K, ot, d):
    return {td: cl for td, cl in c.execute("SELECT trade_date,close FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND strike=? AND option_type=? AND trade_date>=? AND trade_date<=? AND close>0", (E, K, ot, d, E))}
# preload paths
paths = []
for E in mexp():
    tgt = dt.date.fromisoformat(E) - dt.timedelta(days=28); d = None
    for k in range(6):
        cd = (tgt - dt.timedelta(days=k)).isoformat()
        if cd in tset and cd in NF: d = cd; break
    if not d or dte(E, d) < 1: continue
    K = round(NF[d][0]/50)*50
    ce = openleg(d, E, K, "CE"); pe = openleg(d, E, K, "PE")
    if not ce or not pe or ce[1] < 50 or pe[1] < 50: continue
    credit = ce[0] + pe[0]; ecl = closeseries(E, K, "CE", d); pcl = closeseries(E, K, "PE", d)
    espot = NF[d][1]; path = []
    for x in tdays:
        if x < d or x > E or dte(E, x) < 1: continue
        if x in ecl and x in pcl and x in NF:
            path.append((dte(E, x), ecl[x] + pcl[x], abs(NF[x][1]-espot)/espot))
    if not path: continue
    paths.append({"m": E[:7], "credit": credit, "vix": VIX.get(d), "path": path})
print(f"preloaded {len(paths)} monthly trades (2011-2026)\n")
def run(PT, STOP, EARLY, VIXMIN):
    series = {}
    for tr in paths:
        if VIXMIN and (tr["vix"] is None or tr["vix"] < VIXMIN): continue
        cr = tr["credit"]; exitmark = tr["path"][-1][1]
        for d_x, mark, move in tr["path"]:
            if (PT and mark <= (1-PT)*cr) or (STOP and move >= STOP) or (EARLY and d_x <= EARLY):
                exitmark = mark; break
        series[tr["m"]] = round((cr-exitmark)*QTY - COST*10 - SLIP*(cr+exitmark)*QTY)
    mm = sorted(series); pl = [series[m] for m in mm]; n = len(pl); tot = sum(pl)
    cum=pk=mdd=0
    for x in pl: cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    return dict(n=n, tot=tot, cal=(tot/(n/12))/abs(mdd) if mdd else 0, mdd=mdd, win=100*sum(1 for x in pl if x>0)/n)
grid = []
for PT in [None, 0.4, 0.5, 0.6]:
    for STOP in [None, 0.02, 0.03, 0.04]:
        for EARLY in [1, 5, 7]:
            for VIXMIN in [None, 13, 15]:
                s = run(PT, STOP, EARLY, VIXMIN); grid.append((s["cal"], PT, STOP, EARLY, VIXMIN, s))
base = run(None, None, 1, None)
print(f"BASELINE (hold to DTE-1, no mgmt): n={base['n']} total ₹{base['tot']:+,} Calmar {base['cal']:.2f} maxDD ₹{base['mdd']:+,} win {base['win']:.0f}%\n")
grid.sort(key=lambda t: t[0], reverse=True)
print(f"{'PT':>4} {'stop':>5} {'earlyDTE':>8} {'VIX':>4} | {'n':>3} {'total':>11} {'Calmar':>7} {'maxDD':>11} {'win':>4}")
for cal, PT, STOP, EARLY, VIXMIN, s in grid[:15]:
    print(f"{('' if PT is None else int(PT*100)):>3}{'%' if PT else ' '} {('-' if STOP is None else str(int(STOP*100))+'%'):>5} {('DTE-'+str(EARLY)):>8} {('-' if VIXMIN is None else '>='+str(VIXMIN)):>4} | {s['n']:>3} {s['tot']:>+11,} {cal:>7.2f} {s['mdd']:>+11,} {s['win']:>3.0f}%")
c.close()
