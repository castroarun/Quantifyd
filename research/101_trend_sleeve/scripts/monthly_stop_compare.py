# -*- coding: utf-8 -*-
"""Stop-type bake-off on the 10-lot DTE-28 monthly straddle (VIX>=15 gate + 40% PT + close DTE-5 base,
2011-2026). Compare: no stop / underlying-move stop / combined-PREMIUM stop (mark >= M*credit) /
LEG-RATIO stop (one side >= R* the other). All checked at daily close."""
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
tset = {r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY'")}; tdays = sorted(tset)
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def openleg(d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (d, E, K, ot)).fetchone()
    return (r[0] if r and r[0] else (r[1] if r else None), r[2] or 0 if r else 0) if r else None
def cser(E, K, ot, d):
    return {td: cl for td, cl in c.execute("SELECT trade_date,close FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND strike=? AND option_type=? AND trade_date>=? AND trade_date<=? AND close>0", (E, K, ot, d, E))}
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
    credit = ce[0] + pe[0]; ecl = cser(E, K, "CE", d); pcl = cser(E, K, "PE", d); espot = NF[d][1]; path = []
    for x in tdays:
        if x < d or x > E or dte(E, x) < 1: continue
        if x in ecl and x in pcl and x in NF:
            path.append((dte(E, x), ecl[x], pcl[x], abs(NF[x][1]-espot)/espot))
    if path: paths.append({"m": E[:7], "credit": credit, "vix": VIX.get(d), "path": path})

def run(MOVE=None, PREM=None, LEGR=None, PT=0.4, EARLY=5, VIXMIN=15):
    series = {}
    for tr in paths:
        if VIXMIN and (tr["vix"] is None or tr["vix"] < VIXMIN): continue
        cr = tr["credit"]; exitmark = tr["path"][-1][1] + tr["path"][-1][2]
        for dx, ce, pe, mv in tr["path"]:
            mark = ce + pe; ratio = max(ce, pe)/min(ce, pe) if min(ce, pe) > 0 else 99
            if (PT and mark <= (1-PT)*cr) or (MOVE and mv >= MOVE) or (PREM and mark >= PREM*cr) or (LEGR and ratio >= LEGR) or (EARLY and dx <= EARLY):
                exitmark = mark; break
        series[tr["m"]] = round((cr-exitmark)*QTY - COST*10 - SLIP*(cr+exitmark)*QTY)
    pl = list(series.values()); n = len(pl); tot = sum(pl); cum=pk=mdd=0
    for x in pl: cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    return n, tot, (tot/(n/12))/abs(mdd) if mdd else 0, mdd, 100*sum(1 for x in pl if x>0)/n
configs = [
    ("no stop (PT+DTE5 only)", dict()),
    ("underlying move 3%", dict(MOVE=0.03)), ("underlying move 4%", dict(MOVE=0.04)),
    ("combined-premium 1.5x", dict(PREM=1.5)), ("combined-premium 2.0x", dict(PREM=2.0)), ("combined-premium 2.5x", dict(PREM=2.5)),
    ("leg-ratio 2x (one dbl)", dict(LEGR=2.0)), ("leg-ratio 3x (one triple)", dict(LEGR=3.0)),
    ("move4% + prem2x", dict(MOVE=0.04, PREM=2.0)),
]
for VG, vlbl in [(15, "WITH VIX>=15 gate"), (None, "NO VIX gate (all months)")]:
    print(f"\n=== {vlbl} · 40% PT + close DTE-5, 10-lot DTE-28 straddle, 2011-2026 ===")
    print(f"{'stop type':>26} | {'n':>3} {'total':>11} {'Calmar':>7} {'maxDD':>11} {'win':>4}")
    for name, kw in configs:
        n, tot, cal, mdd, win = run(**kw, VIXMIN=VG)
        print(f"{name:>26} | {n:>3} {tot:>+11,} {cal:>7.2f} {mdd:>+11,} {win:>3.0f}%")
c.close()
