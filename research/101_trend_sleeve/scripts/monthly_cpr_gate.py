# -*- coding: utf-8 -*-
"""CPR-width entry gate on top of the managed straddle (VIX>=15 + 40% PT + prem-2.5x stop + DTE-5),
10-lot DTE-28, 2016-2026. Test PRIOR-MONTH monthly CPR and PRIOR-DAY daily CPR, skip narrow (compression)
or wide. Causal (uses only completed prior periods)."""
import sqlite3, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP, QTY = 160, 0.003, 500
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
NFO = {d: (o, h, l, cl) for d, o, h, l, cl in c.execute("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 AND date>='2013-01-01'")}
VIX = {d: cl for d, cl in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
ndays = sorted(NFO); prevday = {ndays[i]: ndays[i-1] for i in range(1, len(ndays))}
# monthly OHLC
mo = {}
for d in ndays:
    k = d[:7]; o, h, l, cl = NFO[d]
    if k not in mo: mo[k] = [o, h, l, cl]
    else: mo[k][1] = max(mo[k][1], h); mo[k][2] = min(mo[k][2], l); mo[k][3] = cl
mkeys = sorted(mo); prevmon = {mkeys[i]: mkeys[i-1] for i in range(1, len(mkeys))}
def cprw(H, L, C, spot):
    P = (H+L+C)/3; BC = (H+L)/2; TC = 2*P-BC
    return abs(TC-BC)/spot
def mexp():
    ex = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2016-01-01'")})
    bm = defaultdict(list)
    for e in ex: bm[e[:7]].append(e)
    return sorted(max(v) for v in bm.values())
tset = {r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2016-01-01'")}; tdays = sorted(tset)
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def openleg(d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (d, E, K, ot)).fetchone()
    return (r[0] if r and r[0] else (r[1] if r else None), r[2] or 0 if r else 0) if r else None
def cser(E, K, ot, d):
    return {td: cl for td, cl in c.execute("SELECT trade_date,close FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND strike=? AND option_type=? AND trade_date>=? AND trade_date<=? AND close>0", (E, K, ot, d, E))}
trades = []
for E in mexp():
    tgt = dt.date.fromisoformat(E) - dt.timedelta(days=28); d = None
    for k in range(6):
        cd = (tgt - dt.timedelta(days=k)).isoformat()
        if cd in tset and cd in NFO: d = cd; break
    if not d or dte(E, d) < 1: continue
    vx = VIX.get(d)
    if not (vx and vx >= 15): continue                       # managed already gates VIX>=15
    K = round(NFO[d][0]/50)*50
    ce = openleg(d, E, K, "CE"); pe = openleg(d, E, K, "PE")
    if not ce or not pe or ce[1] < 50 or pe[1] < 50: continue
    credit = ce[0]+pe[0]; ecl = cser(E, K, "CE", d); pcl = cser(E, K, "PE", d)
    hold = [x for x in tdays if d <= x <= E and dte(E, x) >= 1]; xd = hold[-1] if hold else d
    if xd not in ecl or xd not in pcl: continue
    exd, emark = xd, ecl[xd]+pcl[xd]
    for x in hold:
        if x not in ecl or x not in pcl: continue
        mark = ecl[x]+pcl[x]
        if mark <= 0.6*credit or mark >= 2.5*credit or dte(E, x) <= 5: emark = mark; break
    mpl = round((credit-emark)*QTY - COST*10 - SLIP*(credit+emark)*QTY)
    espot = NFO[d][3]
    # prior-month monthly CPR
    pmk = prevmon.get(d[:7]); mcpr = cprw(mo[pmk][1], mo[pmk][2], mo[pmk][3], espot) if pmk in mo else None
    # prior-day daily CPR
    pd = prevday.get(d); dcpr = cprw(NFO[pd][1], NFO[pd][2], NFO[pd][3], espot) if pd in NFO else None
    trades.append({"m": E[:7], "pnl": mpl, "mcpr": mcpr, "dcpr": dcpr})
def stats(sel):
    pl = [t["pnl"] for t in sel]; n = len(pl); tot = sum(pl); cum=pk=mdd=0
    for x in pl: cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    return n, tot, (tot/(n/12))/abs(mdd) if n and mdd else 0, mdd, 100*sum(1 for x in pl if x>0)/n if n else 0
print("CPR-width gate on the MANAGED straddle (VIX>=15 + PT40 + prem2.5x + DTE-5), 10-lot, 2016-2026\n")
print(f"{'gate':>34} | {'n':>3} {'total':>11} {'Calmar':>7} {'maxDD':>11} {'win':>4}")
def show(lbl, sel):
    n, tot, cal, mdd, win = stats(sel); print(f"{lbl:>34} | {n:>3} {tot:>+11,} {cal:>7.2f} {mdd:>+11,} {win:>3.0f}%")
show("baseline (managed, no CPR)", trades)
for th in [0.005, 0.008, 0.012]:
    show(f"skip MONTHLY-CPR < {th*100:.1f}%", [t for t in trades if t["mcpr"] is None or t["mcpr"] >= th])
    show(f"skip MONTHLY-CPR > {th*100:.1f}%", [t for t in trades if t["mcpr"] is None or t["mcpr"] <= th])
for th in [0.0010, 0.0015, 0.0020]:
    show(f"skip DAILY-CPR < {th*100:.2f}%", [t for t in trades if t["dcpr"] is None or t["dcpr"] >= th])
c.close()
