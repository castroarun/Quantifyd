# -*- coding: utf-8 -*-
"""FAR-OTM WINGS for margin reduction on the managed straddle (VIX>=15 + 40%PT + prem2.5x + DTE-5),
10-lot DTE-28, 2016-2026. Buy 5/7/10%-OTM CE+PE (same monthly expiry) at entry, close at the straddle's
exit. Measure the PREMIUM DRAG (the wings barely pay) — the point is the margin/hedge benefit, not P&L."""
import sqlite3, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP, QTY = 160, 0.003, 500
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
NF = {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 AND date>='2015-06-01'")}
VIX = {d: cl for d, cl in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
def mexp():
    ex = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2016-01-01'")})
    bm = defaultdict(list)
    for e in ex: bm[e[:7]].append(e)
    return sorted(max(v) for v in bm.values())
tset = {r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2016-01-01'")}; tdays = sorted(tset)
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def leg(d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (d, E, K, ot)).fetchone()
    return (r[0] if r and r[0] else (r[1] if r else None), r[1] if r else None, r[2] or 0 if r else 0) if r else None
def cser(E, K, ot, d):
    return {td: cl for td, cl in c.execute("SELECT trade_date,close FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND strike=? AND option_type=? AND trade_date>=? AND trade_date<=? AND close>0", (E, K, ot, d, E))}
managed = {}; exits = {}
for E in mexp():
    tgt = dt.date.fromisoformat(E) - dt.timedelta(days=28); d = None
    for k in range(6):
        cd = (tgt - dt.timedelta(days=k)).isoformat()
        if cd in tset and cd in NF: d = cd; break
    if not d or dte(E, d) < 1: continue
    vx = VIX.get(d)
    if not (vx and vx >= 15): continue
    K = round(NF[d][0]/50)*50
    ce = leg(d, E, K, "CE"); pe = leg(d, E, K, "PE")
    if not ce or not pe or ce[2] < 50 or pe[2] < 50: continue
    credit = ce[0]+pe[0]; ecl = cser(E, K, "CE", d); pcl = cser(E, K, "PE", d)
    hold = [x for x in tdays if d <= x <= E and dte(E, x) >= 1]; xd = hold[-1] if hold else d
    if xd not in ecl or xd not in pcl: continue
    exd, emark = xd, ecl[xd]+pcl[xd]
    for x in hold:
        if x not in ecl or x not in pcl: continue
        mark = ecl[x]+pcl[x]
        if mark <= 0.6*credit or mark >= 2.5*credit or dte(E, x) <= 5: exd, emark = x, mark; break
    managed[E[:7]] = (d, E, K, exd, round((credit-emark)*QTY - COST*10 - SLIP*(credit+emark)*QTY))
def wingdrag(pct):
    tot_straddle = 0; tot_fly = 0; wingcosts = []
    for m, (d, E, K, exd, spl) in managed.items():
        Kc = round(K*(1+pct/100)/50)*50; Kp = round(K*(1-pct/100)/50)*50
        wc = leg(d, E, Kc, "CE"); wp = leg(d, E, Kp, "PE")
        if not wc or not wp: tot_straddle += spl; tot_fly += spl; continue
        wcx = leg(exd, E, Kc, "CE"); wpx = leg(exd, E, Kp, "PE")
        wc_out = wcx[1] if wcx and wcx[1] else 0; wp_out = wpx[1] if wpx and wpx[1] else 0
        wingpl = round(((wc_out-wc[0])+(wp_out-wp[0]))*QTY - SLIP*(wc[0]+wp[0])*QTY)
        wingcosts.append(wc[0]+wp[0])
        tot_straddle += spl; tot_fly += spl + wingpl
    avgcost = sum(wingcosts)/len(wingcosts) if wingcosts else 0
    return tot_straddle, tot_fly, avgcost, len(managed)
print("FAR-OTM WINGS on the managed straddle (10-lot, 2016-2026) — premium drag:\n")
print(f"{'wing':>6} | {'straddle total':>15} {'+wings total':>15} {'drag':>11} {'avg wing cost/mo':>16}")
for pct in [5, 7, 10, 12, 15, 20]:
    ts, tf, ac, n = wingdrag(pct)
    print(f"{pct:>4}% | {ts:>+15,} {tf:>+15,} {tf-ts:>+11,} {'~'+str(round(ac))+' pts (₹'+str(round(ac*QTY)):>13}/mo)")
print(f"\n({n} managed trades. Wing cost in pts x qty {QTY} = rupee cost of the wings per month.)")
print("Margin: naked 10-lot straddle ~₹13-15L SPAN; hedged (any long wing) ~₹6-7L (Zerodha hedge benefit)")
print("-> frees ~₹6-8L -> 6% liquid = +₹36-48k/yr, OR ~1.8-2x the lots on the same capital.")
c.close()
