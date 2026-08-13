# -*- coding: utf-8 -*-
"""WEEKLY rolled far-OTM wings (vs monthly) as the margin hedge on the managed straddle. For each
managed monthly trade (2019-2026, weeklies exist), buy X%-OTM CE+PE in the NEAREST WEEKLY, roll to a
new weekly each week until the straddle exits. Much less time value than the 28-DTE monthly wing."""
import sqlite3, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP, QTY = 160, 0.003, 500
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
NF = {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 AND date>='2018-06-01'")}
VIX = {d: cl for d, cl in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
allexp = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2019-01-01'")})
def mexp():
    bm = defaultdict(list)
    for e in allexp: bm[e[:7]].append(e)
    return sorted(max(v) for v in bm.values())
tset = {r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2019-01-01'")}; tdays = sorted(tset)
def nextday(x):
    i = tdays.index(x) if x in tdays else -1
    return tdays[i+1] if 0 <= i < len(tdays)-1 else None
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def leg(d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (d, E, K, ot)).fetchone()
    return (r[0] if r and r[0] else (r[1] if r else None), r[1] if r else None) if r else None
def cser(E, K, ot, d):
    return {td: cl for td, cl in c.execute("SELECT trade_date,close FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND strike=? AND option_type=? AND trade_date>=? AND trade_date<=? AND close>0", (E, K, ot, d, E))}
managed = {}
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
    if not ce or not pe: continue
    credit = ce[0]+pe[0]; ecl = cser(E, K, "CE", d); pcl = cser(E, K, "PE", d)
    hold = [x for x in tdays if d <= x <= E and dte(E, x) >= 1]; xd = hold[-1] if hold else d
    if xd not in ecl or xd not in pcl: continue
    exd, emark = xd, ecl[xd]+pcl[xd]
    for x in hold:
        if x not in ecl or x not in pcl: continue
        mark = ecl[x]+pcl[x]
        if mark <= 0.6*credit or mark >= 2.5*credit or dte(E, x) <= 5: exd, emark = x, mark; break
    managed[E[:7]] = (d, K, exd, round((credit-emark)*QTY - COST*10 - SLIP*(credit+emark)*QTY))
def weekly_wing_drag(pct):
    tot_str = 0; tot_fly = 0; nwk = 0
    for m, (d, K, exd, spl) in managed.items():
        Kc = round(K*(1+pct/100)/50)*50; Kp = round(K*(1-pct/100)/50)*50; wpl = 0
        weeklies = [e for e in allexp if d <= e]
        seg = d
        for we in weeklies:
            if seg > exd: break
            wc = leg(seg, we, Kc, "CE"); wp = leg(seg, we, Kp, "PE")
            if not wc or not wp or wc[0] is None or wp[0] is None: seg = nextday(we) or (exd+"z"); continue
            close_at = min(we, exd)
            if close_at == we:
                s = NF.get(we, (0, K))[1]; out = max(s-Kc, 0) + max(Kp-s, 0)              # settle intrinsic
            else:
                cx = leg(exd, we, Kc, "CE"); px = leg(exd, we, Kp, "PE")                  # sell at exd close
                out = (cx[1] if cx and cx[1] else 0) + (px[1] if px and px[1] else 0)
            wpl += ((out-(wc[0]+wp[0]))*QTY - SLIP*(wc[0]+wp[0])*QTY); nwk += 1
            if we >= exd: break
            seg = nextday(we)
            if not seg or seg > exd: break
        tot_str += spl; tot_fly += spl + round(wpl)
    return tot_str, tot_fly, nwk
print("WEEKLY rolled far-OTM wings on the managed straddle (10-lot, 2019-2026):\n")
print(f"{'wing':>5} | {'straddle total':>15} {'+weekly-wings':>15} {'drag':>12} {'wing-legs':>10}")
for pct in [3, 4, 5, 7]:
    ts, tf, nwk = weekly_wing_drag(pct)
    print(f"{pct:>4}% | {ts:>+15,} {tf:>+15,} {tf-ts:>+12,} {nwk:>10}")
print(f"\n(vs the MONTHLY 5% wing which dragged −₹33.9L. n={len(managed)} managed trades, 2019+.)")
c.close()
