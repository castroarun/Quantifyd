# -*- coding: utf-8 -*-
"""SENSEX DTE-N short straddle study on real BSE bhavcopy (bse_options_bhav, 2024->now).
Entry = option OPEN at DTE-N (nearest trade day <= expiry-N days); exit = DTE-1 CLOSE.
Naked + 3% next-week fly. Net 0.3% slip + Rs160/RT. OI>=100 ATM / >=25 wings. India VIX 13-28 gate.
Reports: DTE sweep (gated+ungated), DTE-3 naked vs fly, per-year DTE-3."""
import sqlite3, datetime as dt, statistics
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
SYM, STEP, COST, SLIP = "SENSEX", 100, 160, 0.003
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
vix = {d: cl for d, cl in c.execute("SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
sp = defaultdict(list)
for d, u in c.execute("SELECT trade_date, underlying FROM bse_options_bhav WHERE symbol=? AND underlying>0", (SYM,)):
    sp[d].append(u)
spot = {d: statistics.median(v) for d, v in sp.items()}
tdates = sorted(spot); tset = set(tdates)
exps = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM bse_options_bhav WHERE symbol=?", (SYM,))})
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def leg(d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest,lot FROM bse_options_bhav WHERE symbol=? AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (SYM, d, E, K, ot)).fetchone()
    if not r: return None
    o, cl, oi, lot = r
    return (o if o and o > 0 else cl, cl, oi or 0, lot or 10)

def run(N, use_vix, wings):
    naked = []; fly = []
    for E in exps:
        target = dt.date.fromisoformat(E) - dt.timedelta(days=N)
        d = None
        for k in range(4):
            cd = (target - dt.timedelta(days=k)).isoformat()
            if cd in tset: d = cd; break
        if not d or dte(E, d) < 1 or d not in spot: continue
        vx = vix.get(d)
        if use_vix and not (vx and 13 <= vx <= 28): continue
        K = round(spot[d]/STEP)*STEP
        ce = leg(d, E, K, "CE"); pe = leg(d, E, K, "PE")
        if not ce or not pe or ce[2] < 100 or pe[2] < 100: continue
        hold = [x for x in tdates if d <= x <= E and dte(E, x) >= 1]; xd = hold[-1] if hold else d
        cf = leg(xd, E, K, "CE"); pf = leg(xd, E, K, "PE")
        if not cf or not pf: continue
        QTY = ce[3]*10
        nk = round((ce[0]+pe[0]-(cf[1]+pf[1]))*QTY - COST - SLIP*(ce[0]+pe[0]+cf[1]+pf[1])*QTY)
        naked.append((xd, nk))
        fl = nk
        if wings:
            i = exps.index(E); WE = exps[i+1] if i+1 < len(exps) else None
            wcK = round(K*1.03/STEP)*STEP; wpK = round(K*0.97/STEP)*STEP
            wn = None
            if WE:
                wc = leg(d, WE, wcK, "CE"); wp = leg(d, WE, wpK, "PE"); wcf = leg(xd, WE, wcK, "CE"); wpf = leg(xd, WE, wpK, "PE")
                if wc and wp and wcf and wpf and wc[2] >= 25 and wp[2] >= 25:
                    wn = ((wcf[1]-wc[0])+(wpf[1]-wp[0]))*QTY - SLIP*(wc[0]+wp[0]+wcf[1]+wpf[1])*QTY
            if wn is not None: fl = round(nk + wn)
        fly.append((xd, fl))
    return naked, fly

def st(pl):
    p = [x[1] for x in pl]; n = len(p)
    if not n: return "n=0"
    m = sum(p)/n; sd = (sum((x-m)**2 for x in p)/n)**0.5
    cum = pk = mdd = 0
    for x in p: cum += x; pk = max(pk, cum); mdd = min(mdd, cum-pk)
    return f"n={n:>3} total={round(sum(p)):>+10,} mean={round(m):>+7,} win={round(100*sum(1 for x in p if x>0)/n):>3}% M/SD={m/sd if sd else 0:>5.2f} worst={round(min(p)):>+9,} maxDD={round(mdd):>+9,}"

lot0 = c.execute("SELECT lot FROM bse_options_bhav WHERE symbol=? ORDER BY trade_date DESC LIMIT 1", (SYM,)).fetchone()
print(f"SENSEX current lot={lot0[0] if lot0 else '?'}, 10-lot book. Trade days {tdates[0]}..{tdates[-1]} ({len(tdates)})")
print("\n== DTE sweep, naked, ungated vs VIX 13-28 ==")
for N in [1, 2, 3, 4, 5]:
    print(f" DTE-{N} naked      : {st(run(N, False, False)[0])}")
    print(f" DTE-{N} naked+VIX  : {st(run(N, True, False)[0])}")
nk, fl = run(3, False, True)
print("\n== DTE-3 (the NIFTY-optimal) ==")
print(" naked        :", st(nk))
print(" fly 3% next-wk:", st(fl))
yr = defaultdict(list)
for xd, v in nk: yr[xd[:4]].append(v)
print("\n== DTE-3 naked, per year ==")
for y in sorted(yr):
    print(f"  {y}: {st([(0, v) for v in yr[y]])}")
c.close()
