# -*- coding: utf-8 -*-
"""SBI long-options cushion vs BankNifty, on the 10-LOT DTE-28 monthly NIFTY straddle (2016-2026, the
window SBI options exist). Which cushion better lifts the combined Calmar + pays in the crash months?
All ₹ presented at 10 LOTS (straddle = 500 NIFTY units)."""
import sqlite3, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP = 160, 0.003
NSTRAD_QTY = 500          # 10 NIFTY lots (50 units/lot basis)
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
def spot(sym): return {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0 AND date>='2015-06-01'", (sym,))}
NF, BN, SB = spot("NIFTY50"), spot("BANKNIFTY"), spot("SBIN")
def mexp(sym, frm="2016-01-01"):
    ex = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol=? AND expiry_date>=?", (sym, frm))})
    bm = defaultdict(list)
    for e in ex: bm[e[:7]].append(e)
    return sorted(max(v) for v in bm.values())
tdays = sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2016-01-01'")}); tset = set(tdays)
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def leg(sym, d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol=? AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (sym, d, E, K, ot)).fetchone()
    return (r[0] if r and r[0] else (r[1] if r else None), r[1] if r else None, r[2] or 0 if r else 0) if r else None
# NIFTY straddle DTE-28, 10 lots
strad = {}
for E in mexp("NIFTY"):
    tgt = dt.date.fromisoformat(E) - dt.timedelta(days=28); d = None
    for k in range(6):
        cd = (tgt - dt.timedelta(days=k)).isoformat()
        if cd in tset and cd in NF: d = cd; break
    if not d or dte(E, d) < 1: continue
    K = round(NF[d][0]/50)*50
    ce = leg("NIFTY", d, E, K, "CE"); pe = leg("NIFTY", d, E, K, "PE")
    if not ce or not pe or ce[2] < 50 or pe[2] < 50: continue
    hold = [x for x in tdays if d <= x <= E and dte(E, x) >= 1]; xd = hold[-1] if hold else d
    cf = leg("NIFTY", xd, E, K, "CE"); pf = leg("NIFTY", xd, E, K, "PE")
    if not cf or not pf: continue
    strad[E[:7]] = round((ce[0]+pe[0]-(cf[1]+pf[1]))*NSTRAD_QTY - COST*10 - SLIP*(ce[0]+pe[0]+cf[1]+pf[1])*NSTRAD_QTY)
def long_strangle(sym, sp, step, lot, pct):
    out = defaultdict(float)
    for E in mexp(sym):
        if E not in sp: continue
        d = None
        for k in range(8):
            cd = (dt.date.fromisoformat(E)-dt.timedelta(days=28+k)).isoformat()
            if cd in tset and cd in sp: d = cd; break
        if not d: continue
        atm = round(sp[d][0]/step)*step; Kc = round(atm*(1+pct/100)/step)*step; Kp = round(atm*(1-pct/100)/step)*step
        ce = leg(sym, d, E, Kc, "CE"); pe = leg(sym, d, E, Kp, "PE")
        if not ce or not pe or ce[2] < 20 or pe[2] < 20: continue
        out[E[:7]] = (max(sp[E][1]-Kc,0)+max(Kp-sp[E][1],0) - (ce[0]+pe[0]) - SLIP*(ce[0]+pe[0]))*lot
    return out
BNFc = long_strangle("BANKNIFTY", BN, 100, 15, 1)      # BNF 1% strangle, lot 15
SBIc = long_strangle("SBIN", SB, 10, 750, 5)           # SBI 5% strangle, lot 750
def calmar(series):
    mm = sorted(series); pl = [series[m] for m in mm]; cum=pk=mdd=0
    for x in pl: cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    return (sum(pl)/(len(mm)/12))/abs(mdd) if mdd else 0, mdd, sum(pl)
mm = sorted(strad); c0, m0, t0 = calmar(strad)
print(f"10-LOT DTE-28 NIFTY straddle (2016-2026, {len(mm)} months): Calmar {c0:.2f} total ₹{t0:+,} maxDD ₹{m0:+,}\n")
for name, cush in [("BankNifty 1% (lot15)", BNFc), ("SBI 5% (lot750)", SBIc)]:
    best = (c0, 0, m0, t0)
    line = []
    for w in [0, 1, 2, 3, 5, 8, 12]:
        comb = {m: strad[m] + w*cush.get(m, 0) for m in mm}
        cal, mdd, tot = calmar(comb); line.append(f"w{w}:{cal:.2f}")
        if cal > best[0]: best = (cal, w, mdd, tot)
    print(f"{name}: " + " ".join(line))
    print(f"   -> best w={best[1]}: Calmar {best[0]:.2f}, total ₹{best[3]:+,}, maxDD ₹{best[2]:+,}\n")
print("CRASH MONTHS — cushion payoff at w=3 (10-lot straddle basis):")
for cm in ["2018-02", "2018-10", "2020-03", "2022-06", "2025-04"]:
    if cm in strad:
        print(f"  {cm}: straddle ₹{strad[cm]:>+10,} | BNF×3 ₹{3*BNFc.get(cm,0):>+10,.0f} | SBI×3 ₹{3*SBIc.get(cm,0):>+11,.0f}")
c.close()
