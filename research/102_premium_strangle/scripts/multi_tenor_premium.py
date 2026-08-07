# -*- coding: utf-8 -*-
"""Premium-strangle optimization phase 2: TENOR comparison (weekly/bi-weekly/monthly) + indicator gates
(CPR prior-day, ADX, ATR) on top of VIX>=15 + premium blow-up stop. NIFTY, 10 lots (qty 750), 2019-2026."""
import sqlite3, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP, QTY = 160, 0.003, 750
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
rows = c.execute("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 AND date>='2018-06-01' ORDER BY date").fetchall()
dates = [r[0] for r in rows]; O={};H={};L={};C={}
for d,o,h,l,cl in rows: O[d]=o;H[d]=h;L[d]=l;C[d]=cl
VIX = {d: cl for d, cl in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
# ATR14, ADX14 (Wilder)
ATR={};ADX={};pdm=0;ndm=0;tr=0;adx=0
for i in range(1,len(dates)):
    d=dates[i];p=dates[i-1]
    up=H[d]-H[p]; dn=L[p]-L[d]
    pd=up if (up>dn and up>0) else 0; nd=dn if (dn>up and dn>0) else 0
    t=max(H[d]-L[d], abs(H[d]-C[p]), abs(L[d]-C[p]))
    if i<=14: tr+=t; pdm+=pd; ndm+=nd
    else: tr=tr-tr/14+t; pdm=pdm-pdm/14+pd; ndm=ndm-ndm/14+nd
    if i>=14 and tr>0:
        ATR[d]=tr/14
        pdi=100*pdm/tr; ndi=100*ndm/tr; dx=100*abs(pdi-ndi)/(pdi+ndi) if (pdi+ndi) else 0
        adx=(adx*13+dx)/14 if adx else dx; ADX[d]=adx
# prior-day CPR width / spot
prevd={dates[i]:dates[i-1] for i in range(1,len(dates))}
def cpr(d):
    p=prevd.get(d)
    if not p: return None
    P=(H[p]+L[p]+C[p])/3; BC=(H[p]+L[p])/2; TC=2*P-BC
    return abs(TC-BC)/O[d] if d in O else None
exps = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2019-01-01'")})
tdays = sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2019-01-01'")}); tset=set(tdays)
def dte(E,d): return (dt.date.fromisoformat(E)-dt.date.fromisoformat(d)).days
def nrst(d,E,ot,P):
    return c.execute("SELECT strike,close FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date=? AND expiry_date=? AND option_type=? AND close>0 AND open_interest>0 ORDER BY ABS(close-?) LIMIT 1",(d,E,ot,P)).fetchone()
def cser(E,K,ot,d):
    return {td:cl for td,cl in c.execute("SELECT trade_date,close FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND strike=? AND option_type=? AND trade_date>=? AND trade_date<=? AND close>0",(E,K,ot,d,E))}
def build(P, entry_tb, dmax):
    trades=[]
    for E in exps:
        before=[x for x in tdays if x<E]
        if len(before)<entry_tb: continue
        d=before[-entry_tb]
        if dte(E,d)<1 or dte(E,d)>dmax or d not in O: continue
        ce=nrst(d,E,"CE",P); pe=nrst(d,E,"PE",P)
        if not ce or not pe: continue
        cr=ce[1]+pe[1]
        if cr<P: continue
        ecl=cser(E,ce[0],"CE",d); pcl=cser(E,pe[0],"PE",d)
        path=[(dte(E,x),ecl[x]+pcl[x]) for x in [t for t in tdays if d<=t<=E] if x in ecl and x in pcl]
        if len(path)<2: continue
        trades.append({"d":d,"cr":cr,"vix":VIX.get(d),"cpr":cpr(d),"adx":ADX.get(d),"atr":(ATR.get(d,0)/O[d] if d in O else None),"path":path})
    return trades
def run(tr, PREMx=3.0, VIXMIN=15, gate=None):
    pl=[]
    for t in tr:
        if VIXMIN and (t["vix"] is None or t["vix"]<VIXMIN): continue
        if gate and not gate(t): continue
        cr=t["cr"]; ex=t["path"][-1][1]
        for dx,mark in t["path"]:
            if PREMx and mark>=PREMx*cr: ex=mark; break
        pl.append(round((cr-ex)*QTY - COST*10 - SLIP*(cr+ex)*QTY))
    n=len(pl);tot=sum(pl);cum=pk=mdd=0
    for x in pl: cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    return n,tot,(tot/7.4)/abs(mdd) if mdd else 0,mdd,100*sum(1 for x in pl if x>0)/n if n else 0
print("TENOR comparison — premium strangle, VIX>=15 + premSL 3x, 10 lots (qty 750), 2019-2026\n")
print(f"{'tenor':>10} {'P':>4} | {'n':>3} {'total':>11} {'Calmar':>7} {'maxDD':>11} {'win':>4}")
best=None
for tenor,tb,dmax in [("weekly",4,12),("bi-weekly",9,20),("monthly",20,45)]:
    for P in [15,20]:
        tr=build(P,tb,dmax); n,tot,cal,mdd,win=run(tr)
        print(f"{tenor:>10} Rs{P:>2} | {n:>3} {tot:>+11,} {cal:>7.2f} {mdd:>+11,} {win:>3.0f}%")
        if n and (best is None or cal>best[0]): best=(cal,tenor,tb,dmax,P,tr)
cal,tenor,tb,dmax,P,tr=best
print(f"\nGATES on the best tenor ({tenor} Rs{P}, VIX>=15 + premSL 3x):")
gates=[("baseline",None),("+ skip CPR<0.10%",lambda t:t["cpr"] is None or t["cpr"]>=0.0010),
       ("+ ADX<20 (ranging)",lambda t:t["adx"] is None or t["adx"]<20),("+ ADX<25",lambda t:t["adx"] is None or t["adx"]<25),
       ("+ ATR<1.2%",lambda t:t["atr"] is None or t["atr"]<0.012),("+ ATR>0.8%",lambda t:t["atr"] is None or t["atr"]>0.008)]
for lbl,g in gates:
    n,tot,ca,mdd,win=run(tr,gate=g); print(f"  {lbl:>22}: n={n:>3} total ₹{tot:>+11,} Calmar {ca:>5.2f} maxDD ₹{mdd:>+11,} win {win:>3.0f}%")
c.close()
