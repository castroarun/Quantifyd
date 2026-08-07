# -*- coding: utf-8 -*-
"""G3 — express a NIFTY trend signal through LEVERAGED/CONVEX bullish option structures vs cash B&H.
Gate: long only when trend-up (200DMA / 50>200), else 6% liquid. Structures: long ATM/OTM call, call debit
spread, call backspread. Monthly, NIFTY 2015-26, sweep size (lots). Capital ₹20L. Can this beat B&H?"""
import sqlite3, datetime as dt
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"
SLIP,CLpl,CAP,LOT=0.003,40,2000000,75   # CLpl=Rs40/lot/leg-txn (=Rs400 at 10 lots)
c=sqlite3.connect(DB);c.execute("PRAGMA busy_timeout=60000")
rows=c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 AND date>='2013-01-01' ORDER BY date").fetchall()
Dl=[r[0] for r in rows];O={};Cl={}
for d,o,cl in rows:O[d]=o;Cl[d]=cl
clist=[Cl[d] for d in Dl]
SMA={}
for i,d in enumerate(Dl):
    if i>=199: SMA[d]=(sum(clist[i-199:i+1])/200, sum(clist[i-49:i+1])/50)
def trend(d,kind):
    s=SMA.get(d)
    if not s: return False
    if kind=="200dma": return Cl[d]>s[0]
    if kind=="50>200": return s[1]>s[0]
    return True  # always
exps=sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2015-01-01' AND expiry_date<='2026-08-01'")})
tdays=sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2015-01-01'")})
def dte(E,d):return (dt.date.fromisoformat(E)-dt.date.fromisoformat(d)).days
def loadchain(E,d0):
    ch=defaultdict(lambda:{"CE":{},"PE":{}})
    for td,ot,K,cl,oi in c.execute("SELECT trade_date,option_type,strike,close,open_interest FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND trade_date>=? AND trade_date<=? AND close>0",(E,d0,E)):
        if oi and oi>0: ch[td][ot][K]=cl
    return ch
def pk(day,ot,off,sp):
    dd=day.get(ot,{})
    if not dd:return None
    tgt=sp*(1+off);return min(dd,key=lambda k:abs(k-tgt))
cycles=[]
for E in exps:
    before=[x for x in tdays if x<E]
    cand=[x for x in before if 26<=dte(E,x)<=32]
    if not cand:continue
    d=cand[-1]
    if d in O and d in SMA: cycles.append((E,d))
chc={}
for E,d in cycles: chc[(E,d)]=loadchain(E,d)
ws=[x for x in tdays if x>=cycles[0][1] and x in Cl];BH=Cl[ws[-1]]/O[ws[0]]-1
yrs=(dt.date.fromisoformat(cycles[-1][0])-dt.date.fromisoformat(cycles[0][1])).days/365.25
STR={"LONG_CALL_ATM":[("CE",+1,1,0.0)],"LONG_CALL_OTM2":[("CE",+1,1,0.02)],
     "CALL_DEBIT_3":[("CE",+1,1,0.0),("CE",-1,1,0.03)],
     "CALL_BACKSPREAD":[("CE",-1,1,0.005),("CE",+1,2,0.02)]}
def cycle_pnl(ch,d,E,legs,L):
    sp=O[d];QTY=L*LOT;Lg=[]
    for ot,sg,rt,off in legs:
        K=pk(ch[d],ot,off,sp)
        if K is None or K not in ch[d][ot]:return None
        Lg.append((ot,sg,rt,K,ch[d][ot][K]))
    days=[x for x in tdays if d<x<=E and x in ch]
    if not days:return None
    vE=None
    for x in reversed(days):
        v=0;ok=True
        for ot,sg,rt,K,_ in Lg:
            p=ch[x].get(ot,{}).get(K)
            if p is None:ok=False;break
            v+=sg*rt*p
        if ok:vE=v;break
    if vE is None:return None
    v0=sum(sg*rt*p for _,sg,rt,_,p in Lg)
    gross=sum(rt*abs(p) for _,_,rt,_,p in Lg)*2
    pnl=(vE-v0)*QTY-CLpl*L*sum(rt for _,_,rt,_,_ in Lg)*2-SLIP*gross*QTY
    debit=max(0.0,v0)*QTY   # capital at risk (net debit)
    return pnl,debit
def run(name,legs,sig,L):
    pls=[]
    for E,d in cycles:
        ch=chc[(E,d)]
        if d not in ch:continue
        liq_full=CAP*0.06/12
        if not trend(d,sig):
            pls.append(liq_full);continue      # flat -> full liquid
        r=cycle_pnl(ch,d,E,legs,L)
        if r is None:pls.append(liq_full);continue
        pnl,debit=r
        liq=CAP*0.06/12*max(0.0,(CAP-debit))/CAP
        pls.append(pnl+liq)
    n=len(pls);tot=sum(pls);cum=pk_=mdd=0
    for x in pls:cum+=x;pk_=max(pk_,cum);mdd=min(mdd,cum-pk_)
    dep=sum(1 for E,d in cycles if trend(d,sig))
    return n,tot,(tot/yrs),(100*(tot/yrs)/CAP),mdd,dep
print(f"G3 — NIFTY trend-gated bullish option book vs B&H. Window ~{yrs:.1f}y, CAP ₹20L.")
print(f"NIFTY B&H = ₹{round(BH*CAP):+,} ({BH*100:.0f}%) | CAGR {100*(BH/yrs):.1f}%/yr additive\n")
print(f"{'structure':>16} {'signal':>8} {'lots':>4} {'total':>13} {'CAGR':>6} {'Calmar':>7} {'maxDD':>13} {'deployed':>8}")
best=[]
for name,legs in STR.items():
    for sig in ["always","200dma","50>200"]:
        for L in [3,6,10]:
            n,tot,ann,cagr,mdd,dep=run(name,legs,sig,L)
            cal=ann/abs(mdd) if mdd else 0
            beat="  <-- beats B&H" if tot>BH*CAP else ""
            print(f"{name:>16} {sig:>8} {L:>4} {tot:>+13,.0f} {cagr:>5.1f}% {cal:>7.2f} {mdd:>+13,.0f} {dep:>3}/{len(cycles)}{beat}")
            best.append((tot,cal,name,sig,L,mdd,cagr))
    print()
print("TOP by Calmar:")
for tot,cal,name,sig,L,mdd,cagr in sorted(best,key=lambda t:-t[1])[:5]:
    print(f"  {name} {sig} {L}lot: CAGR {cagr:.1f}% Calmar {cal:.2f} DD ₹{mdd:+,.0f} total ₹{tot:+,.0f}")
c.close()
