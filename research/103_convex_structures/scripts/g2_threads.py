# -*- coding: utf-8 -*-
"""G2 — (A) signal-TIMED convexity: Batman/backspreads/long-strangle deployed only on low-IV / ATR-contraction /
CPR-compression gates.  (B) credit + tail ADJUSTMENTS: front put ratio & short strangle with roll-tested-wing /
convert-to-fly / combined-stop.  NIFTY monthly 2015-26, 10 lots (qty 750), EOD, net of costs. Ranked vs B&H."""
import sqlite3, datetime as dt
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"
SLIP,QTY,CL=0.003,750,400
c=sqlite3.connect(DB);c.execute("PRAGMA busy_timeout=60000")
rows=c.execute("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 AND date>='2014-06-01' ORDER BY date").fetchall()
D=[r[0] for r in rows];O={};H={};L={};Cl={}
for d,o,h,l,cl in rows:O[d]=o;H[d]=h;L[d]=l;Cl[d]=cl
idx={d:i for i,d in enumerate(D)}
VIX={d:v for d,v in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
ATR={};tr=0
for i in range(1,len(D)):
    d=D[i];p=D[i-1];t=max(H[d]-L[d],abs(H[d]-Cl[p]),abs(L[d]-Cl[p]))
    tr=tr+t if i<=14 else tr-tr/14+t
    if i>=14 and tr>0: ATR[d]=tr/14
def atr_pct(d):return ATR[d]/O[d] if d in ATR and d in O else None
def atr_contract(d):
    i=idx.get(d)
    if i is None or i<20 or d not in ATR: return False
    d20=D[i-20]
    return d20 in ATR and ATR[d]<0.85*ATR[d20]
def cprw(d):
    i=idx.get(d)
    if i is None or i<1: return None
    p=D[i-1];P=(H[p]+L[p]+Cl[p])/3;BC=(H[p]+L[p])/2;TC=2*P-BC
    return abs(TC-BC)/O[d] if d in O else None
exps=sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2015-01-01' AND expiry_date<='2026-08-01'")})
tdays=sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2015-01-01'")})
def dte(E,d):return (dt.date.fromisoformat(E)-dt.date.fromisoformat(d)).days
def loadchain(E,d0):
    ch=defaultdict(lambda:{"CE":{},"PE":{}})
    for td,ot,K,cl,oi in c.execute("SELECT trade_date,option_type,strike,close,open_interest FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND trade_date>=? AND trade_date<=? AND close>0",(E,d0,E)):
        if oi and oi>0: ch[td][ot][K]=cl
    return ch
def pk(day,ot,kind,arg,sp):
    dd=day.get(ot,{})
    if not dd:return None
    if kind=="prem":return min(dd,key=lambda k:abs(dd[k]-arg))
    tgt=sp if kind=="atm" else sp*(1+arg)
    return min(dd,key=lambda k:abs(k-tgt))
def stat(pls):
    n=len(pls);tot=sum(pls);cum=pk_=mdd=0
    for x in pls:cum+=x;pk_=max(pk_,cum);mdd=min(mdd,cum-pk_)
    return n,tot,(sum(pls)/n if n else 0),(100*sum(1 for x in pls if x>0)/n if n else 0),mdd
# monthly entry days
cycles=[]
for E in exps:
    before=[x for x in tdays if x<E]
    cand=[x for x in before if 26<=dte(E,x)<=32]
    if not cand: continue
    d=cand[-1]
    if d not in O: continue
    cycles.append((E,d))
BH=None
ws=[x for x in tdays if x>=cycles[0][1] and x in O];BH=Cl[ws[-1]]/O[ws[0]]-1
CONVEX={
 "LONG_STRANGLE":[("PE",+1,1,("off",-0.015)),("CE",+1,1,("off",0.015))],
 "PUT_BACKSPREAD":[("PE",-1,1,("off",-0.005)),("PE",+1,2,("off",-0.02))],
 "CALL_BACKSPREAD":[("CE",-1,1,("off",0.005)),("CE",+1,2,("off",0.02))],
 "BATMAN":[("CE",+1,1,("off",0.01)),("CE",-1,2,("off",0.02)),("CE",+1,1,("off",0.03)),
           ("PE",+1,1,("off",-0.01)),("PE",-1,2,("off",-0.02)),("PE",+1,1,("off",-0.03))],
}
GATES={"always":lambda d:True,"VIX<14":lambda d:VIX.get(d,99)<14,
 "ATR-contract":lambda d:atr_contract(d),"CPR<0.10%":lambda d:(cprw(d) or 9)<0.0010,
 "VIX<14 & contract":lambda d:VIX.get(d,99)<14 and atr_contract(d)}
def naked_pnl(ch,d,E,legs):
    sp=O[d];L_=[]
    for ot,sg,rt,(k,a) in legs:
        K=pk(ch[d],ot,k,a,sp)
        if K is None or K not in ch[d][ot]:return None
        L_.append((ot,sg,rt,K,ch[d][ot][K]))
    days=[x for x in tdays if d<x<=E and x in ch]
    if not days:return None
    vE=None
    for x in reversed(days):
        v=0;ok=True
        for ot,sg,rt,K,_ in L_:
            p=ch[x].get(ot,{}).get(K)
            if p is None:ok=False;break
            v+=sg*rt*p
        if ok:vE=v;break
    if vE is None:return None
    v0=sum(sg*rt*p for _,sg,rt,_,p in L_)
    gross=sum(rt*abs(p) for _,_,rt,_,p in L_)*2
    return round((vE-v0)*QTY-CL*sum(rt for _,_,rt,_,_ in L_)*2-SLIP*gross*QTY)
print("G2 — NIFTY monthly, 10 lots. B&H over window = ₹{:+,} ({:.0f}%)\n".format(round(BH*2000000),BH*100))
print("=== THREAD A — signal-timed convexity (deploy only on gate; skipped cycles = ₹0) ===")
print(f"{'structure':>16} {'gate':>18} {'n':>4} {'total':>12} {'avg':>9} {'win':>5} {'eqDD':>12}")
chcache={}
for E,d in cycles: chcache[(E,d)]=loadchain(E,d)
for name,legs in CONVEX.items():
    for gname,gf in GATES.items():
        pls=[]
        for E,d in cycles:
            if not gf(d):continue
            ch=chcache[(E,d)]
            if d not in ch:continue
            v=naked_pnl(ch,d,E,legs)
            if v is not None:pls.append(v)
        if len(pls)<6:continue
        n,tot,avg,win,mdd=stat(pls)
        print(f"{name:>16} {gname:>18} {n:>4} {tot:>+12,} {avg:>+9,.0f} {win:>4.0f}% {mdd:>+12,}")
    print()
# THREAD B — credit + adjustments (cash-flow sim)
def simB(ch,d,E,base,adj):
    sp=O[d];days=[x for x in tdays if d<=x<=E and x in ch]
    legs=[]  # dict ot,sg,rt,K,ep
    cash=0.0;gross=0.0;txn=0
    for ot,sg,rt,(k,a) in base:
        K=pk(ch[d],ot,k,a,sp)
        if K is None or K not in ch[d][ot]:return None
        p=ch[d][ot][K];cash-=sg*rt*p;gross+=rt*p;txn+=rt;legs.append([ot,sg,rt,K,p])
    adjusted=False
    for x in days[1:]:
        cc=ch[x]
        # mark shorts to detect tested wing (put side)
        if not adjusted and adj in ("convert_fly","roll_out"):
            for lg in legs:
                ot,sg,rt,K,ep=lg
                if ot=="PE" and sg<0:                      # the naked short put(s)
                    m=cc.get("PE",{}).get(K)
                    if m and m>=2.0*ep:                    # tested wing (short doubled)
                        puts=cc.get("PE",{})
                        if adj=="convert_fly":             # buy protective puts below K -> defined-risk fly
                            Kp=min((k for k in puts if k<K),key=lambda k:abs(k-(K-sp*0.015)),default=None)
                            if Kp is not None:
                                pp=puts[Kp];cash-=(+1)*rt*pp;gross+=rt*pp;txn+=rt
                                legs.append(["PE",+1,rt,Kp,pp])
                        else:                              # roll_out: buy back tested short, re-sell further OTM
                            cash+=sg*rt*m;gross+=rt*m;txn+=rt          # close tested short (sg=-1 -> pay m)
                            Kn=min((k for k in puts if k<K),key=lambda k:abs(k-(K-sp*0.02)),default=None)
                            if Kn is not None:
                                pn=puts[Kn];cash-=sg*rt*pn;gross+=rt*pn;txn+=rt   # sell new short (sg=-1 -> receive pn)
                                lg[3]=Kn;lg[4]=pn
                        adjusted=True
                        break
        if adj=="stop3x":
            v=sum(sg*rt*cc.get(ot,{}).get(K,ep) for ot,sg,rt,K,ep in legs)
            v0=sum(sg*rt*ep for ot,sg,rt,K,ep in legs)
            # for a credit structure v0<0 (net short); loss if (v - v0) very negative
            if (v-v0)<= -3*abs(v0) and abs(v0)>0:
                for ot,sg,rt,K,ep in legs:
                    p=cc.get(ot,{}).get(K,ep);cash+=sg*rt*p;gross+=rt*p;txn+=rt
                return round(cash*QTY-CL*txn*2-SLIP*gross*QTY)
    xl=days[-1];cc=ch[xl]
    for ot,sg,rt,K,ep in legs:
        p=cc.get(ot,{}).get(K,ep);cash+=sg*rt*p;gross+=rt*p;txn+=rt
    return round(cash*QTY-CL*txn*2-SLIP*gross*QTY)
BASES={"FRONT_PUT_RATIO":[("PE",+1,1,("off",-0.01)),("PE",-1,2,("off",-0.025))],
       "SHORT_STRANGLE":[("PE",-1,1,("prem",20)),("CE",-1,1,("prem",20))]}
print("=== THREAD B — credit + tail adjustments ===")
print(f"{'structure':>16} {'adjustment':>14} {'n':>4} {'total':>12} {'avg':>9} {'win':>5} {'eqDD':>12}")
for name,base in BASES.items():
    for adj in ["none","stop3x","convert_fly","roll_out"]:
        pls=[]
        for E,d in cycles:
            ch=chcache[(E,d)]
            if d not in ch:continue
            v=simB(ch,d,E,base,adj)
            if v is not None:pls.append(v)
        if len(pls)<6:continue
        n,tot,avg,win,mdd=stat(pls)
        print(f"{name:>16} {adj:>14} {n:>4} {tot:>+12,} {avg:>+9,.0f} {win:>4.0f}% {mdd:>+12,}")
    print()
c.close()
