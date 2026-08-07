# -*- coding: utf-8 -*-
"""Adjustment/exit method bake-off on the validated weekly Rs20 strangle (VIX>=15 + ATR<1.2%).
Methods: hold+premSL3x (baseline), underlying-move stop, re-deploy after SL, roll-recenter on leg
double, roll-untested-in on leg double. NIFTY 10 lots (qty 750), 2019-2026. Clean cash-flow P&L."""
import sqlite3, datetime as dt
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"
SLIP,QTY,CAP,CL=0.003,750,2000000,400   # CL=Rs400 per leg transaction (open OR close) => baseline 4txn=Rs1600
c=sqlite3.connect(DB);c.execute("PRAGMA busy_timeout=30000")
rows=c.execute("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 AND date>='2018-06-01' ORDER BY date").fetchall()
dates=[r[0] for r in rows];O={};H={};L={};Cc={}
for d,o,h,l,cl in rows:O[d]=o;H[d]=h;L[d]=l;Cc[d]=cl
VIX={d:v for d,v in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
ATR={};tr=0
for i in range(1,len(dates)):
    d=dates[i];p=dates[i-1];t=max(H[d]-L[d],abs(H[d]-Cc[p]),abs(L[d]-Cc[p]))
    tr=tr+t if i<=14 else tr-tr/14+t
    if i>=14 and tr>0: ATR[d]=(tr/14)/O[d]
exps=sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2019-01-01'")})
tdays=sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2019-01-01'")})
def dte(E,d):return (dt.date.fromisoformat(E)-dt.date.fromisoformat(d)).days
def nrst(d,E,ot,P):return c.execute("SELECT strike,close FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date=? AND expiry_date=? AND option_type=? AND close>0 AND open_interest>0 ORDER BY ABS(close-?) LIMIT 1",(d,E,ot,P)).fetchone()
def loadchain(E,d):
    ch=defaultdict(lambda:{"CE":{},"PE":{}})
    for td,ot,K,cl in c.execute("SELECT trade_date,option_type,strike,close FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND trade_date>=? AND trade_date<=? AND close>0 AND open_interest>0",(E,d,E)):
        ch[td][ot][K]=cl
    return ch
trades=[]
for E in exps:
    before=[x for x in tdays if x<E]
    if len(before)<4: continue
    d=before[-4]
    if dte(E,d)<1 or dte(E,d)>12 or d not in O: continue
    if not (VIX.get(d) and VIX[d]>=15 and ATR.get(d) is not None and ATR[d]<0.012): continue
    ce=nrst(d,E,"CE",20);pe=nrst(d,E,"PE",20)
    if not ce or not pe or ce[1]+pe[1]<20: continue
    ch=loadchain(E,d);days=[x for x in tdays if d<=x<=E]
    if len(days)<2: continue
    trades.append({"d":d,"m":d[:7],"E":E,"Kc":ce[0],"Kp":pe[0],"pc":ce[1],"pp":pe[1],"days":days,"ch":ch})
print(f"{len(trades)} trades in book\n")
def cnr(cc,ot,tgt):
    dd=cc.get(ot,{})
    if not dd: return None
    K=min(dd,key=lambda k:abs(dd[k]-tgt));return (K,dd[K])
def sim(t,method,param):
    ch=t["ch"];s0=Cc[t["d"]];days=t["days"]
    ceK,ceP=t["Kc"],t["pc"];peK,peP=t["Kp"],t["pp"]
    cash=ceP+peP;gross=ceP+peP;txn=2;nr=0
    for x in days[1:]:
        cc=ch.get(x,{});mc=cc.get("CE",{}).get(ceK);mp=cc.get("PE",{}).get(peK)
        if mc is None or mp is None: continue
        oc=ceP+peP
        if method=="umove" and abs(Cc[x]/s0-1)>=param:
            cash-=mc+mp;gross+=mc+mp;txn+=2;return pnl(cash,gross,txn)
        if method in("baseline","umove","redeploy") and mc+mp>=3.0*oc:
            cash-=mc+mp;gross+=mc+mp;txn+=2
            if method=="redeploy" and nr<param and x!=days[-1]:
                nc=cnr(cc,"CE",20);npe=cnr(cc,"PE",20)
                if nc and npe:
                    ceK,ceP=nc;peK,peP=npe;cash+=ceP+peP;gross+=ceP+peP;txn+=2;nr+=1;continue
            return pnl(cash,gross,txn)
        if method=="recenter" and (mc>=2*ceP or mp>=2*peP) and nr<param and x!=days[-1]:
            cash-=mc+mp;gross+=mc+mp;txn+=2
            nc=cnr(cc,"CE",20);npe=cnr(cc,"PE",20)
            if nc and npe:
                ceK,ceP=nc;peK,peP=npe;cash+=ceP+peP;gross+=ceP+peP;txn+=2;nr+=1;continue
            return pnl(cash,gross,txn)
        if method=="rollin" and nr<param and x!=days[-1]:
            if mc>=2*ceP:                       # CE tested (spot up) -> roll PE up to match CE mark
                cash-=mp;gross+=mp;txn+=1;np_=cnr(cc,"PE",mc)
                if np_:peK,peP=np_;cash+=peP;gross+=peP;txn+=1;nr+=1;continue
            if mp>=2*peP:                        # PE tested (spot down) -> roll CE down to match PE mark
                cash-=mc;gross+=mc;txn+=1;np_=cnr(cc,"CE",mp)
                if np_:ceK,ceP=np_;cash+=ceP;gross+=ceP;txn+=1;nr+=1;continue
    xl=days[-1];cc=ch.get(xl,{});mc=cc.get("CE",{}).get(ceK,0);mp=cc.get("PE",{}).get(peK,0)
    cash-=mc+mp;gross+=mc+mp;txn+=2;return pnl(cash,gross,txn)
def pnl(cash,gross,txn):return round(cash*QTY-CL*txn-SLIP*gross*QTY)
months=sorted({d[:7] for d in tdays})
def rep(lbl,pls):
    by=defaultdict(float)
    for t,v in zip(trades,pls): by[t["m"]]+=v
    win=100*sum(1 for v in pls if v>0)/len(pls)
    out=[]
    for wy in [False,True]:
        s=[by.get(m,0)+(10000 if wy else 0) for m in months];tot=sum(s);cum=pk=mdd=0
        for x in s:cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
        cal=(tot/(len(months)/12))/abs(mdd) if mdd else 0
        out.append((tot,100*(tot/(len(months)/12))/CAP,cal,mdd))
    b,y=out[0],out[1]
    print(f"  {lbl:>26}: base ₹{b[0]:>+9,.0f} Cal {b[2]:>4.2f} DD ₹{b[3]:>+9,.0f} | +liq CAGR {y[1]:>4.1f}% Cal {y[2]:>4.2f} DD ₹{y[3]:>+9,.0f} win {win:.0f}%")
print("ADJUSTMENT / EXIT bake-off — weekly Rs20 strangle (VIX>=15 + ATR<1.2%), 10 lots:\n")
rep("baseline hold+premSL3x",[sim(t,"baseline",0) for t in trades])
for x in [0.010,0.015,0.020]: rep(f"underlying-move stop {x*100:.1f}%",[sim(t,"umove",x) for t in trades])
rep("re-deploy after SL (x3)",[sim(t,"redeploy",3) for t in trades])
rep("roll-recenter on 2x (x3)",[sim(t,"recenter",3) for t in trades])
rep("roll-untested-IN on 2x (x3)",[sim(t,"rollin",3) for t in trades])
c.close()
