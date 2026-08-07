# -*- coding: utf-8 -*-
"""(A) Intraday 3x stop (CE_high+PE_high conservative trigger) vs daily-close stop, both with redeploy.
(B) CAGR on realistic margin bases. Weekly Rs20 strangle, VIX>=15 + ATR<1.2%, 10 lots (qty 750), 2019-2026."""
import sqlite3, datetime as dt
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"
SLIP,QTY,CL=0.003,750,400
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
    for td,ot,K,o,h,cl in c.execute("SELECT trade_date,option_type,strike,open,high,close FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND trade_date>=? AND trade_date<=? AND close>0 AND open_interest>0",(E,d,E)):
        ch[td][ot][K]=(o if o>0 else cl,h if h>0 else cl,cl)
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
    trades.append({"d":d,"m":d[:7],"Kc":ce[0],"Kp":pe[0],"pc":ce[1],"pp":pe[1],"days":days,"ch":ch})
def cnr(cc,ot,tgt):
    dd=cc.get(ot,{})
    if not dd:return None
    K=min(dd,key=lambda k:abs(dd[k][2]-tgt));return (K,dd[K][2])
def sim(t,intraday,redeploy=3):
    ch=t["ch"];days=t["days"];ceK,ceP=t["Kc"],t["pc"];peK,peP=t["Kp"],t["pp"]
    cash=ceP+peP;gross=ceP+peP;txn=2;nr=0
    for x in days[1:]:
        cc=ch.get(x,{});C_=cc.get("CE",{}).get(ceK);P_=cc.get("PE",{}).get(peK)
        if not C_ or not P_: continue
        oc=ceP+peP;om=C_[0]+P_[0];hm=C_[1]+P_[1];clm=C_[2]+P_[2]
        fired=False;fill=None
        if intraday:
            if om>=3*oc: fired=True;fill=om            # gapped through the stop at open
            elif hm>=3*oc: fired=True;fill=3*oc         # SL-M fills at the stop level
        else:
            if clm>=3*oc: fired=True;fill=clm           # daily-close stop
        if fired:
            cash-=fill;gross+=fill;txn+=2
            if nr<redeploy and x!=days[-1]:
                nc=cnr(cc,"CE",20);npe=cnr(cc,"PE",20)
                if nc and npe: ceK,ceP=nc;peK,peP=npe;cash+=ceP+peP;gross+=ceP+peP;txn+=2;nr+=1;continue
            return round(cash*QTY-CL*txn-SLIP*gross*QTY)
    xl=days[-1];cc=ch.get(xl,{});C_=cc.get("CE",{}).get(ceK,(0,0,0));P_=cc.get("PE",{}).get(peK,(0,0,0))
    cash-=C_[2]+P_[2];gross+=C_[2]+P_[2];txn+=2
    return round(cash*QTY-CL*txn-SLIP*gross*QTY)
months=sorted({d[:7] for d in tdays})
def curve(pls):
    by=defaultdict(float)
    for t,v in zip(trades,pls):by[t["m"]]+=v
    return by
def kpi(by,cap,liq):
    s=[by.get(m,0)+(cap*0.06/12 if liq else 0) for m in months];tot=sum(s);cum=pk=mdd=0
    for x in s:cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    return tot,100*(tot/(len(months)/12))/cap,(tot/(len(months)/12))/abs(mdd) if mdd else 0,mdd
print(f"{len(trades)} trades\n")
for lbl,intr in [("daily-CLOSE 3x + redeploy",False),("INTRADAY 3x + redeploy",True)]:
    pls=[sim(t,intr) for t in trades];by=curve(pls);win=100*sum(1 for v in pls if v>0)/len(pls)
    t20,cg,cal,dd=kpi(by,2000000,False);t20y,cgy,caly,ddy=kpi(by,2000000,True)
    print(f"{lbl}:  P&L ₹{t20:+,.0f}  win {win:.0f}%  maxDD ₹{dd:+,.0f}  Calmar {cal:.2f}  (+liq Calmar {caly:.2f})")
print("\nCAGR by capital base (INTRADAY-stop book, the realistic one) — strangle-only, then +6% liquid on the base:")
pls=[sim(t,True) for t in trades];by=curve(pls)
print(f"  {'capital base':>16} | {'strangle CAGR':>13} | {'+6% liquid CAGR':>15} | {'Calmar+liq':>10}")
for cap,note in [(2000000,'parked, conservative'),(1500000,'margin+buffer'),(1200000,'~est 10-lot margin'),(1000000,'tight margin')]:
    tot,cg,cal,dd=kpi(by,cap,False);toty,cgy,caly,ddy=kpi(by,cap,True)
    print(f"  Rs{cap/100000:>5.0f}L ({note:>20}) | {cg:>12.1f}% | {cgy:>14.1f}% | {caly:>10.2f}")
c.close()
