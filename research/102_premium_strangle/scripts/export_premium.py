# -*- coding: utf-8 -*-
"""Export the premium-strangle study to JSON for the /app tearsheet. 4 books (progression):
A weekly Rs20 all weeks | B +VIX>=15 | C +VIX>=15+ATR<1.2% | D final (intraday 3x stop + redeploy).
NIFTY 10 lots (qty 750), 2019-2026, net of costs, on Rs20L. 6% liquid-yield curves too."""
import sqlite3, datetime as dt, json
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"
SLIP,QTY,CL,CAP=0.003,750,400,2000000
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
    ce=nrst(d,E,"CE",20);pe=nrst(d,E,"PE",20)
    if not ce or not pe or ce[1]+pe[1]<20: continue
    ch=loadchain(E,d);days=[x for x in tdays if d<=x<=E]
    if len(days)<2: continue
    trades.append({"d":d,"m":d[:7],"E":E,"Kc":ce[0],"Kp":pe[0],"pc":ce[1],"pp":pe[1],
                   "vix":round(VIX.get(d,0),1),"atr":ATR.get(d),"days":days,"ch":ch})
def cnr(cc,ot,tgt):
    dd=cc.get(ot,{})
    if not dd:return None
    K=min(dd,key=lambda k:abs(dd[k][2]-tgt));return (K,dd[K][2])
def sim(t,intraday,redeploy,detail=False):
    ch=t["ch"];days=t["days"];ceK,ceP=t["Kc"],t["pc"];peK,peP=t["Kp"],t["pp"]
    cash=ceP+peP;gross=ceP+peP;txn=2;nr=0;reason="expiry";exd=days[-1]
    for x in days[1:]:
        cc=ch.get(x,{});C_=cc.get("CE",{}).get(ceK);P_=cc.get("PE",{}).get(peK)
        if not C_ or not P_: continue
        oc=ceP+peP;om=C_[0]+P_[0];hm=C_[1]+P_[1];clm=C_[2]+P_[2]
        fired=False;fill=None
        if intraday:
            if om>=3*oc: fired=True;fill=om
            elif hm>=3*oc: fired=True;fill=3*oc
        else:
            if clm>=3*oc: fired=True;fill=clm
        if fired:
            cash-=fill;gross+=fill;txn+=2
            if nr<redeploy and x!=days[-1]:
                nc=cnr(cc,"CE",20);npe=cnr(cc,"PE",20)
                if nc and npe: ceK,ceP=nc;peK,peP=npe;cash+=ceP+peP;gross+=ceP+peP;txn+=2;nr+=1;continue
            reason="3x stop"+(f" (+{nr} redeploy)" if nr else "");exd=x
            pnl=round(cash*QTY-CL*txn-SLIP*gross*QTY)
            return (pnl,dict(exd=x,exCE=round(C_[2],1),exPE=round(P_[2],1),reason=reason,nr=nr)) if detail else pnl
    xl=days[-1];cc=ch.get(xl,{});C_=cc.get("CE",{}).get(ceK,(0,0,0));P_=cc.get("PE",{}).get(peK,(0,0,0))
    cash-=C_[2]+P_[2];gross+=C_[2]+P_[2];txn+=2
    reason="expiry"+(f" (+{nr} redeploy)" if nr else "");pnl=round(cash*QTY-CL*txn-SLIP*gross*QTY)
    return (pnl,dict(exd=xl,exCE=round(C_[2],1),exPE=round(P_[2],1),reason=reason,nr=nr)) if detail else pnl
months=sorted({d[:7] for d in tdays})
def curve_kpi(pls_by_month,liq):
    cum=0;cur=[];pk=mdd=0
    for m in months:
        cum+=pls_by_month.get(m,0)+(CAP*0.06/12 if liq else 0);cur.append([m,round(cum)])
        pk=max(pk,cum);mdd=min(mdd,cum-pk)
    tot=cum;yr=len(months)/12
    return cur,dict(total=round(tot),cagr=round(100*(tot/yr)/CAP,1),calmar=round((tot/yr)/abs(mdd),2) if mdd else 0,mdd=round(mdd))
BOOKS=[("Weekly ₹20 · all weeks",lambda t:True,False,0),
       ("+ VIX≥15",lambda t:t["vix"]>=15,False,0),
       ("+ VIX≥15 + ATR<1.2%",lambda t:t["vix"]>=15 and t["atr"] is not None and t["atr"]<0.012,False,0),
       ("Final · intraday stop + redeploy",lambda t:t["vix"]>=15 and t["atr"] is not None and t["atr"]<0.012,True,3)]
out={"months":months,"curves":{},"curvesY":{},"kpi":{},"kpiY":{},"trades":[]}
for name,flt,intr,rd in BOOKS:
    sel=[t for t in trades if flt(t)];by=defaultdict(float);wins=0
    for t in sel:
        v=sim(t,intr,rd);by[t["m"]]+=v;wins+=1 if v>0 else 0
    cur,kpi=curve_kpi(by,False);curY,kpiY=curve_kpi(by,True)
    kpi["win"]=round(100*wins/len(sel));kpiY["win"]=kpi["win"]
    out["curves"][name]=cur;out["curvesY"][name]=curY;out["kpi"][name]=kpi;out["kpiY"][name]=kpiY
    print(f"{name:>34}: n={len(sel):>3} total ₹{kpi['total']:>+9,} CAGR {kpi['cagr']:>4.1f}% Cal {kpi['calmar']:.2f} DD ₹{kpi['mdd']:>+9,} win {kpi['win']}% | +liq CAGR {kpiY['cagr']:.1f}% Cal {kpiY['calmar']:.2f}")
# blotter for final book
final=[t for t in trades if t["vix"]>=15 and t["atr"] is not None and t["atr"]<0.012]
for t in final:
    v,dd=sim(t,True,3,detail=True)
    out["trades"].append(dict(m=t["m"],entry=t["d"],exit=dd["exd"],vix=t["vix"],atr=round(t["atr"]*100,2),
        Kc=int(t["Kc"]),Kp=int(t["Kp"]),ceIn=round(t["pc"],1),peIn=round(t["pp"],1),
        ceOut=dd["exCE"],peOut=dd["exPE"],reason=dd["reason"],nr=dd["nr"],total=v))
json.dump(out,open("/home/arun/quantifyd/research/102_premium_strangle/results/premium_curves.json","w"))
print(f"\nwrote premium_curves.json — {len(out['trades'])} blotter rows, {len(months)} months")
c.close()
