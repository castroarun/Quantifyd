# -*- coding: utf-8 -*-
"""G1 — convex/defined-risk option STRUCTURE bake-off on NIFTY (naked, hold-to-expiry), weekly + monthly.
Families: short strangle, iron fly, put/call backspread, front put ratio, Batman (double fly), broken-wing put fly.
Ranked vs NIFTY buy-&-hold. 10 lots (qty 750), EOD bhav, net of costs. G1 finds which families have a pulse."""
import sqlite3, datetime as dt, csv, os
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"
SLIP,QTY,CL=0.003,750,400
OUT="/home/arun/quantifyd/research/103_convex_structures/results/g1_nifty.csv"
c=sqlite3.connect(DB);c.execute("PRAGMA busy_timeout=60000")
spot={d:o for d,o in c.execute("SELECT date,open FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND open>0")}
exps=sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2015-01-01'")})
tdays=sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2015-01-01'")})
tset=set(tdays)
def dte(E,d):return (dt.date.fromisoformat(E)-dt.date.fromisoformat(d)).days
def loadchain(E,d0):
    ch=defaultdict(lambda:{"CE":{},"PE":{}})
    for td,ot,K,cl,oi in c.execute("SELECT trade_date,option_type,strike,close,open_interest FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND trade_date>=? AND trade_date<=? AND close>0",(E,d0,E)):
        if oi and oi>0: ch[td][ot][K]=cl
    return ch
def pick(day,ot,kind,arg,sp):
    dd=day.get(ot,{})
    if not dd: return None
    if kind=="atm": tgt=sp
    elif kind=="off": tgt=sp*(1+arg)
    elif kind=="prem":
        K=min(dd,key=lambda k:abs(dd[k]-arg));return K
    K=min(dd,key=lambda k:abs(k-tgt));return K
# structures: list of legs (otype, sign(+long/-short), ratio, (kind,arg))
STRUCTS={
 "SHORT_STRANGLE":[("PE",-1,1,("prem",20)),("CE",-1,1,("prem",20))],
 "IRON_FLY":     [("CE",-1,1,("atm",0)),("PE",-1,1,("atm",0)),("CE",+1,1,("off",0.02)),("PE",+1,1,("off",-0.02))],
 "PUT_BACKSPREAD":[("PE",-1,1,("off",-0.005)),("PE",+1,2,("off",-0.02))],
 "CALL_BACKSPREAD":[("CE",-1,1,("off",0.005)),("CE",+1,2,("off",0.02))],
 "PUT_RATIO_FRONT":[("PE",+1,1,("off",-0.01)),("PE",-1,2,("off",-0.025))],
 "BATMAN":       [("CE",+1,1,("off",0.01)),("CE",-1,2,("off",0.02)),("CE",+1,1,("off",0.03)),
                  ("PE",+1,1,("off",-0.01)),("PE",-1,2,("off",-0.02)),("PE",+1,1,("off",-0.03))],
 "BWB_PUT":      [("PE",+1,1,("off",-0.01)),("PE",-1,2,("off",-0.02)),("PE",+1,1,("off",-0.025))],
}
def build_legs(day,legs,sp):
    out=[]
    for ot,sg,rt,(kind,arg) in legs:
        K=pick(day,ot,kind,arg,sp)
        if K is None or K not in day[ot]: return None
        out.append((ot,sg,rt,K,day[ot][K]))
    return out
def value(chday,legs):
    v=0
    for ot,sg,rt,K,_ in legs:
        p=chday.get(ot,{}).get(K)
        if p is None: return None
        v+=sg*rt*p
    return v
def run_tenor(tenor, exps_use, tb_or_dte):
    results={}
    for name,legs in STRUCTS.items():
        pls=[];entrymonth=[]
        for E in exps_use:
            before=[x for x in tdays if x<E]
            if len(before)<5: continue
            if tenor=="weekly":
                d=before[-4]
                if dte(E,d)<1 or dte(E,d)>12: continue
            else: # monthly: ~28 cal days before
                cand=[x for x in before if dte(E,x)>=26 and dte(E,x)<=32]
                if not cand: continue
                d=cand[-1]
            if d not in spot: continue
            ch=loadchain(E,d)
            if d not in ch: continue
            sp=spot[d]
            L=build_legs(ch[d],legs,sp)
            if not L: continue
            v0=value(ch[d],L)
            days=[x for x in tdays if d<x<=E and x in ch]
            if not days: continue
            vE=None
            for x in reversed(days):
                vv=value(ch[x],L)
                if vv is not None: vE=vv;break
            if vE is None: continue
            gross=sum(rt*abs(p) for _,_,rt,_,p in L)*2
            pnl=round((vE-v0)*QTY - CL*sum(rt for _,_,rt,_,_ in L)*2 - SLIP*gross*QTY)
            pls.append(pnl);entrymonth.append(d[:7])
        if not pls: continue
        cum=pk=mdd=0
        for x in pls: cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
        results[name]=dict(n=len(pls),total=sum(pls),avg=sum(pls)/len(pls),
            win=100*sum(1 for x in pls if x>0)/len(pls),worst=min(pls),mdd=mdd)
    return results
def bh(exps_use):
    ds=[x for x in tdays if x>=(exps_use[0])][:1] or [tdays[0]]
    d0=min(spot);d1=max(spot)
    # window = first entry to last expiry
    start=[x for x in tdays if x in spot][0];end=[x for x in tdays if x in spot][-1]
    return start,end,(spot[end]/spot[start]-1)
print("G1 — NIFTY structure bake-off (10 lots=qty 750, naked hold-to-expiry, net of costs)\n")
os.makedirs(os.path.dirname(OUT),exist_ok=True)
w=csv.writer(open(OUT,"w",newline=""));w.writerow(["tenor","structure","n","total","avg_per_trade","win_pct","worst_trade","equity_maxDD"])
for tenor,exps_use in [("weekly",[e for e in exps if e>='2019-02-01']),("monthly",[e for e in exps if e>='2015-01-01'])]:
    res=run_tenor(tenor,exps_use,None)
    yrs=(dt.date.fromisoformat(exps_use[-1])-dt.date.fromisoformat(exps_use[0])).days/365.25
    print(f"=== {tenor.upper()} ({exps_use[0][:7]}→{exps_use[-1][:7]}, ~{yrs:.1f}y) ===")
    print(f"{'structure':>17} {'n':>4} {'total':>12} {'avg/trade':>10} {'win':>5} {'worst':>11} {'eqMaxDD':>11}")
    for name in sorted(res,key=lambda k:-res[k]['total']):
        r=res[name];print(f"{name:>17} {r['n']:>4} {r['total']:>+12,} {r['avg']:>+10,.0f} {r['win']:>4.0f}% {r['worst']:>+11,} {r['mdd']:>+11,}")
        w.writerow([tenor,name,r['n'],r['total'],round(r['avg']),round(r['win'],1),r['worst'],r['mdd']])
    # NIFTY B&H over same window on Rs20L
    ws=[x for x in tdays if x>=exps_use[0] and x in spot];bhret=spot[ws[-1]]/spot[ws[0]]-1
    print(f"{'NIFTY B&H (Rs20L)':>17} {'':>4} {round(bhret*2000000):>+12,} {'':>10} {'':>5}  ({bhret*100:.0f}% over {yrs:.1f}y)\n")
c.close();print("wrote",OUT)
