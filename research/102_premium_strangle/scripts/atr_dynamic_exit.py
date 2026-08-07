# -*- coding: utf-8 -*-
"""Does exiting when ATR BREACHES 1.2% during the hold beat holding through? Weekly Rs20 premium
strangle, VIX>=15 + ATR<1.2% entry + premSL 3x. NIFTY 10 lots (qty 750), 2019-2026."""
import sqlite3, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP, QTY, CAP = 160, 0.003, 750, 2000000
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
rows = c.execute("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 AND date>='2018-06-01' ORDER BY date").fetchall()
dates=[r[0] for r in rows];O={};H={};L={};C={}
for d,o,h,l,cl in rows:O[d]=o;H[d]=h;L[d]=l;C[d]=cl
VIX={d:cl for d,cl in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
ATR={};tr=0
for i in range(1,len(dates)):
    d=dates[i];p=dates[i-1];t=max(H[d]-L[d],abs(H[d]-C[p]),abs(L[d]-C[p]))
    tr=tr+t if i<=14 else tr-tr/14+t
    if i>=14 and tr>0: ATR[d]=(tr/14)/O[d]
exps=sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2019-01-01'")})
tdays=sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2019-01-01'")});tset=set(tdays)
def dte(E,d):return (dt.date.fromisoformat(E)-dt.date.fromisoformat(d)).days
def nrst(d,E,ot,P):return c.execute("SELECT strike,close FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date=? AND expiry_date=? AND option_type=? AND close>0 AND open_interest>0 ORDER BY ABS(close-?) LIMIT 1",(d,E,ot,P)).fetchone()
def cser(E,K,ot,d):return {td:cl for td,cl in c.execute("SELECT trade_date,close FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND strike=? AND option_type=? AND trade_date>=? AND trade_date<=? AND close>0",(E,K,ot,d,E))}
trades=[]
for E in exps:
    before=[x for x in tdays if x<E]
    if len(before)<4: continue
    d=before[-4]
    if dte(E,d)<1 or dte(E,d)>12 or d not in O: continue
    if not (VIX.get(d) and VIX[d]>=15 and ATR.get(d) is not None and ATR[d]<0.012): continue
    ce=nrst(d,E,"CE",20);pe=nrst(d,E,"PE",20)
    if not ce or not pe: continue
    cr=ce[1]+pe[1]
    if cr<20: continue
    ecl=cser(E,ce[0],"CE",d);pcl=cser(E,pe[0],"PE",d)
    path=[(x,ecl[x]+pcl[x]) for x in [t for t in tdays if d<=t<=E] if x in ecl and x in pcl]
    if len(path)<2: continue
    trades.append({"d":d,"m":d[:7],"cr":cr,"path":path})
def book(atr_breach_exit):
    by=defaultdict(float);pl=[]
    for t in trades:
        cr=t["cr"];ex=t["path"][-1][1]
        for x,mark in t["path"]:
            if mark>=3.0*cr: ex=mark;break
            if atr_breach_exit and x!=t["d"] and ATR.get(x) is not None and ATR[x]>=0.012: ex=mark;break
        v=round((cr-ex)*QTY-COST*10-SLIP*(cr+ex)*QTY);pl.append(v);by[t["m"]]+=v
    return pl,by
def rep(lbl,pl,by):
    months=sorted({d[:7] for d in tdays})
    for wy in [False,True]:
        s=[by.get(m,0)+(10000 if wy else 0) for m in months];tot=sum(s);cum=pk=mdd=0
        for x in s:cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
        cal=(tot/(len(months)/12))/abs(mdd) if mdd else 0
        win=100*sum(1 for x in pl if x>0)/len(pl)
        tag="+liquid" if wy else "base   "
        print(f"  {lbl:>22} {tag}: total ₹{tot:>+10,} CAGR {100*(tot/(len(months)/12))/CAP:>4.1f}% Calmar {cal:>4.2f} maxDD ₹{mdd:>+10,} win {win:.0f}% n={len(pl)}")
print("ATR-breach EXIT vs HOLD-THROUGH — weekly Rs20 strangle (VIX>=15 + ATR<1.2% entry + premSL 3x), 10 lots:\n")
pl1,by1=book(False);rep("HOLD through breach",pl1,by1)
pl2,by2=book(True);rep("EXIT on ATR>=1.2%",pl2,by2)
c.close()
