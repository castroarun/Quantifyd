# -*- coding: utf-8 -*-
"""Validate the ATR gate on the weekly Rs20 premium strangle (walk-forward) + add 6% liquid yield.
NIFTY 10 lots (qty 750), 2019-2026. Best book: VIX>=15 + ATR<thr + premSL 3x."""
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
    if not (VIX.get(d) and VIX[d]>=15): continue
    ce=nrst(d,E,"CE",20);pe=nrst(d,E,"PE",20)
    if not ce or not pe: continue
    cr=ce[1]+pe[1]
    if cr<20: continue
    ecl=cser(E,ce[0],"CE",d);pcl=cser(E,pe[0],"PE",d)
    path=[(dte(E,x),ecl[x]+pcl[x]) for x in [t for t in tdays if d<=t<=E] if x in ecl and x in pcl]
    if len(path)<2: continue
    cr2=cr;ex=path[-1][1]
    for dx,mk in path:
        if mk>=3.0*cr2: ex=mk;break
    trades.append({"d":d,"m":d[:7],"atr":ATR.get(d),"pnl":round((cr2-ex)*QTY-COST*10-SLIP*(cr2+ex)*QTY)})
def stats(sel):
    pl=[t["pnl"] for t in sel];n=len(pl);tot=sum(pl);cum=pk=mdd=0
    for x in pl:cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    yr=(dt.date.fromisoformat(max(t["d"] for t in sel))-dt.date.fromisoformat(min(t["d"] for t in sel))).days/365.25 or 1
    return n,tot,(tot/yr)/abs(mdd) if mdd else 0,mdd,100*sum(1 for x in pl if x>0)/n if n else 0,yr
print("WALK-FORWARD — ATR gate on weekly Rs20 strangle (VIX>=15 + premSL 3x), 10 lots:\n")
for tr_lbl,te_lbl,split in [("2019-22","2023-26","2023-01"),("2023-26","2019-22r","2023-01")]:
    if split=="2023-01" and tr_lbl.startswith("2019"):
        train=[t for t in trades if t["d"]<"2023-01-01"];test=[t for t in trades if t["d"]>="2023-01-01"]
    else:
        train=[t for t in trades if t["d"]>="2023-01-01"];test=[t for t in trades if t["d"]<"2023-01-01"]
    best=None
    for thr in [0.010,0.012,0.015,0.020,None]:
        sub=[t for t in train if thr is None or (t["atr"] is None or t["atr"]<thr)]
        if len(sub)<8: continue
        cal=stats(sub)[2]
        if best is None or cal>best[0]: best=(cal,thr)
    thr=best[1];tst=[t for t in test if thr is None or (t["atr"] is None or t["atr"]<thr)]
    n,tot,cal,mdd,win,yr=stats(tst)
    print(f"  train {tr_lbl} picks ATR<{thr} -> OOS {te_lbl}: n={n} Calmar {cal:.2f} total ₹{tot:+,} win {win:.0f}%")
# full-sample + liquid yield
print("\nFULL-SAMPLE ATR<1.2% book + 6% liquid yield (10 lots, Rs20L capital):")
sel=[t for t in trades if t["atr"] is None or t["atr"]<0.012]
by=defaultdict(float)
for t in sel: by[t["m"]]+=t["pnl"]
months=sorted({d[:7] for d in tdays})
def mm_stats(withyield):
    pl=[by.get(m,0)+(10000 if withyield else 0) for m in months]  # Rs10k/mo = 6% on Rs20L
    tot=sum(pl);cum=pk=mdd=0
    for x in pl:cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    yr=len(months)/12
    return tot,(tot/yr)/abs(mdd) if mdd else 0,mdd,100*(tot/yr)/CAP
for wy,lbl in [(False,"strangle only"),(True,"+ 6% liquid")]:
    tot,cal,mdd,cagr=mm_stats(wy)
    print(f"  {lbl:>14}: total ₹{tot:+,} CAGR {cagr:.1f}% Calmar {cal:.2f} maxDD ₹{mdd:+,} ({len(sel)} trades)")
c.close()
