#!/usr/bin/env python3
"""
System A theta decomposition — answers "is the theta capture enough?".
For each SYS-A morning, sell the OTM leg at 09:20 and hold PASSIVELY to 15:15
(no stop/target/flip). Split the P&L into:
  theta_pnl = decay with spot FROZEN  (the pure time-decay credit)
  delta_pnl = price change from the actual spot move at the later time
  total     = theta_pnl + delta_pnl - cost
Shows how thin morning theta is vs the directional swing. Pure stdlib, VPS.
"""
import sqlite3, math
from datetime import datetime, timedelta
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"; SYM="NIFTY50"
R=0.065; COST=1.0; EXPIRY_WD=3; THR_FAR=0.005; NARROW=0.0008; EXIT=(15,15)

def N(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def bs(S,K,T,s,k):
    if T<=1e-9 or s<=1e-9: return max(0.0,(S-K) if k=="C" else (K-S))
    srt=s*math.sqrt(T); d1=(math.log(S/K)+(R+0.5*s*s)*T)/srt; d2=d1-srt
    return (S*N(d1)-K*math.exp(-R*T)*N(d2)) if k=="C" else (K*math.exp(-R*T)*N(-d2)-S*N(-d1))

con=sqlite3.connect(DB)
bars=[(datetime.strptime(d,"%Y-%m-%d %H:%M:%S"),float(o),float(h),float(l),float(c))
      for d,o,h,l,c in con.execute("SELECT date,open,high,low,close FROM market_data_unified "
      "WHERE symbol=? AND timeframe='5minute' ORDER BY date",(SYM,))]
days=[(datetime.strptime(d.split()[0],"%Y-%m-%d").date(),float(h),float(l),float(c))
      for d,h,l,c in con.execute("SELECT date,high,low,close FROM market_data_unified "
      "WHERE symbol=? AND timeframe='day' ORDER BY date",(SYM,))]
con.close()
cpr={}; iv={}
for i in range(1,len(days)):
    _,ph,pl,pc=days[i-1]; P=(ph+pl+pc)/3; BC=(ph+pl)/2; TC=2*P-BC
    cpr[days[i][0]]=dict(BC=min(BC,TC),TC=max(BC,TC),R1=2*P-pl,S1=2*P-ph,width=abs(TC-BC))
    if i>=10:
        cl=[days[j][3] for j in range(i-10,i)]; lr=[math.log(cl[k+1]/cl[k]) for k in range(len(cl)-1)]
        m=sum(lr)/len(lr); v=sum((x-m)**2 for x in lr)/(len(lr)-1); iv[days[i][0]]=max(math.sqrt(v*252),0.08)
    else: iv[days[i][0]]=0.12
byday=defaultdict(list)
for i,b in enumerate(bars): byday[b[0].date()].append(i)

def expiry_dt(d):
    off=(EXPIRY_WD-d.weekday())%7; e=d+timedelta(days=off)
    return datetime(e.year,e.month,e.day,15,30),off

def run(markup):
    rows=[]
    for d in sorted(byday):
        idx=byday[d]
        if len(idx)<2 or d not in cpr or d not in iv: continue
        lv=cpr[d]; spot=bars[idx[0]][1]; fc=bars[idx[0]][4]
        if lv["width"]/spot<NARROW: continue
        if lv["BC"]<=fc<=lv["TC"]: continue
        if fc>lv["TC"]: dist=(fc-lv["TC"])/spot; side="bull"
        else: dist=(lv["BC"]-fc)/spot; side="bear"
        if dist>THR_FAR: continue
        ei=idx[1]; S0=bars[ei][1]; sig=iv[d]*markup; edt,dte=expiry_dt(d)
        # find 15:15 bar
        ex=idx[-1]
        for i in idx[1:]:
            if (bars[i][0].hour,bars[i][0].minute)>=EXIT: ex=i; break
        S1=bars[ex][4]
        T0=max((edt-bars[ei][0]).total_seconds()/(365*86400),1e-6)
        T1=max((edt-bars[ex][0]).total_seconds()/(365*86400),1e-6)
        K=round(S0*0.996/50)*50 if side=="bull" else round(S0*1.004/50)*50
        kind="P" if side=="bull" else "C"
        P0=bs(S0,K,T0,sig,kind)
        if P0<=0.5: continue
        Pth=bs(S0,K,T1,sig,kind)            # spot frozen -> pure theta
        Pac=bs(S1,K,T1,sig,kind)            # actual spot
        theta=P0-Pth; delta=Pth-Pac; tot=P0-Pac-COST
        hrs=(bars[ex][0]-bars[ei][0]).total_seconds()/3600
        rows.append(dict(dte=dte,P0=P0,theta=theta,delta=delta,tot=tot,hrs=hrs,
                         bkt=("0DTE" if dte==0 else str(dte) if dte<3 else "3+")))
    return rows

def agg(rows,label):
    n=len(rows)
    if not n: return
    mt=sum(r["theta"] for r in rows)/n; md=sum(r["delta"] for r in rows)/n
    mtot=sum(r["tot"] for r in rows)/n; mp0=sum(r["P0"] for r in rows)/n
    sd=(sum((r["delta"]-md)**2 for r in rows)/n)**0.5
    win=100*sum(1 for r in rows if r["tot"]>0)/n; hrs=sum(r["hrs"] for r in rows)/n
    print(f"  [{label}] n={n}  avg_premium={mp0:5.1f}  hold={hrs:.1f}h")
    print(f"     theta credit : {mt:+6.2f} pts  ({100*mt/mp0:4.1f}% of premium, {mt/hrs:+.2f}/hr)")
    print(f"     delta P&L    : {md:+6.2f} pts  (stdev {sd:5.1f}  <- the coin-flip swing)")
    print(f"     NET/trade    : {mtot:+6.2f} pts  win={win:.0f}%   (= theta + delta - {COST} cost)")

for mk in (1.0,1.3):
    print("="*64); print(f"PASSIVE HOLD-TO-15:15, IV = realized x {mk}"); print("="*64)
    rows=run(mk); agg(rows,"ALL")
    for b in ("0DTE","1","2","3+"): agg([r for r in rows if r["bkt"]==b],"DTE "+b)
