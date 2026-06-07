#!/usr/bin/env python3
"""
REC 15-min Supertrend deep-dive: COST SENSITIVITY (make-or-break) + per-year +
long/short split, for the robust OOS plateau configs. Pure stdlib, VPS.
"""
import sqlite3, math
from datetime import datetime
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"; SYM="RECLTD"; STEP=3  # 15min=3x5min
CONFIGS=[(7,3.0,"both"),(21,3.0,"both"),(14,2.5,"both"),(10,3.0,"short")]
COSTS=[0.0005,0.0010,0.0015,0.0020,0.0030]

con=sqlite3.connect(DB)
r5=[(datetime.strptime(d,"%Y-%m-%d %H:%M:%S"),float(o),float(h),float(l),float(c))
    for d,o,h,l,c in con.execute("SELECT date,open,high,low,close FROM market_data_unified "
    "WHERE symbol=? AND timeframe='5minute' ORDER BY date",(SYM,))]
con.close()
byd=defaultdict(list)
for x in r5: byd[x[0].date()].append(x)
bars=[]
for d in sorted(byd):
    day=byd[d]
    for k in range(0,len(day),STEP):
        g=day[k:k+STEP]
        if g: bars.append((g[0][0],g[0][1],max(z[2] for z in g),min(z[3] for z in g),g[-1][4]))
n=len(bars); closes=[b[4] for b in bars]; dts=[b[0].date() for b in bars]
TEST_LO=dts[n//2]
ANN=252*(375/15)

def st(period,mult):
    m=len(bars); tr=[0.0]*m
    for i in range(m):
        _,o,h,l,c=bars[i]; tr[i]=h-l if i==0 else max(h-l,abs(h-bars[i-1][4]),abs(l-bars[i-1][4]))
    atr=[0.0]*m; atr[period-1]=sum(tr[:period])/period
    for i in range(period,m): atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    d=[0]*m; fub=flb=0.0
    for i in range(m):
        _,o,h,l,c=bars[i]; mid=(h+l)/2
        if i<period: d[i]=1; continue
        bub=mid+mult*atr[i]; blb=mid-mult*atr[i]
        fub=bub if (bub<fub or bars[i-1][4]>fub or fub==0) else fub
        flb=blb if (blb>flb or bars[i-1][4]<flb or flb==0) else flb
        prev=d[i-1] or 1
        d[i]=(-1 if c<flb else 1) if prev==1 else (1 if c>fub else -1)
    return d

def pos_of(d,mode):
    out=[0]*n
    for i in range(n):
        if mode=="both": out[i]=d[i]
        elif mode=="long": out[i]=1 if d[i]==1 else 0
        elif mode=="short": out[i]=-1 if d[i]==-1 else 0
    return out

def metric(pos,cost,lo=None,hi=None):
    eq=1.0;peak=1.0;dd=0.0;daily=[];flips=0;pp=0
    for i in range(1,n):
        if lo and dts[i]<lo: pp=pos[i-1]; continue
        if hi and dts[i]>=hi: break
        ret=closes[i]/closes[i-1]-1; p=pos[i-1]; c=cost if p!=pp else 0
        if p!=pp: flips+=1
        r=p*ret-c; eq*=(1+r); peak=max(peak,eq); dd=min(dd,eq/peak-1); daily.append(r); pp=p
    if not daily: return None
    span=((hi or dts[-1])-(lo or dts[1])).days/365.25; span=max(0.2,span)
    cagr=eq**(1/span)-1; mean=sum(daily)/len(daily); sd=(sum((x-mean)**2 for x in daily)/len(daily))**0.5
    return dict(eq=eq,cagr=cagr,dd=dd,sharpe=(mean/sd*math.sqrt(ANN)) if sd>0 else 0,
                calmar=cagr/abs(dd) if dd<0 else 0,flips=flips)

def longshort(pos,cost):
    segs=[];pp=0;seg=1.0;sd=0
    for i in range(1,n):
        ret=closes[i]/closes[i-1]-1; p=pos[i-1]
        if p!=pp and pp!=0: segs.append((sd,seg-1)); seg=1.0
        seg*=(1+p*ret); sd=p; pp=p
    if sd!=0: segs.append((sd,seg-1))
    o={}
    for tag,dd in (("LONG",1),("SHORT",-1)):
        s=[r for d,r in segs if d==dd]
        if s:
            comp=1.0
            for r in s: comp*=(1+r)
            o[tag]=(len(s),100*sum(1 for r in s if r>0)/len(s),100*(comp-1))
    return o

def peryear(pos,cost):
    yr=defaultdict(lambda:[1.0,1.0]);pp=0
    for i in range(1,n):
        y=dts[i].year; ret=closes[i]/closes[i-1]-1; p=pos[i-1]; c=cost if p!=pp else 0
        yr[y][0]*=(1+p*ret-c); yr[y][1]*=(1+ret); pp=p
    return yr

L=[];P=L.append
P("="*80);P(f"REC 15-min Supertrend DEEP-DIVE | {dts[1]}..{dts[-1]} ({n} bars) | TEST from {TEST_LO}");P("="*80)
bhf=closes[-1]/closes[0]-1; bht=[closes[i] for i in range(n) if dts[i]>=TEST_LO]
P(f"Buy&Hold: full {100*bhf:+.1f}%  |  test {100*(bht[-1]/bht[0]-1):+.1f}%")
for (p,mu,mode) in CONFIGS:
    d=st(p,mu); pos=pos_of(d,mode)
    P(f"\n{'='*60}\nCONFIG  ST({p},{mu}) {mode}")
    P("  COST SENSITIVITY (make-or-break):")
    P("    cost/flip | FULL: CAGR Sharpe Calmar MaxDD | TEST(OOS): CAGR Sharpe Calmar MaxDD")
    for cst in COSTS:
        f=metric(pos,cst); t=metric(pos,cst,lo=TEST_LO)
        P(f"    {cst*100:5.2f}%   | {100*f['cagr']:+6.1f} {f['sharpe']:5.2f} {f['calmar']:5.2f} {100*f['dd']:6.1f} "
          f"| {100*t['cagr']:+6.1f} {t['sharpe']:5.2f} {t['calmar']:5.2f} {100*t['dd']:6.1f}  (flips {t['flips']})")
    ls=longshort(pos,0.0010)
    P("  LONG/SHORT (@0.10%): "+"  ".join(f"{k}: n={v[0]} win={v[1]:.0f}% comp={v[2]:+.0f}%" for k,v in ls.items()))
    yr=peryear(pos,0.0010)
    P("  PER-YEAR (@0.10% strat vs B&H): "+"  ".join(f"{y}:{100*(yr[y][0]-1):+.0f}%/{100*(yr[y][1]-1):+.0f}%" for y in sorted(yr)))

# EXECUTION-LAG robustness: act 0 / 1 / 2 bars after the signal (15-min each)
def shift(arr,k): return [arr[max(0,i-k)] for i in range(len(arr))]
P(f"\n{'='*60}\nEXECUTION-LAG ROBUSTNESS (@0.10%/flip, OOS test) — does it survive a delayed fill?")
P("    config           lag0(same-close)  lag1(+15m)   lag2(+30m)")
for (p,mu,mode) in CONFIGS:
    d=st(p,mu); base=pos_of(d,mode); line=f"    ST({p},{mu}) {mode:6s} "
    for lag in (0,1,2):
        t=metric(shift(base,lag),0.0010,lo=TEST_LO)
        line+=f"  CAGR {100*t['cagr']:+5.0f}% Sh {t['sharpe']:4.2f} |"
    P(line)
print("\n".join(L))
