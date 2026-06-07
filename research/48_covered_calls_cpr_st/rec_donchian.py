#!/usr/bin/env python3
"""
REC — always-on Donchian channel breakout (Turtle-style), vs Supertrend.
Long when close breaks the prior N-bar HIGH; short when it breaks the prior
N-bar LOW; hold until the opposite breakout (always in market). Same realistic
next-bar-OPEN execution + 0.10%/flip + OOS split as the ST tests, so it's an
apples-to-apples comparison. Run: python3 rec_donchian.py [day|15]  (default 15)
Pure stdlib, VPS.
"""
import sqlite3, math, sys
from datetime import datetime
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"; SYM="RECLTD"
TF=sys.argv[1] if len(sys.argv)>1 else "15"
COST=0.0010
NS=[10,20,40,55] if TF!="day" else [10,20,55]
MODES=["both","long","short"]

con=sqlite3.connect(DB)
if TF=="day":
    bars=[(datetime.strptime(d.split()[0],"%Y-%m-%d").date(),float(o),float(h),float(l),float(c))
          for d,o,h,l,c in con.execute("SELECT date,open,high,low,close FROM market_data_unified "
          "WHERE symbol=? AND timeframe='day' AND date>=? ORDER BY date",(SYM,"2019-01-01"))]
else:
    step=int(TF)//5
    r5=[(datetime.strptime(d,"%Y-%m-%d %H:%M:%S"),float(o),float(h),float(l),float(c))
        for d,o,h,l,c in con.execute("SELECT date,open,high,low,close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date",(SYM,))]
    byd=defaultdict(list)
    for x in r5: byd[x[0].date()].append(x)
    bars=[]
    for d in sorted(byd):
        day=byd[d]
        for k in range(0,len(day),step):
            g=day[k:k+step]
            if g: bars.append((g[0][0] if TF!="day" else g[0][0].date(),g[0][1],max(z[2] for z in g),min(z[3] for z in g),g[-1][4]))
con.close()
n=len(bars)
opens=[b[1] for b in bars]; highs=[b[2] for b in bars]; lows=[b[3] for b in bars]; closes=[b[4] for b in bars]
dts=[(b[0] if TF=="day" else b[0].date()) for b in bars]
TEST_LO=dts[n//2] if TF!="day" else __import__("datetime").date(2023,1,1)
ANN=252 if TF=="day" else 252*(375/int(TF))

def donchian_dir(N):
    d=[0]*n
    for i in range(n):
        if i<N: d[i]=1; continue
        up=max(highs[i-N:i]); dn=min(lows[i-N:i])
        if closes[i]>=up: d[i]=1
        elif closes[i]<=dn: d[i]=-1
        else: d[i]=d[i-1] or 1
    return d

def metrics(d,mode,lo=None):
    daily=[];eq=1.0;peak=1.0;dd=0.0;pp=0;flips=0
    for i in range(1,n-1):
        des=d[i-1] if mode=="both" else (-1 if (mode=="short" and d[i-1]==-1) else (1 if (mode=="long" and d[i-1]==1) else 0))
        if lo and dts[i]<lo: pp=des; continue
        ret=opens[i+1]/opens[i]-1; c=COST if des!=pp else 0
        if des!=pp: flips+=1
        r=des*ret-c; eq*=(1+r);peak=max(peak,eq);dd=min(dd,eq/peak-1); daily.append(r); pp=des
    if len(daily)<30 or eq<=0: return None
    span=(dts[-1]-(lo or dts[1])).days/365.25; span=max(0.2,span)
    cagr=eq**(1/span)-1; mean=sum(daily)/len(daily); sd=(sum((x-mean)**2 for x in daily)/len(daily))**0.5
    return dict(cagr=cagr,dd=dd,sharpe=(mean/sd*math.sqrt(ANN)) if sd>0 else 0,
                calmar=cagr/abs(dd) if dd<0 else 0,flips=flips)

L=[];P=L.append
P("="*80);P(f"REC always-on DONCHIAN breakout | TF={TF} | {dts[1]}..{dts[-1]} ({n} bars) | {COST*100:.2f}%/flip, next-open fills");P("="*80)
bh=[closes[i] for i in range(n) if dts[i]>=TEST_LO]
P(f"Buy&Hold (test): {100*(bh[-1]/bh[0]-1):+.1f}%   TEST from {TEST_LO}")
P("\n  N    mode    | TEST: CAGR  Sharpe Calmar MaxDD  flips")
best=None
for N in NS:
    d=donchian_dir(N)
    for mode in MODES:
        te=metrics(d,mode,lo=TEST_LO)
        if not te: continue
        P(f"  {N:3d}  {mode:6s}  | {100*te['cagr']:+6.1f} {te['sharpe']:6.2f} {te['calmar']:6.2f} {100*te['dd']:6.1f}  {te['flips']}")
        if best is None or te["sharpe"]>best[1]["sharpe"]: best=((N,mode),te)
if best:
    P(f"\n  Best Donchian (test Sharpe): N={best[0][0]} {best[0][1]} -> Sharpe {best[1]['sharpe']:.2f}, CAGR {100*best[1]['cagr']:+.1f}%, Calmar {best[1]['calmar']:.2f}")
    P("  (compare to 15m Supertrend(7,3) both realistic: ~+10-17% CAGR, Sharpe ~0.5 OOS)")
print("\n".join(L))
