#!/usr/bin/env python3
"""
Basket validation: does 15-min always-on Supertrend(7,3) generalize beyond REC?
Runs ST(7,3) both AND short on every F&O name with enough 5-min history,
realistic next-bar-OPEN execution + 0.10%/flip, OOS test split. Per-name CAGR/
Sharpe/MaxDD vs buy-and-hold. The verdict = how many names beat B&H out-of-sample,
and whether it works on names that ROSE (regime generalization). Pure stdlib, VPS.
"""
import sqlite3, math
from datetime import datetime
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"; STEP=3
COST=0.0010; PERIOD=7; MULT=3.0; MIN_BARS=20000   # ~full 2yr of 5-min
ANN=252*(375/15)

con=sqlite3.connect(DB)
syms=[r[0] for r in con.execute(
    "SELECT symbol FROM market_data_unified WHERE timeframe='5minute' "
    "GROUP BY symbol HAVING COUNT(*)>=? ORDER BY symbol",(MIN_BARS,))]

def load15(sym):
    r5=[(datetime.strptime(d,"%Y-%m-%d %H:%M:%S"),float(o),float(h),float(l),float(c))
        for d,o,h,l,c in con.execute("SELECT date,open,high,low,close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date",(sym,))]
    byd=defaultdict(list)
    for x in r5: byd[x[0].date()].append(x)
    bars=[]
    for d in sorted(byd):
        day=byd[d]
        for k in range(0,len(day),STEP):
            g=day[k:k+STEP]
            if g: bars.append((g[0][0],g[0][1],max(z[2] for z in g),min(z[3] for z in g),g[-1][4]))
    return bars

def st(bars,period,mult):
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

def metrics(bars,d,mode,lo):
    n=len(bars); opens=[b[1] for b in bars]; dts=[b[0].date() for b in bars]
    daily=[];eq=1.0;peak=1.0;dd=0.0;pp=0
    for i in range(1,n-1):
        des=d[i-1] if mode=="both" else (-1 if (mode=="short" and d[i-1]==-1) else (1 if (mode=="long" and d[i-1]==1) else 0))
        if dts[i]<lo: pp=des; continue
        ret=opens[i+1]/opens[i]-1; c=COST if des!=pp else 0
        r=des*ret-c; eq*=(1+r);peak=max(peak,eq);dd=min(dd,eq/peak-1); daily.append(r); pp=des
    if len(daily)<50 or eq<=0: return None
    span=(dts[-1]-lo).days/365.25; span=max(0.2,span)
    cagr=eq**(1/span)-1; mean=sum(daily)/len(daily); sd=(sum((x-mean)**2 for x in daily)/len(daily))**0.5
    return dict(cagr=cagr,dd=dd,sharpe=(mean/sd*math.sqrt(ANN)) if sd>0 else 0,
                calmar=cagr/abs(dd) if dd<0 else 0)

L=[];P=L.append
P("="*92);P(f"BASKET — 15-min always-on Supertrend({PERIOD},{MULT}) | realistic next-open fills, {COST*100:.2f}%/flip");P("="*92)
P(f"Symbols with >={MIN_BARS} 5-min bars: {len(syms)}")
P("\n  symbol        B&H(test)  | BOTH: CAGR Sharpe Calmar MaxDD | SHORT: CAGR Sharpe MaxDD  | beats B&H?")
rows=[]
for sym in syms:
    bars=load15(sym); n=len(bars)
    if n<MIN_BARS//3: continue
    dts=[b[0].date() for b in bars]; lo=dts[n//2]
    closes=[b[4] for b in bars]
    bh=[closes[i] for i in range(n) if dts[i]>=lo]
    bhret=bh[-1]/bh[0]-1 if len(bh)>1 else 0
    d=st(bars,PERIOD,MULT)
    mb=metrics(bars,d,"both",lo); ms=metrics(bars,d,"short",lo)
    if not mb: continue
    beat = mb["cagr"]>bhret
    rows.append((sym,bhret,mb,ms,beat))
    P(f"  {sym:12s} {100*bhret:+7.1f}%  | {100*mb['cagr']:+6.1f} {mb['sharpe']:5.2f} {mb['calmar']:5.2f} {100*mb['dd']:6.1f} | "
      f"{100*ms['cagr']:+6.1f} {ms['sharpe']:5.2f} {100*ms['dd']:6.1f}  | {'YES' if beat else 'no'}")
con.close()

if rows:
    nb=sum(1 for r in rows if r[4]); tot=len(rows)
    upn=[r for r in rows if r[1]>0]; dnn=[r for r in rows if r[1]<=0]
    P(f"\n  VERDICT: BOTH beats B&H on {nb}/{tot} names.")
    import statistics as S
    P(f"  median BOTH Sharpe={S.median(r[2]['sharpe'] for r in rows):.2f}  CAGR={100*S.median(r[2]['cagr'] for r in rows):+.1f}%  "
      f"median SHORT Sharpe={S.median(r[3]['sharpe'] for r in rows):.2f}")
    if upn:
        P(f"  on names that ROSE (n={len(upn)}): BOTH median Sharpe={S.median(r[2]['sharpe'] for r in upn):.2f} "
          f"CAGR={100*S.median(r[2]['cagr'] for r in upn):+.1f}%  beats B&H {sum(1 for r in upn if r[4])}/{len(upn)}")
    if dnn:
        P(f"  on names that FELL (n={len(dnn)}): BOTH median Sharpe={S.median(r[2]['sharpe'] for r in dnn):.2f} "
          f"CAGR={100*S.median(r[2]['cagr'] for r in dnn):+.1f}%  beats B&H {sum(1 for r in dnn if r[4])}/{len(dnn)}")
print("\n".join(L))
