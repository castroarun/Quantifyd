#!/usr/bin/env python3
"""
REC Ltd (RECLTD) — always-on Supertrend(7,3) futures system, daily, last 2yr.
Always in the market: long when ST bullish, short when bearish, flip on signal.
No look-ahead: ST computed on close of day t; position effective day t+1.
Reports gross AND net of cost + cost sensitivity, vs buy-and-hold, per-year.
Underlying daily OHLC used as continuous-futures proxy. Pure stdlib, VPS.
"""
import sqlite3, math
from datetime import datetime
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"; SYM="RECLTD"
START="2024-05-15"; PERIOD=7; MULT=3.0

con=sqlite3.connect(DB)
bars=[(datetime.strptime(d.split()[0],"%Y-%m-%d").date(),float(o),float(h),float(l),float(c))
      for d,o,h,l,c in con.execute("SELECT date,open,high,low,close FROM market_data_unified "
      "WHERE symbol=? AND timeframe='day' AND date>=? ORDER BY date",(SYM,START))]
con.close()
n=len(bars)

def supertrend(bars,period,mult):
    m=len(bars); tr=[0.0]*m
    for i in range(m):
        _,o,h,l,c=bars[i]
        tr[i]=h-l if i==0 else max(h-l,abs(h-bars[i-1][4]),abs(l-bars[i-1][4]))
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

direction=supertrend(bars,PERIOD,MULT)

def run(cost_per_flip):
    eq=1.0; peak=1.0; dd=0.0; bh=1.0
    pos_prev=0; flips=0; daily=[]
    seg_ret=1.0; segs=[]; long_days=short_days=0
    for i in range(1,n):
        c0=bars[i-1][4]; c1=bars[i][4]; ret=c1/c0-1
        pos=direction[i-1]          # tradable: signal from prior close
        if pos==1: long_days+=1
        elif pos==-1: short_days+=1
        sr=pos*ret
        cost=0.0
        if pos!=pos_prev and pos_prev!=0:
            cost=cost_per_flip; flips+=1
            segs.append(seg_ret-1); seg_ret=1.0
        seg_ret*= (1+sr)
        eq*=(1+sr-cost); peak=max(peak,eq); dd=min(dd,eq/peak-1)
        bh*=(1+ret)
        daily.append(sr-cost); pos_prev=pos
    segs.append(seg_ret-1)
    yrs=(bars[-1][0]-bars[1][0]).days/365.25
    cagr=eq**(1/yrs)-1; bhcagr=bh**(1/yrs)-1
    mean=sum(daily)/len(daily); sd=(sum((x-mean)**2 for x in daily)/len(daily))**0.5
    sharpe=(mean/sd)*math.sqrt(252) if sd>0 else 0
    wins=sum(1 for s in segs if s>0)
    return dict(eq=eq,cagr=cagr,dd=dd,sharpe=sharpe,flips=flips,bh=bh,bhcagr=bhcagr,
                trades=len(segs),win=100*wins/len(segs) if segs else 0,
                long_days=long_days,short_days=short_days,yrs=yrs)

def peryear(cost):
    yr=defaultdict(lambda:[1.0,1.0]); pos_prev=0
    for i in range(1,n):
        y=bars[i][0].year; c0=bars[i-1][4]; c1=bars[i][4]; ret=c1/c0-1
        pos=direction[i-1]; cost_i=cost if (pos!=pos_prev and pos_prev!=0) else 0
        yr[y][0]*=(1+pos*ret-cost_i); yr[y][1]*=(1+ret); pos_prev=pos
    return yr

def long_short_breakdown(cost):
    """Split P&L into LONG segments vs SHORT segments."""
    segs=[]; pos_prev=0; seg_ret=1.0; seg_dir=0
    for i in range(1,n):
        ret=bars[i][4]/bars[i-1][4]-1; pos=direction[i-1]
        if pos!=pos_prev and pos_prev!=0:
            segs.append((seg_dir,seg_ret-1)); seg_ret=1.0
        cost_i=cost if (pos!=pos_prev and pos_prev!=0) else 0
        seg_ret*=(1+pos*ret-cost_i); seg_dir=pos; pos_prev=pos
    segs.append((seg_dir,seg_ret-1))
    out={}
    for tag,d in (("LONG",1),("SHORT",-1)):
        s=[r for dd,r in segs if dd==d]
        if s:
            wins=sum(1 for r in s if r>0)
            # compounded contribution of just these segments
            comp=1.0
            for r in s: comp*=(1+r)
            out[tag]=dict(n=len(s),win=100*wins/len(s),avg=100*sum(s)/len(s),
                          best=100*max(s),worst=100*min(s),comp=100*(comp-1))
    return out

L=[];P=L.append
P("="*70);P(f"RECLTD — always-on Supertrend({PERIOD},{MULT}) daily | {bars[1][0]}..{bars[-1][0]} ({n} bars)");P("="*70)
g=run(0.0)
P(f"\nBuy & Hold:   total {100*(g['bh']-1):+7.1f}%   CAGR {100*g['bhcagr']:+6.1f}%")
P(f"\nGROSS (0 cost): total {100*(g['eq']-1):+7.1f}%  CAGR {100*g['cagr']:+6.1f}%  "
  f"Sharpe {g['sharpe']:.2f}  MaxDD {100*g['dd']:.1f}%")
P(f"  trades(flips)={g['flips']}  segments={g['trades']}  win%={g['win']:.0f}  "
  f"days long/short={g['long_days']}/{g['short_days']}")
P("\nNET cost sensitivity (cost charged per flip, % of notional):")
P("    cost/flip   total      CAGR     Sharpe   MaxDD    trades")
for cpf in (0.0005,0.0010,0.0020,0.0030):
    r=run(cpf)
    P(f"    {cpf*100:5.2f}%    {100*(r['eq']-1):+7.1f}%  {100*r['cagr']:+6.1f}%   {r['sharpe']:5.2f}  {100*r['dd']:6.1f}%   {r['flips']}")
P("\nLONG vs SHORT breakdown (@0.10%/flip):")
P("    side    trades  win%   avg/trade   best     worst   compounded")
bd=long_short_breakdown(0.0010)
for tag in ("LONG","SHORT"):
    if tag in bd:
        b=bd[tag]
        P(f"    {tag:6s}   {b['n']:3d}   {b['win']:4.0f}   {b['avg']:+7.2f}%   "
          f"{b['best']:+6.1f}%  {b['worst']:+6.1f}%   {b['comp']:+7.1f}%")

P("\nPer-year (strategy @0.10%/flip  vs  buy&hold):")
yr=peryear(0.0010)
for y in sorted(yr):
    P(f"    {y}:  strat {100*(yr[y][0]-1):+7.1f}%   B&H {100*(yr[y][1]-1):+7.1f}%")
print("\n".join(L))
