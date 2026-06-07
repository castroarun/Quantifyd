#!/usr/bin/env python3
"""
REC Ltd — Supertrend positional system optimization (daily + intraday).
Flexible engine: timeframe, ST(period,mult), direction mode, 200-period regime
filter. Walk-forward OOS (train->test) + per-year. Net of cost. Pure stdlib, VPS.

Modes:
  both     : always-on (pos = ST dir)
  long     : long when ST bull, else flat
  short    : short when ST bear, else flat
  regime   : long only if ST bull AND price>MA200; short only if ST bear AND price<MA200; else flat

Run: python3 rec_st_sweep.py [day|15|30|60]   (default day)
Underlying OHLC as continuous-futures proxy; cost charged per flip.
"""
import sqlite3, math, sys
from datetime import datetime
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"; SYM="RECLTD"
TF=sys.argv[1] if len(sys.argv)>1 else "day"
COST=0.0005                       # 0.05% per flip (sensitivity already shown minor)
PERIODS=[7,10,14,21]; MULTS=[2.0,2.5,3.0,3.5,4.0]
MODES=["both","long","short","regime"]
MA_N=200
DAILY_START="2019-01-01"          # long span for walk-forward; per-year shown
TRAIN_END=datetime(2023,1,1).date()

con=sqlite3.connect(DB)
if TF=="day":
    raw=[(datetime.strptime(d.split()[0],"%Y-%m-%d").date(),float(o),float(h),float(l),float(c))
         for d,o,h,l,c in con.execute("SELECT date,open,high,low,close FROM market_data_unified "
         "WHERE symbol=? AND timeframe='day' AND date>=? ORDER BY date",(SYM,DAILY_START))]
    bars=[(dt,o,h,l,c) for (dt,o,h,l,c) in raw]
    dts=[b[0] for b in bars]
else:
    step=int(TF)//5
    r5=[(datetime.strptime(d,"%Y-%m-%d %H:%M:%S"),float(o),float(h),float(l),float(c))
        for d,o,h,l,c in con.execute("SELECT date,open,high,low,close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date",(SYM,))]
    # resample within each day into N-min buckets
    byday=defaultdict(list)
    for x in r5: byday[x[0].date()].append(x)
    bars=[]
    for d in sorted(byday):
        day=byday[d]
        for k in range(0,len(day),step):
            grp=day[k:k+step]
            if not grp: continue
            bars.append((grp[0][0], grp[0][1], max(g[2] for g in grp),
                         min(g[3] for g in grp), grp[-1][4]))
    dts=[b[0].date() for b in bars]
con.close()
n=len(bars); closes=[b[4] for b in bars]
# intraday data starts 2024-03; the 2023 split would leave train empty -> use midpoint
if TF!="day":
    TRAIN_END=dts[n//2]

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

def ma(series,N):
    out=[None]*len(series); s=0.0
    for i,v in enumerate(series):
        s+=v
        if i>=N: s-=series[i-N]
        if i>=N-1: out[i]=s/N
    return out
ma200=ma(closes,MA_N)

def positions(direction,mode):
    pos=[0]*n
    for i in range(n):
        d=direction[i]; c=closes[i]; m=ma200[i]
        if mode=="both": p=d
        elif mode=="long": p=1 if d==1 else 0
        elif mode=="short": p=-1 if d==-1 else 0
        else: # regime
            if d==1 and m is not None and c>m: p=1
            elif d==-1 and m is not None and c<m: p=-1
            else: p=0
        pos[i]=p
    return pos

def metrics(pos, lo=None, hi=None):
    eq=1.0;peak=1.0;dd=0.0;daily=[];flips=0;pp=0;segs=[];seg=1.0;sdir=0
    for i in range(1,n):
        if lo and dts[i]<lo:
            pp=pos[i-1]; continue
        if hi and dts[i]>=hi: break
        ret=closes[i]/closes[i-1]-1; p=pos[i-1]
        cost=COST if (p!=pp) else 0.0
        if p!=pp:
            flips+=1
            if sdir!=0: segs.append((sdir,seg-1))
            seg=1.0
        seg*=(1+ (p*ret) ); sdir=p
        r=p*ret-cost
        eq*=(1+r); peak=max(peak,eq); dd=min(dd,eq/peak-1); daily.append(r); pp=p
    if sdir!=0: segs.append((sdir,seg-1))
    if not daily or eq<=0: return None
    yrs=max(0.1,(min(hi,dts[-1]) if hi else dts[-1] and dts[-1]).__sub__(lo or dts[1]).days/365.25) if False else None
    # compute years from the daily count by tf
    span_days=( (hi or dts[-1]) - (lo or dts[1]) ).days if (hi or lo) else (dts[-1]-dts[1]).days
    yrs=max(0.2,span_days/365.25)
    cagr=eq**(1/yrs)-1
    mean=sum(daily)/len(daily); sd=(sum((x-mean)**2 for x in daily)/len(daily))**0.5
    ann=252 if TF=="day" else 252*(375/int(TF))
    sharpe=(mean/sd)*math.sqrt(ann) if sd>0 else 0
    wins=sum(1 for _,s in segs if s>0)
    calmar=cagr/abs(dd) if dd<0 else 0
    return dict(eq=eq,cagr=cagr,dd=dd,sharpe=sharpe,calmar=calmar,flips=flips,
                trades=len(segs),win=100*wins/len(segs) if segs else 0,segs=segs)

# buy & hold over test window
def bh(lo,hi):
    s=[closes[i] for i in range(n) if (not lo or dts[i]>=lo) and (not hi or dts[i]<hi)]
    if len(s)<2: return 0,0
    span=((hi or dts[-1])-(lo or dts[1])).days/365.25
    tot=s[-1]/s[0]-1; return tot, (s[-1]/s[0])**(1/max(0.2,span))-1

L=[];P=L.append
P("="*82);P(f"REC SUPERTREND SWEEP — TF={TF} | {dts[1]}..{dts[-1]} ({n} bars) | cost {COST*100:.2f}%/flip");P("="*82)
bt,bc=bh(TRAIN_END,None)
P(f"\nBuy&Hold (TEST {TRAIN_END}+):  total {100*bt:+.1f}%  CAGR {100*bc:+.1f}%")

rows=[]
for p in PERIODS:
    for mu in MULTS:
        d=supertrend(bars,p,mu)
        for mode in MODES:
            pos=positions(d,mode)
            full=metrics(pos)
            tr=metrics(pos,hi=TRAIN_END)
            te=metrics(pos,lo=TRAIN_END)
            if full and tr and te:
                rows.append((p,mu,mode,full,tr,te))

# HONEST walk-forward: pick best by TRAIN calmar, report its TEST
valid=[r for r in rows if r[4]["trades"]>=5]
by_train=sorted(valid,key=lambda r:r[4]["calmar"],reverse=True)
P("\n[WALK-FORWARD] best by TRAIN Calmar -> its true OOS TEST result:")
P("    rank  P/mult  mode    | train: CAGR Calmar | TEST: CAGR  Calmar Sharpe MaxDD win% trades")
for k,(p,mu,mode,full,tr,te) in enumerate(by_train[:8]):
    P(f"    {k+1:2d}   {p:2d}/{mu:.1f} {mode:6s} | {100*tr['cagr']:+6.1f} {tr['calmar']:5.2f} | "
      f"{100*te['cagr']:+6.1f} {te['calmar']:6.2f} {te['sharpe']:5.2f} {100*te['dd']:6.1f} {te['win']:4.0f} {te['trades']}")

P("\n[UPPER BOUND] best by TEST Calmar (what's achievable; overfit-prone):")
by_test=sorted(valid,key=lambda r:r[5]["calmar"],reverse=True)
for p,mu,mode,full,tr,te in by_test[:8]:
    P(f"    {p:2d}/{mu:.1f} {mode:6s}  TEST CAGR {100*te['cagr']:+6.1f} Calmar {te['calmar']:5.2f} "
      f"Sharpe {te['sharpe']:5.2f} MaxDD {100*te['dd']:6.1f} win {te['win']:3.0f}% | train Calmar {tr['calmar']:5.2f}")
print("\n".join(L))
