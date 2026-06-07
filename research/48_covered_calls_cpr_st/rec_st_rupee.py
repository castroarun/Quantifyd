#!/usr/bin/env python3
"""
REC 15-min Supertrend — HONEST rupee-per-1-lot P&L.
Fixed 1 lot. Two execution models:
  optimistic : signal bar's CLOSE fill (close-to-close)  [the rosy backtest]
  realistic  : NEXT-BAR OPEN fill (signal@close -> fill@next open)  [truth]
Cost 0.10%/flip of notional. Reports Rs P&L/lot, Rs MaxDD, return on margin and
on full notional, per-year. Pure stdlib, VPS.
"""
import sqlite3, math
from datetime import datetime
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"; SYM="RECLTD"; STEP=3
LOT=1400; COSTPCT=0.0010; MARGIN_PCT=0.20
CONFIGS=[(7,3.0,"both"),(21,3.0,"both"),(10,3.0,"short")]

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
n=len(bars)
opens=[b[1] for b in bars]; closes=[b[4] for b in bars]; dts=[b[0].date() for b in bars]
TEST_LO=dts[n//2]

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

def desired(d,mode):
    out=[0]*n
    for i in range(n):
        if mode=="both": out[i]=d[i]
        elif mode=="long": out[i]=1 if d[i]==1 else 0
        elif mode=="short": out[i]=-1 if d[i]==-1 else 0
    return out

def run_rupee(d,mode,execmodel,lo=None,hi=None):
    """Returns list of (date, rupee_pnl_this_step) and trade count."""
    des=desired(d,mode); pnl=[]; flips=0; pp=0
    for i in range(1,n):
        if execmodel=="optimistic":
            # hold des[i-1] over close[i-1]->close[i]
            pos=des[i-1]; px0=closes[i-1]; px1=closes[i]; flip_px=closes[i-1]; dt=dts[i]
        else: # realistic: hold des[i-1] over open[i]->open[i+1]
            if i+1>=n: break
            pos=des[i-1]; px0=opens[i]; px1=opens[i+1]; flip_px=opens[i]; dt=dts[i]
        if lo and dt<lo: pp=pos; continue
        if hi and dt>=hi: break
        rupee=pos*(px1-px0)*LOT
        if pos!=pp:
            rupee-=COSTPCT*flip_px*LOT; flips+=1
        pnl.append((dt,rupee)); pp=pos
    return pnl,flips

def summarize(pnl,flips,label):
    if not pnl: return None
    total=sum(r for _,r in pnl)
    eq=0.0; peak=0.0; dd=0.0
    for _,r in pnl:
        eq+=r; peak=max(peak,eq); dd=min(dd,eq-peak)
    span=(pnl[-1][0]-pnl[0][0]).days/365.25; span=max(0.2,span)
    notional=closes[0]*LOT; margin=notional*MARGIN_PCT
    peryr=total/span
    return dict(label=label,total=total,dd=dd,flips=flips,span=span,peryr=peryr,
                notional=notional,margin=margin,
                roc_margin=peryr/margin*100, roc_notional=peryr/notional*100)

def peryear(pnl):
    yr=defaultdict(float)
    for dt,r in pnl: yr[dt.year]+=r
    return yr

L=[];P=L.append
P("="*84);P(f"REC 15-min Supertrend — Rs P&L per 1 LOT (lot={LOT}, ~Rs{closes[0]*LOT/1e5:.2f}L notional/lot)");P("="*84)
P(f"Cost {COSTPCT*100:.2f}%/flip | margin@{int(MARGIN_PCT*100)}%~Rs{closes[0]*LOT*MARGIN_PCT/1e5:.2f}L/lot | TEST(OOS) from {TEST_LO}")
# Buy&Hold rupee per lot (test)
bh_test=(closes[-1]-closes[[i for i in range(n) if dts[i]>=TEST_LO][0]])*LOT
P(f"Buy&Hold (test, 1 lot long held): Rs{bh_test:+,.0f}")

for (p,mu,mode) in CONFIGS:
    d=st(p,mu)
    P(f"\n{'='*64}\nST({p},{mu}) {mode}")
    P("    exec model    | FULL period            | TEST (OOS)")
    P("                  | Rs P&L/yr  MaxDD  ROC%m | Rs P&L  Rs P&L/yr  MaxDD   ROC%marg ROC%notnl flips")
    for em in ("optimistic","realistic"):
        full,ff=run_rupee(d,mode,em); sf=summarize(full,ff,em)
        tp,tf=run_rupee(d,mode,em,lo=TEST_LO); st_=summarize(tp,tf,em)
        if sf and st_:
            P(f"    {em:11s}   | {sf['peryr']:+9,.0f} {sf['dd']:+8,.0f} {sf['roc_margin']:5.0f} | "
              f"{st_['total']:+8,.0f} {st_['peryr']:+9,.0f} {st_['dd']:+8,.0f} {st_['roc_margin']:7.0f} {st_['roc_notional']:8.0f}  {st_['flips']}")
    # per-year rupee (realistic)
    rp,_=run_rupee(d,mode,"realistic")
    yr=peryear(rp)
    P("    per-year Rs (realistic): "+"  ".join(f"{y}:{v:+,.0f}" for y,v in sorted(yr.items())))
print("\n".join(L))
