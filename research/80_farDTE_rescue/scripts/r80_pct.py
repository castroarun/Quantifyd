#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G11 — REDO with PERCENTAGE strikes. The previous run was not one strategy.

Fixed +/-100pt shorts and 250pt wings across 2015-2026 is NOT a single strategy: NIFTY went from
8,000 to 24,000, so +/-100 points was 1.25% OTM in 2015 and 0.42% OTM today -- deep-OTM-and-cheap
then, nearly-ATM-and-fat now. The wings drifted the same way (3% wide -> 1% wide). And the stop
levels (125-225 POINTS) meant completely different things at each index level, which is almost
certainly why tighter stops looked monotonically better.

So: strikes and wings as a PERCENTAGE of spot, and the stop as a MULTIPLE OF THE CREDIT (both
scale-free). Now it is one strategy across the whole sample.
"""
import json, math, sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
import numpy as np

ROOT="/home/arun/quantifyd"
CAL=json.load(open(f"{ROOT}/research/80_farDTE_rescue/results/engine_calib.json"))["iv_mult_by_dte"]
QTY,BROK,SLIP=130,20,0.01

def ncdf(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
def bs(S,K,T,iv,k):
    if T<=0 or iv<=0: return max(0.0,(S-K) if k=="CE" else (K-S))
    d1=(math.log(S/K)+0.5*iv*iv*T)/(iv*math.sqrt(T)); d2=d1-iv*math.sqrt(T)
    return S*ncdf(d1)-K*ncdf(d2) if k=="CE" else K*ncdf(-d2)-S*ncdf(-d1)
def ivm(dd):
    d=int(round(dd))
    if str(d) in CAL: return CAL[str(d)]
    ks=sorted(int(k) for k in CAL); return CAL[str(min(ks,key=lambda k:abs(k-d)))]
def nx(d):
    t=1 if d>=date(2025,9,1) else 3; a=(t-d.weekday())%7
    return d if a==0 else d+timedelta(days=a)

c=sqlite3.connect(f"file:{ROOT}/backtest_data/market_data.db?mode=ro",uri=True)
n5=c.execute("SELECT date,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date").fetchall()
v5={str(d):v for d,v in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='5minute'")}
vday={r[0][:10]:r[1] for r in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
c.close()
bars=defaultdict(list)
for d,cl in n5: bars[str(d)[:10]].append((str(d),cl))
for k in bars: bars[k].sort()
sess=sorted(bars); sidx={d:i for i,d in enumerate(sess)}

def T_yr(ts,exp):
    return max((datetime(exp.year,exp.month,exp.day,15,30)-datetime.fromisoformat(ts)).total_seconds()/(365*24*3600),1e-6)

def run(short_pct, wing_pct, stop_mult):
    tr=[]
    for day in sess:
        y,m,dd=map(int,day.split("-")); d0=date(y,m,dd); e=nx(d0)
        if (e-d0).days!=6: continue
        ts0,S0=bars[day][-1]
        v0=v5.get(ts0,vday.get(day))
        if not v0: continue
        T0=T_yr(ts0,e); iv0=(v0/100)*ivm(T0*365)
        Kc=round(S0*(1+short_pct)/50)*50; Kp=round(S0*(1-short_pct)/50)*50
        Kcw=round(S0*(1+short_pct+wing_pct)/50)*50; Kpw=round(S0*(1-short_pct-wing_pct)/50)*50
        cred=(bs(S0,Kc,T0,iv0,"CE")+bs(S0,Kp,T0,iv0,"PE")
              -bs(S0,Kcw,T0,iv0,"CE")-bs(S0,Kpw,T0,iv0,"PE"))
        if cred<2: continue
        hold=[]
        for s in sess[sidx[day]+1:]:
            hold.append(s); sy,sm,sd=map(int,s.split("-"))
            if (e-date(sy,sm,sd)).days<=4: break
        val=cred; stopped=False
        for s in hold:
            for ts,Sx in bars[s]:
                vv=v5.get(ts,vday.get(s,v0)); Tx=T_yr(ts,e); ivx=(vv/100)*ivm(Tx*365)
                val=(bs(Sx,Kc,Tx,ivx,"CE")+bs(Sx,Kp,Tx,ivx,"PE")
                     -bs(Sx,Kcw,Tx,ivx,"CE")-bs(Sx,Kpw,Tx,ivx,"PE"))
                if stop_mult and val>=cred*stop_mult:
                    stopped=True; break
            if stopped: break
        wing_pts=Kcw-Kc
        tr.append(dict(year=day[:4],pnl=(cred*(1-SLIP)-val*(1+SLIP))*QTY-4*BROK,
                       stopped=stopped, margin=wing_pts*QTY+20000, cred=cred))
    return tr

def show(tr,lab):
    v=np.array([t["pnl"] for t in tr],float)
    eq=np.cumsum(v); mdd=(eq-np.maximum.accumulate(eq)).min()
    ys=sorted({t["year"] for t in tr}); ann=v.sum()/len(ys)
    ym=[np.mean([t["pnl"] for t in tr if t["year"]==y]) for y in ys]
    mg=np.mean([t["margin"] for t in tr])
    print(f"{lab:<34s} {len(v):>4d} {v.mean():>+8.0f} {ann:>+9.0f} {100*(v>0).mean():>4.0f}% "
          f"{mdd:>+10.0f} {ann/abs(mdd):>6.2f} {v.min():>+9.0f} {100*np.mean([t['stopped'] for t in tr]):>5.0f}% "
          f"{sum(1 for x in ym if x>0):>2d}/{len(ym)} {100*ann/mg:>7.0f}%")

print("=== PERCENTAGE strikes — one consistent strategy across 2015-2026 ===")
print("    Wed close -> Fri close | stop = multiple of CREDIT (scale-free) | 5-min marking\n")
print(f"{'shorts / wings / stop':<34s} {'n':>4s} {'mean':>8s} {'annual':>9s} {'win':>5s} "
      f"{'maxDD':>10s} {'Calmar':>6s} {'worst':>9s} {'fires':>5s} {'+yrs':>5s} {'ret/mgn':>7s}")
print("-"*126)
for sp in (0.004, 0.006, 0.008):
    for wp in (0.006, 0.010):
        for st in (None, 1.5, 2.0):
            lab=f"{sp*100:.1f}% / {wp*100:.1f}% / {('x'+str(st)) if st else 'wings only'}"
            show(run(sp,wp,st), lab)
    print()
print("DONE")
