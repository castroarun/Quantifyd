#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G12 — the two questions that gate the condor: strike PEAK, and portfolio correlation.

PART A — extend the strike sweep until it turns. Wider shorts kept improving at the edge of the last
sweep (0.4->0.6->0.8% => Calmar 0.98->1.15->1.45), which usually means the optimum is outside the
range tested. Sweep shorts 0.4 .. 1.6% and find where Calmar peaks.

PART B — is the condor a NEW bet or the SAME short-vol bet again? My first attempt was INVALID (I
booked P&L on the entry day, not when it lands, and my 916 proxy lost money). Fix both:
  * proxy the intraday short-vol book with NAS-OPT itself -- research/79 validated its synthetic
    version against 11 real live days to within 5%, so it is the trustworthy short-vol proxy.
  * book each position's P&L as a DAILY mark-to-market change, so the two series are time-aligned.
Then the real question: NAS-OPT trades Mon/Tue (0/1-DTE); the condor holds Wed->Fri. They are
RARELY in the market on the same calendar day. So does trading different days of the week give tail
DIVERSIFICATION (a one-day crash hits only one book) -- or does a multi-week vol regime hit both?
Measured: day-of-week overlap, correlation on overlapping days, and the COMBINED drawdown vs each
book alone. All on the calibrated engine, 5-min marking, 2015-2026.
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
def mn(ts): return int(ts[11:13])*60+int(ts[14:16])
def T_yr(ts,exp): return max((datetime(exp.year,exp.month,exp.day,15,30)-datetime.fromisoformat(ts)).total_seconds()/(365*24*3600),1e-6)

# ═══════════════════ PART A — strike peak ═══════════════════
def condor_trades(sp_pct, wp_pct=0.010, stop=2.0):
    tr=[]
    for day in sess:
        y,m,dd=map(int,day.split("-")); d0=date(y,m,dd); e=nx(d0)
        if (e-d0).days!=6: continue
        ts0,S0=bars[day][-1]; v0=v5.get(ts0,vday.get(day))
        if not v0: continue
        T0=T_yr(ts0,e); iv0=(v0/100)*ivm(T0*365)
        Kc=round(S0*(1+sp_pct)/50)*50; Kp=round(S0*(1-sp_pct)/50)*50
        Kcw=round(S0*(1+sp_pct+wp_pct)/50)*50; Kpw=round(S0*(1-sp_pct-wp_pct)/50)*50
        cred=(bs(S0,Kc,T0,iv0,"CE")+bs(S0,Kp,T0,iv0,"PE")-bs(S0,Kcw,T0,iv0,"CE")-bs(S0,Kpw,T0,iv0,"PE"))
        if cred<2: continue
        # daily marks Wed..Fri
        marks={day:cred}; hold=[]
        for s in sess[sidx[day]+1:]:
            hold.append(s); sy,sm,sd=map(int,s.split("-"))
            if (e-date(sy,sm,sd)).days<=4: break
        val=cred; stopped=False
        for s in hold:
            last=val
            for ts,Sx in bars[s]:
                vv=v5.get(ts,vday.get(s,v0)); Tx=T_yr(ts,e); ivx=(vv/100)*ivm(Tx*365)
                val=(bs(Sx,Kc,Tx,ivx,"CE")+bs(Sx,Kp,Tx,ivx,"PE")-bs(Sx,Kcw,Tx,ivx,"CE")-bs(Sx,Kpw,Tx,ivx,"PE"))
                if stop and val>=cred*stop: stopped=True; break
            marks[s]=val
            if stopped: break
        pnl=(cred*(1-SLIP)-val*(1+SLIP))*QTY-4*BROK
        # per-day MTM delta series for the correlation test
        days_sorted=sorted(marks); daily={}
        prev=cred
        for i,s in enumerate(days_sorted):
            if i==0: continue                       # entry day: credit taken, no MTM change booked
            daily[s]=(prev-marks[s])*QTY            # short: value down => profit
            prev=marks[s]
        # apply costs/slippage to the final exit day
        if days_sorted[1:]:
            daily[days_sorted[-1]]-=4*BROK
        tr.append(dict(day=day,year=day[:4],pnl=pnl,stopped=stopped,daily=daily))
    return tr

def sumstats(tr):
    v=np.array([t["pnl"] for t in tr],float)
    eq=np.cumsum(v); mdd=(eq-np.maximum.accumulate(eq)).min()
    ys=sorted({t["year"] for t in tr}); ann=v.sum()/len(ys)
    return dict(n=len(v),mean=v.mean(),ann=ann,win=100*(v>0).mean(),mdd=mdd,cal=ann/abs(mdd),
                worst=v.min(),pos=sum(1 for y in ys if np.mean([t['pnl'] for t in tr if t['year']==y])>0),ny=len(ys))

print("=== PART A: strike PEAK (wings 1.0%, stop x2, 5-min marks) ===\n")
print(f"{'shorts':>7s} {'n':>4s} {'mean':>8s} {'annual':>9s} {'win':>5s} {'maxDD':>10s} {'Calmar':>7s} {'worst':>9s} {'+yrs':>6s}")
best=None
for spc in (0.004,0.006,0.008,0.010,0.012,0.014,0.016):
    tr=condor_trades(spc); st=sumstats(tr)
    print(f"{spc*100:>6.1f}% {st['n']:>4d} {st['mean']:>+8.0f} {st['ann']:>+9.0f} {st['win']:>4.0f}% "
          f"{st['mdd']:>+10.0f} {st['cal']:>7.2f} {st['worst']:>+9.0f} {st['pos']:>2d}/{st['ny']}")
    if best is None or st['cal']>best[0]: best=(st['cal'],spc,tr)
print(f"\nPEAK Calmar at shorts = {best[1]*100:.1f}%  (Calmar {best[0]:.2f})")

# ═══════════════════ PART B — correlation vs NAS-OPT ═══════════════════
# NAS-OPT synthetic: 0.4%-OTM strangle, 0/1-DTE, 09:20 entry, 0.4% move-stop, 14:45 exit
def nasopt_daily():
    out={}
    for day in sess:
        b=bars[day]
        if len(b)<40: continue
        v0=v5.get(b[-1][0],vday.get(day))
        if not v0: continue
        y,m,dd=map(int,day.split("-")); d0=date(y,m,dd); e=nx(d0); dte=(e-d0).days
        if dte>1: continue                          # NAS-OPT: 0/1-DTE only
        ent=next((x for x in b if mn(x[0])>=9*60+20),None)
        if not ent: continue
        S0=ent[1]; Kc=round(S0*1.004/50)*50; Kp=round(S0*0.996/50)*50
        T0=T_yr(ent[0],e); iv=(v0/100)*ivm(T0*365)
        c0=bs(S0,Kc,T0,iv,"CE"); p0=bs(S0,Kp,T0,iv,"PE")
        win=[x for x in b if 9*60+20<=mn(x[0])<=14*60+45]
        ex=None
        for ts,Sx in win:
            if abs(Sx-S0)/S0>=0.004:
                Tx=T_yr(ts,e); ivx=(v5.get(ts,v0)/100)*ivm(Tx*365)
                ex=bs(Sx,Kc,Tx,ivx,"CE")+bs(Sx,Kp,Tx,ivx,"PE"); break
        if ex is None:
            ts,Sx=win[-1]; Tx=T_yr(ts,e); ivx=(v5.get(ts,v0)/100)*ivm(Tx*365)
            ex=bs(Sx,Kc,Tx,ivx,"CE")+bs(Sx,Kp,Tx,ivx,"PE")
        out[day]=((c0+p0)*(1-SLIP)-ex*(1+SLIP))*QTY-2*BROK
    return out

print("\n\n=== PART B: is the condor a NEW bet or the SAME short-vol bet? ===\n")
cond_tr=best[2]
cond_daily=defaultdict(float)
for t in cond_tr:
    for d,pnl in t["daily"].items(): cond_daily[d]+=pnl
nas=nasopt_daily()

# day-of-week: when is each book actually in the market?
wd_name=["Mon","Tue","Wed","Thu","Fri"]
print("days each book has NON-ZERO P&L, by weekday:")
print(f"  {'':<8s}" + "".join(f"{w:>7s}" for w in wd_name))
for lab,ser in [("NAS-OPT",nas),("Condor",cond_daily)]:
    cnt=[sum(1 for d in ser if datetime.fromisoformat(d).weekday()==i and abs(ser[d])>1e-6) for i in range(5)]
    print(f"  {lab:<8s}" + "".join(f"{x:>7d}" for x in cnt))

both=[d for d in cond_daily if d in nas and abs(nas[d])>1e-6 and abs(cond_daily[d])>1e-6]
print(f"\ncalendar days BOTH books are simultaneously in the market: {len(both)}  "
      f"(of {len(nas)} NAS-OPT days, {len([d for d in cond_daily if abs(cond_daily[d])>1e-6])} condor days)")
if len(both)>=20:
    a=np.array([cond_daily[d] for d in both]); bb=np.array([nas[d] for d in both])
    print(f"correlation of daily P&L on those overlapping days: {np.corrcoef(a,bb)[0,1]:+.3f}")

# combined book: same capital pool, different days -> just add the daily series
alld=sorted(set(cond_daily)|set(nas))
cs=np.array([cond_daily.get(d,0.0) for d in alld])
ns=np.array([nas.get(d,0.0) for d in alld])
def dd(x):
    eq=np.cumsum(x); return (eq-np.maximum.accumulate(eq)).min()
print(f"\ncombined-book drawdown (same capital, different days of the week):")
print(f"  NAS-OPT alone   total {ns.sum():+,.0f}   maxDD {dd(ns):+,.0f}")
print(f"  Condor alone    total {cs.sum():+,.0f}   maxDD {dd(cs):+,.0f}")
print(f"  BOTH together   total {(cs+ns).sum():+,.0f}   maxDD {dd(cs+ns):+,.0f}")
print(f"  sum of separate DDs = {dd(ns)+dd(cs):+,.0f}   -> combined is "
      f"{'BETTER than the sum (diversified)' if dd(cs+ns)>dd(ns)+dd(cs) else 'WORSE/equal (concentrated)'}")

# the tail: condor's 10 worst days -> what was NAS-OPT doing?
print("\ncondor's 10 worst days -> NAS-OPT's P&L that day (was it also in the market and losing?):")
worst=sorted([d for d in cond_daily if abs(cond_daily[d])>1e-6], key=lambda d: cond_daily[d])[:10]
both_lose=0
for d in worst:
    nv=nas.get(d)
    active = nv is not None and abs(nv)>1e-6
    if active and nv<0: both_lose+=1
    wd=wd_name[datetime.fromisoformat(d).weekday()] if datetime.fromisoformat(d).weekday()<5 else "?"
    print(f"  {d} {wd}  condor {cond_daily[d]:+,.0f}   NAS-OPT {('flat/not in market' if not active else format(nv,'+,.0f'))}")
print(f"\n  on {both_lose} of the condor's 10 worst days, NAS-OPT was ALSO in the market and losing.")
print("\nDONE")
