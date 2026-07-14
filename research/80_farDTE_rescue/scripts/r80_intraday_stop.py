#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G8 — mark the stop INTRADAY, not at daily closes. (Arun was right to push.)

G5/G7 checked "has the premium doubled?" only at the DAILY CLOSE. That is optimistic: in reality
the premium blows through the stop intraday and you exit there, worse. There was no excuse for it --
we have NIFTY 5-min AND India VIX 5-min for 11 years.

This re-runs the Wed->Fri condor with the position marked every 5 MINUTES:
  * spot  = NIFTY 5-min close
  * IV    = India VIX 5-min (falls back to the daily close if a bar is missing)
  * T     = exact time to the Tuesday 15:30 expiry, in years, decremented bar by bar
  * stop  = exit AT THAT BAR if the combined premium >= 2x the credit
  * else exit at the Friday 15:15 bar

Then compares, honestly, against the daily-close-marked version. If the edge only existed because
the stop was marked lazily, this is where it dies.
Reads only.
"""
import json, math, sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np

ROOT = "/home/arun/quantifyd"
CAL = json.load(open(f"{ROOT}/research/80_farDTE_rescue/results/engine_calib.json"))["iv_mult_by_dte"]
QTY, BROK, SLIP = 130, 20, 0.01
STOP_X = 2.0

def ncdf(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
def bs(S,K,T,iv,k):
    if T<=0 or iv<=0: return max(0.0,(S-K) if k=="CE" else (K-S))
    d1=(math.log(S/K)+0.5*iv*iv*T)/(iv*math.sqrt(T)); d2=d1-iv*math.sqrt(T)
    return S*ncdf(d1)-K*ncdf(d2) if k=="CE" else K*ncdf(-d2)-S*ncdf(-d1)
def ivm(dte_days):
    d=int(round(dte_days))
    if str(d) in CAL: return CAL[str(d)]
    ks=sorted(int(k) for k in CAL); return CAL[str(min(ks,key=lambda k:abs(k-d)))]
def nx(d):
    t=1 if d>=date(2025,9,1) else 3; a=(t-d.weekday())%7
    return d if a==0 else d+timedelta(days=a)

c=sqlite3.connect(f"file:{ROOT}/backtest_data/market_data.db?mode=ro",uri=True)
n5=c.execute("SELECT date, close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date").fetchall()
v5=c.execute("SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='5minute' ORDER BY date").fetchall()
vday={r[0][:10]:r[1] for r in c.execute("SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
c.close()

VIX5={str(d):v for d,v in v5}
bars=defaultdict(list)
for d,cl in n5:
    bars[str(d)[:10]].append((str(d), cl))
for k in bars: bars[k].sort()
sessions=sorted(bars)
sidx={d:i for i,d in enumerate(sessions)}

EXPIRY_HOUR=15.5     # options expire at 15:30

def T_years(ts, exp):
    dt=datetime.fromisoformat(ts)
    exp_dt=datetime(exp.year, exp.month, exp.day, 15, 30)
    return max((exp_dt-dt).total_seconds()/(365.0*24*3600), 1e-6)

def run(width, wing, exit_friday=True, intraday_stop=True):
    tr=[]
    for day in sessions:
        y,m,dd=map(int,day.split("-")); d0=date(y,m,dd); e=nx(d0)
        if (e-d0).days!=6: continue                      # enter Wednesday
        # entry at the last bar of Wednesday
        ts0, S0 = bars[day][-1]
        v0 = VIX5.get(ts0, vday.get(day))
        if not v0: continue
        T0=T_years(ts0, e); iv0=(v0/100.0)*ivm(T0*365)
        Kc,Kp=round((S0+width)/50)*50, round((S0-width)/50)*50
        Kcw,Kpw=(Kc+wing, Kp-wing) if wing else (None,None)
        cred=bs(S0,Kc,T0,iv0,"CE")+bs(S0,Kp,T0,iv0,"PE"); legs=2
        if wing:
            cred-=bs(S0,Kcw,T0,iv0,"CE")+bs(S0,Kpw,T0,iv0,"PE"); legs=4
        if cred<2: continue

        # the sessions we hold through
        i0=sidx[day]
        hold=[]
        for s in sessions[i0+1:]:
            hold.append(s)
            sy,sm,sd=map(int,s.split("-"))
            left=(e-date(sy,sm,sd)).days
            if exit_friday and left<=4: break            # Friday
            if left<=0: break                            # expiry
        if not hold: continue

        val=cred; stopped=False
        for s in hold:
            blist = bars[s] if intraday_stop else [bars[s][-1]]
            for ts, Sx in blist:
                vxx = VIX5.get(ts, vday.get(s, v0))
                Tx = T_years(ts, e)
                ivx=(vxx/100.0)*ivm(Tx*365)
                val=bs(Sx,Kc,Tx,ivx,"CE")+bs(Sx,Kp,Tx,ivx,"PE")
                if wing: val-=bs(Sx,Kcw,Tx,ivx,"CE")+bs(Sx,Kpw,Tx,ivx,"PE")
                if val>=cred*STOP_X:
                    stopped=True; break
            if stopped: break
        pnl=(cred*(1-SLIP)-val*(1+SLIP))*QTY-legs*BROK
        margin=(wing*QTY+20000) if wing else 0.12*S0*QTY
        tr.append(dict(day=day, year=day[:4], pnl=pnl, stopped=stopped, margin=margin))
    return tr

def stats(tr, lab):
    v=np.array([t["pnl"] for t in tr],float)
    eq=np.cumsum(v); mdd=(eq-np.maximum.accumulate(eq)).min()
    yrs=len({t["year"] for t in tr}); ann=v.sum()/yrs
    stopped=100*np.mean([t["stopped"] for t in tr])
    ym=[np.mean([t["pnl"] for t in tr if t["year"]==y]) for y in sorted({t["year"] for t in tr})]
    print(f"{lab:<42s} {len(v):>4d} {v.mean():>+8.0f} {ann:>+9.0f} {100*(v>0).mean():>4.0f}% "
          f"{mdd:>+10.0f} {ann/abs(mdd):>6.2f} {v.min():>+9.0f} {stopped:>6.0f}% {sum(1 for x in ym if x>0):>2d}/{len(ym)}")

print("=== THE HONEST TEST: stop marked every 5 MINUTES, not at the daily close ===")
print("    Wed entry -> Fri exit | 2 lots | 1% slippage | stop = premium doubles | 2015-2026\n")
print(f"{'variant':<42s} {'n':>4s} {'mean':>8s} {'annual':>9s} {'win':>5s} "
      f"{'maxDD':>10s} {'Calmar':>6s} {'worst':>9s} {'stopped':>7s} {'+yrs':>5s}")
print("-"*118)
for w, wing in ((100,250),(150,250),(100,None)):
    lab = f"CONDOR {w}/{wing}" if wing else f"NAKED {w}pt"
    stats(run(w,wing,True,False), f"{lab}  DAILY-close stop  (the old, lazy way)")
    stats(run(w,wing,True,True),  f"{lab}  INTRADAY stop     (honest)")
    print()
print("DONE")
