#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G7 — deploy the IDLE capital, and give it back before NAS-OPT needs it.

Arun's point, and it reframes the whole thing: NAS-OPT only trades 0/1-DTE (Mon + Tue). The margin
sits IDLE Wed / Thu / Fri. So the question is not "what is the best condor" -- it is:

    what earns on the days the capital is doing nothing, and is FLAT again before Monday?

The condor from G5 holds Wed -> Tue expiry, which COLLIDES with NAS-OPT on Mon and Tue -- it needs
margin exactly when NAS-OPT does. So price the exit-day variants, and judge them on RETURN PER
MARGIN-DAY, not on P&L:

    exit FRIDAY  (dte_left 4) -- flat over the weekend, zero collision. Gives up weekend theta.
    exit MONDAY  (dte_left 1) -- captures the WEEKEND decay (3 calendar days, no market risk
                                except the Monday gap). Collides with NAS-OPT on Monday only.
    hold to TUESDAY EXPIRY    -- maximum theta, maximum collision.

Weekend theta is the interesting one: Fri->Mon is 3 calendar days of decay for ONE session of gap
risk. That is the best decay-per-unit-risk in the week, and exiting on Friday throws it away.
Reads only.
"""
import json, math, sqlite3
from datetime import date, timedelta
import numpy as np

ROOT = "/home/arun/quantifyd"
CAL = json.load(open(f"{ROOT}/research/80_farDTE_rescue/results/engine_calib.json"))["iv_mult_by_dte"]
QTY, BROK, SLIP = 130, 20, 0.01

def ncdf(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
def bs(S,K,T,iv,k):
    if T<=0 or iv<=0: return max(0.0,(S-K) if k=="CE" else (K-S))
    d1=(math.log(S/K)+0.5*iv*iv*T)/(iv*math.sqrt(T)); d2=d1-iv*math.sqrt(T)
    return S*ncdf(d1)-K*ncdf(d2) if k=="CE" else K*ncdf(-d2)-S*ncdf(-d1)
def ivm(d):
    if str(d) in CAL: return CAL[str(d)]
    ks=sorted(int(k) for k in CAL); return CAL[str(min(ks,key=lambda k:abs(k-d)))]
def nx(d):
    t=1 if d>=date(2025,9,1) else 3; a=(t-d.weekday())%7
    return d if a==0 else d+timedelta(days=a)

c=sqlite3.connect(f"file:{ROOT}/backtest_data/market_data.db?mode=ro",uri=True)
px={r[0][:10]:r[1] for r in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day'")}
vx={r[0][:10]:r[1] for r in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
c.close()
sess=sorted(d for d in px if d in vx); idx={d:i for i,d in enumerate(sess)}

# exit policies, expressed as "get out once this many days to expiry remain"
POLICIES = [("exit FRIDAY (flat for NAS-OPT)", 4),
            ("exit MONDAY (keeps weekend theta)", 1),
            ("hold to TUESDAY expiry", 0)]

def run(width, wing, exit_at):
    tr=[]
    for day in sess:
        y,m,dd=map(int,day.split("-")); d0=date(y,m,dd); e=nx(d0)
        if (e-d0).days!=6: continue          # enter Wednesday
        es=e.isoformat()
        if es not in idx: continue
        S0,v0=px[day],vx[day]; iv0=(v0/100)*ivm(6); T0=6/365
        Kc,Kp=round((S0+width)/50)*50, round((S0-width)/50)*50
        Kcw,Kpw=(Kc+wing,Kp-wing) if wing else (None,None)
        cred=bs(S0,Kc,T0,iv0,"CE")+bs(S0,Kp,T0,iv0,"PE"); legs=2
        if wing:
            cred-=bs(S0,Kcw,T0,iv0,"CE")+bs(S0,Kpw,T0,iv0,"PE"); legs=4
        if cred<2: continue
        val=cred; held=0; exit_day=day
        for s in sess[idx[day]+1:idx[es]+1]:
            Sx,vv=px[s],vx.get(s,v0); sy,sm,sd=map(int,s.split("-"))
            left=(e-date(sy,sm,sd)).days
            held+=1; exit_day=s
            if left<=0:
                val=max(0.,Sx-Kc)+max(0.,Kp-Sx)
                if wing: val-=max(0.,Sx-Kcw)+max(0.,Kpw-Sx)
                break
            ivv=(vv/100)*ivm(left); Tx=left/365
            val=bs(Sx,Kc,Tx,ivv,"CE")+bs(Sx,Kp,Tx,ivv,"PE")
            if wing: val-=bs(Sx,Kcw,Tx,ivv,"CE")+bs(Sx,Kpw,Tx,ivv,"PE")
            if val>=cred*2.0: break            # stop: premium doubled
            if left<=exit_at: break            # the capital-aware exit
        pnl=(cred*(1-SLIP)-val*(1+SLIP))*QTY-legs*BROK
        margin=(wing*QTY+20000) if wing else 0.12*S0*QTY
        tr.append(dict(day=day,year=day[:4],pnl=pnl,margin=margin,held=held,exit_day=exit_day))
    return tr

def show(tr,lab):
    v=np.array([t["pnl"] for t in tr],float)
    eq=np.cumsum(v); mdd=(eq-np.maximum.accumulate(eq)).min()
    yrs=len({t["year"] for t in tr}); ann=v.sum()/yrs
    marg=np.mean([t["margin"] for t in tr]); mdays=np.mean([t["held"] for t in tr])
    # return per rupee of margin per day it is tied up -- the metric Arun actually cares about
    rpmd = v.mean()/(marg*mdays)
    print(f"{lab:<36s} {len(v):>4d} {v.mean():>+8.0f} {ann:>+9.0f} {100*(v>0).mean():>4.0f}% "
          f"{mdd:>+10.0f} {ann/abs(mdd):>6.2f} {mdays:>6.1f} {10000*rpmd:>9.1f}")

print("=== DEPLOY THE IDLE CAPITAL: enter Wed, get out before NAS-OPT needs the margin back ===")
print("    2 lots (130 qty) | 1% slippage | stop = premium doubles | 2015-2026\n")
print(f"{'policy':<36s} {'n':>4s} {'mean':>8s} {'annual':>9s} {'win':>5s} "
      f"{'maxDD':>10s} {'Calmar':>6s} {'mgn-days':>6s} {'bp/mgn/day':>9s}")
print("-"*104)
for lab,ex in POLICIES:
    show(run(150,250,ex), f"CONDOR 150/250 | {lab}")
print()
for lab,ex in POLICIES:
    show(run(100,250,ex), f"CONDOR 100/250 | {lab}")
print()
for lab,ex in POLICIES:
    show(run(100,None,ex), f"NAKED  100pt   | {lab}")

print("\n\n--- what the WEEKEND is actually worth (Friday close -> Monday close, position unchanged) ---")
tr_f=run(150,250,4); tr_m=run(150,250,1)
mf=np.mean([t["pnl"] for t in tr_f]); mm=np.mean([t["pnl"] for t in tr_m])
print(f"  exit Friday : mean {mf:+,.0f}")
print(f"  exit Monday : mean {mm:+,.0f}")
print(f"  the weekend hold is worth {mm-mf:+,.0f} per trade "
      f"({'WORTH IT' if mm>mf else 'NOT worth the Monday gap risk'})")
print("\nDONE")
