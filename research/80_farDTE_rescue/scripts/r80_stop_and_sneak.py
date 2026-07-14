#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G10 — (a) a REACHABLE stop for the condor, (b) the sneak-in decay trade.

(a) Writing out the worked example exposed that the "2x credit" stop is DEAD on the condor: 2x a
    ~150pt credit is ~300 points, but the structure can never be worth more than the 250-pt wing
    width. The stop is above the ceiling, so it can never fire (backtest: 1%). The wings ARE the
    stop. So: does a stop set BELOW the wing width actually help, or does it just cut winners?
    Sweep the stop as an absolute structure value: none / 225 / 200 / 175 / 150 / 125 pts.

(b) The decay clock (real chain) says decay is LUMPY: on far-DTE days ~14% of the day's decay lands
    in 15:15-15:30 and ~13% in 13:45-14:00, and Wednesday's last two hours decay 2.4x faster than
    its midday. But that was measured only on windows where the index BARELY MOVED -- i.e. it is the
    reward with the risk removed. So now trade it for real: sell the ~100pt-OTM strangle at time T,
    buy it back at 15:25, on EVERY far-DTE day including the ones that move.
    Real recorded quotes only (58 days) -- Black-Scholes cannot show an IV-crush spike, so the engine
    would be useless here, and 33 far-DTE days is thin. Reported as such.
"""
import json, math, sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np

ROOT = "/home/arun/quantifyd"
CAL = json.load(open(f"{ROOT}/research/80_farDTE_rescue/results/engine_calib.json"))["iv_mult_by_dte"]
QTY, BROK, SLIP = 130, 20, 0.01

# ────────────────────────────── (a) the reachable stop ──────────────────────────────
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
n5=c.execute("SELECT date, close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date").fetchall()
v5={str(d):v for d,v in c.execute("SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='5minute'")}
vday={r[0][:10]:r[1] for r in c.execute("SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
c.close()
bars=defaultdict(list)
for d,cl in n5: bars[str(d)[:10]].append((str(d),cl))
for k in bars: bars[k].sort()
sess=sorted(bars); sidx={d:i for i,d in enumerate(sess)}

def T_yr(ts, exp):
    dt=datetime.fromisoformat(ts)
    return max((datetime(exp.year,exp.month,exp.day,15,30)-dt).total_seconds()/(365*24*3600),1e-6)

WING=250
def run_stop(stop_pts):
    tr=[]
    for day in sess:
        y,m,dd=map(int,day.split("-")); d0=date(y,m,dd); e=nx(d0)
        if (e-d0).days!=6: continue
        ts0,S0=bars[day][-1]
        v0=v5.get(ts0, vday.get(day))
        if not v0: continue
        T0=T_yr(ts0,e); iv0=(v0/100)*ivm(T0*365)
        Kc,Kp=round((S0+100)/50)*50, round((S0-100)/50)*50
        Kcw,Kpw=Kc+WING,Kp-WING
        cred=(bs(S0,Kc,T0,iv0,"CE")+bs(S0,Kp,T0,iv0,"PE")
              -bs(S0,Kcw,T0,iv0,"CE")-bs(S0,Kpw,T0,iv0,"PE"))
        if cred<2: continue
        hold=[]
        for s in sess[sidx[day]+1:]:
            hold.append(s); sy,sm,sd=map(int,s.split("-"))
            if (e-date(sy,sm,sd)).days<=4: break        # exit Friday
        val=cred; stopped=False
        for s in hold:
            for ts,Sx in bars[s]:
                vv=v5.get(ts, vday.get(s,v0)); Tx=T_yr(ts,e); ivx=(vv/100)*ivm(Tx*365)
                val=(bs(Sx,Kc,Tx,ivx,"CE")+bs(Sx,Kp,Tx,ivx,"PE")
                     -bs(Sx,Kcw,Tx,ivx,"CE")-bs(Sx,Kpw,Tx,ivx,"PE"))
                if stop_pts and val>=stop_pts:
                    stopped=True; break
            if stopped: break
        pnl=(cred*(1-SLIP)-val*(1+SLIP))*QTY-4*BROK
        tr.append(dict(year=day[:4],pnl=pnl,stopped=stopped))
    return tr

def show(tr,lab):
    v=np.array([t["pnl"] for t in tr],float)
    eq=np.cumsum(v); mdd=(eq-np.maximum.accumulate(eq)).min()
    yrs=len({t["year"] for t in tr}); ann=v.sum()/yrs
    ym=[np.mean([t["pnl"] for t in tr if t["year"]==y]) for y in sorted({t["year"] for t in tr})]
    print(f"{lab:<28s} {len(v):>4d} {v.mean():>+8.0f} {ann:>+9.0f} {100*(v>0).mean():>4.0f}% "
          f"{mdd:>+10.0f} {ann/abs(mdd):>6.2f} {v.min():>+9.0f} {100*np.mean([t['stopped'] for t in tr]):>6.0f}% "
          f"{sum(1 for x in ym if x>0):>2d}/{len(ym)}")

print("=== (a) A REACHABLE STOP FOR THE CONDOR (wing width = 250, so anything >=250 can never fire) ===")
print("    Wed entry -> Fri exit | 100pt shorts / 250pt wings | marked every 5 min\n")
print(f"{'stop (structure value)':<28s} {'n':>4s} {'mean':>8s} {'annual':>9s} {'win':>5s} "
      f"{'maxDD':>10s} {'Calmar':>6s} {'worst':>9s} {'fires':>6s} {'+yrs':>5s}")
print("-"*112)
show(run_stop(None), "NO STOP (wings only)")
for sp in (225, 200, 175, 150, 125):
    show(run_stop(sp), f"close if worth >= {sp} pts")

# ────────────────────────────── (b) the sneak-in decay trade ──────────────────────────────
print("\n\n=== (b) THE SNEAK-IN: sell the far-DTE strangle inside the fast-decay window only ===")
print("    REAL recorded quotes | ~100pt-OTM strangle | exit 15:25 | 2 lots | Rs20/leg + 1% slippage")
print("    NOTE: only 58 recorded days -> ~33 far-DTE. This is THIN. Treat as a look, not a verdict.\n")

oc=sqlite3.connect(f"file:{ROOT}/backtest_data/options_data.db?mode=ro",uri=True)
oc.row_factory=sqlite3.Row

def snap_at(day, t, exp, Kc, Kp):
    ts=oc.execute("SELECT MAX(snapshot_time) FROM option_chain WHERE symbol='NIFTY' "
                  "AND date(snapshot_time)=? AND time(snapshot_time)<=?", (day,t)).fetchone()[0]
    if not ts: return None,None
    rows=oc.execute("SELECT strike,instrument_type,ltp,underlying_spot FROM option_chain "
                    "WHERE symbol='NIFTY' AND snapshot_time=? AND expiry_date=? AND strike IN (?,?) AND ltp>0",
                    (ts,exp,Kc,Kp)).fetchall()
    bk={(int(r["strike"]),r["instrument_type"]):r["ltp"] for r in rows}
    sp=next((r["underlying_spot"] for r in rows if r["underlying_spot"]),None)
    if (Kc,"CE") not in bk or (Kp,"PE") not in bk: return None,None
    return bk[(Kc,"CE")]+bk[(Kp,"PE")], sp

days=[r[0] for r in oc.execute("SELECT DISTINCT date(snapshot_time) FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")]
ENTRIES=["10:00:00","13:00:00","13:45:00","14:00:00","14:30:00","14:45:00","15:00:00"]
res=defaultdict(list)
for day in days:
    exp=oc.execute("SELECT MIN(expiry_date) FROM option_chain WHERE symbol='NIFTY' "
                   "AND date(snapshot_time)=? AND expiry_date>=?",(day,day)).fetchone()[0]
    if not exp: continue
    dte=(datetime.fromisoformat(exp).date()-datetime.fromisoformat(day).date()).days
    if dte<4: continue                                   # far-DTE only
    _,sp0=snap_at(day,"09:20:00",exp,0,0) or (None,None)
    r0=oc.execute("SELECT underlying_spot FROM option_chain WHERE symbol='NIFTY' "
                  "AND date(snapshot_time)=? AND underlying_spot IS NOT NULL LIMIT 1",(day,)).fetchone()
    if not r0: continue
    wd=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][datetime.fromisoformat(day).weekday()]
    for t in ENTRIES:
        # strikes chosen from the spot AT ENTRY (as you would live)
        rs=oc.execute("SELECT MAX(snapshot_time) FROM option_chain WHERE symbol='NIFTY' "
                      "AND date(snapshot_time)=? AND time(snapshot_time)<=?", (day,t)).fetchone()[0]
        if not rs: continue
        sp=oc.execute("SELECT underlying_spot FROM option_chain WHERE symbol='NIFTY' AND snapshot_time=? "
                      "AND underlying_spot IS NOT NULL LIMIT 1",(rs,)).fetchone()
        if not sp: continue
        spot=sp[0]; atm=round(spot/50)*50
        Kc,Kp=atm+100,atm-100
        p_in,_=snap_at(day,t,exp,Kc,Kp)
        p_out,_=snap_at(day,"15:25:00",exp,Kc,Kp)
        if not p_in or not p_out: continue
        pnl=(p_in*(1-SLIP)-p_out*(1+SLIP))*QTY-2*BROK
        res[t].append(dict(day=day,wd=wd,dte=dte,pnl=pnl,cred=p_in))

print(f"{'sell at':>9s} {'n':>4s} {'mean':>8s} {'median':>8s} {'win%':>6s} {'worst':>9s} {'best':>8s} "
      f"{'avg credit':>11s}   {'Wed':>7s} {'Thu':>7s} {'Fri':>7s}")
for t in ENTRIES:
    tl=res[t]
    if len(tl)<10: continue
    v=np.array([x["pnl"] for x in tl],float)
    line=(f"{t[:5]:>9s} {len(v):>4d} {v.mean():>+8.0f} {np.median(v):>+8.0f} {100*(v>0).mean():>5.0f}% "
          f"{v.min():>+9.0f} {v.max():>+8.0f} {np.mean([x['cred'] for x in tl]):>11.1f}   ")
    for wd in ["Wed","Thu","Fri"]:
        w=[x["pnl"] for x in tl if x["wd"]==wd]
        line+=f"{np.mean(w):>+7.0f} " if w else "      - "
    print(line)
oc.close()
print("\nDONE")
