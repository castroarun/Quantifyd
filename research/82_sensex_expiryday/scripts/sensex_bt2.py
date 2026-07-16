#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/81 — SENSEX expiry-day systems on the REAL chain (fast: one query per day).

Both systems, on SENSEX DTE0 (Thu) and DTE1 (Wed):
  A) ATM STRADDLE  (NAS-916 shape): 09:20 SELL ATM CE+PE, per-leg 30% stop, exit 15:15.
  B) OTM STRANGLE  (NAS-OPT shape): 09:20 SELL ~0.4%-OTM strangle, +/-0.4% move-stop, exit 14:45.
Real premiums; reported in points/day (cross-index comparable) + Rs at SENSEX lot 20.
"""
import sqlite3
from collections import defaultdict
from datetime import datetime
import numpy as np

ROOT="/home/arun/quantifyd"
c=sqlite3.connect(f"file:{ROOT}/backtest_data/options_data.db?mode=ro",uri=True); c.row_factory=sqlite3.Row
SYM="SENSEX"; STEP=100; LOT=20; MOVE=0.004; SL=1.30

def tsec(s): return int(s[11:13])*3600+int(s[14:16])*60+int(s[17:19])
def hhmm(sec): return sec//60      # minute bucket

days=[r[0] for r in c.execute("SELECT DISTINCT substr(snapshot_time,1,10) d FROM option_chain WHERE symbol=? ORDER BY d",(SYM,))]
res=defaultdict(list)
T_ENTRY=9*3600+20*60; T_STRAD=15*3600+15*60; T_STRANGLE=14*3600+45*60

import sys
for _i,day in enumerate(days):
    print("  [%d/%d] %s"%(_i+1,len(days),day)); sys.stdout.flush()
    wd=datetime.fromisoformat(day).weekday()
    lo=day+chr(84)+chr(48)+chr(48)+chr(58)+chr(48)+chr(48); hi=day+chr(84)+chr(50)+chr(51)+chr(58)+chr(53)+chr(57); exps=sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=? AND snapshot_time>=? AND snapshot_time<=? AND expiry_date>=?",(SYM,lo,hi,day))})
    if not exps: continue
    E=exps[0]; dte=(datetime.fromisoformat(E).date()-datetime.fromisoformat(day).date()).days
    if dte>1: continue
    # one query: the whole day's nearest-expiry chain + spot
    rows=c.execute("SELECT snapshot_time, strike, instrument_type, ltp, underlying_spot FROM option_chain "
                   "WHERE symbol=? AND snapshot_time>=? AND snapshot_time<=? AND expiry_date=? AND ltp>0 ORDER BY snapshot_time",
                   (SYM,lo,hi,E)).fetchall()
    if not rows: continue
    # per-minute: {(strike,type):ltp}, and spot path
    chain=defaultdict(dict); spot={}
    for r in rows:
        t=tsec(r["snapshot_time"]); chain[t][(int(r["strike"]),r["instrument_type"])]=r["ltp"]
        if r["underlying_spot"]: spot[t]=float(r["underlying_spot"])
    times=sorted(chain)
    # spot at entry
    s0=next((spot[t] for t in times if t>=T_ENTRY and t in spot),None)
    if not s0: continue
    def price(t_target, strike, ot, fallback):
        # nearest snapshot at/just before t_target with that leg
        best=None
        for t in times:
            if t>t_target: break
            if (strike,ot) in chain[t]: best=chain[t][(strike,ot)]
        return best if best is not None else fallback
    def entry_price(strike, ot):
        for t in times:
            if t>=T_ENTRY and (strike,ot) in chain[t]: return chain[t][(strike,ot)]
        return None

    # ---- A) ATM straddle ----
    atm=round(s0/STEP)*STEP
    ce0=entry_price(atm,"CE"); pe0=entry_price(atm,"PE")
    if ce0 and pe0:
        def stop_exit(strike,ot,e):
            cap=e*SL
            for t in times:
                if t<T_ENTRY or t>T_STRAD: continue
                p=chain[t].get((strike,ot))
                if p is not None and p>=cap: return p
            return price(T_STRAD,strike,ot,e)
        cx=stop_exit(atm,"CE",ce0); px=stop_exit(atm,"PE",pe0)
        res[("ATM straddle",dte)].append((ce0-cx)+(pe0-px))

    # ---- B) OTM strangle ----
    ck=round(s0*(1+MOVE)/STEP)*STEP; pk=round(s0*(1-MOVE)/STEP)*STEP
    ce0=entry_price(ck,"CE"); pe0=entry_price(pk,"PE")
    if ce0 and pe0:
        et=None
        for t in times:
            if t<T_ENTRY or t>T_STRANGLE: continue
            s=spot.get(t)
            if s and abs(s-s0)/s0>=MOVE: et=t; break
        tx=et if et else T_STRANGLE
        cx=price(tx,ck,"CE",ce0); px=price(tx,pk,"PE",pe0)
        res[("OTM strangle",dte)].append((ce0-cx)+(pe0-px))
c.close()

print("=== SENSEX expiry-day systems — REAL chain, 59 days (11 Thu / 12 Wed) ===")
print("    entry 09:20 | points/day (gross) | Rs at lot 20\n")
print(f"{'system':<15s} {'DTE':>3s} {'day':>3s} {'n':>3s} {'total':>8s} {'mean':>7s} {'win%':>5s} {'Rs/day':>9s}")
for k in sorted(res):
    sysn,dte=k; v=np.array(res[k]); wd="Thu" if dte==0 else "Wed"
    print(f"{sysn:<15s} {dte:>3d} {wd:>3s} {len(v):>3d} {v.sum():>+8.1f} {v.mean():>+7.1f} {100*(v>0).mean():>4.0f}% {v.mean()*LOT:>+9.0f}")

print("\n--- combined Wed+Thu book, per system ---")
for sysn in ["ATM straddle","OTM strangle"]:
    a=res[(sysn,0)]+res[(sysn,1)]
    if not a: continue
    v=np.array(a)
    print(f"  {sysn:<15s} n={len(v):>2d}  total {v.sum():>+7.1f} pts  mean {v.mean():>+6.1f} pts/day  win {100*(v>0).mean():>3.0f}%  Rs {v.mean()*LOT:>+7.0f}/day")

print("\n(11-12 days/cell is THIN — a first real-chain look, not a verdict.)")
print("DONE")
