#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/82 — SENSEX expiry-day systems on the REAL chain. I/O-lean version.

The DB is 7M SENSEX rows and a concurrent backfill is using the disk, so this:
  * resolves day -> nearest-expiry in ONE grouped scan
  * then touches ONLY the DTE0/DTE1 days (~23 of 60) and only near-ATM strikes
Systems (both, on SENSEX Thu=DTE0 and Wed=DTE1):
  A) ATM STRADDLE  (NAS-916 shape): 09:20 SELL ATM CE+PE, per-leg 30% stop, exit 15:15
  B) OTM STRANGLE  (NAS-OPT shape): 09:20 SELL ~0.4%-OTM strangle, +/-0.4% move-stop, exit 14:45
Points/day (gross) + Rs at SENSEX lot 20.
"""
import sqlite3, sys, time
from collections import defaultdict
from datetime import datetime
import numpy as np

ROOT="/home/arun/quantifyd"
c=sqlite3.connect(f"file:{ROOT}/backtest_data/options_data.db?mode=ro",uri=True)
c.row_factory=sqlite3.Row
SYM="SENSEX"; STEP=100; LOT=20; MOVE=0.004; SL=1.30
T_ENTRY=9*3600+20*60; T_STRAD=15*3600+15*60; T_STRANGLE=14*3600+45*60
def tsec(s): return int(s[11:13])*3600+int(s[14:16])*60+int(s[17:19])

t0=time.time()
# ONE scan: day -> nearest forward expiry
rows=c.execute("SELECT substr(snapshot_time,1,10) d, MIN(expiry_date) e FROM option_chain "
               "WHERE symbol=? AND expiry_date>=substr(snapshot_time,1,10) GROUP BY d ORDER BY d",(SYM,)).fetchall()
day_exp={r["d"]:r["e"] for r in rows}
print("day->expiry map: %d days (%.1fs)"%(len(day_exp),time.time()-t0)); sys.stdout.flush()

targets=[]
for d,e in day_exp.items():
    dte=(datetime.fromisoformat(e).date()-datetime.fromisoformat(d).date()).days
    if dte<=1: targets.append((d,e,dte))
print("DTE0/1 days to process: %d"%len(targets)); sys.stdout.flush()

res=defaultdict(list)
for i,(day,E,dte) in enumerate(sorted(targets)):
    ts=time.time()
    lo,hi=day+"T00:00",day+"T23:59"
    rows=c.execute("SELECT snapshot_time,strike,instrument_type,ltp,underlying_spot FROM option_chain "
                   "WHERE symbol=? AND snapshot_time>=? AND snapshot_time<=? AND expiry_date=? AND ltp>0 "
                   "ORDER BY snapshot_time",(SYM,lo,hi,E)).fetchall()
    if not rows:
        print("  [%d/%d] %s no rows"%(i+1,len(targets),day)); sys.stdout.flush(); continue
    chain=defaultdict(dict); spot={}
    for r in rows:
        t=tsec(r["snapshot_time"]); chain[t][(int(r["strike"]),r["instrument_type"])]=r["ltp"]
        if r["underlying_spot"]: spot[t]=float(r["underlying_spot"])
    times=sorted(chain)
    s0=next((spot[t] for t in times if t>=T_ENTRY and t in spot),None)
    if not s0:
        print("  [%d/%d] %s no spot"%(i+1,len(targets),day)); sys.stdout.flush(); continue
    def entry_px(K,ot):
        for t in times:
            if t>=T_ENTRY and (K,ot) in chain[t]: return chain[t][(K,ot)]
        return None
    def px_at(tt,K,ot,fb):
        best=fb
        for t in times:
            if t>tt: break
            if (K,ot) in chain[t]: best=chain[t][(K,ot)]
        return best
    # A) ATM straddle
    atm=round(s0/STEP)*STEP
    ce0,pe0=entry_px(atm,"CE"),entry_px(atm,"PE")
    if ce0 and pe0:
        def stop_ex(K,ot,e):
            cap=e*SL
            for t in times:
                if T_ENTRY<=t<=T_STRAD:
                    p=chain[t].get((K,ot))
                    if p is not None and p>=cap: return p
            return px_at(T_STRAD,K,ot,e)
        res[("ATM straddle",dte)].append((ce0-stop_ex(atm,"CE",ce0))+(pe0-stop_ex(atm,"PE",pe0)))
    # B) OTM strangle
    ck,pk=round(s0*(1+MOVE)/STEP)*STEP, round(s0*(1-MOVE)/STEP)*STEP
    ce0,pe0=entry_px(ck,"CE"),entry_px(pk,"PE")
    if ce0 and pe0:
        et=None
        for t in times:
            if T_ENTRY<=t<=T_STRANGLE and spot.get(t) and abs(spot[t]-s0)/s0>=MOVE: et=t; break
        tx=et or T_STRANGLE
        res[("OTM strangle",dte)].append((ce0-px_at(tx,ck,"CE",ce0))+(pe0-px_at(tx,pk,"PE",pe0)))
    print("  [%d/%d] %s DTE%d ok (%.1fs)"%(i+1,len(targets),day,dte,time.time()-ts)); sys.stdout.flush()
c.close()

print("\n=== SENSEX expiry-day systems — REAL chain ===")
print("    entry 09:20 | points/day (gross) | Rs at lot 20\n")
print(f"{'system':<15s} {'DTE':>3s} {'day':>3s} {'n':>3s} {'total':>8s} {'mean':>7s} {'win%':>5s} {'Rs/day':>9s}")
for k in sorted(res):
    s_,dte=k; v=np.array(res[k]); wd="Thu" if dte==0 else "Wed"
    print(f"{s_:<15s} {dte:>3d} {wd:>3s} {len(v):>3d} {v.sum():>+8.1f} {v.mean():>+7.1f} {100*(v>0).mean():>4.0f}% {v.mean()*LOT:>+9.0f}")
print("\n--- combined Wed+Thu per system ---")
for s_ in ["ATM straddle","OTM strangle"]:
    a=res[(s_,0)]+res[(s_,1)]
    if not a: continue
    v=np.array(a)
    print(f"  {s_:<15s} n={len(v):>2d}  total {v.sum():>+7.1f} pts  mean {v.mean():>+6.1f} pts/day  win {100*(v>0).mean():>3.0f}%  Rs {v.mean()*LOT:>+7.0f}/day")
print("\n(11-12 days per cell is THIN — first real-chain look, not a verdict.)")
print("DONE")
