#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/81 — SENSEX expiry-day straddle/strangle on the REAL recorded chain.

The plan (Arun): NIFTY expires Tuesday, SENSEX Thursday. NAS already harvests NIFTY's 0/1-DTE
edge (Mon/Tue). Run the SAME expiry-day systems on SENSEX to fill Wed(DTE1)/Thu(DTE0) with the
idle capital. This tests it on the 59 days of REAL SENSEX chain we have (no vol proxy needed).

Systems, each on SENSEX DTE0 (Thu) and DTE1 (Wed):
  A) ATM STRADDLE (the NAS-916 shape): 09:20 SELL ATM CE+PE, per-leg 30% stop, exit 15:15.
  B) OTM STRANGLE (the NAS-OPT shape): 09:20 SELL ~0.4%-OTM strangle, +/-0.4% underlying move-stop,
     exit 14:45.
Reported in POINTS/day (cross-index comparable) and Rs at lot 20. Real chain, real premiums.
"""
import sqlite3
from collections import defaultdict
from datetime import datetime

import numpy as np

ROOT = "/home/arun/quantifyd"
ODB = f"{ROOT}/backtest_data/options_data.db"
SYM = "SENSEX"
STEP = 100                 # SENSEX strike step
LOT = 20                   # SENSEX lot (report Rs at this; points are lot-agnostic)
BROK_PTS = 0.0             # compare gross first; costs added in the summary
MOVE = 0.004               # 0.4% underlying move-stop (strangle)
SL_MULT = 1.30             # 30% per-leg stop (straddle)

c = sqlite3.connect(f"file:{ODB}?mode=ro", uri=True)
c.row_factory = sqlite3.Row


def tsec(s):
    return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def snap_at(day, tt, expiry, strike, ot):
    r = c.execute("SELECT ltp FROM option_chain WHERE symbol=? AND expiry_date=? AND strike=? "
                  "AND instrument_type=? AND substr(snapshot_time,1,10)=? AND snapshot_time<=? "
                  "AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
                  (SYM, expiry, strike, ot, day, f"{day}T{tt}:59")).fetchone()
    return float(r[0]) if r and r[0] else None


def spot_series(day, expiry):
    """(sec, spot, ce_at_strike?, ...) — we fetch per-need; here just the spot path."""
    rows = c.execute("SELECT DISTINCT snapshot_time, underlying_spot FROM option_chain WHERE symbol=? "
                     "AND substr(snapshot_time,1,10)=? AND underlying_spot IS NOT NULL "
                     "ORDER BY snapshot_time", (SYM, day)).fetchall()
    return [(tsec(r[0]), float(r[1]), r[0]) for r in rows]


def leg_path(day, expiry, strike, ot, t0, t1):
    rows = c.execute("SELECT snapshot_time, ltp FROM option_chain WHERE symbol=? AND expiry_date=? "
                     "AND strike=? AND instrument_type=? AND substr(snapshot_time,1,10)=? AND ltp>0 "
                     "ORDER BY snapshot_time", (SYM, expiry, strike, ot, day)).fetchall()
    return [(tsec(r[0]), float(r[1])) for r in rows if t0 <= tsec(r[0]) <= t1]


days = [r[0] for r in c.execute("SELECT DISTINCT substr(snapshot_time,1,10) d FROM option_chain "
                                "WHERE symbol=? ORDER BY d", (SYM,))]

ENTRY_T, STRAD_EXIT, STRANGLE_EXIT = "09:20:00", "15:15:00", "14:45:00"
res = defaultdict(list)   # (system, dte) -> [dict per day]

for day in days:
    wd = datetime.fromisoformat(day).weekday()
    # entry snapshot
    ex = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=? "
                 "AND substr(snapshot_time,1,10)=? AND expiry_date>=?", (SYM, day, day))})
    if not ex:
        continue
    expiry = ex[0]
    dte = (datetime.fromisoformat(expiry).date() - datetime.fromisoformat(day).date()).days
    if dte > 1:
        continue                          # only expiry-day systems: DTE0/DTE1
    sp = spot_series(day, expiry)
    s0 = next((s for t, s, _ in sp if t >= tsec(f"{day}T{ENTRY_T}")), None)
    if not s0:
        continue
    atm = round(s0 / STEP) * STEP

    # ---------- A) ATM straddle ----------
    ce0 = snap_at(day, "09:20", expiry, atm, "CE")
    pe0 = snap_at(day, "09:20", expiry, atm, "PE")
    if ce0 and pe0:
        t0 = tsec(f"{day}T{ENTRY_T}"); t1 = tsec(f"{day}T{STRAD_EXIT}")
        cep = leg_path(day, expiry, atm, "CE", t0, t1)
        pep = leg_path(day, expiry, atm, "PE", t0, t1)
        def exit_leg(path, e):
            cap = e * SL_MULT
            for t, p in path:
                if p >= cap:
                    return p
            return path[-1][1] if path else e
        cx = exit_leg(cep, ce0); px = exit_leg(pep, pe0)
        pnl = (ce0 - cx) + (pe0 - px)
        res[("ATM straddle", dte)].append(dict(day=day, pnl=pnl, credit=ce0 + pe0))

    # ---------- B) OTM strangle (~0.4% OTM) ----------
    ck = round((s0 * (1 + MOVE)) / STEP) * STEP
    pk = round((s0 * (1 - MOVE)) / STEP) * STEP
    ce0 = snap_at(day, "09:20", expiry, ck, "CE")
    pe0 = snap_at(day, "09:20", expiry, pk, "PE")
    if ce0 and pe0:
        t0 = tsec(f"{day}T{ENTRY_T}"); t1 = tsec(f"{day}T{STRANGLE_EXIT}")
        # walk spot; exit both on 0.4% move, else time-exit
        exit_t = None
        for t, s, _ in sp:
            if t < t0 or t > t1:
                continue
            if abs(s - s0) / s0 >= MOVE:
                exit_t = t; break
        cx = snap_at(day, f"{exit_t//3600:02d}:{(exit_t%3600)//60:02d}" if exit_t else "14:45", expiry, ck, "CE") or ce0
        px = snap_at(day, f"{exit_t//3600:02d}:{(exit_t%3600)//60:02d}" if exit_t else "14:45", expiry, pk, "PE") or pe0
        pnl = (ce0 - cx) + (pe0 - px)
        res[("OTM strangle", dte)].append(dict(day=day, pnl=pnl, credit=ce0 + pe0,
                                               stopped=exit_t is not None))
c.close()

print("=== SENSEX expiry-day systems on the REAL chain (59 days: 11 Thu / 12 Wed) ===")
print("    entry 09:20 | points/day, gross | lot 20 for the Rs column\n")
print(f"{'system':<16s} {'DTE':>3s} {'wd':>3s} {'n':>3s} {'total pts':>10s} {'mean pts':>9s} "
      f"{'win%':>5s} {'Rs/day (lot20)':>15s}")
for (sysn, dte) in sorted(res):
    sub = res[(sysn, dte)]
    if not sub:
        continue
    v = np.array([x["pnl"] for x in sub])
    wd = "Thu" if dte == 0 else "Wed"
    print(f"{sysn:<16s} {dte:>3d} {wd:>3s} {len(v):>3d} {v.sum():>+10.1f} {v.mean():>+9.1f} "
          f"{100*(v>0).mean():>4.0f}% {v.mean()*LOT:>+15.0f}")

print("\n--- combined DTE0+DTE1 per system (the full Wed+Thu book) ---")
for sysn in ["ATM straddle", "OTM strangle"]:
    alld = res[(sysn, 0)] + res[(sysn, 1)]
    if not alld:
        continue
    v = np.array([x["pnl"] for x in alld])
    print(f"  {sysn:<16s} n={len(v):>3d}  total {v.sum():>+8.1f} pts  mean {v.mean():>+6.1f} pts/day  "
          f"win {100*(v>0).mean():>3.0f}%  = Rs {v.mean()*LOT:>+7.0f}/day (lot20)")

print("\n--- honest read: 11-12 days per cell is THIN. This is a first look, not a verdict. ---")
print("DONE")
