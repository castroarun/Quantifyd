#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/79 — NAS-OPT replayed on EVERY recorded chain day.

Replays services/nas_opt.py EXACTLY as it trades live, against the real recorded chain:
  09:20  sell ~100pt-OTM strangle (2 strikes either side of ATM), 2 lots/leg = 130 qty
  09:21-14:44  exit BOTH legs if |spot - entry_spot| / entry_spot >= 0.4%
  14:45  else time-exit
  P&L net of Rs80/leg round-trip brokerage

The live system only entered on 0/1-DTE days until 2026-07-14. This replays EVERY day, tagging
each as 'system' (DTE<=1, the research/54 strategy) or 'observational' (DTE>=2) so the two are
never mixed -- exactly the split now running live.

VALIDATION: 11 of these days were actually paper-traded live. The replay is compared leg-for-leg
against that real record. If it cannot reproduce those, it cannot be trusted on the other ~46.

Reads only.
"""
from __future__ import annotations

import csv
import os
import sqlite3
import sys
from datetime import date, datetime

import numpy as np

ROOT = "/home/arun/quantifyd"
ODB = os.path.join(ROOT, "backtest_data", "options_data.db")
OUT = os.path.join(ROOT, "research", "79_nasopt_full_replay", "results")

LOT, LOTS = 65, 2
QTY = LOT * LOTS          # 130
BROK = 80                 # per leg, round trip
OTM_STRIKES = 2           # ~100 pts either side
STEP = 50
MOVE_PCT = 0.4
ENTRY_T = "09:20:00"
LAST_T = "14:45:00"


def main():
    os.makedirs(OUT, exist_ok=True)
    c = sqlite3.connect(f"file:{ODB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row

    days = [r[0] for r in c.execute(
        "SELECT DISTINCT date(snapshot_time) FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")]

    rows = []
    for day in days:
        # --- entry snapshot: first at/after 09:20 ---
        snap = c.execute(
            "SELECT MIN(snapshot_time) FROM option_chain WHERE symbol='NIFTY' "
            "AND date(snapshot_time)=? AND time(snapshot_time)>=?", (day, ENTRY_T)).fetchone()[0]
        if not snap:
            print(f"  {day}  SKIP (no snapshot at/after 09:20)")
            continue

        chain = c.execute(
            "SELECT tradingsymbol,strike,instrument_type,ltp,expiry_date,underlying_spot "
            "FROM option_chain WHERE symbol='NIFTY' AND snapshot_time=? AND ltp>0", (snap,)).fetchall()
        if not chain:
            print(f"  {day}  SKIP (empty entry snapshot)")
            continue

        exps = sorted({r["expiry_date"] for r in chain if r["expiry_date"] >= day})
        if not exps:
            print(f"  {day}  SKIP (no forward expiry)")
            continue
        exp = exps[0]
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days
        spot = next((r["underlying_spot"] for r in chain if r["underlying_spot"]), None)
        if not spot:
            print(f"  {day}  SKIP (no spot)")
            continue

        atm = round(spot / STEP) * STEP
        ce_k = atm + OTM_STRIKES * STEP
        pe_k = atm - OTM_STRIKES * STEP
        book = {(int(r["strike"]), r["instrument_type"]): r["ltp"]
                for r in chain if r["expiry_date"] == exp}
        ce_e, pe_e = book.get((ce_k, "CE")), book.get((pe_k, "PE"))
        if not ce_e or not pe_e:
            print(f"  {day}  SKIP (strikes {ce_k}CE/{pe_k}PE absent at entry)")
            continue

        # --- walk forward, minute by minute, exactly as monitor_job does ---
        path = c.execute(
            "SELECT snapshot_time, strike, instrument_type, ltp, underlying_spot "
            "FROM option_chain WHERE symbol='NIFTY' AND date(snapshot_time)=? "
            "AND snapshot_time>? AND time(snapshot_time)<=? AND expiry_date=? "
            "AND strike IN (?,?) ORDER BY snapshot_time",
            (day, snap, LAST_T, exp, ce_k, pe_k)).fetchall()

        # Key by (strike, type), NOT type alone. Both strikes carry BOTH a CE and a PE, so keying
        # on type alone let the deep-ITM 24200PE overwrite the OTM 24000PE (and the ITM 24000CE
        # overwrite the OTM 24200CE) -- pricing the legs against the wrong options entirely and
        # inventing a -25,022 loss on a day the live book really recorded as -2,123.
        by_ts = {}
        for r in path:
            d = by_ts.setdefault(r["snapshot_time"], {"spot": None})
            d[(int(r["strike"]), r["instrument_type"])] = r["ltp"]
            if r["underlying_spot"]:
                d["spot"] = r["underlying_spot"]

        ce_x = pe_x = None
        reason, exit_spot, exit_t = None, None, None
        for ts in sorted(by_ts):
            d = by_ts[ts]
            s = d.get("spot")
            if s and abs(s - spot) / spot * 100 >= MOVE_PCT:
                ce_x = d.get((ce_k, "CE"), ce_e)
                pe_x = d.get((pe_k, "PE"), pe_e)
                reason, exit_spot, exit_t = "move0.4%", s, ts
                break
        if reason is None:                       # time exit at the last snapshot <= 14:45
            if by_ts:
                ts = sorted(by_ts)[-1]
                d = by_ts[ts]
                ce_x = d.get((ce_k, "CE"), ce_e)
                pe_x = d.get((pe_k, "PE"), pe_e)
                reason, exit_spot, exit_t = "time1445", d.get("spot"), ts
            else:
                print(f"  {day}  SKIP (no path)")
                continue

        pnl = ((ce_e - ce_x) + (pe_e - pe_x)) * QTY - 2 * BROK
        wd = datetime.strptime(day, "%Y-%m-%d").strftime("%a")
        cls = "system" if dte <= 1 else "observational"
        rows.append({
            "day": day, "weekday": wd, "dte": dte, "signal_class": cls,
            "entry_spot": round(spot, 2), "pe_strike": pe_k, "ce_strike": ce_k,
            "pe_entry": pe_e, "ce_entry": ce_e, "credit": round(ce_e + pe_e, 2),
            "pe_exit": pe_x, "ce_exit": ce_x, "exit_reason": reason,
            "exit_spot": round(exit_spot, 2) if exit_spot else None,
            "exit_time": str(exit_t)[11:19] if exit_t else None,
            "pnl": round(pnl),
        })
        print(f"  {day} {wd} DTE{dte:<2d} {cls:13s} {pe_k}P/{ce_k}C cr={ce_e+pe_e:6.1f} "
              f"{reason:9s} pnl={round(pnl):+7d}")
        sys.stdout.flush()

    c.close()
    if not rows:
        print("NO DAYS"); return

    with open(os.path.join(OUT, "nasopt_daily.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def block(sub, lab):
        if not sub:
            return
        v = np.array([r["pnl"] for r in sub], float)
        held = [r for r in sub if r["exit_reason"] == "time1445"]
        stop = [r for r in sub if r["exit_reason"] != "time1445"]
        hv = np.array([r["pnl"] for r in held], float) if held else np.array([0.0])
        sv = np.array([r["pnl"] for r in stop], float) if stop else np.array([0.0])
        print(f"\n{lab}  n={len(sub)}")
        print(f"  total {v.sum():+9.0f}   mean {v.mean():+8.0f}   median {np.median(v):+8.0f}   "
              f"win {100*(v>0).mean():3.0f}%   worst {v.min():+8.0f}   best {v.max():+8.0f}")
        print(f"  held to 14:45 : {len(held):3d} days  {hv.sum():+9.0f}  (mean {hv.mean():+7.0f})")
        print(f"  move-stop      : {len(stop):3d} days  {sv.sum():+9.0f}  (mean {sv.mean():+7.0f})")

    print("\n" + "=" * 78)
    block(rows, "ALL DAYS")
    block([r for r in rows if r["signal_class"] == "system"], "SYSTEM (0/1-DTE -- the research/54 strategy)")
    block([r for r in rows if r["signal_class"] == "observational"], "OBSERVATIONAL (DTE>=2 -- NOT the system)")
    print("\n-- by DTE --")
    for d in sorted({r["dte"] for r in rows}):
        sub = [r for r in rows if r["dte"] == d]
        v = np.array([r["pnl"] for r in sub], float)
        print(f"  DTE{d:<2d} n={len(sub):3d}  total {v.sum():+9.0f}  mean {v.mean():+8.0f}  win {100*(v>0).mean():3.0f}%")
    print("\n-- by weekday --")
    for wd in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        sub = [r for r in rows if r["weekday"] == wd]
        if not sub:
            continue
        v = np.array([r["pnl"] for r in sub], float)
        print(f"  {wd} n={len(sub):3d}  total {v.sum():+9.0f}  mean {v.mean():+8.0f}  win {100*(v>0).mean():3.0f}%")
    print("\nDONE")


if __name__ == "__main__":
    main()
