#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G9 — THE DECAY CLOCK: when, inside the day, does premium actually melt?

Arun's idea: if decay is not spread evenly through the session -- if there are windows where premium
drops fast (he suspects Friday into the close) -- then on the far-DTE days we could sell ONLY inside
that window and be flat the rest of the time. Tiny exposure, no overnight risk.

This CANNOT be answered with the pricing model: Black-Scholes with a constant IV decays smoothly by
construction, so it can never show a "spike". A spike would be a real market effect -- traders
marking IV down into a weekend or a close -- and only REAL QUOTES can reveal it. So this uses the
58 recorded chain days, not the engine.

METHOD (isolating decay from direction, which is the whole difficulty):
  * fix the ~100pt-OTM strangle at 09:20 each day (the legs NAS-OPT would sell)
  * walk the real chain and take the combined premium every 15 minutes
  * for each 15-min window, record d(premium) AND d(spot)
  * to isolate DECAY, keep only windows where the index barely moved (|d spot| < 0.05%, ~12 pts) --
    what is left is time + IV, not delta
  * report the median premium drop per window, by weekday / DTE
  * and cumulatively: what share of the day's total decay lands in each window?

If Arun is right, Friday's last hour will show a disproportionate drop.
Reads only.
"""
import sqlite3
from collections import defaultdict
from datetime import datetime

import numpy as np

ROOT = "/home/arun/quantifyd"
ODB = f"{ROOT}/backtest_data/options_data.db"
QTY = 130
CALM = 0.0005          # |d spot| < 0.05% => the window is "flat", so d(premium) is decay, not delta
STEP = 15              # minutes per window


def mn(ts):
    return int(ts[11:13]) * 60 + int(ts[14:16])


c = sqlite3.connect(f"file:{ODB}?mode=ro", uri=True)
c.row_factory = sqlite3.Row
days = [r[0] for r in c.execute(
    "SELECT DISTINCT date(snapshot_time) FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")]

rows = []           # (day, weekday, dte, bucket_start_min, d_prem, d_spot, prem_level)
for day in days:
    snap0 = c.execute("SELECT MIN(snapshot_time) FROM option_chain WHERE symbol='NIFTY' "
                      "AND date(snapshot_time)=? AND time(snapshot_time)>='09:20:00'", (day,)).fetchone()[0]
    if not snap0:
        continue
    base = c.execute(
        "SELECT strike, instrument_type, ltp, expiry_date, underlying_spot FROM option_chain "
        "WHERE symbol='NIFTY' AND snapshot_time=? AND ltp>0", (snap0,)).fetchall()
    if not base:
        continue
    spot0 = next((r["underlying_spot"] for r in base if r["underlying_spot"]), None)
    if not spot0:
        continue
    exps = sorted({r["expiry_date"] for r in base if r["expiry_date"] >= day})
    if not exps:
        continue
    exp = exps[0]
    dte = (datetime.fromisoformat(exp).date() - datetime.fromisoformat(day).date()).days
    atm = round(spot0 / 50) * 50
    Kc, Kp = atm + 100, atm - 100

    path = c.execute(
        "SELECT snapshot_time, strike, instrument_type, ltp, underlying_spot FROM option_chain "
        "WHERE symbol='NIFTY' AND date(snapshot_time)=? AND expiry_date=? AND strike IN (?,?) "
        "AND ltp>0 ORDER BY snapshot_time", (day, exp, Kc, Kp)).fetchall()
    by_ts = {}
    for r in path:
        d = by_ts.setdefault(r["snapshot_time"], {})
        d[(int(r["strike"]), r["instrument_type"])] = r["ltp"]
        if r["underlying_spot"]:
            d["spot"] = r["underlying_spot"]

    # one sample per 15-min bucket: the LAST snapshot inside it
    buck = {}
    for ts in sorted(by_ts):
        d = by_ts[ts]
        if (Kc, "CE") not in d or (Kp, "PE") not in d or "spot" not in d:
            continue
        b = (mn(ts) // STEP) * STEP
        buck[b] = (d[(Kc, "CE")] + d[(Kp, "PE")], d["spot"])

    ks = sorted(buck)
    wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][datetime.fromisoformat(day).weekday()]
    for i in range(1, len(ks)):
        b0, b1 = ks[i - 1], ks[i]
        if b1 - b0 != STEP:
            continue
        p0, s0 = buck[b0]
        p1, s1 = buck[b1]
        rows.append(dict(day=day, wd=wd, dte=dte, b=b1, dp=p1 - p0,
                         ds=(s1 - s0) / s0, lvl=p0))
c.close()

calm = [r for r in rows if abs(r["ds"]) < CALM]
print(f"=== THE DECAY CLOCK — real chain, {len(days)} days ===")
print(f"    {len(rows)} 15-min windows, of which {len(calm)} were FLAT (|move| < 0.05%) and so show "
      f"pure decay\n")


def clock(sel, lab, minn=6):
    sub = [r for r in calm if sel(r)]
    if len(sub) < 20:
        return
    print(f"\n--- {lab}  (n={len(sub)} flat windows) ---")
    print(f"    {'window':>13s} {'n':>4s} {'median decay':>13s} {'Rs per 130 qty':>15s} {'share of day':>13s}")
    byb = defaultdict(list)
    for r in sub:
        byb[r["b"]].append(r["dp"])
    tot = sum(-np.median(v) for b, v in byb.items() if len(v) >= minn and np.median(v) < 0)
    for b in sorted(byb):
        v = byb[b]
        if len(v) < minn:
            continue
        med = float(np.median(v))
        share = (-med / tot * 100) if (tot > 0 and med < 0) else 0.0
        bar = "#" * int(max(0, -med) * 6)
        print(f"    {b//60:02d}:{b%60:02d}-{(b+STEP)//60:02d}:{(b+STEP)%60:02d} {len(v):>4d} "
              f"{med:>+13.2f} {med*QTY:>+15.0f} {share:>12.0f}%  {bar}")


clock(lambda r: r["dte"] >= 4, "FAR-DTE (4/5/6) — the days we want to sneak into")
clock(lambda r: r["dte"] <= 1, "0/1-DTE (for contrast — the days NAS-OPT already trades)")

print("\n\n=== ARUN'S HYPOTHESIS: does FRIDAY melt into the close? ===")
for wd in ["Wed", "Thu", "Fri"]:
    sub = [r for r in calm if r["wd"] == wd]
    if len(sub) < 20:
        continue
    late = [r["dp"] for r in sub if r["b"] >= 14 * 60]        # 14:00 onwards
    early = [r["dp"] for r in sub if 10 * 60 <= r["b"] < 13 * 60]
    if len(late) < 5 or len(early) < 5:
        continue
    ml, me = float(np.median(late)), float(np.median(early))
    print(f"  {wd}: median decay 10:00-13:00 = {me:+.2f}/window   |   14:00-close = {ml:+.2f}/window   "
          f"-> last hour is {abs(ml/me) if me else 0:.1f}x " +
          ("FASTER" if ml < me else "SLOWER"))

print("\n\n=== can we SNEAK IN? sell the strangle only inside the fastest window ===")
print("    (median decay captured, per 130 qty, before costs -- and what a 0.4% adverse move would cost)")
for wd in ["Wed", "Thu", "Fri"]:
    sub = [r for r in calm if r["wd"] == wd and r["b"] >= 14 * 60]
    if len(sub) < 8:
        continue
    med = float(np.median([r["dp"] for r in sub]))
    n_win = len({r["b"] for r in sub})
    gain = -med * QTY * n_win
    lvl = float(np.median([r["lvl"] for r in sub]))
    print(f"  {wd}: {n_win} windows after 14:00, median decay {med:+.2f}/window "
          f"-> ~Rs{gain:+,.0f} if the index stands still (premium level ~{lvl:.0f})")
print("\nDONE")
