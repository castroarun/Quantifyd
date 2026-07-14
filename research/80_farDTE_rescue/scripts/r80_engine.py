#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 — synthetic NIFTY option engine, CALIBRATED and VALIDATED against the real chain.

Everything downstream (entry timing x holding period x structure x filters, and the directional-
short idea) needs option prices on days the chain recorder never saw. So: Black-Scholes on NIFTY
5-min spot with India VIX as the IV base.

This is only worth anything if it can REPRICE REALITY. So it is calibrated and then validated
against the 58 recorded chain days, leg by leg:

  1. CALIBRATE: India VIX is a 30-day ATM IV. Weekly options do not trade at that IV -- short-dated
     IV is systematically different, and OTM strikes carry skew. Fit a multiplier k(DTE) so that
     BS(iv = VIX * k) reproduces the OBSERVED premium of the actual NAS-OPT legs.
  2. VALIDATE: report the error AFTER calibration. If the engine cannot reprice the real chain to
     within a tolerable error, the whole program is void and I will say so.

Expiry calendar: NIFTY weekly expiry was THURSDAY historically and moved to TUESDAY in 2025. The
switch date is inferred from the recorded chain rather than assumed.

Reads only. Writes the calibration to results/engine_calib.json.
"""
import json
import math
import os
import sqlite3
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

ROOT = "/home/arun/quantifyd"
ODB = f"{ROOT}/backtest_data/options_data.db"
MDB = f"{ROOT}/backtest_data/market_data.db"
OUT = f"{ROOT}/research/80_farDTE_rescue/results"


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs(S, K, T, iv, kind, r=0.0):
    """Black-Scholes. T in YEARS. iv as a decimal (0.14 = 14%)."""
    if T <= 0 or iv <= 0 or S <= 0:
        return max(0.0, (S - K) if kind == "CE" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if kind == "CE":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


# ---------------------------------------------------------------- observed legs
oc = sqlite3.connect(f"file:{ODB}?mode=ro", uri=True)
oc.row_factory = sqlite3.Row
md = sqlite3.connect(f"file:{MDB}?mode=ro", uri=True)
vix_d = {r[0][:10]: r[1] for r in md.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
md.close()

# expiry weekday actually used, straight from the recorder
exp_wd = defaultdict(set)
for r in oc.execute("SELECT DISTINCT date(snapshot_time) d, expiry_date e FROM option_chain "
                    "WHERE symbol='NIFTY'"):
    y, m, dd = map(int, r[1].split("-"))
    exp_wd[r[0][:7]].add(date(y, m, dd).weekday())
print("expiry weekdays seen in the recorded chain (0=Mon):",
      {k: sorted(v) for k, v in sorted(exp_wd.items())})

# sample the real chain: every day, ~09:20, the OTM legs NAS-OPT would sell, ACROSS DTEs
obs = []
days = [r[0] for r in oc.execute(
    "SELECT DISTINCT date(snapshot_time) FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")]
for day in days:
    snap = oc.execute("SELECT MIN(snapshot_time) FROM option_chain WHERE symbol='NIFTY' "
                      "AND date(snapshot_time)=? AND time(snapshot_time)>='09:20:00'",
                      (day,)).fetchone()[0]
    if not snap:
        continue
    rows = oc.execute(
        "SELECT strike, instrument_type, ltp, expiry_date, underlying_spot FROM option_chain "
        "WHERE symbol='NIFTY' AND snapshot_time=? AND ltp>0", (snap,)).fetchall()
    if not rows:
        continue
    spot = next((r["underlying_spot"] for r in rows if r["underlying_spot"]), None)
    v = vix_d.get(day)
    if not spot or not v:
        continue
    exps = sorted({r["expiry_date"] for r in rows if r["expiry_date"] >= day})
    if not exps:
        continue
    exp = exps[0]
    y, m, dd = map(int, day.split("-"))
    ey, em, ed = map(int, exp.split("-"))
    dte = (date(ey, em, ed) - date(y, m, dd)).days
    atm = round(spot / 50) * 50
    for off, kind in ((+100, "CE"), (-100, "PE")):
        k = atm + off
        px = next((r["ltp"] for r in rows if int(r["strike"]) == k
                   and r["instrument_type"] == kind and r["expiry_date"] == exp), None)
        if px:
            obs.append(dict(day=day, dte=dte, spot=spot, K=k, kind=kind, px=float(px), vix=float(v)))
oc.close()

print(f"\nobserved legs sampled from the real chain: {len(obs)}")

# ---------------------------------------------------------------- calibrate k(DTE)
# T uses CALENDAR days/365. Intraday: a 0-DTE option still has hours of life -- give it the
# remaining session (09:20->15:30 = 6.17h) so T is never 0.
def T_of(dte, hours_left=6.17):
    return max((dte + hours_left / 24.0) / 365.0, 1e-6)

calib = {}
print(f"\n{'DTE':>4s} {'n':>4s} {'best k (IV mult)':>17s} {'median abs err':>15s} {'median err':>12s}")
for d in sorted({o["dte"] for o in obs}):
    sub = [o for o in obs if o["dte"] == d]
    if len(sub) < 6:
        continue
    # Zero the MEDIAN SIGNED error, not the absolute one. Minimising |error| leaves a systematic
    # bias in place -- and since every strategy here SELLS this premium, a pricer that is 7% high
    # hands us free money that does not exist. An unbiased pricer cannot manufacture P&L.
    best, bk = None, None
    for k in np.arange(0.60, 2.51, 0.005):
        errs = []
        for o in sub:
            th = bs(o["spot"], o["K"], T_of(d), (o["vix"] / 100.0) * k, o["kind"])
            if o["px"] > 0.5:
                errs.append((th - o["px"]) / o["px"])
        if not errs:
            continue
        e = abs(float(np.median(errs)))          # how far the MEDIAN SIGNED error is from zero
        if best is None or e < best:
            best, bk = e, float(k)
    signed = np.median([
        (bs(o["spot"], o["K"], T_of(d), (o["vix"] / 100.0) * bk, o["kind"]) - o["px"]) / o["px"]
        for o in sub if o["px"] > 0.5])
    calib[str(d)] = bk
    print(f"{d:>4d} {len(sub):>4d} {bk:>17.2f} {100*best:>14.0f}% {100*signed:>11.0f}%")

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/engine_calib.json", "w") as f:
    json.dump({"iv_mult_by_dte": calib}, f, indent=2)

# ---------------------------------------------------------------- validate the STRANGLE credit
print("\n\n=== VALIDATION: can the engine reprice the real 100pt-OTM strangle credit? ===\n")
print(f"{'DTE':>4s} {'n':>4s} {'real credit':>12s} {'synthetic':>11s} {'median err':>12s}")
bad = 0
for d in sorted({o["dte"] for o in obs}):
    sub = [o for o in obs if o["dte"] == d]
    if str(d) not in calib:
        continue
    k = calib[str(d)]
    byday = defaultdict(dict)
    for o in sub:
        byday[o["day"]][o["kind"]] = o
    real, syn = [], []
    for day, legs in byday.items():
        if "CE" not in legs or "PE" not in legs:
            continue
        r_ = legs["CE"]["px"] + legs["PE"]["px"]
        s_ = sum(bs(legs[t]["spot"], legs[t]["K"], T_of(d), (legs[t]["vix"] / 100.0) * k, t)
                 for t in ("CE", "PE"))
        real.append(r_); syn.append(s_)
    if not real:
        continue
    real, syn = np.array(real), np.array(syn)
    err = np.median(np.abs(syn - real) / real)
    if err > 0.25:
        bad += 1
    print(f"{d:>4d} {len(real):>4d} {real.mean():>12.1f} {syn.mean():>11.1f} {100*err:>11.0f}%")

print(f"\nDTE buckets with >25% median error: {bad}")
print("VERDICT:", "ENGINE USABLE" if bad == 0 else "ENGINE NOT TRUSTWORTHY -- do not build on it")
