#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G2a — can ANY exit rule rescue the far-DTE strangle?

G1 showed a smarter exit AT THE SAME 0.4% BAND is capped at +578/day (perfect costless stop). But a
DIFFERENT band, or a premium-based stop, changes the hit rate AND the payoff mix -- so it is not
bounded by that ceiling and must be tested properly.

Sweeps, on the real chain premium path of every day (09:20 -> 14:45):
  * move-stop on the UNDERLYING at 0.2 / 0.3 / 0.4 / 0.5 / 0.6 / 0.8 / 1.0%
  * NO STOP AT ALL (hold to 14:45) -- research/78's lesson: this is the baseline everyone forgets
  * combined-premium SL, relative:  exit if (CE+PE) >= credit x k,  k = 1.2 / 1.3 / 1.5 / 1.75 / 2.0
  * combined-premium SL, absolute:  exit if (CE+PE) - credit >= X pts,  X = 20 / 30 / 50 / 75
  * per-leg SL:                     exit if EITHER leg >= its entry x k, k = 1.5 / 2.0 / 3.0
  * profit target:                  exit if (CE+PE) <= credit x (1-p),  p = 0.3 / 0.5

Reported for FAR-DTE (>=4) and, as a control, for the 0/1-DTE system -- a rule that "fixes" far-DTE
by breaking the good days is not a fix.

Every rule is also reported PER WEEKDAY, because with n=33 far-DTE days a single average is exactly
how you fool yourself.

Reads only.
"""
import csv
import sqlite3
from collections import defaultdict

import numpy as np

ROOT = "/home/arun/quantifyd"
ODB = f"{ROOT}/backtest_data/options_data.db"
DAILY = f"{ROOT}/research/79_nasopt_full_replay/results/nasopt_daily.csv"
QTY, BROK = 130, 80
LAST_T = "14:45:00"


def tsec(s):
    return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def build_rules():
    R = []
    for b in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        R.append((f"move{b:g}%", ("move", b)))
    R.append(("NO STOP", ("none", None)))
    for k in (1.2, 1.3, 1.5, 1.75, 2.0):
        R.append((f"prem x{k:g}", ("premrel", k)))
    for x in (20, 30, 50, 75):
        R.append((f"prem +{x}pt", ("premabs", x)))
    for k in (1.5, 2.0, 3.0):
        R.append((f"leg x{k:g}", ("leg", k)))
    for p in (0.3, 0.5):
        R.append((f"target {int(p*100)}%", ("target", p)))
    return R


RULES = build_rules()


def simulate(path, s0, ce_e, pe_e):
    """path: list of (t, spot, ce, pe). Returns {rule: exit premium (ce+pe)}."""
    credit = ce_e + pe_e
    out = {}
    for name, (kind, p) in RULES:
        px = None
        for t, spot, ce, pe in path:
            tot = ce + pe
            if kind == "move" and spot and abs(spot - s0) / s0 * 100 >= p:
                px = tot; break
            if kind == "premrel" and tot >= credit * p:
                px = tot; break
            if kind == "premabs" and (tot - credit) >= p:
                px = tot; break
            if kind == "leg" and (ce >= ce_e * p or pe >= pe_e * p):
                px = tot; break
            if kind == "target" and tot <= credit * (1 - p):
                px = tot; break
        if px is None:
            px = path[-1][2] + path[-1][3]      # time exit at 14:45
        out[name] = px
    return out


base = {r["day"]: r for r in csv.DictReader(open(DAILY))}
c = sqlite3.connect(f"file:{ODB}?mode=ro", uri=True)
c.row_factory = sqlite3.Row

res = defaultdict(list)   # rule -> list of dicts(day, dte, wd, pnl)
for day, b in sorted(base.items()):
    ce_k, pe_k = int(b["ce_strike"]), int(b["pe_strike"])
    ce_e, pe_e = float(b["ce_entry"]), float(b["pe_entry"])
    s0 = float(b["entry_spot"])
    dte, wd = int(b["dte"]), b["weekday"]

    snap0 = c.execute(
        "SELECT MIN(snapshot_time) FROM option_chain WHERE symbol='NIFTY' "
        "AND date(snapshot_time)=? AND time(snapshot_time)>='09:20:00'", (day,)).fetchone()[0]
    # An option is identified by (expiry, strike, type). Filtering on strike alone pulls the
    # monthlies and far months too, and the last row silently wins -- pricing an intraday strangle
    # against a far-month option. expiry = day + dte.
    from datetime import date as _d, timedelta as _td
    _y, _m, _dd = map(int, day.split("-"))
    expiry = (_d(_y, _m, _dd) + _td(days=dte)).isoformat()
    rows = c.execute(
        "SELECT snapshot_time, strike, instrument_type, ltp, underlying_spot "
        "FROM option_chain WHERE symbol='NIFTY' AND date(snapshot_time)=? AND snapshot_time>=? "
        "AND time(snapshot_time)<=? AND expiry_date=? AND strike IN (?,?) AND ltp>0 "
        "ORDER BY snapshot_time",
        (day, snap0, LAST_T, expiry, ce_k, pe_k)).fetchall()

    by_ts = {}
    for r in rows:
        d = by_ts.setdefault(r["snapshot_time"], {"spot": None})
        d[(int(r["strike"]), r["instrument_type"])] = r["ltp"]
        if r["underlying_spot"]:
            d["spot"] = r["underlying_spot"]

    path = []
    lce, lpe = ce_e, pe_e
    for ts in sorted(by_ts):
        d = by_ts[ts]
        lce = d.get((ce_k, "CE"), lce)
        lpe = d.get((pe_k, "PE"), lpe)
        path.append((tsec(ts), d.get("spot"), lce, lpe))
    if len(path) < 20:
        continue

    sim = simulate(path, s0, ce_e, pe_e)
    for name, exit_prem in sim.items():
        pnl = ((ce_e + pe_e) - exit_prem) * QTY - 2 * BROK
        res[name].append(dict(day=day, dte=dte, wd=wd, pnl=pnl))
c.close()

def show(sel, title):
    print(f"\n\n=== {title} ===")
    n = len(res["NO STOP"] and [r for r in res["NO STOP"] if sel(r)])
    print(f"n = {n} days\n")
    print(f"{'rule':<12s} {'mean':>8s} {'median':>8s} {'total':>9s} {'win%':>6s} {'worst':>9s} "
          f"{'Mon':>7s} {'Tue':>7s} {'Wed':>7s} {'Thu':>7s} {'Fri':>7s}")
    rank = []
    for name, _ in RULES:
        sub = [r for r in res[name] if sel(r)]
        if not sub:
            continue
        v = np.array([r["pnl"] for r in sub], float)
        line = (f"{name:<12s} {v.mean():>+8.0f} {np.median(v):>+8.0f} {v.sum():>+9.0f} "
                f"{100*(v>0).mean():>5.0f}% {v.min():>+9.0f} ")
        for wd in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
            w = [r["pnl"] for r in sub if r["wd"] == wd]
            line += f"{np.mean(w):>+7.0f} " if w else "      - "
        print(line)
        rank.append((v.mean(), name))
    rank.sort(reverse=True)
    print(f"\n  best: {rank[0][1]} ({rank[0][0]:+.0f}/day)   |   "
          f"current live rule: move0.4% ({dict((n_, m) for m, n_ in rank).get('move0.4%', float('nan')):+.0f}/day)")

show(lambda r: r["dte"] >= 4, "FAR-DTE (4/5/6) -- the days we are trying to rescue")
show(lambda r: r["dte"] <= 1, "0/1-DTE (the system) -- CONTROL: a rule must not break these")
print("\nDONE")
