#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G1 — WHY does the far-DTE strangle lose? Decompose before optimising.

A short strangle held 09:20 -> 14:45 has exactly two P&L drivers:
    + THETA   : the premium you keep if the index does not move
    - DELTA/GAMMA : what you give back when it does

research/79 showed far-DTE (4/5/6) loses -441/day while 0/1-DTE makes +1,578. Before sweeping CPR
width / gaps / opening ranges over 33 days (which WILL produce a fake pattern), establish WHICH of
the two drivers breaks. That decides whether conditioning can rescue these days at all:

  * if far-DTE simply MOVES more  -> predicting calm days could rescue it. Worth the sweep.
  * if far-DTE has NO THETA to harvest intraday -> no filter can rescue it. The payoff structure
    itself is wrong, and the honest answer is "do not sell far-DTE premium intraday".

Measures, per DTE, on the real chain:
  - band-hit rate (how often the +/-0.4% move-stop fires)
  - pure decay on days the band never fires  (= the theta actually available in 5 hours)
  - premium given back on days it does fire  (= the delta/gamma cost)
  - decay as a FRACTION of the credit sold   (the honest "how much of what I sold do I keep?")
Reads only.
"""
import csv
import os
import sqlite3
from datetime import datetime

import numpy as np

ROOT = "/home/arun/quantifyd"
CSV = f"{ROOT}/research/79_nasopt_full_replay/results/nasopt_daily.csv"
QTY, BROK = 130, 80

rows = list(csv.DictReader(open(CSV)))
for r in rows:
    for k in ("pnl", "credit", "ce_entry", "pe_entry", "ce_exit", "pe_exit", "entry_spot"):
        r[k] = float(r[k]) if r[k] not in (None, "") else np.nan
    r["dte"] = int(r["dte"])
    r["hit"] = r["exit_reason"] != "time1445"
    r["exit_prem"] = r["ce_exit"] + r["pe_exit"]
    r["decay_pts"] = r["credit"] - r["exit_prem"]          # premium points kept (before costs)
    r["decay_frac"] = r["decay_pts"] / r["credit"] if r["credit"] else np.nan

print("=== per-DTE decomposition (58 real chain days) ===\n")
print(f"{'DTE':>3s} {'n':>3s} {'credit':>8s} {'band-hit':>9s} | "
      f"{'HELD: pts kept':>15s} {'as % of credit':>15s} | {'HIT: pts given':>15s} | {'mean P&L':>9s}")
for d in sorted({r["dte"] for r in rows}):
    sub = [r for r in rows if r["dte"] == d]
    held = [r for r in sub if not r["hit"]]
    hit = [r for r in sub if r["hit"]]
    cred = np.mean([r["credit"] for r in sub])
    hr = 100 * len(hit) / len(sub)
    hk = np.mean([r["decay_pts"] for r in held]) if held else np.nan
    hf = 100 * np.mean([r["decay_frac"] for r in held]) if held else np.nan
    hg = np.mean([r["decay_pts"] for r in hit]) if hit else np.nan
    pnl = np.mean([r["pnl"] for r in sub])
    print(f"{d:>3d} {len(sub):>3d} {cred:>8.1f} {hr:>8.0f}% | {hk:>15.1f} {hf:>14.0f}% | "
          f"{hg:>15.1f} | {pnl:>+9.0f}")

print("\n\n=== the two drivers, as rupees per day (x130 qty, net of Rs160 brokerage) ===\n")
print(f"{'slice':<24s} {'theta captured':>16s} {'delta/gamma cost':>18s} {'net':>10s}")
for lab, sel in [("0/1-DTE (the system)", lambda r: r["dte"] <= 1),
                 ("DTE>=4 (far)", lambda r: r["dte"] >= 4)]:
    sub = [r for r in rows if sel(r)]
    held = [r for r in sub if not r["hit"]]
    hit = [r for r in sub if r["hit"]]
    # theta = what a NO-MOVE day yields; approximate with the held days' decay
    theta = np.mean([r["decay_pts"] for r in held]) * QTY - 2 * BROK if held else 0
    cost = np.mean([r["decay_pts"] for r in hit]) * QTY - 2 * BROK if hit else 0
    p_hit = len(hit) / len(sub)
    net = (1 - p_hit) * theta + p_hit * cost
    print(f"{lab:<24s} {theta:>+16.0f} {cost:>+18.0f} {net:>+10.0f}")

print("\n\n=== the honest ratio: how much of the premium you SELL do you actually KEEP? ===\n")
print(f"{'DTE':>3s} {'credit sold':>12s} {'kept if calm':>14s} {'keep rate':>11s}   {'<- theta available in 5h'}")
for d in sorted({r["dte"] for r in rows}):
    sub = [r for r in rows if r["dte"] == d]
    held = [r for r in sub if not r["hit"]]
    if not held:
        continue
    cred = np.mean([r["credit"] for r in held])
    kept = np.mean([r["decay_pts"] for r in held])
    print(f"{d:>3d} {cred:>12.1f} {kept:>14.1f} {100*kept/cred:>10.0f}%")
