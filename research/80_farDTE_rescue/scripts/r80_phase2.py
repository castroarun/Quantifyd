#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/80 G2 — is "these days MOVE" real, or an 11-day accident?

G1 established the far-DTE strangle cannot be rescued by a better stop: intraday theta on a 4-6 DTE
option is only 6-10% of the premium sold, and a hit costs about as much as a calm day pays. Even a
PERFECT costless stop caps out at +578/day.

But G1 also showed the band (+/-0.4% from the 09:20 spot) is hit 73-82% of the time on Wed/Thu --
i.e. those days MOVE. On n=11 that is worth nothing. This tests it on the full NIFTY 5-min history
(2015-02 -> 2026-03, ~2,700 days), where it is properly powered, and asks:

  1. P(|move from 09:20| >= 0.4% before 14:45) -- by weekday, and PER YEAR (does it hold up?)
  2. how FAR does it go when it goes (a breakout needs range, not just a trigger)
  3. do CPR width / gap / opening range / VIX predict a CALM day (sell) or a TRENDING day (buy)?

Everything is measured from the 09:20 spot -- the exact moment NAS-OPT commits -- so nothing here
uses information the live system would not have.

Reads only.
"""
import sqlite3
from collections import defaultdict

import numpy as np

ROOT = "/home/arun/quantifyd"
BAND = 0.004          # the 0.4% move-stop
ENTRY_MIN = 9 * 60 + 20
LAST_MIN = 14 * 60 + 45

c = sqlite3.connect(f"file:{ROOT}/backtest_data/market_data.db?mode=ro", uri=True)
rows = c.execute(
    "SELECT date, open, high, low, close FROM market_data_unified "
    "WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date").fetchall()

# daily VIX
vix = {r[0][:10]: r[1] for r in c.execute(
    "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
c.close()

days = defaultdict(list)
for d, o, h, l, cl in rows:
    days[str(d)[:10]].append((str(d)[11:16], o, h, l, cl))

def mins(hhmm):
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])

recs = []
prev = None
for day in sorted(days):
    bars = sorted(days[day])
    if len(bars) < 40:
        continue
    day_h = max(b[2] for b in bars)
    day_l = min(b[3] for b in bars)
    day_c = bars[-1][4]
    day_o = bars[0][1]

    entry = next((b for b in bars if mins(b[0]) >= ENTRY_MIN), None)
    if entry is None:
        prev = (day_h, day_l, day_c)
        continue
    s0 = entry[4]
    window = [b for b in bars if ENTRY_MIN <= mins(b[0]) <= LAST_MIN]
    if len(window) < 20 or not s0:
        prev = (day_h, day_l, day_c)
        continue

    hi = max(b[2] for b in window)
    lo = min(b[3] for b in window)
    up = (hi - s0) / s0
    dn = (s0 - lo) / s0
    exc = max(up, dn)                      # max excursion from the 09:20 spot
    hit = exc >= BAND
    end = window[-1][4]
    net = abs(end - s0) / s0               # where it actually ENDED (trend, not just a spike)

    # --- ex-ante features, all known at 09:20 ---
    op = bars[0][1]
    gap = (op - prev[2]) / prev[2] if prev else np.nan          # gap vs prev close
    orb = [b for b in bars if mins(b[0]) < ENTRY_MIN]           # 09:15-09:20 opening range
    or_rng = (max(b[2] for b in orb) - min(b[3] for b in orb)) / s0 if orb else np.nan
    if prev:
        ph, pl, pc = prev
        piv = (ph + pl + pc) / 3.0
        bc = (ph + pl) / 2.0
        tc = 2 * piv - bc
        cpr_w = abs(tc - bc) / piv                              # CPR width (narrow => trend day)
        above_cpr = s0 > max(tc, bc)
        below_cpr = s0 < min(tc, bc)
    else:
        cpr_w, above_cpr, below_cpr = np.nan, None, None

    recs.append(dict(day=day, wd=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][
        __import__("datetime").date(*map(int, day.split("-"))).weekday()],
        year=day[:4], s0=s0, exc=exc, hit=hit, net=net, gap=gap, or_rng=or_rng,
        cpr_w=cpr_w, above_cpr=above_cpr, below_cpr=below_cpr, vix=vix.get(day, np.nan)))
    prev = (day_h, day_l, day_c)

n = len(recs)
print(f"=== {n} days, {recs[0]['day']} -> {recs[-1]['day']} | measured from the 09:20 spot ===\n")

E = np.array([r["exc"] for r in recs])
H = np.array([r["hit"] for r in recs], float)
print(f"P(|move| >= 0.4% between 09:20 and 14:45) = {100*H.mean():.1f}%   "
      f"(median excursion {100*np.median(E):.2f}%)\n")

print("--- by weekday (is Wed/Thu really more volatile?) ---")
print(f"{'wd':>4s} {'n':>5s} {'P(hit 0.4%)':>12s} {'median exc':>11s} {'mean exc':>9s} {'median NET move':>16s}")
for wd in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
    s = [r for r in recs if r["wd"] == wd]
    if not s:
        continue
    h = np.mean([r["hit"] for r in s])
    e = np.array([r["exc"] for r in s])
    nt = np.array([r["net"] for r in s])
    print(f"{wd:>4s} {len(s):>5d} {100*h:>11.1f}% {100*np.median(e):>10.2f}% "
          f"{100*e.mean():>8.2f}% {100*np.median(nt):>15.2f}%")

print("\n--- PER YEAR (does the weekday effect hold up, or is it a recent accident?) ---")
print(f"{'year':>5s} {'n':>4s} " + " ".join(f"{wd:>6s}" for wd in ["Mon","Tue","Wed","Thu","Fri"]) + "   <- P(hit)")
for y in sorted({r["year"] for r in recs}):
    s = [r for r in recs if r["year"] == y]
    line = f"{y:>5s} {len(s):>4d} "
    for wd in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        w = [r["hit"] for r in s if r["wd"] == wd]
        line += f"{100*np.mean(w):>5.0f}% " if w else "    -  "
    print(line)

print("\n\n--- can ANYTHING predict a CALM day (the one that would rescue selling)? ---")
def buckets(key, edges, lab):
    print(f"\n  by {lab}:")
    print(f"    {'bucket':<16s} {'n':>5s} {'P(hit 0.4%)':>12s} {'median exc':>11s}")
    v = np.array([r[key] for r in recs], float)
    ok = ~np.isnan(v)
    for i in range(len(edges) + 1):
        lo = -np.inf if i == 0 else edges[i - 1]
        hi = np.inf if i == len(edges) else edges[i]
        m = ok & (v >= lo) & (v < hi)
        if m.sum() < 30:
            continue
        h = H[m].mean()
        e = np.median(E[m])
        name = (f"< {edges[0]:g}" if i == 0 else
                f">= {edges[-1]:g}" if i == len(edges) else f"{lo:g} - {hi:g}")
        print(f"    {name:<16s} {m.sum():>5d} {100*h:>11.1f}% {100*e:>10.2f}%")

buckets("cpr_w", [0.001, 0.002, 0.004], "CPR width (narrow = trend day)")
buckets("or_rng", [0.001, 0.002, 0.004], "opening range 09:15-09:20 / spot")
buckets("gap", [-0.004, -0.001, 0.001, 0.004], "overnight gap")
buckets("vix", [12, 14, 17, 22], "India VIX (previous close)")

print("\n\n--- correlation of each feature with the day's excursion ---")
for k in ["cpr_w", "or_rng", "vix"]:
    v = np.array([r[k] for r in recs], float)
    m = ~np.isnan(v)
    print(f"  corr(excursion, {k:8s}) = {np.corrcoef(v[m], E[m])[0,1]:+.3f}")
g = np.abs(np.array([r["gap"] for r in recs], float))
m = ~np.isnan(g)
print(f"  corr(excursion, |gap|   ) = {np.corrcoef(g[m], E[m])[0,1]:+.3f}")
print("\nDONE")
