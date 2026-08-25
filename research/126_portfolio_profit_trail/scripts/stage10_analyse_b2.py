#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/126 Stage 10 - ARM B2 ANALYSIS: profit-triggered PORTFOLIO wings.

Nulls, hardest last:
  naked (no defence) | portfolio trail (known to lose) | entry-time wings (known to lose)
  | and the incumbent champion of this whole study: TimeB's CLOCK EXIT.

Reads results/b2_cells.csv. Writes results/b2_summary.txt, results/b2_grid.csv
"""
import csv
import os
import random
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
REP = []


def log(m):
    REP.append(str(m))
    print(m, flush=True)


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def tstat(xs):
    if len(xs) < 3:
        return 0.0
    sd = st.pstdev(xs)
    return st.mean(xs) / (sd / len(xs) ** 0.5) if sd > 0 else 0.0


rows = list(csv.DictReader(open(os.path.join(RES, "b2_cells.csv"))))
log("b2 cells: %d" % len(rows))
cells = defaultdict(dict)
base = {}
peak = {}
for r in rows:
    k = (r["trigger"], int(r["dist_n"]), r["coverage"], r["unwind"])
    cells[k][r["day"]] = r
    base[r["day"]] = float(r["base_net"])
    peak[r["day"]] = float(r["peak_mtm"])
days = sorted(base)
log("days: %d  %s..%s" % (len(days), days[0], days[-1]))
bl = [base[d] for d in days]
log("")
log("NULL 0 - NAKED (the live book as deployed, incl. TIMEB2 replayed): total=%d mean=%d "
    "median=%d win%%=%.1f worst=%d p10=%d"
    % (sum(bl), st.mean(bl), st.median(bl),
       100.0 * sum(1 for x in bl if x > 0) / len(bl), min(bl), pct(bl, 10)))

grid = []
for k, dm in cells.items():
    trig, dist, cov, unw = k
    hedged, armed, wcost, wpaid = [], 0, 0.0, 0
    for d in days:
        r = dm.get(d)
        if not r:
            hedged.append(base[d])
            continue
        hedged.append(float(r["hedged_net"]))
        if r["armed"] == "1" and int(r["n_pairs"]) > 0:
            armed += 1
            wcost += float(r["wing_pnl"])
            if float(r["wing_pnl"]) > 0:
                wpaid += 1
    dl = [h - b for h, b in zip(hedged, bl)]
    grid.append(dict(trigger=trig, dist=dist, coverage=cov, unwind=unw,
                     n=len(hedged), armed=armed,
                     total=round(sum(hedged)), mean=round(st.mean(hedged)),
                     median=round(st.median(hedged)),
                     win=round(100.0 * sum(1 for x in hedged if x > 0) / len(hedged), 1),
                     worst=round(min(hedged)), p10=round(pct(hedged, 10)),
                     d_total=round(sum(dl)), d_mean=round(st.mean(dl)),
                     t=round(tstat(dl), 2),
                     worst_delta=round(min(hedged) - min(bl)),
                     wing_pnl=round(wcost), wing_paid_days=wpaid))

grid.sort(key=lambda r: -r["d_total"])
with open(os.path.join(RES, "b2_grid.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
    w.writeheader()
    for r in grid:
        w.writerow(r)

log("")
log("=" * 118)
log("TOP 20 CELLS by delta vs naked (of %d)" % len(grid))
log("=" * 118)
h = "%-14s %5s %-8s %-8s %6s %9s %8s %9s %9s %7s %10s %6s"
log(h % ("trigger", "dist", "coverage", "unwind", "armed", "total", "mean",
         "d_total", "worst", "t", "worst_del", "paid"))
for r in grid[:20]:
    log(h % (r["trigger"], r["dist"], r["coverage"], r["unwind"], r["armed"],
             r["total"], r["mean"], r["d_total"], r["worst"], r["t"],
             r["worst_delta"], r["wing_paid_days"]))
log("")
log("cells with d_total > 0 : %d of %d" % (sum(1 for r in grid if r["d_total"] > 0), len(grid)))
log("cells improving worst  : %d of %d" % (sum(1 for r in grid if r["worst_delta"] > 0), len(grid)))

# ---- plateau over trigger x distance for the best coverage/unwind
log("")
log("=" * 118)
log("PLATEAU - d_total by TRIGGER x DISTANCE (coverage=ALL, unwind=EOD)")
log("=" * 118)
trigs = sorted({r["trigger"] for r in grid})
dists = sorted({r["dist"] for r in grid})
lk = {(r["trigger"], r["dist"]): r for r in grid if r["coverage"] == "ALL" and r["unwind"] == "EOD"}
log("%-14s" % "trigger" + "".join("%10d" % d for d in dists))
for t in trigs:
    line = "%-14s" % t
    for d in dists:
        r = lk.get((t, d))
        line += "%10s" % (r["d_total"] if r else "-")
    log(line)

# ---- coverage / unwind comparison at the best trigger
log("")
log("=" * 118)
log("COVERAGE and UNWIND (averaged over trigger x distance)")
log("=" * 118)
for key in ("coverage", "unwind"):
    agg = defaultdict(list)
    for r in grid:
        agg[r[key]].append(r["d_total"])
    for kk, v in sorted(agg.items()):
        log("  %-10s %-10s mean d_total=%9d  best=%9d  n_cells=%d"
            % (key, kk, st.mean(v), max(v), len(v)))

# ---- arming frequency by trigger
log("")
log("=" * 118)
log("ARMING FREQUENCY - how often the portfolio ever reaches each trigger")
log("=" * 118)
for t in trigs:
    r = lk.get((t, dists[0]))
    if r:
        log("  %-14s armed on %2d of %d days (%.0f%%)"
            % (t, r["armed"], r["n"], 100.0 * r["armed"] / r["n"]))

open(os.path.join(RES, "b2_summary.txt"), "w").write("\n".join(REP) + "\n")
print("\nwrote results/b2_summary.txt, results/b2_grid.csv")
