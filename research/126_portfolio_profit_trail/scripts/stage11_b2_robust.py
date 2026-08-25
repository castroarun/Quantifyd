#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/126 Stage 11 - ARM B2 robustness: is the gain a real distribution effect or
one lucky day? Plus the OOS split, the super-winner guard, and the clock-exit null.
"""
import csv
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
REP = []


def log(m):
    REP.append(str(m))
    print(m, flush=True)


def pct(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def tstat(xs):
    sd = st.pstdev(xs)
    return st.mean(xs) / (sd / len(xs) ** 0.5) if sd > 0 else 0.0


rows = list(csv.DictReader(open(os.path.join(RES, "b2_cells.csv"))))
cells = defaultdict(dict)
base, peak = {}, {}
for r in rows:
    cells[(r["trigger"], int(r["dist_n"]), r["coverage"], r["unwind"])][r["day"]] = r
    base[r["day"]] = float(r["base_net"])
    peak[r["day"]] = float(r["peak_mtm"])
days = sorted(base)
bl = [base[d] for d in days]

CANDS = [("ABS_15000", 100, "ALL", "EOD"), ("ABS_12000", 100, "ALL", "EOD"),
         ("ABS_20000", 100, "ALL", "EOD"), ("ABS_15000", 250, "ALL", "EOD"),
         ("ABS_12000", 200, "ALL", "EOD"), ("ABS_20000", 300, "ALL", "EOD")]

log("=" * 112)
log("1. IS IT ONE DAY? - per-day contribution of the wing overlay")
log("=" * 112)
for k in CANDS:
    dm = cells[k]
    dl = []
    for d in days:
        r = dm.get(d)
        dl.append((d, float(r["hedged_net"]) - base[d] if r else 0.0))
    nz = [(d, x) for d, x in dl if abs(x) > 1]
    nz.sort(key=lambda x: -x[1])
    tot = sum(x for _, x in dl)
    top1 = nz[0][1] if nz else 0
    top3 = sum(x for _, x in nz[:3])
    log("")
    log("%-30s total=%+8.0f  days_moved=%d  top1=%+.0f (%.0f%%)  top3=%+.0f (%.0f%%)"
        % (str(k), tot, len(nz), top1, 100.0 * top1 / tot if tot else 0,
           top3, 100.0 * top3 / tot if tot else 0))
    log("   best days : %s" % ", ".join("%s %+.0f" % (d, x) for d, x in nz[:4]))
    log("   worst days: %s" % ", ".join("%s %+.0f" % (d, x) for d, x in nz[-4:]))
    ex = [x for _, x in dl]
    log("   ex-top1 total=%+.0f   t(all)=%.2f   t(ex-top1)=%.2f"
        % (tot - top1, tstat(ex),
           tstat([x for d, x in dl if not nz or d != nz[0][0]])))

log("")
log("=" * 112)
log("2. OUT-OF-SAMPLE SPLIT")
log("=" * 112)
mid = len(days) // 2
d_is, d_oos = set(days[:mid]), set(days[mid:])
log("IS %s..%s (n=%d) | OOS %s..%s (n=%d)"
    % (days[0], days[mid - 1], mid, days[mid], days[-1], len(days) - mid))
log("%-30s %11s %11s %11s %11s" % ("cell", "IS d_total", "IS worst", "OOS d_total", "OOS worst"))
for k in CANDS:
    dm = cells[k]
    isd = [float(dm[d]["hedged_net"]) - base[d] for d in days if d in d_is and d in dm]
    ood = [float(dm[d]["hedged_net"]) - base[d] for d in days if d in d_oos and d in dm]
    isn = [float(dm[d]["hedged_net"]) for d in days if d in d_is and d in dm]
    oon = [float(dm[d]["hedged_net"]) for d in days if d in d_oos and d in dm]
    log("%-30s %11d %11d %11d %11d"
        % (str(k), sum(isd), min(isn), sum(ood), min(oon)))
log("%-30s %11s %11d %11s %11d"
    % ("NAKED", "-", min(base[d] for d in days if d in d_is),
       "-", min(base[d] for d in days if d in d_oos)))

log("")
log("=" * 112)
log("3. THE INCUMBENT NULL - TimeB's CLOCK EXIT (flatten the whole book at a fixed time)")
log("=" * 112)
log("(approximated from the recorded portfolio peak/final: a clock exit cannot be priced")
log(" from b2_cells alone, so this is reported from the Arm-A engine's curve in stage2.)")
log("Naked worst=%d  mean=%d" % (min(bl), st.mean(bl)))

open(os.path.join(RES, "b2_robust.txt"), "w").write("\n".join(REP) + "\n")
print("\nwrote results/b2_robust.txt")
