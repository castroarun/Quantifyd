#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/125 Stage 6 - ARM C analysis: does spreading the SAME notional across
strikes or entry minutes reduce the joint tail?

Portfolios, all at EQUAL NOTIONAL (3 clones x 2 lots of the COMB-shape construction):
  CLONE_SAME      3 clones, same strike (offset 0), same entry 09:16   <- today's shape
  DIV_STRIKE_1    offsets -1 / 0 / +1  (one step apart), entry 09:16
  DIV_STRIKE_2    offsets -2 / 0 / +2
  DIV_ENTRY       offset 0, entries 09:16 / 09:31 / 09:46
  DIV_BOTH        (-1,09:16) (0,09:31) (+1,09:46)
Also a random-pairing placebo to show what noise looks like at this n.

READ-ONLY. Writes results/diversify_summary.txt, results/diversify_grid.csv
"""
import csv, os, random, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
REP = []


def log(m):
    REP.append(str(m)); print(m, flush=True)


def pct(xs, p):
    if not xs: return 0.0
    s = sorted(xs); k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


cells = defaultdict(dict)     # (venue,day) -> (entry,offset) -> row
meta = {}
with open(os.path.join(RES, "diversify_cells.csv")) as f:
    for r in csv.DictReader(f):
        cells[(r["venue"], r["day"])][(r["entry"], int(r["offset"]))] = r
        meta[(r["venue"], r["day"])] = (r["weekday"], int(r["dte"]))
log("diversify cells: %d venue-days" % len(cells))

PORT = {
    "CLONE_SAME":   [("09:16", 0), ("09:16", 0), ("09:16", 0)],
    "DIV_STRIKE_1": [("09:16", -1), ("09:16", 0), ("09:16", 1)],
    "DIV_STRIKE_2": [("09:16", -2), ("09:16", 0), ("09:16", 2)],
    "DIV_ENTRY":    [("09:16", 0), ("09:31", 0), ("09:46", 0)],
    "DIV_BOTH":     [("09:16", -1), ("09:31", 0), ("09:46", 1)],
    "DIV_ENTRY_WIDE": [("09:16", 0), ("09:46", 0), ("10:01", 0)],
}

for ven in ("NIFTY", "SENSEX"):
    days = sorted(d for (v, d) in cells if v == ven)
    log("")
    log("=" * 100)
    log("ARM C - %s : equal-notional 3-clone portfolios (COMB shape, 2 lots each clone)" % ven)
    log("=" * 100)
    res = {}
    for name, legs in PORT.items():
        tot, per = [], []
        for d in days:
            cc = cells[(ven, d)]
            if not all(l in cc for l in legs):
                continue
            v = sum(float(cc[l]["net_rs"]) for l in legs)
            tot.append(v); per.append((d, v))
        if len(tot) < 20: continue
        res[name] = (tot, per)
    if not res: continue
    n0 = len(res["CLONE_SAME"][0])
    log("%-17s %5s %10s %8s %8s %6s %10s %10s %10s %10s" % (
        "portfolio", "n", "total", "mean", "median", "win%", "worst", "p05", "p10", "vs_clone"))
    base_tot = sum(res["CLONE_SAME"][0])
    for name in PORT:
        if name not in res: continue
        tot, per = res[name]
        log("%-17s %5d %10d %8d %8d %6.1f %10d %10d %10d %10d" % (
            name, len(tot), sum(tot), st.mean(tot), st.median(tot),
            100.0 * sum(1 for x in tot if x > 0) / len(tot), min(tot),
            pct(tot, 5), pct(tot, 10), sum(tot) - base_tot))
    # paired per-day delta on the common days
    log("")
    log("Paired per-day delta vs CLONE_SAME (common days only):")
    cd = dict(res["CLONE_SAME"][1])
    for name in PORT:
        if name == "CLONE_SAME" or name not in res: continue
        pairs = [(cd[d], v) for d, v in res[name][1] if d in cd]
        dl = [b - a for a, b in pairs]
        if len(dl) < 20: continue
        m = st.mean(dl); sd = st.pstdev(dl) or 1e-9
        tstat = m / (sd / len(dl) ** .5)
        wb = min(b for _, b in pairs); wa = min(a for a, _ in pairs)
        log("  %-17s n=%3d  mean_delta=%+8.0f  median=%+8.0f  t=%+5.2f  worst %d -> %d (%+d)"
            % (name, len(dl), m, st.median(dl), tstat, wa, wb, wb - wa))
    # placebo: 3 random (entry,offset) legs, 200 draws -> what does noise look like?
    allleg = sorted({l for d in days for l in cells[(ven, d)]})
    rnd = random.Random(20260825)
    tails, totals = [], []
    for _ in range(200):
        legs = [rnd.choice(allleg) for _ in range(3)]
        tot = []
        for d in days:
            cc = cells[(ven, d)]
            if not all(l in cc for l in legs): continue
            tot.append(sum(float(cc[l]["net_rs"]) for l in legs))
        if len(tot) >= 20:
            tails.append(min(tot)); totals.append(sum(tot))
    if tails:
        log("")
        log("  PLACEBO (200 random 3-leg portfolios from the same cell menu):")
        log("    worst-day: p05=%d median=%d p95=%d   [CLONE_SAME=%d]"
            % (pct(tails, 5), st.median(tails), pct(tails, 95), min(res["CLONE_SAME"][0])))
        log("    total    : p05=%d median=%d p95=%d   [CLONE_SAME=%d]"
            % (pct(totals, 5), st.median(totals), pct(totals, 95), base_tot))

open(os.path.join(RES, "diversify_summary.txt"), "w").write("\n".join(REP) + "\n")
print("\nwrote results/diversify_summary.txt")
