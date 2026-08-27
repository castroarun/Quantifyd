#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/132 quick QA on the offset atlas: is the gap a real basis or a bad spot feed?

The decisive test is expiry convergence: at DTE0 late in the session the forward MUST
collapse onto the cash index. If it does, the offset is a genuine forward basis and the
recorded spot is fine. If DTE0-late still shows the full offset, the spot print is biased.
"""
import csv
import os
import statistics as S
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
rows = list(csv.DictReader(open(os.path.join(RES, "offset_atlas.csv"))))
for r in rows:
    r["offset"] = float(r["offset"])
    r["minute"] = int(r["minute"])
    r["dte_trd"] = int(r["dte_trd"])
    r["misstrike"] = int(r["misstrike"])
    r["pcp_spread"] = float(r["pcp_spread"])


def q(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    i = min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))
    return v[i]


def blk(title, groups, keyname):
    print("\n" + title)
    print("%-10s %6s %8s %8s %8s %8s %8s %8s %8s" % (
        keyname, "n", "p05", "p25", "med", "p75", "p95", "|med|", "mis%"))
    for k, g in groups:
        o = [x["offset"] for x in g]
        ms = 100.0 * sum(x["misstrike"] for x in g) / len(g)
        print("%-10s %6d %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %7.1f%%" % (
            str(k), len(g), q(o, .05), q(o, .25), q(o, .5), q(o, .75), q(o, .95),
            abs(S.median(o)), ms))


for venue in ("NIFTY", "SENSEX"):
    V = [r for r in rows if r["venue"] == venue]
    step = 50 if venue == "NIFTY" else 100
    print("\n" + "=" * 84)
    print("%s  step=%d  n=%d minutes  %d days" % (
        venue, step, len(V), len({r["day"] for r in V})))
    o = [r["offset"] for r in V]
    print("  offset  p01 %.1f  p05 %.1f  p25 %.1f  MED %.1f  p75 %.1f  p95 %.1f  p99 %.1f" % (
        q(o, .01), q(o, .05), q(o, .25), q(o, .5), q(o, .75), q(o, .95), q(o, .99)))
    print("  |offset| MED %.1f   share |offset| > step/2 (= flips the rounding at the midpoint): %.1f%%" % (
        S.median([abs(x) for x in o]), 100.0 * sum(1 for x in o if abs(x) > step / 2.0) / len(o)))
    print("  MIS-STRIKE RATE: %.1f%% of minutes" % (100.0 * sum(r["misstrike"] for r in V) / len(V)))
    from collections import Counter
    print("  steps off:", dict(Counter(int(r["steps_off"]) for r in V).most_common()))
    print("  PCP cross-strike spread: med %.2f p95 %.2f (noise floor)" % (
        S.median([r["pcp_spread"] for r in V]), q([r["pcp_spread"] for r in V], .95)))

    # by DTE
    ks = sorted({r["dte_trd"] for r in V})
    blk("  -- by trading-DTE --", [(k, [r for r in V if r["dte_trd"] == k]) for k in ks], "DTE")

    # expiry convergence: DTE0 by hour  <-- the decisive spot-feed test
    d0 = [r for r in V if r["dte_trd"] == 0]
    if d0:
        blk("  -- DTE0 by hour (forward MUST converge to spot) --",
            [(h, [r for r in d0 if r["minute"] // 60 == h]) for h in sorted({r["minute"] // 60 for r in d0})],
            "hour")

    # by time of day, all DTE
    blk("  -- by hour, all DTE --",
        [(h, [r for r in V if r["minute"] // 60 == h]) for h in sorted({r["minute"] // 60 for r in V})],
        "hour")

    # per-day medians: is the level stable across days?
    dm = sorted(((d, S.median([r["offset"] for r in V if r["day"] == d]))
                 for d in {r["day"] for r in V}), key=lambda x: x[0])
    meds = [m for _, m in dm]
    print("\n  per-day median offset: n=%d  min %.1f  p25 %.1f  MED %.1f  p75 %.1f  max %.1f"
          % (len(meds), min(meds), q(meds, .25), S.median(meds), q(meds, .75), max(meds)))
    print("  sign flips across days: %d positive, %d negative"
          % (sum(1 for m in meds if m > 0), sum(1 for m in meds if m < 0)))
    # intraday dispersion within a day (basis should drift slowly; LTP noise would not)
    iq = []
    for d in {r["day"] for r in V}:
        oo = [r["offset"] for r in V if r["day"] == d]
        iq.append(q(oo, .75) - q(oo, .25))
    print("  within-day IQR of offset: MED %.1f (vs between-day spread %.1f)"
          % (S.median(iq), q(meds, .95) - q(meds, .05)))
    print("  last 10 recorded days:")
    for d, m in dm[-10:]:
        g = [r for r in V if r["day"] == d]
        print("     %s DTE%-2s %s  med %+7.1f  mis %5.1f%%" % (
            d, g[0]["dte_trd"], g[0]["weekday"], m,
            100.0 * sum(x["misstrike"] for x in g) / len(g)))

# month drift
print("\n" + "=" * 84)
print("MONTHLY DRIFT (median offset / mis-strike rate)")
print("%-8s %-8s %6s %8s %8s" % ("venue", "month", "n", "med", "mis%"))
for venue in ("NIFTY", "SENSEX"):
    V = [r for r in rows if r["venue"] == venue]
    for mo in sorted({r["day"][:7] for r in V}):
        g = [r for r in V if r["day"][:7] == mo]
        print("%-8s %-8s %6d %8.1f %7.1f%%" % (
            venue, mo, len(g), S.median([x["offset"] for x in g]),
            100.0 * sum(x["misstrike"] for x in g) / len(g)))
