#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/141 — the mechanism: what happens to the RE-CENTERED straddle.

The live SENSEX sample (6 re-centers) had 0 of them stop out again. The replay says
that is a lucky draw. This quantifies it: across the 88-day replay, of every cycle-2+
straddle, how many stopped out again, and what did they earn gross vs cost.
Reads results/arms_daily.csv only.
"""
import os
import csv
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = []


def p(m=""):
    OUT.append(m)
    print(m, flush=True)


def main():
    rows = []
    with open(os.path.join(RES, "arms_daily.csv")) as f:
        rows = list(csv.DictReader(f))
    p("# research/141 — re-centered-straddle outcome mix (the mechanism)")
    for venue in ("NIFTY", "SENSEX"):
        p("")
        p("### %s — of every RE-CENTERED (cycle 2+) straddle in the replay" % venue)
        p("")
        p("| arm | cycle-2+ straddles | stopped out AGAIN | ran to 15:15 | re-stop rate |")
        p("|---|---:|---:|---:|---:|")
        agg = defaultdict(lambda: [0, 0])
        for r in rows:
            if r["venue"] != venue:
                continue
            rs = r["reasons"].split("|")[1:]
            for x in rs:
                agg[r["arm"]][0 if x == "SL" else 1] += 1
        for arm in ["RECENTER_1", "RECENTER_2", "RECENTER_3", "RECENTER_5",
                    "RECENTER_3_CD15", "RECENTER_5_CD15", "MOVESTOP_RECENTER", "MOVESTOP_RC1", "MOVESTOP_RC_CD15"]:
            if arm not in agg:
                continue
            sl, tm = agg[arm]
            n = sl + tm
            if not n:
                continue
            p("| %s | %d | %d | %d | **%.0f%%** |" % (arm, n, sl, tm, 100.0 * sl / n))
    with open(os.path.join(RES, "cycle_mix.md"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
