#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/83 P2 — the test that actually matters: is a policy better than HOLD, PAIRED per day?

P1 asked "is the policy's mean > 0", which is the wrong question — HOLD's own mean is +254 and its CI
also spans zero, so "beats zero" says nothing about "beats doing nothing". This pairs each policy
against HOLD on the SAME day and bootstraps the DIFFERENCE.
"""
import io

src = io.open("/home/arun/quantifyd/research/83_giveback_exit_reentry/scripts/r83.py").read()
src = src.split("POLICIES = [")[0]          # reuse the loader + run_policy, skip P1's table
exec(compile(src, "r83", "exec"))

import numpy as np


def series(**kw):
    out = {}
    for s in SIMS:
        r = run_policy(s, **kw)
        if r:
            out[r["day"]] = (r["pnl"], r["dte"])
    return out


hold = series(kind="hold")


def paired(lab, **kw):
    p = series(**kw)
    days = [d for d in p if d in hold]
    diff = np.array([p[d][0] - hold[d][0] for d in days], float)
    rng = np.random.default_rng(3)
    bs = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    better = int((diff > 1).sum()); worse = int((diff < -1).sum())
    strip = np.sort(diff)[:-2].mean() if len(diff) > 4 else float("nan")
    print("%-26s diff mean %+7.0f  MEDIAN %+7.0f  strip2 %+7.0f  better %2d / worse %2d  CI[%+6.0f,%+6.0f]  %s" % (
        lab, diff.mean(), np.median(diff), strip, better, worse, lo, hi,
        "BEATS HOLD" if lo > 0 else "not distinguishable"))


print("\n=== PAIRED vs HOLD — the test that matters (n=%d days, per 1 lot, net of cost) ===\n" % len(hold))
paired("TARGET +1000/lot", kind="target", p1=1000)
paired("TARGET +1500/lot", kind="target", p1=1500)
paired("TARGET +2000/lot", kind="target", p1=2000)
paired("TRAIL arm1000 ret25%", kind="trail", p1=1000, p2=0.25)
paired("EXIT 15:00", kind="exit1500")
paired("TGT1500 -> REENTER 30m", kind="target", p1=1500, reenter_after=30)
paired("TGT2000 -> REENTER 30m", kind="target", p1=2000, reenter_after=30)
print("\nDONE")
