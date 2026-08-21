#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/120 - the adversarial half: block test, concentration, and the Stage-A/Stage-B join.

1) BLOCK TEST - the surface's shape (morning good / midday bad / late so-so) as a 3-parameter
   claim instead of a 110-cell search. Paired by day, both venues, both arms.
2) CONCENTRATION - leave-one-out on every headline book. Does one Friday carry the verdict?
   (research/118's whole lesson.)
3) STAGE-A vs STAGE-B JOIN - rank-correlate each grid cell's 16-Friday net against the SAME
   cell's underlying-excursion risk measured over 274 Fridays. If the correlation is POSITIVE,
   the windows that pay are the windows that are dangerous, and "decays well with least
   possibility of volatile moves" has no solution.
4) PER-DAY table for the books that matter.
"""
import csv, os, math
import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
lines = []
def P(s=""):
    lines.append(s); print(s, flush=True)


def load(fn):
    with open(os.path.join(RES, fn)) as f:
        return list(csv.DictReader(f))


grid = load("stage_a_trades.csv")
volw = load("volclock_windows.csv")
marg = load("marginal_slot_trades.csv")
days = sorted({r["day"] for r in grid})

BLOCKS = {
    "MORNING 09:35-11:05": ["09:35", "09:50", "10:05", "10:20", "10:35", "10:50", "11:05"],
    "MIDDAY  11:20-13:20": ["11:20", "11:35", "11:50", "12:05", "12:20", "12:35", "12:50",
                            "13:05", "13:20"],
    "LATE    13:35-14:30": ["13:35", "13:50", "14:05", "14:20", "14:30"],
}

P("=" * 104)
P("1) BLOCK TEST - the surface's SHAPE as a 3-parameter claim (mean net Rs/lot/Friday, n = all clean Fridays)")
P("   Each block = the average of its cells on that day, so the day is the unit and the")
P("   comparison is paired. Durations 90 and 120 only (45/60 are cost-dominated everywhere).")
P("=" * 104)
P("  %-8s %-7s %-22s %8s %8s %7s %7s" % ("venue", "arm", "block", "mean", "median", "win%", "t"))
blockser = {}
for ven in ("NIFTY", "SENSEX"):
    for arm in ("SL20", "NOSTOP"):
        for bn, starts in BLOCKS.items():
            per = []
            for d in days:
                v = [float(r["net"]) for r in grid
                     if r["venue"] == ven and r["arm"] == arm and r["day"] == d
                     and r["start"] in starts and r["dur"] in ("90", "120")]
                if v:
                    per.append(float(np.mean(v)))
            a = np.array(per)
            t = a.mean() / (a.std(ddof=1) / math.sqrt(len(a)))
            blockser[(ven, arm, bn)] = a
            P("  %-8s %-7s %-22s %8.0f %8.0f %7.0f %7.2f"
              % (ven, arm, bn, a.mean(), np.median(a), 100 * (a > 0).mean(), t))
        P()
P("  MORNING minus MIDDAY, paired by Friday (the single claim the surface makes):")
for ven in ("NIFTY", "SENSEX"):
    for arm in ("SL20", "NOSTOP"):
        d = blockser[(ven, arm, "MORNING 09:35-11:05")] - blockser[(ven, arm, "MIDDAY  11:20-13:20")]
        t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
        p = 2 * (1 - sps.t.cdf(abs(t), len(d) - 1))
        P("    %-7s %-7s  diff = %+7.0f Rs/lot/Fri   t = %5.2f   p = %.4f   positive on %d/%d Fridays"
          % (ven, arm, d.mean(), t, p, int((d > 0).sum()), len(d)))

P()
P("=" * 104)
P("2) CONCENTRATION - leave-one-out on every book that matters (research/118's lesson)")
P("=" * 104)
books = {}
for r in marg:
    books.setdefault(r["book"], {})[r["day"]] = float(r["net"])
P("  %-24s %8s %8s %10s %10s %8s" %
  ("book", "mean", "worst day", "mean ex-worst", "mean ex-best", "sign flips?"))
for bn in ("COMB_NIFTY_full", "A_LIVE_TimeB_N_10_12", "ALT_N_0935_1135", "ALT_N_0950_1150",
           "B_N_1405_1535cap", "B_N_1400_1520", "B_N_1300_1400_SL25",
           "SX_0935_1135", "SX_0950_1120", "SX_1000_1200"):
    if bn not in books:
        continue
    a = np.array([books[bn][d] for d in sorted(books[bn])])
    exw = np.delete(a, a.argmin()).mean()
    exb = np.delete(a, a.argmax()).mean()
    flip = "YES" if (np.sign(a.mean()) != np.sign(exb) or np.sign(a.mean()) != np.sign(exw)) else "no"
    P("  %-24s %8.0f %8.0f %10.0f %10.0f %8s" % (bn, a.mean(), a.min(), exw, exb, flip))

P()
P("=" * 104)
P("3) STAGE-A (clean Fridays of option truth) JOINED TO STAGE-B (274 SENSEX / 542 NIFTY Fridays")
P("   of underlying risk). Does the window that PAYS sit where the index is CALM?")
P("=" * 104)
vb = {}
for r in volw:
    vb[(r["series"], r["scope"], r["start"], r["dur"])] = r
for ven, ser in (("NIFTY", "NIFTY50_5min"), ("SENSEX", "SENSEX_1min")):
    for arm in ("SL20",):
        xs, ys, zs, labs = [], [], [], []
        for r in grid:
            if r["venue"] != ven or r["arm"] != arm:
                continue
            k = (r["venue"], r["arm"], r["start"], r["dur"])
            pass
        cells = {}
        for r in grid:
            if r["venue"] == ven and r["arm"] == arm:
                cells.setdefault((r["start"], r["dur"]), []).append(float(r["net"]))
        for (s, d), v in cells.items():
            b = vb.get((ser, "FRI", s, d))
            if not b or len(v) < len(days):
                continue
            xs.append(float(b["mean_exc_bp"]))
            ys.append(float(np.mean(v)))
            zs.append(float(b["pct_gt_30bp"]))
            labs.append((s, d))
        rho, pv = sps.spearmanr(xs, ys)
        rho2, pv2 = sps.spearmanr(zs, ys)
        P("  %-7s vs %-14s cells=%d" % (ven, ser, len(xs)))
        P("      Spearman( long-run mean excursion , Friday-sample mean net ) = %+.3f  (p=%.4f)" % (rho, pv))
        P("      Spearman( long-run %%days >30bp     , Friday-sample mean net ) = %+.3f  (p=%.4f)" % (rho2, pv2))
        # the calmest cells by the long sample, and what they earned
        order = np.argsort(xs)
        P("      5 CALMEST windows over the long sample, and what they actually earned:")
        for i in order[:5]:
            P("        %-6s %-5s  long-run mean exc %5.1f bp (%4.1f%% days >30bp)   sample net %+6.0f"
              % (labs[i][0], labs[i][1], xs[i], zs[i], ys[i]))
        P("      5 MOST DANGEROUS windows over the long sample, and what they earned:")
        for i in order[-5:]:
            P("        %-6s %-5s  long-run mean exc %5.1f bp (%4.1f%% days >30bp)   sample net %+6.0f"
              % (labs[i][0], labs[i][1], xs[i], zs[i], ys[i]))
        P()

P("=" * 104)
P("4) PER-DAY, the books that matter (net Rs/lot)")
P("=" * 104)
show = ["A_LIVE_TimeB_N_10_12", "ALT_N_0935_1135", "B_N_1405_1535cap", "B_N_1300_1400_SL25",
        "COMB_NIFTY_full", "SX_0935_1135"]
P("  %-12s" % "Friday" + "".join("%22s" % b[:21] for b in show))
for d in days:
    P("  %-12s" % d + "".join("%22s" % ("%+d" % books[b][d] if d in books.get(b, {}) else "-")
                              for b in show))
P("  %-12s" % "MEAN" + "".join("%22s" % ("%+d" % np.mean([books[b][x] for x in sorted(books[b])]))
                               for b in show))

open(os.path.join(RES, "robustness_report.txt"), "w").write("\n".join(lines) + "\n")
