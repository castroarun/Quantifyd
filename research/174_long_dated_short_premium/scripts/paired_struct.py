#!/usr/bin/env python3
"""
research/174 — paired straddle-vs-strangle at the SAME tenor, on shared entry days.

Unpaired medians lie at small n (playbook 6.5). This reports the distribution of
per-trade differences on the entry days both specs actually traded, plus the
tradeability gate the house format requires: win rate, avg win, avg loss, expectancy,
worst losing streak, trades per year.
"""
import csv
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
TAG = os.environ.get("R174_TAG", "")
SLIP = float(os.environ.get("R174_SLIP", "0.0075"))
LOT = 65

# re-derive nets from the trade dump using the same cost model
import sys                                                   # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_lt as E                                        # noqa: E402

rows = list(csv.DictReader(open(RES / ("p5_trades%s.csv" % TAG))))
by = defaultdict(dict)
for r in rows:
    # cost from gross premium turnover; we only stored credit + gross_prem + exit_cost,
    # which is enough for a 2-leg structure where |legs| == gross.
    cr, gp, xc = float(r["credit"]), float(r["gross_prem"]), float(r["exit_cost"])
    # exit gross turnover is approximated by |exit_cost| for 2-leg shorts (both legs
    # bought back); for 4-leg structures this understates it, so those are excluded.
    # 2-leg short structure: split the turnover over two short legs so the brokerage
    # leg-count and the STT side are both right (buying a short back pays no STT).
    entry_px = [(gp / 2.0, -1), (gp / 2.0, -1)]
    exit_px = [(abs(xc) / 2.0, -1), (abs(xc) / 2.0, -1)]
    cost = E.costs_points_legs(entry_px, exit_px, SLIP)
    by[r["spec"]][r["entry_date"]] = dict(net=float(r["gross_pts"]) - cost,
                                          days=float(r["days"]), credit=cr,
                                          date=r["entry_date"])


def gate(vals):
    wins = [v for v in vals if v > 0]
    loss = [v for v in vals if v <= 0]
    streak = cur = 0
    for v in vals:
        cur = cur + 1 if v <= 0 else 0
        streak = max(streak, cur)
    return dict(n=len(vals), wr=100.0 * len(wins) / len(vals),
                avg_win=sum(wins) / len(wins) if wins else 0,
                avg_loss=sum(loss) / len(loss) if loss else 0,
                exp=sum(vals) / len(vals), streak=streak)


print("TRADEABILITY GATE (slip %.2f%%, net of cost, points; 1 pt = Rs %d per lot)\n"
      % (SLIP * 100, LOT))
h = ("%-10s %5s %7s %9s %9s %10s %8s %9s %8s" %
     ("spec", "n", "win%", "avgWin", "avgLoss", "expectancy", "maxLossStreak",
      "tr/yr", "worst"))
print(h)
print("-" * len(h))
for spec, d in sorted(by.items()):
    order = [d[k] for k in sorted(d)]
    vals = [x["net"] for x in order]
    g = gate(vals)
    yrs = (int(max(d)[:4]) - int(min(d)[:4])) + 1
    print("%-10s %5d %7.1f %9.1f %9.1f %10.2f %8d %9.1f %8.1f" %
          (spec, g["n"], g["wr"], g["avg_win"], g["avg_loss"], g["exp"], g["streak"],
           g["n"] / float(yrs), min(vals)))

print("\n\nPAIRED DIFFERENCES on shared entry days")
labs = sorted(by)
base = os.environ.get("R174_BASE", "LIVE45")
if base not in by:
    base = labs[0]
print("baseline = %s\n" % base)
h2 = ("%-10s %7s %10s %10s %8s %9s %10s %11s" %
      ("spec", "n_pair", "medDelta", "meanDelta", "t", "wins", "win%", "worst_delta"))
print(h2)
print("-" * len(h2))
for spec in labs:
    if spec == base:
        continue
    common = sorted(set(by[spec]) & set(by[base]))
    if len(common) < 10:
        continue
    d = [by[spec][k]["net"] - by[base][k]["net"] for k in common]
    sd = st.stdev(d) if len(d) > 1 else 0.0
    m = sum(d) / len(d)
    print("%-10s %7d %10.2f %10.2f %8.2f %9d %10.1f %11.1f" %
          (spec, len(d), st.median(d), m,
           m / (sd / math.sqrt(len(d))) if sd else 0,
           sum(1 for x in d if x > 0), 100.0 * sum(1 for x in d if x > 0) / len(d),
           min(d)))
