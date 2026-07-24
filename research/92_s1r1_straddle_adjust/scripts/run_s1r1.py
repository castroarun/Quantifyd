# -*- coding: utf-8 -*-
"""Run the S1/R1 straddle-adjust + breakout-flip backtest over the recorded NIFTY chain."""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/arun/quantifyd")
sys.path.insert(0, HERE)
import numpy as np
from engine_s1r1 import load_day, daily_ohlc, pivots, sim_straddle, sim_breakout  # noqa

RES = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RES, exist_ok=True)

print("computing daily OHLC + pivots ...", flush=True)
OHLC = daily_ohlc()
dl = sorted(OHLC.keys())
prevmap = {dl[i]: dl[i - 1] for i in range(1, len(dl))}
print("days with OHLC:", len(dl), flush=True)

STRAD = ["hold", "cut", "price", "premium"]
BRK = ["price", "premium"]
res = {("strad", m): [] for m in STRAD}
res.update({("brk", m): [] for m in BRK})

n = len(dl)
for i, day in enumerate(dl, 1):
    if day not in prevmap:
        continue
    r1, s1 = pivots(OHLC[prevmap[day]])
    b = load_day(day)
    if b is None:
        continue
    for m in STRAD:
        r = sim_straddle(b, r1, s1, mode=m)
        if r:
            res[("strad", m)].append((day, r["pnl"], r.get("bdir")))
    for m in BRK:
        r = sim_breakout(b, r1, s1, mode=m)
        if r:
            res[("brk", m)].append((day, r["pnl"], None))
    if i % 8 == 0 or i == n:
        print("  replay %d/%d (%s)" % (i, n, day), flush=True)


def summ(rows):
    p = np.array([x[1] for x in rows], float)
    if not len(p):
        return None
    cum = np.cumsum(p)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    return dict(total=round(float(p.sum())), n=len(p), win=round(100 * float((p > 0).mean())),
                maxdd=round(dd), best=round(float(p.max())), worst=round(float(p.min())),
                calmar=(round(float(p.sum()) / abs(dd), 2) if dd < 0 else None))


order = [("strad", "hold"), ("strad", "cut"), ("strad", "price"), ("strad", "premium"),
         ("brk", "price"), ("brk", "premium")]
label = {("strad", "hold"): "STRADDLE hold", ("strad", "cut"): "STRADDLE cut-only",
         ("strad", "price"): "STRADDLE trail-PRICE", ("strad", "premium"): "STRADDLE trail-PREMIUM",
         ("brk", "price"): "BREAKOUT trail-PRICE", ("brk", "premium"): "BREAKOUT trail-PREMIUM"}

print("\n=== S1/R1 systems — 64d, 2 lots (QTY 130), ST(7,3) ===")
print("%-24s %9s %4s %5s %7s %9s %9s %9s" %
      ("mode", "total", "n", "win%", "calmar", "maxDD", "best", "worst"))
rows_csv = []
for k in order:
    s = summ(res[k])
    if not s:
        continue
    print("%-24s %+9d %4d %5d %7s %+9d %+9d %+9d" %
          (label[k], s["total"], s["n"], s["win"],
           ("%.2f" % s["calmar"]) if s["calmar"] else "-", s["maxdd"], s["best"], s["worst"]))
    rows_csv.append(dict(mode=label[k], **s))

with open(os.path.join(RES, "s1r1_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode", "total", "n", "win", "calmar", "maxdd", "best", "worst"])
    w.writeheader()
    for r in rows_csv:
        w.writerow({k: r.get(k) for k in ["mode", "total", "n", "win", "calmar", "maxdd", "best", "worst"]})

# break-day-only view (straddle family only differs from HOLD on break days)
bd_days = set(d for d, _, bd in res[("strad", "cut")] if bd)
alln = len(res[("strad", "hold")])
print("\nBreak days (a 5-min close beyond R1/S1): %d of %d" % (len(bd_days), alln))
print("On BREAK DAYS only — straddle family (HOLD vs the adjustments):")
for k in [("strad", "hold"), ("strad", "cut"), ("strad", "price"), ("strad", "premium")]:
    rows = [(d, p) for d, p, bd in res[k] if d in bd_days]
    tot = sum(p for _, p in rows)
    win = round(100 * sum(1 for _, p in rows if p > 0) / max(1, len(rows)))
    print("  %-24s break-day total %+8d  win %d%%" % (label[k], tot, win))
