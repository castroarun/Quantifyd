#!/usr/bin/env python3
"""
research/174 — read p1_summary*.csv and answer the tenor question in margin terms.

Margin per lot is MEASURED from Kite (results/margin_by_tenor.json), interpolated by DTE,
because that is the binding constraint for a long-dated short, not the point P&L.
"""
import csv
import json
import os
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
TAG = os.environ.get("R174_TAG", "")
SLIP = os.environ.get("R174_SLIP", "0.0075")

MJ = json.load(open(RES / "margin_by_tenor.json"))
MROWS = sorted([(r["dte"], r["STR"], r.get("STG2.5"), r.get("IC2.5w7"), r.get("WS7"))
                for r in MJ["rows"] if r.get("STR")])


def margin_at(dte, col=1):
    """Linear interpolation of the measured Kite margin curve, flat outside the range."""
    xs = [r[0] for r in MROWS]
    ys = [r[col] for r in MROWS]
    if dte <= xs[0]:
        return ys[0]
    if dte >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if dte <= xs[i]:
            f = (dte - xs[i - 1]) / float(xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


rows = [r for r in csv.DictReader(open(RES / ("p1_summary%s.csv" % TAG)))
        if r["slip"] == SLIP]
LOT = 65

print("NIFTY short ATM straddle — tenor bake-off. Slippage %s of premium, vol floor 25,"
      " no stop.\nMargin per lot MEASURED from Kite %s (spot %.0f).\n"
      % (SLIP, MJ["asof"], MJ["spot"]))

hdr = ("%5s %5s %6s %5s %6s %7s %8s %7s %6s %8s %9s %8s %9s %9s %8s" %
       ("tenor", "exit", "target", "n", "indep", "win%", "avgCred", "avgNet", "t",
        "days", "pts/lotyr", "marginL", "ret%/yr", "beSlip%", "medVol"))
print(hdr)
print("-" * len(hdr))
best = {}
for r in rows:
    if r["vol_floor"] != "25" or r["target"] not in ("0.5", "none"):
        continue
    T, X = int(r["tenor"]), int(r["exit_dte"])
    n = int(r["n"])
    if n < 6:
        continue
    # Capital you must actually HAVE is the ENTRY requirement (margin declines as the
    # contract ages). We rank on that and show the mid-life figure for reference.
    marg = margin_at(T)
    marg_mid = margin_at((T + X) / 2.0)
    ply = float(r["pts_per_lotyear"])
    ret = 100.0 * ply * LOT / marg
    key = (T, r["target"])
    rec = (ret, r, marg, ply)
    if key not in best or ret > best[key][0]:
        best[key] = rec
    print("%5d %5d %6s %5d %6s %7s %8s %7s %6s %8s %9s %8.2f %9.1f %8.2f %8s" %
          (T, X, r["target"], n, r["n_indep"], r["win_rate"], r["avg_credit"],
           r["avg_net"], r["t_stat"], r["avg_days"], r["pts_per_lotyear"],
           marg / 1e5, ret, 100 * float(r["breakeven_slip"]), r["med_entry_vol"]))

print("\n\nBEST EXIT PER TENOR, ranked by return on measured margin (target=0.5 rows)")
print("%5s %5s %5s %7s %6s %9s %8s %9s %10s %9s" %
      ("tenor", "exit", "n", "avgNet", "t", "pts/lotyr", "marginL", "ret%/yr",
       "window", "medVol"))
for (T, tg), (ret, r, marg, ply) in sorted(best.items(), key=lambda kv: -kv[1][0]):
    if tg != "0.5":
        continue
    print("%5d %5s %5s %7s %6s %9s %8.2f %9.1f %10s %9s" %
          (T, r["exit_dte"], r["n"], r["avg_net"], r["t_stat"], ply, marg / 1e5, ret,
           r["first"][:7] + ".." + r["last"][2:7], r["med_entry_vol"]))
