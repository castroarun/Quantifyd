#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/132 Stage E — attribute REAL rupees to the accidental directional bet.

"Rs per 100 index points" is an exposure, not an outcome. This converts it into the
rupees the unintended delta actually contributed, by multiplying each entry's net delta
by the index move that actually occurred between entry and exit.

delta_pnl ~= net_delta x (spot_exit - spot_entry) x qty

That is a first-order attribution (delta is not held constant over the holding period —
gamma on a short straddle makes the true contribution worse than linear on big moves), so
it is a LOWER bound on the damage a large move would do. It is reported alongside the
realised index move so the reader can see that the sign was a coin flip that landed well.

READ-ONLY. Writes results/delta_attrib.csv.
"""
import csv
import json
import os
import statistics as S
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common132 import CHAIN, Q, RES, VENUE, ro, load_day, hm2m, log_line

OUT = os.path.join(RES, "delta_attrib.csv")


def q(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    k = min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))
    return v[k]


def main():
    aud = {(r["src"], r["book"], r["day"], r["k_actual"]): r
           for r in csv.DictReader(open(os.path.join(RES, "entry_audit.csv")))}
    st = json.load(open(Q + "backtest_data/csl_paper_state.json"))
    c = ro(CHAIN)
    cache = {}
    rows = []
    for r in st["records"]:
        k = ("CSL", r["book"], r["day"], str(int(r["strike"])))
        a = aud.get(k)
        if not a or not a["net_delta"]:
            continue
        key = (r["sym"], r["day"])
        if key not in cache:
            if len(cache) > 12:
                cache.clear()
            cache[key] = load_day(c, r["sym"], r["day"])
        d = cache[key]
        if not d:
            continue
        _fexp, spot, _chain = d
        m0, m1 = int(a["entry_minute"]), hm2m(r["exit_ts"][:5])
        s0 = s1 = None
        for dd in range(0, 8):
            s0 = s0 or spot.get(m0 + dd) or spot.get(m0 - dd)
            s1 = s1 or spot.get(m1 + dd) or spot.get(m1 - dd)
        if not s0 or not s1:
            continue
        nd, qty = float(a["net_delta"]), float(a["qty"])
        move = s1 - s0
        rows.append(dict(
            book=r["book"], venue=r["sym"], day=r["day"], dte=r["dte"],
            misstrike=int(a["misstrike"]), steps_off=a["steps_off"],
            net_delta=round(nd, 4), qty=int(qty),
            spot_entry=round(s0, 1), spot_exit=round(s1, 1), move=round(move, 1),
            rs_per_100pt=round(nd * qty * 100),
            delta_pnl=round(nd * move * qty),
            booked_pnl=r["pnl"], reason=r["reason"]))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("=" * 88)
    print("STAGE E — rupees actually attributable to the UNINTENDED delta (CSL books)")
    print("=" * 88)
    print("delta_pnl = net_delta x realised index move x qty. First-order; gamma makes the")
    print("true figure WORSE than this on large moves, so it is a lower bound on the damage.\n")
    print("%-8s %5s %10s %10s %10s %10s %10s" % (
        "venue", "n", "sum", "mean", "med", "p10", "p90"))
    for ven in ("NIFTY", "SENSEX"):
        G = [r for r in rows if r["venue"] == ven]
        if not G:
            continue
        for lbl, H in (("all", G), ("mis-struck", [x for x in G if x["misstrike"]])):
            v = [x["delta_pnl"] for x in H]
            print("%-8s %5d %+10.0f %+10.0f %+10.0f %+10.0f %+10.0f   (%s)" % (
                ven, len(v), sum(v), S.mean(v), S.median(v), q(v, .1), q(v, .9), lbl))
    allm = [r for r in rows if r["misstrike"]]
    v = [r["delta_pnl"] for r in allm]
    print("\nALL mis-struck CSL entries: n=%d  TOTAL %+.0f  |per trade| med %.0f p90 %.0f max %.0f"
          % (len(v), sum(v), S.median([abs(x) for x in v]), q([abs(x) for x in v], .9),
             max(abs(x) for x in v)))
    print("  positive (the accident helped) %d / negative (it hurt) %d"
          % (sum(1 for x in v if x > 0), sum(1 for x in v if x < 0)))
    bp = [abs(float(r["booked_pnl"])) for r in allm]
    print("  |delta P&L| as a share of |booked P&L| on the same trade: median %.0f%%"
          % (100 * S.median([abs(r["delta_pnl"]) / max(abs(float(r["booked_pnl"])), 1)
                             for r in allm])))
    print("\nRealised index move over the holding period (the coin that was flipped):")
    for ven in ("NIFTY", "SENSEX"):
        G = [r for r in rows if r["venue"] == ven]
        if G:
            mv = [r["move"] for r in G]
            print("  %-7s n=%2d  med %+7.1f pts  |move| med %6.1f  max %6.1f"
                  % (ven, len(G), S.median(mv), S.median([abs(x) for x in mv]),
                     max(abs(x) for x in mv)))
    print("\nWorst single accidental bets:")
    print("%-24s %-7s %-11s %6s %8s %9s %10s %10s" % (
        "book", "venue", "day", "steps", "delta", "move", "delta P&L", "booked"))
    for r in sorted(allm, key=lambda x: -abs(x["delta_pnl"]))[:10]:
        print("%-24s %-7s %-11s %6s %8.3f %+9.1f %+10.0f %10s" % (
            r["book"][:24], r["venue"], r["day"], r["steps_off"], r["net_delta"],
            r["move"], r["delta_pnl"], r["booked_pnl"]))
    log_line("STAGE E done: %d rows -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
