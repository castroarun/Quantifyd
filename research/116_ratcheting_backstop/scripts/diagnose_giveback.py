#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/116 - WHY the ratchet fails: the anatomy of give-back on a decaying straddle.

Walks the same frozen constructions minute by minute (no defence, full window) and asks:
  1. WHERE in the window does peak open profit occur?  (if it's at the end, nothing to trail)
  2. Once deep in profit, how far back does it actually come?  (Arun's "big bad crack")
  3. How often does a deep-in-profit day come ALL THE WAY back to the static stop?
"""
import sqlite3, csv, os, sys, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_ratchet_sweep import (constructions, trading_days, dte_of, load_day,
                               build_series, VENUE, CHG, BACKSTOP, CHAIN, EXCLUDE_DAYS)

RES = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "giveback_anatomy.csv")
DEEP = 0.40      # "deep in profit" = open profit >= 40% of the entry credit


def pctl(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))]


def main():
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    cons = constructions()
    rows = []
    for venue in ("NIFTY", "SENSEX"):
        V = VENUE[venue]
        lot = V["lot"]
        days = trading_days(c, venue)
        vcons = [x for x in cons if x[1] == venue]
        for day in days:
            if day in EXCLUDE_DAYS:
                continue
            d = load_day(c, venue, day)
            if not d:
                continue
            fexp, per, spot = d
            dte = dte_of(day, fexp, days)
            for name, _, book, cells in vcons:
                if dte not in cells:
                    continue
                e_win, x_win, sl = cells[dte]
                b = build_series(per, spot, V["step"], e_win, x_win)
                if not b:
                    continue
                K, credit, e_hm, ser = b
                stop_lvl = credit * min(1 + sl, BACKSTOP) if sl is not None else credit * BACKSTOP
                n = len(ser)
                pnl = [(credit - comb) for _, comb in ser]
                peak = max(pnl)
                ipeak = pnl.index(peak)
                final = pnl[-1]
                # deepest retrace from a running peak, over the whole window
                run, mae = -1e9, 0.0
                for x in pnl:
                    run = max(run, x)
                    mae = max(mae, run - x)
                # first minute open profit crossed DEEP*credit, and what happened after
                deep_i = next((i for i, x in enumerate(pnl) if x >= DEEP * credit), None)
                after_min = after_stop = None
                if deep_i is not None:
                    tail = pnl[deep_i:]
                    after_min = min(tail)
                    after_stop = 1 if any(credit - x >= stop_lvl for x in tail) else 0
                rows.append(dict(
                    construction=name, venue=venue, day=day, dte=dte, credit=round(credit, 2),
                    n_min=n, peak_frac_window=round(ipeak / max(1, n - 1), 3),
                    peak_rs=round(peak * lot), final_rs=round(final * lot),
                    giveback_rs=round((peak - final) * lot),
                    mae_from_peak_rs=round(mae * lot),
                    stop_dist_at_peak_rs=round((stop_lvl - (credit - peak)) * lot),
                    went_deep=1 if deep_i is not None else 0,
                    after_deep_min_rs=(round(after_min * lot) if after_min is not None else ""),
                    after_deep_hit_stop=(after_stop if after_stop is not None else "")))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", OUT, len(rows), "construction-days")

    def blk(sub, label):
        if not sub:
            return
        pf = [r["peak_frac_window"] for r in sub]
        gb = [r["giveback_rs"] for r in sub]
        mae = [r["mae_from_peak_rs"] for r in sub]
        sd = [r["stop_dist_at_peak_rs"] for r in sub]
        late = 100.0 * sum(1 for x in pf if x >= 0.90) / len(pf)
        print("\n-- %s (n=%d) --" % (label, len(sub)))
        print("  peak occurs at window fraction: p25=%.2f p50=%.2f p75=%.2f  | in last 10%% of window: %.0f%% of days"
              % (pctl(pf, 25), pctl(pf, 50), pctl(pf, 75), late))
        print("  give-back peak->close (Rs/lot):  p50=%d p90=%d max=%d  mean=%d"
              % (pctl(gb, 50), pctl(gb, 90), max(gb), st.mean(gb)))
        print("  worst retrace from running peak: p50=%d p90=%d max=%d"
              % (pctl(mae, 50), pctl(mae, 90), max(mae)))
        print("  UNUSED stop distance at the peak: p50=%d p90=%d max=%d  <-- what Arun is worried about"
              % (pctl(sd, 50), pctl(sd, 90), max(sd)))
        deep = [r for r in sub if r["went_deep"]]
        if deep:
            hit = sum(1 for r in deep if r["after_deep_hit_stop"] == 1)
            gaveall = sum(1 for r in deep if r["after_deep_min_rs"] <= 0)
            print("  days that went DEEP (open profit >= %d%% of credit): %d/%d (%.0f%%)"
                  % (DEEP * 100, len(deep), len(sub), 100.0 * len(deep) / len(sub)))
            print("    ...of those, later touched the STATIC STOP: %d (%.1f%%)" % (hit, 100.0 * hit / len(deep)))
            print("    ...of those, later went back to <= Rs0 open:  %d (%.1f%%)" % (gaveall, 100.0 * gaveall / len(deep)))

    blk(rows, "ALL CONSTRUCTIONS")
    for cn in sorted({r["construction"] for r in rows}):
        blk([r for r in rows if r["construction"] == cn], cn)
    for ven in ("NIFTY", "SENSEX"):
        for dte in sorted({r["dte"] for r in rows if r["venue"] == ven}):
            blk([r for r in rows if r["venue"] == ven and r["dte"] == dte], "%s DTE%d" % (ven, dte))


if __name__ == "__main__":
    main()
