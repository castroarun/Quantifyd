#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/132 Stage A — the spot-vs-forward offset atlas.

For every recorded venue x day x minute, read the synthetic forward off the chain by
put-call parity and compare it to the recorded cash index print. Emit the offset, the
strike each level rounds to, and whether they differ.

This is the statistically solid half of the study: 90 recorded days x 2 venues x ~370
minutes, versus 52 actual CSL entries.

READ-ONLY. Writes results/offset_atlas.csv + results/stage.log.
"""
import csv
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common132 import (CHAIN, RES, VENUE, WD, LOG, ro, all_days, load_day,
                       read_forward, log_line, m2hm, trading_dte, SESS_END_M)

OUT = os.path.join(RES, "offset_atlas.csv")
FG = ["venue", "day", "weekday", "expiry", "dte_trd", "dte_cal", "minute", "hm",
      "spot", "fwd", "offset", "pcp_spread", "k_spot", "k_fwd", "misstrike", "steps_off"]


def main():
    os.makedirs(RES, exist_ok=True)
    c = ro(CHAIN)
    fout = open(OUT, "w", newline="")
    w = csv.DictWriter(fout, fieldnames=FG)
    w.writeheader()
    log_line("=== STAGE A: offset atlas ===")
    for sym, V in VENUE.items():
        step = V["step"]
        days = all_days(c, sym)
        log_line("%s: %d candidate days %s..%s (step %d)" % (sym, len(days), days[0], days[-1], step))
        kept = skipped = 0
        for day in days:
            d = load_day(c, sym, day)
            if not d:
                log_line("  %s %s SKIP (holiday/partial/no data)" % (sym, day))
                skipped += 1
                continue
            fexp, spot, chain = d
            mins = sorted(m for m in chain if 9 * 60 + 15 <= m <= SESS_END_M)
            if len(mins) < 200:
                log_line("  %s %s SKIP thin (%d mins)" % (sym, day, len(mins)))
                skipped += 1
                continue
            dte_t = trading_dte(day, fexp)
            dte_c = (date.fromisoformat(fexp) - date.fromisoformat(day)).days
            wd = WD[date.fromisoformat(day).weekday()]
            n = 0
            rows = []
            for mi in mins:
                sp = spot.get(mi)
                if not sp:
                    continue
                rf = read_forward(chain[mi], sp, step)
                if rf is None:
                    continue
                F, kref, spread = rf
                if spread > 0.25 * step:          # unreliable print — PCP does not hold
                    continue
                k_spot = int(round(sp / step) * step)
                k_fwd = int(round(F / step) * step)
                rows.append(dict(
                    venue=sym, day=day, weekday=wd, expiry=fexp, dte_trd=dte_t,
                    dte_cal=dte_c, minute=mi, hm=m2hm(mi), spot=round(sp, 2),
                    fwd=round(F, 2), offset=round(F - sp, 2), pcp_spread=round(spread, 2),
                    k_spot=k_spot, k_fwd=k_fwd,
                    misstrike=int(k_spot != k_fwd),
                    steps_off=int(round((k_fwd - k_spot) / step))))
                n += 1
            if n < 100:
                log_line("  %s %s SKIP few readable forwards (%d)" % (sym, day, n))
                skipped += 1
                continue
            w.writerows(rows)
            fout.flush()
            kept += 1
            ms = sum(r["misstrike"] for r in rows)
            offs = [r["offset"] for r in rows]
            log_line("  %s %s DTE%d %s: %d min, offset med %.1f [%.1f..%.1f], misstrike %d/%d = %.1f%%"
                     % (sym, day, dte_t, wd, n, sorted(offs)[n // 2], min(offs), max(offs),
                        ms, n, 100.0 * ms / n))
        log_line("%s: kept %d days, skipped %d" % (sym, kept, skipped))
    fout.close()
    log_line("STAGE A done -> %s" % OUT)


if __name__ == "__main__":
    main()
