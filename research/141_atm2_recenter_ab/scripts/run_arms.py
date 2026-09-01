#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/141 main sweep — every arm x every recorded day x both venues.

Writes results/arms_daily.csv incrementally (flush per day). READ-ONLY on the DB.
"""
import os
import sys
import csv
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/arun/quantifyd/research/132_strike_misselection_cost/scripts")

from common132 import CHAIN, VENUE, WD, ro, load_day, all_days, trading_dte  # noqa: E402
from engine141 import ARMS, replay_day  # noqa: E402

RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "arms_daily.csv")
LOG = os.path.join(RES, "run_arms.log")

FIELDS = ["venue", "day", "weekday", "expiry", "dte_trd", "dte_cal", "arm",
          "n_cycles", "n_recenters", "stop_fired", "credit0",
          "gross", "cost", "net", "gross_c1", "cost_c1", "net_c1",
          "gross_after", "cost_after", "net_after",
          "first_exit", "last_exit", "strikes", "reasons"]


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    c = ro(CHAIN)
    fo = open(OUT, "w", newline="")
    w = csv.DictWriter(fo, fieldnames=FIELDS)
    w.writeheader()
    for sym in ("NIFTY", "SENSEX"):
        days = all_days(c, sym)
        log("%s: %d candidate days %s..%s" % (sym, len(days), days[0], days[-1]))
        kept = 0
        for day in days:
            d = load_day(c, sym, day)
            if not d:
                log("  %s %s SKIP (holiday/partial/no data)" % (sym, day))
                continue
            fexp, spot, chain = d
            dte_t = trading_dte(day, fexp)
            dte_c = (date.fromisoformat(fexp) - date.fromisoformat(day)).days
            wd = WD[date.fromisoformat(day).weekday()]
            nrows = 0
            summ = []
            for arm, cfg in ARMS:
                r = replay_day(spot, chain, sym, cfg)
                if not r:
                    continue
                row = dict(venue=sym, day=day, weekday=wd, expiry=fexp,
                           dte_trd=dte_t, dte_cal=dte_c, arm=arm)
                row.update({k: (round(v, 2) if isinstance(v, float) else v)
                            for k, v in r.items()})
                w.writerow(row)
                nrows += 1
                if arm in ("ONE_AND_DONE", "RECENTER_5"):
                    summ.append("%s net%+.0f rc%d" % (arm[:4], r["net"], r["n_recenters"]))
            kept += 1
            log("  %s %s %s dte_t=%d rows=%d  %s" % (sym, day, wd, dte_t, nrows, " | ".join(summ)))
            fo.flush()
        log("%s: kept %d days" % (sym, kept))
    fo.close()
    log("DONE -> %s" % OUT)


if __name__ == "__main__":
    main()
