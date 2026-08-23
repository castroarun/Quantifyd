#!/usr/bin/env python3
"""How close did trades actually get to the 50% target / 200% stop?

The monitoring-frequency conclusion rests on this: if the combined premium never got
near the trigger levels, no check frequency can matter. Prints, per trade, the minimum
and maximum premium ratio on (a) the real daily CLOSE basis and (b) the real daily
leg-low / leg-high basis (the absolute intraday bound), plus 5-min spot coverage gaps.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine45 import (connect, trading_days, monthly_expiries, nifty_daily_close,
                      nifty_5min, build_trade)

WIN_LO, WIN_HI = "2019-01-01", "2026-06-30"


def main():
    con = connect()
    days = trading_days(con, "2018-06-01")
    spot = nifty_daily_close(con)
    exps = monthly_expiries(con, days, "2018-06-01", "2026-08-31")

    trades = []
    for ym, exp in exps.items():
        t = build_trade(con, exp, days, spot)
        if t and WIN_LO <= t["entry_date"] <= WIN_HI:
            trades.append(t)

    d0 = min(t["entry_date"] for t in trades)
    d1 = max(t["time_exit_date"] for t in trades)
    bars = nifty_5min(con, d0, d1)

    print("=== how near did the premium get to the triggers? (ratio to entry credit) ===")
    print("%-12s %8s %8s | %8s %8s | %s" %
          ("entry", "minClose", "maxClose", "minLegLo", "maxLegHi", "5min-gap-days"))
    near_target, near_stop, gaps_total = 0, 0, 0
    rows = []
    for t in trades:
        cr = t["credit"]
        pc = t["path"][1:]
        minc = min(r["comb"] for r in pc) / cr
        maxc = max(r["comb"] for r in pc) / cr
        los = [r["lo"] for r in pc if r["lo"]]
        his = [r["hi"] for r in pc if r["hi"]]
        minl = (min(los) / cr) if los else float("nan")
        maxh = (max(his) / cr) if his else float("nan")
        gap = sum(1 for r in pc if r["date"] not in bars)
        gaps_total += gap
        rows.append((t["entry_date"], minc, maxc, minl, maxh, gap))
        if minl <= 0.50:
            near_target += 1
        if maxh >= 2.00:
            near_stop += 1
    for r in sorted(rows, key=lambda x: x[3]):
        print("%-12s %8.2f %8.2f | %8.2f %8.2f | %d" % r)

    print("\ntrades whose LEG-LOW sum pierced 0.50x  : %d / %d" % (near_target, len(trades)))
    print("trades whose LEG-HIGH sum pierced 2.00x : %d / %d" % (near_stop, len(trades)))
    print("total trade-days with no 5-min spot bars: %d" % gaps_total)

    # of the leg-low piercers, how low did the CLOSE basis actually get?
    print("\nFor trades whose leg-low pierced 0.50x, the daily-CLOSE minimum ratio was:")
    for e, minc, maxc, minl, maxh, gap in rows:
        if minl <= 0.50:
            print("  %s  legLow=%.2f  close=%.2f  -> %s" %
                  (e, minl, minc,
                   "CLOSE also under 0.50" if minc <= 0.50 else "close never under 0.50"))


if __name__ == "__main__":
    main()
