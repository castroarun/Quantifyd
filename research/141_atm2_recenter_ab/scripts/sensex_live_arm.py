#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/141 — the LIVE re-center arm.

SENSEX ATM2 kept `move_stop_pct=0.004` + `move_stop_reenter=True` when NIFTY ATM2 was
switched to the rupee stop + one-and-done (r/96 scope fix, commit c95f10a, 2026-07-29).
It has therefore been running the RE-CENTER arm with recorded fills ever since. The NIFTY
916-ATM2 book ran the same arm BEFORE 2026-07-28. Both ledgers are direct evidence.

For each trading day: order the recorded strangles, find the ones that follow a MOVE_STOP
(= a re-center), and total what the re-centered straddles actually earned, per lot, and
after re-pricing the extra round trip with the research/122 measured cost model.

READ-ONLY on every DB. Writes results/live_arm.md only.
"""
import os
import sys
import sqlite3
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/arun/quantifyd/research/132_strike_misselection_cost/scripts")
from common132 import cost_per_lot  # noqa: E402

RES = os.path.join(HERE, "..", "results")
Q = "/home/arun/quantifyd/backtest_data/"
OUT = []

BOOKS = [
    # (label, db, lot, day_from, day_to, note)
    ("SENSEX ATM2 (live re-center arm, kept by c95f10a)",
     "sensex_atm2_trading.db", 20, "2026-07-27", "9999-12-31",
     "0.4% move-stop + re-center; matrix live on DTE0/DTE1"),
    ("NIFTY 916-ATM2 (re-center arm, BEFORE research/96)",
     "nas_916_atm2_trading.db", 65, "0000-01-01", "2026-07-28",
     "0.4% move-stop + re-center, the pre-July behaviour"),
    ("NIFTY squeeze-ATM2 (re-center arm, BEFORE research/96)",
     "nas_atm2_trading.db", 65, "0000-01-01", "2026-07-28",
     "0.4% move-stop + re-center, the pre-July behaviour"),
]
STOPS = ("MOVE_STOP", "RUPEE_STOP")


def p(m=""):
    OUT.append(m)
    print(m, flush=True)


def run(label, db, lot, d0, d1, note):
    c = sqlite3.connect("file:%s%s?mode=ro" % (Q, db), uri=True)
    rows = list(c.execute(
        "SELECT id, trade_date, call_entry_premium, put_entry_premium, "
        "call_exit_premium, put_exit_premium, lots, gross_pnl, net_pnl, exit_reason "
        "FROM nas_atm_trades WHERE trade_date>=? AND trade_date<=? ORDER BY trade_date, id",
        (d0, d1)))
    rows = [r for r in rows if (r[6] or 0) > 0]
    by = defaultdict(list)
    for r in rows:
        by[r[1]].append(r)
    p("")
    p("### %s" % label)
    p("")
    p("`%s`, lot %d, %s. Days with a recorded strangle: %d, strangles: %d."
      % (db, lot, note, len(by), len(rows)))
    p("")
    p("| day | lots | cycle 1 reason | cycle-1 net ₹ | re-centers | re-center net ₹ | re-center ₹/lot | re-center exits |")
    p("|---|---:|---|---:|---:|---:|---:|---|")
    tot_rc, tot_rc_pnl, tot_rc_perlot, tot_rc_cost = 0, 0.0, 0.0, 0.0
    n_rc_stopped, n_stop_days, n_stop_no_rc = 0, 0, 0
    for day in sorted(by):
        ts = by[day]
        if ts[0][9] not in STOPS:
            continue
        n_stop_days += 1
        rc = ts[1:]
        lots = ts[0][6] or 1
        if not rc:
            n_stop_no_rc += 1
            p("| %s | %d | %s | %+.0f | 0 | — | — | (no re-center) |"
              % (day, lots, ts[0][9], ts[0][8] or 0))
            continue
        rcp = sum((r[8] or 0) for r in rc)
        rcl = sum(((r[8] or 0) / float(r[6] or 1)) for r in rc)
        ccost = 0.0
        for r in rc:
            cred = (r[2] or 0) + (r[3] or 0)
            exi = (r[4] or 0) + (r[5] or 0)
            reason = "SL" if r[9] in STOPS else "TIME"
            if r[9] in STOPS:
                n_rc_stopped += 1
            ccost += cost_per_lot(cred, exi, lot, reason) * (r[6] or 1)
        tot_rc += len(rc)
        tot_rc_pnl += rcp
        tot_rc_perlot += rcl
        tot_rc_cost += ccost
        p("| %s | %d | %s | %+.0f | %d | %+.0f | %+.0f | %s |"
          % (day, lots, ts[0][9], ts[0][8] or 0, len(rc), rcp, rcl,
             ", ".join(r[9] for r in rc)))
    p("")
    if tot_rc:
        p("**Live re-center total: %d re-centers over %d stop-days "
          "(%d stop-days ended with no re-center).**" % (tot_rc, n_stop_days, n_stop_no_rc))
        p("")
        p("- as recorded (book's own ₹160/strangle brokerage): **%+.0f** total, "
          "**%+.0f ₹/lot** across all re-centers (mean **%+.0f ₹/lot** each)"
          % (tot_rc_pnl, tot_rc_perlot, tot_rc_perlot / tot_rc))
        p("- re-priced with the research/122 measured cost model "
          "(charges + slippage by exit type): extra-cycle cost **₹%.0f** total"
          % tot_rc_cost)
        p("- of the %d re-centered straddles, **%d stopped out again** (%.0f%%) — the "
          "re-center's own risk of repeating the loss"
          % (tot_rc, n_rc_stopped, 100.0 * n_rc_stopped / tot_rc))
    else:
        p("No re-centers recorded in this window.")


def main():
    os.makedirs(RES, exist_ok=True)
    p("# research/141 — the live re-center arm (recorded books)")
    for b in BOOKS:
        run(*b)
    with open(os.path.join(RES, "live_arm.md"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
