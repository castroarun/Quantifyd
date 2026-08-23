#!/usr/bin/env python3
"""
Backfill stale DAILY bars for the Market Mosaic universe.

Audit on 2026-08-21 found 136 of the 375 mosaic symbols (36%) stale: 133 stop at
2026-05-15, GSPL at 2026-05-07, JBCHEPHARM at 2026-07-16 and TATAMOTORS at
2025-10-17 — while NIFTY50 and 239 other names run to 2026-08-21. Every Mosaic
view (treemap colour, sector medians, quadrant position, the Leaders list) was
therefore mixing May numbers with August numbers under a single "as of" date.

Daily bars only. Additive. VPS-only by the data_manager guard. Nothing is deleted.
Run:  python3 research/119_45dte_short_straddle/scripts/backfill_stale_daily.py
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta

sys.path.insert(0, "/home/arun/quantifyd")

from kiteconnect import KiteConnect
from services.data_manager import get_data_manager

DB = "/home/arun/quantifyd/backtest_data/market_data.db"
SYMS_FILE = "/tmp/mosaic_syms.txt"
FROM = datetime(2026, 4, 1)          # generous overlap before the 2026-05 cliff
TATAMOTORS_FROM = datetime(2025, 9, 1)   # that one is stale since 2025-10-17


def last_bar(con, sym):
    r = con.execute("SELECT MAX(date) FROM market_data_unified "
                    "WHERE symbol=? AND timeframe='day'", (sym,)).fetchone()
    return (r[0] or "")[:10]


def main():
    syms = [l.strip() for l in open(SYMS_FILE) if l.strip()]
    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    before = {s: last_bar(con, s) for s in syms}
    con.close()

    current = max(before.values())
    stale = sorted([s for s, d in before.items() if d != current])
    print("universe %d | current bar %s | stale %d" % (len(syms), current, len(stale)))
    if not stale:
        print("nothing to do")
        return

    tok = json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))
    kite = KiteConnect(api_key=os.environ["KITE_API_KEY"])
    kite.set_access_token(tok["access_token"])
    print("kite user:", kite.profile().get("user_id"))

    dm = get_data_manager(kite=kite)
    to_date = datetime.now()

    # TATAMOTORS needs a deeper window than the rest
    groups = [([s for s in stale if s != "TATAMOTORS"], FROM),
              ([s for s in stale if s == "TATAMOTORS"], TATAMOTORS_FROM)]

    tot_ok = tot_fail = 0
    all_errors = []
    for group, frm in groups:
        if not group:
            continue
        print("\ndownloading %d symbols from %s" % (len(group), frm.date()))

        def cb(i, n, sym, status):
            if status != "downloading":
                print("  [%3d/%3d] %-14s %s" % (i, n, sym, status), flush=True)

        ok, fail, errors = dm.download_data(group, timeframe="day",
                                            from_date=frm, to_date=to_date,
                                            progress_callback=cb)
        tot_ok += ok
        tot_fail += fail
        all_errors += errors

    print("\ndownloaded ok=%d fail=%d" % (tot_ok, tot_fail))
    for e in all_errors[:20]:
        print("  ERR", e)

    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    after = {s: last_bar(con, s) for s in syms}
    con.close()
    fixed = [s for s in stale if after[s] != before[s]]
    still = [s for s in syms if after[s] != max(after.values())]
    print("\nadvanced: %d symbols" % len(fixed))
    print("still stale: %d %s" % (len(still), still[:15]))
    print("universe now current to:", max(after.values()))


if __name__ == "__main__":
    main()
