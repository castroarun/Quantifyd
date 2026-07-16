#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SENSEX 1-minute downloader -> market_data.db (symbol='SENSEX', timeframe='minute').

WHY 1-min: the expiry-day work (research/82) is about intraday behaviour on SENSEX's Thursday
expiry. 5-min bars blur exactly the moves the move-stop reacts to; 1-min is the right resolution.

Runs on the VPS only (the canonical host for all Kite downloads, per project rule).

Usage:
  dl_sensex_1min.py                -> TODAY only (the daily cron; ~375 rows, trivial)
  dl_sensex_1min.py --backfill     -> 2021-01-01 -> today, 55-day chunks, RESUMABLE
                                      (skips any day already present; safe to re-run)
  dl_sensex_1min.py --days N       -> last N days (gap repair)

Kite caps the 'minute' interval at ~60 days per request, so the backfill chunks at 55.
Idempotent: every row is checked before insert, so nothing is ever duplicated.
"""
import sqlite3
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, "/home/arun/quantifyd")
from services.kite_service import get_kite

DB = "/home/arun/quantifyd/backtest_data/market_data.db"
TOKEN = 265           # SENSEX index
SYMBOL = "SENSEX"
TF = "minute"         # matches Kite's interval name, like the existing 5minute/30minute/day
CHUNK_DAYS = 55       # Kite's 'minute' limit is ~60d/request
FIRST = datetime(2021, 1, 1)


def upsert(cur, rows):
    n = 0
    for r in rows:
        d = r["date"]
        ds = d.strftime("%Y-%m-%d %H:%M:%S") if hasattr(d, "strftime") else str(d)
        if cur.execute("SELECT 1 FROM market_data_unified WHERE symbol=? AND timeframe=? AND date=?",
                       (SYMBOL, TF, ds)).fetchone():
            continue
        cur.execute("INSERT INTO market_data_unified (symbol,timeframe,date,open,high,low,close,volume,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
                    (SYMBOL, TF, ds, r["open"], r["high"], r["low"], r["close"], r.get("volume", 0) or 0))
        n += 1
    return n


def main():
    k = get_kite()
    con = sqlite3.connect(DB, timeout=120)   # generous: the DB is often under backfill load
    cur = con.cursor()

    if "--backfill" in sys.argv:
        start, end = FIRST, datetime.now()
    elif "--days" in sys.argv:
        n = int(sys.argv[sys.argv.index("--days") + 1])
        start, end = datetime.now() - timedelta(days=n), datetime.now()
    else:
        # daily cron: just today
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end = datetime.now()

    total = 0
    cur_dt = start
    while cur_dt <= end:
        to = min(cur_dt + timedelta(days=CHUNK_DAYS), end)
        try:
            data = k.historical_data(TOKEN, cur_dt, to, TF)
            got = upsert(cur, data or [])
            con.commit()
            total += got
            if got or "--backfill" in sys.argv:
                print("  %s..%s  +%d rows (total %d)" % (cur_dt.date(), to.date(), got, total))
                sys.stdout.flush()
        except Exception as e:
            print("  %s..%s ERR %s" % (cur_dt.date(), to.date(), str(e)[:70]))
            sys.stdout.flush()
        cur_dt = to + timedelta(days=1)
        time.sleep(0.4)          # Kite rate limit (3 req/s)

    r = cur.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM market_data_unified "
                    "WHERE symbol=? AND timeframe=?", (SYMBOL, TF)).fetchone()
    print("SENSEX %s: %d rows total | %s .. %s  (+%d this run)" % (TF, r[0], r[1], r[2], total))
    con.close()


if __name__ == "__main__":
    main()
