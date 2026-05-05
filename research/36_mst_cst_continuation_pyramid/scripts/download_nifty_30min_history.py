"""Download NIFTY 50 30-min history from 2020-01-01 to today via Kite.

Inserts into market_data_unified (symbol='NIFTY50', timeframe='30minute').
Idempotent — uses INSERT OR IGNORE on (symbol, timeframe, date) composite.
"""
from __future__ import annotations
import sqlite3
import time
from datetime import datetime, date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "backtest_data" / "market_data.db"

NIFTY_TOKEN = 256265
SYMBOL = "NIFTY50"
TIMEFRAME = "30minute"
START_DATE = date(2020, 1, 1)
CHUNK_DAYS = 60                 # safe for 30-min interval
KITE_RATE_LIMIT_S = 0.4          # 0.35s = 3 req/s; we use 0.4 for safety


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from services.kite_service import get_kite
    kite = get_kite()

    end_date = date.today()
    print(f"Downloading {SYMBOL} {TIMEFRAME} from {START_DATE} to {end_date}")

    con = sqlite3.connect(DB)
    cur = con.cursor()

    # Composite unique index on (symbol, timeframe, date) should already exist; verify
    cur.execute("PRAGMA index_list(market_data_unified)")
    indexes = [row for row in cur.fetchall()]
    print(f"Indexes on market_data_unified: {[r[1] for r in indexes]}")

    # Pre-count what we have
    cur.execute(
        "SELECT COUNT(*), MIN(date), MAX(date) FROM market_data_unified "
        "WHERE symbol=? AND timeframe=?", (SYMBOL, TIMEFRAME))
    pre_count, pre_min, pre_max = cur.fetchone()
    print(f"Pre-existing {SYMBOL} {TIMEFRAME}: {pre_count} rows, {pre_min} -> {pre_max}")

    chunk_start = START_DATE
    total_inserted = 0
    api_calls = 0
    errors = 0

    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end_date)
        try:
            data = kite.historical_data(
                instrument_token=NIFTY_TOKEN,
                from_date=chunk_start,
                to_date=chunk_end,
                interval=TIMEFRAME,
            )
            api_calls += 1
            if data:
                rows = [
                    (SYMBOL, TIMEFRAME, d['date'].strftime('%Y-%m-%d %H:%M:%S'),
                     d['open'], d['high'], d['low'], d['close'], d.get('volume', 0))
                    for d in data
                ]
                cur.executemany(
                    "INSERT OR IGNORE INTO market_data_unified "
                    "(symbol, timeframe, date, open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                total_inserted += cur.rowcount
                con.commit()
                print(f"  {chunk_start} -> {chunk_end}: {len(data)} bars fetched, {cur.rowcount} new inserted (cumulative new: {total_inserted})")
            else:
                print(f"  {chunk_start} -> {chunk_end}: no data")
        except Exception as e:
            errors += 1
            print(f"  {chunk_start} -> {chunk_end}: ERROR {type(e).__name__}: {e}")
            if errors >= 3:
                print("Too many errors. Aborting.")
                break

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(KITE_RATE_LIMIT_S)

    # Post-count
    cur.execute(
        "SELECT COUNT(*), MIN(date), MAX(date) FROM market_data_unified "
        "WHERE symbol=? AND timeframe=?", (SYMBOL, TIMEFRAME))
    post_count, post_min, post_max = cur.fetchone()
    con.close()

    print(f"\nDone. API calls: {api_calls}. New rows inserted: {total_inserted}. Errors: {errors}")
    print(f"Final {SYMBOL} {TIMEFRAME}: {post_count} rows, {post_min} -> {post_max}")


if __name__ == "__main__":
    main()
