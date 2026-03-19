"""
Backfill Historical Data from 2005-01-01
=========================================

Downloads daily OHLCV data from Kite API for all Nifty 500 stocks,
going back to 2005-01-01. Only fetches data for date ranges not already
in the database (smart backfill - won't re-download existing data).

Usage:
    python backfill_historical_data.py

Requirements:
    - Valid Kite API access token (login via the web app first)
    - KITE_API_KEY and KITE_API_SECRET in .env file
"""

import sqlite3
import time
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from kiteconnect import KiteConnect

# Configuration
DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'
BACKFILL_FROM = datetime(2005, 1, 1)
BACKFILL_TO = datetime.now()
CHUNK_SIZE_DAYS = 1800  # Kite allows ~2000 candles per request for daily
RATE_LIMIT_SLEEP = 0.35  # 3 req/sec


def get_kite():
    """Initialize Kite with stored access token."""
    api_key = os.getenv("KITE_API_KEY", "")
    token_file = Path(__file__).parent / 'backtest_data' / 'access_token.json'

    if not api_key:
        print("ERROR: KITE_API_KEY not found in .env")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)

    if token_file.exists():
        import json
        data = json.loads(token_file.read_text())
        access_token = data.get("access_token", "")
        if access_token:
            kite.set_access_token(access_token)
            print(f"Kite API initialized with stored token")
            return kite

    print("ERROR: No access token found. Login via the web app first.")
    sys.exit(1)


def get_instrument_tokens(kite):
    """Get instrument tokens from cache or Kite API."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if instrument cache exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='instrument_cache'
    """)
    has_cache = cursor.fetchone() is not None

    tokens = {}
    if has_cache:
        cursor.execute("""
            SELECT tradingsymbol, instrument_token
            FROM instrument_cache
            WHERE exchange = 'NSE' AND instrument_type = 'EQ'
        """)
        tokens = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()

    if len(tokens) < 100:
        print("Instrument cache empty or stale, fetching from Kite API...")
        instruments = kite.instruments("NSE")
        tokens = {}
        for inst in instruments:
            if inst['instrument_type'] == 'EQ':
                tokens[inst['tradingsymbol']] = inst['instrument_token']
        print(f"  Got {len(tokens)} instrument tokens from Kite")

        # Cache them
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS instrument_cache (
                tradingsymbol TEXT PRIMARY KEY,
                instrument_token INTEGER NOT NULL,
                instrument_type TEXT,
                exchange TEXT,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for sym, tok in tokens.items():
            conn.execute("""
                INSERT OR REPLACE INTO instrument_cache
                (tradingsymbol, instrument_token, instrument_type, exchange)
                VALUES (?, ?, 'EQ', 'NSE')
            """, (sym, tok))
        conn.commit()
        conn.close()

    return tokens


def get_symbols_needing_backfill():
    """Get symbols that need historical data backfill."""
    conn = sqlite3.connect(DB_PATH)

    # Get all symbols and their earliest date
    df = conn.execute("""
        SELECT symbol, MIN(date) as first_date, COUNT(*) as rows
        FROM market_data_unified
        WHERE timeframe = 'day'
        GROUP BY symbol
        ORDER BY symbol
    """).fetchall()

    conn.close()

    needs_backfill = []
    already_good = 0

    for sym, first_date, rows in df:
        first_dt = datetime.strptime(first_date[:10], '%Y-%m-%d')
        # Need backfill if first date is after our target
        if first_dt > BACKFILL_FROM + timedelta(days=365):  # 1 year grace for IPOs
            needs_backfill.append({
                'symbol': sym,
                'first_date': first_date,
                'rows': rows,
                'backfill_to': first_date,  # Download up to their current earliest date
            })
        else:
            already_good += 1

    return needs_backfill, already_good


def fetch_and_store(kite, symbol, instrument_token, from_date, to_date):
    """Fetch data from Kite and store in DB. Returns number of new rows inserted."""
    conn = sqlite3.connect(DB_PATH)

    # Get existing dates for this symbol to avoid duplicates
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT date FROM market_data_unified
        WHERE symbol = ? AND timeframe = 'day'
    """, (symbol,))
    existing_dates = {row[0] for row in cursor.fetchall()}

    total_new = 0
    current = from_date

    while current < to_date:
        chunk_end = min(current + timedelta(days=CHUNK_SIZE_DAYS), to_date)

        try:
            data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=current,
                to_date=chunk_end,
                interval='day'
            )

            if data:
                new_rows = []
                for candle in data:
                    date_str = candle['date'].strftime('%Y-%m-%d') if hasattr(candle['date'], 'strftime') else str(candle['date'])[:10]
                    if date_str not in existing_dates:
                        new_rows.append((
                            symbol, 'day', date_str,
                            candle['open'], candle['high'],
                            candle['low'], candle['close'],
                            candle.get('volume', 0)
                        ))
                        existing_dates.add(date_str)

                if new_rows:
                    cursor.executemany("""
                        INSERT OR IGNORE INTO market_data_unified
                        (symbol, timeframe, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, new_rows)
                    total_new += len(new_rows)

        except Exception as e:
            error_str = str(e)
            if 'TokenException' in error_str or 'InvalidToken' in error_str:
                print(f"\n  TOKEN EXPIRED! Please re-login and restart.")
                conn.commit()
                conn.close()
                sys.exit(1)
            # Other errors: skip this chunk, continue
            pass

        time.sleep(RATE_LIMIT_SLEEP)
        current = chunk_end + timedelta(days=1)

    conn.commit()
    conn.close()
    return total_new


def main():
    print("=" * 80)
    print("HISTORICAL DATA BACKFILL")
    print(f"Target: {BACKFILL_FROM.strftime('%Y-%m-%d')} to {BACKFILL_TO.strftime('%Y-%m-%d')}")
    print("=" * 80)

    # Step 1: Connect to Kite
    print("\n[1/4] Connecting to Kite API...")
    kite = get_kite()

    # Verify connection with a simple API call
    try:
        profile = kite.profile()
        print(f"  Logged in as: {profile.get('user_name', 'Unknown')}")
    except Exception as e:
        print(f"  ERROR: Kite connection failed: {e}")
        print("  Please login via the web app and try again.")
        sys.exit(1)

    # Step 2: Get instrument tokens
    print("\n[2/4] Loading instrument tokens...")
    tokens = get_instrument_tokens(kite)
    print(f"  {len(tokens)} instrument tokens available")

    # Step 3: Identify symbols needing backfill
    print("\n[3/4] Identifying symbols needing backfill...")
    needs_backfill, already_good = get_symbols_needing_backfill()
    print(f"  Already have full history: {already_good}")
    print(f"  Need backfill: {len(needs_backfill)}")

    if not needs_backfill:
        print("\nAll symbols already have data from 2005 or earlier. Nothing to do.")
        return

    # Show what we'll download
    print(f"\n  First 10 needing backfill:")
    for item in needs_backfill[:10]:
        print(f"    {item['symbol']:>15s}: current from {item['first_date'][:10]}, {item['rows']} rows")
    if len(needs_backfill) > 10:
        print(f"    ... and {len(needs_backfill) - 10} more")

    # Step 4: Download
    print(f"\n[4/4] Downloading historical data...")
    print(f"  Estimated API calls: ~{len(needs_backfill) * 5}")
    print(f"  Estimated time: ~{len(needs_backfill) * 5 * 0.35 / 60:.1f} minutes")
    print()

    total_new_rows = 0
    successful = 0
    failed = 0
    failed_symbols = []
    start_time = time.time()

    for idx, item in enumerate(needs_backfill, 1):
        sym = item['symbol']
        instrument_token = tokens.get(sym)

        if not instrument_token:
            failed += 1
            failed_symbols.append(f"{sym}: no instrument token")
            continue

        # Backfill from 2005 to their current earliest date
        to_date_str = item['first_date'][:10]
        to_date = datetime.strptime(to_date_str, '%Y-%m-%d') - timedelta(days=1)

        if to_date <= BACKFILL_FROM:
            successful += 1
            continue

        try:
            new_rows = fetch_and_store(kite, sym, instrument_token, BACKFILL_FROM, to_date)
            total_new_rows += new_rows
            successful += 1

            elapsed = time.time() - start_time
            rate = idx / elapsed * 60 if elapsed > 0 else 0
            remaining = (len(needs_backfill) - idx) / rate if rate > 0 else 0

            status = f"+{new_rows} rows" if new_rows > 0 else "no new data (IPO after 2005?)"
            print(f"  [{idx}/{len(needs_backfill)}] {sym:>15s}: {status} "
                  f"(~{remaining:.0f}min left)")

        except Exception as e:
            failed += 1
            failed_symbols.append(f"{sym}: {str(e)[:50]}")
            print(f"  [{idx}/{len(needs_backfill)}] {sym:>15s}: FAILED - {str(e)[:60]}")

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print("BACKFILL COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  New rows inserted: {total_new_rows:,}")

    # Check new DB size
    db_size = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"  Database size: {db_size:.1f} MB")

    # New date coverage
    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("""
        SELECT COUNT(DISTINCT symbol) as syms,
               COUNT(*) as rows,
               MIN(date) as min_date
        FROM market_data_unified WHERE timeframe='day'
    """).fetchone()
    print(f"  Total symbols: {r[0]}")
    print(f"  Total daily rows: {r[1]:,}")
    print(f"  Earliest date: {r[2]}")
    conn.close()

    if failed_symbols:
        print(f"\n  Failed symbols ({len(failed_symbols)}):")
        for fs in failed_symbols[:20]:
            print(f"    - {fs}")
        if len(failed_symbols) > 20:
            print(f"    ... and {len(failed_symbols) - 20} more")


if __name__ == '__main__':
    main()
