#!/usr/bin/env python3
"""
Download 5-minute and 30-minute F&O stock data for CPR intraday backtesting.
Crash-proof: tracks progress per symbol+timeframe, resumes from where it stopped.

Usage: python download_5min_fno.py

Requirements:
    - Valid Kite API access token (login via the web app first)
    - KITE_API_KEY in .env file
    - instrument_cache table populated in market_data.db
"""

import sqlite3
import time
import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from kiteconnect import KiteConnect

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'
TOKEN_FILE = Path(__file__).parent / 'backtest_data' / 'access_token.json'
PROGRESS_FILE = Path(__file__).parent / 'backtest_data' / 'download_5min_progress.json'

START_DATE = datetime(2024, 3, 16)
END_DATE = datetime(2026, 3, 16)
RATE_LIMIT = 0.35  # seconds between API calls (3 req/sec limit)
MAX_RETRIES = 3

# Chunk sizes: how many calendar days per Kite API request
# 5-min: 75 candles/day * 7 days = 525 candles (well within 2000 limit)
# 30-min: 12 candles/day * 40 days = 480 candles (well within 2000 limit)
CHUNK_SIZES = {
    '5minute': 7,
}

TIMEFRAMES = ['5minute']  # 30-min can be aggregated from 5-min candles


# ---------------------------------------------------------------------------
# Kite API setup
# ---------------------------------------------------------------------------
def get_kite():
    """Initialize KiteConnect with stored access token."""
    api_key = os.getenv("KITE_API_KEY", "")
    if not api_key:
        print("ERROR: KITE_API_KEY not found in .env file")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)

    if not TOKEN_FILE.exists():
        print(f"ERROR: No access token file found at {TOKEN_FILE}")
        print("       Login via the web app first (http://127.0.0.1:5000/zerodha/login)")
        sys.exit(1)

    data = json.loads(TOKEN_FILE.read_text())
    access_token = data.get("access_token", "")
    if not access_token:
        print("ERROR: access_token is empty in token file")
        sys.exit(1)

    kite.set_access_token(access_token)

    # Verify the token works
    try:
        profile = kite.profile()
        print(f"Kite API connected: {profile.get('user_name', 'OK')}")
    except Exception as e:
        print(f"ERROR: Kite API connection failed: {e}")
        print("       Please re-login via the web app and try again.")
        sys.exit(1)

    return kite


# ---------------------------------------------------------------------------
# Instrument tokens
# ---------------------------------------------------------------------------
def get_instrument_tokens(kite, conn):
    """
    Load instrument tokens from the instrument_cache table.
    If cache is empty or stale (> 30 days), refresh from Kite API.
    Returns dict: {symbol: instrument_token}
    """
    cursor = conn.cursor()

    # Check cache age
    cursor.execute("""
        SELECT tradingsymbol, instrument_token, cached_at
        FROM instrument_cache
        WHERE exchange = 'NSE' AND instrument_type = 'EQ'
    """)
    rows = cursor.fetchall()

    tokens = {}
    cache_stale = True

    if rows:
        tokens = {row[0]: row[1] for row in rows}
        # Check if cache is recent (within 30 days)
        try:
            latest_cache = max(row[2] for row in rows if row[2])
            cache_dt = datetime.fromisoformat(latest_cache.replace('T', ' ').split('.')[0])
            cache_age_days = (datetime.now() - cache_dt).days
            cache_stale = cache_age_days > 30
            if not cache_stale:
                print(f"Instrument cache: {len(tokens)} tokens (cached {cache_age_days}d ago)")
        except Exception:
            cache_stale = True

    if cache_stale or len(tokens) < 100:
        print("Refreshing instrument cache from Kite API...")
        try:
            instruments = kite.instruments("NSE")
            tokens = {}
            for inst in instruments:
                if inst['instrument_type'] == 'EQ':
                    tokens[inst['tradingsymbol']] = inst['instrument_token']
            print(f"  Fetched {len(tokens)} NSE EQ instrument tokens")

            # Update cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS instrument_cache (
                    tradingsymbol TEXT PRIMARY KEY,
                    instrument_token INTEGER NOT NULL,
                    instrument_type TEXT,
                    exchange TEXT,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            now_str = datetime.now().isoformat()
            for sym, tok in tokens.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO instrument_cache
                    (tradingsymbol, instrument_token, instrument_type, exchange, cached_at)
                    VALUES (?, ?, 'EQ', 'NSE', ?)
                """, (sym, tok, now_str))
            conn.commit()
        except Exception as e:
            print(f"  WARNING: Could not refresh cache: {e}")
            if not tokens:
                print("  ERROR: No instrument tokens available. Cannot continue.")
                sys.exit(1)

    return tokens


# ---------------------------------------------------------------------------
# Progress tracking (crash-proof)
# ---------------------------------------------------------------------------
def load_progress():
    """Load progress state from disk. Returns dict of completed symbol+timeframe combos."""
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text())
            return data
        except Exception:
            return {}
    return {}


def save_progress(progress):
    """Save progress state to disk after each symbol completes."""
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, default=str))


def is_complete(progress, symbol, timeframe):
    """Check if a symbol+timeframe is already fully downloaded."""
    key = f"{symbol}|{timeframe}"
    return progress.get(key, {}).get('complete', False)


def mark_complete(progress, symbol, timeframe, candles, duration):
    """Mark a symbol+timeframe as fully downloaded."""
    key = f"{symbol}|{timeframe}"
    progress[key] = {
        'complete': True,
        'candles': candles,
        'duration_sec': round(duration, 1),
        'finished_at': datetime.now().isoformat(),
    }
    save_progress(progress)


# ---------------------------------------------------------------------------
# Existing data coverage check
# ---------------------------------------------------------------------------
def get_existing_coverage(conn, symbol, timeframe):
    """
    Check what date range we already have for this symbol+timeframe.
    Returns (min_date, max_date, count) or (None, None, 0) if no data.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MIN(date), MAX(date), COUNT(*)
        FROM market_data_unified
        WHERE symbol = ? AND timeframe = ?
    """, (symbol, timeframe))
    row = cursor.fetchone()

    if row and row[0]:
        return row[0], row[1], row[2]
    return None, None, 0


def get_existing_dates(conn, symbol, timeframe):
    """Get set of all existing datetime strings for duplicate filtering."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT date FROM market_data_unified
        WHERE symbol = ? AND timeframe = ?
    """, (symbol, timeframe))
    return {row[0] for row in cursor.fetchall()}


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------
def download_chunk_with_retry(kite, instrument_token, start, end, timeframe):
    """
    Download a single chunk from Kite API with retry logic.
    Returns list of candle dicts, or None on permanent failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=start,
                to_date=end,
                interval=timeframe,
            )
            return data if data else []

        except Exception as e:
            error_str = str(e)

            # Token expired - unrecoverable
            if 'TokenException' in error_str or 'InvalidToken' in error_str:
                print(f"\n  TOKEN EXPIRED! Please re-login and restart the script.")
                sys.exit(1)

            # Rate limit hit
            if 'TooManyRequests' in error_str or '429' in error_str:
                wait = 2 ** attempt  # 2s, 4s, 8s
                print(f" [rate-limited, waiting {wait}s]", end='', flush=True)
                time.sleep(wait)
                continue

            # Network / transient errors
            if attempt < MAX_RETRIES:
                wait = 1.5 ** attempt  # 1.5s, 2.25s, 3.375s
                print(f" [retry {attempt}/{MAX_RETRIES}: {error_str[:60]}]", end='', flush=True)
                time.sleep(wait)
                continue
            else:
                print(f" [FAILED after {MAX_RETRIES} retries: {error_str[:80]}]", end='', flush=True)
                return None

    return None


def download_symbol_timeframe(kite, conn, symbol, instrument_token, timeframe, start_date, end_date):
    """
    Download all missing data for one symbol+timeframe.
    Inserts into DB after each chunk. Returns total new candles inserted.
    """
    chunk_days = CHUNK_SIZES[timeframe]

    # Load existing dates for dedup
    existing_dates = get_existing_dates(conn, symbol, timeframe)

    # Determine effective start: if we have some data, start from day after max date
    min_dt, max_dt, existing_count = get_existing_coverage(conn, symbol, timeframe)

    effective_start = start_date
    if max_dt:
        # Parse max date - could be "2025-10-27 15:25:00" format
        try:
            max_datetime = datetime.strptime(max_dt[:10], '%Y-%m-%d')
            # If existing data covers beyond our start, begin from next day
            if max_datetime >= start_date:
                effective_start = max_datetime + timedelta(days=1)
        except Exception:
            pass

    if effective_start >= end_date:
        return 0  # Already fully covered

    total_new = 0
    current = effective_start
    cursor = conn.cursor()
    chunk_count = 0

    while current < end_date:
        chunk_end = min(current + timedelta(days=chunk_days), end_date)
        chunk_count += 1

        data = download_chunk_with_retry(kite, instrument_token, current, chunk_end, timeframe)

        if data is not None and len(data) > 0:
            # Filter duplicates and insert
            new_rows = []
            for candle in data:
                # Format date string
                if hasattr(candle['date'], 'strftime'):
                    date_str = candle['date'].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    date_str = str(candle['date'])[:19]

                if date_str not in existing_dates:
                    new_rows.append((
                        symbol, timeframe, date_str,
                        candle['open'], candle['high'],
                        candle['low'], candle['close'],
                        candle.get('volume', 0),
                    ))
                    existing_dates.add(date_str)

            if new_rows:
                cursor.executemany("""
                    INSERT OR IGNORE INTO market_data_unified
                    (symbol, timeframe, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, new_rows)
                conn.commit()
                total_new += len(new_rows)

        # Rate limit
        time.sleep(RATE_LIMIT)
        current = chunk_end + timedelta(days=1)

    return total_new


# ---------------------------------------------------------------------------
# ETA / progress formatting
# ---------------------------------------------------------------------------
def format_duration(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


def estimate_chunks(start_date, end_date, chunk_days):
    """Estimate how many API chunks a symbol will need."""
    total_days = (end_date - start_date).days
    return max(1, (total_days + chunk_days - 1) // chunk_days)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    load_dotenv()

    print("=" * 70)
    print("F&O INTRADAY DATA DOWNLOADER")
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Rate limit: {RATE_LIMIT}s between API calls")
    print("=" * 70)

    # Step 1: Connect to Kite
    print("\n[Setup] Connecting to Kite API...")
    kite = get_kite()

    # Step 2: Open DB connection
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent write performance

    # Step 3: Get instrument tokens
    print("[Setup] Loading instrument tokens...")
    tokens = get_instrument_tokens(kite, conn)

    # Step 4: Get F&O symbols
    from services.data_manager import FNO_LOT_SIZES
    symbols = sorted(FNO_LOT_SIZES.keys())
    total_symbols = len(symbols)
    print(f"[Setup] F&O universe: {total_symbols} stocks")

    # Step 5: Check which symbols have tokens
    missing_tokens = [s for s in symbols if s not in tokens]
    if missing_tokens:
        print(f"\n  WARNING: {len(missing_tokens)} symbols missing instrument tokens:")
        for s in missing_tokens:
            print(f"    - {s}")
        symbols = [s for s in symbols if s in tokens]
        total_symbols = len(symbols)
        print(f"  Proceeding with {total_symbols} symbols that have tokens.\n")

    # Step 6: Load crash-proof progress
    progress = load_progress()
    already_done = sum(1 for s in symbols for tf in TIMEFRAMES if is_complete(progress, s, tf))
    total_jobs = total_symbols * len(TIMEFRAMES)
    if already_done > 0:
        print(f"[Resume] {already_done}/{total_jobs} symbol+timeframe combos already complete")
        print(f"         (Delete {PROGRESS_FILE.name} to force re-download)\n")

    # Step 7: Download loop
    overall_start = time.time()
    completed_count = already_done
    skipped_full_coverage = 0
    total_candles_downloaded = 0
    errors = []

    for timeframe in TIMEFRAMES:
        chunk_days = CHUNK_SIZES[timeframe]
        est_chunks_per_symbol = estimate_chunks(START_DATE, END_DATE, chunk_days)
        est_seconds_per_symbol = est_chunks_per_symbol * RATE_LIMIT + 2  # overhead

        print(f"\n{'=' * 70}")
        print(f"TIMEFRAME: {timeframe}")
        print(f"  Chunk size: {chunk_days} days | ~{est_chunks_per_symbol} API calls/symbol")
        print(f"  Estimated time per symbol: ~{format_duration(est_seconds_per_symbol)}")
        print(f"{'=' * 70}\n")

        tf_start = time.time()
        tf_symbols_done = 0

        for i, symbol in enumerate(symbols, 1):
            # Skip if already complete (crash recovery)
            if is_complete(progress, symbol, timeframe):
                continue

            # Check existing coverage in DB
            min_dt, max_dt, existing_count = get_existing_coverage(conn, symbol, timeframe)

            instrument_token = tokens[symbol]
            sym_start = time.time()

            # Print progress header
            coverage_note = ""
            if existing_count > 0:
                coverage_note = f" (existing: {existing_count} candles to {max_dt[:10]})"
            print(f"[{i}/{total_symbols}] {symbol:15s} {timeframe}: ", end='', flush=True)
            if coverage_note:
                print(coverage_note, end='', flush=True)

            try:
                new_candles = download_symbol_timeframe(
                    kite, conn, symbol, instrument_token,
                    timeframe, START_DATE, END_DATE
                )

                sym_elapsed = time.time() - sym_start
                total_candles_downloaded += new_candles

                if new_candles == 0 and existing_count > 0:
                    print(f" SKIP (already covered) [{sym_elapsed:.1f}s]")
                    skipped_full_coverage += 1
                elif new_candles == 0:
                    print(f" 0 candles (no data?) [{sym_elapsed:.1f}s]")
                else:
                    print(f" +{new_candles} candles [{sym_elapsed:.1f}s]")

                # Mark complete and save progress
                mark_complete(progress, symbol, timeframe, new_candles, sym_elapsed)
                completed_count += 1
                tf_symbols_done += 1

            except SystemExit:
                raise  # Token expired - propagate
            except Exception as e:
                sym_elapsed = time.time() - sym_start
                error_msg = f"{symbol} {timeframe}: {str(e)[:100]}"
                errors.append(error_msg)
                print(f" ERROR: {str(e)[:80]} [{sym_elapsed:.1f}s]")
                # Don't mark as complete so it retries on next run

            # ETA calculation
            if tf_symbols_done > 0:
                tf_elapsed = time.time() - tf_start
                avg_per_symbol = tf_elapsed / tf_symbols_done
                remaining_in_tf = sum(
                    1 for s in symbols[i:]
                    if not is_complete(progress, s, timeframe)
                )
                if remaining_in_tf > 0:
                    eta_seconds = remaining_in_tf * avg_per_symbol
                    print(f"         ETA for {timeframe}: ~{format_duration(eta_seconds)} "
                          f"({remaining_in_tf} symbols remaining)")

        tf_elapsed = time.time() - tf_start
        print(f"\n  {timeframe} complete: {tf_symbols_done} symbols in {format_duration(tf_elapsed)}")

    # Step 8: Final summary
    overall_elapsed = time.time() - overall_start

    print(f"\n{'=' * 70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time:           {format_duration(overall_elapsed)}")
    print(f"  Symbols processed:    {total_symbols}")
    print(f"  Jobs completed:       {completed_count}/{total_jobs}")
    print(f"  Skipped (full data):  {skipped_full_coverage}")
    print(f"  New candles inserted: {total_candles_downloaded:,}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for err in errors:
            print(f"    - {err}")

    # Print DB summary
    print(f"\n  Database coverage after download:")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timeframe, COUNT(DISTINCT symbol) as symbols,
               COUNT(*) as total_candles,
               MIN(date) as min_date, MAX(date) as max_date
        FROM market_data_unified
        WHERE timeframe IN ('5minute', '30minute')
        GROUP BY timeframe
        ORDER BY timeframe
    """)
    for row in cursor.fetchall():
        print(f"    {row[0]:10s}: {row[1]:3d} symbols, {row[2]:>10,} candles "
              f"({row[3][:10]} to {row[4][:10]})")

    conn.close()

    # Clean up progress file if everything succeeded
    if not errors and completed_count == total_jobs:
        print(f"\n  All downloads successful. Progress file retained for reference.")
        print(f"  Delete {PROGRESS_FILE} to force re-download on next run.")

    print("\nDone!")


if __name__ == '__main__':
    main()
