"""
Download 2 years of 5-minute data for NIFTY50 and BANKNIFTY indices.
Uses Kite API with chunked fetching (7 days per request, 2000 candle limit).
Saves to centralized market_data.db.

Run after refreshing Kite access token:
    python download_index_5min.py
"""

import sqlite3
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────
DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'
TOKEN_PATH = Path(__file__).parent / 'backtest_data' / 'access_token.json'

# NSE index instrument tokens (fixed, don't change)
INDICES = {
    'NIFTY50': 256265,       # NSE:NIFTY 50
    'BANKNIFTY': 260105,     # NSE:NIFTY BANK
}

TARGET_START = datetime(2024, 3, 1)
TARGET_END = datetime(2026, 3, 27)
CHUNK_DAYS = 7       # 5-min: max 2000 candles = ~7 trading days
RATE_LIMIT = 0.35    # seconds between API calls


def get_kite():
    """Initialize Kite Connect with saved access token."""
    from kiteconnect import KiteConnect
    from config import KITE_API_KEY

    with open(TOKEN_PATH) as f:
        token_data = json.load(f)

    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(token_data['access_token'])
    return kite


def get_existing_range(symbol: str) -> tuple:
    """Get existing 5-min data date range from DB."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute(
        "SELECT MIN(date), MAX(date), COUNT(*) FROM market_data_unified "
        "WHERE symbol = ? AND timeframe = '5minute'",
        (symbol,)
    )
    row = cur.fetchone()
    conn.close()

    if row[0] is None:
        return None, None, 0
    return row[0], row[1], row[2]


def save_chunk(symbol: str, candles: list):
    """Save candles to DB, skip duplicates."""
    if not candles:
        return 0

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    inserted = 0

    for c in candles:
        dt = c['date']
        if hasattr(dt, 'strftime'):
            dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            dt_str = str(dt)

        try:
            cur.execute(
                "INSERT INTO market_data_unified (symbol, timeframe, date, open, high, low, close, volume) "
                "VALUES (?, '5minute', ?, ?, ?, ?, ?, ?)",
                (symbol, dt_str, c['open'], c['high'], c['low'], c['close'], c['volume'])
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass  # duplicate

    conn.commit()
    conn.close()
    return inserted


def download_index(kite, symbol: str, token: int):
    """Download 5-min data for an index, filling gaps."""
    min_date, max_date, count = get_existing_range(symbol)
    logger.info(f"\n{'='*60}")
    logger.info(f"{symbol} (token {token})")
    logger.info(f"  Existing: {count} rows, {min_date} to {max_date}")

    # Determine what to download
    # Gap 1: TARGET_START to existing min_date (backfill)
    # Gap 2: existing max_date to TARGET_END (forward fill, if needed)
    gaps = []

    if count == 0:
        gaps.append((TARGET_START, TARGET_END))
    else:
        existing_min = datetime.strptime(min_date[:10], '%Y-%m-%d')
        existing_max = datetime.strptime(max_date[:10], '%Y-%m-%d')

        if existing_min > TARGET_START:
            gaps.append((TARGET_START, existing_min - timedelta(days=1)))
        if existing_max < TARGET_END:
            gaps.append((existing_max + timedelta(days=1), TARGET_END))

    if not gaps:
        logger.info(f"  Already complete — no gaps to fill")
        return

    total_days = sum((g[1] - g[0]).days for g in gaps)
    total_chunks = sum(((g[1] - g[0]).days // CHUNK_DAYS) + 1 for g in gaps)
    logger.info(f"  Gaps to fill: {len(gaps)}, ~{total_days} days, ~{total_chunks} API calls")

    total_inserted = 0
    chunk_num = 0

    for gap_start, gap_end in gaps:
        logger.info(f"  Filling: {gap_start.date()} to {gap_end.date()}")
        current = gap_start

        while current < gap_end:
            chunk_end = min(current + timedelta(days=CHUNK_DAYS), gap_end)
            chunk_num += 1

            try:
                data = kite.historical_data(
                    instrument_token=token,
                    from_date=current,
                    to_date=chunk_end,
                    interval='5minute'
                )

                inserted = save_chunk(symbol, data)
                total_inserted += inserted
                logger.info(
                    f"    [{chunk_num}/{total_chunks}] {current.date()} → {chunk_end.date()} | "
                    f"{len(data)} fetched, {inserted} new"
                )

            except Exception as e:
                logger.error(f"    [{chunk_num}/{total_chunks}] {current.date()} → {chunk_end.date()} ERROR: {e}")

            time.sleep(RATE_LIMIT)
            current = chunk_end + timedelta(days=1)

    logger.info(f"  Done: {total_inserted} new rows inserted for {symbol}")


def main():
    logger.info("Connecting to Kite...")
    kite = get_kite()

    # Quick test
    try:
        data = kite.historical_data(256265, datetime(2026, 3, 18), datetime(2026, 3, 18), '5minute')
        logger.info(f"Token OK — test fetch got {len(data)} candles")
    except Exception as e:
        logger.error(f"Token FAILED: {e}")
        logger.error("Please refresh your Kite access token first!")
        return

    for symbol, token in INDICES.items():
        download_index(kite, symbol, token)

    # Final stats
    logger.info(f"\n{'='*60}")
    logger.info("FINAL DATA STATUS:")
    for symbol in INDICES:
        min_d, max_d, count = get_existing_range(symbol)
        logger.info(f"  {symbol}: {count} rows, {min_d} to {max_d}")


if __name__ == '__main__':
    main()
