"""
Download NIFTY 50 and BANK NIFTY 5-minute data for 36 months.
Higher timeframes (15m, 30m, 60m) will be resampled from 5-min data.
Daily data also downloaded directly for accuracy.
"""

import json
import sqlite3
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from kiteconnect import KiteConnect

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'
TOKEN_PATH = Path(__file__).parent / 'backtest_data' / 'access_token.json'

# Index instrument tokens
INDICES = {
    'NIFTY50': {'token': 256265, 'name': 'NIFTY 50'},
    'BANKNIFTY': {'token': 260105, 'name': 'NIFTY BANK'},
}

START_DATE = datetime(2023, 3, 19)
END_DATE = datetime(2026, 3, 19)

# Chunk sizes (days per API call) for each timeframe
CHUNK_SIZES = {
    '5minute': 7,
    'day': 1800,
}


def get_kite():
    with open(TOKEN_PATH) as f:
        data = json.load(f)
    kite = KiteConnect(api_key='kitefront')
    kite.set_access_token(data['access_token'])
    return kite


def fetch_data(kite, token, timeframe, from_date, to_date):
    """Fetch data in chunks respecting API limits."""
    chunk_days = CHUNK_SIZES.get(timeframe, 7)
    all_data = []
    current = from_date

    while current < to_date:
        chunk_end = min(current + timedelta(days=chunk_days), to_date)
        try:
            data = kite.historical_data(
                instrument_token=token,
                from_date=current,
                to_date=chunk_end,
                interval=timeframe
            )
            if data:
                all_data.extend(data)
        except Exception as e:
            logger.error(f"  Error {current.date()}-{chunk_end.date()}: {e}")

        time.sleep(0.35)  # Rate limit
        current = chunk_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates(subset=['date'])
    return df


def store_data(symbol, timeframe, df):
    """Store data in unified database."""
    if df.empty:
        return 0

    conn = sqlite3.connect(DB_PATH)

    # Get existing dates
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT date FROM market_data_unified WHERE symbol=? AND timeframe=?",
        (symbol, timeframe)
    )
    existing = {r[0] for r in cur.fetchall()}

    # Prepare insert data
    records = []
    for _, row in df.iterrows():
        if timeframe == 'day':
            date_str = row['date'].strftime('%Y-%m-%d')
        else:
            date_str = row['date'].strftime('%Y-%m-%d %H:%M:%S')

        if date_str in existing:
            continue

        records.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'date': date_str,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row.get('volume', 0),
        })

    if records:
        pd.DataFrame(records).to_sql(
            'market_data_unified', conn, if_exists='append', index=False
        )
        logger.info(f"  Inserted {len(records)} new candles")

    conn.commit()
    conn.close()
    return len(records)


def main():
    kite = get_kite()
    logger.info(f"Downloading {START_DATE.date()} to {END_DATE.date()}")

    for symbol, info in INDICES.items():
        for timeframe in ['5minute', 'day']:
            logger.info(f"\n{'='*50}")
            logger.info(f"Downloading {symbol} {timeframe}...")

            df = fetch_data(kite, info['token'], timeframe, START_DATE, END_DATE)

            if df.empty:
                logger.error(f"  No data fetched!")
                continue

            logger.info(f"  Fetched {len(df)} candles: {df['date'].min()} to {df['date'].max()}")
            inserted = store_data(symbol, timeframe, df)
            logger.info(f"  Done: {inserted} new records stored")

    # Verify
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol, timeframe, MIN(date), MAX(date), COUNT(*)
        FROM market_data_unified
        WHERE symbol IN ('NIFTY50', 'BANKNIFTY')
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
    """)
    print("\n=== VERIFICATION ===")
    for r in cur.fetchall():
        print(f"{r[0]:12s} | {r[1]:10s} | {r[2]} to {r[3]} | {r[4]:,} rows")
    conn.close()


if __name__ == '__main__':
    main()
