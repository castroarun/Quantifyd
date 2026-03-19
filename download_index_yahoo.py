"""
Download NIFTY 50 and BANK NIFTY data from Yahoo Finance.
- Daily: 36 months (full period)
- Hourly: ~24 months (max available)
- 5-min: ~60 days (max available)
Store in unified database for backtesting.
"""

import sqlite3
import pandas as pd
import yfinance as yf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'

INDICES = {
    'NIFTY50': '^NSEI',
    'BANKNIFTY': '^NSEBANK',
}

DOWNLOADS = [
    {'interval': '1d',  'start': '2023-03-19', 'end': '2026-03-20', 'timeframe': 'day'},
    {'interval': '1h',  'period': 'max',  'timeframe': '60minute'},
    {'interval': '5m',  'period': '60d',  'timeframe': '5minute'},
]


def store_data(symbol, timeframe, df):
    if df.empty:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get existing dates
    cur.execute(
        "SELECT DISTINCT date FROM market_data_unified WHERE symbol=? AND timeframe=?",
        (symbol, timeframe)
    )
    existing = {r[0] for r in cur.fetchall()}

    records = []
    for idx, row in df.iterrows():
        ts = pd.Timestamp(idx)
        if timeframe == 'day':
            date_str = ts.strftime('%Y-%m-%d')
        else:
            # Convert UTC to IST (+5:30)
            ts_ist = ts.tz_convert('Asia/Kolkata') if ts.tzinfo else ts + pd.Timedelta(hours=5, minutes=30)
            date_str = ts_ist.strftime('%Y-%m-%d %H:%M:%S')

        if date_str in existing:
            continue

        records.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'date': date_str,
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'volume': int(row['Volume']) if pd.notna(row['Volume']) else 0,
        })

    if records:
        pd.DataFrame(records).to_sql(
            'market_data_unified', conn, if_exists='append', index=False
        )

    conn.commit()
    conn.close()
    return len(records)


def main():
    for symbol, ticker in INDICES.items():
        for dl in DOWNLOADS:
            tf = dl['timeframe']
            interval = dl['interval']

            print(f"\n{'='*50}")
            print(f"Downloading {symbol} {tf} (interval={interval})...")

            try:
                if 'period' in dl:
                    df = yf.download(ticker, period=dl['period'], interval=interval,
                                     progress=False, auto_adjust=True)
                else:
                    df = yf.download(ticker, start=dl['start'], end=dl['end'],
                                     interval=interval, progress=False, auto_adjust=True)

                if df.empty:
                    print(f"  No data!")
                    continue

                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                print(f"  Fetched {len(df)} rows: {df.index.min()} to {df.index.max()}")
                inserted = store_data(symbol, tf, df)
                print(f"  Inserted {inserted} new records")

            except Exception as e:
                print(f"  Error: {e}")

    # Verification
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol, timeframe, MIN(date), MAX(date), COUNT(*)
        FROM market_data_unified
        WHERE symbol IN ('NIFTY50', 'BANKNIFTY')
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
    """)
    print(f"\n{'='*50}")
    print("=== DATA VERIFICATION ===")
    for r in cur.fetchall():
        print(f"  {r[0]:12s} | {r[1]:10s} | {r[2]} to {r[3]} | {r[4]:,} rows")
    conn.close()


if __name__ == '__main__':
    main()
