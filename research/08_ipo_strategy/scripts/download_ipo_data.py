"""
Download expanded IPO stock data via yfinance.
Target: All NSE equities listed after 2010 that aren't already in market_data.db.
Stores in same DB, same table (market_data_unified) for seamless backtesting.
"""
import sqlite3, sys, os, time, logging
from datetime import datetime
from pathlib import Path

logging.disable(logging.WARNING)
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd

DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'
CUTOFF_DATE = '2010-01-01'
PROGRESS_FILE = Path(__file__).parent / 'ipo_download_progress.txt'


def get_candidate_symbols():
    """Get NSE equity symbols not already in DB, filtered for real stocks."""
    conn = sqlite3.connect(DB_PATH)

    cur = conn.execute('''
        SELECT tradingsymbol FROM instrument_cache
        WHERE exchange = 'NSE' AND instrument_type = 'EQ'
    ''')
    all_syms = [r[0] for r in cur.fetchall()]

    skip_suffixes = ['-BE', '-SM', '-IT', '-BZ', '-BL', '-IL', '-ST', '-IV']
    skip_contains = ['NAV', 'ETF', 'LIQUID', 'GOLD', 'NIFTY', 'SILVER',
                     'BANKBEES', 'JUNIOR', 'BEES', 'INAV']
    equity = []
    for s in all_syms:
        if any(c.isdigit() for c in s):
            continue
        if any(s.endswith(suf) for suf in skip_suffixes):
            continue
        if any(kw in s for kw in skip_contains):
            continue
        if len(s) < 2:
            continue
        equity.append(s)

    cur2 = conn.execute('SELECT DISTINCT symbol FROM market_data_unified')
    existing = {r[0] for r in cur2.fetchall()}
    conn.close()

    new_syms = sorted([s for s in equity if s not in existing])

    # Skip already-attempted symbols (from previous partial runs)
    done = set()
    if PROGRESS_FILE.exists():
        done = set(PROGRESS_FILE.read_text().strip().split('\n'))
    new_syms = [s for s in new_syms if s not in done]

    return new_syms, existing, done


def download_single(sym, conn):
    """Download one stock. Returns 'ipo', 'old', or 'fail'."""
    yf_sym = f'{sym}.NS'
    try:
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(start='2009-01-01', auto_adjust=True)

        if df is None or len(df) < 5:
            return 'fail'

        df = df.dropna(subset=['Close'])
        if len(df) < 5:
            return 'fail'

        first_date = df.index.min().tz_localize(None) if df.index.min().tzinfo else df.index.min()
        if first_date < pd.Timestamp(CUTOFF_DATE):
            return 'old'

        # IPO stock — store it
        rows = []
        for idx, row in df.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            rows.append((
                sym, date_str,
                round(float(row.get('Open', 0)), 2),
                round(float(row.get('High', 0)), 2),
                round(float(row.get('Low', 0)), 2),
                round(float(row.get('Close', 0)), 2),
                int(row.get('Volume', 0)),
                'day'
            ))

        conn.executemany('''
            INSERT OR IGNORE INTO market_data_unified
            (symbol, date, open, high, low, close, volume, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows)

        return 'ipo'

    except Exception:
        return 'fail'


def main():
    symbols, existing, done = get_candidate_symbols()
    print(f'Existing symbols in DB: {len(existing)}')
    print(f'Previously attempted: {len(done)}')
    print(f'Remaining to check: {len(symbols)}')
    print(f'Target: stocks listed after {CUTOFF_DATE}')
    print(f'DB: {DB_PATH}\n')

    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS market_data_unified (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            timeframe TEXT DEFAULT 'day',
            PRIMARY KEY (symbol, date, timeframe)
        )
    ''')

    total = len(symbols)
    ipo_count = 0
    old_count = 0
    fail_count = 0

    progress_f = open(PROGRESS_FILE, 'a')

    for i, sym in enumerate(symbols, 1):
        if i % 25 == 1 or i == 1:
            print(f'\n--- Progress: {i}/{total} (IPOs found: {ipo_count}) ---')

        result = download_single(sym, conn)

        if result == 'ipo':
            ipo_count += 1
            conn.commit()
            print(f'  [{i}] {sym}: IPO ({ipo_count} total)')
        elif result == 'old':
            old_count += 1
        else:
            fail_count += 1

        # Track progress
        progress_f.write(f'{sym}\n')
        progress_f.flush()

        # Print periodic summary
        if i % 100 == 0:
            print(f'  --- {i}/{total}: IPO={ipo_count} Old={old_count} Fail={fail_count} ---')
            sys.stdout.flush()

        # Rate limit: ~2 req/sec
        time.sleep(0.4)

    conn.commit()
    conn.close()
    progress_f.close()

    print(f'\n=== DOWNLOAD COMPLETE ===')
    print(f'New IPO stocks added: {ipo_count}')
    print(f'Pre-2010 stocks skipped: {old_count}')
    print(f'Failed/no data: {fail_count}')

    # Final count
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute('''
        SELECT COUNT(*) FROM (
            SELECT symbol FROM market_data_unified
            GROUP BY symbol HAVING MIN(date) >= ?
        )
    ''', (CUTOFF_DATE,))
    total_ipo = cur.fetchone()[0]
    cur2 = conn.execute('SELECT COUNT(DISTINCT symbol) FROM market_data_unified')
    total_all = cur2.fetchone()[0]
    conn.close()

    print(f'\nDB now has {total_all} total symbols')
    print(f'IPO candidates (listed >= {CUTOFF_DATE}): {total_ipo}')


if __name__ == '__main__':
    main()
