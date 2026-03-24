"""
Historical Options Data Backfill Script
=========================================
Downloads 5-minute OHLCV data for NIFTY, BANKNIFTY, SENSEX options
from Kite Connect API into options_data.db.

Usage:
    python backfill_options_data.py                    # All 3 indices, all listed contracts
    python backfill_options_data.py --symbol NIFTY     # Single index
    python backfill_options_data.py --resume            # Resume interrupted download
    python backfill_options_data.py --dump-instruments   # Save today's instruments list only

Limitations:
    - Kite instruments() only returns CURRENTLY LISTED contracts
    - For NIFTY weekly options: typically 4-5 weeks + 2-3 monthly expiries are listed
    - Expired contracts are NOT available — so we can only backfill from listing date
    - To build a 2-year dataset: run this script daily to capture instruments before they expire
    - 5-min chunk size: 7 days per API call, rate limit: 3 req/sec

Strategy:
    1. Every day, dump all NFO/BFO instruments to instruments_archive table
    2. Download 5-min OHLCV for every option contract we have a token for
    3. Over time, the dataset grows as we capture contracts before expiry
    4. Resume-safe: tracks what's already downloaded
"""

import os
import sys
import csv
import time
import sqlite3
import logging
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from services.options_data_manager import OPTIONS_DB_PATH, INDEX_CONFIG, get_options_db

# ─── Constants ───────────────────────────────────────────────

CHUNK_SIZE_DAYS = 7         # 5-min: 7 days = ~525 candles per request (under 2000 limit)
RATE_LIMIT_SLEEP = 0.35     # 3 req/sec
TIMEFRAME = '5minute'

# How far back to try downloading (Kite typically has ~2 years for listed contracts)
MAX_LOOKBACK_DAYS = 730     # 2 years


# ─── Database Extensions ────────────────────────────────────

def init_backfill_tables(db_path=None):
    """Create additional tables for backfill tracking."""
    db_path = db_path or OPTIONS_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        -- Archive of all instrument tokens (saved daily)
        CREATE TABLE IF NOT EXISTS instruments_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dump_date DATE NOT NULL,
            exchange VARCHAR(10) NOT NULL,
            instrument_token INTEGER NOT NULL,
            tradingsymbol VARCHAR(60) NOT NULL,
            name VARCHAR(20),
            instrument_type VARCHAR(10),
            strike REAL,
            expiry DATE,
            lot_size INTEGER,
            tick_size REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_ia_date_name
            ON instruments_archive(dump_date, name);
        CREATE INDEX IF NOT EXISTS idx_ia_token
            ON instruments_archive(instrument_token);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ia_date_token
            ON instruments_archive(dump_date, instrument_token);

        -- 5-min OHLCV data for options (the main backfill target)
        CREATE TABLE IF NOT EXISTS option_ohlc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument_token INTEGER NOT NULL,
            tradingsymbol VARCHAR(60) NOT NULL,
            symbol VARCHAR(15) NOT NULL,
            instrument_type VARCHAR(5) NOT NULL,
            strike REAL NOT NULL,
            expiry DATE NOT NULL,
            timeframe VARCHAR(10) DEFAULT '5minute',
            date TIMESTAMP NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            oi INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_ohlc_token_date
            ON option_ohlc(instrument_token, date);
        CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_strike_date
            ON option_ohlc(symbol, strike, instrument_type, date);
        CREATE INDEX IF NOT EXISTS idx_ohlc_expiry
            ON option_ohlc(expiry, symbol);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlc_unique
            ON option_ohlc(instrument_token, date);

        -- Download progress tracking
        CREATE TABLE IF NOT EXISTS backfill_progress (
            instrument_token INTEGER PRIMARY KEY,
            tradingsymbol VARCHAR(60),
            symbol VARCHAR(15),
            earliest_date DATE,
            latest_date DATE,
            total_candles INTEGER DEFAULT 0,
            status VARCHAR(20) DEFAULT 'pending',
            last_error TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Backfill tables initialized")


# ─── Instrument Dump ────────────────────────────────────────

def dump_instruments(kite, exchanges=None):
    """
    Save today's full instruments list to instruments_archive.
    This is the KEY step — run daily to capture tokens before contracts expire.
    """
    if exchanges is None:
        exchanges = list(set(cfg['exchange'] for cfg in INDEX_CONFIG.values()))

    db_path = OPTIONS_DB_PATH
    conn = sqlite3.connect(db_path)
    today = date.today().isoformat()
    total_saved = 0

    for exchange in exchanges:
        logger.info(f"Fetching instruments from {exchange}...")
        try:
            instruments = kite.instruments(exchange)
            logger.info(f"  Got {len(instruments)} instruments from {exchange}")

            # Filter to our index options only
            index_names = set(cfg['name_filter'] for cfg in INDEX_CONFIG.values()
                              if cfg['exchange'] == exchange)

            option_instruments = [
                inst for inst in instruments
                if inst.get('name') in index_names
                and inst.get('instrument_type') in ('CE', 'PE', 'FUT')
            ]

            logger.info(f"  {len(option_instruments)} index option/future instruments")

            for inst in option_instruments:
                expiry = inst.get('expiry')
                if hasattr(expiry, 'isoformat'):
                    expiry = expiry.isoformat()

                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO instruments_archive
                        (dump_date, exchange, instrument_token, tradingsymbol, name,
                         instrument_type, strike, expiry, lot_size, tick_size)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        today, exchange, inst['instrument_token'],
                        inst['tradingsymbol'], inst.get('name'),
                        inst.get('instrument_type'), inst.get('strike'),
                        expiry, inst.get('lot_size'), inst.get('tick_size'),
                    ))
                    total_saved += 1
                except sqlite3.IntegrityError:
                    pass  # Already dumped today

            time.sleep(RATE_LIMIT_SLEEP)

        except Exception as e:
            logger.error(f"  Failed to fetch {exchange} instruments: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Instruments dump complete: {total_saved} saved for {today}")
    return total_saved


# ─── Historical Data Download ───────────────────────────────

def get_contracts_to_download(symbol=None):
    """
    Get list of option contracts to download, from instruments_archive.
    Returns list of dicts with instrument_token, tradingsymbol, etc.
    """
    conn = sqlite3.connect(OPTIONS_DB_PATH)
    conn.row_factory = sqlite3.Row

    sql = """
        SELECT DISTINCT instrument_token, tradingsymbol, name as symbol,
               instrument_type, strike, expiry, exchange
        FROM instruments_archive
        WHERE instrument_type IN ('CE', 'PE')
    """
    params = []
    if symbol:
        sql += " AND name = ?"
        params.append(symbol)

    sql += " ORDER BY name, expiry, strike, instrument_type"
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    contracts = [dict(r) for r in rows]
    logger.info(f"Found {len(contracts)} option contracts to download"
                f"{f' for {symbol}' if symbol else ''}")
    return contracts


def get_download_progress(instrument_token):
    """Check what's already been downloaded for a contract."""
    conn = sqlite3.connect(OPTIONS_DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM backfill_progress WHERE instrument_token=?",
        (instrument_token,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_progress(instrument_token, tradingsymbol, symbol,
                    earliest_date, latest_date, total_candles, status='done'):
    conn = sqlite3.connect(OPTIONS_DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO backfill_progress
        (instrument_token, tradingsymbol, symbol, earliest_date, latest_date,
         total_candles, status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (instrument_token, tradingsymbol, symbol,
          earliest_date, latest_date, total_candles, status,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()


def download_contract(kite, contract, from_date=None, to_date=None):
    """
    Download 5-min OHLCV for a single option contract.

    Returns: (candles_inserted, earliest_date, latest_date) or None on error
    """
    token = contract['instrument_token']
    tsym = contract['tradingsymbol']
    symbol = contract['symbol']
    inst_type = contract['instrument_type']
    strike = contract['strike']
    expiry = contract['expiry']

    # Check progress — skip if already done
    progress = get_download_progress(token)
    if progress and progress.get('status') == 'done':
        return None  # Already downloaded

    # Determine date range
    if to_date is None:
        # Download up to yesterday (or expiry, whichever is earlier)
        to_date = date.today() - timedelta(days=1)
        if expiry:
            exp_date = date.fromisoformat(str(expiry)[:10]) if isinstance(expiry, str) else expiry
            if exp_date < to_date:
                to_date = exp_date

    if from_date is None:
        # If we have progress, start from where we left off
        if progress and progress.get('latest_date'):
            from_date = date.fromisoformat(progress['latest_date']) + timedelta(days=1)
        else:
            # Go back as far as possible
            from_date = to_date - timedelta(days=MAX_LOOKBACK_DAYS)

    if from_date >= to_date:
        update_progress(token, tsym, symbol, str(from_date), str(to_date), 0, 'done')
        return None

    # Download in chunks
    conn = sqlite3.connect(OPTIONS_DB_PATH)
    total_candles = 0
    earliest = None
    latest = None
    current = datetime.combine(from_date, datetime.min.time())
    end = datetime.combine(to_date, datetime.max.time().replace(microsecond=0))

    try:
        while current < end:
            chunk_end = min(
                current + timedelta(days=CHUNK_SIZE_DAYS),
                end
            )

            try:
                data = kite.historical_data(
                    instrument_token=token,
                    from_date=current,
                    to_date=chunk_end,
                    interval=TIMEFRAME,
                    oi=True,  # Include OI
                )

                if data:
                    for candle in data:
                        dt = candle['date']
                        if hasattr(dt, 'isoformat'):
                            dt_str = dt.isoformat()
                        else:
                            dt_str = str(dt)

                        try:
                            conn.execute("""
                                INSERT OR IGNORE INTO option_ohlc
                                (instrument_token, tradingsymbol, symbol, instrument_type,
                                 strike, expiry, timeframe, date, open, high, low, close,
                                 volume, oi)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                token, tsym, symbol, inst_type,
                                strike, str(expiry)[:10], TIMEFRAME,
                                dt_str, candle['open'], candle['high'],
                                candle['low'], candle['close'],
                                candle.get('volume', 0), candle.get('oi', 0),
                            ))
                        except sqlite3.IntegrityError:
                            pass

                    total_candles += len(data)

                    if not earliest:
                        earliest = str(data[0]['date'])[:10]
                    latest = str(data[-1]['date'])[:10]

                time.sleep(RATE_LIMIT_SLEEP)

            except Exception as e:
                err = str(e)
                if 'Too many requests' in err or '429' in err:
                    logger.warning(f"  Rate limited on {tsym}, waiting 5s...")
                    time.sleep(5)
                    continue
                elif 'No data' in err or 'cdl' in err.lower():
                    # No data for this range — move on
                    pass
                else:
                    logger.warning(f"  Chunk error {tsym} ({current.date()} to {chunk_end.date()}): {err}")

            current = chunk_end + timedelta(days=1)

        conn.commit()

        # Update progress
        status = 'done' if total_candles > 0 else 'empty'
        update_progress(token, tsym, symbol,
                        earliest or str(from_date),
                        latest or str(to_date),
                        total_candles, status)

        return total_candles, earliest, latest

    except Exception as e:
        update_progress(token, tsym, symbol, str(from_date), str(to_date), total_candles, 'error')
        logger.error(f"  Download failed for {tsym}: {e}")
        return None
    finally:
        conn.close()


# ─── Main Orchestration ─────────────────────────────────────

def run_backfill(kite, symbol=None, resume=True):
    """
    Full backfill pipeline:
    1. Dump today's instruments
    2. Get list of contracts to download
    3. Download 5-min data for each
    """
    init_backfill_tables()

    # Step 1: Dump instruments
    logger.info("=" * 60)
    logger.info("STEP 1: Dumping current instruments")
    logger.info("=" * 60)
    dump_instruments(kite)

    # Step 2: Get contracts
    logger.info("=" * 60)
    logger.info("STEP 2: Getting contracts to download")
    logger.info("=" * 60)
    contracts = get_contracts_to_download(symbol)

    if not contracts:
        logger.warning("No contracts found. Run --dump-instruments first.")
        return

    # Filter out already-completed if resuming
    if resume:
        pending = []
        for c in contracts:
            progress = get_download_progress(c['instrument_token'])
            if not progress or progress.get('status') not in ('done', 'empty'):
                pending.append(c)
        logger.info(f"Resume mode: {len(contracts) - len(pending)} already done, "
                     f"{len(pending)} remaining")
        contracts = pending

    # Step 3: Download
    logger.info("=" * 60)
    logger.info(f"STEP 3: Downloading 5-min data for {len(contracts)} contracts")
    logger.info("=" * 60)

    total_contracts = len(contracts)
    total_candles = 0
    errors = 0
    start_time = time.time()

    for i, contract in enumerate(contracts, 1):
        tsym = contract['tradingsymbol']
        sym = contract['symbol']
        strike = contract['strike']
        inst_type = contract['instrument_type']
        expiry = str(contract.get('expiry', ''))[:10]

        print(f"[{i}/{total_contracts}] {sym} {strike}{inst_type} exp={expiry} ({tsym}) ...",
              end='', flush=True)

        result = download_contract(kite, contract)

        if result is None:
            print(" SKIP (done/empty)")
        elif isinstance(result, tuple):
            candles, earliest, latest = result
            total_candles += candles
            print(f" {candles} candles ({earliest} to {latest})")
        else:
            errors += 1
            print(" ERROR")

        sys.stdout.flush()

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"BACKFILL COMPLETE")
    logger.info(f"  Contracts: {total_contracts}")
    logger.info(f"  Candles: {total_candles:,}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 60)


def print_stats():
    """Print current database statistics."""
    db = get_options_db()
    stats = db.get_stats()

    print("\n=== Options Data Database Stats ===")
    print(f"DB: {OPTIONS_DB_PATH}")
    print(f"Size: {stats.get('db_size_mb', 0)} MB")
    print(f"Total chain snapshots: {stats.get('total_rows', 0):,} rows")
    print(f"Today: {stats.get('today_snapshots', 0)} snapshots, {stats.get('today_rows', 0):,} rows")

    # OHLC stats
    conn = sqlite3.connect(OPTIONS_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT COUNT(*) as cnt FROM option_ohlc").fetchone()
        ohlc_count = row['cnt'] if row else 0
        print(f"\n=== 5-min OHLCV Data ===")
        print(f"Total candles: {ohlc_count:,}")

        rows = conn.execute("""
            SELECT symbol, COUNT(*) as candles,
                   COUNT(DISTINCT tradingsymbol) as contracts,
                   MIN(date) as earliest, MAX(date) as latest
            FROM option_ohlc GROUP BY symbol
        """).fetchall()
        for r in rows:
            print(f"  {r['symbol']}: {r['candles']:,} candles, "
                  f"{r['contracts']} contracts, "
                  f"{r['earliest'][:10]} to {r['latest'][:10]}")

        # Progress stats
        rows = conn.execute("""
            SELECT symbol, status, COUNT(*) as cnt
            FROM backfill_progress GROUP BY symbol, status
        """).fetchall()
        if rows:
            print(f"\n=== Download Progress ===")
            for r in rows:
                print(f"  {r['symbol']} [{r['status']}]: {r['cnt']} contracts")

        # Instruments archive
        rows = conn.execute("""
            SELECT dump_date, COUNT(*) as cnt
            FROM instruments_archive
            GROUP BY dump_date ORDER BY dump_date DESC LIMIT 5
        """).fetchall()
        if rows:
            print(f"\n=== Instruments Archive ===")
            for r in rows:
                print(f"  {r['dump_date']}: {r['cnt']} instruments")

    finally:
        conn.close()


# ─── CLI ─────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backfill historical options data from Kite API')
    parser.add_argument('--symbol', choices=['NIFTY', 'BANKNIFTY', 'SENSEX'],
                        help='Single index to download (default: all)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume interrupted download (default: True)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Re-download everything from scratch')
    parser.add_argument('--dump-instruments', action='store_true',
                        help='Only save instruments list, do not download data')
    parser.add_argument('--stats', action='store_true',
                        help='Print database stats and exit')

    args = parser.parse_args()

    if args.stats:
        init_backfill_tables()
        print_stats()
        sys.exit(0)

    # Need Kite auth
    from services.kite_service import get_kite
    try:
        kite = get_kite()
        logger.info("Kite authenticated OK")
    except Exception as e:
        logger.error(f"Kite auth failed: {e}")
        logger.error("Make sure you've logged in via /login on the dashboard first")
        sys.exit(1)

    if args.dump_instruments:
        init_backfill_tables()
        dump_instruments(kite)
        print_stats()
    else:
        resume = not args.no_resume
        run_backfill(kite, symbol=args.symbol, resume=resume)
        print_stats()
