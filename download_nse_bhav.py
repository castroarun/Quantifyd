"""
NSE F&O Bhav Copy Downloader
=============================
Downloads daily option chain data (Bhav Copy) for NIFTY and BANKNIFTY options
from NSE archives and stores in SQLite database.

URL Formats:
  Old (before 2024-07-08): nsearchives.nseindia.com/content/historical/DERIVATIVES/{YEAR}/{MON}/fo{DD}{MON}{YYYY}bhav.csv.zip
  New (2024-07-08 onwards): nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip

Usage:
  python download_nse_bhav.py                    # Download full range (Mar 2024 - Mar 2026)
  python download_nse_bhav.py --test              # Test with 5 recent trading days
  python download_nse_bhav.py --calibrate         # Run calibration analysis only (no download)
  python download_nse_bhav.py --start 2025-01-01 --end 2025-01-31  # Custom range
"""

import os
import sys
import csv
import io
import time
import sqlite3
import zipfile
import argparse
import math
from datetime import datetime, timedelta, date
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' library required. Install with: pip install requests")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

DB_DIR = Path(__file__).parent / 'backtest_data'
DB_PATH = DB_DIR / 'market_data.db'

# Date when NSE switched from old to new URL format
FORMAT_SWITCH_DATE = date(2024, 7, 8)

# Symbols to filter
TARGET_SYMBOLS = {'NIFTY', 'BANKNIFTY'}

# Instruments that are index options
TARGET_INSTRUMENTS = {'OPTIDX'}

# Month abbreviations for old URL format
MONTH_ABBR = {
    1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
    7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
}

# HTTP headers to mimic browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.nseindia.com/',
    'Connection': 'keep-alive',
}

# Rate limit between downloads (seconds)
RATE_LIMIT = 2.0

# Request timeout (seconds)
REQUEST_TIMEOUT = 30


# ============================================================================
# Database Setup
# ============================================================================

def init_db():
    """Create the nse_options_bhav table if it doesn't exist."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS nse_options_bhav (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT,
            symbol TEXT,
            expiry_date TEXT,
            strike REAL,
            option_type TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            settle_price REAL,
            contracts INTEGER,
            value_in_lakhs REAL,
            open_interest INTEGER,
            change_in_oi INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(trade_date, symbol, expiry_date, strike, option_type)
        )
    ''')
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_bhav_symbol_date
        ON nse_options_bhav(symbol, trade_date)
    ''')
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_bhav_expiry
        ON nse_options_bhav(symbol, expiry_date, strike, option_type)
    ''')
    conn.commit()
    conn.close()
    print(f"Database ready: {DB_PATH}")


def get_existing_dates():
    """Get set of dates already downloaded."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    try:
        cur.execute("SELECT DISTINCT trade_date FROM nse_options_bhav")
        dates = {row[0] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        dates = set()
    conn.close()
    return dates


def insert_rows(rows):
    """Insert rows into database, ignoring duplicates."""
    if not rows:
        return 0
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.executemany('''
        INSERT OR IGNORE INTO nse_options_bhav
        (trade_date, symbol, expiry_date, strike, option_type,
         open, high, low, close, settle_price,
         contracts, value_in_lakhs, open_interest, change_in_oi)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', rows)
    inserted = cur.rowcount
    conn.commit()
    conn.close()
    return inserted


# ============================================================================
# URL Construction
# ============================================================================

def get_old_url(d):
    """Old format URL (before 2024-07-08)."""
    mon = MONTH_ABBR[d.month]
    dd = f"{d.day:02d}"
    yyyy = str(d.year)
    return f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mon}/fo{dd}{mon}{yyyy}bhav.csv.zip"


def get_new_url(d):
    """New UDiFF format URL (2024-07-08 onwards)."""
    ds = d.strftime('%Y%m%d')
    return f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{ds}_F_0000.csv.zip"


def get_url(d):
    """Get the appropriate URL for the given date."""
    if d < FORMAT_SWITCH_DATE:
        return get_old_url(d)
    else:
        return get_new_url(d)


# ============================================================================
# CSV Parsing
# ============================================================================

def parse_date_field(date_str):
    """Parse date from various formats to YYYY-MM-DD."""
    date_str = date_str.strip()
    # Try DD-MON-YYYY (old format: 28-MAR-2024)
    for fmt in ('%d-%b-%Y', '%d-%B-%Y', '%Y-%m-%d', '%d/%m/%Y', '%Y%m%d'):
        try:
            return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return date_str


def safe_float(val, default=0.0):
    """Safely convert to float."""
    try:
        return float(val.strip()) if val and val.strip() else default
    except (ValueError, AttributeError):
        return default


def safe_int(val, default=0):
    """Safely convert to int."""
    try:
        return int(float(val.strip())) if val and val.strip() else default
    except (ValueError, AttributeError):
        return default


# --- Old format columns ---
# INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP

# --- New UDiFF format columns (verified) ---
# TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,
# XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,
# OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,
# UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,
# TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,
# Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
# FinInstrmTp: IDO=Index Option, IDF=Index Future, STO=Stock Option, STF=Stock Future

def parse_old_format(csv_text, trade_date_str):
    """Parse old format bhav copy CSV."""
    rows = []
    reader = csv.DictReader(io.StringIO(csv_text))

    # Normalize column names (strip whitespace)
    if reader.fieldnames:
        reader.fieldnames = [f.strip() for f in reader.fieldnames]

    nifty_count = 0
    bn_count = 0

    for row in reader:
        # Normalize keys
        row = {k.strip(): v for k, v in row.items()}

        instrument = row.get('INSTRUMENT', '').strip()
        symbol = row.get('SYMBOL', '').strip()

        if instrument not in TARGET_INSTRUMENTS:
            continue
        if symbol not in TARGET_SYMBOLS:
            continue

        option_type = row.get('OPTION_TYP', '').strip()
        if option_type not in ('CE', 'PE'):
            continue

        expiry = parse_date_field(row.get('EXPIRY_DT', ''))
        strike = safe_float(row.get('STRIKE_PR'))

        db_row = (
            trade_date_str,
            symbol,
            expiry,
            strike,
            option_type,
            safe_float(row.get('OPEN')),
            safe_float(row.get('HIGH')),
            safe_float(row.get('LOW')),
            safe_float(row.get('CLOSE')),
            safe_float(row.get('SETTLE_PR')),
            safe_int(row.get('CONTRACTS')),
            safe_float(row.get('VAL_INLAKH')),
            safe_int(row.get('OPEN_INT')),
            safe_int(row.get('CHG_IN_OI')),
        )
        rows.append(db_row)

        if symbol == 'NIFTY':
            nifty_count += 1
        else:
            bn_count += 1

    return rows, nifty_count, bn_count


def parse_new_format(csv_text, trade_date_str):
    """Parse new UDiFF format bhav copy CSV."""
    rows = []
    reader = csv.DictReader(io.StringIO(csv_text))

    if reader.fieldnames:
        reader.fieldnames = [f.strip() for f in reader.fieldnames]

    # Detect column names from first pass
    fieldnames = reader.fieldnames or []

    nifty_count = 0
    bn_count = 0

    for row in reader:
        row = {k.strip(): v for k, v in row.items()}

        # New UDiFF format:
        #   TckrSymb = "NIFTY" or "BANKNIFTY" (exact symbol)
        #   FinInstrmTp = "IDO" (Index Derivative Option), "IDF" (Index Derivative Future),
        #                 "STO" (Stock Option), "STF" (Stock Future)
        #   OptnTp = "CE" or "PE"
        #   StrkPric = strike price

        # Filter for index options only (IDO)
        fin_type = row.get('FinInstrmTp', '').strip()
        if fin_type != 'IDO':
            continue

        # Match symbol exactly from TckrSymb
        symbol = row.get('TckrSymb', '').strip()
        if symbol not in TARGET_SYMBOLS:
            continue

        option_type = row.get('OptnTp', '').strip()
        if option_type not in ('CE', 'PE'):
            continue

        # Parse expiry date
        expiry_raw = row.get('XpryDt', '') or row.get('FininstrmActlXpryDt', '')
        expiry = parse_date_field(expiry_raw) if expiry_raw else ''

        strike = safe_float(row.get('StrkPric'))

        # Price fields
        open_p = safe_float(row.get('OpnPric'))
        high_p = safe_float(row.get('HghPric'))
        low_p = safe_float(row.get('LwPric'))
        close_p = safe_float(row.get('ClsPric')) or safe_float(row.get('LastPric'))
        settle_p = safe_float(row.get('SttlmPric'))

        # Volume and OI
        contracts = safe_int(row.get('TtlTradgVol'))
        val_raw = safe_float(row.get('TtlTrfVal'))
        val_lakh = val_raw / 100000.0  # Convert to lakhs
        oi = safe_int(row.get('OpnIntrst'))
        chg_oi = safe_int(row.get('ChngInOpnIntrst'))

        db_row = (
            trade_date_str,
            symbol,
            expiry,
            strike,
            option_type,
            open_p,
            high_p,
            low_p,
            close_p,
            settle_p,
            contracts,
            val_lakh,
            oi,
            chg_oi,
        )
        rows.append(db_row)

        if symbol == 'NIFTY':
            nifty_count += 1
        else:
            bn_count += 1

    return rows, nifty_count, bn_count


def parse_csv(csv_text, trade_date, trade_date_str):
    """Auto-detect format and parse."""
    # Check first line for format detection
    first_line = csv_text.split('\n')[0] if csv_text else ''

    if 'INSTRUMENT' in first_line.upper() and 'SYMBOL' in first_line.upper():
        return parse_old_format(csv_text, trade_date_str)
    elif 'TckrSymb' in first_line or 'FinInstrmTp' in first_line:
        return parse_new_format(csv_text, trade_date_str)
    else:
        # Try old format first, then new
        rows, n, b = parse_old_format(csv_text, trade_date_str)
        if rows:
            return rows, n, b
        return parse_new_format(csv_text, trade_date_str)


# ============================================================================
# Download Logic
# ============================================================================

def create_session():
    """Create a requests session with proper headers."""
    session = requests.Session()
    session.headers.update(HEADERS)

    # First hit nseindia.com to get cookies
    try:
        resp = session.get('https://www.nseindia.com/', timeout=10)
        print(f"Session init: status={resp.status_code}, cookies={len(session.cookies)}")
    except Exception as e:
        print(f"Session init warning: {e}")

    return session


def download_and_parse(session, d):
    """Download bhav copy for a date, extract CSV, parse and return rows.

    Returns: (rows, nifty_count, bn_count, status)
        status: 'ok', 'holiday', 'error', 'blocked'
    """
    url = get_url(d)
    trade_date_str = d.strftime('%Y-%m-%d')

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)

        if resp.status_code == 404:
            return [], 0, 0, 'holiday'

        if resp.status_code == 403:
            return [], 0, 0, 'blocked'

        if resp.status_code != 200:
            return [], 0, 0, f'http_{resp.status_code}'

        # Check if we got actual zip content
        content_type = resp.headers.get('Content-Type', '')
        if 'text/html' in content_type and len(resp.content) < 5000:
            return [], 0, 0, 'blocked'

        # Extract CSV from ZIP
        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
        except zipfile.BadZipFile:
            return [], 0, 0, 'bad_zip'

        csv_files = [f for f in zf.namelist() if f.lower().endswith('.csv')]
        if not csv_files:
            return [], 0, 0, 'no_csv'

        csv_text = zf.read(csv_files[0]).decode('utf-8', errors='replace')

        rows, nifty_count, bn_count = parse_csv(csv_text, d, trade_date_str)

        return rows, nifty_count, bn_count, 'ok'

    except requests.exceptions.Timeout:
        return [], 0, 0, 'timeout'
    except requests.exceptions.ConnectionError:
        return [], 0, 0, 'conn_error'
    except Exception as e:
        return [], 0, 0, f'error: {str(e)[:50]}'


def generate_trading_days(start_date, end_date):
    """Generate weekdays (Mon-Fri) in the date range."""
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Mon=0 to Fri=4
            yield current
        current += timedelta(days=1)


def download_bhav_data(start_date, end_date):
    """Download bhav copy data for the given date range."""
    init_db()

    existing_dates = get_existing_dates()
    print(f"Already have {len(existing_dates)} dates in database")

    # Generate list of trading days to download
    all_days = list(generate_trading_days(start_date, end_date))
    to_download = [d for d in all_days if d.strftime('%Y-%m-%d') not in existing_dates]

    print(f"Date range: {start_date} to {end_date}")
    print(f"Total weekdays: {len(all_days)}, To download: {len(to_download)}, Skipping: {len(all_days) - len(to_download)}")

    if not to_download:
        print("Nothing to download - all dates already in database.")
        return

    # Create session
    session = create_session()
    time.sleep(1)

    # Track stats
    total_rows = 0
    success_count = 0
    holiday_count = 0
    error_count = 0
    blocked_count = 0
    consecutive_errors = 0

    for i, d in enumerate(to_download, 1):
        trade_date_str = d.strftime('%Y-%m-%d')

        print(f"[{i}/{len(to_download)}] {trade_date_str} ... ", end='', flush=True)

        rows, nifty_count, bn_count, status = download_and_parse(session, d)

        if status == 'ok' and rows:
            inserted = insert_rows(rows)
            total_rows += inserted
            success_count += 1
            consecutive_errors = 0
            print(f"{inserted} options saved (NIFTY: {nifty_count}, BANKNIFTY: {bn_count})")
        elif status == 'holiday':
            holiday_count += 1
            consecutive_errors = 0
            print("holiday/no data")
        elif status == 'blocked':
            blocked_count += 1
            consecutive_errors += 1
            print("BLOCKED by NSE")
        elif status == 'ok' and not rows:
            holiday_count += 1
            consecutive_errors = 0
            print("no NIFTY/BN options found")
        else:
            error_count += 1
            consecutive_errors += 1
            print(f"FAILED: {status}")

        # If blocked multiple times, try refreshing session
        if blocked_count > 0 and blocked_count % 3 == 0:
            print("  -> Refreshing session...")
            session = create_session()
            time.sleep(3)

        # If too many consecutive errors, abort
        if consecutive_errors >= 10:
            print(f"\n*** ABORTING: {consecutive_errors} consecutive errors. NSE may be blocking us. ***")
            print("Try again later or use a VPN.")
            break

        # Rate limit
        time.sleep(RATE_LIMIT)

        sys.stdout.flush()

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Successful days:  {success_count}")
    print(f"  Holidays/no data: {holiday_count}")
    print(f"  Blocked:          {blocked_count}")
    print(f"  Errors:           {error_count}")
    print(f"  Total rows saved: {total_rows}")

    print_db_summary()


def print_db_summary():
    """Print summary of data in the database."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    try:
        cur.execute("SELECT COUNT(*) FROM nse_options_bhav")
        total = cur.fetchone()[0]

        cur.execute("SELECT MIN(trade_date), MAX(trade_date) FROM nse_options_bhav")
        min_date, max_date = cur.fetchone()

        cur.execute("SELECT symbol, COUNT(*) FROM nse_options_bhav GROUP BY symbol")
        by_symbol = dict(cur.fetchall())

        cur.execute("SELECT COUNT(DISTINCT trade_date) FROM nse_options_bhav")
        date_count = cur.fetchone()[0]

        print(f"\n  Database totals:")
        print(f"    Total rows:     {total:,}")
        print(f"    Date range:     {min_date} to {max_date}")
        print(f"    Trading days:   {date_count}")
        for sym, cnt in sorted(by_symbol.items()):
            print(f"    {sym:15s}  {cnt:,} rows")

        # Sample ATM data quality check
        cur.execute("""
            SELECT trade_date, symbol, strike, option_type, close, open_interest
            FROM nse_options_bhav
            WHERE symbol = 'NIFTY' AND option_type = 'CE' AND close > 0
            ORDER BY trade_date DESC, open_interest DESC
            LIMIT 5
        """)
        sample = cur.fetchall()
        if sample:
            print(f"\n  Sample data (NIFTY CE, highest OI, recent):")
            print(f"    {'Date':12s} {'Strike':>8s} {'Close':>8s} {'OI':>10s}")
            for row in sample:
                print(f"    {row[0]:12s} {row[2]:8.0f} {row[4]:8.2f} {row[5]:10,}")

    except sqlite3.OperationalError as e:
        print(f"  No data in database yet: {e}")

    conn.close()


# ============================================================================
# Calibration Analysis
# ============================================================================

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Cumulative normal distribution (approximation)
    call = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return call


def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    put = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return put


def norm_cdf(x):
    """Cumulative normal distribution function (Abramowitz & Stegun approximation)."""
    if x >= 0:
        t = 1.0 / (1.0 + 0.2316419 * x)
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        p = d * math.exp(-0.5 * x * x) * (
            t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
            t * (-1.821255978 + t * 1.330274429))))
        )
        return 1.0 - p
    else:
        return 1.0 - norm_cdf(-x)


def run_calibration():
    """Compare real ATM straddle premiums with Black-Scholes estimates."""
    conn = sqlite3.connect(str(DB_PATH))

    # Check if we have bhav data
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nse_options_bhav WHERE symbol = 'NIFTY'")
        bhav_count = cur.fetchone()[0]
        if bhav_count == 0:
            print("No NIFTY bhav data found. Run download first.")
            conn.close()
            return
    except sqlite3.OperationalError:
        print("No nse_options_bhav table found. Run download first.")
        conn.close()
        return

    # Check if we have 5-min spot data for NIFTY (index) or NIFTY 50 proxy
    # We use daily data from market_data_unified as spot reference
    try:
        cur.execute("""
            SELECT DISTINCT date FROM market_data_unified
            WHERE symbol = 'NIFTY50' AND timeframe = 'day'
            ORDER BY date
        """)
        spot_dates_raw = cur.fetchall()
    except sqlite3.OperationalError:
        spot_dates_raw = []

    if not spot_dates_raw:
        # Try other symbols that track Nifty
        try:
            cur.execute("""
                SELECT DISTINCT symbol FROM market_data_unified
                WHERE timeframe = 'day' AND symbol LIKE '%NIFTY%'
                LIMIT 10
            """)
            nifty_syms = [r[0] for r in cur.fetchall()]
            print(f"  Available NIFTY-like symbols in daily data: {nifty_syms}")
        except:
            pass

        print("\nNo NIFTY 50 daily spot data found in market_data_unified.")
        print("Will use settle_price from options data as spot proxy.")
        print("(For proper calibration, download Nifty 50 index daily data)")

        # Alternative: use the mid-point of ATM options as a proxy
        # Or use the most liquid strike's OI to infer spot
        _calibrate_without_spot(conn)
        conn.close()
        return

    spot_dates = {r[0] for r in spot_dates_raw}

    # Get bhav dates
    cur.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol = 'NIFTY' ORDER BY trade_date")
    bhav_dates = [r[0] for r in cur.fetchall()]

    # Find overlapping dates
    overlap_dates = sorted([d for d in bhav_dates if d in spot_dates])
    print(f"\nCalibration: {len(overlap_dates)} overlapping dates between spot and bhav data")

    if not overlap_dates:
        print("No overlapping dates found.")
        _calibrate_without_spot(conn)
        conn.close()
        return

    _calibrate_with_spot(conn, overlap_dates)
    conn.close()


def _calibrate_without_spot(conn):
    """Calibration using highest-OI strike as ATM proxy."""
    cur = conn.cursor()

    # For each trading day, find the nearest-expiry options and the highest OI strike
    cur.execute("""
        SELECT trade_date, MIN(expiry_date) as nearest_expiry
        FROM nse_options_bhav
        WHERE symbol = 'NIFTY' AND expiry_date >= trade_date
        GROUP BY trade_date
        ORDER BY trade_date
    """)
    date_expiry = cur.fetchall()

    if not date_expiry:
        print("No valid data for calibration.")
        return

    iv_levels = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    r = 0.07  # Risk-free rate (India ~7%)

    straddle_data = []

    for trade_date, nearest_expiry in date_expiry:
        # Get the highest OI CE strike (proxy for ATM)
        cur.execute("""
            SELECT strike, close, open_interest FROM nse_options_bhav
            WHERE symbol = 'NIFTY' AND trade_date = ? AND expiry_date = ?
              AND option_type = 'CE' AND close > 0 AND open_interest > 0
            ORDER BY open_interest DESC
            LIMIT 1
        """, (trade_date, nearest_expiry))
        ce_row = cur.fetchone()

        if not ce_row:
            continue

        atm_strike = ce_row[0]
        ce_close = ce_row[1]

        # Get corresponding PE
        cur.execute("""
            SELECT close FROM nse_options_bhav
            WHERE symbol = 'NIFTY' AND trade_date = ? AND expiry_date = ?
              AND option_type = 'PE' AND strike = ? AND close > 0
        """, (trade_date, nearest_expiry, atm_strike))
        pe_row = cur.fetchone()

        if not pe_row:
            continue

        pe_close = pe_row[0]
        straddle = ce_close + pe_close

        # Days to expiry
        td = datetime.strptime(trade_date, '%Y-%m-%d')
        ed = datetime.strptime(nearest_expiry, '%Y-%m-%d')
        dte = (ed - td).days
        if dte <= 0:
            continue

        T = dte / 365.0
        S = atm_strike  # Use ATM strike as spot proxy
        K = atm_strike

        bs_straddles = {}
        for iv in iv_levels:
            bs_ce = black_scholes_call(S, K, T, r, iv)
            bs_pe = black_scholes_put(S, K, T, r, iv)
            bs_straddles[iv] = bs_ce + bs_pe

        straddle_data.append({
            'date': trade_date,
            'expiry': nearest_expiry,
            'dte': dte,
            'strike': atm_strike,
            'real_straddle': straddle,
            'ce': ce_close,
            'pe': pe_close,
            **{f'bs_{iv}': bs_straddles[iv] for iv in iv_levels}
        })

    if not straddle_data:
        print("No straddle data computed.")
        return

    _print_calibration_results(straddle_data, iv_levels)


def _calibrate_with_spot(conn, overlap_dates):
    """Calibration using actual spot data."""
    cur = conn.cursor()

    iv_levels = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    r = 0.07

    straddle_data = []

    for trade_date in overlap_dates:
        # Get spot close
        cur.execute("""
            SELECT close FROM market_data_unified
            WHERE symbol = 'NIFTY50' AND timeframe = 'day' AND date = ?
        """, (trade_date,))
        spot_row = cur.fetchone()
        if not spot_row:
            continue
        spot = spot_row[0]

        # Get nearest expiry
        cur.execute("""
            SELECT MIN(expiry_date) FROM nse_options_bhav
            WHERE symbol = 'NIFTY' AND trade_date = ? AND expiry_date >= ?
        """, (trade_date, trade_date))
        exp_row = cur.fetchone()
        if not exp_row or not exp_row[0]:
            continue
        nearest_expiry = exp_row[0]

        # Find ATM strike (nearest to spot)
        cur.execute("""
            SELECT DISTINCT strike FROM nse_options_bhav
            WHERE symbol = 'NIFTY' AND trade_date = ? AND expiry_date = ?
            ORDER BY ABS(strike - ?)
            LIMIT 1
        """, (trade_date, nearest_expiry, spot))
        strike_row = cur.fetchone()
        if not strike_row:
            continue
        atm_strike = strike_row[0]

        # Get CE and PE close
        cur.execute("""
            SELECT option_type, close FROM nse_options_bhav
            WHERE symbol = 'NIFTY' AND trade_date = ? AND expiry_date = ?
              AND strike = ? AND option_type IN ('CE', 'PE') AND close > 0
        """, (trade_date, nearest_expiry, atm_strike))
        opt_rows = dict(cur.fetchall())

        if 'CE' not in opt_rows or 'PE' not in opt_rows:
            continue

        ce_close = opt_rows['CE']
        pe_close = opt_rows['PE']
        straddle = ce_close + pe_close

        dte = (datetime.strptime(nearest_expiry, '%Y-%m-%d') - datetime.strptime(trade_date, '%Y-%m-%d')).days
        if dte <= 0:
            continue

        T = dte / 365.0

        bs_straddles = {}
        for iv in iv_levels:
            bs_ce = black_scholes_call(spot, atm_strike, T, r, iv)
            bs_pe = black_scholes_put(spot, atm_strike, T, r, iv)
            bs_straddles[iv] = bs_ce + bs_pe

        straddle_data.append({
            'date': trade_date,
            'expiry': nearest_expiry,
            'dte': dte,
            'spot': spot,
            'strike': atm_strike,
            'real_straddle': straddle,
            'ce': ce_close,
            'pe': pe_close,
            **{f'bs_{iv}': bs_straddles[iv] for iv in iv_levels}
        })

    if not straddle_data:
        print("No straddle data computed.")
        return

    _print_calibration_results(straddle_data, iv_levels)


def _print_calibration_results(straddle_data, iv_levels):
    """Print calibration summary table."""
    print(f"\n{'=' * 80}")
    print("CALIBRATION: Real ATM Straddle vs Black-Scholes Estimates")
    print(f"{'=' * 80}")
    print(f"  Data points: {len(straddle_data)}")
    print(f"  Date range:  {straddle_data[0]['date']} to {straddle_data[-1]['date']}")

    # Compute averages
    avg_real = sum(d['real_straddle'] for d in straddle_data) / len(straddle_data)
    avg_dte = sum(d['dte'] for d in straddle_data) / len(straddle_data)

    print(f"  Avg DTE:     {avg_dte:.1f} days")
    print(f"  Avg real ATM straddle: {avg_real:.2f}")

    print(f"\n  {'IV':>6s}  {'Avg BS Straddle':>16s}  {'Ratio (Real/BS)':>16s}  {'Match?':>8s}")
    print(f"  {'-'*6}  {'-'*16}  {'-'*16}  {'-'*8}")

    best_iv = None
    best_ratio_diff = float('inf')

    for iv in iv_levels:
        key = f'bs_{iv}'
        avg_bs = sum(d[key] for d in straddle_data) / len(straddle_data)
        ratio = avg_real / avg_bs if avg_bs > 0 else 0

        ratio_diff = abs(ratio - 1.0)
        match = ''
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_iv = iv

        if ratio_diff < 0.05:
            match = '***'
        elif ratio_diff < 0.10:
            match = '**'
        elif ratio_diff < 0.20:
            match = '*'

        print(f"  {iv:6.0%}  {avg_bs:16.2f}  {ratio:16.3f}  {match:>8s}")

    print(f"\n  BEST FIT IV: {best_iv:.0%} (ratio closest to 1.0)")

    # Group by DTE buckets
    dte_buckets = {'0-3 DTE': [], '4-7 DTE': [], '8-14 DTE': [], '15+ DTE': []}
    for d in straddle_data:
        dte = d['dte']
        if dte <= 3:
            dte_buckets['0-3 DTE'].append(d)
        elif dte <= 7:
            dte_buckets['4-7 DTE'].append(d)
        elif dte <= 14:
            dte_buckets['8-14 DTE'].append(d)
        else:
            dte_buckets['15+ DTE'].append(d)

    print(f"\n  Breakdown by DTE bucket (using best IV = {best_iv:.0%}):")
    print(f"  {'Bucket':>10s}  {'Count':>6s}  {'Avg Real':>10s}  {'Avg BS':>10s}  {'Ratio':>8s}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")

    key = f'bs_{best_iv}'
    for bucket_name, bucket_data in dte_buckets.items():
        if not bucket_data:
            continue
        avg_r = sum(d['real_straddle'] for d in bucket_data) / len(bucket_data)
        avg_b = sum(d[key] for d in bucket_data) / len(bucket_data)
        ratio = avg_r / avg_b if avg_b > 0 else 0
        print(f"  {bucket_name:>10s}  {len(bucket_data):>6d}  {avg_r:>10.2f}  {avg_b:>10.2f}  {ratio:>8.3f}")

    # Show a few sample rows
    print(f"\n  Sample rows (last 10 days):")
    print(f"  {'Date':>12s}  {'Expiry':>12s}  {'DTE':>4s}  {'Strike':>8s}  {'CE':>8s}  {'PE':>8s}  {'Straddle':>10s}  {'BS@{0:.0%}'.format(best_iv):>10s}")
    for d in straddle_data[-10:]:
        bs_val = d[f'bs_{best_iv}']
        print(f"  {d['date']:>12s}  {d['expiry']:>12s}  {d['dte']:>4d}  {d['strike']:>8.0f}  {d['ce']:>8.2f}  {d['pe']:>8.2f}  {d['real_straddle']:>10.2f}  {bs_val:>10.2f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='NSE F&O Bhav Copy Downloader')
    parser.add_argument('--start', type=str, default='2024-03-01',
                        help='Start date YYYY-MM-DD (default: 2024-03-01)')
    parser.add_argument('--end', type=str, default='2026-03-28',
                        help='End date YYYY-MM-DD (default: 2026-03-28)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: download only 5 recent trading days')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run calibration analysis only (no download)')
    parser.add_argument('--summary', action='store_true',
                        help='Print database summary only')
    args = parser.parse_args()

    if args.summary:
        print_db_summary()
        return

    if args.calibrate:
        run_calibration()
        return

    if args.test:
        # Test with 5 recent trading days
        end = date.today()
        start = end - timedelta(days=10)
        print(f"TEST MODE: downloading {start} to {end}")
    else:
        start = datetime.strptime(args.start, '%Y-%m-%d').date()
        end = datetime.strptime(args.end, '%Y-%m-%d').date()

    download_bhav_data(start, end)

    # Run calibration if we have data
    print("\n\nRunning calibration analysis...")
    run_calibration()


if __name__ == '__main__':
    main()
