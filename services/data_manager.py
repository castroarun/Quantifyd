"""
Centralized Data Manager for Covered Calls Backtester
======================================================

Unified interface for downloading, storing, and accessing market data.
Adapted from quantflow/kiteconnect01 for covered call backtesting.

Key Features:
- Single unified database (market_data.db)
- Support for daily and intraday timeframes
- Focus on F&O eligible stocks for covered calls
- NSE lot sizes for position sizing
- Thread-safe operations
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import time
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

# Database configuration
DB_DIR = Path(__file__).parent.parent / 'backtest_data'
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / 'market_data.db'

# Thread lock for database operations
db_lock = threading.Lock()

# =============================================================================
# F&O Stock Universe - Stocks eligible for covered calls
# =============================================================================

# F&O stocks with lot sizes (as of Nov 2024)
# These are the primary candidates for covered call strategies
FNO_LOT_SIZES = {
    # NIFTY 50 Heavyweights
    'RELIANCE': 250,
    'TCS': 150,
    'HDFCBANK': 550,
    'INFY': 300,
    'ICICIBANK': 700,
    'HINDUNILVR': 300,
    'ITC': 1600,
    'SBIN': 750,
    'BHARTIARTL': 475,
    'KOTAKBANK': 400,
    'LT': 150,
    'AXISBANK': 600,
    'ASIANPAINT': 300,
    'MARUTI': 100,
    'TITAN': 375,
    'BAJFINANCE': 125,
    'HCLTECH': 350,
    'SUNPHARMA': 350,
    'ULTRACEMCO': 100,
    'NESTLEIND': 50,
    'WIPRO': 1500,
    'ONGC': 1925,
    'NTPC': 2875,
    'POWERGRID': 2700,
    'M&M': 350,
    'TATAMOTORS': 575,
    'TECHM': 600,
    'JSWSTEEL': 675,
    'INDUSINDBK': 450,
    'ADANIPORTS': 500,
    'BAJAJFINSV': 125,
    'HINDALCO': 1400,
    'COALINDIA': 2175,
    'DIVISLAB': 200,
    'GRASIM': 425,
    'TATACONSUM': 675,
    'DRREDDY': 125,
    'CIPLA': 650,
    'EICHERMOT': 175,
    'BRITANNIA': 200,
    'APOLLOHOSP': 125,
    'HEROMOTOCO': 300,
    'SBILIFE': 375,
    'BPCL': 1800,
    'TATASTEEL': 500,
    'BAJAJ-AUTO': 125,
    'HDFCLIFE': 1100,
    'SHREECEM': 25,
    'ADANIENT': 250,

    # Additional F&O stocks commonly used
    'VEDL': 1550,
    'TATAPOWER': 1350,
    'GAIL': 2775,
    'JINDALSTEL': 1750,
    'DLF': 1650,
    'GODREJPROP': 750,
    'SIEMENS': 75,
    'HAVELLS': 625,
    'PIDILITIND': 375,
    'DABUR': 1250,
    'MARICO': 1200,
    'COLPAL': 350,
    'MUTHOOTFIN': 500,
    'CHOLAFIN': 625,
    'BEL': 3350,
    'HAL': 175,
    'IOC': 6500,
    'IRCTC': 175,
    'COFORGE': 200,
    'PERSISTENT': 150,
    'MCX': 250,
    'CUMMINSIND': 250,
    'VOLTAS': 350,
    'AMBUJACEM': 1200,
    'TRENT': 175,
    'ZOMATO': 1800,
    'PAYTM': 875,
    'DELHIVERY': 750,
    'PNB': 6000,
    'BANKBARODA': 3950,
    'FEDERALBNK': 4000,
    'IDFCFIRSTB': 7500,
}

# NIFTY 50 stocks (primary universe for covered calls)
NIFTY_50 = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'KOTAKBANK',
    'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'TITAN',
    'BAJFINANCE', 'HCLTECH', 'SUNPHARMA', 'ULTRACEMCO', 'NESTLEIND', 'WIPRO', 'ONGC',
    'NTPC', 'POWERGRID', 'M&M', 'TATAMOTORS', 'TECHM', 'JSWSTEEL', 'INDUSINDBK',
    'ADANIPORTS', 'BAJAJFINSV', 'HINDALCO', 'COALINDIA', 'DIVISLAB', 'GRASIM',
    'TATACONSUM', 'DRREDDY', 'CIPLA', 'EICHERMOT', 'BRITANNIA', 'APOLLOHOSP',
    'HEROMOTOCO', 'SBILIFE', 'BPCL', 'TATASTEEL', 'BAJAJ-AUTO', 'HDFCLIFE',
    'SHREECEM', 'ADANIENT'
]

# Top 10 most liquid F&O stocks (recommended for covered calls)
TOP_10_LIQUID = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
    'SBIN', 'BHARTIARTL', 'AXISBANK', 'ITC', 'KOTAKBANK'
]

# Stock universes
STOCK_UNIVERSES = {
    'nifty_50': NIFTY_50,
    'top_10_liquid': TOP_10_LIQUID,
    'all_fno': list(FNO_LOT_SIZES.keys())
}


def get_lot_size(symbol: str) -> int:
    """Get the F&O lot size for a stock"""
    return FNO_LOT_SIZES.get(symbol, 1)


def get_stock_universe(universe_name: str) -> List[str]:
    """Get list of stocks for a named universe"""
    return STOCK_UNIVERSES.get(universe_name, NIFTY_50)


# =============================================================================
# CentralizedDataManager Class
# =============================================================================

class CentralizedDataManager:
    """
    Centralized manager for all market data operations

    Handles:
    - Data downloads from Zerodha Kite API
    - Data storage in unified database
    - Data retrieval with caching
    - Data validation and quality checks
    """

    def __init__(self, kite=None):
        """
        Initialize the data manager

        Args:
            kite: KiteConnect instance (optional, can be set later)
        """
        self.kite = kite
        self.db_path = DB_PATH
        self._init_database()

        # In-memory cache for frequently accessed data
        self._cache = {}
        self._cache_lock = threading.Lock()

    def _init_database(self):
        """Initialize the unified database schema"""
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Main market data table - unified storage for all timeframes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data_unified (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    date DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, date)
                )
            """)

            # Indexes for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe
                ON market_data_unified(symbol, timeframe)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_date
                ON market_data_unified(date)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_composite
                ON market_data_unified(symbol, timeframe, date)
            """)

            # Download status tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS download_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date DATE,
                    end_date DATE,
                    total_candles INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_source TEXT DEFAULT 'zerodha',
                    UNIQUE(symbol, timeframe)
                )
            """)

            # Data quality metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_candles INTEGER,
                    missing_candles INTEGER,
                    quality_score REAL,
                    issues TEXT
                )
            """)

            conn.commit()
            conn.close()

        logger.info(f"Centralized database initialized at: {self.db_path}")

    def set_kite(self, kite):
        """Set or update the Kite connection"""
        self.kite = kite

    def download_data(
        self,
        symbols: List[str],
        timeframe: str = 'day',
        from_date: datetime = None,
        to_date: datetime = None,
        progress_callback=None
    ) -> Tuple[int, int, List[str]]:
        """
        Download market data for multiple symbols

        Args:
            symbols: List of stock symbols
            timeframe: Data interval ('day' recommended for covered calls)
            from_date: Start date (default: 2 years ago)
            to_date: End date (default: today)
            progress_callback: Optional callback(index, total, symbol, status)

        Returns:
            (successful_count, failed_count, error_list)
        """
        if not self.kite:
            raise ValueError("Kite connection not set. Call set_kite() first.")

        # Default date range for covered calls: 2 years
        if from_date is None:
            from_date = datetime.now() - timedelta(days=730)
        if to_date is None:
            to_date = datetime.now()

        # Validate timeframe
        valid_timeframes = ['day', '60minute', '15minute', '5minute']
        if timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe. Must be one of: {valid_timeframes}")

        successful = 0
        failed = 0
        errors = []

        total = len(symbols)
        logger.info(f"Starting download: {total} symbols, timeframe={timeframe}")

        for idx, symbol in enumerate(symbols, 1):
            try:
                if progress_callback:
                    progress_callback(idx, total, symbol, 'downloading')

                logger.info(f"[{idx}/{total}] Downloading {symbol} ({timeframe})...")

                # Fetch data from Kite API
                df = self._fetch_from_kite(symbol, timeframe, from_date, to_date)

                # Store in database
                self._store_data(symbol, timeframe, df)

                if progress_callback:
                    progress_callback(idx, total, symbol, 'completed')

                logger.info(f"✓ {symbol} completed ({len(df)} candles)")
                successful += 1

            except Exception as e:
                error_msg = f"{symbol}: {str(e)}"
                logger.error(f"✗ {error_msg}")
                errors.append(error_msg)
                failed += 1

                if progress_callback:
                    progress_callback(idx, total, symbol, f'failed: {str(e)}')

        logger.info(f"Download completed! Success: {successful}, Failed: {failed}")
        return successful, failed, errors

    def _fetch_from_kite(
        self,
        symbol: str,
        timeframe: str,
        from_date: datetime,
        to_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch data from Kite API in chunks

        Handles the 2000 candle limit by breaking into chunks
        """
        # Get instrument token
        instruments = self.kite.instruments("NSE")
        instrument_token = None

        for inst in instruments:
            if inst['tradingsymbol'] == symbol and inst['instrument_type'] == 'EQ':
                instrument_token = inst['instrument_token']
                break

        if not instrument_token:
            raise ValueError(f"Instrument token not found for {symbol}")

        # Determine chunk size based on timeframe
        chunk_sizes = {
            '5minute': 7,
            '15minute': 20,
            '60minute': 80,
            'day': 1800
        }

        chunk_size_days = chunk_sizes.get(timeframe, 1800)

        # Fetch in chunks
        all_data = []
        current_date = from_date

        while current_date < to_date:
            chunk_end = min(
                current_date + timedelta(days=chunk_size_days),
                to_date
            )

            try:
                logger.debug(f"  Fetching chunk: {current_date.date()} to {chunk_end.date()}")

                data = self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=current_date,
                    to_date=chunk_end,
                    interval=timeframe
                )

                if data:
                    df_chunk = pd.DataFrame(data)
                    all_data.append(df_chunk)
                    logger.debug(f"  Fetched {len(df_chunk)} candles")

                # Respect rate limit (3 req/sec)
                time.sleep(0.35)

            except Exception as e:
                logger.error(f"  Error fetching chunk: {e}")

            current_date = chunk_end + timedelta(days=1)

        if not all_data:
            raise ValueError(f"No data fetched for {symbol}")

        # Combine all chunks
        df = pd.concat(all_data, ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').drop_duplicates(subset=['date'])

        logger.info(f"Total candles fetched for {symbol}: {len(df)}")
        return df

    def _store_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Store data in the unified database"""
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Prepare data
            df_insert = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            df_insert['symbol'] = symbol
            df_insert['timeframe'] = timeframe

            # Format datetime
            if timeframe == 'day':
                df_insert['date'] = df_insert['date'].dt.strftime('%Y-%m-%d')
            else:
                df_insert['date'] = df_insert['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Get existing dates to filter duplicates
            cursor.execute("""
                SELECT DISTINCT date FROM market_data_unified
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            existing_dates = {row[0] for row in cursor.fetchall()}

            # Filter out duplicates
            df_new = df_insert[~df_insert['date'].isin(existing_dates)].copy()

            if len(df_new) > 0:
                logger.info(f"Inserting {len(df_new)} new candles for {symbol}")

                # Insert in chunks
                chunk_size = 100
                for i in range(0, len(df_new), chunk_size):
                    chunk = df_new.iloc[i:i+chunk_size]
                    chunk.to_sql('market_data_unified', conn, if_exists='append', index=False)

            # Update download status
            cursor.execute("""
                INSERT OR REPLACE INTO download_status
                (symbol, timeframe, start_date, end_date, total_candles, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                symbol,
                timeframe,
                df['date'].min().strftime('%Y-%m-%d'),
                df['date'].max().strftime('%Y-%m-%d'),
                len(df)
            ))

            conn.commit()
            conn.close()

        # Invalidate cache
        self._invalidate_cache(symbol, timeframe)

    def load_data(
        self,
        symbol: str,
        timeframe: str = 'day',
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load market data from database

        Args:
            symbol: Stock symbol
            timeframe: Data interval (default 'day')
            from_date: Optional start date filter
            to_date: Optional end date filter
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data indexed by date
        """
        cache_key = f"{symbol}_{timeframe}_{from_date}_{to_date}"

        if use_cache:
            with self._cache_lock:
                if cache_key in self._cache:
                    return self._cache[cache_key].copy()

        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT date, open, high, low, close, volume
            FROM market_data_unified
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        if from_date:
            query += " AND date >= ?"
            params.append(from_date.strftime('%Y-%m-%d'))

        if to_date:
            query += " AND date <= ?"
            params.append(to_date.strftime('%Y-%m-%d'))

        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if len(df) == 0:
            raise ValueError(f"No {timeframe} data found for {symbol}")

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        if use_cache:
            with self._cache_lock:
                self._cache[cache_key] = df.copy()

        return df

    def calculate_historical_volatility(
        self,
        symbol: str,
        lookback_days: int = 252
    ) -> float:
        """
        Calculate annualized historical volatility for IV estimation

        Args:
            symbol: Stock symbol
            lookback_days: Number of trading days to look back

        Returns:
            Annualized volatility (e.g., 0.25 for 25%)
        """
        df = self.load_data(symbol, 'day')

        # Get last N days of data
        df = df.tail(lookback_days)

        if len(df) < 20:
            raise ValueError(f"Insufficient data for {symbol}")

        # Calculate daily returns
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()

        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)

        return volatility

    def get_available_symbols(self, timeframe: str = 'day') -> List[str]:
        """Get list of symbols with data for a given timeframe"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT symbol
            FROM market_data_unified
            WHERE timeframe = ?
            ORDER BY symbol
        """, (timeframe,))

        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()

        return symbols

    def get_date_range(self, symbol: str, timeframe: str = 'day') -> Tuple[datetime, datetime]:
        """Get the date range of available data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT MIN(date), MAX(date)
            FROM market_data_unified
            WHERE symbol = ? AND timeframe = ?
        """, (symbol, timeframe))

        result = cursor.fetchone()
        conn.close()

        if result and result[0] and result[1]:
            return (pd.to_datetime(result[0]), pd.to_datetime(result[1]))
        else:
            raise ValueError(f"No data found for {symbol} ({timeframe})")

    def get_download_status(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> pd.DataFrame:
        """Get download status for symbols"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT symbol, timeframe, start_date, end_date,
                   total_candles, last_updated, data_source
            FROM download_status
        """
        params = []

        conditions = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if timeframe:
            conditions.append("timeframe = ?")
            params.append(timeframe)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timeframe, symbol"

        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()

        return df

    def get_database_summary(self) -> Dict:
        """Get comprehensive summary of all data in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        summary = {
            'total_records': 0,
            'total_symbols': 0,
            'timeframes': {},
            'database_size_mb': 0
        }

        cursor.execute("SELECT COUNT(*) FROM market_data_unified")
        summary['total_records'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data_unified")
        summary['total_symbols'] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT
                timeframe,
                COUNT(*) as record_count,
                COUNT(DISTINCT symbol) as symbol_count,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM market_data_unified
            GROUP BY timeframe
        """)

        for row in cursor.fetchall():
            timeframe, record_count, symbol_count, min_date, max_date = row
            summary['timeframes'][timeframe] = {
                'records': record_count,
                'symbols': symbol_count,
                'date_range': {'start': min_date, 'end': max_date}
            }

        try:
            db_size_bytes = self.db_path.stat().st_size
            summary['database_size_mb'] = round(db_size_bytes / (1024 * 1024), 2)
        except Exception:
            pass

        conn.close()
        return summary

    def _invalidate_cache(self, symbol: str = None, timeframe: str = None):
        """Invalidate cache entries"""
        with self._cache_lock:
            if symbol is None and timeframe is None:
                self._cache.clear()
            else:
                keys_to_remove = [
                    key for key in self._cache.keys()
                    if (symbol and symbol in key) or (timeframe and timeframe in key)
                ]
                for key in keys_to_remove:
                    del self._cache[key]


# =============================================================================
# Singleton Instance
# =============================================================================

_manager_instance = None
_manager_lock = threading.Lock()


def get_data_manager(kite=None) -> CentralizedDataManager:
    """
    Get singleton instance of CentralizedDataManager

    Args:
        kite: Optional Kite connection to set

    Returns:
        CentralizedDataManager instance
    """
    global _manager_instance

    with _manager_lock:
        if _manager_instance is None:
            _manager_instance = CentralizedDataManager(kite)
        elif kite is not None:
            _manager_instance.set_kite(kite)

    return _manager_instance
