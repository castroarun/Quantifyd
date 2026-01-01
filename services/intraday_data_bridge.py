"""
Intraday Data Bridge - Access 30-minute candle data from quantflow project

This module provides a bridge to load intraday data stored in the quantflow
project's centralized database for use in CPR-based covered call backtesting.
"""

import sqlite3
import logging
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class IntradayDataBridge:
    """
    Bridge to access intraday data from quantflow project.

    The quantflow project stores 30-minute candle data in SQLite which is
    used for CPR-based entry signal detection.
    """

    # Path to quantflow database
    QUANTFLOW_DB_PATH = Path("C:/Users/Castro/Documents/Projects/quantflow/kiteconnect01/backtest_data/market_data.db")

    # Local fallback path (if data is copied locally)
    LOCAL_FALLBACK_PATH = Path("backtest_data/intraday_data.db")

    # Market timing (IST)
    MARKET_OPEN_TIME = dtime(9, 15)
    MARKET_CLOSE_TIME = dtime(15, 30)

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the intraday data bridge.

        Args:
            db_path: Optional custom path to the database. If not provided,
                     uses QUANTFLOW_DB_PATH or LOCAL_FALLBACK_PATH.
        """
        if db_path:
            self.db_path = db_path
        elif self.QUANTFLOW_DB_PATH.exists():
            self.db_path = self.QUANTFLOW_DB_PATH
        elif self.LOCAL_FALLBACK_PATH.exists():
            self.db_path = self.LOCAL_FALLBACK_PATH
        else:
            self.db_path = self.QUANTFLOW_DB_PATH  # Will fail on access if missing

        self._cache: Dict[str, pd.DataFrame] = {}
        logger.info(f"IntradayDataBridge initialized with DB: {self.db_path}")

    def is_available(self) -> bool:
        """Check if the intraday database is available."""
        return self.db_path.exists()

    def load_30min_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load 30-minute candle data for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            start_date: Start date for data
            end_date: End date for data
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: date (index), open, high, low, close, volume
        """
        cache_key = f"{symbol}_30min_{start_date.date()}_{end_date.date()}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        if not self.is_available():
            logger.warning(f"Intraday database not available at {self.db_path}")
            return pd.DataFrame()

        try:
            conn = sqlite3.connect(self.db_path)

            # Query the unified market data table
            query = """
                SELECT date, open, high, low, close, volume
                FROM market_data_unified
                WHERE symbol = ?
                AND timeframe = '30minute'
                AND date BETWEEN ? AND ?
                ORDER BY date ASC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d 23:59:59')],
                parse_dates=['date']
            )

            conn.close()

            if not df.empty:
                df.set_index('date', inplace=True)
                df.index = pd.to_datetime(df.index)

                # Cache the result
                if use_cache:
                    self._cache[cache_key] = df.copy()

            logger.info(f"Loaded {len(df)} 30-min candles for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error loading 30-min data for {symbol}: {e}")
            return pd.DataFrame()

    def get_first_candle_of_week(
        self,
        symbol: str,
        week_start: datetime,
        intraday_df: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[float], Optional[datetime]]:
        """
        Get the first 30-minute candle of a week (Monday 09:15-09:45).

        Args:
            symbol: Stock symbol
            week_start: Monday of the target week
            intraday_df: Optional pre-loaded intraday data

        Returns:
            Tuple of (close_price, candle_datetime) or (None, None) if not found
        """
        try:
            # Ensure week_start is a Monday
            if week_start.weekday() != 0:
                # Adjust to previous Monday
                week_start = week_start - timedelta(days=week_start.weekday())

            # Load data if not provided
            if intraday_df is None or intraday_df.empty:
                week_end = week_start + timedelta(days=6)
                intraday_df = self.load_30min_data(symbol, week_start, week_end)

            if intraday_df.empty:
                return None, None

            # Filter for Monday
            monday_data = intraday_df[intraday_df.index.date == week_start.date()]

            if monday_data.empty:
                # Try Tuesday if Monday is a holiday
                tuesday = week_start + timedelta(days=1)
                monday_data = intraday_df[intraday_df.index.date == tuesday.date()]

            if monday_data.empty:
                logger.warning(f"No intraday data found for week of {week_start.date()} for {symbol}")
                return None, None

            # Get the first candle (should be 09:15-09:45 or close to it)
            first_candle = monday_data.iloc[0]
            candle_time = monday_data.index[0]

            return float(first_candle['close']), candle_time

        except Exception as e:
            logger.error(f"Error getting first candle for {symbol} week {week_start}: {e}")
            return None, None

    def get_first_candle_of_day(
        self,
        symbol: str,
        target_date: datetime,
        intraday_df: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[float], Optional[datetime]]:
        """
        Get the first 30-minute candle of a specific day (09:15-09:45).

        Args:
            symbol: Stock symbol
            target_date: Target date
            intraday_df: Optional pre-loaded intraday data

        Returns:
            Tuple of (close_price, candle_datetime) or (None, None) if not found
        """
        try:
            # Load data if not provided
            if intraday_df is None or intraday_df.empty:
                intraday_df = self.load_30min_data(
                    symbol,
                    target_date,
                    target_date + timedelta(days=1)
                )

            if intraday_df.empty:
                return None, None

            # Filter for target date
            day_data = intraday_df[intraday_df.index.date == target_date.date()]

            if day_data.empty:
                return None, None

            # Get the first candle
            first_candle = day_data.iloc[0]
            candle_time = day_data.index[0]

            return float(first_candle['close']), candle_time

        except Exception as e:
            logger.error(f"Error getting first candle for {symbol} on {target_date}: {e}")
            return None, None

    def get_available_symbols(self, timeframe: str = '30minute') -> List[str]:
        """
        Get list of symbols available in the intraday database.

        Args:
            timeframe: Timeframe to check (default '30minute')

        Returns:
            List of available symbol names
        """
        if not self.is_available():
            return []

        try:
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

        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []

    def get_date_range(
        self,
        symbol: str,
        timeframe: str = '30minute'
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the date range of available data for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe to check

        Returns:
            Tuple of (min_date, max_date) or (None, None)
        """
        if not self.is_available():
            return None, None

        try:
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
                min_date = pd.to_datetime(result[0])
                max_date = pd.to_datetime(result[1])
                return min_date.to_pydatetime(), max_date.to_pydatetime()

            return None, None

        except Exception as e:
            logger.error(f"Error getting date range for {symbol}: {e}")
            return None, None

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of available intraday data.

        Returns:
            Dictionary with summary statistics
        """
        if not self.is_available():
            return {"status": "unavailable", "db_path": str(self.db_path)}

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Count by timeframe
            cursor.execute("""
                SELECT timeframe, COUNT(DISTINCT symbol), COUNT(*)
                FROM market_data_unified
                GROUP BY timeframe
            """)

            timeframes = {}
            for row in cursor.fetchall():
                timeframes[row[0]] = {
                    "symbols": row[1],
                    "candles": row[2]
                }

            # Get overall date range for 30-minute data
            cursor.execute("""
                SELECT MIN(date), MAX(date)
                FROM market_data_unified
                WHERE timeframe = '30minute'
            """)
            result = cursor.fetchone()

            conn.close()

            return {
                "status": "available",
                "db_path": str(self.db_path),
                "timeframes": timeframes,
                "30min_date_range": {
                    "start": result[0] if result else None,
                    "end": result[1] if result else None
                }
            }

        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {"status": "error", "error": str(e)}

    def clear_cache(self):
        """Clear the internal data cache."""
        self._cache.clear()
        logger.info("Intraday data cache cleared")


# Singleton instance
_bridge_instance: Optional[IntradayDataBridge] = None


def get_intraday_bridge(db_path: Optional[Path] = None) -> IntradayDataBridge:
    """
    Get or create the IntradayDataBridge singleton.

    Args:
        db_path: Optional custom database path

    Returns:
        IntradayDataBridge instance
    """
    global _bridge_instance

    if _bridge_instance is None:
        _bridge_instance = IntradayDataBridge(db_path)

    return _bridge_instance