"""
Momentum Filter for Momentum + Quality Strategy
================================================

Filters Nifty 500 stocks by price momentum:
- ATH proximity: stock must be within 10% of its 52-week high
- Bulk screening across the full universe

Uses daily OHLCV data from the centralized data manager.
"""

import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'


@dataclass
class MomentumResult:
    """Result of momentum screening for a single stock."""
    symbol: str
    current_price: float
    high_52w: float
    low_52w: float
    distance_from_ath: float  # 0.0 = at ATH, 0.10 = 10% below
    passes: bool
    as_of_date: Optional[datetime] = None

    @property
    def proximity_pct(self) -> float:
        """How close to ATH as a percentage (100% = at ATH)."""
        return (1 - self.distance_from_ath) * 100


@dataclass
class MomentumScreenResult:
    """Result of screening the full universe."""
    as_of_date: datetime
    total_scanned: int
    total_passed: int
    threshold: float
    results: List[MomentumResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.total_passed / self.total_scanned if self.total_scanned else 0

    @property
    def passed(self) -> List[MomentumResult]:
        return [r for r in self.results if r.passes]

    @property
    def passed_symbols(self) -> List[str]:
        return [r.symbol for r in self.results if r.passes]


def calculate_momentum(
    symbol: str,
    as_of_date: datetime = None,
    threshold: float = 0.10,
    db_path: Path = None,
) -> MomentumResult:
    """
    Calculate ATH proximity momentum for a single stock.

    Args:
        symbol: NSE trading symbol
        as_of_date: Date to evaluate (default: latest available)
        threshold: Max distance from 52-week high to pass (default 0.10 = 10%)
        db_path: Path to market data database

    Returns:
        MomentumResult with pass/fail and metrics
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)

    if as_of_date is None:
        # Get the latest date available for this symbol
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(date) FROM market_data_unified "
            "WHERE symbol = ? AND timeframe = 'day'",
            (symbol,)
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            conn.close()
            raise ValueError(f"No daily data found for {symbol}")
        as_of_date = datetime.strptime(row[0][:10], '%Y-%m-%d')

    # 52-week lookback
    lookback_start = as_of_date - timedelta(days=365)

    query = """
        SELECT date, high, low, close
        FROM market_data_unified
        WHERE symbol = ? AND timeframe = 'day'
          AND date >= ? AND date <= ?
        ORDER BY date
    """
    df = pd.read_sql_query(
        query, conn,
        params=(symbol, lookback_start.strftime('%Y-%m-%d'), as_of_date.strftime('%Y-%m-%d'))
    )
    conn.close()

    if len(df) < 20:
        raise ValueError(f"Insufficient data for {symbol}: {len(df)} days (need >= 20)")

    high_52w = df['high'].max()
    low_52w = df['low'].min()
    current_price = df.iloc[-1]['close']

    distance = (high_52w - current_price) / high_52w if high_52w > 0 else 1.0

    return MomentumResult(
        symbol=symbol,
        current_price=round(current_price, 2),
        high_52w=round(high_52w, 2),
        low_52w=round(low_52w, 2),
        distance_from_ath=round(distance, 4),
        passes=distance <= threshold,
        as_of_date=as_of_date,
    )


def screen_momentum(
    symbols: List[str],
    as_of_date: datetime = None,
    threshold: float = 0.10,
    db_path: Path = None,
) -> MomentumScreenResult:
    """
    Screen multiple stocks for ATH proximity momentum.

    This is the bulk screening function used by ScreeningAgent.
    Runs a single SQL query per symbol for efficiency.

    Args:
        symbols: List of NSE trading symbols to screen
        as_of_date: Date to evaluate (default: latest available)
        threshold: Max distance from 52-week high (default 0.10 = 10%)
        db_path: Path to market data database

    Returns:
        MomentumScreenResult with all results and pass/fail counts
    """
    db_path = db_path or DB_PATH
    results = []
    errors = []

    for symbol in symbols:
        try:
            result = calculate_momentum(symbol, as_of_date, threshold, db_path)
            results.append(result)
        except ValueError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"{symbol}: {e}")

    # Sort by proximity (closest to ATH first)
    results.sort(key=lambda r: r.distance_from_ath)

    passed = [r for r in results if r.passes]
    eval_date = as_of_date or (results[0].as_of_date if results else datetime.now())

    if errors:
        logger.warning(f"Momentum screen: {len(errors)} errors out of {len(symbols)}")

    logger.info(
        f"Momentum screen: {len(passed)}/{len(results)} passed "
        f"(threshold={threshold:.0%}, date={eval_date.date()})"
    )

    return MomentumScreenResult(
        as_of_date=eval_date,
        total_scanned=len(results),
        total_passed=len(passed),
        threshold=threshold,
        results=results,
        errors=errors,
    )


def screen_momentum_fast(
    symbols: List[str],
    as_of_date: datetime = None,
    threshold: float = 0.10,
    db_path: Path = None,
) -> MomentumScreenResult:
    """
    Fast bulk momentum screen using a single SQL query.

    Much faster than per-symbol queries for large universes (500+ stocks).
    Computes 52-week high/low and latest close in one pass.

    Args:
        symbols: List of NSE trading symbols
        as_of_date: Date to evaluate (default: latest in DB)
        threshold: Max distance from 52-week high
        db_path: Path to database

    Returns:
        MomentumScreenResult
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)

    # Determine as_of_date if not provided
    if as_of_date is None:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(date) FROM market_data_unified WHERE timeframe = 'day'"
        )
        row = cursor.fetchone()
        as_of_date = datetime.strptime(row[0][:10], '%Y-%m-%d') if row and row[0] else datetime.now()

    lookback_start = as_of_date - timedelta(days=365)

    # Build parameterized query for all symbols at once
    placeholders = ','.join('?' * len(symbols))

    # Get 52-week high/low for each symbol
    range_query = f"""
        SELECT symbol, MAX(high) as high_52w, MIN(low) as low_52w
        FROM market_data_unified
        WHERE symbol IN ({placeholders})
          AND timeframe = 'day'
          AND date >= ? AND date <= ?
        GROUP BY symbol
    """
    range_params = symbols + [lookback_start.strftime('%Y-%m-%d'), as_of_date.strftime('%Y-%m-%d')]
    df_range = pd.read_sql_query(range_query, conn, params=range_params)

    # Get latest close for each symbol (on or before as_of_date)
    close_query = f"""
        SELECT symbol, close, date
        FROM market_data_unified
        WHERE timeframe = 'day'
          AND (symbol, date) IN (
              SELECT symbol, MAX(date)
              FROM market_data_unified
              WHERE symbol IN ({placeholders})
                AND timeframe = 'day'
                AND date <= ?
              GROUP BY symbol
          )
    """
    close_params = symbols + [as_of_date.strftime('%Y-%m-%d')]
    df_close = pd.read_sql_query(close_query, conn, params=close_params)
    conn.close()

    # Merge range and close data
    df = df_range.merge(df_close[['symbol', 'close']], on='symbol', how='inner')

    results = []
    found_symbols = set(df['symbol'].tolist())
    errors = []

    for _, row in df.iterrows():
        high_52w = row['high_52w']
        current_price = row['close']
        distance = (high_52w - current_price) / high_52w if high_52w > 0 else 1.0

        results.append(MomentumResult(
            symbol=row['symbol'],
            current_price=round(current_price, 2),
            high_52w=round(high_52w, 2),
            low_52w=round(row['low_52w'], 2),
            distance_from_ath=round(distance, 4),
            passes=distance <= threshold,
            as_of_date=as_of_date,
        ))

    # Track missing symbols
    for s in symbols:
        if s not in found_symbols:
            errors.append(f"No daily data found for {s}")

    results.sort(key=lambda r: r.distance_from_ath)
    passed = [r for r in results if r.passes]

    logger.info(
        f"Momentum screen (fast): {len(passed)}/{len(results)} passed "
        f"(threshold={threshold:.0%}, date={as_of_date.date()})"
    )

    return MomentumScreenResult(
        as_of_date=as_of_date,
        total_scanned=len(results),
        total_passed=len(passed),
        threshold=threshold,
        results=results,
        errors=errors,
    )
