"""
Consolidation & Breakout Detection for Momentum + Quality Strategy
===================================================================

Detects stocks in consolidation (sideways price action) and breakout events
for the topup mechanism.

Consolidation criteria:
- Price stays within a narrow range (default 5%) for N days (default 20)
- At least 2 local highs and 2 local lows (genuine sideways, not just drift)

Breakout criteria:
- Close > Range High × 1.02 (2% buffer above consolidation range)
- Volume > 1.5× 20-day average volume

Uses daily OHLCV data from the centralized data manager.
"""

import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'


@dataclass
class ConsolidationZone:
    """Detected consolidation zone for a stock."""
    symbol: str
    is_consolidating: bool
    start_date: Optional[datetime] = None
    days_in_range: int = 0
    range_high: float = 0.0
    range_low: float = 0.0
    range_pct: float = 0.0  # Range as % of midpoint
    local_highs: int = 0
    local_lows: int = 0
    current_price: float = 0.0
    as_of_date: Optional[datetime] = None

    @property
    def range_midpoint(self) -> float:
        return (self.range_high + self.range_low) / 2 if self.range_high > 0 else 0

    @property
    def breakout_level(self) -> float:
        """Price level that would trigger a breakout (2% above range high)."""
        return self.range_high * 1.02


@dataclass
class BreakoutSignal:
    """Breakout event detected for a stock."""
    symbol: str
    is_breakout: bool
    breakout_date: Optional[datetime] = None
    breakout_price: float = 0.0
    range_high: float = 0.0
    breakout_pct: float = 0.0  # How far above range high
    volume: float = 0.0
    avg_volume_20d: float = 0.0
    volume_ratio: float = 0.0  # Volume / avg volume
    consolidation: Optional[ConsolidationZone] = None

    @property
    def volume_confirmed(self) -> bool:
        return self.volume_ratio >= 1.5


@dataclass
class ConsolidationScreenResult:
    """Result of screening multiple stocks for consolidation/breakout."""
    as_of_date: datetime
    total_scanned: int
    consolidating: List[ConsolidationZone] = field(default_factory=list)
    breakouts: List[BreakoutSignal] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _count_local_extrema(prices: pd.Series, order: int = 3) -> Tuple[int, int]:
    """
    Count local highs and lows in a price series.

    A local high is a point higher than `order` neighbors on each side.
    A local low is a point lower than `order` neighbors on each side.

    Args:
        prices: Price series (typically close prices)
        order: Number of neighbors to compare on each side

    Returns:
        (num_local_highs, num_local_lows)
    """
    values = prices.values
    n = len(values)
    highs = 0
    lows = 0

    for i in range(order, n - order):
        is_high = True
        is_low = True
        for j in range(1, order + 1):
            if values[i] <= values[i - j] or values[i] <= values[i + j]:
                is_high = False
            if values[i] >= values[i - j] or values[i] >= values[i + j]:
                is_low = False
        if is_high:
            highs += 1
        if is_low:
            lows += 1

    return highs, lows


def detect_consolidation(
    symbol: str,
    as_of_date: datetime = None,
    lookback_days: int = 60,
    min_consolidation_days: int = 20,
    max_range_pct: float = 0.05,
    min_local_extrema: int = 2,
    db_path: Path = None,
) -> ConsolidationZone:
    """
    Detect if a stock is currently in consolidation.

    Looks backward from as_of_date to find the longest recent period
    where price stayed within a narrow range.

    Args:
        symbol: NSE trading symbol
        as_of_date: Date to evaluate (default: latest available)
        lookback_days: How many trading days to look back
        min_consolidation_days: Minimum days in range to qualify (default 20)
        max_range_pct: Maximum price range as % of midpoint (default 5%)
        min_local_extrema: Min local highs AND lows needed (default 2)
        db_path: Path to market data database

    Returns:
        ConsolidationZone with detection results
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)

    if as_of_date is None:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(date) FROM market_data_unified "
            "WHERE symbol = ? AND timeframe = 'day'",
            (symbol,)
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            conn.close()
            return ConsolidationZone(symbol=symbol, is_consolidating=False)
        as_of_date = datetime.strptime(row[0][:10], '%Y-%m-%d')

    calendar_lookback = as_of_date - timedelta(days=int(lookback_days * 1.5))

    query = """
        SELECT date, open, high, low, close, volume
        FROM market_data_unified
        WHERE symbol = ? AND timeframe = 'day'
          AND date >= ? AND date <= ?
        ORDER BY date
    """
    df = pd.read_sql_query(
        query, conn,
        params=(symbol, calendar_lookback.strftime('%Y-%m-%d'), as_of_date.strftime('%Y-%m-%d'))
    )
    conn.close()

    if len(df) < min_consolidation_days:
        return ConsolidationZone(
            symbol=symbol, is_consolidating=False,
            current_price=df.iloc[-1]['close'] if len(df) > 0 else 0,
            as_of_date=as_of_date,
        )

    df['date'] = pd.to_datetime(df['date'])

    # Take the last `lookback_days` trading days
    df = df.tail(lookback_days).reset_index(drop=True)
    current_price = df.iloc[-1]['close']

    # Sliding window: find the longest consolidation ending at the current date
    # Start from the end and expand backward
    best_zone = None
    best_days = 0

    for start_idx in range(len(df) - min_consolidation_days, -1, -1):
        window = df.iloc[start_idx:]
        range_high = window['high'].max()
        range_low = window['low'].min()
        midpoint = (range_high + range_low) / 2

        if midpoint == 0:
            continue

        range_pct = (range_high - range_low) / midpoint

        if range_pct <= max_range_pct:
            days = len(window)
            if days > best_days:
                # Check local extrema
                num_highs, num_lows = _count_local_extrema(window['close'])

                if num_highs >= min_local_extrema and num_lows >= min_local_extrema:
                    best_days = days
                    best_zone = ConsolidationZone(
                        symbol=symbol,
                        is_consolidating=True,
                        start_date=pd.to_datetime(window.iloc[0]['date']),
                        days_in_range=days,
                        range_high=round(range_high, 2),
                        range_low=round(range_low, 2),
                        range_pct=round(range_pct, 4),
                        local_highs=num_highs,
                        local_lows=num_lows,
                        current_price=round(current_price, 2),
                        as_of_date=as_of_date,
                    )
        else:
            # Once range exceeds threshold expanding backward, stop
            break

    if best_zone and best_zone.days_in_range >= min_consolidation_days:
        return best_zone

    return ConsolidationZone(
        symbol=symbol, is_consolidating=False,
        current_price=round(current_price, 2),
        as_of_date=as_of_date,
    )


def detect_breakout(
    symbol: str,
    consolidation: ConsolidationZone = None,
    as_of_date: datetime = None,
    volume_multiplier: float = 1.5,
    breakout_buffer: float = 0.02,
    db_path: Path = None,
) -> BreakoutSignal:
    """
    Detect if a stock has broken out of its consolidation range.

    Args:
        symbol: NSE trading symbol
        consolidation: Pre-computed consolidation zone (if None, will detect)
        as_of_date: Date to check for breakout
        volume_multiplier: Min volume vs 20-day avg (default 1.5x)
        breakout_buffer: % above range high to confirm (default 2%)
        db_path: Path to database

    Returns:
        BreakoutSignal with detection results
    """
    db_path = db_path or DB_PATH

    if consolidation is None:
        consolidation = detect_consolidation(symbol, as_of_date, db_path=db_path)

    if not consolidation.is_consolidating:
        return BreakoutSignal(
            symbol=symbol, is_breakout=False,
            consolidation=consolidation,
        )

    conn = sqlite3.connect(db_path)

    if as_of_date is None:
        as_of_date = consolidation.as_of_date or datetime.now()

    # Get recent data for volume analysis
    vol_lookback = as_of_date - timedelta(days=40)
    query = """
        SELECT date, close, volume
        FROM market_data_unified
        WHERE symbol = ? AND timeframe = 'day'
          AND date >= ? AND date <= ?
        ORDER BY date
    """
    df = pd.read_sql_query(
        query, conn,
        params=(symbol, vol_lookback.strftime('%Y-%m-%d'), as_of_date.strftime('%Y-%m-%d'))
    )
    conn.close()

    if len(df) < 5:
        return BreakoutSignal(
            symbol=symbol, is_breakout=False,
            consolidation=consolidation,
        )

    latest = df.iloc[-1]
    close_price = latest['close']
    volume = latest['volume']

    # 20-day average volume (excluding today)
    avg_volume = df['volume'].iloc[-21:-1].mean() if len(df) > 20 else df['volume'].iloc[:-1].mean()

    breakout_level = consolidation.range_high * (1 + breakout_buffer)
    is_price_breakout = close_price > breakout_level
    volume_ratio = volume / avg_volume if avg_volume > 0 else 0
    is_volume_confirmed = volume_ratio >= volume_multiplier

    is_breakout = is_price_breakout and is_volume_confirmed

    return BreakoutSignal(
        symbol=symbol,
        is_breakout=is_breakout,
        breakout_date=as_of_date if is_breakout else None,
        breakout_price=round(close_price, 2),
        range_high=round(consolidation.range_high, 2),
        breakout_pct=round((close_price - consolidation.range_high) / consolidation.range_high, 4),
        volume=round(volume, 0),
        avg_volume_20d=round(avg_volume, 0),
        volume_ratio=round(volume_ratio, 2),
        consolidation=consolidation,
    )


def screen_consolidation_breakout(
    symbols: List[str],
    as_of_date: datetime = None,
    min_consolidation_days: int = 20,
    max_range_pct: float = 0.05,
    volume_multiplier: float = 1.5,
    db_path: Path = None,
) -> ConsolidationScreenResult:
    """
    Screen multiple stocks for consolidation and breakout signals.

    Used by MonitoringAgent to check portfolio stocks daily.

    Args:
        symbols: List of symbols to screen
        as_of_date: Date to evaluate
        min_consolidation_days: Min days in range
        max_range_pct: Max range as % of midpoint
        volume_multiplier: Min volume ratio for breakout
        db_path: Path to database

    Returns:
        ConsolidationScreenResult with all findings
    """
    db_path = db_path or DB_PATH
    consolidating = []
    breakouts = []
    errors = []

    for symbol in symbols:
        try:
            zone = detect_consolidation(
                symbol, as_of_date,
                min_consolidation_days=min_consolidation_days,
                max_range_pct=max_range_pct,
                db_path=db_path,
            )

            if zone.is_consolidating:
                consolidating.append(zone)

                # Check for breakout
                signal = detect_breakout(
                    symbol, zone, as_of_date,
                    volume_multiplier=volume_multiplier,
                    db_path=db_path,
                )
                if signal.is_breakout:
                    breakouts.append(signal)

        except Exception as e:
            errors.append(f"{symbol}: {e}")

    eval_date = as_of_date or datetime.now()

    logger.info(
        f"Consolidation screen: {len(consolidating)} consolidating, "
        f"{len(breakouts)} breakouts out of {len(symbols)} stocks "
        f"(date={eval_date.date() if isinstance(eval_date, datetime) else eval_date})"
    )

    return ConsolidationScreenResult(
        as_of_date=eval_date,
        total_scanned=len(symbols),
        consolidating=consolidating,
        breakouts=breakouts,
        errors=errors,
    )
