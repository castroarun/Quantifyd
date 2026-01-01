"""
IV Percentile Service
=====================

Calculates and tracks Implied Volatility percentile for adaptive strike selection.

IV Percentile tells us where current IV ranks compared to historical IV:
- IV Percentile < 25%: Low volatility - sell closer to ATM (higher delta)
- IV Percentile 25-50%: Normal volatility - standard delta
- IV Percentile 50-75%: Elevated volatility - sell further OTM
- IV Percentile > 75%: High volatility - sell very OTM or use collar

Target Delta Mapping:
- IV < 25%:  Delta 0.35 (3-4% OTM)
- IV 25-50%: Delta 0.30 (4-5% OTM)
- IV 50-75%: Delta 0.25 (5-7% OTM)
- IV > 75%:  Delta 0.20 (7-10% OTM)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent / 'data'
IV_DB_PATH = DATA_DIR / 'iv_history.db'


@dataclass
class IVMetrics:
    """IV metrics for a symbol"""
    symbol: str
    current_iv: float
    iv_percentile: float
    iv_rank: float
    iv_52w_high: float
    iv_52w_low: float
    target_delta: float
    target_otm_pct: float
    regime: str  # 'LOW', 'NORMAL', 'ELEVATED', 'HIGH'
    date: datetime


class IVPercentileService:
    """
    Service for calculating and tracking IV percentile

    Uses 252-day lookback for percentile calculation (1 trading year)
    """

    # Target delta mapping based on IV percentile
    DELTA_MAP = {
        'LOW': 0.35,      # IV < 25%
        'NORMAL': 0.30,   # IV 25-50%
        'ELEVATED': 0.25, # IV 50-75%
        'HIGH': 0.20      # IV > 75%
    }

    # Approximate OTM % for each regime
    OTM_MAP = {
        'LOW': 0.035,      # 3.5% OTM
        'NORMAL': 0.045,   # 4.5% OTM
        'ELEVATED': 0.06,  # 6% OTM
        'HIGH': 0.085      # 8.5% OTM
    }

    def __init__(self, lookback_days: int = 252):
        """
        Initialize IV Percentile Service

        Args:
            lookback_days: Number of days for percentile calculation (default 252 = 1 year)
        """
        self.lookback_days = lookback_days
        self._ensure_database()

    def _ensure_database(self):
        """Create IV history database if it doesn't exist"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(IV_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS iv_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                iv REAL NOT NULL,
                hv_20 REAL,
                hv_60 REAL,
                iv_percentile REAL,
                iv_rank REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_iv_symbol_date
            ON iv_history(symbol, date)
        ''')

        conn.commit()
        conn.close()

    def calculate_iv_from_hv(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """
        Estimate IV from historical volatility

        In absence of real options data, we use HV * 1.2 as IV proxy
        (Options typically trade at 10-30% premium to realized vol)

        Args:
            returns: Series of log returns
            annualize: Whether to annualize (default True)

        Returns:
            Estimated IV
        """
        if len(returns) < 10:
            return 0.20  # Default 20% IV

        # Calculate standard deviation
        std = returns.std()

        # Annualize if needed
        if annualize:
            hv = std * np.sqrt(252)
        else:
            hv = std

        # Apply IV premium (options typically trade above HV)
        iv = hv * 1.2

        # Clamp to reasonable range
        return max(0.10, min(0.80, iv))

    def calculate_iv_percentile(
        self,
        current_iv: float,
        historical_ivs: List[float]
    ) -> float:
        """
        Calculate IV percentile

        Percentile = % of historical observations where IV was lower than current

        Args:
            current_iv: Current implied volatility
            historical_ivs: List of historical IV values

        Returns:
            IV percentile (0-100)
        """
        if not historical_ivs:
            return 50.0  # Default to middle

        below_count = sum(1 for iv in historical_ivs if iv < current_iv)
        percentile = (below_count / len(historical_ivs)) * 100

        return percentile

    def calculate_iv_rank(
        self,
        current_iv: float,
        iv_high: float,
        iv_low: float
    ) -> float:
        """
        Calculate IV Rank

        IV Rank = (Current IV - 52w Low) / (52w High - 52w Low) * 100

        Args:
            current_iv: Current IV
            iv_high: 52-week high IV
            iv_low: 52-week low IV

        Returns:
            IV rank (0-100)
        """
        if iv_high == iv_low:
            return 50.0

        iv_rank = ((current_iv - iv_low) / (iv_high - iv_low)) * 100
        return max(0, min(100, iv_rank))

    def get_regime(self, iv_percentile: float) -> str:
        """
        Determine IV regime based on percentile

        Args:
            iv_percentile: IV percentile (0-100)

        Returns:
            Regime string: 'LOW', 'NORMAL', 'ELEVATED', 'HIGH'
        """
        if iv_percentile < 25:
            return 'LOW'
        elif iv_percentile < 50:
            return 'NORMAL'
        elif iv_percentile < 75:
            return 'ELEVATED'
        else:
            return 'HIGH'

    def get_target_delta(self, iv_percentile: float) -> float:
        """
        Get target delta for strike selection based on IV percentile

        Args:
            iv_percentile: IV percentile (0-100)

        Returns:
            Target delta (0.20-0.35)
        """
        regime = self.get_regime(iv_percentile)
        return self.DELTA_MAP[regime]

    def get_target_otm_pct(self, iv_percentile: float) -> float:
        """
        Get target OTM percentage based on IV percentile

        Args:
            iv_percentile: IV percentile (0-100)

        Returns:
            Target OTM percentage (0.035-0.085)
        """
        regime = self.get_regime(iv_percentile)
        return self.OTM_MAP[regime]

    def calculate_metrics(
        self,
        symbol: str,
        stock_data: pd.DataFrame,
        current_date: datetime
    ) -> IVMetrics:
        """
        Calculate all IV metrics for a symbol

        Args:
            symbol: Stock symbol
            stock_data: DataFrame with OHLCV data (must have 'close' column)
            current_date: Current date for calculation

        Returns:
            IVMetrics dataclass with all metrics
        """
        # Get data up to current date
        mask = stock_data.index <= current_date
        data = stock_data.loc[mask].tail(self.lookback_days + 30)

        if len(data) < 30:
            # Not enough data, return defaults
            return IVMetrics(
                symbol=symbol,
                current_iv=0.20,
                iv_percentile=50.0,
                iv_rank=50.0,
                iv_52w_high=0.30,
                iv_52w_low=0.15,
                target_delta=0.30,
                target_otm_pct=0.045,
                regime='NORMAL',
                date=current_date
            )

        # Calculate log returns
        returns = np.log(data['close'] / data['close'].shift(1)).dropna()

        # Calculate rolling IV estimates (20-day HV * 1.2)
        rolling_ivs = []
        for i in range(20, len(returns)):
            window = returns.iloc[i-20:i]
            hv = window.std() * np.sqrt(252)
            iv_estimate = hv * 1.2
            rolling_ivs.append(iv_estimate)

        if len(rolling_ivs) < 10:
            # Still not enough
            current_iv = self.calculate_iv_from_hv(returns.tail(20))
            return IVMetrics(
                symbol=symbol,
                current_iv=current_iv,
                iv_percentile=50.0,
                iv_rank=50.0,
                iv_52w_high=current_iv * 1.3,
                iv_52w_low=current_iv * 0.7,
                target_delta=0.30,
                target_otm_pct=0.045,
                regime='NORMAL',
                date=current_date
            )

        # Current IV (most recent estimate)
        current_iv = rolling_ivs[-1]

        # Historical IVs for percentile (exclude current)
        historical_ivs = rolling_ivs[:-1]

        # Calculate percentile
        iv_percentile = self.calculate_iv_percentile(current_iv, historical_ivs)

        # Calculate 52-week high/low
        lookback_ivs = rolling_ivs[-min(252, len(rolling_ivs)):]
        iv_52w_high = max(lookback_ivs)
        iv_52w_low = min(lookback_ivs)

        # Calculate IV rank
        iv_rank = self.calculate_iv_rank(current_iv, iv_52w_high, iv_52w_low)

        # Get regime and targets
        regime = self.get_regime(iv_percentile)
        target_delta = self.get_target_delta(iv_percentile)
        target_otm_pct = self.get_target_otm_pct(iv_percentile)

        return IVMetrics(
            symbol=symbol,
            current_iv=current_iv,
            iv_percentile=iv_percentile,
            iv_rank=iv_rank,
            iv_52w_high=iv_52w_high,
            iv_52w_low=iv_52w_low,
            target_delta=target_delta,
            target_otm_pct=target_otm_pct,
            regime=regime,
            date=current_date
        )

    def save_iv_history(
        self,
        symbol: str,
        date: datetime,
        iv: float,
        hv_20: float = None,
        hv_60: float = None,
        iv_percentile: float = None,
        iv_rank: float = None
    ):
        """Save IV data point to history database"""
        conn = sqlite3.connect(IV_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO iv_history
            (symbol, date, iv, hv_20, hv_60, iv_percentile, iv_rank)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, date.date(), iv, hv_20, hv_60, iv_percentile, iv_rank))

        conn.commit()
        conn.close()

    def get_iv_history(
        self,
        symbol: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Get IV history for a symbol

        Args:
            symbol: Stock symbol
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with IV history
        """
        conn = sqlite3.connect(IV_DB_PATH)

        query = "SELECT * FROM iv_history WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.date())

        if end_date:
            query += " AND date <= ?"
            params.append(end_date.date())

        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        return df

    def build_iv_history(
        self,
        symbol: str,
        stock_data: pd.DataFrame,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Build IV history for a symbol from stock data

        Args:
            symbol: Stock symbol
            stock_data: DataFrame with OHLCV
            progress_callback: Optional callback(pct, message)

        Returns:
            DataFrame with IV metrics over time
        """
        results = []
        dates = stock_data.index.tolist()

        # Start after we have enough data for rolling calculation
        start_idx = 60  # Need at least 60 days

        for i, date in enumerate(dates[start_idx:], start=start_idx):
            if progress_callback and i % 50 == 0:
                pct = (i / len(dates)) * 100
                progress_callback(pct, f"Calculating IV for {date.date()}")

            metrics = self.calculate_metrics(symbol, stock_data, date)

            results.append({
                'date': date,
                'iv': metrics.current_iv,
                'iv_percentile': metrics.iv_percentile,
                'iv_rank': metrics.iv_rank,
                'regime': metrics.regime,
                'target_delta': metrics.target_delta
            })

            # Save to database
            self.save_iv_history(
                symbol=symbol,
                date=date,
                iv=metrics.current_iv,
                iv_percentile=metrics.iv_percentile,
                iv_rank=metrics.iv_rank
            )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df.set_index('date', inplace=True)

        return df


# Singleton instance
_iv_service = None


def get_iv_service(lookback_days: int = 252) -> IVPercentileService:
    """Get or create IV Percentile Service instance"""
    global _iv_service
    if _iv_service is None:
        _iv_service = IVPercentileService(lookback_days)
    return _iv_service


# Convenience functions

def get_adaptive_delta(
    symbol: str,
    stock_data: pd.DataFrame,
    current_date: datetime
) -> Tuple[float, IVMetrics]:
    """
    Get adaptive delta based on IV percentile

    Args:
        symbol: Stock symbol
        stock_data: DataFrame with OHLCV
        current_date: Current date

    Returns:
        Tuple of (target_delta, IVMetrics)
    """
    service = get_iv_service()
    metrics = service.calculate_metrics(symbol, stock_data, current_date)
    return metrics.target_delta, metrics


def get_iv_regime_color(regime: str) -> str:
    """Get color for IV regime display"""
    colors = {
        'LOW': '#22c55e',      # Green
        'NORMAL': '#3b82f6',   # Blue
        'ELEVATED': '#f59e0b', # Orange
        'HIGH': '#ef4444'      # Red
    }
    return colors.get(regime, '#6b7280')


def get_regime_description(regime: str) -> str:
    """Get description for IV regime"""
    descriptions = {
        'LOW': 'Low IV - Sell closer to ATM for better premium',
        'NORMAL': 'Normal IV - Standard 0.30 delta strikes',
        'ELEVATED': 'Elevated IV - Sell further OTM for safety',
        'HIGH': 'High IV - Very OTM strikes, consider collar'
    }
    return descriptions.get(regime, 'Unknown regime')
