"""
CPR (Central Pivot Range) Intraday Backtesting Engine
=====================================================

Intraday strategy for Indian F&O stocks using Central Pivot Range,
clean candle analysis, and SuperTrend signals on 5-minute data.

Strategy Flow:
1. Pre-market: Calculate weekly and daily CPRs from prior OHLC data
2. Filter: Only trade stocks with narrow weekly CPR (< threshold)
3. Wait for first 30-min candle (9:15-9:45) to close
4. Clean candle check: Low wick-to-body ratio
5. Proximity check: First candle close near weekly CPR levels
6. Entry: SuperTrend(7,3) flip on 5-min after clean candle confirmation
7. Exit: CPR breach on 30-min, SuperTrend flip, EOD, SL, or target

Data sources:
- Daily OHLCV from market_data_unified (timeframe='day')
- 5-minute OHLCV from market_data_unified (timeframe='5minute')
- 30-min candles resampled from 5-min data
"""

import sqlite3
import logging
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from math import sqrt

from .technical_indicators import (
    calc_atr, calc_rsi, calc_supertrend,
    calc_stochastics, calc_bollinger_bands, calc_keltner_channels,
)
from .data_manager import FNO_LOT_SIZES

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'

# Market timing constants (IST)
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
FIRST_CANDLE_END = dtime(9, 45)
EOD_EXIT_TIME = dtime(15, 20)  # Exit 10 min before close

# Trading cost constants
SLIPPAGE_PCT = 0.05 / 100  # 0.05%
BROKERAGE_PER_TRADE = 20.0  # Rs 20 flat per F&O trade
TRADING_DAYS_PER_YEAR = 252

# Available 5-min symbols — dynamically loaded from DB
def _load_intraday_symbols() -> List[str]:
    """Query DB for all symbols that have 5-minute data."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT symbol FROM market_data_unified "
            "WHERE timeframe='5minute' ORDER BY symbol"
        )
        symbols = [row[0] for row in cur.fetchall()]
        conn.close()
        return symbols if symbols else []
    except Exception:
        return []

INTRADAY_SYMBOLS = _load_intraday_symbols()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CPRIntradayConfig:
    """All configurable parameters for CPR intraday backtesting."""

    # Universe & time period
    symbols: List[str] = field(default_factory=lambda: INTRADAY_SYMBOLS.copy())
    start_date: str = '2024-01-01'
    end_date: str = '2025-10-27'
    initial_capital: float = 1_000_000  # Rs 10 Lakh

    # CPR parameters
    narrow_cpr_threshold: float = 1.0  # Weekly CPR width < 1% = narrow

    # Clean candle parameters
    max_wick_pct: float = 20.0  # Max wick as % of body for "clean" candle

    # Proximity to CPR levels
    cpr_proximity_pct: float = 1.0  # First candle close within 1% of CPR

    # SuperTrend parameters
    st_period: int = 7
    st_multiplier: float = 3.0

    # Risk management
    max_loss_per_trade_pct: float = 1.0  # Max 1% capital risk per trade
    position_size_pct: float = 5.0  # 5% of capital per position
    max_positions: int = 5  # Max concurrent positions
    stop_loss_pct: float = 1.5  # 1.5% SL from entry
    target_pct: float = 3.0  # 3% target from entry

    # Optional indicator filters
    use_rsi: bool = False
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    use_stochastic: bool = False
    stoch_k: int = 14
    stoch_d: int = 3

    use_bollinger: bool = False
    bb_period: int = 20
    bb_std: float = 2.0

    use_kc: bool = False
    kc_period: int = 20
    kc_multiplier: float = 1.5

    # Regime filter (NIFTYBEES-based)
    # 'none', 'above_sma50', 'below_sma50', 'above_sma200', 'below_sma200',
    # 'rvol_high', 'rvol_low'
    regime_filter: str = 'none'
    regime_sma_period: int = 50         # For SMA-based filters
    regime_rvol_threshold: float = 12.0  # For RVol filter (annualized %)


# =============================================================================
# Trade & Result Dataclasses
# =============================================================================

@dataclass
class IntradayTrade:
    """Record of a single intraday trade."""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    exit_reason: str  # 'SUPERTREND_FLIP', 'CPR_BREACH', 'EOD', 'SL', 'TARGET'
    pnl: float
    pnl_pct: float
    quantity: int
    gross_pnl: float = 0.0  # Before costs
    brokerage: float = 0.0
    slippage_cost: float = 0.0


@dataclass
class CPRBacktestResult:
    """Complete backtest output with all performance metrics."""
    config: CPRIntradayConfig

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_trades_per_day: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_holding_minutes: float = 0.0

    # Detailed data
    trades: List[IntradayTrade] = field(default_factory=list)
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    per_symbol_stats: Dict[str, dict] = field(default_factory=dict)
    equity_curve: Dict[str, float] = field(default_factory=dict)

    # Exit reason breakdown
    exit_reason_counts: Dict[str, int] = field(default_factory=dict)

    # Costs
    total_brokerage: float = 0.0
    total_slippage: float = 0.0

    # Trading days info
    total_trading_days: int = 0
    days_with_trades: int = 0
    days_with_setups: int = 0


# =============================================================================
# Active Position Tracker (internal)
# =============================================================================

@dataclass
class _ActivePosition:
    """Internal tracker for an open intraday position."""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_time: datetime
    entry_price: float
    quantity: int
    stop_loss: float
    target: float


# =============================================================================
# CPR Intraday Engine
# =============================================================================

class CPRIntradayEngine:
    """
    CPR-based intraday backtesting engine for Indian F&O stocks.

    Uses weekly and daily Central Pivot Range with SuperTrend confirmation
    on 5-minute data. Constructs 30-minute candles by resampling 5-min data.

    Usage:
        config = CPRIntradayConfig(symbols=['RELIANCE', 'HDFCBANK'])
        engine = CPRIntradayEngine(config)
        result = engine.run()
    """

    def __init__(
        self,
        config: CPRIntradayConfig,
        preloaded_daily: Optional[Dict[str, pd.DataFrame]] = None,
        preloaded_5min: Optional[Dict[str, pd.DataFrame]] = None,
        preloaded_niftybees: Optional[pd.DataFrame] = None,
    ):
        self.config = config
        self._daily_data = preloaded_daily or {}
        self._5min_data = preloaded_5min or {}
        self._lot_sizes = FNO_LOT_SIZES
        self._niftybees = preloaded_niftybees  # For regime filter

    # -----------------------------------------------------------------
    # Data Loading
    # -----------------------------------------------------------------

    @staticmethod
    def preload_data(
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Load all needed data once from SQLite.

        Returns:
            (daily_data, five_min_data) dicts keyed by symbol.
            Daily data includes extra lookback for weekly CPR calc.
        """
        daily_data: Dict[str, pd.DataFrame] = {}
        five_min_data: Dict[str, pd.DataFrame] = {}

        # Need ~2 weeks lookback for weekly CPR on the first trading day
        lookback_start = (
            datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=30)
        ).strftime('%Y-%m-%d')

        conn = sqlite3.connect(str(DB_PATH))

        for sym in symbols:
            # Daily data
            query_daily = """
                SELECT date, open, high, low, close, volume
                FROM market_data_unified
                WHERE symbol = ? AND timeframe = 'day'
                  AND date >= ? AND date <= ?
                ORDER BY date
            """
            df_daily = pd.read_sql_query(
                query_daily, conn, params=(sym, lookback_start, end_date)
            )
            if df_daily.empty:
                logger.warning(f"No daily data for {sym}, skipping")
                continue
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily = df_daily.set_index('date').sort_index()
            daily_data[sym] = df_daily

            # 5-minute data
            query_5min = """
                SELECT date, open, high, low, close, volume
                FROM market_data_unified
                WHERE symbol = ? AND timeframe = '5minute'
                  AND date >= ? AND date <= ?
                ORDER BY date
            """
            # 5-min data: use start_date (no lookback needed, ST warmup uses first few days)
            df_5min = pd.read_sql_query(
                query_5min, conn, params=(sym, start_date, end_date + ' 23:59:59')
            )
            if df_5min.empty:
                logger.warning(f"No 5-min data for {sym}, skipping")
                continue
            df_5min['date'] = pd.to_datetime(df_5min['date'])
            df_5min = df_5min.set_index('date').sort_index()
            five_min_data[sym] = df_5min

        # Also load NIFTYBEES daily data for regime filter
        niftybees_df = pd.read_sql_query(
            """SELECT date, open, high, low, close, volume
               FROM market_data_unified
               WHERE symbol = 'NIFTYBEES' AND timeframe = 'day'
                 AND date >= ? AND date <= ?
               ORDER BY date""",
            conn, params=(lookback_start, end_date),
        )
        if not niftybees_df.empty:
            niftybees_df['date'] = pd.to_datetime(niftybees_df['date'])
            niftybees_df = niftybees_df.set_index('date').sort_index()

        conn.close()
        return daily_data, five_min_data, niftybees_df

    def _ensure_data_loaded(self) -> None:
        """Load data if not preloaded."""
        if not self._daily_data or not self._5min_data:
            print("Loading data from database...", flush=True)
            self._daily_data, self._5min_data, niftybees = self.preload_data(
                self.config.symbols, self.config.start_date, self.config.end_date
            )
            if self._niftybees is None:
                self._niftybees = niftybees
            print(
                f"Loaded data for {len(self._daily_data)} symbols "
                f"(daily) and {len(self._5min_data)} symbols (5-min)",
                flush=True,
            )

    # -----------------------------------------------------------------
    # Regime Filter
    # -----------------------------------------------------------------

    def _prepare_regime_data(self) -> Optional[pd.DataFrame]:
        """Pre-compute regime indicators on NIFTYBEES data."""
        cfg = self.config
        if cfg.regime_filter == 'none' or self._niftybees is None or self._niftybees.empty:
            return None

        nb = self._niftybees.copy()

        # SMAs
        for p in [50, 200]:
            nb[f'sma{p}'] = nb['close'].rolling(p).mean()

        # True Range & ATR for RVol
        tr = pd.DataFrame({
            'hl': nb['high'] - nb['low'],
            'hc': abs(nb['high'] - nb['close'].shift(1)),
            'lc': abs(nb['low'] - nb['close'].shift(1)),
        }).max(axis=1)
        nb['atr14'] = tr.rolling(14).mean()

        # 20-day realized vol (annualized %)
        nb['ret'] = nb['close'].pct_change()
        nb['rvol20'] = nb['ret'].rolling(20).std() * (252 ** 0.5) * 100

        return nb

    def _check_regime(self, regime_data: Optional[pd.DataFrame],
                      check_date: pd.Timestamp) -> bool:
        """Return True if regime filter allows trading on this date."""
        cfg = self.config
        if cfg.regime_filter == 'none' or regime_data is None:
            return True

        # Find the most recent NIFTYBEES data on or before check_date
        mask = regime_data.index <= check_date
        if not mask.any():
            return True  # No data yet, allow
        row = regime_data.loc[mask].iloc[-1]

        sma_col = f'sma{cfg.regime_sma_period}'

        if cfg.regime_filter == 'above_sma50':
            return bool(row['close'] > row.get('sma50', 0))
        elif cfg.regime_filter == 'below_sma50':
            return bool(row['close'] < row.get('sma50', float('inf')))
        elif cfg.regime_filter == 'above_sma200':
            return bool(row['close'] > row.get('sma200', 0))
        elif cfg.regime_filter == 'below_sma200':
            return bool(row['close'] < row.get('sma200', float('inf')))
        elif cfg.regime_filter == 'rvol_high':
            return bool(row.get('rvol20', 0) >= cfg.regime_rvol_threshold)
        elif cfg.regime_filter == 'rvol_low':
            return bool(row.get('rvol20', 999) < cfg.regime_rvol_threshold)
        return True

    # -----------------------------------------------------------------
    # CPR Calculations
    # -----------------------------------------------------------------

    def calculate_daily_cpr(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily CPR levels from previous day's OHLC.

        Returns DataFrame indexed by date with columns:
        pivot, bc, tc, r1, r2, s1, s2, cpr_width_pct
        """
        prev = daily_df.shift(1)
        cpr = pd.DataFrame(index=daily_df.index)

        cpr['pivot'] = (prev['high'] + prev['low'] + prev['close']) / 3
        cpr['bc'] = (prev['high'] + prev['low']) / 2
        cpr['tc'] = 2 * cpr['pivot'] - cpr['bc']
        cpr['r1'] = 2 * cpr['pivot'] - prev['low']
        cpr['r2'] = cpr['pivot'] + (prev['high'] - prev['low'])
        cpr['s1'] = 2 * cpr['pivot'] - prev['high']
        cpr['s2'] = cpr['pivot'] - (prev['high'] - prev['low'])
        cpr['cpr_width_pct'] = (
            (cpr['tc'] - cpr['bc']).abs() / cpr['pivot'] * 100
        )

        return cpr.dropna()

    def calculate_weekly_cpr(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weekly CPR from previous week's OHLC (Mon-Fri).

        Returns DataFrame with one row per week (indexed by the Monday
        of the FOLLOWING week, i.e. when the CPR is applicable).
        Columns: pivot, bc, tc, r1, r2, s1, s2, cpr_width_pct
        """
        df = daily_df.copy()
        df['week'] = df.index.isocalendar().week.astype(int)
        df['year'] = df.index.year

        weekly_ohlc = df.groupby(['year', 'week']).agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            first_date=('open', lambda _: _.index[0] if len(_) > 0 else None),
        )

        # Drop the current (incomplete) week if needed
        weekly_ohlc = weekly_ohlc.dropna()

        rows = []
        week_keys = list(weekly_ohlc.index)
        for i in range(1, len(week_keys)):
            prev = weekly_ohlc.loc[week_keys[i - 1]]
            curr_key = week_keys[i]

            h, l, c = prev['high'], prev['low'], prev['close']
            pivot = (h + l + c) / 3
            bc = (h + l) / 2
            tc = 2 * pivot - bc
            r1 = 2 * pivot - l
            r2 = pivot + (h - l)
            s1 = 2 * pivot - h
            s2 = pivot - (h - l)
            width_pct = abs(tc - bc) / pivot * 100

            # Get the first trading day of the current week to use as index
            curr_dates = df[
                (df['year'] == curr_key[0]) & (df['week'] == curr_key[1])
            ].index
            if len(curr_dates) == 0:
                continue

            rows.append({
                'date': curr_dates[0],
                'pivot': pivot,
                'bc': bc,
                'tc': tc,
                'r1': r1,
                'r2': r2,
                's1': s1,
                's2': s2,
                'cpr_width_pct': width_pct,
            })

        if not rows:
            return pd.DataFrame()

        wcpr = pd.DataFrame(rows).set_index('date').sort_index()
        return wcpr

    def _get_weekly_cpr_for_date(
        self, weekly_cpr: pd.DataFrame, trade_date: pd.Timestamp
    ) -> Optional[pd.Series]:
        """Get the applicable weekly CPR for a given trading date."""
        if weekly_cpr.empty:
            return None
        # The weekly CPR valid for `trade_date` is the one whose index (start
        # of the week) is <= trade_date.
        applicable = weekly_cpr[weekly_cpr.index <= trade_date]
        if applicable.empty:
            return None
        return applicable.iloc[-1]

    # -----------------------------------------------------------------
    # 30-Minute Candle Resampling
    # -----------------------------------------------------------------

    def resample_to_30min(self, five_min_df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 5-min candles to 30-min candles.

        Groups every 6 consecutive 5-min candles within market hours:
        9:15-9:45, 9:45-10:15, 10:15-10:45, ... 15:00-15:30

        Returns DataFrame with columns: open, high, low, close, volume
        """
        df = five_min_df.copy()

        # Add date column for grouping
        df['trade_date'] = df.index.date

        # Assign each candle to its 30-min bucket
        # The candle at 9:15 belongs to the 9:15-9:45 bucket, etc.
        minutes_from_open = (
            (df.index.hour - 9) * 60 + df.index.minute - 15
        )
        df['bucket'] = minutes_from_open // 30

        result = df.groupby(['trade_date', 'bucket']).agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            volume=('volume', 'sum'),
            bucket_start=('open', lambda _: _.index[0]),
        ).reset_index()

        result = result.set_index('bucket_start').sort_index()
        result = result.drop(columns=['trade_date', 'bucket'], errors='ignore')

        return result

    # -----------------------------------------------------------------
    # SuperTrend Calculation
    # -----------------------------------------------------------------

    def calculate_supertrend(
        self,
        df: pd.DataFrame,
        period: Optional[int] = None,
        multiplier: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Add SuperTrend indicator columns to a DataFrame.

        Uses the existing calc_supertrend from technical_indicators.

        Returns df with added columns:
        - supertrend: The SuperTrend line value
        - st_direction: 1 for uptrend, -1 for downtrend
        - st_flip_up: Just flipped to uptrend (buy)
        - st_flip_down: Just flipped to downtrend (sell)
        """
        period = period or self.config.st_period
        multiplier = multiplier or self.config.st_multiplier

        df = df.copy()
        st_line, st_dir = calc_supertrend(df, atr_period=period, multiplier=multiplier)
        df['supertrend'] = st_line
        df['st_direction'] = st_dir
        df['st_flip_up'] = (
            (df['st_direction'] == 1)
            & (df['st_direction'].shift(1) == -1)
        )
        df['st_flip_down'] = (
            (df['st_direction'] == -1)
            & (df['st_direction'].shift(1) == 1)
        )
        return df

    # -----------------------------------------------------------------
    # Clean Candle Check
    # -----------------------------------------------------------------

    @staticmethod
    def is_clean_candle(
        o: float, h: float, l: float, c: float,
        max_wick_pct: float = 20.0,
    ) -> Tuple[bool, str]:
        """
        Check if a candle is "clean" (small wicks relative to body).

        A clean candle has upper and lower wicks each <= max_wick_pct
        of the body size.

        Returns:
            (is_clean, direction) where direction is 'RED' or 'GREEN'.
            If body is zero (doji), returns (False, 'DOJI').
        """
        body = abs(c - o)
        if body < 1e-9:
            return False, 'DOJI'

        if c >= o:
            # Green candle
            upper_wick = h - c
            lower_wick = o - l
            direction = 'GREEN'
        else:
            # Red candle
            upper_wick = h - o
            lower_wick = c - l
            direction = 'RED'

        upper_wick_ratio = (upper_wick / body) * 100
        lower_wick_ratio = (lower_wick / body) * 100

        is_clean = (upper_wick_ratio <= max_wick_pct) and (lower_wick_ratio <= max_wick_pct)
        return is_clean, direction

    # -----------------------------------------------------------------
    # Proximity Check
    # -----------------------------------------------------------------

    @staticmethod
    def is_near_cpr(
        price: float,
        cpr_levels: pd.Series,
        proximity_pct: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        Check if a price is within proximity_pct of any weekly CPR level.

        Returns:
            (is_near, nearest_level_name)
        """
        levels = {
            'tc': cpr_levels['tc'],
            'pivot': cpr_levels['pivot'],
            'bc': cpr_levels['bc'],
            'r1': cpr_levels['r1'],
            's1': cpr_levels['s1'],
        }

        best_dist = float('inf')
        best_name = ''

        for name, level in levels.items():
            if pd.isna(level) or level == 0:
                continue
            dist_pct = abs(price - level) / level * 100
            if dist_pct < best_dist:
                best_dist = dist_pct
                best_name = name

        return best_dist <= proximity_pct, best_name

    # -----------------------------------------------------------------
    # Optional Indicator Filters
    # -----------------------------------------------------------------

    def _add_optional_indicators(self, df_5min: pd.DataFrame) -> pd.DataFrame:
        """Add optional indicator columns to 5-min data if enabled."""
        cfg = self.config
        df = df_5min

        if cfg.use_rsi:
            df = df.copy() if df is df_5min else df
            df['rsi'] = calc_rsi(df['close'], period=cfg.rsi_period)

        if cfg.use_stochastic:
            df = df.copy() if df is df_5min else df
            stoch_k, stoch_d = calc_stochastics(
                df, k_period=cfg.stoch_k, d_period=cfg.stoch_d
            )
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d

        if cfg.use_bollinger:
            df = df.copy() if df is df_5min else df
            bb_mid, bb_upper, bb_lower = calc_bollinger_bands(
                df, period=cfg.bb_period, std_dev=cfg.bb_std
            )
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower

        if cfg.use_kc:
            df = df.copy() if df is df_5min else df
            kc_mid, kc_upper, kc_lower = calc_keltner_channels(
                df, ema_period=cfg.kc_period, multiplier=cfg.kc_multiplier
            )
            df['kc_upper'] = kc_upper
            df['kc_lower'] = kc_lower

        return df

    def _check_optional_filters(
        self, row: pd.Series, direction: str
    ) -> bool:
        """
        Check optional indicator filters for trade confirmation.

        Returns True if all enabled filters confirm the direction.
        """
        cfg = self.config

        if cfg.use_rsi:
            rsi_val = row.get('rsi', 50.0)
            if direction == 'SHORT' and rsi_val < cfg.rsi_overbought:
                return False  # Need overbought for shorts
            if direction == 'LONG' and rsi_val > cfg.rsi_oversold:
                return False  # Need oversold for longs

        if cfg.use_stochastic:
            stoch_k = row.get('stoch_k', 50.0)
            if direction == 'SHORT' and stoch_k < 80:
                return False
            if direction == 'LONG' and stoch_k > 20:
                return False

        if cfg.use_bollinger:
            close = row.get('close', 0)
            bb_upper = row.get('bb_upper', float('inf'))
            bb_lower = row.get('bb_lower', 0)
            if direction == 'SHORT' and close < bb_upper:
                return False
            if direction == 'LONG' and close > bb_lower:
                return False

        if cfg.use_kc:
            close = row.get('close', 0)
            kc_upper = row.get('kc_upper', float('inf'))
            kc_lower = row.get('kc_lower', 0)
            if direction == 'SHORT' and close < kc_upper:
                return False
            if direction == 'LONG' and close > kc_lower:
                return False

        return True

    # -----------------------------------------------------------------
    # Position Sizing
    # -----------------------------------------------------------------

    def _calc_quantity(
        self, symbol: str, price: float, capital: float
    ) -> int:
        """
        Calculate position size in lot multiples.

        Position size = config.position_size_pct of capital,
        rounded down to nearest lot.
        """
        lot_size = self._lot_sizes.get(symbol, 1)
        max_notional = capital * (self.config.position_size_pct / 100)
        lots = int(max_notional / (price * lot_size))
        return max(lots, 1) * lot_size

    def _apply_slippage(self, price: float, direction: str, is_entry: bool) -> float:
        """
        Apply slippage to a price.

        Entry buy/short-cover: price goes up (worse fill).
        Entry sell/short: price goes down (worse fill).
        """
        if (direction == 'LONG' and is_entry) or (direction == 'SHORT' and not is_entry):
            return price * (1 + SLIPPAGE_PCT)
        else:
            return price * (1 - SLIPPAGE_PCT)

    # -----------------------------------------------------------------
    # Single Day Execution
    # -----------------------------------------------------------------

    def run_single_day(
        self,
        symbol: str,
        trade_date: pd.Timestamp,
        daily_cpr: pd.Series,
        weekly_cpr: Optional[pd.Series],
        five_min_day: pd.DataFrame,
        thirty_min_day: pd.DataFrame,
        five_min_with_st: pd.DataFrame,
        capital: float,
        current_positions_count: int,
    ) -> List[IntradayTrade]:
        """
        Run the CPR strategy for one stock on one day.

        Args:
            symbol: Stock symbol
            trade_date: The trading date
            daily_cpr: CPR levels for this day (from prev day's OHLC)
            weekly_cpr: Weekly CPR levels applicable to this week
            five_min_day: 5-min candles for this day only
            thirty_min_day: 30-min candles for this day only
            five_min_with_st: Full 5-min data with SuperTrend pre-calculated
            capital: Current available capital
            current_positions_count: Number of currently open positions

        Returns:
            List of IntradayTrade objects for this day.
        """
        cfg = self.config
        trades: List[IntradayTrade] = []

        if five_min_day.empty or thirty_min_day.empty:
            return trades

        if weekly_cpr is None:
            return trades

        # ---- Step 1: Check weekly CPR narrowness ----
        if weekly_cpr['cpr_width_pct'] >= cfg.narrow_cpr_threshold:
            return trades  # Skip: wide CPR

        # ---- Step 2: Get first 30-min candle ----
        first_30 = thirty_min_day.iloc[0]
        f30_open = first_30['open']
        f30_high = first_30['high']
        f30_low = first_30['low']
        f30_close = first_30['close']

        # ---- Step 3: Clean candle check ----
        is_clean, candle_dir = self.is_clean_candle(
            f30_open, f30_high, f30_low, f30_close,
            max_wick_pct=cfg.max_wick_pct,
        )
        if not is_clean:
            return trades  # Skip: not a clean candle

        # ---- Step 4: Proximity check ----
        is_near, nearest_level = self.is_near_cpr(
            f30_close, weekly_cpr, proximity_pct=cfg.cpr_proximity_pct,
        )
        if not is_near:
            return trades  # Skip: first candle not near CPR

        # ---- Step 5: Determine trade direction ----
        # Clean RED candle near CPR -> look for SHORT signals
        # Clean GREEN candle near CPR -> look for LONG signals
        trade_direction = 'SHORT' if candle_dir == 'RED' else 'LONG'

        # ---- Step 6: Scan 5-min candles AFTER first 30-min for SuperTrend signals ----
        # Get 5-min candles after 9:45 and up to EOD exit time
        candle_end_time = thirty_min_day.index[0]  # First 30-min candle start
        # We want candles AFTER the first 30-min candle closes (i.e. >= 9:45)
        first_candle_close_time = pd.Timestamp(
            year=trade_date.year, month=trade_date.month, day=trade_date.day,
            hour=9, minute=45
        )
        eod_time = pd.Timestamp(
            year=trade_date.year, month=trade_date.month, day=trade_date.day,
            hour=EOD_EXIT_TIME.hour, minute=EOD_EXIT_TIME.minute,
        )

        # Filter 5-min data for post-first-candle period
        # Use day-scoped data (already filtered by date) for O(1) vs O(n)
        day_st = five_min_with_st
        post_first_candle = day_st[day_st.index >= first_candle_close_time]

        if post_first_candle.empty:
            return trades

        # ---- Step 7: Look for SuperTrend entry signals ----
        position: Optional[_ActivePosition] = None

        for idx, row in post_first_candle.iterrows():
            candle_time = idx

            # If we hit EOD exit time, close any open position
            if candle_time >= eod_time:
                if position is not None:
                    exit_price = self._apply_slippage(
                        row['close'], position.direction, is_entry=False
                    )
                    trade = self._close_position(position, candle_time, exit_price, 'EOD')
                    trades.append(trade)
                    position = None
                break

            # ---- Check exits for open position ----
            if position is not None:
                exit_price = None
                exit_reason = None

                # Check stop loss
                if position.direction == 'LONG':
                    if row['low'] <= position.stop_loss:
                        exit_price = self._apply_slippage(
                            position.stop_loss, position.direction, is_entry=False
                        )
                        exit_reason = 'SL'
                    elif row['high'] >= position.target:
                        exit_price = self._apply_slippage(
                            position.target, position.direction, is_entry=False
                        )
                        exit_reason = 'TARGET'
                    elif row.get('st_flip_down', False):
                        exit_price = self._apply_slippage(
                            row['close'], position.direction, is_entry=False
                        )
                        exit_reason = 'SUPERTREND_FLIP'
                else:  # SHORT
                    if row['high'] >= position.stop_loss:
                        exit_price = self._apply_slippage(
                            position.stop_loss, position.direction, is_entry=False
                        )
                        exit_reason = 'SL'
                    elif row['low'] <= position.target:
                        exit_price = self._apply_slippage(
                            position.target, position.direction, is_entry=False
                        )
                        exit_reason = 'TARGET'
                    elif row.get('st_flip_up', False):
                        exit_price = self._apply_slippage(
                            row['close'], position.direction, is_entry=False
                        )
                        exit_reason = 'SUPERTREND_FLIP'

                # Check 30-min CPR breach exit
                if exit_reason is None:
                    # Find the 30-min candle this 5-min bar belongs to
                    thirty_min_candles_so_far = thirty_min_day[
                        thirty_min_day.index <= candle_time
                    ]
                    if len(thirty_min_candles_so_far) > 0:
                        latest_30 = thirty_min_candles_so_far.iloc[-1]
                        # For SHORT: exit if 30-min closes ABOVE weekly TC
                        if position.direction == 'SHORT' and latest_30['close'] > weekly_cpr['tc']:
                            # Only trigger on a 30-min boundary
                            time_in_day = (candle_time.hour - 9) * 60 + candle_time.minute - 15
                            if time_in_day > 0 and time_in_day % 30 == 0:
                                exit_price = self._apply_slippage(
                                    row['close'], position.direction, is_entry=False
                                )
                                exit_reason = 'CPR_BREACH'
                        # For LONG: exit if 30-min closes BELOW weekly BC
                        elif position.direction == 'LONG' and latest_30['close'] < weekly_cpr['bc']:
                            time_in_day = (candle_time.hour - 9) * 60 + candle_time.minute - 15
                            if time_in_day > 0 and time_in_day % 30 == 0:
                                exit_price = self._apply_slippage(
                                    row['close'], position.direction, is_entry=False
                                )
                                exit_reason = 'CPR_BREACH'

                if exit_price is not None and exit_reason is not None:
                    trade = self._close_position(position, candle_time, exit_price, exit_reason)
                    trades.append(trade)
                    position = None
                    # After closing, we can look for re-entry on same day
                    continue

            # ---- Check for new entry ----
            if position is not None:
                continue  # Already in a position for this symbol

            if current_positions_count + len([
                t for t in trades if t.exit_reason != ''
            ]) >= cfg.max_positions:
                continue  # Max positions reached

            # SuperTrend signal check
            entry_signal = False
            if trade_direction == 'SHORT' and row.get('st_flip_down', False):
                entry_signal = True
            elif trade_direction == 'LONG' and row.get('st_flip_up', False):
                entry_signal = True

            if not entry_signal:
                continue

            # Optional indicator filter check
            if not self._check_optional_filters(row, trade_direction):
                continue

            # ---- Execute entry ----
            entry_price = self._apply_slippage(
                row['close'], trade_direction, is_entry=True
            )
            quantity = self._calc_quantity(symbol, entry_price, capital)

            if trade_direction == 'LONG':
                sl = entry_price * (1 - cfg.stop_loss_pct / 100)
                tgt = entry_price * (1 + cfg.target_pct / 100)
            else:
                sl = entry_price * (1 + cfg.stop_loss_pct / 100)
                tgt = entry_price * (1 - cfg.target_pct / 100)

            position = _ActivePosition(
                symbol=symbol,
                direction=trade_direction,
                entry_time=candle_time,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=sl,
                target=tgt,
            )

        # ---- End of day: close any remaining position ----
        if position is not None:
            last_candle = five_min_day.iloc[-1]
            exit_price = self._apply_slippage(
                last_candle['close'], position.direction, is_entry=False
            )
            trade = self._close_position(
                position, five_min_day.index[-1], exit_price, 'EOD'
            )
            trades.append(trade)

        return trades

    def _close_position(
        self,
        pos: _ActivePosition,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
    ) -> IntradayTrade:
        """Create an IntradayTrade from closing a position."""
        if pos.direction == 'LONG':
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        brokerage = BROKERAGE_PER_TRADE * 2  # Entry + exit
        slippage_cost = (
            pos.entry_price * SLIPPAGE_PCT * pos.quantity
            + exit_price * SLIPPAGE_PCT * pos.quantity
        )
        net_pnl = gross_pnl - brokerage  # Slippage already in prices

        notional = pos.entry_price * pos.quantity
        pnl_pct = (net_pnl / notional * 100) if notional > 0 else 0.0

        return IntradayTrade(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_date=pos.entry_time,
            entry_price=pos.entry_price,
            exit_date=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            quantity=pos.quantity,
            gross_pnl=gross_pnl,
            brokerage=brokerage,
            slippage_cost=slippage_cost,
        )

    # -----------------------------------------------------------------
    # Weekly-Bias Day Execution (bias already set from Monday's candle)
    # -----------------------------------------------------------------

    def _run_biased_day(
        self,
        symbol: str,
        trade_date: pd.Timestamp,
        bias: str,  # 'SHORT' or 'LONG'
        daily_cpr: pd.Series,
        weekly_cpr: pd.Series,
        five_min_day: pd.DataFrame,
        thirty_min_day: pd.DataFrame,
        five_min_with_st: pd.DataFrame,
        capital: float,
        current_positions_count: int,
        is_bias_day: bool = False,
    ) -> Tuple[List[IntradayTrade], bool]:
        """
        Execute SuperTrend signals for one day with a pre-established weekly bias.
        No CPR/clean candle checks here — those were done on Monday.

        On the bias-setting day (is_bias_day=True), only scan AFTER 9:45
        (first 30-min candle close). On subsequent days, scan from open.

        Returns:
            (trades, bias_breached) — bias_breached=True if a 30-min candle
            closed past the weekly CPR, cancelling the weekly bias.
        """
        cfg = self.config
        trades: List[IntradayTrade] = []
        bias_breached = False

        if five_min_day.empty or five_min_with_st.empty:
            return trades, False

        # Time boundaries
        eod_time = pd.Timestamp(
            year=trade_date.year, month=trade_date.month, day=trade_date.day,
            hour=EOD_EXIT_TIME.hour, minute=EOD_EXIT_TIME.minute,
        )

        # Build a set of 30-min boundary times where CPR breach matters
        # Each boundary: check the just-closed 30-min candle
        breach_candles: Dict[pd.Timestamp, pd.Series] = {}
        for bar_idx, bar30 in thirty_min_day.iterrows():
            # The 30-min candle starting at bar_idx closes 30 min later
            bar_close_time = bar_idx + pd.Timedelta(minutes=30)
            breach_candles[bar_close_time] = bar30

        # On bias day, skip candles before 9:45 (first 30-min candle close)
        scan_data = five_min_with_st
        if is_bias_day:
            first_candle_close = pd.Timestamp(
                year=trade_date.year, month=trade_date.month, day=trade_date.day,
                hour=9, minute=45,
            )
            scan_data = five_min_with_st[five_min_with_st.index >= first_candle_close]
            if scan_data.empty:
                return trades, False

        # Scan 5-min candles for the day
        position: Optional[_ActivePosition] = None

        for idx, row in scan_data.iterrows():
            candle_time = idx

            # EOD exit
            if candle_time >= eod_time:
                if position is not None:
                    exit_price = self._apply_slippage(
                        row['close'], position.direction, is_entry=False
                    )
                    trade = self._close_position(position, candle_time, exit_price, 'EOD')
                    trades.append(trade)
                    position = None
                break

            # ---- Check exits for open position ----
            if position is not None:
                exit_price = None
                exit_reason = None

                if position.direction == 'LONG':
                    if row['low'] <= position.stop_loss:
                        exit_price = self._apply_slippage(
                            position.stop_loss, position.direction, is_entry=False)
                        exit_reason = 'SL'
                    elif row['high'] >= position.target:
                        exit_price = self._apply_slippage(
                            position.target, position.direction, is_entry=False)
                        exit_reason = 'TARGET'
                    elif row.get('st_flip_down', False):
                        exit_price = self._apply_slippage(
                            row['close'], position.direction, is_entry=False)
                        exit_reason = 'SUPERTREND_FLIP'
                else:  # SHORT
                    if row['high'] >= position.stop_loss:
                        exit_price = self._apply_slippage(
                            position.stop_loss, position.direction, is_entry=False)
                        exit_reason = 'SL'
                    elif row['low'] <= position.target:
                        exit_price = self._apply_slippage(
                            position.target, position.direction, is_entry=False)
                        exit_reason = 'TARGET'
                    elif row.get('st_flip_up', False):
                        exit_price = self._apply_slippage(
                            row['close'], position.direction, is_entry=False)
                        exit_reason = 'SUPERTREND_FLIP'

                if exit_price is not None and exit_reason is not None:
                    trade = self._close_position(position, candle_time, exit_price, exit_reason)
                    trades.append(trade)
                    position = None
                    continue

            # ---- Check CPR breach at 30-min boundaries ----
            if candle_time in breach_candles:
                bar30 = breach_candles[candle_time]
                if bias == 'SHORT' and bar30['close'] > weekly_cpr['tc']:
                    bias_breached = True
                    if position is not None:
                        exit_price = self._apply_slippage(
                            row['close'], position.direction, is_entry=False)
                        trade = self._close_position(
                            position, candle_time, exit_price, 'CPR_BREACH')
                        trades.append(trade)
                        position = None
                    break  # Stop trading — bias cancelled
                elif bias == 'LONG' and bar30['close'] < weekly_cpr['bc']:
                    bias_breached = True
                    if position is not None:
                        exit_price = self._apply_slippage(
                            row['close'], position.direction, is_entry=False)
                        trade = self._close_position(
                            position, candle_time, exit_price, 'CPR_BREACH')
                        trades.append(trade)
                        position = None
                    break  # Stop trading — bias cancelled

            # ---- Check for new entry (SuperTrend signal in bias direction) ----
            if position is not None:
                continue

            if current_positions_count + len(trades) >= cfg.max_positions:
                continue

            entry_signal = False
            if bias == 'SHORT' and row.get('st_flip_down', False):
                entry_signal = True
            elif bias == 'LONG' and row.get('st_flip_up', False):
                entry_signal = True

            if not entry_signal:
                continue

            # Optional indicator filter
            if not self._check_optional_filters(row, bias):
                continue

            # Execute entry
            entry_price = self._apply_slippage(row['close'], bias, is_entry=True)
            quantity = self._calc_quantity(symbol, entry_price, capital)

            if bias == 'LONG':
                sl = entry_price * (1 - cfg.stop_loss_pct / 100)
                tgt = entry_price * (1 + cfg.target_pct / 100)
            else:
                sl = entry_price * (1 + cfg.stop_loss_pct / 100)
                tgt = entry_price * (1 - cfg.target_pct / 100)

            position = _ActivePosition(
                symbol=symbol,
                direction=bias,
                entry_time=candle_time,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=sl,
                target=tgt,
            )

        # End of day: close remaining position
        if position is not None:
            last_candle = scan_data.iloc[-1]
            exit_price = self._apply_slippage(
                last_candle['close'], position.direction, is_entry=False)
            trade = self._close_position(
                position, scan_data.index[-1], exit_price, 'EOD')
            trades.append(trade)

        return trades, bias_breached

    # -----------------------------------------------------------------
    # Main Backtest Loop
    # -----------------------------------------------------------------

    def run(self) -> CPRBacktestResult:
        """
        Run the full CPR intraday backtest across all symbols and dates.

        Returns a CPRBacktestResult with all metrics and trade details.
        """
        cfg = self.config
        self._ensure_data_loaded()

        # Filter symbols to those with actual data
        valid_symbols = [
            s for s in cfg.symbols
            if s in self._daily_data and s in self._5min_data
        ]
        if not valid_symbols:
            print("ERROR: No valid symbols with both daily and 5-min data.", flush=True)
            return CPRBacktestResult(config=cfg)

        print(
            f"Running CPR intraday backtest: {len(valid_symbols)} symbols, "
            f"{cfg.start_date} to {cfg.end_date}",
            flush=True,
        )

        # ---- Pre-compute CPR and SuperTrend for all symbols ----
        print("Pre-computing CPR levels and SuperTrend...", flush=True)

        symbol_daily_cpr: Dict[str, pd.DataFrame] = {}
        symbol_weekly_cpr: Dict[str, pd.DataFrame] = {}
        symbol_5min_st: Dict[str, pd.DataFrame] = {}
        symbol_30min: Dict[str, pd.DataFrame] = {}

        for sym in valid_symbols:
            daily_df = self._daily_data[sym]
            five_min_df = self._5min_data[sym]

            # CPR
            symbol_daily_cpr[sym] = self.calculate_daily_cpr(daily_df)
            symbol_weekly_cpr[sym] = self.calculate_weekly_cpr(daily_df)

            # SuperTrend on 5-min
            st_df = self.calculate_supertrend(five_min_df)
            # Add optional indicators
            st_df = self._add_optional_indicators(st_df)
            symbol_5min_st[sym] = st_df

            # 30-min resampled
            symbol_30min[sym] = self.resample_to_30min(five_min_df)

        # ---- Pre-group data by date for O(1) lookups (perf optimization) ----
        print("Pre-grouping data by date for fast lookups...", flush=True)

        sym_5min_by_date: Dict[str, Dict] = {}
        sym_30min_by_date: Dict[str, Dict] = {}
        sym_5min_raw_by_date: Dict[str, Dict] = {}

        start_dt = pd.Timestamp(cfg.start_date)
        end_dt = pd.Timestamp(cfg.end_date)
        all_dates: Set[pd.Timestamp] = set()

        for sym in valid_symbols:
            # Group SuperTrend-enriched 5-min by date
            st_df = symbol_5min_st[sym]
            sym_5min_by_date[sym] = {d: grp for d, grp in st_df.groupby(st_df.index.date)}

            # Group raw 5-min by date
            raw_5 = self._5min_data[sym]
            sym_5min_raw_by_date[sym] = {d: grp for d, grp in raw_5.groupby(raw_5.index.date)}

            # Group 30-min by date
            m30 = symbol_30min[sym]
            sym_30min_by_date[sym] = {d: grp for d, grp in m30.groupby(m30.index.date)}

            # Collect trading dates
            for d in sym_5min_by_date[sym]:
                d_ts = pd.Timestamp(d)
                if start_dt <= d_ts <= end_dt:
                    all_dates.add(d_ts)

        trading_dates = sorted(all_dates)
        total_days = len(trading_dates)

        if total_days == 0:
            print("ERROR: No trading days found in the date range.", flush=True)
            return CPRBacktestResult(config=cfg)

        # ---- Group trading dates by ISO week ----
        from collections import defaultdict
        weeks: Dict[tuple, List[pd.Timestamp]] = defaultdict(list)
        for d in trading_dates:
            iso = d.isocalendar()
            weeks[(iso[0], iso[1])].append(d)
        sorted_weeks = sorted(weeks.keys())
        total_weeks = len(sorted_weeks)
        total_days = len(trading_dates)

        print(f"Processing {total_days} trading days across {total_weeks} weeks...",
              flush=True)

        # ---- Prepare regime filter data ----
        regime_data = self._prepare_regime_data()
        regime_skipped_weeks = 0

        # ---- Week-by-week backtest loop ----
        all_trades: List[IntradayTrade] = []
        capital = cfg.initial_capital
        daily_pnl: Dict[str, float] = {}
        equity_curve: Dict[str, float] = {}
        equity_curve[trading_dates[0].strftime('%Y-%m-%d')] = capital
        days_with_setups = 0
        days_with_trades = 0
        day_counter = 0

        progress_interval = max(1, total_days // 20)

        for week_idx, week_key in enumerate(sorted_weeks):
            week_days = weeks[week_key]
            first_day = week_days[0]

            # --- Regime filter check (weekly level) ---
            if not self._check_regime(regime_data, first_day):
                regime_skipped_weeks += 1
                # Still process days for equity curve / day counter
                for trade_date in week_days:
                    day_counter += 1
                    td = trade_date.date()
                    date_str = trade_date.strftime('%Y-%m-%d')
                    daily_pnl[date_str] = 0.0
                    equity_curve[date_str] = capital
                continue

            # --- Per-symbol: determine weekly bias from first day's 1st 30-min candle ---
            # weekly_bias[sym] = 'SHORT' | 'LONG' | None
            weekly_bias: Dict[str, Optional[str]] = {}
            weekly_cpr_cache: Dict[str, Optional[pd.Series]] = {}
            bias_cancelled: Dict[str, bool] = {}

            for sym in valid_symbols:
                w_cpr = symbol_weekly_cpr[sym]
                weekly_cpr_row = self._get_weekly_cpr_for_date(w_cpr, first_day)
                weekly_cpr_cache[sym] = weekly_cpr_row
                weekly_bias[sym] = None
                bias_cancelled[sym] = False

                if weekly_cpr_row is None:
                    continue

                # Check narrow CPR
                if weekly_cpr_row['cpr_width_pct'] >= cfg.narrow_cpr_threshold:
                    continue  # Not narrow enough

                # Get first 30-min candle of the week
                fd = first_day.date()
                thirty_min_day = sym_30min_by_date[sym].get(fd)
                if thirty_min_day is None or thirty_min_day.empty:
                    continue

                first_30 = thirty_min_day.iloc[0]
                is_clean, candle_dir = self.is_clean_candle(
                    first_30['open'], first_30['high'],
                    first_30['low'], first_30['close'],
                    max_wick_pct=cfg.max_wick_pct,
                )
                if not is_clean:
                    continue

                # Proximity check
                is_near, _ = self.is_near_cpr(
                    first_30['close'], weekly_cpr_row,
                    proximity_pct=cfg.cpr_proximity_pct,
                )
                if not is_near:
                    continue

                # Set weekly bias
                weekly_bias[sym] = 'SHORT' if candle_dir == 'RED' else 'LONG'

            # Count how many stocks have a bias this week
            active_biases = {s: b for s, b in weekly_bias.items() if b is not None}

            # --- Process each day of the week ---
            for trade_date in week_days:
                day_counter += 1
                td = trade_date.date()

                if day_counter % progress_interval == 0 or day_counter == total_days:
                    pct = day_counter / total_days * 100
                    active_count = sum(1 for s, b in weekly_bias.items()
                                       if b and not bias_cancelled.get(s, False))
                    print(
                        f"  [{day_counter}/{total_days}] {trade_date.strftime('%Y-%m-%d')} "
                        f"({pct:.0f}%) | Capital: {capital:,.0f} | Trades: {len(all_trades)} "
                        f"| Active biases: {active_count}",
                        flush=True,
                    )

                day_trades: List[IntradayTrade] = []
                day_had_setup = False

                for sym in valid_symbols:
                    # Skip if no weekly bias or bias was cancelled
                    if weekly_bias[sym] is None or bias_cancelled.get(sym, False):
                        continue

                    day_had_setup = True
                    w_cpr_row = weekly_cpr_cache[sym]

                    # O(1) lookups
                    five_min_day = sym_5min_by_date[sym].get(td)
                    thirty_min_day = sym_30min_by_date[sym].get(td)
                    five_min_raw = sym_5min_raw_by_date[sym].get(td)

                    if five_min_day is None or thirty_min_day is None or five_min_raw is None:
                        continue
                    if five_min_day.empty or thirty_min_day.empty:
                        continue

                    # Get daily CPR
                    d_cpr = symbol_daily_cpr[sym]
                    if trade_date not in d_cpr.index:
                        applicable = d_cpr[d_cpr.index <= trade_date]
                        if applicable.empty:
                            continue
                        day_cpr_row = applicable.iloc[-1]
                    else:
                        day_cpr_row = d_cpr.loc[trade_date]

                    # Run strategy for this day — bias already established
                    # CPR breach detection happens INSIDE _run_biased_day
                    sym_trades, breached = self._run_biased_day(
                        symbol=sym,
                        trade_date=trade_date,
                        bias=weekly_bias[sym],
                        daily_cpr=day_cpr_row,
                        weekly_cpr=w_cpr_row,
                        five_min_day=five_min_raw,
                        thirty_min_day=thirty_min_day,
                        five_min_with_st=five_min_day,
                        capital=capital,
                        current_positions_count=len(day_trades),
                        is_bias_day=(trade_date == first_day),
                    )
                    day_trades.extend(sym_trades)
                    if breached:
                        bias_cancelled[sym] = True

                # Aggregate day's P&L
                day_pnl_total = sum(t.pnl for t in day_trades)
                date_str = trade_date.strftime('%Y-%m-%d')
                daily_pnl[date_str] = day_pnl_total
                capital += day_pnl_total
                equity_curve[date_str] = capital

                all_trades.extend(day_trades)

                if day_had_setup:
                    days_with_setups += 1
                if len(day_trades) > 0:
                    days_with_trades += 1

        # ---- Calculate aggregate metrics ----
        result = self._calculate_metrics(
            all_trades, daily_pnl, equity_curve, valid_symbols,
            total_days, days_with_setups, days_with_trades,
        )

        self._print_summary(result)

        if regime_skipped_weeks > 0:
            print(f"\nRegime-Filtered Weeks: {regime_skipped_weeks}/{total_weeks} "
                  f"({regime_skipped_weeks/total_weeks*100:.0f}% skipped)")

        return result

    # -----------------------------------------------------------------
    # Metrics Calculation
    # -----------------------------------------------------------------

    def _calculate_metrics(
        self,
        trades: List[IntradayTrade],
        daily_pnl: Dict[str, float],
        equity_curve: Dict[str, float],
        symbols: List[str],
        total_trading_days: int,
        days_with_setups: int,
        days_with_trades: int,
    ) -> CPRBacktestResult:
        """Calculate all performance metrics from trade list."""
        cfg = self.config
        result = CPRBacktestResult(config=cfg)
        result.trades = trades
        result.daily_pnl = daily_pnl
        result.equity_curve = equity_curve
        result.total_trading_days = total_trading_days
        result.days_with_setups = days_with_setups
        result.days_with_trades = days_with_trades

        if not trades:
            return result

        # Basic counts
        result.total_trades = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(trades) * 100 if trades else 0

        # P&L
        result.total_pnl = sum(t.pnl for t in trades)
        result.total_pnl_pct = result.total_pnl / cfg.initial_capital * 100

        result.avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        result.avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

        # Profit factor
        gross_wins = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))
        result.profit_factor = (
            gross_wins / gross_losses if gross_losses > 0 else float('inf')
        )

        # Costs
        result.total_brokerage = sum(t.brokerage for t in trades)
        result.total_slippage = sum(t.slippage_cost for t in trades)

        # Exit reason breakdown
        exit_counts: Dict[str, int] = {}
        for t in trades:
            exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
        result.exit_reason_counts = exit_counts

        # Avg trades per day
        result.avg_trades_per_day = (
            len(trades) / total_trading_days if total_trading_days > 0 else 0
        )

        # Average holding time (minutes)
        holding_times = []
        for t in trades:
            delta = t.exit_date - t.entry_date
            holding_times.append(delta.total_seconds() / 60)
        result.avg_holding_minutes = np.mean(holding_times) if holding_times else 0

        # Consecutive wins/losses
        max_consec_w, max_consec_l = 0, 0
        curr_w, curr_l = 0, 0
        for t in trades:
            if t.pnl > 0:
                curr_w += 1
                curr_l = 0
                max_consec_w = max(max_consec_w, curr_w)
            else:
                curr_l += 1
                curr_w = 0
                max_consec_l = max(max_consec_l, curr_l)
        result.max_consecutive_wins = max_consec_w
        result.max_consecutive_losses = max_consec_l

        # Drawdown from equity curve
        eq_values = list(equity_curve.values())
        if eq_values:
            running_max = eq_values[0]
            max_dd = 0.0
            for val in eq_values:
                if val > running_max:
                    running_max = val
                dd = (running_max - val) / running_max * 100
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown = max_dd

        # Sharpe ratio (annualized, using daily returns)
        daily_returns = list(daily_pnl.values())
        if len(daily_returns) > 1:
            dr_array = np.array(daily_returns)
            mean_daily = np.mean(dr_array)
            std_daily = np.std(dr_array, ddof=1)
            if std_daily > 0:
                result.sharpe_ratio = (
                    mean_daily / std_daily * sqrt(TRADING_DAYS_PER_YEAR)
                )

            # Sortino ratio (need at least 2 downside observations for ddof=1)
            downside = dr_array[dr_array < 0]
            if len(downside) > 1:
                downside_std = np.std(downside, ddof=1)
                if downside_std > 0:
                    result.sortino_ratio = (
                        mean_daily / downside_std * sqrt(TRADING_DAYS_PER_YEAR)
                    )

        # Per-symbol stats
        for sym in symbols:
            sym_trades = [t for t in trades if t.symbol == sym]
            if not sym_trades:
                continue
            sym_wins = [t for t in sym_trades if t.pnl > 0]
            sym_pnl = sum(t.pnl for t in sym_trades)
            result.per_symbol_stats[sym] = {
                'total_trades': len(sym_trades),
                'winning_trades': len(sym_wins),
                'win_rate': len(sym_wins) / len(sym_trades) * 100,
                'total_pnl': sym_pnl,
                'avg_pnl': sym_pnl / len(sym_trades),
                'best_trade': max(t.pnl for t in sym_trades),
                'worst_trade': min(t.pnl for t in sym_trades),
            }

        return result

    # -----------------------------------------------------------------
    # Summary Printing
    # -----------------------------------------------------------------

    def _print_summary(self, result: CPRBacktestResult) -> None:
        """Print a formatted summary of backtest results."""
        cfg = result.config

        print("\n" + "=" * 70)
        print("CPR INTRADAY BACKTEST RESULTS")
        print("=" * 70)

        print(f"\nPeriod: {cfg.start_date} to {cfg.end_date}")
        print(f"Symbols: {len([s for s in cfg.symbols if s in self._daily_data])}")
        print(f"Initial Capital: Rs {cfg.initial_capital:,.0f}")
        print(f"Narrow CPR Threshold: {cfg.narrow_cpr_threshold}%")
        print(f"Max Wick %: {cfg.max_wick_pct}%")
        print(f"CPR Proximity: {cfg.cpr_proximity_pct}%")
        print(f"SuperTrend: ({cfg.st_period}, {cfg.st_multiplier})")

        print(f"\n--- Performance ---")
        final_cap = list(result.equity_curve.values())[-1] if result.equity_curve else cfg.initial_capital
        print(f"Final Capital: Rs {final_cap:,.0f}")
        print(f"Total P&L: Rs {result.total_pnl:,.0f} ({result.total_pnl_pct:.2f}%)")
        print(f"Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {result.sortino_ratio:.2f}")

        print(f"\n--- Trade Statistics ---")
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning: {result.winning_trades} | Losing: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.1f}%")
        print(f"Avg Win: Rs {result.avg_win:,.0f} | Avg Loss: Rs {result.avg_loss:,.0f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Avg Trades/Day: {result.avg_trades_per_day:.2f}")
        print(f"Avg Holding: {result.avg_holding_minutes:.0f} min")
        print(f"Max Consec Wins: {result.max_consecutive_wins} | "
              f"Max Consec Losses: {result.max_consecutive_losses}")

        print(f"\n--- Trading Days ---")
        print(f"Total Trading Days: {result.total_trading_days}")
        print(f"Days With Setups: {result.days_with_setups}")
        print(f"Days With Trades: {result.days_with_trades}")
        # regime info is printed in run() after _print_summary

        print(f"\n--- Costs ---")
        print(f"Total Brokerage: Rs {result.total_brokerage:,.0f}")
        print(f"Total Slippage: Rs {result.total_slippage:,.0f}")

        if result.exit_reason_counts:
            print(f"\n--- Exit Reasons ---")
            for reason, count in sorted(
                result.exit_reason_counts.items(), key=lambda x: -x[1]
            ):
                pct = count / result.total_trades * 100
                print(f"  {reason:20s}: {count:4d} ({pct:.1f}%)")

        if result.per_symbol_stats:
            print(f"\n--- Per-Symbol Breakdown ---")
            print(f"{'Symbol':>12s} {'Trades':>7s} {'WinRate':>8s} {'PnL':>12s} {'AvgPnL':>10s}")
            print("-" * 55)
            for sym, stats in sorted(
                result.per_symbol_stats.items(), key=lambda x: -x[1]['total_pnl']
            ):
                print(
                    f"{sym:>12s} {stats['total_trades']:>7d} "
                    f"{stats['win_rate']:>7.1f}% "
                    f"{stats['total_pnl']:>12,.0f} "
                    f"{stats['avg_pnl']:>10,.0f}"
                )

        print("\n" + "=" * 70)


# =============================================================================
# Convenience Runner
# =============================================================================

def run_cpr_backtest(
    symbols: Optional[List[str]] = None,
    start_date: str = '2024-01-01',
    end_date: str = '2025-10-27',
    initial_capital: float = 1_000_000,
    narrow_cpr_threshold: float = 1.0,
    max_wick_pct: float = 20.0,
    cpr_proximity_pct: float = 1.0,
    st_period: int = 7,
    st_multiplier: float = 3.0,
    **kwargs,
) -> CPRBacktestResult:
    """
    Convenience function to run a CPR intraday backtest.

    Usage:
        result = run_cpr_backtest(
            symbols=['RELIANCE', 'HDFCBANK'],
            start_date='2024-01-01',
            end_date='2025-10-27',
        )
    """
    if symbols is None:
        symbols = INTRADAY_SYMBOLS.copy()

    config = CPRIntradayConfig(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        narrow_cpr_threshold=narrow_cpr_threshold,
        max_wick_pct=max_wick_pct,
        cpr_proximity_pct=cpr_proximity_pct,
        st_period=st_period,
        st_multiplier=st_multiplier,
        **kwargs,
    )

    # Preload data
    print("Preloading data...", flush=True)
    daily_data, five_min_data = CPRIntradayEngine.preload_data(
        symbols, start_date, end_date
    )
    print(
        f"Data loaded: {len(daily_data)} daily, {len(five_min_data)} 5-min",
        flush=True,
    )

    engine = CPRIntradayEngine(
        config,
        preloaded_daily=daily_data,
        preloaded_5min=five_min_data,
    )
    return engine.run()


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    import logging
    logging.disable(logging.WARNING)

    result = run_cpr_backtest()
    print(f"\nBacktest complete. {result.total_trades} trades generated.")
