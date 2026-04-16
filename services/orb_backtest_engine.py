"""
ORB (Opening Range Breakout) Backtest Engine
=============================================

Intraday backtest engine for the Opening Range Breakout strategy on NSE stocks.

Data source: market_data.db (5-min and daily OHLCV)
Supported symbols: Any stock with 5-min data in market_data_unified table.

Flow:
1. Preload all 5-min and daily data into memory
2. For each trading day:
   a. Build Opening Range from first N minutes
   b. Compute daily filters (CPR, pivot S/R, inside day, NR4, MTF EMA, BB)
   c. Walk 5-min bars after OR: detect breakouts, apply filters, manage trades
   d. EOD squareoff at configurable time
3. Collect all trades, compute performance stats
"""

import sqlite3
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ORBConfig:
    """All configurable parameters for the ORB backtest."""

    # Core ORB
    or_minutes: int = 15              # 15, 30, 60
    last_entry_time: str = '14:00'
    eod_exit_time: str = '15:20'
    max_trades_per_day: int = 1       # 1, 2, 3
    allow_longs: bool = True
    allow_shorts: bool = True

    # Stop Loss
    sl_type: str = 'or_opposite'      # 'or_opposite', 'fixed_pct', 'atr_multiple'
    fixed_sl_pct: float = 1.0         # % from entry
    atr_sl_multiple: float = 1.5
    atr_period: int = 14

    # Target
    target_type: str = 'r_multiple'   # 'r_multiple', 'fixed_pct', 'or_range_multiple'
    r_multiple: float = 1.5
    fixed_target_pct: float = 2.0
    or_range_multiple: float = 1.0

    # Filter: VWAP
    use_vwap_filter: bool = False

    # Filter: RSI (higher TF)
    use_rsi_filter: bool = False
    rsi_period: int = 14
    rsi_timeframe: str = '15min'
    rsi_long_threshold: float = 60.0
    rsi_short_threshold: float = 40.0

    # Filter: CPR Direction
    use_cpr_dir_filter: bool = False

    # Filter: CPR Width
    use_cpr_width_filter: bool = False
    cpr_width_threshold_pct: float = 0.5

    # Filter: Virgin CPR
    use_virgin_cpr_filter: bool = False

    # Filter: Previous Day CPR position
    use_prev_cpr_filter: bool = False

    # Filter: Previous Day High/Low Break
    use_prev_hl_filter: bool = False

    # Filter: Pivot S/R proximity
    use_pivot_sr_filter: bool = False
    pivot_sr_buffer_pct: float = 0.2

    # Filter: Bollinger Bands
    use_bb_filter: bool = False
    bb_period: int = 20
    bb_std: float = 2.0

    # Filter: Multi-timeframe trend (daily EMA)
    use_mtf_filter: bool = False
    mtf_ema_period: int = 20

    # Filter: Inside Day
    use_inside_day_filter: bool = False

    # Filter: Narrow Range (NR4)
    use_narrow_range_filter: bool = False
    nr_lookback: int = 4

    def label(self) -> str:
        """Generate a short config label for CSV output."""
        parts = [f'OR{self.or_minutes}']
        parts.append(f'SL_{self.sl_type}')
        parts.append(f'TGT_{self.target_type}')
        if self.target_type == 'r_multiple':
            parts.append(f'R{self.r_multiple}')
        filters = []
        if self.use_vwap_filter: filters.append('VWAP')
        if self.use_rsi_filter: filters.append('RSI')
        if self.use_cpr_dir_filter: filters.append('CPRd')
        if self.use_cpr_width_filter: filters.append('CPRw')
        if self.use_virgin_cpr_filter: filters.append('VCPR')
        if self.use_prev_cpr_filter: filters.append('PCPR')
        if self.use_prev_hl_filter: filters.append('PHL')
        if self.use_pivot_sr_filter: filters.append('PVT')
        if self.use_bb_filter: filters.append('BB')
        if self.use_mtf_filter: filters.append('MTF')
        if self.use_inside_day_filter: filters.append('ID')
        if self.use_narrow_range_filter: filters.append(f'NR{self.nr_lookback}')
        if filters:
            parts.append('_'.join(filters))
        return '_'.join(parts)


# =============================================================================
# Backtest Result
# =============================================================================

@dataclass
class BacktestResult:
    """Complete ORB backtest output for one symbol."""
    symbol: str
    config_label: str
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    total_pnl_pts: float
    avg_win_pts: float
    avg_loss_pts: float
    profit_factor: float
    max_drawdown_pts: float
    avg_trade_pts: float
    long_trades: int
    short_trades: int
    long_win_rate: float
    short_win_rate: float
    target_exits: int
    sl_exits: int
    eod_exits: int
    avg_holding_bars: float
    sharpe: float
    trades: list = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary for quick review."""
        return (
            f"{self.symbol} | {self.config_label} | "
            f"Trades={self.total_trades} WR={self.win_rate:.1f}% "
            f"PF={self.profit_factor:.2f} PnL={self.total_pnl_pts:.1f}pts "
            f"Sharpe={self.sharpe:.2f} MaxDD={self.max_drawdown_pts:.1f}pts"
        )


# =============================================================================
# Helper: Indicator computations (vectorized)
# =============================================================================

def _calc_rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI with proper seeding."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range on OHLC dataframe."""
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def _calc_bollinger(series: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: returns (upper, middle, lower)."""
    middle = series.rolling(period).mean()
    rolling_std = series.rolling(period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return upper, middle, lower


def _resample_5min_to_15min(df_5min: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 5-min OHLCV to 15-min.
    Groups bars into 15-min windows aligned to session start (09:15).
    """
    df = df_5min.copy()
    df = df.set_index('date')
    resampled = df.resample('15min', origin='start_day', offset='15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(subset=['open'])
    resampled = resampled.reset_index()
    return resampled


# =============================================================================
# Engine
# =============================================================================

class ORBBacktestEngine:
    """
    Opening Range Breakout backtest engine.

    Usage:
        config = ORBConfig(or_minutes=15, r_multiple=2.0)
        data = ORBBacktestEngine.preload_data(['ADANIENT', 'TATASTEEL'])
        engine = ORBBacktestEngine(config)
        result = engine.run('ADANIENT', data['ADANIENT'])
    """

    def __init__(self, config: ORBConfig):
        self.config = config
        self._last_entry_time = self._parse_time(config.last_entry_time)
        self._eod_exit_time = self._parse_time(config.eod_exit_time)
        # OR end time: session open + or_minutes
        or_end_minutes = 9 * 60 + 15 + config.or_minutes
        self._or_end_time = time(or_end_minutes // 60, or_end_minutes % 60)

    @staticmethod
    def _parse_time(t_str: str) -> time:
        parts = t_str.split(':')
        return time(int(parts[0]), int(parts[1]))

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    @staticmethod
    def preload_data(
        symbols: List[str],
        db_path: Optional[Path] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load all 5-min and daily data for given symbols.

        Returns:
            dict[symbol] = {'5min': DataFrame, 'day': DataFrame}
            Each DataFrame has columns: date, open, high, low, close, volume
            date column is pandas Timestamp.
        """
        db = db_path or DB_PATH
        conn = sqlite3.connect(str(db))

        result = {}
        for sym in symbols:
            sym_data = {}

            # 5-min data
            query_5m = (
                "SELECT date, open, high, low, close, volume "
                "FROM market_data_unified "
                "WHERE symbol = ? AND timeframe = '5minute' "
                "ORDER BY date"
            )
            df_5m = pd.read_sql_query(query_5m, conn, params=(sym,))
            df_5m['date'] = pd.to_datetime(df_5m['date'])
            sym_data['5min'] = df_5m

            # Daily data
            query_day = (
                "SELECT date, open, high, low, close, volume "
                "FROM market_data_unified "
                "WHERE symbol = ? AND timeframe = 'day' "
                "ORDER BY date"
            )
            df_day = pd.read_sql_query(query_day, conn, params=(sym,))
            df_day['date'] = pd.to_datetime(df_day['date'])
            sym_data['day'] = df_day

            result[sym] = sym_data
            logger.info(f"Loaded {sym}: {len(df_5m)} 5-min bars, {len(df_day)} daily bars")

        conn.close()
        return result

    # -------------------------------------------------------------------------
    # Pre-compute daily-level indicators (vectorized)
    # -------------------------------------------------------------------------

    def _prepare_daily_indicators(self, df_day: pd.DataFrame) -> pd.DataFrame:
        """
        Add all daily-level indicator columns to the daily dataframe.
        Returns a copy with extra columns.
        """
        df = df_day.copy()

        # CPR: computed from PREVIOUS day's HLC
        # Pivot = (H + L + C) / 3, BC = (H + L) / 2, TC = Pivot + (Pivot - BC)
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3.0
        df['bc'] = (df['high'] + df['low']) / 2.0
        df['tc'] = df['pivot'] + (df['pivot'] - df['bc'])
        df['cpr_width'] = (df['tc'] - df['bc']).abs()
        df['cpr_width_pct'] = df['cpr_width'] / df['pivot'] * 100.0

        # Pivot support/resistance
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])

        # Daily range
        df['day_range'] = df['high'] - df['low']

        # Inside day: today's range < yesterday's range
        df['is_inside_day'] = df['day_range'] < df['day_range'].shift(1)

        # NR4: today's range is narrowest in last N days
        if self.config.nr_lookback > 1:
            df['range_min_n'] = df['day_range'].rolling(
                self.config.nr_lookback
            ).min()
            df['is_narrow_range'] = df['day_range'] == df['range_min_n']
        else:
            df['is_narrow_range'] = True

        # Bollinger Bands on daily close
        if self.config.use_bb_filter:
            _, bb_mid, _ = _calc_bollinger(
                df['close'], self.config.bb_period, self.config.bb_std
            )
            df['bb_middle'] = bb_mid
        else:
            df['bb_middle'] = np.nan

        # Daily EMA for MTF filter
        if self.config.use_mtf_filter:
            df['daily_ema'] = _calc_ema(df['close'], self.config.mtf_ema_period)
        else:
            df['daily_ema'] = np.nan

        # ATR for ATR-based SL
        if self.config.sl_type == 'atr_multiple':
            df['atr'] = _calc_atr(df, self.config.atr_period)
        else:
            df['atr'] = np.nan

        # Previous day close vs CPR for prev_cpr_filter
        # Bullish if prev close > prev TC, Bearish if prev close < prev BC
        df['prev_close'] = df['close'].shift(1)
        df['prev_tc'] = df['tc'].shift(1)
        df['prev_bc'] = df['bc'].shift(1)
        df['prev_cpr_bullish'] = df['prev_close'] > df['prev_tc']
        df['prev_cpr_bearish'] = df['prev_close'] < df['prev_bc']

        # Previous day high/low
        df['prev_day_high'] = df['high'].shift(1)
        df['prev_day_low'] = df['low'].shift(1)

        return df

    def _build_daily_lookup(self, df_day: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Build a dict mapping trade_date_str -> daily row (from PREVIOUS trading day).
        For a given trading date, the CPR/pivot/filter values come from the
        most recent daily bar BEFORE that date.
        """
        df = self._prepare_daily_indicators(df_day)
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

        # Create lookup: for each date, store that row's indicator values
        # When we look up a trading date, we find the most recent daily bar < that date
        lookup = {}
        date_strs = df['date_str'].tolist()
        for i in range(1, len(df)):
            # Row i has indicators computed from row i's own OHLC,
            # but CPR/pivot from row i should be used for the NEXT trading day.
            # So we map "next trading day after row i" to row i.
            # Instead, build a sorted list and do bisect lookup.
            pass

        # Simpler: return the prepared df, and do date-based lookup at runtime
        return df

    # -------------------------------------------------------------------------
    # Main backtest loop
    # -------------------------------------------------------------------------

    def run(self, symbol: str, data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Run full ORB backtest on one symbol.

        Args:
            symbol: Stock symbol
            data: {'5min': DataFrame, 'day': DataFrame}

        Returns:
            BacktestResult with all stats and trade list
        """
        cfg = self.config
        df_5min = data['5min'].copy()
        df_day = data['day'].copy()

        if df_5min.empty:
            return self._empty_result(symbol)

        # Prepare daily indicators
        daily_df = self._prepare_daily_indicators(df_day)
        daily_dates = daily_df['date'].dt.normalize().values  # numpy datetime64 array
        daily_rows = daily_df.to_dict('records')

        # Prepare 15-min RSI if needed
        rsi_15min_series = None
        if cfg.use_rsi_filter:
            df_15min = _resample_5min_to_15min(df_5min)
            rsi_15min_series = _calc_rsi_wilder(df_15min['close'], cfg.rsi_period)
            df_15min['rsi'] = rsi_15min_series
            # Build lookup: timestamp -> RSI value
            rsi_lookup = {}
            for idx, row in df_15min.iterrows():
                rsi_lookup[row['date']] = row['rsi']
            self._rsi_lookup = rsi_lookup
            self._rsi_15min_df = df_15min

        # Group 5-min bars by trading day
        df_5min['trade_date'] = df_5min['date'].dt.normalize()
        trading_days = df_5min.groupby('trade_date')

        all_trades: List[dict] = []
        daily_pnls: List[float] = []

        for trade_date, day_bars in trading_days:
            day_bars = day_bars.sort_values('date').reset_index(drop=True)
            trade_date_np = pd.Timestamp(trade_date).to_datetime64()

            # Find the most recent daily bar BEFORE this trade date
            prev_daily_idx = np.searchsorted(daily_dates, trade_date_np, side='left') - 1
            if prev_daily_idx < 0:
                # No previous daily data available, skip this day
                daily_pnls.append(0.0)
                continue

            prev_daily = daily_rows[prev_daily_idx]

            # For filters that need 2-days-ago data
            prev2_daily = daily_rows[prev_daily_idx - 1] if prev_daily_idx >= 1 else None

            # Run the intraday simulation for this day
            day_trades = self._simulate_day(
                symbol, day_bars, prev_daily, prev2_daily, trade_date
            )

            day_pnl = sum(t['pnl_pts'] for t in day_trades)
            daily_pnls.append(day_pnl)
            all_trades.extend(day_trades)

        return self._compute_result(symbol, all_trades, daily_pnls)

    # -------------------------------------------------------------------------
    # Single day simulation
    # -------------------------------------------------------------------------

    def _simulate_day(
        self,
        symbol: str,
        day_bars: pd.DataFrame,
        prev_daily: dict,
        prev2_daily: Optional[dict],
        trade_date: pd.Timestamp
    ) -> List[dict]:
        """
        Simulate one trading day. Walk through 5-min bars, build OR,
        detect breakouts, manage positions.

        Returns list of trade dicts for this day.
        """
        cfg = self.config
        bars = day_bars.to_dict('records')
        n_bars = len(bars)

        if n_bars < 3:
            return []

        # ---- Daily-level filter pre-checks ----

        # CPR values from previous day (to be used as today's levels)
        pivot = prev_daily['pivot']
        tc = prev_daily['tc']
        bc = prev_daily['bc']
        cpr_width_pct = prev_daily['cpr_width_pct']
        r1 = prev_daily['r1']
        s1 = prev_daily['s1']
        r2 = prev_daily['r2']
        s2 = prev_daily['s2']

        # CPR width filter: skip wide CPR days
        if cfg.use_cpr_width_filter and cpr_width_pct > cfg.cpr_width_threshold_pct:
            return []

        # Inside day filter: only trade if YESTERDAY was inside day
        if cfg.use_inside_day_filter:
            if prev2_daily is None:
                return []
            prev_range = prev_daily['high'] - prev_daily['low']
            prev2_range = prev2_daily['high'] - prev2_daily['low']
            if prev_range >= prev2_range:
                return []

        # Narrow range filter: only trade if yesterday was NR(N)
        if cfg.use_narrow_range_filter:
            if not prev_daily.get('is_narrow_range', False):
                return []

        # Prev CPR filter
        allow_long_prev_cpr = True
        allow_short_prev_cpr = True
        if cfg.use_prev_cpr_filter:
            allow_long_prev_cpr = prev_daily.get('prev_cpr_bullish', True)
            allow_short_prev_cpr = prev_daily.get('prev_cpr_bearish', True)

        # Prev day high/low for filter
        prev_day_high = prev_daily['high']
        prev_day_low = prev_daily['low']

        # Bollinger Bands middle from previous daily bar
        bb_middle = prev_daily.get('bb_middle', np.nan)

        # Daily EMA from previous daily bar
        daily_ema = prev_daily.get('daily_ema', np.nan)

        # ATR from previous daily bar (for ATR-based SL)
        daily_atr = prev_daily.get('atr', np.nan)

        # ---- Build Opening Range ----
        or_high = -np.inf
        or_low = np.inf
        or_bar_count = cfg.or_minutes // 5  # number of 5-min bars in OR

        if n_bars < or_bar_count:
            return []

        for i in range(or_bar_count):
            bar = bars[i]
            or_high = max(or_high, bar['high'])
            or_low = min(or_low, bar['low'])

        or_range = or_high - or_low
        if or_range <= 0:
            return []

        # ---- Walk bars after OR ----
        trades: List[dict] = []
        trades_today = 0
        position = None  # dict with entry details
        prev_close = bars[or_bar_count - 1]['close']  # last OR bar close

        # VWAP accumulators (from session start)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        for i in range(or_bar_count):
            b = bars[i]
            tp = (b['high'] + b['low'] + b['close']) / 3.0
            cum_tp_vol += tp * b['volume']
            cum_vol += b['volume']

        # Virgin CPR tracker
        cpr_virgin = True
        # Check OR bars against CPR
        cpr_top = max(tc, bc)
        cpr_bot = min(tc, bc)
        for i in range(or_bar_count):
            b = bars[i]
            if b['low'] <= cpr_top and b['high'] >= cpr_bot:
                cpr_virgin = False
                break

        for i in range(or_bar_count, n_bars):
            bar = bars[i]
            bar_time = bar['date'].time() if isinstance(bar['date'], datetime) else bar['date'].to_pydatetime().time()
            bar_close = bar['close']
            bar_high = bar['high']
            bar_low = bar['low']

            # Update VWAP
            tp = (bar_high + bar_low + bar_close) / 3.0
            cum_tp_vol += tp * bar['volume']
            cum_vol += bar['volume']
            vwap = cum_tp_vol / cum_vol if cum_vol > 0 else bar_close

            # Update virgin CPR
            if cpr_virgin and bar_low <= cpr_top and bar_high >= cpr_bot:
                cpr_virgin = False

            # ---- Position management (check exits FIRST) ----
            if position is not None:
                exit_price = None
                exit_reason = None
                direction = position['direction']

                # Check SL hit (using bar high/low for intrabar stop)
                if direction == 'long':
                    if bar_low <= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'sl'
                    elif bar_high >= position['target']:
                        exit_price = position['target']
                        exit_reason = 'target'
                else:  # short
                    if bar_high >= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'sl'
                    elif bar_low <= position['target']:
                        exit_price = position['target']
                        exit_reason = 'target'

                # EOD exit
                if exit_price is None and bar_time >= self._eod_exit_time:
                    exit_price = bar_close
                    exit_reason = 'eod'

                if exit_price is not None:
                    pnl = (exit_price - position['entry_price']) if direction == 'long' else (position['entry_price'] - exit_price)
                    holding_bars = i - position['entry_bar_idx']
                    trade_rec = {
                        'symbol': symbol,
                        'date': trade_date.strftime('%Y-%m-%d') if hasattr(trade_date, 'strftime') else str(trade_date)[:10],
                        'direction': direction,
                        'entry_time': position['entry_time'],
                        'entry_price': position['entry_price'],
                        'sl': position['sl'],
                        'target': position['target'],
                        'exit_time': str(bar_time),
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_pts': round(pnl, 2),
                        'or_high': or_high,
                        'or_low': or_low,
                        'or_range': or_range,
                        'holding_bars': holding_bars,
                    }
                    trades.append(trade_rec)
                    position = None

                # If still in position, update prev_close and continue
                if position is not None:
                    prev_close = bar_close
                    continue

            # ---- No position: look for breakout entries ----
            if position is not None:
                prev_close = bar_close
                continue

            # Max trades per day check
            if trades_today >= cfg.max_trades_per_day:
                prev_close = bar_close
                continue

            # Last entry time check
            if bar_time >= self._last_entry_time:
                prev_close = bar_close
                continue

            # EOD exit time check (don't enter if already at exit time)
            if bar_time >= self._eod_exit_time:
                prev_close = bar_close
                continue

            # Detect breakout
            long_breakout = (bar_close > or_high and prev_close <= or_high)
            short_breakout = (bar_close < or_low and prev_close >= or_low)

            if not long_breakout and not short_breakout:
                prev_close = bar_close
                continue

            # Determine direction
            if long_breakout and short_breakout:
                # Extremely rare: both triggered on same bar, skip
                prev_close = bar_close
                continue

            direction = 'long' if long_breakout else 'short'

            # Direction permission
            if direction == 'long' and not cfg.allow_longs:
                prev_close = bar_close
                continue
            if direction == 'short' and not cfg.allow_shorts:
                prev_close = bar_close
                continue

            entry_price = bar_close

            # ---- Apply filters ----
            if not self._pass_filters(
                cfg, direction, entry_price, vwap, bar, trade_date,
                pivot, tc, bc, r1, s1, r2, s2, cpr_virgin,
                allow_long_prev_cpr, allow_short_prev_cpr,
                prev_day_high, prev_day_low,
                bb_middle, daily_ema, daily_atr
            ):
                prev_close = bar_close
                continue

            # ---- Compute SL and Target ----
            sl, target = self._compute_sl_target(
                cfg, direction, entry_price, or_high, or_low, or_range, daily_atr
            )

            position = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_time': str(bar_time),
                'entry_bar_idx': i,
                'sl': sl,
                'target': target,
            }
            trades_today += 1
            prev_close = bar_close

        # End of day: force close any open position at last bar
        if position is not None:
            last_bar = bars[-1]
            direction = position['direction']
            exit_price = last_bar['close']
            pnl = (exit_price - position['entry_price']) if direction == 'long' else (position['entry_price'] - exit_price)
            holding_bars = (n_bars - 1) - position['entry_bar_idx']
            trade_rec = {
                'symbol': symbol,
                'date': trade_date.strftime('%Y-%m-%d') if hasattr(trade_date, 'strftime') else str(trade_date)[:10],
                'direction': direction,
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'sl': position['sl'],
                'target': position['target'],
                'exit_time': str(last_bar['date'].time() if isinstance(last_bar['date'], datetime) else last_bar['date'].to_pydatetime().time()),
                'exit_price': exit_price,
                'exit_reason': 'eod',
                'pnl_pts': round(pnl, 2),
                'or_high': or_high,
                'or_low': or_low,
                'or_range': or_range,
                'holding_bars': holding_bars,
            }
            trades.append(trade_rec)

        return trades

    # -------------------------------------------------------------------------
    # Filter evaluation
    # -------------------------------------------------------------------------

    def _pass_filters(
        self,
        cfg: ORBConfig,
        direction: str,
        entry_price: float,
        vwap: float,
        bar: dict,
        trade_date: pd.Timestamp,
        pivot: float, tc: float, bc: float,
        r1: float, s1: float, r2: float, s2: float,
        cpr_virgin: bool,
        allow_long_prev_cpr: bool, allow_short_prev_cpr: bool,
        prev_day_high: float, prev_day_low: float,
        bb_middle: float, daily_ema: float, daily_atr: float
    ) -> bool:
        """Evaluate all enabled filters. Returns True if trade passes ALL filters."""

        is_long = (direction == 'long')

        # VWAP filter: long above VWAP, short below VWAP
        if cfg.use_vwap_filter:
            if is_long and entry_price < vwap:
                return False
            if not is_long and entry_price > vwap:
                return False

        # RSI filter (15-min timeframe)
        if cfg.use_rsi_filter:
            rsi_val = self._get_rsi_at_time(bar['date'])
            if rsi_val is not None and not np.isnan(rsi_val):
                if is_long and rsi_val < cfg.rsi_long_threshold:
                    return False
                if not is_long and rsi_val > cfg.rsi_short_threshold:
                    return False

        # CPR direction filter: long above TC, short below BC
        if cfg.use_cpr_dir_filter:
            if is_long and entry_price < tc:
                return False
            if not is_long and entry_price > bc:
                return False

        # Virgin CPR filter: only trade on virgin CPR days
        if cfg.use_virgin_cpr_filter:
            if not cpr_virgin:
                return False

        # Prev CPR filter
        if cfg.use_prev_cpr_filter:
            if is_long and not allow_long_prev_cpr:
                return False
            if not is_long and not allow_short_prev_cpr:
                return False

        # Prev day high/low break filter
        if cfg.use_prev_hl_filter:
            if is_long and entry_price <= prev_day_high:
                return False
            if not is_long and entry_price >= prev_day_low:
                return False

        # Pivot S/R proximity filter
        if cfg.use_pivot_sr_filter:
            buffer = cfg.pivot_sr_buffer_pct / 100.0 * entry_price
            levels = [s1, s2, r1, r2, pivot]
            for lvl in levels:
                if abs(entry_price - lvl) < buffer:
                    return False

        # Bollinger Bands filter: long above middle, short below middle
        if cfg.use_bb_filter and not np.isnan(bb_middle):
            if is_long and entry_price < bb_middle:
                return False
            if not is_long and entry_price > bb_middle:
                return False

        # MTF filter: long above daily EMA, short below daily EMA
        if cfg.use_mtf_filter and not np.isnan(daily_ema):
            if is_long and entry_price < daily_ema:
                return False
            if not is_long and entry_price > daily_ema:
                return False

        return True

    def _get_rsi_at_time(self, bar_date) -> Optional[float]:
        """Get the most recent 15-min RSI value at or before bar_date."""
        if not hasattr(self, '_rsi_15min_df') or self._rsi_15min_df is None:
            return None

        ts = pd.Timestamp(bar_date)
        # Find the most recent 15-min bar at or before this timestamp
        mask = self._rsi_15min_df['date'] <= ts
        if not mask.any():
            return None
        idx = self._rsi_15min_df.loc[mask].index[-1]
        return self._rsi_15min_df.at[idx, 'rsi']

    # -------------------------------------------------------------------------
    # SL / Target computation
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_sl_target(
        cfg: ORBConfig,
        direction: str,
        entry_price: float,
        or_high: float,
        or_low: float,
        or_range: float,
        daily_atr: float
    ) -> Tuple[float, float]:
        """
        Compute stop loss and target price for a trade.

        Returns:
            (sl_price, target_price)
        """
        is_long = (direction == 'long')

        # ---- Stop Loss ----
        if cfg.sl_type == 'or_opposite':
            sl = or_low if is_long else or_high
        elif cfg.sl_type == 'fixed_pct':
            offset = entry_price * cfg.fixed_sl_pct / 100.0
            sl = entry_price - offset if is_long else entry_price + offset
        elif cfg.sl_type == 'atr_multiple':
            if np.isnan(daily_atr) or daily_atr <= 0:
                # Fallback to OR opposite
                sl = or_low if is_long else or_high
            else:
                offset = daily_atr * cfg.atr_sl_multiple
                sl = entry_price - offset if is_long else entry_price + offset
        else:
            sl = or_low if is_long else or_high

        # Risk = distance from entry to SL
        risk = abs(entry_price - sl)
        if risk <= 0:
            risk = or_range * 0.5  # fallback

        # ---- Target ----
        if cfg.target_type == 'r_multiple':
            reward = risk * cfg.r_multiple
            target = entry_price + reward if is_long else entry_price - reward
        elif cfg.target_type == 'fixed_pct':
            offset = entry_price * cfg.fixed_target_pct / 100.0
            target = entry_price + offset if is_long else entry_price - offset
        elif cfg.target_type == 'or_range_multiple':
            offset = or_range * cfg.or_range_multiple
            target = entry_price + offset if is_long else entry_price - offset
        else:
            target = entry_price + risk * 1.5 if is_long else entry_price - risk * 1.5

        return round(sl, 2), round(target, 2)

    # -------------------------------------------------------------------------
    # Result computation
    # -------------------------------------------------------------------------

    def _compute_result(
        self, symbol: str, trades: List[dict], daily_pnls: List[float]
    ) -> BacktestResult:
        """Aggregate all trades into a BacktestResult."""

        if not trades:
            return self._empty_result(symbol)

        total = len(trades)
        pnls = [t['pnl_pts'] for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        n_win = len(winners)
        n_loss = len(losers)
        win_rate = n_win / total * 100.0 if total > 0 else 0.0

        total_pnl = sum(pnls)
        avg_win = np.mean(winners) if winners else 0.0
        avg_loss = np.mean(losers) if losers else 0.0

        gross_profit = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            float('inf') if gross_profit > 0 else 0.0
        )

        # Max drawdown in points (cumulative PnL drawdown)
        cum_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        avg_trade = np.mean(pnls) if pnls else 0.0

        # Direction breakdown
        long_trades = [t for t in trades if t['direction'] == 'long']
        short_trades = [t for t in trades if t['direction'] == 'short']
        n_long = len(long_trades)
        n_short = len(short_trades)
        long_wins = sum(1 for t in long_trades if t['pnl_pts'] > 0)
        short_wins = sum(1 for t in short_trades if t['pnl_pts'] > 0)
        long_wr = long_wins / n_long * 100.0 if n_long > 0 else 0.0
        short_wr = short_wins / n_short * 100.0 if n_short > 0 else 0.0

        # Exit type counts
        target_exits = sum(1 for t in trades if t['exit_reason'] == 'target')
        sl_exits = sum(1 for t in trades if t['exit_reason'] == 'sl')
        eod_exits = sum(1 for t in trades if t['exit_reason'] == 'eod')

        # Average holding bars
        avg_bars = np.mean([t['holding_bars'] for t in trades]) if trades else 0.0

        # Sharpe ratio on daily PnLs
        daily_arr = np.array(daily_pnls)
        if len(daily_arr) > 1 and np.std(daily_arr) > 0:
            sharpe = (np.mean(daily_arr) / np.std(daily_arr)) * np.sqrt(252)
        else:
            sharpe = 0.0

        return BacktestResult(
            symbol=symbol,
            config_label=self.config.label(),
            total_trades=total,
            winners=n_win,
            losers=n_loss,
            win_rate=round(win_rate, 2),
            total_pnl_pts=round(total_pnl, 2),
            avg_win_pts=round(float(avg_win), 2),
            avg_loss_pts=round(float(avg_loss), 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pts=round(max_dd, 2),
            avg_trade_pts=round(float(avg_trade), 2),
            long_trades=n_long,
            short_trades=n_short,
            long_win_rate=round(long_wr, 2),
            short_win_rate=round(short_wr, 2),
            target_exits=target_exits,
            sl_exits=sl_exits,
            eod_exits=eod_exits,
            avg_holding_bars=round(float(avg_bars), 1),
            sharpe=round(sharpe, 2),
            trades=trades,
        )

    def _empty_result(self, symbol: str) -> BacktestResult:
        """Return a zeroed-out result when no trades are generated."""
        return BacktestResult(
            symbol=symbol,
            config_label=self.config.label(),
            total_trades=0,
            winners=0,
            losers=0,
            win_rate=0.0,
            total_pnl_pts=0.0,
            avg_win_pts=0.0,
            avg_loss_pts=0.0,
            profit_factor=0.0,
            max_drawdown_pts=0.0,
            avg_trade_pts=0.0,
            long_trades=0,
            short_trades=0,
            long_win_rate=0.0,
            short_win_rate=0.0,
            target_exits=0,
            sl_exits=0,
            eod_exits=0,
            avg_holding_bars=0.0,
            sharpe=0.0,
            trades=[],
        )


# =============================================================================
# Convenience: run all symbols with one call
# =============================================================================

def run_orb_backtest(
    symbols: List[str],
    config: Optional[ORBConfig] = None,
    db_path: Optional[Path] = None
) -> Dict[str, BacktestResult]:
    """
    Run ORB backtest across multiple symbols.

    Args:
        symbols: List of stock symbols
        config: ORBConfig (uses defaults if None)
        db_path: Override database path

    Returns:
        dict[symbol] = BacktestResult
    """
    if config is None:
        config = ORBConfig()

    data = ORBBacktestEngine.preload_data(symbols, db_path)
    engine = ORBBacktestEngine(config)

    results = {}
    for sym in symbols:
        if sym in data:
            result = engine.run(sym, data[sym])
            results[sym] = result
            print(result.summary())
        else:
            print(f"WARNING: No data for {sym}, skipping")

    return results
