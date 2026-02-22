"""
IPO Launch Strategy Backtester

Strategy: Buy recently-listed stocks when price breaches the initial ATH
(high of first N days) for the first time, with volume confirmation.
Ride with configurable exit (supertrend, trailing SL, fixed target, time-based).

Indian market focus: cash-settled, no lot size constraints for equity delivery.
"""
import sqlite3
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from services.technical_indicators import (
    calc_supertrend, calc_atr, calc_ema, calc_rsi, calc_adx, calc_mfi,
    calc_stochastics, calc_macd, calc_bollinger_bands
)

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'


@dataclass
class IPOConfig:
    """Configuration for IPO strategy backtest."""
    # Universe filter
    min_ipo_age_days: int = 10       # Min days of trading before eligible
    max_ipo_age_days: int = 50       # Max days — only trade early-stage IPOs
    ipo_start_year: int = 2015       # Only consider stocks listed after this year

    # Entry signal
    ath_lookback_days: int = 5       # ATH = high of first N days
    vol_multiplier: float = 1.5      # Volume must be >= X times 20-day avg
    vol_avg_period: int = 20         # Period for volume moving average
    ignore_day1_volume: bool = True  # Ignore IPO listing day volume in avg

    # Exit strategy: 'supertrend', 'trailing_sl', 'fixed_target', 'time_exit',
    #                'ema_cross', 'atr_trail', 'chandelier'
    exit_strategy: str = 'supertrend'

    # Supertrend exit params
    st_atr_period: int = 10
    st_multiplier: float = 3.0

    # Trailing SL params
    trail_pct: float = 15.0          # Trail stop at X% below peak

    # Fixed target + SL params
    target_pct: float = 30.0         # Take profit at X%
    stop_loss_pct: float = 10.0      # Stop loss at X%

    # Time exit params
    max_hold_days: int = 30          # Exit after N days

    # EMA cross exit params
    ema_fast: int = 5
    ema_slow: int = 20

    # ATR trailing stop params
    atr_trail_period: int = 14
    atr_trail_mult: float = 2.0

    # Chandelier exit params
    chandelier_period: int = 22
    chandelier_mult: float = 3.0

    # --- ENTRY QUALITY FILTERS ---
    # RSI filter: only enter when RSI > threshold (momentum confirmation)
    use_rsi_filter: bool = False
    rsi_min: float = 50.0           # Min RSI to allow entry
    rsi_period: int = 14

    # EMA trend filter: close must be above EMA
    use_ema_filter: bool = False
    ema_filter_period: int = 10     # Close > EMA(N)

    # ADX trend strength filter
    use_adx_filter: bool = False
    adx_min: float = 20.0           # Min ADX for trending market
    adx_period: int = 14

    # MFI (volume-weighted RSI) filter
    use_mfi_filter: bool = False
    mfi_min: float = 50.0
    mfi_period: int = 14

    # Minimum breakout strength: close must be X% above ATH
    min_breakout_pct: float = 0.0   # e.g. 2.0 means close > ATH * 1.02

    # Gap-up filter: open > previous close (positive sentiment)
    require_gap_up: bool = False

    # Listing gain filter: IPO listed positive (day 1 close > day 1 open)
    require_listing_gain: bool = False

    # Consecutive close: require N bars closing above ATH (confirmation)
    require_consec_close: int = 1   # 1 = just today, 2 = today + yesterday

    # Price above VWAP-like: close > avg price since listing
    require_above_avg_price: bool = False

    # --- HYBRID EXIT STRATEGIES ---
    # Breakeven stop: after +X%, move SL to entry price
    use_breakeven_stop: bool = False
    breakeven_trigger_pct: float = 5.0  # Move SL to entry after this gain

    # Trail tightening: after N days, tighten trail %
    use_time_tightening: bool = False
    tighten_after_days: int = 20
    tightened_trail_pct: float = 8.0

    # Hybrid exit modes: combine target + trail
    # 'none', 'target_or_trail', 'target_with_breakeven'
    hybrid_exit: str = 'none'
    hybrid_target_pct: float = 15.0
    hybrid_trail_pct: float = 10.0
    hybrid_sl_pct: float = 7.0

    # Position sizing
    initial_capital: float = 10_000_000  # Rs.1 Crore
    position_size_pct: float = 5.0       # 5% of capital per trade
    max_positions: int = 10              # Max concurrent positions


@dataclass
class IPOTrade:
    """Record of a single IPO trade."""
    symbol: str
    ipo_date: str                   # First trading date
    entry_date: str
    entry_price: float
    exit_date: str = ''
    exit_price: float = 0.0
    exit_reason: str = ''
    return_pct: float = 0.0
    pnl: float = 0.0
    hold_days: int = 0
    ipo_age_at_entry: int = 0       # Days since IPO when signal fired
    peak_price: float = 0.0         # Highest price during hold
    peak_return_pct: float = 0.0    # Max unrealized gain


@dataclass
class IPOResult:
    """Backtest result for IPO strategy."""
    config: IPOConfig
    trades: List[IPOTrade]
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    avg_winner_pct: float = 0.0
    avg_loser_pct: float = 0.0
    max_return_pct: float = 0.0
    max_loss_pct: float = 0.0
    total_pnl: float = 0.0
    avg_hold_days: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0         # avg_win * win_rate - avg_loss * loss_rate
    # Equity curve
    final_capital: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0


class IPOStrategyBacktester:
    """Backtests the IPO breakout strategy across all qualifying stocks."""

    def __init__(self, config: IPOConfig = None, preloaded_data: Dict[str, pd.DataFrame] = None):
        self.config = config or IPOConfig()
        self.trades: List[IPOTrade] = []
        self._preloaded_data = preloaded_data

    @staticmethod
    def preload_data(ipo_start_year: int = 2015, min_bars: int = 10) -> Dict[str, pd.DataFrame]:
        """Load all IPO stock data once. Reuse across multiple configs."""
        conn = sqlite3.connect(DB_PATH)
        cutoff = f'{ipo_start_year}-01-01'
        ipo_list = pd.read_sql_query(f"""
            SELECT symbol, MIN(date) as first_date, COUNT(*) as bars
            FROM market_data_unified
            WHERE timeframe = 'day'
            GROUP BY symbol
            HAVING MIN(date) >= '{cutoff}'
            ORDER BY first_date
        """, conn)

        if ipo_list.empty:
            conn.close()
            return {}

        symbols = ipo_list['symbol'].tolist()

        placeholders = ','.join('?' * len(symbols))
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM market_data_unified
            WHERE symbol IN ({placeholders})
              AND timeframe = 'day'
            ORDER BY symbol, date
        """
        df = pd.read_sql_query(query, conn, params=symbols)
        conn.close()

        df['date'] = pd.to_datetime(df['date'])
        data = {}
        for symbol, group in df.groupby('symbol'):
            sdf = group.set_index('date').sort_index()
            sdf = sdf[['open', 'high', 'low', 'close', 'volume']]
            if len(sdf) >= min_bars:
                data[symbol] = sdf

        logger.info(f"Preloaded {len(data)} IPO stocks since {ipo_start_year}")
        return data

    def _load_ipo_stocks(self) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data for all IPO candidates. Uses preloaded data if available."""
        cfg = self.config

        if self._preloaded_data is not None:
            # Filter preloaded data by min_ipo_age_days
            return {s: df for s, df in self._preloaded_data.items()
                    if len(df) >= cfg.min_ipo_age_days}

        conn = sqlite3.connect(DB_PATH)
        cutoff = f'{cfg.ipo_start_year}-01-01'
        ipo_list = pd.read_sql_query(f"""
            SELECT symbol, MIN(date) as first_date, COUNT(*) as bars
            FROM market_data_unified
            WHERE timeframe = 'day'
            GROUP BY symbol
            HAVING MIN(date) >= '{cutoff}'
            ORDER BY first_date
        """, conn)

        if ipo_list.empty:
            conn.close()
            return {}

        symbols = ipo_list['symbol'].tolist()
        logger.info(f"Found {len(symbols)} IPO candidates since {cfg.ipo_start_year}")

        placeholders = ','.join('?' * len(symbols))
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM market_data_unified
            WHERE symbol IN ({placeholders})
              AND timeframe = 'day'
            ORDER BY symbol, date
        """
        df = pd.read_sql_query(query, conn, params=symbols)
        conn.close()

        df['date'] = pd.to_datetime(df['date'])
        data = {}
        for symbol, group in df.groupby('symbol'):
            sdf = group.set_index('date').sort_index()
            sdf = sdf[['open', 'high', 'low', 'close', 'volume']]
            if len(sdf) >= cfg.min_ipo_age_days:
                data[symbol] = sdf

        return data

    def _check_entry_signal(self, df: pd.DataFrame, bar_idx: int) -> bool:
        """
        Check if entry signal fires on bar_idx.
        Conditions:
        1. Stock age is between min_ipo_age_days and max_ipo_age_days
        2. Close breaches ATH of first ath_lookback_days (first time)
        3. Volume >= vol_multiplier × avg volume of past vol_avg_period days
        """
        cfg = self.config

        if bar_idx < cfg.min_ipo_age_days:
            return False
        if bar_idx > cfg.max_ipo_age_days:
            return False

        # ATH of first N days
        ath_end = min(cfg.ath_lookback_days, bar_idx)
        initial_high = df['high'].iloc[:ath_end].max()

        # Current close must breach initial ATH
        current_close = df['close'].iloc[bar_idx]
        if current_close <= initial_high:
            return False

        # Check this is the FIRST breach: no prior close above initial_high
        # (after the ATH formation period)
        for i in range(ath_end, bar_idx):
            if df['close'].iloc[i] > initial_high:
                return False  # Already breached before

        # Volume check: current volume >= vol_multiplier × 20-day avg
        # Ignore day 1 volume if configured
        vol_start = 1 if cfg.ignore_day1_volume else 0
        vol_end = bar_idx  # Don't include current bar in average

        if vol_end - vol_start < 5:
            # Not enough volume history
            return False

        vol_lookback_start = max(vol_start, vol_end - cfg.vol_avg_period)
        avg_vol = df['volume'].iloc[vol_lookback_start:vol_end].mean()

        if avg_vol <= 0:
            return False

        current_vol = df['volume'].iloc[bar_idx]
        vol_ratio = current_vol / avg_vol

        if vol_ratio < cfg.vol_multiplier:
            return False

        # --- QUALITY FILTERS ---
        current_close = df['close'].iloc[bar_idx]

        # Min breakout strength
        if cfg.min_breakout_pct > 0:
            breakout_pct = (current_close - initial_high) / initial_high * 100
            if breakout_pct < cfg.min_breakout_pct:
                return False

        # Consecutive close above ATH
        if cfg.require_consec_close > 1:
            for j in range(1, cfg.require_consec_close):
                prev_idx = bar_idx - j
                if prev_idx < ath_end or df['close'].iloc[prev_idx] <= initial_high:
                    return False

        # Gap-up: open > previous close
        if cfg.require_gap_up:
            if bar_idx > 0 and df['open'].iloc[bar_idx] <= df['close'].iloc[bar_idx - 1]:
                return False

        # Listing gain: day 1 close > day 1 open
        if cfg.require_listing_gain:
            if df['close'].iloc[0] <= df['open'].iloc[0]:
                return False

        # Price above average price since listing
        if cfg.require_above_avg_price:
            avg_price = df['close'].iloc[:bar_idx].mean()
            if current_close <= avg_price:
                return False

        # RSI filter — reject if not enough data (stricter)
        if cfg.use_rsi_filter:
            if bar_idx < cfg.rsi_period + 1:
                return False  # Not enough data = reject
            rsi = calc_rsi(df['close'].iloc[:bar_idx + 1], cfg.rsi_period)
            if rsi.iloc[-1] < cfg.rsi_min:
                return False

        # EMA trend filter
        if cfg.use_ema_filter:
            if bar_idx < cfg.ema_filter_period:
                return False
            ema = calc_ema(df['close'].iloc[:bar_idx + 1], cfg.ema_filter_period)
            if current_close <= ema.iloc[-1]:
                return False

        # ADX trend strength filter
        if cfg.use_adx_filter:
            if bar_idx < cfg.adx_period + 1:
                return False
            adx, plus_di, minus_di = calc_adx(df.iloc[:bar_idx + 1], cfg.adx_period)
            if adx.iloc[-1] < cfg.adx_min or plus_di.iloc[-1] <= minus_di.iloc[-1]:
                return False

        # MFI filter
        if cfg.use_mfi_filter:
            if bar_idx < cfg.mfi_period + 1:
                return False
            mfi = calc_mfi(df.iloc[:bar_idx + 1], cfg.mfi_period)
            if mfi.iloc[-1] < cfg.mfi_min:
                return False

        return True

    def _run_exit(self, df: pd.DataFrame, entry_idx: int,
                  entry_price: float) -> Tuple[int, float, str]:
        """
        Run exit strategy from entry_idx forward.
        Returns (exit_idx, exit_price, exit_reason).
        """
        cfg = self.config
        strategy = cfg.exit_strategy

        # Check for hybrid exit mode first
        if cfg.hybrid_exit != 'none':
            return self._run_hybrid_exit(df, entry_idx, entry_price)

        peak_price = entry_price

        # Pre-compute indicators if needed
        if strategy == 'supertrend' and len(df) > entry_idx + 2:
            st_val, st_dir = calc_supertrend(df, cfg.st_atr_period, cfg.st_multiplier)
        elif strategy == 'ema_cross':
            ema_f = calc_ema(df['close'], cfg.ema_fast)
            ema_s = calc_ema(df['close'], cfg.ema_slow)
        elif strategy == 'atr_trail':
            atr = calc_atr(df, cfg.atr_trail_period)
        elif strategy == 'chandelier':
            atr = calc_atr(df, cfg.chandelier_period)

        breakeven_active = False

        for i in range(entry_idx + 1, len(df)):
            current_close = df['close'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            days_held = i - entry_idx
            peak_price = max(peak_price, current_high)
            return_pct = (current_close - entry_price) / entry_price * 100
            peak_return = (peak_price - entry_price) / entry_price * 100

            # --- Breakeven stop overlay ---
            if cfg.use_breakeven_stop and peak_return >= cfg.breakeven_trigger_pct:
                breakeven_active = True
                if current_low <= entry_price:
                    return i, entry_price, 'breakeven_stop'

            # --- Time-based trail tightening overlay ---
            effective_trail = cfg.trail_pct
            if cfg.use_time_tightening and days_held >= cfg.tighten_after_days:
                effective_trail = cfg.tightened_trail_pct

            # --- Always check hard stop loss (if any position has it) ---
            if strategy in ('supertrend', 'trailing_sl', 'ema_cross',
                            'atr_trail', 'chandelier'):
                if return_pct <= -30:
                    return i, current_close, 'hard_stop_30'

            # --- Strategy-specific exits ---
            if strategy == 'supertrend':
                if i < len(st_dir) and st_dir.iloc[i] == -1 and st_dir.iloc[i-1] == 1:
                    return i, current_close, 'supertrend_flip'

            elif strategy == 'trailing_sl':
                trail_stop = peak_price * (1 - effective_trail / 100)
                if current_low <= trail_stop:
                    return i, trail_stop, 'trailing_sl'

            elif strategy == 'fixed_target':
                if return_pct >= cfg.target_pct:
                    return i, current_close, 'target_hit'
                if return_pct <= -cfg.stop_loss_pct:
                    return i, current_close, 'stop_loss'

            elif strategy == 'time_exit':
                if days_held >= cfg.max_hold_days:
                    return i, current_close, 'time_exit'
                if return_pct <= -cfg.stop_loss_pct:
                    return i, current_close, 'stop_loss'

            elif strategy == 'ema_cross':
                if (i < len(ema_f) and ema_f.iloc[i] < ema_s.iloc[i]
                        and ema_f.iloc[i-1] >= ema_s.iloc[i-1]):
                    return i, current_close, 'ema_cross_down'

            elif strategy == 'atr_trail':
                if i < len(atr):
                    atr_stop = peak_price - cfg.atr_trail_mult * atr.iloc[i]
                    if current_low <= atr_stop:
                        return i, max(atr_stop, current_low), 'atr_trail_stop'

            elif strategy == 'chandelier':
                if i < len(atr):
                    chandelier_stop = peak_price - cfg.chandelier_mult * atr.iloc[i]
                    if current_low <= chandelier_stop:
                        return i, max(chandelier_stop, current_low), 'chandelier_exit'

        return len(df) - 1, df['close'].iloc[-1], 'end_of_data'

    def _run_hybrid_exit(self, df: pd.DataFrame, entry_idx: int,
                         entry_price: float) -> Tuple[int, float, str]:
        """
        Hybrid exit: combines target + trail + breakeven + time cap.
        Modes:
        - 'target_or_trail': Target profit OR trailing SL, whichever first
        - 'target_with_breakeven': Target profit, SL, and breakeven stop after trigger
        """
        cfg = self.config
        peak_price = entry_price
        breakeven_active = False

        for i in range(entry_idx + 1, len(df)):
            current_close = df['close'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            days_held = i - entry_idx
            peak_price = max(peak_price, current_high)
            return_pct = (current_close - entry_price) / entry_price * 100
            peak_return = (peak_price - entry_price) / entry_price * 100

            # Hard stop at 30%
            if return_pct <= -30:
                return i, current_close, 'hard_stop_30'

            if cfg.hybrid_exit == 'target_or_trail':
                # Target hit
                if return_pct >= cfg.hybrid_target_pct:
                    return i, current_close, 'hybrid_target'
                # Stop loss
                if return_pct <= -cfg.hybrid_sl_pct:
                    return i, current_close, 'hybrid_sl'
                # Trailing SL (kicks in once we're in profit)
                if peak_return > 0:
                    trail_stop = peak_price * (1 - cfg.hybrid_trail_pct / 100)
                    if current_low <= trail_stop:
                        return i, trail_stop, 'hybrid_trail'
                # Optional time cap
                if cfg.max_hold_days > 0 and days_held >= cfg.max_hold_days:
                    return i, current_close, 'hybrid_time'

            elif cfg.hybrid_exit == 'target_with_breakeven':
                # Target hit
                if return_pct >= cfg.hybrid_target_pct:
                    return i, current_close, 'hybrid_target'
                # Breakeven stop after trigger
                if peak_return >= cfg.breakeven_trigger_pct:
                    breakeven_active = True
                if breakeven_active and current_low <= entry_price:
                    return i, entry_price, 'hybrid_breakeven'
                # Hard SL before breakeven kicks in
                if not breakeven_active and return_pct <= -cfg.hybrid_sl_pct:
                    return i, current_close, 'hybrid_sl'
                # Time cap
                if cfg.max_hold_days > 0 and days_held >= cfg.max_hold_days:
                    return i, current_close, 'hybrid_time'

        return len(df) - 1, df['close'].iloc[-1], 'end_of_data'

    def run(self, progress_callback=None) -> IPOResult:
        """Run the full IPO strategy backtest."""
        cfg = self.config
        self.trades = []

        # Load IPO stock data
        ipo_data = self._load_ipo_stocks()
        total_stocks = len(ipo_data)

        if progress_callback:
            progress_callback(0, f'Scanning {total_stocks} IPO stocks...')

        capital = cfg.initial_capital
        equity_curve = [capital]

        for idx, (symbol, df) in enumerate(ipo_data.items()):
            if progress_callback and idx % 20 == 0:
                progress_callback(int(idx / total_stocks * 100),
                                  f'{idx}/{total_stocks} stocks scanned')

            # Scan for entry signal
            for bar_idx in range(cfg.min_ipo_age_days, min(cfg.max_ipo_age_days + 1, len(df))):
                if self._check_entry_signal(df, bar_idx):
                    entry_price = df['close'].iloc[bar_idx]
                    entry_date = df.index[bar_idx]
                    ipo_date = df.index[0]

                    # Run exit
                    exit_idx, exit_price, exit_reason = self._run_exit(
                        df, bar_idx, entry_price)

                    exit_date = df.index[exit_idx]
                    return_pct = (exit_price - entry_price) / entry_price * 100
                    position_value = capital * (cfg.position_size_pct / 100)
                    pnl = position_value * (return_pct / 100)
                    hold_days = (exit_date - entry_date).days

                    peak_price = df['high'].iloc[bar_idx:exit_idx+1].max()
                    peak_return = (peak_price - entry_price) / entry_price * 100

                    trade = IPOTrade(
                        symbol=symbol,
                        ipo_date=ipo_date.strftime('%Y-%m-%d'),
                        entry_date=entry_date.strftime('%Y-%m-%d'),
                        entry_price=round(entry_price, 2),
                        exit_date=exit_date.strftime('%Y-%m-%d'),
                        exit_price=round(exit_price, 2),
                        exit_reason=exit_reason,
                        return_pct=round(return_pct, 2),
                        pnl=round(pnl, 2),
                        hold_days=hold_days,
                        ipo_age_at_entry=bar_idx,
                        peak_price=round(peak_price, 2),
                        peak_return_pct=round(peak_return, 2),
                    )
                    self.trades.append(trade)

                    # Update capital
                    capital += pnl
                    equity_curve.append(capital)

                    break  # Only one trade per IPO stock

        # Build result
        result = self._build_result(equity_curve)
        if progress_callback:
            progress_callback(100, f'Done: {result.total_trades} trades')
        return result

    def _build_result(self, equity_curve: List[float]) -> IPOResult:
        """Compute aggregate metrics from trades."""
        trades = self.trades
        cfg = self.config

        if not trades:
            return IPOResult(config=cfg, trades=[], total_trades=0,
                             final_capital=cfg.initial_capital)

        total = len(trades)
        winners = [t for t in trades if t.return_pct > 0]
        losers = [t for t in trades if t.return_pct <= 0]

        win_count = len(winners)
        lose_count = len(losers)
        win_rate = win_count / total * 100

        avg_return = np.mean([t.return_pct for t in trades])
        avg_winner = np.mean([t.return_pct for t in winners]) if winners else 0
        avg_loser = np.mean([t.return_pct for t in losers]) if losers else 0

        max_return = max(t.return_pct for t in trades)
        max_loss = min(t.return_pct for t in trades)

        total_pnl = sum(t.pnl for t in trades)
        avg_hold = np.mean([t.hold_days for t in trades])

        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        expectancy = (win_rate / 100 * avg_winner) + ((1 - win_rate / 100) * avg_loser)

        # Equity curve metrics
        final_capital = equity_curve[-1]
        total_return_pct = (final_capital - cfg.initial_capital) / cfg.initial_capital * 100

        # Approximate CAGR (based on first and last trade dates)
        if trades:
            first_date = datetime.strptime(trades[0].entry_date, '%Y-%m-%d')
            last_date = datetime.strptime(trades[-1].exit_date, '%Y-%m-%d')
            years = max((last_date - first_date).days / 365.25, 0.1)
            if final_capital > 0 and cfg.initial_capital > 0:
                cagr = ((final_capital / cfg.initial_capital) ** (1 / years) - 1) * 100
            else:
                cagr = 0
        else:
            cagr = 0

        # Max drawdown from equity curve
        peak = equity_curve[0]
        max_dd = 0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Simple Sharpe approximation from trade returns
        returns = [t.return_pct for t in trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns))
        else:
            sharpe = 0

        return IPOResult(
            config=cfg,
            trades=trades,
            total_trades=total,
            winners=win_count,
            losers=lose_count,
            win_rate=round(win_rate, 1),
            avg_return_pct=round(avg_return, 2),
            avg_winner_pct=round(avg_winner, 2),
            avg_loser_pct=round(avg_loser, 2),
            max_return_pct=round(max_return, 2),
            max_loss_pct=round(max_loss, 2),
            total_pnl=round(total_pnl, 0),
            avg_hold_days=round(avg_hold, 1),
            profit_factor=round(profit_factor, 2),
            expectancy=round(expectancy, 2),
            final_capital=round(final_capital, 0),
            total_return_pct=round(total_return_pct, 2),
            cagr=round(cagr, 2),
            sharpe=round(sharpe, 2),
            max_drawdown=round(max_dd, 2),
        )
