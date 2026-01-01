"""
CPR-Based Covered Call Backtest Service
========================================

Strategy for covered call trading based on Weekly Central Pivot Range (CPR).

Strategy Logic:
1. Existing Position Assumption - Already own stocks, looking for covered call opportunities
2. Weekly Narrow CPR Filter - Ignore weeks where CPR width < threshold (default 0.5%)
3. Entry: 1st 30-min candle closes BELOW weekly CPR -> sell call at configurable OTM%
4. Premium Rollout: If sold call premium doubles -> roll to further OTM with similar premium
5. Exits:
   - Price closes above R1 (current or previous week's, whichever closer)
   - Premium erodes by target % (default 75%)
   - DTE threshold exit (default 10 days)

Configurable Parameters:
- narrow_cpr_threshold: 0.5% (skip weeks with narrow CPR)
- otm_strike_pct: 5% (strike selection)
- dte_min/max: 30-35 (target DTE range)
- premium_double_threshold: 2.0x (trigger rollout)
- premium_erosion_target: 75% (profit booking)
- dte_exit_threshold: 10 (days to exit)
- enable_r1_exit: True (exit on R1 breach)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import calendar

from .greeks_calculator import GreeksCalculator
from .data_manager import get_lot_size, FNO_LOT_SIZES
from .metrics_calculator import MetricsCalculator
from .backtest_db import get_backtest_db
from .intraday_data_bridge import get_intraday_bridge, IntradayDataBridge

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================

class CPRExitReason(Enum):
    """Exit reasons for CPR-based covered call positions"""
    R1_EXIT = "R1_EXIT"  # Price closed above R1
    PROFIT_TARGET = "PROFIT_TARGET"  # Premium eroded by target %
    DTE_EXIT = "DTE_EXIT"  # Exited due to DTE threshold
    EXPIRY = "EXPIRY"  # Option expired
    ASSIGNED = "ASSIGNED"  # Stock called away at strike
    ROLLOUT = "ROLLOUT"  # Position rolled to further OTM
    END_OF_BACKTEST = "END_OF_BACKTEST"  # Backtest period ended


@dataclass
class CPRBacktestConfig:
    """Configuration for CPR-based covered call backtest"""

    # Basic settings
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float = 1000000.0
    risk_free_rate: float = 0.07
    default_iv: float = 0.20

    # DTE targeting (configurable)
    dte_min: int = 30  # Minimum days to expiry
    dte_max: int = 35  # Maximum days to expiry

    # CPR settings (configurable)
    narrow_cpr_threshold: float = 0.5  # Ignore weeks with CPR < 0.5% of price

    # Entry settings (configurable)
    otm_strike_pct: float = 5.0  # 5% OTM strike

    # Premium doubling rollout (configurable)
    enable_premium_rollout: bool = True
    premium_double_threshold: float = 2.0  # Roll when premium doubles

    # Profit booking (configurable)
    premium_erosion_target: float = 75.0  # Book profit at 75% erosion
    dte_exit_threshold: int = 10  # Exit by 10 DTE

    # R1 exit logic (configurable)
    enable_r1_exit: bool = True  # Exit when price closes above R1
    use_closer_r1: bool = True  # Use whichever R1 is closer (current or previous week)

    # Position sizing
    position_size: int = 1  # Number of lots per position

    # Backtest name for saving
    name: str = ""


@dataclass
class CPRWeeklyData:
    """Weekly CPR calculation result"""
    week_start: datetime
    week_end: datetime
    prev_week_high: float
    prev_week_low: float
    prev_week_close: float
    pivot: float
    bc: float  # Bottom Central
    tc: float  # Top Central
    r1: float
    r2: float
    s1: float
    s2: float
    cpr_width: float  # TC - BC (absolute)
    cpr_width_pct: float  # cpr_width / pivot * 100
    is_narrow: bool  # True if cpr_width_pct < threshold
    cpr_top: float  # max(bc, tc)
    cpr_bottom: float  # min(bc, tc)
    first_candle_close: Optional[float] = None  # First 30-min candle close of the week
    first_candle_time: Optional[datetime] = None


@dataclass
class CPRPosition:
    """Represents an open CPR-based covered call position"""
    symbol: str
    lot_size: int
    entry_date: datetime
    stock_price_at_entry: float
    strike_price: float
    premium_received: float
    expiry_date: datetime
    iv_at_entry: float = 0.0
    delta_at_entry: float = 0.0
    theta_at_entry: float = 0.0

    # CPR tracking
    entry_week_cpr: Dict[str, float] = field(default_factory=dict)
    entry_candle_close: float = 0.0

    # Rollout tracking
    rollout_count: int = 0
    original_strike: float = 0.0
    original_premium: float = 0.0
    total_premium_collected: float = 0.0
    rollout_dates: List[datetime] = field(default_factory=list)

    # Week tracking for multi-week positions
    current_week_cpr: Dict[str, float] = field(default_factory=dict)
    previous_week_r1: float = 0.0
    weeks_held: int = 0


@dataclass
class CPRTrade:
    """Completed CPR trade record"""
    symbol: str
    lot_size: int
    entry_date: datetime
    stock_entry_price: float
    strike_price: float
    premium_received: float
    expiry_date: datetime
    exit_date: datetime
    stock_exit_price: float
    option_exit_price: float
    exit_reason: str
    stock_pnl: float
    option_pnl: float
    total_pnl: float
    return_pct: float
    # CPR-specific
    entry_cpr_pivot: float = 0.0
    entry_cpr_bc: float = 0.0
    entry_cpr_tc: float = 0.0
    entry_cpr_r1: float = 0.0
    entry_first_candle_close: float = 0.0
    rollout_count: int = 0
    total_premium_collected: float = 0.0
    weeks_held: int = 0
    iv_at_entry: float = 0.0
    delta_at_entry: float = 0.0


# =============================================================================
# CPR Covered Call Engine
# =============================================================================

class CPRCoveredCallEngine:
    """
    CPR-based covered call backtest engine.

    Strategy:
    1. Already own stocks (existing position assumption)
    2. Filter out narrow CPR weeks (< threshold)
    3. Entry: 1st 30-min candle closes BELOW weekly CPR -> sell call at OTM%
    4. Premium doubling rollout: If sold call premium doubles -> roll to further OTM
    5. Exit conditions:
       - Price closes ABOVE R1 (current or previous week's, whichever closer)
       - Premium erodes by target %
       - DTE threshold reached
    """

    def __init__(self, config: CPRBacktestConfig):
        """
        Initialize the CPR engine.

        Args:
            config: CPRBacktestConfig with all parameters
        """
        self.config = config
        self.greeks = GreeksCalculator(risk_free_rate=config.risk_free_rate)
        self.metrics = MetricsCalculator(risk_free_rate=config.risk_free_rate)
        self.intraday_bridge = get_intraday_bridge()

        # State
        self.positions: Dict[str, CPRPosition] = {}
        self.trades: List[CPRTrade] = []
        self.equity_curve: Dict[datetime, float] = {}
        self.cash = config.initial_capital

        # Weekly CPR cache
        self.weekly_cpr_cache: Dict[str, Dict[datetime, CPRWeeklyData]] = {}

        # Statistics
        self.stats = {
            'entry_signals': 0,
            'entries_taken': 0,
            'narrow_cpr_skipped': 0,
            'candle_above_cpr_skipped': 0,
            'r1_exits': 0,
            'profit_target_exits': 0,
            'dte_exits': 0,
            'rollouts': 0,
            'expiries': 0,
            'assignments': 0,
        }

        logger.info(f"CPRCoveredCallEngine initialized with config: {config}")

    def run_backtest(
        self,
        stock_data: Dict[str, pd.DataFrame],
        intraday_data: Optional[Dict[str, pd.DataFrame]] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run complete CPR-based backtest.

        Args:
            stock_data: Dictionary of symbol -> daily OHLCV DataFrame
            intraday_data: Optional dict of symbol -> 30-min DataFrame
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting CPR backtest from {self.config.start_date} to {self.config.end_date}")

        # Get all trading days
        all_dates = set()
        for symbol, df in stock_data.items():
            dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
            all_dates.update(dates.to_pydatetime())

        trading_days = sorted([d for d in all_dates
                               if self.config.start_date <= d <= self.config.end_date])

        if not trading_days:
            logger.error("No trading days found in date range")
            return self._empty_result()

        total_days = len(trading_days)

        # Pre-load intraday data if not provided
        if intraday_data is None:
            intraday_data = self._load_intraday_data(stock_data.keys())

        # Pre-calculate weekly CPR for all weeks
        self._precalculate_weekly_cpr(stock_data, intraday_data)

        # Process each trading day
        for i, current_date in enumerate(trading_days):
            self._process_day(current_date, stock_data, intraday_data)

            # Update equity curve
            self._update_equity_curve(current_date, stock_data)

            # Progress callback
            if progress_callback and i % 10 == 0:
                progress_callback(i / total_days * 100, f"Processing {current_date.date()}")

        # Close all remaining positions at end
        self._close_all_positions(trading_days[-1], stock_data, CPRExitReason.END_OF_BACKTEST)

        # Calculate final metrics
        results = self._calculate_results()

        logger.info(f"CPR Backtest complete: {len(self.trades)} trades, "
                   f"Total Return: {results['total_return']:.2f}%")

        return results

    def _load_intraday_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Load intraday data for all symbols."""
        intraday_data = {}

        for symbol in symbols:
            df = self.intraday_bridge.load_30min_data(
                symbol,
                self.config.start_date - timedelta(days=14),  # Extra days for CPR calc
                self.config.end_date
            )
            if not df.empty:
                intraday_data[symbol] = df

        return intraday_data

    def _precalculate_weekly_cpr(
        self,
        stock_data: Dict[str, pd.DataFrame],
        intraday_data: Dict[str, pd.DataFrame]
    ):
        """Pre-calculate weekly CPR for all symbols and weeks."""

        for symbol, daily_df in stock_data.items():
            self.weekly_cpr_cache[symbol] = {}

            # Get all Mondays in the date range
            current = self.config.start_date
            while current <= self.config.end_date:
                # Find Monday of this week
                monday = current - timedelta(days=current.weekday())

                # Calculate CPR for this week
                cpr_data = self._calculate_weekly_cpr(
                    symbol, monday, daily_df,
                    intraday_data.get(symbol, pd.DataFrame())
                )

                if cpr_data:
                    self.weekly_cpr_cache[symbol][monday] = cpr_data

                # Move to next week
                current = monday + timedelta(days=7)

    def _calculate_weekly_cpr(
        self,
        symbol: str,
        week_start: datetime,
        daily_df: pd.DataFrame,
        intraday_df: pd.DataFrame
    ) -> Optional[CPRWeeklyData]:
        """
        Calculate weekly CPR from previous week's data.

        CPR Formula:
        - Pivot (P) = (High + Low + Close) / 3
        - Bottom Central (BC) = (High + Low) / 2
        - Top Central (TC) = (P - BC) + P = 2P - BC
        - R1 = (2 x P) - Low
        - R2 = P + (High - Low)
        - S1 = (2 x P) - High
        - S2 = P - (High - Low)
        """
        try:
            # Ensure week_start is a Monday
            if week_start.weekday() != 0:
                week_start = week_start - timedelta(days=week_start.weekday())

            week_end = week_start + timedelta(days=6)

            # Get previous week's data
            prev_week_start = week_start - timedelta(days=7)
            prev_week_end = week_start - timedelta(days=1)

            # Filter daily data for previous week
            if isinstance(daily_df.index, pd.DatetimeIndex):
                mask = (daily_df.index >= prev_week_start) & (daily_df.index <= prev_week_end)
            else:
                daily_df.index = pd.to_datetime(daily_df.index)
                mask = (daily_df.index >= prev_week_start) & (daily_df.index <= prev_week_end)

            prev_week_data = daily_df[mask]

            if prev_week_data.empty or len(prev_week_data) < 3:
                return None

            # Get H/L/C from previous week
            prev_high = prev_week_data['high'].max() if 'high' in prev_week_data else prev_week_data['High'].max()
            prev_low = prev_week_data['low'].min() if 'low' in prev_week_data else prev_week_data['Low'].min()
            prev_close = prev_week_data['close'].iloc[-1] if 'close' in prev_week_data else prev_week_data['Close'].iloc[-1]

            # Calculate CPR components
            pivot = (prev_high + prev_low + prev_close) / 3
            bc = (prev_high + prev_low) / 2
            tc = 2 * pivot - bc

            # Resistance and Support levels
            r1 = (2 * pivot) - prev_low
            r2 = pivot + (prev_high - prev_low)
            s1 = (2 * pivot) - prev_high
            s2 = pivot - (prev_high - prev_low)

            # CPR width
            cpr_width = abs(tc - bc)
            cpr_width_pct = (cpr_width / pivot) * 100
            is_narrow = cpr_width_pct < self.config.narrow_cpr_threshold

            cpr_top = max(bc, tc)
            cpr_bottom = min(bc, tc)

            # Get first 30-min candle of the week
            first_candle_close = None
            first_candle_time = None

            if not intraday_df.empty:
                first_candle_close, first_candle_time = self.intraday_bridge.get_first_candle_of_week(
                    symbol, week_start, intraday_df
                )

            return CPRWeeklyData(
                week_start=week_start,
                week_end=week_end,
                prev_week_high=prev_high,
                prev_week_low=prev_low,
                prev_week_close=prev_close,
                pivot=pivot,
                bc=bc,
                tc=tc,
                r1=r1,
                r2=r2,
                s1=s1,
                s2=s2,
                cpr_width=cpr_width,
                cpr_width_pct=cpr_width_pct,
                is_narrow=is_narrow,
                cpr_top=cpr_top,
                cpr_bottom=cpr_bottom,
                first_candle_close=first_candle_close,
                first_candle_time=first_candle_time
            )

        except Exception as e:
            logger.error(f"Error calculating CPR for {symbol} week {week_start}: {e}")
            return None

    def _get_week_cpr(self, symbol: str, date: datetime) -> Optional[CPRWeeklyData]:
        """Get CPR data for the week containing the given date."""
        monday = date - timedelta(days=date.weekday())
        return self.weekly_cpr_cache.get(symbol, {}).get(monday)

    def _process_day(
        self,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame],
        intraday_data: Dict[str, pd.DataFrame]
    ):
        """Process a single trading day."""

        # 1. Check for R1 exits on existing positions
        self._check_r1_exits(current_date, stock_data)

        # 2. Check for profit target exits
        self._check_profit_target_exits(current_date, stock_data)

        # 3. Check for DTE exits
        self._check_dte_exits(current_date, stock_data)

        # 4. Check for premium doubling (rollout)
        self._check_premium_rollouts(current_date, stock_data)

        # 5. Check for expiries
        self._check_expiries(current_date, stock_data)

        # 6. Check for new entries (only on Mondays or first trading day of week)
        if current_date.weekday() == 0:  # Monday
            self._check_new_entries(current_date, stock_data, intraday_data)

    def _check_new_entries(
        self,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame],
        intraday_data: Dict[str, pd.DataFrame]
    ):
        """Check for new entry signals on Monday."""

        for symbol in self.config.symbols:
            # Skip if already have position
            if symbol in self.positions:
                continue

            # Get weekly CPR
            cpr_data = self._get_week_cpr(symbol, current_date)
            if not cpr_data:
                continue

            # Check narrow CPR filter
            if cpr_data.is_narrow:
                self.stats['narrow_cpr_skipped'] += 1
                logger.debug(f"{symbol}: Skipping narrow CPR week (width: {cpr_data.cpr_width_pct:.2f}%)")
                continue

            # Check first candle condition
            if cpr_data.first_candle_close is None:
                logger.debug(f"{symbol}: No first candle data available")
                continue

            # Entry condition: First 30-min candle closes BELOW CPR
            if cpr_data.first_candle_close >= cpr_data.cpr_bottom:
                self.stats['candle_above_cpr_skipped'] += 1
                logger.debug(f"{symbol}: First candle ({cpr_data.first_candle_close:.2f}) "
                           f"not below CPR bottom ({cpr_data.cpr_bottom:.2f})")
                continue

            self.stats['entry_signals'] += 1

            # Get current stock price
            daily_df = stock_data.get(symbol)
            if daily_df is None or daily_df.empty:
                continue

            try:
                if current_date in daily_df.index:
                    row = daily_df.loc[current_date]
                else:
                    # Find nearest date
                    idx = daily_df.index.get_indexer([current_date], method='ffill')[0]
                    if idx < 0:
                        continue
                    row = daily_df.iloc[idx]

                stock_price = row['close'] if 'close' in row else row['Close']
            except Exception as e:
                logger.error(f"Error getting stock price for {symbol}: {e}")
                continue

            # Find appropriate expiry (30-35 DTE)
            expiry_date = self._find_expiry_date(current_date)
            if not expiry_date:
                continue

            dte = (expiry_date - current_date).days
            if dte < self.config.dte_min or dte > self.config.dte_max:
                logger.debug(f"{symbol}: DTE {dte} outside range [{self.config.dte_min}, {self.config.dte_max}]")
                continue

            # Open position
            self._open_position(symbol, current_date, stock_price, expiry_date, cpr_data, stock_data)

    def _open_position(
        self,
        symbol: str,
        entry_date: datetime,
        stock_price: float,
        expiry_date: datetime,
        cpr_data: CPRWeeklyData,
        stock_data: Dict[str, pd.DataFrame]
    ):
        """Open a new CPR-based covered call position."""

        try:
            lot_size = get_lot_size(symbol) * self.config.position_size

            # Calculate strike at OTM%
            strike_price = stock_price * (1 + self.config.otm_strike_pct / 100)
            strike_price = self._round_to_strike(strike_price, stock_price)

            # Calculate premium using Black-Scholes
            dte = (expiry_date - entry_date).days
            time_to_expiry = dte / 365.0
            iv = self._estimate_iv(symbol, stock_data.get(symbol))

            premium = self.greeks.calculate_option_price(
                spot=stock_price,
                strike=strike_price,
                time_to_expiry=time_to_expiry,
                volatility=iv,
                option_type='CE'
            )

            if premium <= 0:
                logger.warning(f"{symbol}: Invalid premium calculated: {premium}")
                return

            # Get Greeks
            greeks = self.greeks.calculate_greeks(
                spot=stock_price,
                strike=strike_price,
                time_to_expiry=time_to_expiry,
                volatility=iv,
                option_type='CE'
            )

            # Check capital
            position_cost = stock_price * lot_size
            if position_cost > self.cash:
                logger.debug(f"{symbol}: Insufficient capital (need {position_cost}, have {self.cash})")
                return

            # Create position
            position = CPRPosition(
                symbol=symbol,
                lot_size=lot_size,
                entry_date=entry_date,
                stock_price_at_entry=stock_price,
                strike_price=strike_price,
                premium_received=premium,
                expiry_date=expiry_date,
                iv_at_entry=iv,
                delta_at_entry=greeks.get('delta', 0),
                theta_at_entry=greeks.get('theta', 0),
                entry_week_cpr={
                    'pivot': cpr_data.pivot,
                    'bc': cpr_data.bc,
                    'tc': cpr_data.tc,
                    'r1': cpr_data.r1,
                    'r2': cpr_data.r2,
                    'cpr_top': cpr_data.cpr_top,
                    'cpr_bottom': cpr_data.cpr_bottom,
                },
                entry_candle_close=cpr_data.first_candle_close or 0.0,
                original_strike=strike_price,
                original_premium=premium,
                total_premium_collected=premium,
                current_week_cpr={
                    'r1': cpr_data.r1,
                },
                previous_week_r1=cpr_data.r1,
                weeks_held=1,
            )

            # Update state
            self.positions[symbol] = position
            self.cash -= position_cost
            self.cash += premium * lot_size  # Collect premium

            self.stats['entries_taken'] += 1

            logger.info(f"ENTRY: {symbol} @ {stock_price:.2f}, Strike: {strike_price:.2f}, "
                       f"Premium: {premium:.2f}, Expiry: {expiry_date.date()}")

        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {e}")

    def _check_r1_exits(self, current_date: datetime, stock_data: Dict[str, pd.DataFrame]):
        """Check for R1 exit conditions."""

        if not self.config.enable_r1_exit:
            return

        positions_to_close = []

        for symbol, position in self.positions.items():
            daily_df = stock_data.get(symbol)
            if daily_df is None:
                continue

            try:
                if current_date in daily_df.index:
                    row = daily_df.loc[current_date]
                else:
                    idx = daily_df.index.get_indexer([current_date], method='ffill')[0]
                    if idx < 0:
                        continue
                    row = daily_df.iloc[idx]

                current_close = row['close'] if 'close' in row else row['Close']
            except:
                continue

            # Get current week's CPR
            current_cpr = self._get_week_cpr(symbol, current_date)
            if not current_cpr:
                continue

            # Update current week's R1
            position.current_week_cpr['r1'] = current_cpr.r1

            # Determine which R1 to use
            if self.config.use_closer_r1:
                current_r1 = current_cpr.r1
                prev_r1 = position.previous_week_r1
                # Use whichever is closer to price
                target_r1 = min(current_r1, prev_r1, key=lambda r: abs(r - current_close))
            else:
                target_r1 = current_cpr.r1

            # Check R1 exit condition
            if current_close > target_r1:
                positions_to_close.append((symbol, current_close, CPRExitReason.R1_EXIT))
                self.stats['r1_exits'] += 1

            # Update previous week's R1 if we're on a new week's Monday
            if current_date.weekday() == 0:
                position.previous_week_r1 = current_cpr.r1
                position.weeks_held += 1

        # Close positions
        for symbol, exit_price, reason in positions_to_close:
            self._close_position(symbol, current_date, exit_price, stock_data, reason)

    def _check_profit_target_exits(self, current_date: datetime, stock_data: Dict[str, pd.DataFrame]):
        """Check for premium erosion profit target exits."""

        positions_to_close = []

        for symbol, position in self.positions.items():
            daily_df = stock_data.get(symbol)
            if daily_df is None:
                continue

            try:
                if current_date in daily_df.index:
                    row = daily_df.loc[current_date]
                else:
                    idx = daily_df.index.get_indexer([current_date], method='ffill')[0]
                    if idx < 0:
                        continue
                    row = daily_df.iloc[idx]

                stock_price = row['close'] if 'close' in row else row['Close']
            except:
                continue

            # Calculate current option price
            dte = (position.expiry_date - current_date).days
            if dte <= 0:
                continue

            time_to_expiry = dte / 365.0
            current_option_price = self.greeks.calculate_option_price(
                spot=stock_price,
                strike=position.strike_price,
                time_to_expiry=time_to_expiry,
                volatility=position.iv_at_entry,
                option_type='CE'
            )

            # Calculate premium erosion
            erosion_pct = (1 - current_option_price / position.premium_received) * 100

            if erosion_pct >= self.config.premium_erosion_target:
                positions_to_close.append((symbol, stock_price, CPRExitReason.PROFIT_TARGET))
                self.stats['profit_target_exits'] += 1

        for symbol, exit_price, reason in positions_to_close:
            self._close_position(symbol, current_date, exit_price, stock_data, reason)

    def _check_dte_exits(self, current_date: datetime, stock_data: Dict[str, pd.DataFrame]):
        """Check for DTE threshold exits."""

        positions_to_close = []

        for symbol, position in self.positions.items():
            dte = (position.expiry_date - current_date).days

            if dte <= self.config.dte_exit_threshold:
                daily_df = stock_data.get(symbol)
                if daily_df is None:
                    continue

                try:
                    if current_date in daily_df.index:
                        row = daily_df.loc[current_date]
                    else:
                        idx = daily_df.index.get_indexer([current_date], method='ffill')[0]
                        if idx < 0:
                            continue
                        row = daily_df.iloc[idx]

                    stock_price = row['close'] if 'close' in row else row['Close']
                    positions_to_close.append((symbol, stock_price, CPRExitReason.DTE_EXIT))
                    self.stats['dte_exits'] += 1
                except:
                    continue

        for symbol, exit_price, reason in positions_to_close:
            self._close_position(symbol, current_date, exit_price, stock_data, reason)

    def _check_premium_rollouts(self, current_date: datetime, stock_data: Dict[str, pd.DataFrame]):
        """Check for premium doubling (trigger rollout)."""

        if not self.config.enable_premium_rollout:
            return

        for symbol, position in list(self.positions.items()):
            daily_df = stock_data.get(symbol)
            if daily_df is None:
                continue

            try:
                if current_date in daily_df.index:
                    row = daily_df.loc[current_date]
                else:
                    idx = daily_df.index.get_indexer([current_date], method='ffill')[0]
                    if idx < 0:
                        continue
                    row = daily_df.iloc[idx]

                stock_price = row['close'] if 'close' in row else row['Close']
            except:
                continue

            # Calculate current option price
            dte = (position.expiry_date - current_date).days
            if dte <= 0:
                continue

            time_to_expiry = dte / 365.0
            current_option_price = self.greeks.calculate_option_price(
                spot=stock_price,
                strike=position.strike_price,
                time_to_expiry=time_to_expiry,
                volatility=position.iv_at_entry,
                option_type='CE'
            )

            # Check if premium has doubled
            if current_option_price >= position.premium_received * self.config.premium_double_threshold:
                self._execute_rollout(symbol, current_date, stock_price, current_option_price, stock_data)

    def _execute_rollout(
        self,
        symbol: str,
        current_date: datetime,
        stock_price: float,
        current_option_price: float,
        stock_data: Dict[str, pd.DataFrame]
    ):
        """
        Execute premium doubling rollout:
        1. Buy back current call at market
        2. Sell new call at further OTM strike with similar premium
        """

        position = self.positions.get(symbol)
        if not position:
            return

        try:
            # Cost to buy back current option
            buyback_cost = current_option_price * position.lot_size

            # Find new strike at further OTM (target similar premium)
            target_premium = position.original_premium
            new_strike = self._find_strike_for_premium(
                stock_price,
                position.expiry_date - current_date,
                position.iv_at_entry,
                target_premium
            )

            if new_strike <= position.strike_price:
                logger.warning(f"{symbol}: Could not find higher strike for rollout")
                return

            # Calculate new premium
            dte = (position.expiry_date - current_date).days
            time_to_expiry = dte / 365.0
            new_premium = self.greeks.calculate_option_price(
                spot=stock_price,
                strike=new_strike,
                time_to_expiry=time_to_expiry,
                volatility=position.iv_at_entry,
                option_type='CE'
            )

            # Update position
            self.cash -= buyback_cost
            self.cash += new_premium * position.lot_size

            position.strike_price = new_strike
            position.premium_received = new_premium
            position.total_premium_collected += new_premium - current_option_price
            position.rollout_count += 1
            position.rollout_dates.append(current_date)

            self.stats['rollouts'] += 1

            logger.info(f"ROLLOUT: {symbol} from {position.strike_price:.2f} to {new_strike:.2f}, "
                       f"New Premium: {new_premium:.2f}")

        except Exception as e:
            logger.error(f"Error executing rollout for {symbol}: {e}")

    def _find_strike_for_premium(
        self,
        stock_price: float,
        days_remaining: timedelta,
        iv: float,
        target_premium: float
    ) -> float:
        """Find strike price that would give approximately target premium."""

        dte = days_remaining.days if isinstance(days_remaining, timedelta) else days_remaining
        time_to_expiry = dte / 365.0

        # Start from 10% OTM and search up
        for otm_pct in range(10, 25, 1):
            strike = stock_price * (1 + otm_pct / 100)
            strike = self._round_to_strike(strike, stock_price)

            premium = self.greeks.calculate_option_price(
                spot=stock_price,
                strike=strike,
                time_to_expiry=time_to_expiry,
                volatility=iv,
                option_type='CE'
            )

            if premium <= target_premium * 1.2:  # Within 20% of target
                return strike

        # Fallback: return 15% OTM
        return stock_price * 1.15

    def _check_expiries(self, current_date: datetime, stock_data: Dict[str, pd.DataFrame]):
        """Check for option expiries."""

        positions_to_close = []

        for symbol, position in self.positions.items():
            if current_date >= position.expiry_date:
                daily_df = stock_data.get(symbol)
                if daily_df is None:
                    continue

                try:
                    if current_date in daily_df.index:
                        row = daily_df.loc[current_date]
                    else:
                        idx = daily_df.index.get_indexer([current_date], method='ffill')[0]
                        if idx < 0:
                            continue
                        row = daily_df.iloc[idx]

                    stock_price = row['close'] if 'close' in row else row['Close']

                    # Check if assigned (stock price >= strike)
                    if stock_price >= position.strike_price:
                        positions_to_close.append((symbol, position.strike_price, CPRExitReason.ASSIGNED))
                        self.stats['assignments'] += 1
                    else:
                        positions_to_close.append((symbol, stock_price, CPRExitReason.EXPIRY))
                        self.stats['expiries'] += 1
                except:
                    continue

        for symbol, exit_price, reason in positions_to_close:
            self._close_position(symbol, current_date, exit_price, stock_data, reason)

    def _close_position(
        self,
        symbol: str,
        exit_date: datetime,
        stock_exit_price: float,
        stock_data: Dict[str, pd.DataFrame],
        exit_reason: CPRExitReason
    ):
        """Close a position and record the trade."""

        position = self.positions.get(symbol)
        if not position:
            return

        try:
            # Calculate option exit price
            dte = max(0, (position.expiry_date - exit_date).days)

            if dte > 0 and exit_reason not in [CPRExitReason.ASSIGNED, CPRExitReason.EXPIRY]:
                time_to_expiry = dte / 365.0
                option_exit_price = self.greeks.calculate_option_price(
                    spot=stock_exit_price,
                    strike=position.strike_price,
                    time_to_expiry=time_to_expiry,
                    volatility=position.iv_at_entry,
                    option_type='CE'
                )
            else:
                # At expiry
                if exit_reason == CPRExitReason.ASSIGNED:
                    option_exit_price = stock_exit_price - position.strike_price
                else:
                    option_exit_price = max(0, stock_exit_price - position.strike_price)

            # Calculate P&L
            stock_pnl = (stock_exit_price - position.stock_price_at_entry) * position.lot_size
            option_pnl = (position.total_premium_collected - option_exit_price) * position.lot_size
            total_pnl = stock_pnl + option_pnl

            initial_investment = position.stock_price_at_entry * position.lot_size
            return_pct = (total_pnl / initial_investment) * 100 if initial_investment > 0 else 0

            # Create trade record
            trade = CPRTrade(
                symbol=symbol,
                lot_size=position.lot_size,
                entry_date=position.entry_date,
                stock_entry_price=position.stock_price_at_entry,
                strike_price=position.strike_price,
                premium_received=position.premium_received,
                expiry_date=position.expiry_date,
                exit_date=exit_date,
                stock_exit_price=stock_exit_price,
                option_exit_price=option_exit_price,
                exit_reason=exit_reason.value,
                stock_pnl=stock_pnl,
                option_pnl=option_pnl,
                total_pnl=total_pnl,
                return_pct=return_pct,
                entry_cpr_pivot=position.entry_week_cpr.get('pivot', 0),
                entry_cpr_bc=position.entry_week_cpr.get('bc', 0),
                entry_cpr_tc=position.entry_week_cpr.get('tc', 0),
                entry_cpr_r1=position.entry_week_cpr.get('r1', 0),
                entry_first_candle_close=position.entry_candle_close,
                rollout_count=position.rollout_count,
                total_premium_collected=position.total_premium_collected,
                weeks_held=position.weeks_held,
                iv_at_entry=position.iv_at_entry,
                delta_at_entry=position.delta_at_entry,
            )

            self.trades.append(trade)

            # Update cash
            if exit_reason == CPRExitReason.ASSIGNED:
                # Stock called away at strike
                self.cash += position.strike_price * position.lot_size
            else:
                # Keep stock, buy back option if needed
                self.cash += stock_exit_price * position.lot_size
                if dte > 0:
                    self.cash -= option_exit_price * position.lot_size

            # Remove position
            del self.positions[symbol]

            logger.info(f"EXIT ({exit_reason.value}): {symbol} @ {stock_exit_price:.2f}, "
                       f"P&L: {total_pnl:.2f} ({return_pct:.2f}%)")

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    def _close_all_positions(
        self,
        exit_date: datetime,
        stock_data: Dict[str, pd.DataFrame],
        exit_reason: CPRExitReason
    ):
        """Close all remaining positions at end of backtest."""

        for symbol in list(self.positions.keys()):
            daily_df = stock_data.get(symbol)
            if daily_df is None:
                continue

            try:
                if exit_date in daily_df.index:
                    row = daily_df.loc[exit_date]
                else:
                    idx = daily_df.index.get_indexer([exit_date], method='ffill')[0]
                    if idx < 0:
                        continue
                    row = daily_df.iloc[idx]

                stock_price = row['close'] if 'close' in row else row['Close']
                self._close_position(symbol, exit_date, stock_price, stock_data, exit_reason)
            except:
                continue

    def _update_equity_curve(self, current_date: datetime, stock_data: Dict[str, pd.DataFrame]):
        """Update equity curve with current portfolio value."""

        portfolio_value = self.cash

        for symbol, position in self.positions.items():
            daily_df = stock_data.get(symbol)
            if daily_df is None:
                continue

            try:
                if current_date in daily_df.index:
                    row = daily_df.loc[current_date]
                else:
                    idx = daily_df.index.get_indexer([current_date], method='ffill')[0]
                    if idx < 0:
                        continue
                    row = daily_df.iloc[idx]

                stock_price = row['close'] if 'close' in row else row['Close']

                # Add stock value
                portfolio_value += stock_price * position.lot_size

                # Subtract option liability
                dte = (position.expiry_date - current_date).days
                if dte > 0:
                    time_to_expiry = dte / 365.0
                    option_price = self.greeks.calculate_option_price(
                        spot=stock_price,
                        strike=position.strike_price,
                        time_to_expiry=time_to_expiry,
                        volatility=position.iv_at_entry,
                        option_type='CE'
                    )
                    portfolio_value -= option_price * position.lot_size
            except:
                continue

        self.equity_curve[current_date] = portfolio_value

    def _find_expiry_date(self, current_date: datetime) -> Optional[datetime]:
        """Find the monthly expiry date that falls within DTE range."""

        # Find last Thursday of current and next month
        for month_offset in range(0, 3):
            year = current_date.year
            month = current_date.month + month_offset

            if month > 12:
                month -= 12
                year += 1

            # Find last Thursday
            last_day = calendar.monthrange(year, month)[1]
            last_date = datetime(year, month, last_day)

            # Go back to find Thursday (weekday 3)
            days_to_subtract = (last_date.weekday() - 3) % 7
            expiry = last_date - timedelta(days=days_to_subtract)

            dte = (expiry - current_date).days

            if self.config.dte_min <= dte <= self.config.dte_max:
                return expiry

        return None

    def _round_to_strike(self, price: float, reference_price: float) -> float:
        """Round price to valid strike interval."""

        if reference_price < 100:
            interval = 2.5
        elif reference_price < 500:
            interval = 5
        elif reference_price < 1000:
            interval = 10
        elif reference_price < 2500:
            interval = 25
        elif reference_price < 5000:
            interval = 50
        else:
            interval = 100

        return round(price / interval) * interval

    def _estimate_iv(self, symbol: str, daily_df: Optional[pd.DataFrame]) -> float:
        """Estimate implied volatility from historical data."""

        if daily_df is None or len(daily_df) < 30:
            return self.config.default_iv

        try:
            close_col = 'close' if 'close' in daily_df.columns else 'Close'
            closes = daily_df[close_col].tail(30)

            log_returns = np.log(closes / closes.shift(1)).dropna()

            if len(log_returns) < 10:
                return self.config.default_iv

            # Annualize
            hv = log_returns.std() * np.sqrt(252)

            # Add premium for IV
            iv = hv * 1.2

            return max(0.10, min(0.80, iv))

        except:
            return self.config.default_iv

    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate final backtest results and metrics."""

        if not self.trades:
            return self._empty_result()

        # Convert trades to DataFrame
        trades_df = pd.DataFrame([vars(t) for t in self.trades])

        # Basic metrics
        total_pnl = trades_df['total_pnl'].sum()
        total_return = (total_pnl / self.config.initial_capital) * 100

        # Win rate
        winning_trades = trades_df[trades_df['total_pnl'] > 0]
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0

        # Sharpe ratio from equity curve
        equity_series = pd.Series(self.equity_curve)
        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            sharpe = self.metrics.calculate_sharpe_ratio(returns)
        else:
            sharpe = 0

        # Max drawdown
        if len(equity_series) > 1:
            max_dd = self.metrics.calculate_max_drawdown(equity_series)
        else:
            max_dd = 0

        # Average trade metrics
        avg_pnl = trades_df['total_pnl'].mean()
        avg_return = trades_df['return_pct'].mean()
        avg_holding_days = (trades_df['exit_date'] - trades_df['entry_date']).dt.days.mean()

        # Exit reason breakdown
        exit_breakdown = trades_df['exit_reason'].value_counts().to_dict()

        # Annualized return
        days_in_backtest = (self.config.end_date - self.config.start_date).days
        years = days_in_backtest / 365
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'avg_pnl': avg_pnl,
            'avg_return': avg_return,
            'avg_holding_days': avg_holding_days,
            'total_pnl': total_pnl,
            'final_value': list(self.equity_curve.values())[-1] if self.equity_curve else self.config.initial_capital,
            'exit_breakdown': exit_breakdown,
            'stats': self.stats,
            'trades': [vars(t) for t in self.trades],
            'equity_curve': {str(k): v for k, v in self.equity_curve.items()},
            'config': {
                'symbols': self.config.symbols,
                'start_date': str(self.config.start_date.date()),
                'end_date': str(self.config.end_date.date()),
                'narrow_cpr_threshold': self.config.narrow_cpr_threshold,
                'otm_strike_pct': self.config.otm_strike_pct,
                'dte_min': self.config.dte_min,
                'dte_max': self.config.dte_max,
                'premium_double_threshold': self.config.premium_double_threshold,
                'premium_erosion_target': self.config.premium_erosion_target,
                'dte_exit_threshold': self.config.dte_exit_threshold,
                'enable_r1_exit': self.config.enable_r1_exit,
                'use_closer_r1': self.config.use_closer_r1,
            }
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'total_return': 0,
            'annualized_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'total_trades': 0,
            'avg_pnl': 0,
            'avg_return': 0,
            'avg_holding_days': 0,
            'total_pnl': 0,
            'final_value': self.config.initial_capital,
            'exit_breakdown': {},
            'stats': self.stats,
            'trades': [],
            'equity_curve': {},
            'config': {},
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def run_cpr_backtest(
    config: CPRBacktestConfig,
    stock_data: Dict[str, pd.DataFrame],
    intraday_data: Optional[Dict[str, pd.DataFrame]] = None,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Run a CPR-based covered call backtest.

    Args:
        config: CPRBacktestConfig with all parameters
        stock_data: Dictionary of symbol -> daily OHLCV DataFrame
        intraday_data: Optional dict of symbol -> 30-min DataFrame
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with backtest results
    """
    engine = CPRCoveredCallEngine(config)
    return engine.run_backtest(stock_data, intraday_data, progress_callback)