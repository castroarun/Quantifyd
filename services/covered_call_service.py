"""
Covered Call Backtest Service
=============================

Core engine for simulating covered call strategies on Indian stocks.

Strategy Overview:
- Buy and hold stock
- Sell OTM call options monthly
- Collect premium as income
- Handle assignment when stock exceeds strike

Strike Selection Methods:
- DELTA_30: Sell calls with ~0.30 delta
- DELTA_40: Sell calls with ~0.40 delta
- OTM_2PCT: Sell calls 2% above spot
- OTM_5PCT: Sell calls 5% above spot
- ATM: Sell at-the-money calls

Exit Strategies:
- HOLD_TO_EXPIRY: Hold until expiration
- PROFIT_TARGET: Close when X% of max profit captured (configurable)
- STOP_LOSS: Close when loss exceeds Nx premium received (configurable)
- PROFIT_TARGET_AND_STOP_LOSS: Combined profit target and stop loss
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import calendar

from .greeks_calculator import GreeksCalculator
from .data_manager import get_lot_size, FNO_LOT_SIZES
from .metrics_calculator import MetricsCalculator
from .backtest_db import get_backtest_db
from .iv_percentile import get_iv_service, IVMetrics

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================

class StrikeMethod(Enum):
    DELTA_30 = "DELTA_30"
    DELTA_40 = "DELTA_40"
    OTM_2PCT = "OTM_2PCT"
    OTM_5PCT = "OTM_5PCT"
    ATM = "ATM"
    ADAPTIVE_DELTA = "ADAPTIVE_DELTA"  # IV Percentile-based delta selection
    ATR_BASED = "ATR_BASED"  # ATR-based OTM distance calculation
    BOLLINGER_UPPER = "BOLLINGER_UPPER"  # Strike at/near Upper Bollinger Band (resistance)
    PIVOT_R1 = "PIVOT_R1"  # Strike at R1 resistance (pivot point)
    PIVOT_R2 = "PIVOT_R2"  # Strike at R2 resistance (pivot point)


class ExitStrategy(Enum):
    HOLD_TO_EXPIRY = "HOLD_TO_EXPIRY"
    PROFIT_TARGET = "PROFIT_TARGET"
    STOP_LOSS = "STOP_LOSS"
    PROFIT_TARGET_AND_STOP_LOSS = "PROFIT_TARGET_AND_STOP_LOSS"


class TrendFilterMode(Enum):
    """
    Multi-timeframe EMA trend filter modes for entry conditions.

    NONE: No trend filter applied
    BEARISH: Only enter when 20 EMA < 50 EMA (bearish/neutral trend - original)
    BULLISH: Only enter when 20 EMA > 50 EMA (bullish trend - ride the trend)
    GOLDEN_CROSS: Only enter when 50 EMA > 200 EMA (long-term bullish)
    DEATH_CROSS: Only enter when 50 EMA < 200 EMA (long-term bearish)
    BULLISH_ALIGNED: Only enter when 20 > 50 > 200 EMA (full bullish alignment)
    """
    NONE = "NONE"
    BEARISH = "BEARISH"
    BULLISH = "BULLISH"
    GOLDEN_CROSS = "GOLDEN_CROSS"
    DEATH_CROSS = "DEATH_CROSS"
    BULLISH_ALIGNED = "BULLISH_ALIGNED"


class VWAPMode(Enum):
    """
    VWAP (Volume Weighted Average Price) filter modes for entry conditions.

    ABOVE: Only enter when price > VWAP (bullish - institutional buying support)
    BELOW: Only enter when price < VWAP (bearish - institutional selling)
    """
    ABOVE = "ABOVE"
    BELOW = "BELOW"


@dataclass
class BacktestConfig:
    """Configuration for a backtest run"""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    strike_method: StrikeMethod
    exit_strategy: ExitStrategy
    initial_capital: float = 1000000.0
    position_size: int = 1  # Number of lots per position
    risk_free_rate: float = 0.07
    default_iv: float = 0.20  # 20% IV if not available
    # Customizable exit parameters
    profit_target_pct: float = 50.0  # Close when X% of max profit captured
    stop_loss_multiple: float = 2.0  # Close when loss exceeds Nx premium received
    # Stop-loss adjustment (roll-up)
    allow_sl_adjustment: bool = False  # If True, roll up to higher strike on SL instead of closing
    # Trend filter - DEPRECATED: use trend_filter_mode instead
    use_trend_filter: bool = False  # Legacy: If True, only enter when 20 EMA < 50 EMA (bearish/neutral)
    # Multi-timeframe EMA trend filter mode
    trend_filter_mode: str = "NONE"  # NONE, BEARISH, BULLISH, GOLDEN_CROSS, DEATH_CROSS, BULLISH_ALIGNED
    # RSI filter
    use_rsi_filter: bool = False  # If True, only enter when RSI is within range
    rsi_period: int = 14  # RSI calculation period
    rsi_min: float = 40.0  # Minimum RSI for entry (avoid oversold)
    rsi_max: float = 70.0  # Maximum RSI for entry (avoid overbought)
    # ATR-based strike selection
    atr_multiplier: float = 1.5  # Strike = Current Price + (ATR x Multiplier)
    # Stochastic oscillator filter
    use_stochastic_filter: bool = False  # If True, only enter on stochastic sell signal
    stochastic_k_period: int = 14  # %K lookback period
    stochastic_d_period: int = 3  # %D smoothing period
    stochastic_smoothing: int = 3  # %K smoothing (Slow Stochastic)
    stochastic_overbought: float = 70.0  # Overbought threshold
    # Advanced exit strategies - DTE-based exit
    use_dte_exit: bool = False  # Exit when DTE falls below threshold
    dte_exit_threshold: int = 7  # Days to expiry to trigger exit
    # Advanced exit strategies - Trailing stop
    use_trailing_stop: bool = False  # Enable trailing stop on profits
    trailing_stop_activation: float = 25.0  # Activate trail at X% profit (of max premium)
    trailing_stop_distance: float = 15.0  # Trail distance in % of max profit
    # Supertrend filter
    use_supertrend_filter: bool = False  # If True, only enter when price > Supertrend (bullish)
    supertrend_period: int = 10  # ATR period for Supertrend
    supertrend_multiplier: float = 3.0  # ATR multiplier for Supertrend
    # VWAP filter
    use_vwap_filter: bool = False  # If True, filter entries based on VWAP
    vwap_mode: str = "ABOVE"  # ABOVE (price > VWAP = bullish) or BELOW
    vwap_period: int = 1  # Rolling period in days (1 = intraday/daily reset, >1 = rolling)
    # ADX (Average Directional Index) filter - trend strength indicator
    use_adx_filter: bool = False  # If True, only enter when ADX > threshold (trending market)
    adx_period: int = 14  # ADX calculation period
    adx_threshold: float = 25.0  # Minimum ADX value for entry (>25 = trending market)
    adx_require_bullish: bool = True  # If True, also require +DI > -DI (uptrend)
    # Williams %R filter - momentum oscillator
    use_williams_filter: bool = False  # If True, only enter when Williams %R drops from overbought
    williams_period: int = 14  # Williams %R lookback period
    williams_overbought: float = -20.0  # Above -20 is overbought (0 to -20 range)
    williams_oversold: float = -80.0  # Below -80 is oversold (-80 to -100 range)
    # MACD filter - Moving Average Convergence Divergence
    use_macd_filter: bool = False  # If True, apply MACD-based entry filter
    macd_fast: int = 12  # Fast EMA period
    macd_slow: int = 26  # Slow EMA period
    macd_signal: int = 9  # Signal line EMA period
    macd_mode: str = "BULLISH"  # BULLISH (MACD>Signal) or REVERSAL (histogram turning negative)
    # Bollinger Bands filter and strike selection
    use_bollinger_filter: bool = False  # If True, only enter when price is below upper band (not overbought)
    bollinger_period: int = 20  # SMA period for middle band (standard: 20)
    bollinger_std: float = 2.0  # Standard deviations for bands (standard: 2.0)


@dataclass
class Position:
    """Represents an open covered call position"""
    symbol: str
    lot_size: int
    entry_date: datetime
    stock_entry_price: float
    strike_price: float
    premium_received: float
    expiry_date: datetime
    delta_at_entry: float = 0.0
    theta_at_entry: float = 0.0
    iv_at_entry: float = 0.0
    # Adjustment tracking
    adjusted: bool = False  # True if position has been rolled up once
    original_premium: float = 0.0  # Original premium before adjustment
    adjustment_date: datetime = None  # Date when adjustment was made
    adjustment_cost: float = 0.0  # Cost to buy back original option
    # Trailing stop tracking
    profit_high_water_mark: float = 0.0  # Highest profit % achieved (for trailing stop)
    trailing_stop_active: bool = False  # True when trailing stop is activated


@dataclass
class Trade:
    """Completed trade record"""
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
    delta_at_entry: float = 0.0
    theta_at_entry: float = 0.0
    iv_at_entry: float = 0.0
    strike_method: str = ""
    exit_strategy: str = ""


# =============================================================================
# Covered Call Engine
# =============================================================================

class CoveredCallEngine:
    """
    Core backtesting engine for covered call strategies

    Simulates:
    1. Monthly option selling cycle
    2. Strike selection based on method
    3. Premium collection
    4. Early exit based on strategy
    5. Assignment handling at expiry
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize the engine

        Args:
            config: BacktestConfig with all parameters
        """
        self.config = config
        self.greeks = GreeksCalculator(risk_free_rate=config.risk_free_rate)
        self.metrics = MetricsCalculator(risk_free_rate=config.risk_free_rate)
        self.iv_service = get_iv_service()

        # State
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: Dict[datetime, float] = {}
        self.cash = config.initial_capital
        self.stock_holdings: Dict[str, Dict] = {}  # symbol -> {qty, avg_price}
        self.iv_history: Dict[str, List[IVMetrics]] = {}  # Track IV metrics per symbol

    def run_backtest(
        self,
        stock_data: Dict[str, pd.DataFrame],
        progress_callback=None
    ) -> Dict:
        """
        Run the complete backtest

        Args:
            stock_data: Dict mapping symbol to DataFrame with OHLCV data
            progress_callback: Optional callback(progress_pct, message)

        Returns:
            Dict with all results including metrics, trades, equity curve
        """
        logger.info(f"Starting backtest: {len(self.config.symbols)} symbols, "
                    f"{self.config.start_date.date()} to {self.config.end_date.date()}")

        # Generate all trading dates
        all_dates = self._get_trading_dates(stock_data)
        total_days = len(all_dates)

        logger.info(f"Total trading days: {total_days}")

        # Process each day
        for idx, current_date in enumerate(all_dates):
            if progress_callback and idx % 10 == 0:
                pct = (idx / total_days) * 100
                progress_callback(pct, f"Processing {current_date.date()}")

            self._process_day(current_date, stock_data)

        # Close any remaining positions at end
        self._close_all_positions(self.config.end_date, stock_data)

        # Calculate final metrics
        results = self._compile_results(stock_data)

        logger.info(f"Backtest completed: {len(self.trades)} trades, "
                    f"Total Return: {results['metrics']['total_return']:.2f}%")

        return results

    def _get_trading_dates(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> List[datetime]:
        """Get all unique trading dates from stock data"""
        all_dates = set()
        for symbol, df in stock_data.items():
            dates = pd.to_datetime(df.index).tolist()
            all_dates.update(dates)

        # Filter to date range
        filtered = [
            d for d in all_dates
            if self.config.start_date <= d <= self.config.end_date
        ]

        return sorted(filtered)

    def _process_day(
        self,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ):
        """Process a single trading day"""

        # 1. Check for expiries and handle them
        self._check_expiries(current_date, stock_data)

        # 2. Check early exit conditions for open positions
        # Run if exit_strategy is not HOLD_TO_EXPIRY, OR if advanced exits are enabled
        has_advanced_exits = (
            self.config.use_dte_exit or
            self.config.use_trailing_stop
        )
        if self.config.exit_strategy != ExitStrategy.HOLD_TO_EXPIRY or has_advanced_exits:
            self._check_early_exits(current_date, stock_data)

        # 3. Check for new entry opportunities (monthly cycle)
        self._check_new_entries(current_date, stock_data)

        # 4. Update equity curve
        self._update_equity(current_date, stock_data)

    def _check_expiries(
        self,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ):
        """Check and handle option expiries"""
        expired_symbols = []

        for symbol, position in self.positions.items():
            if current_date >= position.expiry_date:
                expired_symbols.append(symbol)

        for symbol in expired_symbols:
            self._handle_expiry(symbol, current_date, stock_data)

    def _handle_expiry(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ):
        """Handle option expiry - check for assignment"""
        position = self.positions[symbol]

        # Get current stock price
        stock_price = self._get_price(symbol, current_date, stock_data)
        if stock_price is None:
            # Use last known price
            stock_price = position.stock_entry_price

        # Determine if assigned (stock above strike)
        if stock_price >= position.strike_price:
            # ASSIGNED - stock called away at strike
            exit_reason = "ASSIGNED"
            exit_stock_price = position.strike_price  # Sold at strike
            option_exit_price = 0  # Option expires worthless for holder
        else:
            # EXPIRED - option expires worthless, keep stock
            exit_reason = "EXPIRY"
            exit_stock_price = stock_price
            option_exit_price = 0

        self._close_position(
            symbol, current_date, exit_stock_price,
            option_exit_price, exit_reason
        )

    def _check_early_exits(
        self,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ):
        """Check for early exit conditions"""
        symbols_to_close = []
        symbols_to_adjust = []

        # Determine which exit rules to apply
        strategy = self.config.exit_strategy
        check_profit_target = strategy in [
            ExitStrategy.PROFIT_TARGET,
            ExitStrategy.PROFIT_TARGET_AND_STOP_LOSS
        ]
        check_stop_loss = strategy in [
            ExitStrategy.STOP_LOSS,
            ExitStrategy.PROFIT_TARGET_AND_STOP_LOSS
        ]

        for symbol, position in self.positions.items():
            stock_price = self._get_price(symbol, current_date, stock_data)
            if stock_price is None:
                continue

            # Calculate current option value
            dte = (position.expiry_date - current_date).days
            if dte <= 0:
                continue

            iv = position.iv_at_entry
            time_to_expiry = dte / 365.0

            current_option_price = self.greeks.calculate_option_price(
                spot=stock_price,
                strike=position.strike_price,
                time_to_expiry=time_to_expiry,
                volatility=iv,
                option_type='CE'
            )

            # Calculate P&L on option
            option_pnl = position.premium_received - current_option_price
            max_profit = position.premium_received
            profit_pct = (option_pnl / max_profit) * 100 if max_profit > 0 else 0

            # === ADVANCED EXIT: DTE-based exit ===
            # Close position when DTE falls below threshold to avoid gamma risk
            if self.config.use_dte_exit:
                if dte <= self.config.dte_exit_threshold:
                    symbols_to_close.append((
                        symbol, stock_price, current_option_price,
                        f"DTE_EXIT_{dte}D"
                    ))
                    logger.debug(f"DTE exit triggered for {symbol}: DTE={dte} <= {self.config.dte_exit_threshold}")
                    continue

            # === ADVANCED EXIT: Trailing stop ===
            # Lock in profits as option decays
            if self.config.use_trailing_stop:
                # Update high water mark if we're in profit
                if profit_pct > position.profit_high_water_mark:
                    position.profit_high_water_mark = profit_pct

                # Check if trailing stop should be activated
                if not position.trailing_stop_active:
                    if profit_pct >= self.config.trailing_stop_activation:
                        position.trailing_stop_active = True
                        logger.debug(f"Trailing stop activated for {symbol}: "
                                    f"Profit={profit_pct:.1f}% >= {self.config.trailing_stop_activation}%")

                # If trailing stop is active, check if we should exit
                if position.trailing_stop_active:
                    # Calculate how much we've given back from the high
                    drawdown_from_high = position.profit_high_water_mark - profit_pct

                    if drawdown_from_high >= self.config.trailing_stop_distance:
                        symbols_to_close.append((
                            symbol, stock_price, current_option_price,
                            f"TRAILING_STOP_{int(position.profit_high_water_mark)}PCT_HWM"
                        ))
                        logger.debug(f"Trailing stop exit for {symbol}: "
                                    f"HWM={position.profit_high_water_mark:.1f}%, "
                                    f"Current={profit_pct:.1f}%, "
                                    f"Drawdown={drawdown_from_high:.1f}%")
                        continue

            # Check profit target
            if check_profit_target:
                target_pct = self.config.profit_target_pct / 100.0
                if option_pnl >= max_profit * target_pct:
                    symbols_to_close.append((
                        symbol, stock_price, current_option_price,
                        f"PROFIT_TARGET_{int(self.config.profit_target_pct)}PCT"
                    ))
                    continue

            # Check stop loss
            if check_stop_loss:
                stop_multiple = self.config.stop_loss_multiple
                if option_pnl <= -stop_multiple * position.premium_received:
                    # Check if we should adjust instead of closing
                    if self.config.allow_sl_adjustment and not position.adjusted:
                        # Roll up to higher strike instead of closing
                        premium_lost = abs(option_pnl)
                        symbols_to_adjust.append((
                            symbol, stock_price, current_option_price,
                            premium_lost, time_to_expiry, iv
                        ))
                    else:
                        # Close the position (or already adjusted once)
                        reason = f"STOP_LOSS_{stop_multiple}X"
                        if position.adjusted:
                            reason += "_AFTER_ADJ"
                        symbols_to_close.append((
                            symbol, stock_price, current_option_price, reason
                        ))

        # Process adjustments first
        for symbol, stock_price, buyback_price, premium_lost, tte, iv in symbols_to_adjust:
            self._adjust_position(
                symbol, current_date, stock_price, buyback_price,
                premium_lost, tte, iv
            )

        # Then close remaining positions
        for symbol, stock_price, option_price, reason in symbols_to_close:
            self._close_position(symbol, current_date, stock_price, option_price, reason)

    def _adjust_position(
        self,
        symbol: str,
        current_date: datetime,
        stock_price: float,
        buyback_price: float,
        premium_lost: float,
        time_to_expiry: float,
        iv: float
    ):
        """
        Adjust position by rolling up to a higher strike when stop loss is hit.

        The new strike is selected to be further OTM by approximately the lost premium,
        to attempt to recover the loss with additional time decay.
        """
        position = self.positions[symbol]

        # Save original premium if not already saved
        if position.original_premium == 0:
            position.original_premium = position.premium_received

        # Calculate how much further OTM we need to go
        # New strike should be current strike + (premium_lost / delta estimate)
        # Simplified: move strike up by premium_lost as percentage of stock price
        otm_increase = premium_lost / stock_price  # e.g., lost ₹50 on ₹1000 stock = 5% more OTM

        # Generate available strikes
        strike_interval = self._get_strike_interval(stock_price)
        strikes = self._generate_strikes(stock_price, strike_interval)

        # Find new strike that is at least current_strike + (otm_increase * stock_price)
        min_new_strike = position.strike_price + (otm_increase * stock_price)

        # Filter to strikes above minimum
        valid_strikes = [s for s in strikes if s >= min_new_strike]

        if not valid_strikes:
            # Can't find a suitable strike, just take the highest available
            new_strike = max(strikes)
        else:
            # Take the closest valid strike
            new_strike = min(valid_strikes)

        # Calculate new option premium
        new_premium = self.greeks.calculate_option_price(
            spot=stock_price,
            strike=new_strike,
            time_to_expiry=time_to_expiry,
            volatility=iv,
            option_type='CE'
        )

        # Calculate new delta
        new_delta = self.greeks.calculate_delta(
            spot=stock_price,
            strike=new_strike,
            time_to_expiry=time_to_expiry,
            volatility=iv,
            option_type='CE'
        )

        # Update cash: pay to buy back old option, receive new premium
        self.cash -= buyback_price * position.lot_size  # Buy back old
        self.cash += new_premium * position.lot_size     # Sell new

        logger.info(f"ADJUSTMENT {symbol}: Rolled from {position.strike_price:.0f} to {new_strike:.0f}, "
                    f"Buyback: {buyback_price:.2f}, New Premium: {new_premium:.2f}, "
                    f"Net Cost: {(buyback_price - new_premium):.2f}")

        # Update position
        position.adjustment_date = current_date
        position.adjustment_cost = buyback_price
        position.strike_price = new_strike
        position.premium_received = new_premium
        position.delta_at_entry = new_delta
        position.adjusted = True

    def _check_new_entries(
        self,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ):
        """Check for new entry opportunities"""
        # Only enter on monthly expiry cycle
        # NSE options expire on last Thursday of month
        if not self._is_entry_day(current_date):
            return

        for symbol in self.config.symbols:
            # Skip if already have position
            if symbol in self.positions:
                continue

            stock_price = self._get_price(symbol, current_date, stock_data)
            if stock_price is None:
                continue

            # Apply multi-timeframe trend filter based on mode
            if not self._check_multi_timeframe_trend_filter(symbol, current_date, stock_data):
                continue  # Skip entry when trend filter fails

            # Apply RSI filter if enabled (RSI between min and max)
            if self.config.use_rsi_filter:
                if not self._check_rsi_filter(symbol, current_date, stock_data):
                    continue  # Skip entry when RSI filter fails

            # Apply Stochastic filter if enabled (%K crossing below %D from overbought)
            if self.config.use_stochastic_filter:
                if not self._check_stochastic_filter(symbol, current_date, stock_data):
                    continue  # Skip entry when Stochastic filter fails

            # Apply Supertrend filter if enabled (price > Supertrend = bullish)
            if self.config.use_supertrend_filter:
                if not self._check_supertrend_filter(symbol, current_date, stock_data):
                    continue  # Skip entry when Supertrend filter fails

            # Apply VWAP filter if enabled (price vs VWAP for institutional support)
            if self.config.use_vwap_filter:
                if not self._check_vwap_filter(symbol, current_date, stock_data):
                    continue  # Skip entry when VWAP filter fails

            # Apply ADX filter if enabled (trend strength - ADX > threshold)
            if self.config.use_adx_filter:
                if not self._check_adx_filter(symbol, current_date, stock_data):
                    continue  # Skip entry when ADX filter fails

            # Apply Bollinger Bands filter if enabled (price below upper band = not overbought)
            if self.config.use_bollinger_filter:
                if not self._check_bollinger_filter(symbol, current_date, stock_data):
                    continue  # Skip entry when Bollinger filter fails

            # Apply MACD filter if enabled (MACD-based momentum filter)
            if self.config.use_macd_filter:
                if not self._check_macd_filter(symbol, current_date, stock_data):
                    continue  # Skip entry when MACD filter fails

            # Apply Williams %R filter if enabled (momentum turning from overbought)
            if self.config.use_williams_filter:
                if not self._check_williams_filter(symbol, current_date, stock_data):
                    continue  # Skip entry when Williams %R filter fails

            # Check if we have enough capital
            lot_size = get_lot_size(symbol)
            required_capital = stock_price * lot_size * self.config.position_size

            if self.cash < required_capital:
                continue

            # Open new position
            self._open_position(symbol, current_date, stock_price, stock_data)

    def _is_entry_day(self, current_date: datetime) -> bool:
        """
        Check if this is a valid entry day (start of new monthly cycle)

        Enters at the beginning of each month or when previous option expires
        """
        # Entry on first trading day of month
        if current_date.day <= 5:
            # Check if this is first occurrence this month
            return True

        return False

    def _get_monthly_expiry(self, current_date: datetime) -> datetime:
        """Get the monthly expiry date (last Thursday of month)"""
        year = current_date.year
        month = current_date.month

        # If we're past 20th, look at next month
        if current_date.day > 20:
            month += 1
            if month > 12:
                month = 1
                year += 1

        # Find last Thursday
        last_day = calendar.monthrange(year, month)[1]
        last_date = datetime(year, month, last_day)

        # Work backwards to find Thursday (weekday 3)
        while last_date.weekday() != 3:
            last_date -= timedelta(days=1)

        return last_date

    def _open_position(
        self,
        symbol: str,
        entry_date: datetime,
        stock_price: float,
        stock_data: Dict[str, pd.DataFrame]
    ):
        """Open a new covered call position"""
        lot_size = get_lot_size(symbol)

        # Get expiry date
        expiry_date = self._get_monthly_expiry(entry_date)
        dte = (expiry_date - entry_date).days

        if dte < 7:
            # Too close to expiry, skip
            return

        time_to_expiry = dte / 365.0

        # Calculate IV metrics for adaptive strategy
        iv_metrics = None
        if self.config.strike_method == StrikeMethod.ADAPTIVE_DELTA:
            if symbol in stock_data:
                iv_metrics = self.iv_service.calculate_metrics(
                    symbol, stock_data[symbol], entry_date
                )
                # Track IV history
                if symbol not in self.iv_history:
                    self.iv_history[symbol] = []
                self.iv_history[symbol].append(iv_metrics)

        # Calculate IV from historical volatility
        if iv_metrics:
            iv = iv_metrics.current_iv
        else:
            iv = self._estimate_iv(symbol, entry_date, stock_data)

        # Select strike based on method
        strike_price = self._select_strike(
            symbol, stock_price, time_to_expiry, iv, iv_metrics,
            entry_date, stock_data
        )

        # Calculate option premium
        premium = self.greeks.calculate_option_price(
            spot=stock_price,
            strike=strike_price,
            time_to_expiry=time_to_expiry,
            volatility=iv,
            option_type='CE'
        )

        # Calculate Greeks
        delta = self.greeks.calculate_delta(
            spot=stock_price,
            strike=strike_price,
            time_to_expiry=time_to_expiry,
            volatility=iv,
            option_type='CE'
        )

        theta = self.greeks.calculate_theta(
            spot=stock_price,
            strike=strike_price,
            time_to_expiry=time_to_expiry,
            volatility=iv,
            option_type='CE'
        )

        # Create position
        position = Position(
            symbol=symbol,
            lot_size=lot_size * self.config.position_size,
            entry_date=entry_date,
            stock_entry_price=stock_price,
            strike_price=strike_price,
            premium_received=premium,
            expiry_date=expiry_date,
            delta_at_entry=delta,
            theta_at_entry=theta,
            iv_at_entry=iv
        )

        self.positions[symbol] = position

        # Update cash (buy stock, receive premium)
        stock_cost = stock_price * position.lot_size
        premium_received = premium * position.lot_size

        self.cash -= stock_cost
        self.cash += premium_received

        logger.debug(f"Opened position: {symbol} @ {stock_price:.2f}, "
                     f"Strike: {strike_price:.2f}, Premium: {premium:.2f}")

    def _close_position(
        self,
        symbol: str,
        exit_date: datetime,
        stock_exit_price: float,
        option_exit_price: float,
        exit_reason: str
    ):
        """Close a position and record the trade"""
        position = self.positions.pop(symbol)

        # Calculate P&L
        stock_pnl = (stock_exit_price - position.stock_entry_price) * position.lot_size
        option_pnl = (position.premium_received - option_exit_price) * position.lot_size
        total_pnl = stock_pnl + option_pnl

        # Return percentage based on initial stock investment
        initial_investment = position.stock_entry_price * position.lot_size
        return_pct = (total_pnl / initial_investment) * 100 if initial_investment > 0 else 0

        # Create trade record
        trade = Trade(
            symbol=symbol,
            lot_size=position.lot_size,
            entry_date=position.entry_date,
            stock_entry_price=position.stock_entry_price,
            strike_price=position.strike_price,
            premium_received=position.premium_received,
            expiry_date=position.expiry_date,
            exit_date=exit_date,
            stock_exit_price=stock_exit_price,
            option_exit_price=option_exit_price,
            exit_reason=exit_reason,
            stock_pnl=stock_pnl,
            option_pnl=option_pnl,
            total_pnl=total_pnl,
            return_pct=return_pct,
            delta_at_entry=position.delta_at_entry,
            theta_at_entry=position.theta_at_entry,
            iv_at_entry=position.iv_at_entry,
            strike_method=self.config.strike_method.value,
            exit_strategy=self.config.exit_strategy.value
        )

        self.trades.append(trade)

        # Update cash
        if exit_reason == "ASSIGNED":
            # Stock sold at strike
            self.cash += position.strike_price * position.lot_size
        else:
            # Stock still held, option bought back
            self.cash += stock_exit_price * position.lot_size
            self.cash -= option_exit_price * position.lot_size

        logger.debug(f"Closed position: {symbol}, Reason: {exit_reason}, "
                     f"P&L: {total_pnl:.2f}")

    def _close_all_positions(
        self,
        exit_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ):
        """Close all remaining positions at end of backtest"""
        symbols_to_close = list(self.positions.keys())

        for symbol in symbols_to_close:
            stock_price = self._get_price(symbol, exit_date, stock_data)
            if stock_price is None:
                stock_price = self.positions[symbol].stock_entry_price

            # Calculate current option value
            position = self.positions[symbol]
            dte = max((position.expiry_date - exit_date).days, 0)

            if dte > 0:
                time_to_expiry = dte / 365.0
                option_price = self.greeks.calculate_option_price(
                    spot=stock_price,
                    strike=position.strike_price,
                    time_to_expiry=time_to_expiry,
                    volatility=position.iv_at_entry,
                    option_type='CE'
                )
            else:
                option_price = max(0, stock_price - position.strike_price)

            self._close_position(
                symbol, exit_date, stock_price, option_price, "END_OF_BACKTEST"
            )

    def _select_strike(
        self,
        symbol: str,
        stock_price: float,
        time_to_expiry: float,
        iv: float,
        iv_metrics: IVMetrics = None,
        current_date: datetime = None,
        stock_data: Dict[str, pd.DataFrame] = None
    ) -> float:
        """Select strike based on configured method"""
        # Generate available strikes (rounded to nearest 50 for NIFTY-like stocks)
        strike_interval = self._get_strike_interval(stock_price)
        strikes = self._generate_strikes(stock_price, strike_interval)

        method = self.config.strike_method

        if method == StrikeMethod.ATR_BASED:
            # Calculate ATR-based strike: Current Price + (ATR x Multiplier)
            atr = self._calculate_atr(symbol, current_date, stock_data)
            target = stock_price + (atr * self.config.atr_multiplier)
            logger.debug(f"ATR-Based: Price={stock_price:.2f}, ATR={atr:.2f}, "
                        f"Multiplier={self.config.atr_multiplier}, Target={target:.2f}")
            return min(strikes, key=lambda x: abs(x - target))

        elif method == StrikeMethod.ADAPTIVE_DELTA:
            # Use IV percentile to determine target delta
            if iv_metrics:
                target_delta = iv_metrics.target_delta
                logger.debug(f"Adaptive: IV%={iv_metrics.iv_percentile:.1f}, "
                            f"Regime={iv_metrics.regime}, Delta={target_delta}")
            else:
                target_delta = 0.30  # Default if no IV metrics

            strike, _ = self.greeks.find_strike_by_delta(
                spot=stock_price,
                time_to_expiry=time_to_expiry,
                volatility=iv,
                target_delta=target_delta,
                available_strikes=strikes,
                option_type='CE'
            )
            return strike

        elif method == StrikeMethod.DELTA_30:
            target_delta = 0.30
            strike, _ = self.greeks.find_strike_by_delta(
                spot=stock_price,
                time_to_expiry=time_to_expiry,
                volatility=iv,
                target_delta=target_delta,
                available_strikes=strikes,
                option_type='CE'
            )
            return strike

        elif method == StrikeMethod.DELTA_40:
            target_delta = 0.40
            strike, _ = self.greeks.find_strike_by_delta(
                spot=stock_price,
                time_to_expiry=time_to_expiry,
                volatility=iv,
                target_delta=target_delta,
                available_strikes=strikes,
                option_type='CE'
            )
            return strike

        elif method == StrikeMethod.OTM_2PCT:
            target = stock_price * 1.02
            return min(strikes, key=lambda x: abs(x - target))

        elif method == StrikeMethod.OTM_5PCT:
            target = stock_price * 1.05
            return min(strikes, key=lambda x: abs(x - target))

        elif method == StrikeMethod.ATM:
            return min(strikes, key=lambda x: abs(x - stock_price))

        elif method == StrikeMethod.PIVOT_R1:
            # Calculate Pivot Points and use R1 as strike target
            pivots = self._calculate_pivot_points(symbol, current_date, stock_data)
            if pivots and 'r1' in pivots:
                target = pivots['r1']
                # Only use R1 if it's above current price (OTM)
                if target <= stock_price:
                    target = stock_price * 1.02  # Fallback to 2% OTM
                logger.debug(f"PIVOT_R1: Price={stock_price:.2f}, R1={pivots['r1']:.2f}, Target={target:.2f}")
                return min(strikes, key=lambda x: abs(x - target))
            else:
                # Fallback to 2% OTM if pivot calculation fails
                target = stock_price * 1.02
                return min(strikes, key=lambda x: abs(x - target))

        elif method == StrikeMethod.PIVOT_R2:
            # Calculate Pivot Points and use R2 as strike target
            pivots = self._calculate_pivot_points(symbol, current_date, stock_data)
            if pivots and 'r2' in pivots:
                target = pivots['r2']
                # Only use R2 if it's above current price (OTM)
                if target <= stock_price:
                    target = stock_price * 1.05  # Fallback to 5% OTM
                logger.debug(f"PIVOT_R2: Price={stock_price:.2f}, R2={pivots['r2']:.2f}, Target={target:.2f}")
                return min(strikes, key=lambda x: abs(x - target))
            else:
                # Fallback to 5% OTM if pivot calculation fails
                target = stock_price * 1.05
                return min(strikes, key=lambda x: abs(x - target))

        elif method == StrikeMethod.BOLLINGER_UPPER:
            # Select strike at or near the Upper Bollinger Band (natural resistance)
            # Upper Band = 20 SMA + (2 x StdDev) - acts as dynamic resistance
            if symbol in stock_data:
                df = stock_data[symbol]
                mask = df.index <= current_date
                historical_data = df.loc[mask]

                if len(historical_data) >= self.config.bollinger_period + 5:
                    upper_band, middle_band, _ = self._calculate_bollinger_bands(
                        historical_data,
                        period=self.config.bollinger_period,
                        std_dev=self.config.bollinger_std
                    )

                    current_upper = upper_band.iloc[-1]
                    current_middle = middle_band.iloc[-1]

                    if not pd.isna(current_upper) and current_upper > stock_price:
                        target = current_upper
                        logger.debug(f"BOLLINGER_UPPER: Price={stock_price:.2f}, "
                                    f"Middle={current_middle:.2f}, Upper={current_upper:.2f}, "
                                    f"Target={target:.2f}")
                        return min(strikes, key=lambda x: abs(x - target))

            # Fallback to 2% OTM if Bollinger calculation fails
            target = stock_price * 1.02
            logger.debug(f"BOLLINGER_UPPER fallback: Price={stock_price:.2f}, Target={target:.2f}")
            return min(strikes, key=lambda x: abs(x - target))

        else:
            # Default to OTM 2%
            target = stock_price * 1.02
            return min(strikes, key=lambda x: abs(x - target))

    def _generate_strikes(
        self,
        stock_price: float,
        interval: float
    ) -> List[float]:
        """Generate available strike prices around current price"""
        # Generate strikes from 90% to 120% of current price
        start = round((stock_price * 0.9) / interval) * interval
        end = round((stock_price * 1.2) / interval) * interval

        strikes = []
        current = start
        while current <= end:
            strikes.append(current)
            current += interval

        return strikes

    def _get_strike_interval(self, stock_price: float) -> float:
        """Determine strike interval based on stock price"""
        if stock_price < 100:
            return 2.5
        elif stock_price < 500:
            return 5
        elif stock_price < 1000:
            return 10
        elif stock_price < 2500:
            return 25
        elif stock_price < 5000:
            return 50
        else:
            return 100

    def _estimate_iv(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Estimate IV from historical volatility"""
        if symbol not in stock_data:
            return self.config.default_iv

        df = stock_data[symbol]

        # Get last 30 days of data
        mask = df.index <= current_date
        recent_data = df.loc[mask].tail(30)

        if len(recent_data) < 10:
            return self.config.default_iv

        # Calculate historical volatility
        returns = np.log(recent_data['close'] / recent_data['close'].shift(1)).dropna()
        hv = returns.std() * np.sqrt(252)

        # Add 20% premium for IV over HV (typical for options)
        iv = hv * 1.2

        # Clamp to reasonable range
        return max(0.10, min(0.80, iv))

    def _calculate_atr(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame],
        period: int = 14
    ) -> float:
        """
        Calculate Average True Range (ATR) for strike selection.

        ATR measures market volatility by decomposing the entire range of an asset
        price for that period. Used to determine OTM distance for strike selection.

        Args:
            symbol: Stock symbol
            current_date: Current date
            stock_data: Dict of stock DataFrames with OHLCV data
            period: ATR lookback period (default 14)

        Returns:
            ATR value, or 2% of stock price as fallback
        """
        if symbol not in stock_data:
            # Fallback: return 0 (will use stock price directly)
            return 0.0

        df = stock_data[symbol]

        # Get data up to current date
        mask = df.index <= current_date
        historical_data = df.loc[mask].tail(period + 1)

        if len(historical_data) < period + 1:
            # Not enough data, use simple range estimate
            if len(historical_data) > 0:
                avg_price = historical_data['close'].mean()
                return avg_price * 0.02  # 2% fallback
            return 0.0

        # Calculate True Range components
        high = historical_data['high']
        low = historical_data['low']
        close = historical_data['close']

        # True Range = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR as simple moving average of True Range
        atr = true_range.tail(period).mean()

        logger.debug(f"ATR for {symbol}: {atr:.2f} (period={period})")

        return atr

    def _calculate_pivot_points(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Calculate Standard Pivot Points for strike selection.

        Pivot Points are technical analysis indicators used to identify potential
        support and resistance levels. For covered calls, resistance levels (R1, R2)
        serve as natural strike price targets.

        Formulas (Standard Pivot Points):
        - Pivot (P) = (High + Low + Close) / 3
        - R1 = (2 x P) - Low
        - R2 = P + (High - Low)
        - R3 = High + 2 x (P - Low)
        - S1 = (2 x P) - High
        - S2 = P - (High - Low)
        - S3 = Low - 2 x (High - P)

        Args:
            symbol: Stock symbol
            current_date: Current date
            stock_data: Dict of stock DataFrames with OHLCV data

        Returns:
            Dict with pivot, r1, r2, r3, s1, s2, s3 values
        """
        if symbol not in stock_data:
            return {}

        df = stock_data[symbol]

        # Get data up to current date (use previous day's OHLC for pivot calculation)
        mask = df.index <= current_date
        historical_data = df.loc[mask].tail(2)

        if len(historical_data) < 1:
            return {}

        # Use the previous completed day for pivot calculation
        if len(historical_data) >= 2:
            prev_day = historical_data.iloc[-2]
        else:
            prev_day = historical_data.iloc[-1]

        high = prev_day['high']
        low = prev_day['low']
        close = prev_day['close']

        # Calculate Pivot Point
        pivot = (high + low + close) / 3

        # Calculate Resistance Levels
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        # Calculate Support Levels
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        logger.debug(f"Pivot Points for {symbol}: P={pivot:.2f}, R1={r1:.2f}, R2={r2:.2f}, R3={r3:.2f}")

        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            's1': s1,
            's2': s2,
            's3': s3
        }

    def _check_trend_filter(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Check if trend filter condition is met: 20 EMA < 50 EMA

        This filters entries to only occur in bearish/neutral trends,
        which can help avoid selling calls when stock is strongly bullish.

        Args:
            symbol: Stock symbol
            current_date: Current date
            stock_data: Dict of stock DataFrames

        Returns:
            True if trend filter passes (20 EMA < 50 EMA), False otherwise
        """
        if symbol not in stock_data:
            return True  # Allow entry if no data

        df = stock_data[symbol]

        # Get data up to current date
        mask = df.index <= current_date
        historical_data = df.loc[mask]

        if len(historical_data) < 50:
            # Not enough data for 50 EMA, allow entry
            return True

        # Calculate EMAs on closing prices
        close_prices = historical_data['close']
        ema_20 = close_prices.ewm(span=20, adjust=False).mean()
        ema_50 = close_prices.ewm(span=50, adjust=False).mean()

        # Get current EMA values
        current_ema_20 = ema_20.iloc[-1]
        current_ema_50 = ema_50.iloc[-1]

        # Trend filter passes when 20 EMA is below 50 EMA
        filter_passed = current_ema_20 < current_ema_50

        if not filter_passed:
            logger.debug(f"Trend filter blocked entry for {symbol}: "
                        f"EMA20={current_ema_20:.2f} >= EMA50={current_ema_50:.2f}")

        return filter_passed

    def _check_multi_timeframe_trend_filter(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Multi-timeframe EMA trend filter based on configured mode.

        Modes:
        - NONE: No filter, always allow entry
        - BEARISH: 20 EMA < 50 EMA (bearish/neutral trend)
        - BULLISH: 20 EMA > 50 EMA (bullish trend)
        - GOLDEN_CROSS: 50 EMA > 200 EMA (long-term bullish)
        - DEATH_CROSS: 50 EMA < 200 EMA (long-term bearish)
        - BULLISH_ALIGNED: 20 > 50 > 200 EMA (full bullish alignment)

        Args:
            symbol: Stock symbol
            current_date: Current date
            stock_data: Dict of stock DataFrames

        Returns:
            True if filter passes, False otherwise
        """
        # Handle legacy use_trend_filter flag for backward compatibility
        mode = self.config.trend_filter_mode
        if mode == "NONE" and self.config.use_trend_filter:
            mode = "BEARISH"  # Legacy behavior

        if mode == "NONE":
            return True  # No filter applied

        if symbol not in stock_data:
            return True  # Allow entry if no data

        df = stock_data[symbol]

        # Get data up to current date
        mask = df.index <= current_date
        historical_data = df.loc[mask]

        # Calculate required EMAs on closing prices
        close_prices = historical_data['close']

        # Check data requirements based on mode
        if mode in ["GOLDEN_CROSS", "DEATH_CROSS", "BULLISH_ALIGNED"]:
            if len(historical_data) < 200:
                return True  # Not enough data for 200 EMA, allow entry
            ema_200 = close_prices.ewm(span=200, adjust=False).mean().iloc[-1]
        else:
            ema_200 = None

        if len(historical_data) < 50:
            return True  # Not enough data for 50 EMA, allow entry

        ema_20 = close_prices.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = close_prices.ewm(span=50, adjust=False).mean().iloc[-1]

        # Apply filter based on mode
        filter_passed = False

        if mode == "BEARISH":
            # Only enter when 20 EMA < 50 EMA (bearish/neutral)
            filter_passed = ema_20 < ema_50
            if not filter_passed:
                logger.debug(f"BEARISH filter blocked {symbol}: "
                            f"EMA20={ema_20:.2f} >= EMA50={ema_50:.2f}")

        elif mode == "BULLISH":
            # Only enter when 20 EMA > 50 EMA (bullish)
            filter_passed = ema_20 > ema_50
            if not filter_passed:
                logger.debug(f"BULLISH filter blocked {symbol}: "
                            f"EMA20={ema_20:.2f} <= EMA50={ema_50:.2f}")

        elif mode == "GOLDEN_CROSS":
            # Only enter when 50 EMA > 200 EMA (long-term bullish)
            filter_passed = ema_50 > ema_200
            if not filter_passed:
                logger.debug(f"GOLDEN_CROSS filter blocked {symbol}: "
                            f"EMA50={ema_50:.2f} <= EMA200={ema_200:.2f}")

        elif mode == "DEATH_CROSS":
            # Only enter when 50 EMA < 200 EMA (long-term bearish)
            filter_passed = ema_50 < ema_200
            if not filter_passed:
                logger.debug(f"DEATH_CROSS filter blocked {symbol}: "
                            f"EMA50={ema_50:.2f} >= EMA200={ema_200:.2f}")

        elif mode == "BULLISH_ALIGNED":
            # Only enter when 20 > 50 > 200 EMA (full bullish alignment)
            filter_passed = ema_20 > ema_50 > ema_200
            if not filter_passed:
                logger.debug(f"BULLISH_ALIGNED filter blocked {symbol}: "
                            f"EMA20={ema_20:.2f}, EMA50={ema_50:.2f}, EMA200={ema_200:.2f}")

        else:
            # Unknown mode, allow entry
            filter_passed = True

        return filter_passed

    def _calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smoothing: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Slow Stochastic Oscillator (%K and %D).

        Slow Stochastic uses smoothed %K (which becomes the slow %K)
        and then smooths that to get %D.

        Formula:
        - Raw %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        - Slow %K = SMA(Raw %K, smoothing)  # This is what we display as %K
        - %D = SMA(Slow %K, d_period)

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            k_period: Lookback period for %K (default 14)
            d_period: %D smoothing period (default 3)
            smoothing: %K smoothing period for Slow Stochastic (default 3)

        Returns:
            Tuple of (slow_k, slow_d) Series
        """
        # Calculate lowest low and highest high over k_period
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()

        # Calculate raw %K (Fast Stochastic %K)
        # Avoid division by zero
        range_hl = highest_high - lowest_low
        range_hl = range_hl.replace(0, np.nan)

        raw_k = 100 * (df['close'] - lowest_low) / range_hl

        # Slow %K = SMA of Raw %K
        slow_k = raw_k.rolling(window=smoothing).mean()

        # %D = SMA of Slow %K
        slow_d = slow_k.rolling(window=d_period).mean()

        return slow_k, slow_d

    def _check_stochastic_filter(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Check if Stochastic oscillator gives a sell signal for covered calls.

        Sell signal for covered calls (good time to sell premium):
        - %K was recently above overbought level (70 by default)
        - %K just crossed below %D (bearish crossover)
        - This indicates momentum is turning, stock may consolidate or decline
        - Good environment for selling calls (premium decay)

        Args:
            symbol: Stock symbol
            current_date: Current date
            stock_data: Dict of stock DataFrames

        Returns:
            True if stochastic filter passes (sell signal), False otherwise
        """
        if symbol not in stock_data:
            return True  # Allow entry if no data

        df = stock_data[symbol]

        # Get data up to current date
        mask = df.index <= current_date
        historical_data = df.loc[mask].copy()

        # Need enough data for stochastic calculation
        min_required = self.config.stochastic_k_period + self.config.stochastic_smoothing + self.config.stochastic_d_period + 5
        if len(historical_data) < min_required:
            logger.debug(f"Stochastic filter: Not enough data for {symbol}, allowing entry")
            return True

        # Calculate Slow Stochastic
        slow_k, slow_d = self._calculate_stochastic(
            historical_data,
            k_period=self.config.stochastic_k_period,
            d_period=self.config.stochastic_d_period,
            smoothing=self.config.stochastic_smoothing
        )

        # Get current and previous values
        if len(slow_k) < 2 or len(slow_d) < 2:
            return True

        current_k = slow_k.iloc[-1]
        previous_k = slow_k.iloc[-2]
        current_d = slow_d.iloc[-1]
        previous_d = slow_d.iloc[-2]

        # Check for NaN values
        if pd.isna(current_k) or pd.isna(previous_k) or pd.isna(current_d) or pd.isna(previous_d):
            logger.debug(f"Stochastic filter: NaN values for {symbol}, allowing entry")
            return True

        overbought = self.config.stochastic_overbought

        # Check for sell signal:
        # 1. Previous %K was above overbought OR current %K still above overbought threshold - 10
        #    (was recently overbought)
        # 2. %K just crossed below %D (bearish crossover)
        was_overbought = previous_k > overbought or current_k > (overbought - 10)
        bearish_crossover = previous_k >= previous_d and current_k < current_d

        filter_passed = was_overbought and bearish_crossover

        if filter_passed:
            logger.debug(f"Stochastic SELL signal for {symbol}: "
                        f"K={current_k:.1f}, D={current_d:.1f}, "
                        f"prev_K={previous_k:.1f}, prev_D={previous_d:.1f}")
        else:
            logger.debug(f"Stochastic filter blocked entry for {symbol}: "
                        f"K={current_k:.1f}, D={current_d:.1f}, "
                        f"was_overbought={was_overbought}, crossover={bearish_crossover}")

        return filter_passed

    def _calculate_rsi(
        self,
        prices: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over the period

        Args:
            prices: Series of closing prices
            period: RSI calculation period (default 14)

        Returns:
            Series of RSI values
        """
        delta = prices.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)
        avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _check_rsi_filter(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Check if RSI filter condition is met: RSI between rsi_min and rsi_max

        RSI between 40-70 indicates bullish but not overbought conditions,
        which is ideal for covered call writing.

        Returns:
            True if RSI filter passes (RSI within range), False otherwise
        """
        if symbol not in stock_data:
            return True
        df = stock_data[symbol]
        mask = df.index <= current_date
        historical_data = df.loc[mask]
        min_periods = self.config.rsi_period + 10
        if len(historical_data) < min_periods:
            return True
        close_prices = historical_data['close']
        rsi = self._calculate_rsi(close_prices, self.config.rsi_period)
        current_rsi = rsi.iloc[-1]
        if pd.isna(current_rsi):
            return True
        filter_passed = self.config.rsi_min <= current_rsi <= self.config.rsi_max
        if not filter_passed:
            logger.debug(f"RSI filter blocked entry for {symbol}: "
                        f"RSI={current_rsi:.2f} (range: {self.config.rsi_min}-{self.config.rsi_max})")
        return filter_passed

    def _calculate_supertrend(
        self,
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Supertrend indicator.

        Supertrend is a trend-following indicator that uses ATR to determine
        trend direction. When price is above Supertrend, the trend is bullish.
        When price is below, the trend is bearish.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period (default 10)
            multiplier: ATR multiplier (default 3.0)

        Returns:
            Tuple of (supertrend, trend_direction) Series
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        hl2 = (high + low) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        final_upper = pd.Series(index=df.index, dtype=float)
        final_lower = pd.Series(index=df.index, dtype=float)
        supertrend = pd.Series(index=df.index, dtype=float)
        trend_direction = pd.Series(index=df.index, dtype=int)

        start_idx = period - 1 if period > 0 else 0

        for i in range(len(df)):
            if i < start_idx:
                final_upper.iloc[i] = np.nan
                final_lower.iloc[i] = np.nan
                supertrend.iloc[i] = np.nan
                trend_direction.iloc[i] = 0
                continue

            if i == start_idx:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                    final_upper.iloc[i] = basic_upper.iloc[i]
                else:
                    final_upper.iloc[i] = final_upper.iloc[i-1]

            if i == start_idx:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                    final_lower.iloc[i] = basic_lower.iloc[i]
                else:
                    final_lower.iloc[i] = final_lower.iloc[i-1]

            if i == start_idx:
                if close.iloc[i] <= final_upper.iloc[i]:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    trend_direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = final_lower.iloc[i]
                    trend_direction.iloc[i] = 1
            else:
                prev_trend = trend_direction.iloc[i-1]
                if prev_trend == -1:
                    if close.iloc[i] > final_upper.iloc[i]:
                        supertrend.iloc[i] = final_lower.iloc[i]
                        trend_direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = final_upper.iloc[i]
                        trend_direction.iloc[i] = -1
                else:
                    if close.iloc[i] < final_lower.iloc[i]:
                        supertrend.iloc[i] = final_upper.iloc[i]
                        trend_direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = final_lower.iloc[i]
                        trend_direction.iloc[i] = 1

        return supertrend, trend_direction

    def _check_supertrend_filter(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Check if Supertrend filter condition is met: price > Supertrend (bullish trend).

        Args:
            symbol: Stock symbol
            current_date: Current date
            stock_data: Dict of stock DataFrames

        Returns:
            True if Supertrend filter passes (bullish), False otherwise
        """
        if symbol not in stock_data:
            return True

        df = stock_data[symbol]
        mask = df.index <= current_date
        historical_data = df.loc[mask].copy()

        min_required = self.config.supertrend_period + 10
        if len(historical_data) < min_required:
            logger.debug(f"Supertrend filter: Not enough data for {symbol}, allowing entry")
            return True

        supertrend, trend_direction = self._calculate_supertrend(
            historical_data,
            period=self.config.supertrend_period,
            multiplier=self.config.supertrend_multiplier
        )

        current_close = historical_data['close'].iloc[-1]
        current_supertrend = supertrend.iloc[-1]
        current_trend = trend_direction.iloc[-1]

        if pd.isna(current_supertrend) or pd.isna(current_trend):
            logger.debug(f"Supertrend filter: NaN values for {symbol}, allowing entry")
            return True

        filter_passed = current_trend == 1

        if filter_passed:
            logger.debug(f"Supertrend BULLISH for {symbol}: Close={current_close:.2f}, ST={current_supertrend:.2f}")
        else:
            logger.debug(f"Supertrend filter blocked entry for {symbol}: Close={current_close:.2f}, ST={current_supertrend:.2f} (bearish)")

        return filter_passed

    def _calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Bollinger Bands consist of:
        - Middle Band: 20-period Simple Moving Average (SMA)
        - Upper Band: Middle Band + (2 x Standard Deviation)
        - Lower Band: Middle Band - (2 x Standard Deviation)

        For covered calls, the Upper Band serves as a natural resistance level
        and can be used for strike selection.

        Args:
            df: DataFrame with 'close' column
            period: SMA period (default 20)
            std_dev: Number of standard deviations (default 2.0)

        Returns:
            Tuple of (upper_band, middle_band, lower_band) Series
        """
        close = df['close']

        # Middle Band = 20-period SMA
        middle_band = close.rolling(window=period).mean()

        # Standard Deviation
        rolling_std = close.rolling(window=period).std()

        # Upper and Lower Bands
        upper_band = middle_band + (std_dev * rolling_std)
        lower_band = middle_band - (std_dev * rolling_std)

        return upper_band, middle_band, lower_band

    def _check_bollinger_filter(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Check if Bollinger Bands filter condition is met: price is below upper band.

        Entry Filter Logic:
        - Only enter when price is BELOW the upper Bollinger Band
        - This avoids entering when price is overbought/extended
        - Best to sell covered calls when stock has room to move up (but not too much)

        Args:
            symbol: Stock symbol
            current_date: Current date
            stock_data: Dict of stock DataFrames

        Returns:
            True if price is below upper band (filter passes), False otherwise
        """
        if symbol not in stock_data:
            return True  # Allow entry if no data

        df = stock_data[symbol]
        mask = df.index <= current_date
        historical_data = df.loc[mask].copy()

        # Need enough data for Bollinger calculation
        min_required = self.config.bollinger_period + 5
        if len(historical_data) < min_required:
            logger.debug(f"Bollinger filter: Not enough data for {symbol}, allowing entry")
            return True

        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = self._calculate_bollinger_bands(
            historical_data,
            period=self.config.bollinger_period,
            std_dev=self.config.bollinger_std
        )

        # Get current values
        current_close = historical_data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_middle = middle_band.iloc[-1]

        # Check for NaN values
        if pd.isna(current_upper) or pd.isna(current_middle):
            logger.debug(f"Bollinger filter: NaN values for {symbol}, allowing entry")
            return True

        # Filter passes when price is below upper band (not overbought)
        filter_passed = current_close < current_upper

        if filter_passed:
            # Calculate how far below upper band (as % of band width)
            band_width = current_upper - current_middle
            distance_from_upper = current_upper - current_close
            pct_below_upper = (distance_from_upper / band_width) * 100 if band_width > 0 else 0
            logger.debug(f"Bollinger filter PASSED for {symbol}: "
                        f"Price={current_close:.2f}, Upper={current_upper:.2f}, "
                        f"Middle={middle_band.iloc[-1]:.2f} ({pct_below_upper:.1f}% below upper)")
        else:
            logger.debug(f"Bollinger filter BLOCKED entry for {symbol}: "
                        f"Price={current_close:.2f} >= Upper Band={current_upper:.2f} (overbought)")

        return filter_passed

    def _calculate_vwap(
        self,
        df: pd.DataFrame,
        period: int = 1
    ) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).

        VWAP = Cumulative(Typical Price x Volume) / Cumulative(Volume)
        Typical Price = (High + Low + Close) / 3

        VWAP is a key indicator used by institutional traders to assess
        whether a stock is trading at fair value. Price above VWAP indicates
        bullish sentiment and institutional buying support.

        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume' columns
            period: Rolling period in days for VWAP calculation
                    - 1: Daily reset (traditional intraday VWAP, calculated for single day)
                    - >1: Rolling VWAP over N days (anchored VWAP variant)

        Returns:
            Series of VWAP values
        """
        # Check if volume column exists
        if 'volume' not in df.columns:
            logger.warning("Volume column not found, cannot calculate VWAP")
            return pd.Series(index=df.index, dtype=float)

        # Calculate Typical Price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Calculate VWAP based on period
        if period == 1:
            # Single day VWAP (for daily data, each row is one day)
            # VWAP = Typical Price (since we have one data point per day)
            # For daily bars, VWAP is essentially the typical price of that day
            vwap = typical_price
        else:
            # Rolling VWAP over N days
            # Cumulative sum of (Typical Price x Volume) / Cumulative sum of Volume
            tp_volume = typical_price * df['volume']
            cumsum_tp_vol = tp_volume.rolling(window=period, min_periods=1).sum()
            cumsum_vol = df['volume'].rolling(window=period, min_periods=1).sum()

            # Avoid division by zero
            cumsum_vol = cumsum_vol.replace(0, np.nan)
            vwap = cumsum_tp_vol / cumsum_vol

        return vwap

    def _check_vwap_filter(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Check if VWAP filter condition is met based on configured mode.

        VWAP (Volume Weighted Average Price) is used by institutional traders
        to assess fair value. Price above VWAP indicates buying pressure from
        institutions, which provides support for covered call positions.

        Modes:
        - ABOVE: Price > VWAP (bullish - institutional support, good for covered calls)
        - BELOW: Price < VWAP (bearish - selling pressure)

        Args:
            symbol: Stock symbol
            current_date: Current date
            stock_data: Dict of stock DataFrames with OHLCV data

        Returns:
            True if VWAP filter passes, False otherwise
        """
        if symbol not in stock_data:
            return True  # Allow entry if no data

        df = stock_data[symbol]

        # Get data up to current date
        mask = df.index <= current_date
        historical_data = df.loc[mask].copy()

        # Need at least vwap_period days of data
        min_required = max(self.config.vwap_period, 5)
        if len(historical_data) < min_required:
            logger.debug(f"VWAP filter: Not enough data for {symbol}, allowing entry")
            return True

        # Check if volume data is available
        if 'volume' not in historical_data.columns:
            logger.debug(f"VWAP filter: No volume data for {symbol}, allowing entry")
            return True

        # Calculate VWAP
        vwap = self._calculate_vwap(
            historical_data,
            period=self.config.vwap_period
        )

        # Get current values
        current_close = historical_data['close'].iloc[-1]
        current_vwap = vwap.iloc[-1]

        # Check for NaN values
        if pd.isna(current_close) or pd.isna(current_vwap):
            logger.debug(f"VWAP filter: NaN values for {symbol}, allowing entry")
            return True

        # Apply filter based on mode
        mode = self.config.vwap_mode.upper()
        filter_passed = False

        if mode == "ABOVE":
            # Bullish: Price > VWAP indicates institutional buying support
            filter_passed = current_close > current_vwap
            if filter_passed:
                logger.debug(f"VWAP filter PASSED for {symbol}: "
                            f"Close={current_close:.2f} > VWAP={current_vwap:.2f} (bullish)")
            else:
                logger.debug(f"VWAP filter blocked entry for {symbol}: "
                            f"Close={current_close:.2f} <= VWAP={current_vwap:.2f}")

        elif mode == "BELOW":
            # Bearish: Price < VWAP indicates selling pressure
            filter_passed = current_close < current_vwap
            if filter_passed:
                logger.debug(f"VWAP filter PASSED for {symbol}: "
                            f"Close={current_close:.2f} < VWAP={current_vwap:.2f} (bearish)")
            else:
                logger.debug(f"VWAP filter blocked entry for {symbol}: "
                            f"Close={current_close:.2f} >= VWAP={current_vwap:.2f}")

        else:
            # Unknown mode, allow entry
            logger.warning(f"Unknown VWAP mode '{mode}', allowing entry")
            filter_passed = True

        return filter_passed

    def _calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence) indicator.

        MACD is a trend-following momentum indicator that shows the relationship
        between two exponential moving averages of a security's price.

        Formula:
        - MACD Line = Fast EMA - Slow EMA (typically 12-day EMA - 26-day EMA)
        - Signal Line = 9-day EMA of MACD Line
        - Histogram = MACD Line - Signal Line

        Args:
            df: DataFrame with 'close' column
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram) Series
        """
        close = df['close']

        # Calculate EMAs
        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()

        # MACD Line = Fast EMA - Slow EMA
        macd_line = fast_ema - slow_ema

        # Signal Line = 9-period EMA of MACD Line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # Histogram = MACD Line - Signal Line
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _check_macd_filter(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Check if MACD filter condition is met for covered call entry.

        Two modes are supported:
        1. BULLISH: MACD line > Signal line (bullish momentum)
           - Good for riding uptrends while collecting premium
        2. REVERSAL: Histogram is turning negative (momentum fading)
           - Good for selling calls when upward momentum is weakening
           - Premium decay is favorable when stock is topping out

        Args:
            symbol: Stock symbol
            current_date: Current date
            stock_data: Dict of stock DataFrames

        Returns:
            True if MACD filter passes, False otherwise
        """
        if symbol not in stock_data:
            return True  # Allow entry if no data

        df = stock_data[symbol]

        # Get data up to current date
        mask = df.index <= current_date
        historical_data = df.loc[mask].copy()

        # Need enough data for MACD calculation (slow period + signal period)
        min_required = self.config.macd_slow + self.config.macd_signal + 5
        if len(historical_data) < min_required:
            logger.debug(f"MACD filter: Not enough data for {symbol}, allowing entry")
            return True

        # Calculate MACD
        macd_line, signal_line, histogram = self._calculate_macd(
            historical_data,
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal
        )

        # Get current and previous values
        if len(macd_line) < 2 or len(signal_line) < 2 or len(histogram) < 2:
            return True

        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        previous_histogram = histogram.iloc[-2]

        # Check for NaN values
        if pd.isna(current_macd) or pd.isna(current_signal) or pd.isna(current_histogram):
            logger.debug(f"MACD filter: NaN values for {symbol}, allowing entry")
            return True

        mode = self.config.macd_mode.upper()
        filter_passed = False

        if mode == "BULLISH":
            # MACD line > Signal line = bullish momentum
            filter_passed = current_macd > current_signal
            if filter_passed:
                logger.debug(f"MACD BULLISH signal for {symbol}: "
                            f"MACD={current_macd:.4f} > Signal={current_signal:.4f}")
            else:
                logger.debug(f"MACD filter blocked entry for {symbol}: "
                            f"MACD={current_macd:.4f} <= Signal={current_signal:.4f} (not bullish)")

        elif mode == "REVERSAL":
            # Histogram turning negative = momentum fading
            # Good for premium decay as stock may consolidate/decline
            histogram_turning_down = (
                previous_histogram > 0 and current_histogram < previous_histogram
            ) or (
                current_histogram < 0 and previous_histogram >= 0
            )
            filter_passed = histogram_turning_down
            if filter_passed:
                logger.debug(f"MACD REVERSAL signal for {symbol}: "
                            f"Histogram turning down: prev={previous_histogram:.4f}, curr={current_histogram:.4f}")
            else:
                logger.debug(f"MACD filter blocked entry for {symbol}: "
                            f"Histogram not turning down: prev={previous_histogram:.4f}, curr={current_histogram:.4f}")

        else:
            # Unknown mode, allow entry
            logger.debug(f"MACD filter: Unknown mode '{mode}', allowing entry")
            filter_passed = True

        return filter_passed

    def _calculate_adx(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Average Directional Index (ADX) with +DI and -DI.

        ADX measures trend strength (not direction):
        - ADX > 25: Strong trend (good for trend-following strategies)
        - ADX < 20: Weak trend or ranging market
        - +DI > -DI: Uptrend (bullish)
        - -DI > +DI: Downtrend (bearish)

        Formula:
        1. Calculate True Range (TR)
        2. Calculate +DM (Plus Directional Movement) and -DM (Minus Directional Movement)
        3. Smooth TR, +DM, -DM over period
        4. +DI = 100 * Smoothed(+DM) / Smoothed(TR)
        5. -DI = 100 * Smoothed(-DM) / Smoothed(TR)
        6. DX = 100 * |+DI - -DI| / (+DI + -DI)
        7. ADX = Smoothed(DX) over period

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ADX calculation period (default 14)

        Returns:
            Tuple of (adx, plus_di, minus_di) Series
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate True Range (TR)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement (+DM and -DM)
        # +DM = High - Previous High (if positive and > -(Low - Previous Low))
        # -DM = Previous Low - Low (if positive and > (High - Previous High))
        delta_high = high - high.shift(1)
        delta_low = low.shift(1) - low

        plus_dm = pd.Series(index=df.index, dtype=float)
        minus_dm = pd.Series(index=df.index, dtype=float)

        for i in range(len(df)):
            if i == 0:
                plus_dm.iloc[i] = 0
                minus_dm.iloc[i] = 0
            else:
                dh = delta_high.iloc[i]
                dl = delta_low.iloc[i]

                if dh > dl and dh > 0:
                    plus_dm.iloc[i] = dh
                else:
                    plus_dm.iloc[i] = 0

                if dl > dh and dl > 0:
                    minus_dm.iloc[i] = dl
                else:
                    minus_dm.iloc[i] = 0

        # Smooth TR, +DM, -DM using Wilder's smoothing (similar to EMA with alpha=1/period)
        # First value is sum of first 'period' values
        # Subsequent values: Previous Smoothed - (Previous Smoothed / period) + Current Value
        def wilder_smooth(series: pd.Series, n: int) -> pd.Series:
            smoothed = pd.Series(index=series.index, dtype=float)
            smoothed.iloc[:n] = np.nan

            # First smoothed value is sum of first n values
            first_sum = series.iloc[:n].sum()
            smoothed.iloc[n-1] = first_sum

            # Apply Wilder's smoothing formula
            for i in range(n, len(series)):
                smoothed.iloc[i] = smoothed.iloc[i-1] - (smoothed.iloc[i-1] / n) + series.iloc[i]

            return smoothed

        smoothed_tr = wilder_smooth(true_range, period)
        smoothed_plus_dm = wilder_smooth(plus_dm, period)
        smoothed_minus_dm = wilder_smooth(minus_dm, period)

        # Calculate +DI and -DI
        # Avoid division by zero
        plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
        minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)

        # Calculate DX (Directional Index)
        di_sum = plus_di + minus_di
        di_diff = abs(plus_di - minus_di)
        dx = 100 * di_diff / di_sum.replace(0, np.nan)

        # Calculate ADX as smoothed DX
        adx = wilder_smooth(dx.fillna(0), period)

        return adx, plus_di, minus_di

    def _check_adx_filter(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Check if ADX filter condition is met: ADX > threshold (trending market).

        ADX (Average Directional Index) measures trend strength:
        - ADX > 25: Strong trend - good for covered calls in trending stocks
        - ADX < 20: Weak trend/ranging - may want to avoid or use different strategy

        Optional: Also check +DI > -DI for bullish trend confirmation.

        Args:
            symbol: Stock symbol
            current_date: Current date
            stock_data: Dict of stock DataFrames

        Returns:
            True if ADX filter passes, False otherwise
        """
        if symbol not in stock_data:
            return True  # Allow entry if no data

        df = stock_data[symbol]
        mask = df.index <= current_date
        historical_data = df.loc[mask].copy()

        # Need enough data for ADX calculation (2 * period + buffer)
        min_required = (2 * self.config.adx_period) + 10
        if len(historical_data) < min_required:
            logger.debug(f"ADX filter: Not enough data for {symbol}, allowing entry")
            return True

        # Calculate ADX, +DI, -DI
        adx, plus_di, minus_di = self._calculate_adx(
            historical_data,
            period=self.config.adx_period
        )

        # Get current values
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]

        # Check for NaN values
        if pd.isna(current_adx) or pd.isna(current_plus_di) or pd.isna(current_minus_di):
            logger.debug(f"ADX filter: NaN values for {symbol}, allowing entry")
            return True

        # Check ADX threshold (trend strength)
        adx_condition = current_adx > self.config.adx_threshold

        # Check bullish condition (+DI > -DI) if required
        if self.config.adx_require_bullish:
            bullish_condition = current_plus_di > current_minus_di
            filter_passed = adx_condition and bullish_condition

            if filter_passed:
                logger.debug(f"ADX filter PASSED for {symbol}: "
                            f"ADX={current_adx:.1f} (>{self.config.adx_threshold}), "
                            f"+DI={current_plus_di:.1f} > -DI={current_minus_di:.1f} (bullish)")
            else:
                if not adx_condition:
                    logger.debug(f"ADX filter blocked entry for {symbol}: "
                                f"ADX={current_adx:.1f} <= {self.config.adx_threshold} (weak trend)")
                elif not bullish_condition:
                    logger.debug(f"ADX filter blocked entry for {symbol}: "
                                f"+DI={current_plus_di:.1f} <= -DI={current_minus_di:.1f} (bearish)")
        else:
            filter_passed = adx_condition

            if filter_passed:
                logger.debug(f"ADX filter PASSED for {symbol}: "
                            f"ADX={current_adx:.1f} (>{self.config.adx_threshold})")
            else:
                logger.debug(f"ADX filter blocked entry for {symbol}: "
                            f"ADX={current_adx:.1f} <= {self.config.adx_threshold} (weak trend)")

        return filter_passed

    def _calculate_williams_r(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Williams %R oscillator.

        Williams %R is a momentum indicator that measures overbought and oversold levels.
        It's similar to the Stochastic oscillator but uses a different scale (0 to -100).

        Formula:
        %R = (Highest High - Close) / (Highest High - Lowest Low) x -100

        Interpretation:
        - 0 to -20: Overbought (price near highs, may reverse down)
        - -80 to -100: Oversold (price near lows, may reverse up)
        - -20 to -80: Neutral zone

        For covered calls, we want to sell when %R drops from overbought (above -20)
        This signals momentum is turning, good time to sell premium.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: Lookback period (default 14)

        Returns:
            Series of Williams %R values (range: 0 to -100)
        """
        # Calculate highest high and lowest low over the period
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()

        # Calculate Williams %R
        # Avoid division by zero
        range_hl = highest_high - lowest_low
        range_hl = range_hl.replace(0, np.nan)

        williams_r = ((highest_high - df['close']) / range_hl) * -100

        return williams_r

    def _check_williams_filter(
        self,
        symbol: str,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Check if Williams %R filter condition is met for covered call entry.

        Entry signal for covered calls (good time to sell premium):
        - Williams %R was recently above -20 (overbought territory)
        - Williams %R is now dropping from overbought zone
        - This indicates momentum is turning downward
        - Good environment for selling calls as stock may consolidate/decline

        The signal occurs when:
        1. Previous %R was above overbought threshold (-20)
        2. Current %R has dropped below previous %R (momentum weakening)

        Args:
            symbol: Stock symbol
            current_date: Current date
            stock_data: Dict of stock DataFrames

        Returns:
            True if Williams %R filter passes (sell signal), False otherwise
        """
        if symbol not in stock_data:
            return True  # Allow entry if no data

        df = stock_data[symbol]

        # Get data up to current date
        mask = df.index <= current_date
        historical_data = df.loc[mask].copy()

        # Need enough data for Williams %R calculation
        min_required = self.config.williams_period + 5
        if len(historical_data) < min_required:
            logger.debug(f"Williams %R filter: Not enough data for {symbol}, allowing entry")
            return True

        # Calculate Williams %R
        williams_r = self._calculate_williams_r(
            historical_data,
            period=self.config.williams_period
        )

        # Get current and previous values
        if len(williams_r) < 2:
            return True

        current_wr = williams_r.iloc[-1]
        previous_wr = williams_r.iloc[-2]

        # Check for NaN values
        if pd.isna(current_wr) or pd.isna(previous_wr):
            logger.debug(f"Williams %R filter: NaN values for {symbol}, allowing entry")
            return True

        overbought = self.config.williams_overbought  # -20.0 by default

        # Check for sell signal:
        # 1. Was recently in overbought territory (above -20) OR still near it
        # 2. %R is now dropping (momentum turning)
        was_overbought = previous_wr > overbought or current_wr > (overbought - 10)
        momentum_turning = current_wr < previous_wr  # %R is dropping

        filter_passed = was_overbought and momentum_turning

        if filter_passed:
            logger.debug(f"Williams %R SELL signal for {symbol}: "
                        f"Current %R={current_wr:.1f}, Previous %R={previous_wr:.1f}, "
                        f"Overbought threshold={overbought}")
        else:
            logger.debug(f"Williams %R filter blocked entry for {symbol}: "
                        f"Current %R={current_wr:.1f}, Previous %R={previous_wr:.1f}, "
                        f"was_overbought={was_overbought}, momentum_turning={momentum_turning}")

        return filter_passed

    def _get_price(
        self,
        symbol: str,
        date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> Optional[float]:
        """Get stock price for a given date"""
        if symbol not in stock_data:
            return None

        df = stock_data[symbol]

        # Try exact date first
        if date in df.index:
            return df.loc[date, 'close']

        # Find nearest date
        mask = df.index <= date
        if not mask.any():
            return None

        nearest_date = df.loc[mask].index[-1]
        return df.loc[nearest_date, 'close']

    def _update_equity(
        self,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ):
        """Update equity curve for current date"""
        portfolio_value = self.cash

        # Add value of open positions
        for symbol, position in self.positions.items():
            stock_price = self._get_price(symbol, current_date, stock_data)
            if stock_price is None:
                stock_price = position.stock_entry_price

            # Stock value
            stock_value = stock_price * position.lot_size

            # Option liability (if we had to buy back)
            dte = max((position.expiry_date - current_date).days, 0)
            if dte > 0:
                time_to_expiry = dte / 365.0
                option_value = self.greeks.calculate_option_price(
                    spot=stock_price,
                    strike=position.strike_price,
                    time_to_expiry=time_to_expiry,
                    volatility=position.iv_at_entry,
                    option_type='CE'
                )
            else:
                option_value = max(0, stock_price - position.strike_price)

            # Net position value = stock value - option liability
            position_value = stock_value - (option_value * position.lot_size)
            portfolio_value += position_value

        self.equity_curve[current_date] = portfolio_value

    def _compile_results(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Compile final results"""
        # Create equity series
        equity_series = pd.Series(self.equity_curve)
        equity_series.index = pd.to_datetime(equity_series.index)
        equity_series = equity_series.sort_index()

        # Create trades DataFrame
        trades_df = pd.DataFrame([vars(t) for t in self.trades])

        # Calculate metrics
        metrics = self.metrics.calculate_all_metrics(
            equity_curve=equity_series,
            trades=trades_df,
            initial_capital=self.config.initial_capital
        )

        # Calculate buy-and-hold for comparison
        if stock_data:
            prices_dict = {}
            for symbol, df in stock_data.items():
                mask = (df.index >= self.config.start_date) & (df.index <= self.config.end_date)
                prices_dict[symbol] = df.loc[mask, 'close']

            prices_df = pd.DataFrame(prices_dict)
            bh_return = self.metrics.calculate_buy_hold_return(prices_df)
            metrics['vs_buy_hold'] = metrics['total_return'] - bh_return
            metrics['buy_hold_return'] = bh_return

        return {
            'metrics': metrics,
            'trades': trades_df,
            'equity_curve': equity_series,
            'config': {
                'symbols': self.config.symbols,
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'strike_method': self.config.strike_method.value,
                'exit_strategy': self.config.exit_strategy.value,
                'initial_capital': self.config.initial_capital,
                'trend_filter_mode': self.config.trend_filter_mode
            }
        }


# =============================================================================
# Utility Functions
# =============================================================================

def run_backtest(
    symbols: List[str],
    stock_data: Dict[str, pd.DataFrame],
    start_date: datetime,
    end_date: datetime,
    strike_method: str = "DELTA_30",
    exit_strategy: str = "HOLD_TO_EXPIRY",
    initial_capital: float = 1000000.0,
    profit_target_pct: float = 50.0,
    stop_loss_multiple: float = 2.0,
    allow_sl_adjustment: bool = False,
    use_trend_filter: bool = False,
    trend_filter_mode: str = "NONE",
    use_rsi_filter: bool = False,
    rsi_period: int = 14,
    rsi_min: float = 40.0,
    rsi_max: float = 70.0,
    use_stochastic_filter: bool = False,
    stochastic_k_period: int = 14,
    stochastic_d_period: int = 3,
    stochastic_smoothing: int = 3,
    stochastic_overbought: float = 70.0,
    atr_multiplier: float = 1.5,
    use_supertrend_filter: bool = False,
    supertrend_period: int = 10,
    supertrend_multiplier: float = 3.0,
    use_macd_filter: bool = False,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    macd_mode: str = "BULLISH",
    use_adx_filter: bool = False,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    adx_require_bullish: bool = True,
    use_williams_filter: bool = False,
    williams_period: int = 14,
    williams_overbought: float = -20.0,
    williams_oversold: float = -80.0,
    progress_callback=None
) -> Dict:
    """
    Convenience function to run a backtest

    Args:
        symbols: List of stock symbols
        stock_data: Dict mapping symbol to DataFrame
        start_date: Start date
        end_date: End date
        strike_method: Strike selection method
        exit_strategy: Exit strategy
        initial_capital: Starting capital
        profit_target_pct: Profit target percentage (default 50%)
        stop_loss_multiple: Stop loss multiple of premium (default 2x)
        allow_sl_adjustment: If True, roll up to higher strike on SL instead of closing (once)
        use_trend_filter: DEPRECATED - use trend_filter_mode instead
        trend_filter_mode: Multi-timeframe EMA filter mode (NONE, BEARISH, BULLISH, GOLDEN_CROSS, DEATH_CROSS, BULLISH_ALIGNED)
        use_rsi_filter: If True, only enter when RSI is within range
        rsi_period: RSI calculation period (default 14)
        rsi_min: Minimum RSI for entry (default 40)
        rsi_max: Maximum RSI for entry (default 70)
        use_stochastic_filter: If True, only enter on stochastic sell signal
        stochastic_k_period: %K lookback period (default 14)
        stochastic_d_period: %D smoothing period (default 3)
        stochastic_smoothing: %K smoothing for Slow Stochastic (default 3)
        stochastic_overbought: Overbought threshold (default 70)
        atr_multiplier: ATR multiplier for ATR_BASED strike selection (default 1.5)
        use_supertrend_filter: If True, only enter when price > Supertrend (bullish)
        supertrend_period: ATR period for Supertrend calculation (default 10)
        supertrend_multiplier: ATR multiplier for Supertrend bands (default 3.0)
        use_macd_filter: If True, apply MACD-based entry filter
        macd_fast: Fast EMA period for MACD (default 12)
        macd_slow: Slow EMA period for MACD (default 26)
        macd_signal: Signal line EMA period for MACD (default 9)
        macd_mode: MACD filter mode - BULLISH (MACD>Signal) or REVERSAL (histogram turning down)
        use_adx_filter: If True, only enter when ADX > threshold (trending market)
        adx_period: ADX calculation period (default 14)
        adx_threshold: Minimum ADX value for entry (default 25, indicates trending market)
        adx_require_bullish: If True, also require +DI > -DI for uptrend confirmation
        use_williams_filter: If True, only enter when Williams %R drops from overbought
        williams_period: Williams %R lookback period (default 14)
        williams_overbought: Overbought threshold (default -20, range 0 to -20)
        williams_oversold: Oversold threshold (default -80, range -80 to -100)
        progress_callback: Optional progress callback

    Returns:
        Dict with results
    """
    config = BacktestConfig(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        strike_method=StrikeMethod(strike_method),
        exit_strategy=ExitStrategy(exit_strategy),
        initial_capital=initial_capital,
        profit_target_pct=profit_target_pct,
        stop_loss_multiple=stop_loss_multiple,
        allow_sl_adjustment=allow_sl_adjustment,
        use_trend_filter=use_trend_filter,
        trend_filter_mode=trend_filter_mode,
        use_rsi_filter=use_rsi_filter,
        rsi_period=rsi_period,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        use_stochastic_filter=use_stochastic_filter,
        stochastic_k_period=stochastic_k_period,
        stochastic_d_period=stochastic_d_period,
        stochastic_smoothing=stochastic_smoothing,
        stochastic_overbought=stochastic_overbought,
        atr_multiplier=atr_multiplier,
        use_supertrend_filter=use_supertrend_filter,
        supertrend_period=supertrend_period,
        supertrend_multiplier=supertrend_multiplier,
        use_macd_filter=use_macd_filter,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        macd_mode=macd_mode,
        use_adx_filter=use_adx_filter,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        adx_require_bullish=adx_require_bullish,
        use_williams_filter=use_williams_filter,
        williams_period=williams_period,
        williams_overbought=williams_overbought,
        williams_oversold=williams_oversold
    )

    engine = CoveredCallEngine(config)
    return engine.run_backtest(stock_data, progress_callback)
