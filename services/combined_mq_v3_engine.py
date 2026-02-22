"""
Combined MQ + Breakout V3 Overlay Engine
==========================================

Runs the MQ portfolio (70% equity, 10% debt) with a V3 breakout overlay
(20% capital, 5x futures + Married Put) on MQ portfolio stocks.

Three capital pools:
- MQ Core (70%): 30 momentum+quality stocks, semi-annual rebalance, Darvas topups
- V3 Overlay (20%): 5x leveraged futures on MQ stocks that fire PRIMARY breakout signals
- Debt Reserve (10%): Funds Darvas topups + earns 7% p.a.

V3 overlay only fires on stocks already in the MQ portfolio.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from copy import deepcopy

from .mq_portfolio import MQBacktestConfig, Portfolio, ExitReason
from .mq_backtest_engine import (
    MQBacktestEngine, BacktestResult,
    _load_all_daily_data, _get_trading_days, _get_prices_on_date, _run_screening,
    TRADING_DAYS_PER_YEAR, DB_PATH,
)
from .consolidation_breakout import (
    SYSTEM_PRIMARY, SYSTEM_SNIPER, SYSTEM_BALANCED, SYSTEM_ACTIVE, SYSTEM_HIGH_VOLUME,
    check_system_match, DarvasBoxDetector,
)
from .nifty500_universe import load_nifty500
from .technical_indicators import calc_ema

logger = logging.getLogger(__name__)


# =============================================================================
# V3 Overlay Data Classes
# =============================================================================

@dataclass
class V3OverlayPosition:
    """A leveraged futures position from the V3 overlay."""
    symbol: str
    entry_date: datetime
    entry_price: float
    margin_deployed: float
    notional: float
    highest_high: float
    trail_sl: float
    consol_low: float
    strategy_matched: str
    current_price: float = 0.0

    @property
    def base_return_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def current_notional(self) -> float:
        if self.entry_price <= 0:
            return self.notional
        return self.notional * (self.current_price / self.entry_price)


@dataclass
class V3Trade:
    """Completed V3 overlay trade record."""
    symbol: str
    strategy: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    exit_reason: str
    base_return_pct: float
    leveraged_return_pct: float
    margin_deployed: float
    pnl: float


@dataclass
class CombinedConfig:
    """Configuration for the combined MQ + V3 system."""
    # MQ core config (will be modified: equity=70%, debt=10%)
    start_date: str = '2023-01-01'
    end_date: str = '2025-12-31'
    initial_capital: float = 10_000_000  # Rs.1 Crore

    # Capital allocation
    mq_capital_pct: float = 0.70
    v3_capital_pct: float = 0.20
    debt_capital_pct: float = 0.10

    # V3 SIP mode: monthly contribution to V3 pool (set v3_capital_pct=0 to use SIP only)
    v3_monthly_sip: float = 0.0  # Rs per month added to V3 cash (e.g. 50000)

    # V3 overlay params
    v3_leverage: float = 5.0
    v3_put_cost_pct: float = 3.4
    v3_txn_cost_pct: float = 2.3
    v3_trail_pct: float = 20.0
    v3_max_hold_days: int = 125
    v3_max_concurrent: int = 5
    v3_system_name: str = 'PRIMARY'

    # MQ overrides (pass-through to MQBacktestConfig)
    portfolio_size: int = 30
    hard_stop_loss: float = 0.30
    rebalance_ath_drawdown: float = 0.20
    topup_stop_loss_pct: float = 0.0

    def get_mq_config(self) -> MQBacktestConfig:
        """Build MQ config with adjusted allocation."""
        return MQBacktestConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            equity_allocation_pct=self.mq_capital_pct,
            debt_reserve_pct=self.debt_capital_pct,
            portfolio_size=self.portfolio_size,
            hard_stop_loss=self.hard_stop_loss,
            rebalance_ath_drawdown=self.rebalance_ath_drawdown,
            topup_stop_loss_pct=self.topup_stop_loss_pct,
        )

    @property
    def v3_system(self) -> list:
        systems = {
            'PRIMARY': SYSTEM_PRIMARY,
            'SNIPER': SYSTEM_SNIPER,
            'BALANCED': SYSTEM_BALANCED,
            'ACTIVE': SYSTEM_ACTIVE,
            'HIGH_VOLUME': SYSTEM_HIGH_VOLUME,
        }
        return systems.get(self.v3_system_name, SYSTEM_PRIMARY)


@dataclass
class CombinedResult:
    """Complete combined backtest output."""
    config: CombinedConfig

    # MQ core results
    mq_result: BacktestResult

    # V3 overlay results
    v3_trades: List[V3Trade] = field(default_factory=list)
    v3_total_trades: int = 0
    v3_winning_trades: int = 0
    v3_win_rate: float = 0.0
    v3_avg_leveraged_return: float = 0.0
    v3_profit_factor: float = 0.0
    v3_total_pnl: float = 0.0
    v3_initial_capital: float = 0.0
    v3_final_value: float = 0.0
    v3_equity_curve: Dict[str, float] = field(default_factory=dict)
    v3_strategy_breakdown: Dict[str, Dict] = field(default_factory=dict)

    # Combined metrics
    combined_equity_curve: Dict[str, float] = field(default_factory=dict)
    combined_initial: float = 0.0
    combined_final: float = 0.0
    combined_cagr: float = 0.0
    combined_sharpe: float = 0.0
    combined_max_drawdown: float = 0.0
    combined_calmar: float = 0.0
    combined_total_return_pct: float = 0.0

    # SIP tracking
    v3_total_sip_invested: float = 0.0

    # Capital allocation over time
    capital_allocation: Dict[str, Dict] = field(default_factory=dict)

    # Year breakdown
    yearly_returns: List[Dict] = field(default_factory=list)


# =============================================================================
# Combined Engine
# =============================================================================

class CombinedMQV3Engine:
    """
    Runs MQ portfolio + V3 breakout overlay in a single daily simulation.

    Usage:
        config = CombinedConfig()
        engine = CombinedMQV3Engine(config)
        result = engine.run()
    """

    def __init__(self, config: CombinedConfig = None):
        self.config = config or CombinedConfig()
        self.mq_config = self.config.get_mq_config()

        # Data
        self.universe = None
        self.price_data: Dict[str, pd.DataFrame] = {}

        # V3 overlay state
        self.v3_cash: float = 0.0
        self.v3_positions: Dict[str, V3OverlayPosition] = {}
        self.v3_trades: List[V3Trade] = []
        self.v3_equity_curve: Dict[str, float] = {}

        # Combined tracking
        self.combined_equity_curve: Dict[str, float] = {}
        self.capital_allocation: Dict[str, Dict] = {}

        # Indicator cache: {(symbol, date_str): indicator_dict}
        self._indicator_cache: Dict = {}

    def run(self, quality_scores: Dict = None, progress_callback=None) -> CombinedResult:
        """Run the full combined simulation."""
        cfg = self.config

        # Phase 1: Load data
        logger.info("Combined Engine Phase 1: Loading data...")
        self.universe = load_nifty500()
        sim_start_dt = datetime.strptime(cfg.start_date, '%Y-%m-%d')
        data_start = (sim_start_dt - timedelta(days=400)).strftime('%Y-%m-%d')
        self.price_data = _load_all_daily_data(
            self.universe.symbols, data_start, cfg.end_date
        )

        all_days = _get_trading_days(self.price_data)
        trading_days = [d for d in all_days if d >= sim_start_dt]
        pre_sim_days = len([d for d in all_days if d < sim_start_dt])

        if not trading_days:
            raise ValueError("No trading days found")

        logger.info(f"  {len(trading_days)} trading days, {len(self.price_data)} symbols")

        # Phase 2: Initialize
        logger.info("Combined Engine Phase 2: Initializing portfolios...")

        # MQ portfolio with 70/10 split
        mq_engine = MQBacktestEngine(
            self.mq_config,
            preloaded_universe=self.universe,
            preloaded_price_data=self.price_data,
        )
        mq_engine.quality_scores = quality_scores or {}
        mq_engine.portfolio = Portfolio(self.mq_config)
        mq_engine.portfolio.equity_cash = self.mq_config.equity_capital

        # V3 overlay pool
        self.v3_cash = cfg.initial_capital * cfg.v3_capital_pct
        v3_initial = self.v3_cash
        self.v3_total_sip = 0.0  # Track total SIP contributions
        self.v3_positions = {}
        self.v3_trades = []

        # Warmup
        min_history_days = 60
        warmup_needed = max(0, min_history_days - pre_sim_days)
        warmup_idx = min(warmup_needed, len(trading_days) - 1)
        portfolio_start_day = trading_days[warmup_idx]
        portfolio_built = False
        last_consol_check = trading_days[0]
        last_v3_check = trading_days[0]

        # Phase 3: Daily loop
        logger.info("Combined Engine Phase 3: Running simulation...")
        total_days = len(trading_days)

        for day_idx, current_date in enumerate(trading_days):
            prices = _get_prices_on_date(self.price_data, current_date)
            if not prices:
                continue

            # --- MQ CORE: Update prices + accrue debt ---
            mq_engine.portfolio.update_prices(prices, current_date)

            # Build initial portfolio
            if not portfolio_built and current_date >= portfolio_start_day:
                logger.info("  Building initial MQ portfolio...")
                mq_engine._build_initial_portfolio(current_date)
                portfolio_built = True
                self._record_daily(current_date, mq_engine.portfolio, v3_initial)
                continue

            if not portfolio_built:
                self._record_daily(current_date, mq_engine.portfolio, v3_initial)
                continue

            # --- MQ CORE: Exit checks ---
            mq_engine._check_hard_stops(current_date, prices)

            if self.mq_config.topup_stop_loss_pct > 0:
                reversals = mq_engine.portfolio.check_topup_stop_losses(current_date)
                mq_engine._topup_sl_log.extend(reversals)

            if self.mq_config.daily_ath_drawdown_exit:
                mq_engine._check_daily_ath_drawdown(current_date, prices)

            if self.mq_config.use_technical_filter:
                mq_engine._check_technical_exits(current_date, prices)

            # Semi-annual rebalance
            if current_date.month in self.mq_config.rebalance_months and current_date.day <= 7:
                prev_day = trading_days[day_idx - 1] if day_idx > 0 else None
                if prev_day is None or prev_day.month != current_date.month:
                    # Before rebalance, close V3 positions on stocks being exited
                    mq_engine._run_rebalance(current_date, prices)

            # Darvas topups (weekly)
            if (current_date - last_consol_check).days >= 5:
                mq_engine._check_breakout_topups(current_date)
                last_consol_check = current_date

            # --- V3 SIP: Monthly contribution (first trading day of each month) ---
            if cfg.v3_monthly_sip > 0 and day_idx > 0:
                prev_date = trading_days[day_idx - 1]
                if current_date.month != prev_date.month:
                    self.v3_cash += cfg.v3_monthly_sip
                    self.v3_total_sip += cfg.v3_monthly_sip
                    v3_initial += cfg.v3_monthly_sip  # Adjust baseline for return calc

            # --- V3 OVERLAY: Update + check exits ---
            self._update_v3_prices(prices, current_date, mq_engine.portfolio)

            # --- V3 OVERLAY: Check new entries (weekly) ---
            if (current_date - last_v3_check).days >= 5:
                self._check_v3_entries(current_date, mq_engine.portfolio)
                last_v3_check = current_date

            # --- Record combined equity ---
            self._record_daily(current_date, mq_engine.portfolio, v3_initial)

            # Progress callback
            if progress_callback and day_idx % 50 == 0:
                pct = int(day_idx / total_days * 90)
                mq_val = mq_engine.portfolio.total_value
                v3_val = self._v3_total_value()
                progress_callback(
                    pct,
                    f"Day {day_idx}/{total_days} | MQ: {mq_val:,.0f} | V3: {v3_val:,.0f} | "
                    f"V3 trades: {len(self.v3_positions)}"
                )

        # Phase 4: Close any open V3 positions
        logger.info("Combined Engine Phase 4: Closing open V3 positions...")
        end_prices = _get_prices_on_date(self.price_data, trading_days[-1])
        for symbol in list(self.v3_positions.keys()):
            price = end_prices.get(symbol, self.v3_positions[symbol].current_price)
            self._exit_v3(symbol, price, trading_days[-1], 'END_OF_BACKTEST')

        # Phase 5: Build MQ result
        logger.info("Combined Engine Phase 5: Calculating results...")
        mq_result = mq_engine._build_result()

        if progress_callback:
            progress_callback(95, "Calculating combined metrics...")

        # Phase 6: Build combined result
        result = self._build_combined_result(mq_result, v3_initial)

        if progress_callback:
            progress_callback(100, "Complete")

        return result

    # -------------------------------------------------------------------------
    # V3 Overlay: Price Updates + Exit Checks
    # -------------------------------------------------------------------------

    def _update_v3_prices(self, prices: Dict[str, float], current_date: datetime,
                          mq_portfolio: Portfolio):
        """Update V3 positions, check trail SL, time exit, MQ exit."""
        cfg = self.config

        for symbol in list(self.v3_positions.keys()):
            pos = self.v3_positions[symbol]
            price = prices.get(symbol, pos.current_price)
            pos.current_price = price

            # Get high/low for trail SL check
            df = self.price_data.get(symbol)
            if df is not None and current_date in df.index:
                day_high = df.loc[current_date, 'high']
                day_low = df.loc[current_date, 'low']
            else:
                day_high = price
                day_low = price

            # Update trailing stop
            if day_high > pos.highest_high:
                pos.highest_high = day_high
                pos.trail_sl = pos.highest_high * (1 - cfg.v3_trail_pct / 100)

            # Exit: Trail SL hit
            if day_low <= pos.trail_sl:
                self._exit_v3(symbol, pos.trail_sl, current_date, 'TRAIL_SL')
                continue

            # Exit: Max holding period
            days_held = (current_date - pos.entry_date).days
            if days_held >= cfg.v3_max_hold_days:
                self._exit_v3(symbol, price, current_date, 'TIME_EXIT')
                continue

            # Exit: MQ sold the stock
            if symbol not in mq_portfolio.positions:
                self._exit_v3(symbol, price, current_date, 'MQ_EXIT')
                continue

    # -------------------------------------------------------------------------
    # V3 Overlay: Entry Check
    # -------------------------------------------------------------------------

    def _check_v3_entries(self, current_date: datetime, mq_portfolio: Portfolio):
        """Check MQ portfolio stocks for V3 breakout signals."""
        cfg = self.config

        if len(self.v3_positions) >= cfg.v3_max_concurrent:
            return

        for symbol in list(mq_portfolio.positions.keys()):
            if symbol in self.v3_positions:
                continue
            if len(self.v3_positions) >= cfg.v3_max_concurrent:
                break

            indicators = self._compute_v3_indicators(symbol, current_date)
            if indicators is None:
                continue

            if not indicators.get('is_fresh_breakout', False):
                continue

            if check_system_match(indicators, cfg.v3_system):
                self._enter_v3(symbol, current_date, indicators)

    def _enter_v3(self, symbol: str, current_date: datetime, indicators: dict):
        """Open a V3 overlay position."""
        cfg = self.config
        remaining_slots = cfg.v3_max_concurrent - len(self.v3_positions)
        if remaining_slots <= 0 or self.v3_cash <= 0:
            return

        margin = self.v3_cash / remaining_slots
        margin = min(margin, self.v3_cash)
        notional = margin * cfg.v3_leverage
        entry_price = indicators['close']
        consol_high_60 = indicators.get('consol_high', entry_price)

        # Determine which strategy matched
        strategy = self._identify_strategy(indicators)

        pos = V3OverlayPosition(
            symbol=symbol,
            entry_date=current_date,
            entry_price=entry_price,
            margin_deployed=margin,
            notional=notional,
            highest_high=entry_price,
            trail_sl=entry_price * (1 - cfg.v3_trail_pct / 100),
            consol_low=indicators.get('consol_low', entry_price * 0.85),
            strategy_matched=strategy,
            current_price=entry_price,
        )

        self.v3_cash -= margin
        self.v3_positions[symbol] = pos

        logger.debug(
            f"V3 ENTRY: {symbol} @ {entry_price:.2f} [{strategy}] "
            f"margin={margin:,.0f} notional={notional:,.0f} trail_sl={pos.trail_sl:.2f}"
        )

    def _exit_v3(self, symbol: str, exit_price: float, current_date: datetime, reason: str):
        """Close a V3 overlay position."""
        cfg = self.config
        pos = self.v3_positions.pop(symbol, None)
        if pos is None:
            return

        base_return = (exit_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
        leveraged_return = base_return * cfg.v3_leverage - (cfg.v3_put_cost_pct + cfg.v3_txn_cost_pct) / 100

        # Married put caps max loss: can't lose more than margin + put cost
        # At worst, stock drops to put strike (10% OTM), loss = 10% * leverage = 50%
        # But put kicks in, so max loss from stock = put strike distance
        max_loss_pct = -1.0 + (cfg.v3_put_cost_pct + cfg.v3_txn_cost_pct) / 100  # can't lose more than margin
        leveraged_return = max(leveraged_return, max_loss_pct)

        pnl = pos.margin_deployed * leveraged_return
        returned_capital = pos.margin_deployed + pnl
        returned_capital = max(returned_capital, 0)  # Can't go negative (put protection)
        self.v3_cash += returned_capital

        trade = V3Trade(
            symbol=symbol,
            strategy=pos.strategy_matched,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=current_date,
            exit_price=exit_price,
            exit_reason=reason,
            base_return_pct=round(base_return * 100, 2),
            leveraged_return_pct=round(leveraged_return * 100, 2),
            margin_deployed=round(pos.margin_deployed, 2),
            pnl=round(pnl, 2),
        )
        self.v3_trades.append(trade)

        logger.debug(
            f"V3 EXIT [{reason}]: {symbol} @ {exit_price:.2f}, "
            f"base={base_return:+.1%} lev={leveraged_return:+.1%} pnl={pnl:+,.0f}"
        )

    # -------------------------------------------------------------------------
    # V3 Indicator Computation
    # -------------------------------------------------------------------------

    def _compute_v3_indicators(self, symbol: str, current_date: datetime) -> Optional[dict]:
        """Compute all V3 filter indicators from OHLCV data."""
        cache_key = (symbol, current_date.strftime('%Y-%m-%d'))
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]

        df = self.price_data.get(symbol)
        if df is None:
            return None

        hist = df[df.index <= current_date]
        if len(hist) < 65:  # Need 60+ bars for lookbacks
            return None

        close = hist['close']
        high = hist['high']
        low = hist['low']
        volume = hist['volume']
        latest = hist.iloc[-1]

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi14 = 100 - (100 / (1 + rs))
        rsi14_val = rsi14.iloc[-1] if len(rsi14) > 0 else 0

        # Volume ratio & trend
        vol_sma50 = volume.rolling(50).mean()
        vol_sma10 = volume.rolling(10).mean()
        vol_ratio = (latest['volume'] / vol_sma50.iloc[-1]) if vol_sma50.iloc[-1] > 0 else 0
        vol_trend = (vol_sma10.iloc[-1] / vol_sma50.iloc[-1]) if vol_sma50.iloc[-1] > 0 else 0

        # Breakout % (close vs 60-bar highest high, shifted 1)
        consol_high = high.rolling(60).max().shift(1)
        consol_high_val = consol_high.iloc[-1] if len(consol_high) > 0 else latest['close']
        breakout_pct = ((latest['close'] - consol_high_val) / consol_high_val * 100) if consol_high_val > 0 else 0

        # Consolidation low
        consol_low = low.rolling(60).min().shift(1)
        consol_low_val = consol_low.iloc[-1] if len(consol_low) > 0 else latest['low']

        # ATH proximity (500-bar)
        lookback = min(500, len(high))
        ath = high.tail(lookback).max()
        ath_proximity = (latest['close'] / ath * 100) if ath > 0 else 0

        # Williams %R(14)
        hh14 = high.rolling(14).max()
        ll14 = low.rolling(14).min()
        wr_denom = hh14 - ll14
        williams_r = -100 * (hh14 - close) / wr_denom.replace(0, np.nan)
        wr_val = williams_r.iloc[-1] if len(williams_r) > 0 else -50

        # Momentum 60d
        mom_60d = ((latest['close'] - close.iloc[-61]) / close.iloc[-61] * 100) if len(close) > 60 else 0

        # EMA20 > EMA50
        ema20 = calc_ema(close, 20)
        ema50 = calc_ema(close, 50)
        ema20_above_50 = 1 if ema20.iloc[-1] > ema50.iloc[-1] else 0

        # Weekly EMA20 > EMA50
        weekly = hist.resample('W-FRI').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        if len(weekly) >= 50:
            w_ema20 = calc_ema(weekly['close'], 20)
            w_ema50 = calc_ema(weekly['close'], 50)
            w_ema20_gt_50 = 1 if w_ema20.iloc[-1] > w_ema50.iloc[-1] else 0
        else:
            w_ema20_gt_50 = 0

        # Fresh breakout gate: close > consol_high AND was below within last 5 bars
        above_consol = latest['close'] > consol_high_val
        was_below = False
        if len(hist) > 5:
            for i in range(1, 6):
                if close.iloc[-(i+1)] <= consol_high.iloc[-(i+1)]:
                    was_below = True
                    break
        is_fresh_breakout = above_consol and was_below

        result = {
            'close': latest['close'],
            'rsi14': rsi14_val if not np.isnan(rsi14_val) else 0,
            'volume_ratio': vol_ratio,
            'vol_trend': vol_trend,
            'breakout_pct': breakout_pct,
            'ath_proximity': ath_proximity,
            'williams_r': wr_val if not np.isnan(wr_val) else -50,
            'mom_60d': mom_60d,
            'ema20_above_50': ema20_above_50,
            'w_ema20_gt_50': w_ema20_gt_50,
            'consol_high': consol_high_val,
            'consol_low': consol_low_val,
            'is_fresh_breakout': is_fresh_breakout,
        }

        self._indicator_cache[cache_key] = result
        return result

    def _identify_strategy(self, indicators: dict) -> str:
        """Identify which PRIMARY sub-strategy matched."""
        from .consolidation_breakout import (
            STRATEGY_ALPHA, STRATEGY_T1B, STRATEGY_MOMVOL, check_strategy_match,
        )
        if check_strategy_match(indicators, STRATEGY_ALPHA):
            return 'ALPHA'
        if check_strategy_match(indicators, STRATEGY_T1B):
            return 'T1B'
        if check_strategy_match(indicators, STRATEGY_MOMVOL):
            return 'MOMVOL'
        return 'UNKNOWN'

    # -------------------------------------------------------------------------
    # Equity Tracking
    # -------------------------------------------------------------------------

    def _v3_total_value(self) -> float:
        """Current V3 pool value = cash + unrealized positions."""
        unrealized = 0.0
        for pos in self.v3_positions.values():
            base_ret = pos.base_return_pct
            lev_ret = base_ret * self.config.v3_leverage - (self.config.v3_put_cost_pct + self.config.v3_txn_cost_pct) / 100
            lev_ret = max(lev_ret, -1.0 + (self.config.v3_put_cost_pct + self.config.v3_txn_cost_pct) / 100)
            unrealized += pos.margin_deployed * (1 + lev_ret)
        return self.v3_cash + max(unrealized, 0)

    def _record_daily(self, current_date: datetime, mq_portfolio: Portfolio, v3_initial: float):
        """Record combined equity for the day."""
        date_str = current_date.strftime('%Y-%m-%d')
        mq_val = mq_portfolio.total_value
        v3_val = self._v3_total_value()
        combined = mq_val + v3_val

        self.v3_equity_curve[date_str] = round(v3_val, 2)
        self.combined_equity_curve[date_str] = round(combined, 2)
        self.capital_allocation[date_str] = {
            'mq': round(mq_val, 2),
            'v3': round(v3_val, 2),
            'debt': round(mq_portfolio.debt_fund_balance, 2),
            'total': round(combined, 2),
        }

    # -------------------------------------------------------------------------
    # Build Result
    # -------------------------------------------------------------------------

    def _build_combined_result(self, mq_result: BacktestResult, v3_initial: float) -> CombinedResult:
        """Calculate all combined metrics."""
        cfg = self.config

        # V3 trade stats
        v3_winners = [t for t in self.v3_trades if t.pnl > 0]
        v3_losers = [t for t in self.v3_trades if t.pnl <= 0]
        v3_win_rate = len(v3_winners) / len(self.v3_trades) * 100 if self.v3_trades else 0

        v3_avg_lev_return = np.mean([t.leveraged_return_pct for t in self.v3_trades]) if self.v3_trades else 0

        gross_wins = sum(t.pnl for t in v3_winners)
        gross_losses = abs(sum(t.pnl for t in v3_losers))
        v3_pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        v3_total_pnl = sum(t.pnl for t in self.v3_trades)

        v3_final = self._v3_total_value()

        # Strategy breakdown
        strat_breakdown = {}
        for strat_name in ['ALPHA', 'T1B', 'MOMVOL']:
            strat_trades = [t for t in self.v3_trades if t.strategy == strat_name]
            if strat_trades:
                strat_winners = [t for t in strat_trades if t.pnl > 0]
                strat_breakdown[strat_name] = {
                    'trades': len(strat_trades),
                    'win_rate': round(len(strat_winners) / len(strat_trades) * 100, 1),
                    'avg_return': round(np.mean([t.leveraged_return_pct for t in strat_trades]), 1),
                    'total_pnl': round(sum(t.pnl for t in strat_trades), 2),
                }

        # Combined equity curve metrics
        if self.combined_equity_curve:
            eq_series = pd.Series(self.combined_equity_curve, dtype=float)
            eq_series.index = pd.to_datetime(eq_series.index)
            eq_series = eq_series.sort_index()

            # Total capital deployed = initial + all SIP contributions
            total_invested = cfg.initial_capital + self.v3_total_sip
            combined_initial = total_invested
            combined_final = eq_series.iloc[-1]
            total_return = (combined_final - total_invested) / total_invested

            start_dt = datetime.strptime(cfg.start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(cfg.end_date, '%Y-%m-%d')
            years = (end_dt - start_dt).days / 365.25
            cagr = (combined_final / total_invested) ** (1 / years) - 1 if years > 0 else 0

            daily_returns = eq_series.pct_change().dropna()

            # Sharpe
            if len(daily_returns) > 10 and daily_returns.std() > 0:
                excess = daily_returns.mean() - (0.07 / TRADING_DAYS_PER_YEAR)
                sharpe = (excess / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
            else:
                sharpe = 0.0

            # Max drawdown
            rolling_max = eq_series.cummax()
            drawdowns = (eq_series - rolling_max) / rolling_max
            max_dd = abs(drawdowns.min())

            calmar = cagr / max_dd if max_dd > 0 else 0.0

            # Year-by-year
            yearly = []
            for year in sorted(set(eq_series.index.year)):
                year_data = eq_series[eq_series.index.year == year]
                if len(year_data) < 2:
                    continue
                yr_return = (year_data.iloc[-1] - year_data.iloc[0]) / year_data.iloc[0]

                # V3 year return
                v3_year = {k: v for k, v in self.v3_equity_curve.items()
                           if k.startswith(str(year))}
                if v3_year:
                    v3_vals = list(v3_year.values())
                    v3_yr_ret = (v3_vals[-1] - v3_vals[0]) / v3_vals[0] if v3_vals[0] > 0 else 0
                else:
                    v3_yr_ret = 0

                # MQ year return
                mq_year = {k: v for k, v in mq_result.daily_equity.items()
                           if k.startswith(str(year))}
                if mq_year:
                    mq_vals = list(mq_year.values())
                    mq_yr_ret = (mq_vals[-1] - mq_vals[0]) / mq_vals[0] if mq_vals[0] > 0 else 0
                else:
                    mq_yr_ret = 0

                v3_yr_trades = [t for t in self.v3_trades
                                if t.entry_date.year == year or t.exit_date.year == year]

                yearly.append({
                    'year': year,
                    'mq_return': round(mq_yr_ret * 100, 1),
                    'v3_return': round(v3_yr_ret * 100, 1),
                    'combined_return': round(yr_return * 100, 1),
                    'v3_trades': len(v3_yr_trades),
                })
        else:
            combined_initial = cfg.initial_capital
            combined_final = combined_initial
            total_return = 0
            cagr = 0
            sharpe = 0
            max_dd = 0
            calmar = 0
            yearly = []

        return CombinedResult(
            config=cfg,
            mq_result=mq_result,
            v3_trades=self.v3_trades,
            v3_total_trades=len(self.v3_trades),
            v3_winning_trades=len(v3_winners),
            v3_win_rate=round(v3_win_rate, 1),
            v3_avg_leveraged_return=round(v3_avg_lev_return, 1),
            v3_profit_factor=round(v3_pf, 2) if v3_pf != float('inf') else 999.0,
            v3_total_pnl=round(v3_total_pnl, 2),
            v3_initial_capital=round(v3_initial, 2),
            v3_final_value=round(v3_final, 2),
            v3_equity_curve=self.v3_equity_curve,
            v3_strategy_breakdown=strat_breakdown,
            v3_total_sip_invested=round(self.v3_total_sip, 2),
            combined_equity_curve=self.combined_equity_curve,
            combined_initial=round(combined_initial, 2),
            combined_final=round(combined_final, 2),
            combined_cagr=round(cagr * 100, 2),
            combined_sharpe=round(sharpe, 2),
            combined_max_drawdown=round(max_dd * 100, 2),
            combined_calmar=round(calmar, 2),
            combined_total_return_pct=round(total_return * 100, 2),
            capital_allocation=self.capital_allocation,
            yearly_returns=yearly,
        )


# =============================================================================
# Convenience runner
# =============================================================================

def run_combined_backtest(
    config: CombinedConfig = None,
    progress_callback=None,
) -> CombinedResult:
    """Run a combined MQ + V3 backtest with default or custom config."""
    engine = CombinedMQV3Engine(config)
    return engine.run(progress_callback=progress_callback)
