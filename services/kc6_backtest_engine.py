"""
KC6 Mean Reversion Backtest Engine
====================================

Standalone historical backtest engine for the KC6 strategy.
Can run independently or combined with MQ+ for hybrid systems.

Strategy:
- Entry: Close < KC(6, 1.3 ATR) Lower AND Close > SMA(200)
- Exit priority: SL(5%) -> TP(15%) -> MaxHold(15d) -> Signal(High > KC6 Mid)
- Crash filter: Universe median ATR Ratio >= 1.3x blocks all entries
- Position sizing: X% of total capital per trade, max N concurrent

Uses same market_data.db as MQ engine.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from services.kc6_scanner import compute_indicators, atr_series

logger = logging.getLogger(__name__)


# =============================================================================
# Config & Data Classes
# =============================================================================

@dataclass
class KC6BacktestConfig:
    start_date: str = '2010-01-01'
    end_date: str = '2026-02-16'
    initial_capital: float = 10_000_000

    # Keltner Channel
    kc_ema_period: int = 6
    kc_atr_period: int = 6
    kc_multiplier: float = 1.3

    # Trend filter
    sma_period: int = 200

    # Exit rules
    sl_pct: float = 5.0
    tp_pct: float = 15.0
    max_hold_days: int = 15

    # Crash filter
    atr_ratio_threshold: float = 1.3
    atr_lookback: int = 14
    atr_avg_window: int = 50

    # Position sizing
    max_positions: int = 5
    position_size_pct: float = 0.10  # 10% of capital per position

    # Idle cash
    debt_rate: float = 6.5  # Annual rate for idle cash

    # Transaction costs
    brokerage_pct: float = 0.0003
    stt_pct: float = 0.001
    gst_pct: float = 0.18
    stamp_duty_pct: float = 0.00015
    slippage_pct: float = 0.001


@dataclass
class KC6Position:
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: int
    invested: float
    sl_price: float
    tp_price: float


@dataclass
class KC6Trade:
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    invested: float
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float
    exit_reason: str
    holding_days: int


@dataclass
class KC6BacktestResult:
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    final_value: float = 0.0
    total_return_pct: float = 0.0
    daily_equity: Dict[str, float] = field(default_factory=dict)
    trade_log: List[KC6Trade] = field(default_factory=list)
    exit_reason_counts: Dict[str, int] = field(default_factory=dict)
    crash_filter_days: int = 0
    avg_positions: float = 0.0
    max_concurrent: int = 0


# =============================================================================
# Engine
# =============================================================================

class KC6BacktestEngine:
    """
    Historical backtest engine for KC6 mean reversion strategy.

    Usage:
        # Option 1: Pre-enrich data once, reuse across configs
        enriched = KC6BacktestEngine.precompute_indicators(price_data, symbols)
        engine = KC6BacktestEngine(config, price_data, symbols, precomputed=enriched)
        result = engine.run()

        # Option 2: Auto-enrich (slower, computes each time)
        engine = KC6BacktestEngine(config, price_data, symbols)
        result = engine.run()
    """

    def __init__(self, config: KC6BacktestConfig,
                 price_data: Dict[str, pd.DataFrame],
                 universe_symbols: List[str],
                 precomputed: Optional[Dict[str, pd.DataFrame]] = None):
        self.config = config
        self.raw_price_data = price_data
        self.universe_symbols = universe_symbols

        # State (reset each run)
        self.positions: Dict[str, KC6Position] = {}
        self.trades: List[KC6Trade] = []
        self.cash = config.initial_capital
        self.daily_equity: Dict[str, float] = {}

        # Pre-computed indicator data
        self.enriched_data: Dict[str, pd.DataFrame] = precomputed if precomputed else {}

    @staticmethod
    def precompute_indicators(price_data: Dict[str, pd.DataFrame],
                              universe_symbols: List[str],
                              config: KC6BacktestConfig = None) -> Dict[str, pd.DataFrame]:
        """
        Pre-compute KC6 indicators for all stocks. Call once, reuse across runs.
        Returns dict of symbol -> enriched DataFrame.
        """
        if config is None:
            config = KC6BacktestConfig()

        cfg = {
            'kc_ema_period': config.kc_ema_period,
            'kc_atr_period': config.kc_atr_period,
            'kc_multiplier': config.kc_multiplier,
            'sma_period': config.sma_period,
            'atr_lookback': config.atr_lookback,
            'atr_avg_window': config.atr_avg_window,
        }

        enriched = {}
        for sym in universe_symbols:
            if sym not in price_data:
                continue
            df = price_data[sym].copy()
            if len(df) < config.sma_period + 50:
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            compute_indicators(df, cfg)
            enriched[sym] = df

        return enriched

    # -----------------------------------------------------------------
    # Data Enrichment
    # -----------------------------------------------------------------

    def _enrich_data(self):
        """Compute KC6 indicators for all stocks (skips if precomputed)."""
        if self.enriched_data:
            return  # Already have precomputed data
        self.enriched_data = KC6BacktestEngine.precompute_indicators(
            self.raw_price_data, self.universe_symbols, self.config
        )

    def _build_signal_lookup(self):
        """Pre-compute entry signals and fast numpy price arrays."""
        # Entry candidates by date: date -> [(symbol, close, dip_pct)]
        self._entry_candidates = {}

        # Fast price arrays: symbol -> {date_map, close, high, low, kc6_mid}
        self._fast_prices = {}

        for sym, df in self.enriched_data.items():
            # Build numpy arrays and date-to-index map (O(1) lookup)
            date_map = {d: i for i, d in enumerate(df.index)}
            self._fast_prices[sym] = {
                'date_map': date_map,
                'close': df['close'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'kc6_mid': df['kc6_mid'].values,
            }

            # Vectorized entry signal detection
            mask = (
                df['close'].notna() &
                df['kc6_lower'].notna() &
                df['sma200'].notna() &
                (df['close'] < df['kc6_lower']) &
                (df['close'] > df['sma200'])
            )
            close_arr = df['close'].values
            kc_lower_arr = df['kc6_lower'].values
            signal_indices = np.where(mask.values)[0]

            for idx in signal_indices:
                date = df.index[idx]
                close = float(close_arr[idx])
                kc_lower = float(kc_lower_arr[idx])
                dip_pct = (kc_lower - close) / kc_lower
                if date not in self._entry_candidates:
                    self._entry_candidates[date] = []
                self._entry_candidates[date].append((sym, close, dip_pct))

    def _get_price(self, sym: str, date):
        """Fast O(1) price lookup using numpy arrays. Returns (close, high, low, kc6_mid) or None."""
        fp = self._fast_prices.get(sym)
        if fp is None:
            return None
        idx = fp['date_map'].get(date)
        if idx is None:
            return None
        kc_mid = float(fp['kc6_mid'][idx])
        if np.isnan(kc_mid):
            kc_mid = None
        return (float(fp['close'][idx]), float(fp['high'][idx]),
                float(fp['low'][idx]), kc_mid)

    def _precompute_crash_filter(self, trading_days: List[datetime]):
        """Pre-compute universe median ATR ratio for each trading day."""
        # Build a DataFrame of atr_ratios: rows=dates, cols=symbols
        # Then take median across symbols for each date
        all_ratios = {}
        for sym, df in self.enriched_data.items():
            if 'atr_ratio' not in df.columns:
                continue
            ratio_series = df['atr_ratio'].dropna()
            if len(ratio_series) == 0:
                continue
            all_ratios[sym] = ratio_series

        if not all_ratios:
            self._crash_filter_series = pd.Series(1.0, index=trading_days)
            return

        ratio_df = pd.DataFrame(all_ratios)
        self._crash_filter_series = ratio_df.median(axis=1)

    def _get_crash_filter(self, date: datetime) -> float:
        """Get the universe median ATR ratio for a specific date."""
        if date in self._crash_filter_series.index:
            val = self._crash_filter_series.loc[date]
            if pd.notna(val):
                return float(val)
        # Find nearest prior date
        prior = self._crash_filter_series.index[self._crash_filter_series.index <= date]
        if len(prior) > 0:
            val = self._crash_filter_series.loc[prior[-1]]
            if pd.notna(val):
                return float(val)
        return 1.0

    # -----------------------------------------------------------------
    # Transaction Costs
    # -----------------------------------------------------------------

    def _buy_cost(self, amount: float) -> float:
        brokerage = amount * self.config.brokerage_pct
        gst = brokerage * self.config.gst_pct
        stamp = amount * self.config.stamp_duty_pct
        slippage = amount * self.config.slippage_pct
        return brokerage + gst + stamp + slippage

    def _sell_cost(self, amount: float) -> float:
        brokerage = amount * self.config.brokerage_pct
        gst = brokerage * self.config.gst_pct
        stt = amount * self.config.stt_pct
        slippage = amount * self.config.slippage_pct
        return brokerage + gst + stt + slippage

    # -----------------------------------------------------------------
    # Entry / Exit Logic
    # -----------------------------------------------------------------

    def _check_exits(self, date: datetime) -> List[KC6Trade]:
        """Check all active positions for exit signals (uses pre-built lookup)."""
        completed = []
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            prices = self._get_price(sym, date)
            if prices is None:
                continue

            close, high, low, kc6_mid = prices
            hold_days = (date - pos.entry_date).days

            exit_reason = None
            exit_price = None

            # Priority 1: Stop Loss
            if low <= pos.sl_price:
                exit_reason = 'STOP_LOSS'
                exit_price = pos.sl_price

            # Priority 2: Take Profit
            elif high >= pos.tp_price:
                exit_reason = 'TAKE_PROFIT'
                exit_price = pos.tp_price

            # Priority 3: Max Hold
            elif hold_days >= self.config.max_hold_days:
                exit_reason = 'MAX_HOLD'
                exit_price = close

            # Priority 4: Signal - high > KC6 mid
            elif kc6_mid is not None and high > kc6_mid:
                exit_reason = 'SIGNAL_KC6_MID'
                exit_price = kc6_mid

            if exit_reason:
                sell_value = exit_price * pos.quantity
                sell_costs = self._sell_cost(sell_value)
                gross_pnl = (exit_price - pos.entry_price) * pos.quantity
                buy_costs = self._buy_cost(pos.invested)
                net_pnl = gross_pnl - sell_costs - buy_costs

                trade = KC6Trade(
                    symbol=sym,
                    entry_date=pos.entry_date,
                    exit_date=date,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    quantity=pos.quantity,
                    invested=pos.invested,
                    gross_pnl=gross_pnl,
                    costs=sell_costs + buy_costs,
                    net_pnl=net_pnl,
                    return_pct=(exit_price / pos.entry_price - 1) * 100,
                    exit_reason=exit_reason,
                    holding_days=hold_days,
                )
                self.trades.append(trade)
                completed.append(trade)
                self.cash += sell_value - sell_costs
                del self.positions[sym]

        return completed

    def _scan_and_enter(self, date: datetime,
                        allowed_symbols: Optional[Set[str]] = None):
        """Scan for entry signals using pre-built lookup."""
        if len(self.positions) >= self.config.max_positions:
            return

        slots = self.config.max_positions - len(self.positions)
        raw_candidates = self._entry_candidates.get(date, [])
        if not raw_candidates:
            return

        # Filter by allowed symbols and existing positions
        candidates = []
        for sym, close, dip_pct in raw_candidates:
            if sym in self.positions:
                continue
            if allowed_symbols is not None and sym not in allowed_symbols:
                continue
            candidates.append((sym, close, dip_pct))

        # Sort by dip depth (deepest first)
        candidates.sort(key=lambda x: x[2], reverse=True)

        for sym, close, _ in candidates[:slots]:
            if self._pool_capital_fn:
                # Shared pool mode: borrow from external pool
                pool_capital = self._pool_capital_fn(date)
                kc6_invested = sum(
                    (self._get_price(s, date) or (p.entry_price,))[0] * p.quantity
                    for s, p in self.positions.items()
                )
                available = pool_capital + self.cash - kc6_invested
                position_capital = pool_capital * self.config.position_size_pct
                if position_capital > available:
                    position_capital = available
            else:
                total_value = self._total_value(date)
                position_capital = total_value * self.config.position_size_pct
                if position_capital > self.cash:
                    position_capital = self.cash
            if position_capital < 10000:
                break

            buy_costs = self._buy_cost(position_capital)
            net_capital = position_capital - buy_costs
            quantity = int(net_capital / close)
            if quantity <= 0:
                continue

            invested = close * quantity
            self.cash -= (invested + buy_costs)

            self.positions[sym] = KC6Position(
                symbol=sym,
                entry_date=date,
                entry_price=close,
                quantity=quantity,
                invested=invested,
                sl_price=round(close * (1 - self.config.sl_pct / 100), 2),
                tp_price=round(close * (1 + self.config.tp_pct / 100), 2),
            )

    def _total_value(self, date: datetime) -> float:
        """Current total portfolio value (cash + positions)."""
        value = self.cash
        for sym, pos in self.positions.items():
            prices = self._get_price(sym, date)
            if prices:
                value += prices[0] * pos.quantity  # close price
            else:
                value += pos.entry_price * pos.quantity
        return value

    # -----------------------------------------------------------------
    # Main Run
    # -----------------------------------------------------------------

    def run(self, allowed_symbols_fn=None, pool_capital_fn=None) -> KC6BacktestResult:
        """
        Run the KC6 backtest.

        Args:
            allowed_symbols_fn: Optional callable(date: datetime) -> Set[str]
                If provided, only take entries on returned symbols for that date.
                Used for Case 1 (KC on MQ+ holdings only).
                If None, trade all universe symbols (Case 2/3).
            pool_capital_fn: Optional callable(date: datetime) -> float
                If provided, KC6 "borrows" from an external capital pool
                (e.g. MQ+'s idle cash). Position sizing uses pool capital.
                KC6's own cash tracks net P/L from trades.

        Returns:
            KC6BacktestResult with all metrics.
        """
        # Reset state for fresh run (allows reuse with precomputed data)
        self.positions = {}
        self.trades = []
        self.cash = self.config.initial_capital
        self.daily_equity = {}
        self._pool_capital_fn = pool_capital_fn

        # Step 1: Enrich data (skips if precomputed)
        self._enrich_data()
        if not self.enriched_data:
            return KC6BacktestResult()

        # Step 1b: Build fast signal and price lookups
        if not hasattr(self, '_entry_candidates') or not self._entry_candidates:
            self._build_signal_lookup()

        # Step 2: Get trading days
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)

        # Use the first stock's dates as reference for trading days
        all_dates = set()
        for df in self.enriched_data.values():
            all_dates.update(df.index)
        trading_days = sorted([d for d in all_dates if start <= d <= end])

        if len(trading_days) < 2:
            return KC6BacktestResult()

        # Warmup: need SMA200 + ATR warmup, skip first year
        warmup_end = start + timedelta(days=365)
        sim_days = [d for d in trading_days if d >= warmup_end]
        if len(sim_days) < 10:
            sim_days = trading_days[250:]  # fallback

        # Step 3: Pre-compute crash filter
        self._precompute_crash_filter(trading_days)

        # Step 4: Main simulation loop
        crash_filter_count = 0
        total_positions_sum = 0
        max_concurrent = 0

        for day_idx, date in enumerate(sim_days):
            # Accrue debt on idle cash (daily) - skip in pool mode
            if not self._pool_capital_fn:
                daily_debt_rate = self.config.debt_rate / 100 / 365
                self.cash *= (1 + daily_debt_rate)

            # Check exits first
            self._check_exits(date)

            # Check crash filter
            universe_atr = self._get_crash_filter(date)
            crash_active = universe_atr >= self.config.atr_ratio_threshold

            if crash_active:
                crash_filter_count += 1
            else:
                # Determine allowed symbols
                allowed = None
                if allowed_symbols_fn is not None:
                    allowed = allowed_symbols_fn(date)

                # Scan and enter
                self._scan_and_enter(date, allowed)

            # Track position count
            n_pos = len(self.positions)
            total_positions_sum += n_pos
            max_concurrent = max(max_concurrent, n_pos)

            # Record daily equity
            self.daily_equity[date.strftime('%Y-%m-%d')] = self._total_value(date)

        # Step 5: Close any remaining positions at last day's close
        last_day = sim_days[-1]
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            prices = self._get_price(sym, last_day)
            close = prices[0] if prices else pos.entry_price

            sell_value = close * pos.quantity
            sell_costs = self._sell_cost(sell_value)
            gross_pnl = (close - pos.entry_price) * pos.quantity
            net_pnl = gross_pnl - sell_costs - self._buy_cost(pos.invested)

            trade = KC6Trade(
                symbol=sym,
                entry_date=pos.entry_date,
                exit_date=last_day,
                entry_price=pos.entry_price,
                exit_price=close,
                quantity=pos.quantity,
                invested=pos.invested,
                gross_pnl=gross_pnl,
                costs=sell_costs + self._buy_cost(pos.invested),
                net_pnl=net_pnl,
                return_pct=(close / pos.entry_price - 1) * 100,
                exit_reason='END_OF_PERIOD',
                holding_days=(last_day - pos.entry_date).days,
            )
            self.trades.append(trade)
            self.cash += sell_value - sell_costs
            del self.positions[sym]

        # Step 6: Compute metrics
        return self._build_result(
            sim_days, crash_filter_count,
            total_positions_sum / max(len(sim_days), 1),
            max_concurrent
        )

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    def _build_result(self, sim_days, crash_days, avg_pos, max_conc) -> KC6BacktestResult:
        result = KC6BacktestResult()
        result.trade_log = self.trades
        result.daily_equity = self.daily_equity
        result.crash_filter_days = crash_days
        result.avg_positions = round(avg_pos, 2)
        result.max_concurrent = max_conc
        result.total_trades = len(self.trades)

        if not self.trades:
            result.final_value = self.config.initial_capital
            return result

        # Final value
        result.final_value = self.cash
        result.total_return_pct = (result.final_value / self.config.initial_capital - 1) * 100

        # CAGR
        years = (sim_days[-1] - sim_days[0]).days / 365.25
        if years > 0 and result.final_value > 0:
            result.cagr = ((result.final_value / self.config.initial_capital) ** (1 / years) - 1) * 100
        else:
            result.cagr = 0

        # Win/loss stats
        winners = [t for t in self.trades if t.return_pct > 0]
        losers = [t for t in self.trades if t.return_pct <= 0]
        result.win_rate = len(winners) / len(self.trades) * 100 if self.trades else 0
        result.avg_win_pct = np.mean([t.return_pct for t in winners]) if winners else 0
        result.avg_loss_pct = np.mean([t.return_pct for t in losers]) if losers else 0

        # Profit factor
        total_wins = sum(t.net_pnl for t in winners)
        total_losses = abs(sum(t.net_pnl for t in losers))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Exit reason counts
        reasons = {}
        for t in self.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        result.exit_reason_counts = reasons

        # Sharpe, Sortino, MaxDD from equity curve
        if len(self.daily_equity) > 30:
            eq = pd.Series(self.daily_equity, dtype=float)
            eq.index = pd.to_datetime(eq.index)
            eq = eq.sort_index()

            daily_ret = eq.pct_change().dropna()

            if len(daily_ret) > 0 and daily_ret.std() > 0:
                rf_daily = 0.07 / 252  # 7% risk-free
                excess = daily_ret - rf_daily
                result.sharpe = round(float(excess.mean() / excess.std() * np.sqrt(252)), 2)

                downside = daily_ret[daily_ret < 0]
                if len(downside) > 0 and downside.std() > 0:
                    result.sortino = round(float(excess.mean() / downside.std() * np.sqrt(252)), 2)

            # Max drawdown
            peak = eq.expanding().max()
            dd = ((eq - peak) / peak * 100)
            result.max_drawdown = round(abs(float(dd.min())), 1)

            # Calmar
            if result.max_drawdown > 0:
                result.calmar = round(result.cagr / result.max_drawdown, 2)

        return result


# =============================================================================
# Helper: Extract MQ+ daily holdings from trade log
# =============================================================================

def extract_mq_daily_holdings(mq_result) -> Dict[str, Set[str]]:
    """
    Build a date -> set(symbols) map from MQ+ backtest result.
    Used for Case 1 (KC6 on MQ holdings only).

    Args:
        mq_result: BacktestResult from MQBacktestEngine.run()

    Returns:
        Dict mapping date strings to sets of held symbols.
    """
    # Build intervals from trade_log (completed trades)
    intervals = []  # (symbol, entry_date, exit_date)
    for t in mq_result.trade_log:
        entry = t.entry_date
        exit_d = t.exit_date
        if isinstance(entry, str):
            entry = datetime.strptime(entry, '%Y-%m-%d')
        if isinstance(exit_d, str):
            exit_d = datetime.strptime(exit_d, '%Y-%m-%d')
        intervals.append((t.symbol, entry, exit_d))

    # Add final positions (still open)
    for p in mq_result.final_positions:
        entry = p['entry_date']
        if isinstance(entry, str):
            entry = datetime.strptime(entry, '%Y-%m-%d')
        # Open until end of simulation
        intervals.append((p['symbol'], entry, datetime(2099, 12, 31)))

    # Build daily map from equity curve dates
    daily_holdings = {}
    eq_dates = sorted(mq_result.daily_equity.keys())

    for date_str in eq_dates:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        held = set()
        for sym, entry, exit_d in intervals:
            if entry <= date < exit_d:
                held.add(sym)
        daily_holdings[date_str] = held

    return daily_holdings


# =============================================================================
# Preload helper (reuses MQ engine's data loading)
# =============================================================================

def preload_kc6_data(start_date='2010-01-01', end_date='2026-02-16'):
    """
    Load price data for KC6 backtesting.
    Reuses MQ engine's data loader for consistency.

    Returns:
        (universe_symbols, price_data) tuple
    """
    from services.mq_backtest_engine import MQBacktestEngine
    from services.mq_portfolio import MQBacktestConfig

    config = MQBacktestConfig(
        start_date=start_date,
        end_date=end_date,
    )
    universe, price_data = MQBacktestEngine.preload_data(config)
    return universe.symbols, price_data
