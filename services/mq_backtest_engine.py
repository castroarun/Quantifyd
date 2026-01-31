"""
Momentum + Quality Backtest Engine
====================================

Event-driven backtest loop for the MQ strategy.

Flow:
1. Pre-load all daily OHLCV data for the universe
2. Day 1: Run initial screening → select top 30 → enter positions
3. Daily: Update prices, check hard stops, detect consolidation/breakout topups
4. Semi-annual (Jan/Jul): Rebalance - exit drawdown positions, replace
5. At end: Calculate metrics, return BacktestResult

Limitations (MVP):
- Fundamental exit triggers are skipped (no historical quarterly earnings data)
- Uses current Yahoo Finance fundamentals for initial screening quality gate
- Consolidation/breakout checked weekly (not daily) for performance
"""

import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .mq_portfolio import (
    MQBacktestConfig, Portfolio, Position, Trade, TopupRecord, ExitReason,
)
from .momentum_filter import screen_momentum_fast
from .consolidation_breakout import detect_consolidation, detect_breakout
from .nifty500_universe import load_nifty500, Nifty500Universe

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'

TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# Backtest Result
# =============================================================================

@dataclass
class BacktestResult:
    """Complete backtest output."""
    config: MQBacktestConfig

    # Portfolio state
    initial_capital: float
    final_value: float

    # Performance
    total_return_pct: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    total_topups: int

    # Series data
    daily_equity: Dict[str, float] = field(default_factory=dict)
    daily_debt_fund: Dict[str, float] = field(default_factory=dict)
    trade_log: List[Trade] = field(default_factory=list)
    topup_log: List[TopupRecord] = field(default_factory=list)
    screening_log: List[Dict] = field(default_factory=list)
    rebalance_log: List[Dict] = field(default_factory=list)

    # Breakdown
    sector_allocation: Dict[str, float] = field(default_factory=dict)
    final_positions: List[Dict] = field(default_factory=list)
    exit_reason_counts: Dict[str, int] = field(default_factory=dict)
    exit_reason_pnl: Dict[str, Dict] = field(default_factory=dict)

    @property
    def years(self) -> float:
        start = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.config.end_date, '%Y-%m-%d')
        return (end - start).days / 365.25


# =============================================================================
# Data Loader
# =============================================================================

def _load_all_daily_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    db_path: Path = None,
) -> Dict[str, pd.DataFrame]:
    """
    Pre-load all daily OHLCV data into memory for fast lookups.

    Returns:
        Dict mapping symbol -> DataFrame with date index and OHLCV columns
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)

    placeholders = ','.join('?' * len(symbols))
    query = f"""
        SELECT symbol, date, open, high, low, close, volume
        FROM market_data_unified
        WHERE symbol IN ({placeholders})
          AND timeframe = 'day'
          AND date >= ? AND date <= ?
        ORDER BY symbol, date
    """
    params = symbols + [start_date, end_date]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    data = {}
    for symbol, group in df.groupby('symbol'):
        sdf = group.set_index('date').sort_index()
        sdf = sdf[['open', 'high', 'low', 'close', 'volume']]
        data[symbol] = sdf

    logger.info(f"Loaded daily data: {len(data)} symbols, {len(df)} total candles")
    return data


def _get_trading_days(price_data: Dict[str, pd.DataFrame]) -> List[datetime]:
    """
    Extract unique trading days from the loaded price data.

    Uses the union of all dates across all symbols.
    """
    all_dates = set()
    for df in price_data.values():
        all_dates.update(df.index.tolist())
    return sorted(all_dates)


def _get_prices_on_date(
    price_data: Dict[str, pd.DataFrame],
    date: datetime,
    symbols: List[str] = None,
) -> Dict[str, float]:
    """Get close prices for all symbols on a given date."""
    prices = {}
    check_symbols = symbols or price_data.keys()
    for symbol in check_symbols:
        df = price_data.get(symbol)
        if df is not None and date in df.index:
            prices[symbol] = df.loc[date, 'close']
    return prices


# =============================================================================
# Screening (simplified for backtest — uses momentum + sector limits only)
# =============================================================================

def _run_screening(
    universe: Nifty500Universe,
    price_data: Dict[str, pd.DataFrame],
    as_of_date: datetime,
    config: MQBacktestConfig,
    exclude_symbols: set = None,
    quality_scores: Dict = None,
) -> List[Dict]:
    """
    Run screening at a point in time during backtest.

    Returns ranked list of candidates: [{'symbol', 'sector', 'price', 'distance_from_ath', 'quality_score'}]
    """
    exclude = exclude_symbols or set()
    candidates = []

    # Available symbols that have price data on this date
    available = [s for s in universe.symbols if s in price_data and s not in exclude]

    # Momentum filter: check rolling high proximity
    # Uses 52-week lookback when available, or all available history (min 20 days)
    for symbol in available:
        df = price_data[symbol]

        # Get data up to as_of_date
        hist = df[df.index <= as_of_date]
        if len(hist) < 20:
            continue

        # Use up to 52 weeks of history, but accept whatever is available (min 20 days)
        lookback = as_of_date - timedelta(days=365)
        year_data = hist[hist.index >= lookback]
        if len(year_data) < 20:
            year_data = hist.tail(max(20, len(hist)))  # Use all available

        high_52w = year_data['high'].max()
        current_price = hist.iloc[-1]['close']

        if high_52w <= 0:
            continue

        distance = (high_52w - current_price) / high_52w

        if distance <= config.ath_proximity_threshold:
            stock = universe.get(symbol)
            sector = stock.sector if stock else 'Unknown'

            # Quality score (if available from pre-computed data)
            q_score = 0.5  # Default mid-range score
            if quality_scores and symbol in quality_scores:
                qs = quality_scores[symbol]
                if hasattr(qs, 'composite_score'):
                    q_score = qs.composite_score
                elif isinstance(qs, (int, float)):
                    q_score = qs

            candidates.append({
                'symbol': symbol,
                'sector': sector,
                'price': round(current_price, 2),
                'distance_from_ath': round(distance, 4),
                'quality_score': q_score,
                'is_financial': stock.is_financial if stock else False,
            })

    # Sort by distance from ATH (closest first), then by quality score
    candidates.sort(key=lambda c: (c['distance_from_ath'], -c['quality_score']))

    return candidates


# =============================================================================
# Engine
# =============================================================================

class MQBacktestEngine:
    """
    Event-driven backtest engine for the Momentum + Quality strategy.

    Usage:
        config = MQBacktestConfig()
        engine = MQBacktestEngine(config)
        result = engine.run()
    """

    def __init__(self, config: MQBacktestConfig = None):
        self.config = config or MQBacktestConfig()
        self.universe: Optional[Nifty500Universe] = None
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.portfolio: Optional[Portfolio] = None
        self.quality_scores: Dict = {}
        self._screening_log: List[Dict] = []
        self._rebalance_log: List[Dict] = []

    def run(
        self,
        quality_scores: Dict = None,
        progress_callback=None,
    ) -> BacktestResult:
        """
        Run the full backtest simulation.

        Args:
            quality_scores: Optional pre-computed quality scores {symbol: score}
            progress_callback: Optional callback(day_idx, total_days, date, message)

        Returns:
            BacktestResult with all metrics and logs
        """
        self.quality_scores = quality_scores or {}

        # Phase 1: Load data — extend 1 year before start for momentum lookback
        logger.info("Phase 1: Loading universe and price data...")
        self.universe = load_nifty500()

        sim_start_dt = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        data_start = (sim_start_dt - timedelta(days=400)).strftime('%Y-%m-%d')

        self.price_data = _load_all_daily_data(
            self.universe.symbols,
            data_start,
            self.config.end_date,
        )

        # Separate pre-sim history from simulation period
        all_days = _get_trading_days(self.price_data)
        trading_days = [d for d in all_days if d >= sim_start_dt]
        pre_sim_days = len([d for d in all_days if d < sim_start_dt])

        if not trading_days:
            raise ValueError("No trading days found in the specified period")

        logger.info(f"  {len(trading_days)} trading days from {trading_days[0].date()} to {trading_days[-1].date()}")

        # Phase 2: Initialize portfolio
        self.portfolio = Portfolio(self.config)
        self.portfolio.equity_cash = self.config.equity_capital

        # Determine warmup: need ~60 days of price history for momentum screening
        # Pre-simulation data (loaded from 1yr before) counts toward this
        min_history_days = 60
        warmup_needed = max(0, min_history_days - pre_sim_days)
        warmup_idx = min(warmup_needed, len(trading_days) - 1)
        portfolio_start_day = trading_days[warmup_idx]

        logger.info(
            f"  Pre-sim history: {pre_sim_days} days | "
            f"Warmup: {warmup_needed} sim-days | "
            f"Portfolio construction: {portfolio_start_day.date()}"
        )

        # Phase 3: Daily simulation loop (includes warmup + active trading)
        logger.info(f"Phase 2: Running simulation ({len(trading_days)} days)...")
        total_days = len(trading_days)
        last_consol_check = trading_days[0]
        portfolio_built = False

        for day_idx, current_date in enumerate(trading_days):
            prices = _get_prices_on_date(self.price_data, current_date)
            if not prices:
                continue

            # Update portfolio prices + accrue debt fund
            self.portfolio.update_prices(prices, current_date)

            # Build initial portfolio after warmup completes
            if not portfolio_built and current_date >= portfolio_start_day:
                logger.info("Phase 3: Initial screening and portfolio construction...")
                self._build_initial_portfolio(current_date)
                portfolio_built = True
                continue  # Skip other checks on construction day

            if not portfolio_built:
                continue  # Warmup period — just tracking equity

            # Check hard stop losses
            self._check_hard_stops(current_date, prices)

            # Semi-annual rebalance check (1st trading day of Jan/Jul)
            if current_date.month in self.config.rebalance_months and current_date.day <= 7:
                prev_day = trading_days[day_idx - 1] if day_idx > 0 else None
                if prev_day is None or prev_day.month != current_date.month:
                    self._run_rebalance(current_date, prices)

            # Consolidation/breakout check (weekly for performance)
            if (current_date - last_consol_check).days >= 5:
                self._check_breakout_topups(current_date)
                last_consol_check = current_date

            # Progress callback
            if progress_callback and day_idx % 50 == 0:
                progress_callback(
                    day_idx, total_days, current_date,
                    f"Day {day_idx}/{total_days} | "
                    f"Value: {self.portfolio.total_value:,.0f} | "
                    f"Positions: {self.portfolio.position_count}"
                )

        # Phase 4: Calculate results
        logger.info("Phase 4: Calculating results...")
        return self._build_result()

    # -------------------------------------------------------------------------
    # Initial Portfolio Construction
    # -------------------------------------------------------------------------

    def _build_initial_portfolio(self, start_date: datetime):
        """Screen and build the initial 30-stock portfolio."""
        candidates = _run_screening(
            self.universe, self.price_data, start_date,
            self.config, quality_scores=self.quality_scores,
        )

        entered = 0
        skipped_sector = 0

        for candidate in candidates:
            if entered >= self.config.portfolio_size:
                break

            symbol = candidate['symbol']
            sector = candidate['sector']
            price = candidate['price']

            position = self.portfolio.enter_position(
                symbol, sector, price, start_date
            )
            if position:
                entered += 1
            else:
                skipped_sector += 1

        self._screening_log.append({
            'date': start_date.strftime('%Y-%m-%d'),
            'type': 'initial',
            'candidates': len(candidates),
            'entered': entered,
            'skipped_sector': skipped_sector,
        })

        logger.info(
            f"  Initial portfolio: {entered} positions from {len(candidates)} candidates "
            f"({skipped_sector} skipped for sector limits)"
        )

    # -------------------------------------------------------------------------
    # Hard Stop Check
    # -------------------------------------------------------------------------

    def _check_hard_stops(self, current_date: datetime, prices: Dict[str, float]):
        """Exit positions that hit hard stop loss."""
        to_exit = []
        for symbol in list(self.portfolio.positions.keys()):
            if self.portfolio.check_hard_stop(symbol):
                to_exit.append(symbol)

        for symbol in to_exit:
            price = prices.get(symbol, self.portfolio.positions[symbol].current_price)
            self.portfolio.exit_position(symbol, price, current_date, ExitReason.HARD_STOP)

    # -------------------------------------------------------------------------
    # Semi-Annual Rebalance
    # -------------------------------------------------------------------------

    def _run_rebalance(self, current_date: datetime, prices: Dict[str, float]):
        """
        Semi-annual rebalance:
        1. Exit positions with >20% ATH drawdown
        2. Screen for replacements
        3. Enter replacement positions
        """
        exits = []

        # Step 1: Check ATH drawdown for all positions
        for symbol in list(self.portfolio.positions.keys()):
            if self.portfolio.check_ath_drawdown_exit(symbol):
                position = self.portfolio.positions[symbol]
                price = prices.get(symbol, position.current_price)
                trade = self.portfolio.exit_position(
                    symbol, price, current_date, ExitReason.ATH_DRAWDOWN
                )
                if trade:
                    exits.append(trade)

        if not exits:
            self._rebalance_log.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'exits': 0,
                'entries': 0,
                'message': 'No positions exited',
            })
            return

        # Step 2: Screen for replacements
        current_symbols = set(self.portfolio.positions.keys())
        candidates = _run_screening(
            self.universe, self.price_data, current_date,
            self.config, exclude_symbols=current_symbols,
            quality_scores=self.quality_scores,
        )

        # Step 3: Enter replacement positions
        entries = 0
        capital_per_entry = sum(t.exit_value for t in exits) / len(exits) if exits else 0

        for candidate in candidates:
            if entries >= len(exits):
                break

            symbol = candidate['symbol']
            sector = candidate['sector']
            price = candidate['price']

            # Use recycled capital from exits
            position = self.portfolio.enter_position(
                symbol, sector, price, current_date,
                capital=capital_per_entry,
            )
            if position:
                entries += 1

        self._rebalance_log.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'exits': len(exits),
            'entries': entries,
            'exit_symbols': [t.symbol for t in exits],
            'entry_symbols': [c['symbol'] for c in candidates[:entries]],
        })

        logger.info(
            f"  Rebalance {current_date.date()}: "
            f"exited {len(exits)}, entered {entries}"
        )

    # -------------------------------------------------------------------------
    # Consolidation / Breakout Topups
    # -------------------------------------------------------------------------

    def _check_breakout_topups(self, current_date: datetime):
        """Check portfolio stocks for consolidation breakouts → topups."""
        for symbol in list(self.portfolio.positions.keys()):
            df = self.price_data.get(symbol)
            if df is None:
                continue

            # Only use data up to current_date
            hist = df[df.index <= current_date]
            if len(hist) < 25:
                continue

            # Quick consolidation check using recent data
            recent = hist.tail(60)
            if len(recent) < 20:
                continue

            # Check if in consolidation (last 20+ days in 5% range)
            window = recent.tail(self.config.consolidation_days)
            if len(window) < self.config.consolidation_days:
                continue

            range_high = window['high'].max()
            range_low = window['low'].min()
            midpoint = (range_high + range_low) / 2

            if midpoint <= 0:
                continue

            range_pct = (range_high - range_low) / midpoint

            if range_pct > self.config.consolidation_range_pct:
                continue  # Not in tight consolidation

            # Check breakout: latest close > range_high * 1.02
            latest_close = hist.iloc[-1]['close']
            latest_volume = hist.iloc[-1]['volume']
            breakout_level = range_high * 1.02

            if latest_close <= breakout_level:
                continue  # No price breakout

            # Volume confirmation
            avg_vol = recent['volume'].tail(20).mean()
            if avg_vol <= 0:
                continue

            vol_ratio = latest_volume / avg_vol
            if vol_ratio < self.config.breakout_volume_multiplier:
                continue  # Volume not confirmed

            # Execute topup
            self.portfolio.execute_topup(symbol, latest_close, current_date)

    # -------------------------------------------------------------------------
    # Build Result
    # -------------------------------------------------------------------------

    def _build_result(self) -> BacktestResult:
        """Calculate final metrics and build the result object."""
        portfolio = self.portfolio
        config = self.config

        # Convert daily equity to Series for metric calculation
        if portfolio.daily_equity:
            equity_series = pd.Series(portfolio.daily_equity, dtype=float)
            equity_series.index = pd.to_datetime(equity_series.index)
            equity_series = equity_series.sort_index()
        else:
            equity_series = pd.Series(dtype=float)

        initial = config.initial_capital
        final = portfolio.total_value
        total_return = (final - initial) / initial if initial > 0 else 0

        # CAGR
        start_dt = datetime.strptime(config.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(config.end_date, '%Y-%m-%d')
        years = (end_dt - start_dt).days / 365.25
        cagr = (final / initial) ** (1 / years) - 1 if years > 0 and initial > 0 else 0

        # Daily returns for Sharpe/Sortino
        daily_returns = equity_series.pct_change().dropna()

        # Sharpe ratio
        if len(daily_returns) > 10 and daily_returns.std() > 0:
            excess = daily_returns.mean() - (0.07 / TRADING_DAYS_PER_YEAR)  # 7% risk-free
            sharpe = (excess / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            sharpe = 0.0

        # Sortino ratio
        if len(daily_returns) > 10:
            downside = daily_returns[daily_returns < 0]
            downside_std = downside.std() if len(downside) > 0 else 0
            if downside_std > 0:
                excess = daily_returns.mean() - (0.07 / TRADING_DAYS_PER_YEAR)
                sortino = (excess / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
            else:
                sortino = 0.0
        else:
            sortino = 0.0

        # Max drawdown
        if len(equity_series) > 1:
            rolling_max = equity_series.cummax()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_dd = abs(drawdowns.min())
        else:
            max_dd = 0.0

        # Calmar ratio (CAGR / max drawdown)
        calmar = cagr / max_dd if max_dd > 0 else 0.0

        # Trade stats
        trades = portfolio.trades
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]
        win_rate = len(winners) / len(trades) if trades else 0

        avg_win = np.mean([t.return_pct for t in winners]) if winners else 0
        avg_loss = np.mean([t.return_pct for t in losers]) if losers else 0

        # Exit reason breakdown with P&L stats
        exit_counts = {}
        exit_reason_trades = {}  # reason -> list of trades
        for t in trades:
            reason = t.exit_reason.value if hasattr(t.exit_reason, 'value') else str(t.exit_reason)
            exit_counts[reason] = exit_counts.get(reason, 0) + 1
            exit_reason_trades.setdefault(reason, []).append(t)

        exit_reason_pnl = {}
        for reason, reason_trades in exit_reason_trades.items():
            pnls = [t.net_pnl for t in reason_trades]
            returns = [t.return_pct for t in reason_trades]
            hold_days = [t.holding_days for t in reason_trades]
            reason_winners = [t for t in reason_trades if t.net_pnl > 0]
            exit_reason_pnl[reason] = {
                'count': len(reason_trades),
                'total_pnl': round(sum(pnls), 2),
                'avg_return_pct': round(float(np.mean(returns)) * 100, 2),
                'median_return_pct': round(float(np.median(returns)) * 100, 2),
                'avg_holding_days': round(float(np.mean(hold_days)), 0),
                'win_rate': round(len(reason_winners) / len(reason_trades) * 100, 1),
                'worst_return_pct': round(float(min(returns)) * 100, 2),
                'best_return_pct': round(float(max(returns)) * 100, 2),
            }

        return BacktestResult(
            config=config,
            initial_capital=initial,
            final_value=round(final, 2),
            total_return_pct=round(total_return * 100, 2),
            cagr=round(cagr * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            max_drawdown=round(max_dd * 100, 2),
            calmar_ratio=round(calmar, 2),
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=round(win_rate * 100, 1),
            avg_win_pct=round(avg_win * 100, 2),
            avg_loss_pct=round(avg_loss * 100, 2),
            total_topups=len(portfolio.topup_log),
            daily_equity=portfolio.daily_equity,
            daily_debt_fund=portfolio.daily_debt_fund,
            trade_log=portfolio.trades,
            topup_log=portfolio.topup_log,
            screening_log=self._screening_log,
            rebalance_log=self._rebalance_log,
            sector_allocation={
                s: round(w * 100, 1)
                for s, w in portfolio.sector_weights.items()
            },
            final_positions=portfolio.get_position_summary(),
            exit_reason_counts=exit_counts,
            exit_reason_pnl=exit_reason_pnl,
        )


# =============================================================================
# Convenience runner
# =============================================================================

def run_mq_backtest(
    config: MQBacktestConfig = None,
    quality_scores: Dict = None,
    progress_callback=None,
) -> BacktestResult:
    """
    Run a Momentum + Quality backtest with default or custom config.

    Args:
        config: Backtest configuration (defaults used if None)
        quality_scores: Pre-computed quality scores {symbol: score}
        progress_callback: Optional progress callback

    Returns:
        BacktestResult
    """
    engine = MQBacktestEngine(config)
    return engine.run(quality_scores, progress_callback)
