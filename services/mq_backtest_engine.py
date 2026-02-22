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
from .consolidation_breakout import (
    detect_consolidation, detect_breakout, detect_darvas_boxes, DarvasBoxDetector,
)
from .nifty500_universe import load_nifty500, Nifty500Universe
from .technical_indicators import (
    add_ema_signals, add_rsi_signals, add_stoch_signals,
    add_ichimoku_signals, add_supertrend_signals, add_macd_signals,
    add_adx_signals, add_bollinger_signals, calc_ema,
    calc_rsi, calc_stochastics, calc_bollinger_bands, calc_keltner_channels,
    calc_ichimoku, calc_supertrend, calc_macd, calc_adx, calc_atr,
)
from math import log as _log, sqrt as _sqrt, exp as _exp, erf as _erf

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
    consolidation_log: List[Dict] = field(default_factory=list)
    topup_sl_log: List[Dict] = field(default_factory=list)

    # Covered call overlay
    cc_trades: List[Dict] = field(default_factory=list)
    cc_total_premium: float = 0.0
    cc_total_buyback: float = 0.0
    cc_net_income: float = 0.0
    cc_total_calls_sold: int = 0
    cc_expired_otm: int = 0
    cc_expired_itm: int = 0
    cc_buybacks: int = 0
    cc_rolls: int = 0
    cc_stop_losses: int = 0

    @property
    def years(self) -> float:
        start = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.config.end_date, '%Y-%m-%d')
        return (end - start).days / 365.25


# =============================================================================
# NIFTYBEES Data (for idle cash deployment)
# =============================================================================

def _load_niftybees_data(start_date: str, end_date: str, db_path: Path = None) -> pd.DataFrame:
    """Load NIFTYBEES daily close prices and compute 200 SMA.
    Returns DataFrame with columns: close, sma200, below_sma200."""
    db_path = db_path or DB_PATH
    # Load extra history for 200 SMA warmup
    warmup_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=400)).strftime('%Y-%m-%d')
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT date, close FROM market_data_unified WHERE symbol='NIFTYBEES' AND timeframe='day' AND date >= ? AND date <= ? ORDER BY date",
        conn, params=[warmup_start, end_date]
    )
    conn.close()
    if df.empty:
        logger.warning("No NIFTYBEES data found!")
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df['sma200'] = df['close'].rolling(200).mean()
    df['below_sma200'] = df['close'] < df['sma200']
    df['above_sma200'] = ~df['below_sma200']
    # Compute consecutive days above/below 200 SMA
    cb, ca = 0, 0
    consec_below, consec_above = [], []
    for val in df['below_sma200']:
        if val:
            cb += 1
            ca = 0
        else:
            ca += 1
            cb = 0
        consec_below.append(cb)
        consec_above.append(ca)
    df['consec_below'] = consec_below
    df['consec_above'] = consec_above
    # Trim to requested period
    df = df[df.index >= start_date]
    return df


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
    technical_filter_fn=None,
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

    # Apply technical filter if provided
    if technical_filter_fn is not None:
        before_count = len(candidates)
        candidates = [c for c in candidates if technical_filter_fn(c['symbol'])]
        logger.debug(f"  Technical filter: {before_count} -> {len(candidates)} candidates")

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

    For optimization (load data once, reuse across runs):
        universe, price_data = MQBacktestEngine.preload_data(config)
        for cfg in configs:
            engine = MQBacktestEngine(cfg, universe, price_data)
            result = engine.run()
    """

    @staticmethod
    def preload_data(config: MQBacktestConfig = None):
        """Pre-load universe and price data for reuse across multiple runs."""
        config = config or MQBacktestConfig()
        universe = load_nifty500()
        sim_start_dt = datetime.strptime(config.start_date, '%Y-%m-%d')
        data_start = (sim_start_dt - timedelta(days=400)).strftime('%Y-%m-%d')
        price_data = _load_all_daily_data(
            universe.symbols, data_start, config.end_date
        )
        logger.info(f"Preloaded: {len(price_data)} symbols for {config.start_date} to {config.end_date}")
        return universe, price_data

    def __init__(self, config: MQBacktestConfig = None,
                 preloaded_universe: Optional['Nifty500Universe'] = None,
                 preloaded_price_data: Optional[Dict[str, pd.DataFrame]] = None):
        self.config = config or MQBacktestConfig()
        self.universe: Optional[Nifty500Universe] = preloaded_universe
        self.price_data: Dict[str, pd.DataFrame] = preloaded_price_data or {}
        self.portfolio: Optional[Portfolio] = None
        self.quality_scores: Dict = {}
        self._screening_log: List[Dict] = []
        self._data_preloaded = preloaded_universe is not None and preloaded_price_data is not None
        self._rebalance_log: List[Dict] = []
        self._consolidation_log: List[Dict] = []
        self._topup_sl_log: List[Dict] = []

        # Covered call overlay state
        self._cc_active: Dict[str, Dict] = {}    # symbol -> {strike, expiry_date, premium, shares, entry_price, vol}
        self._cc_trades: List[Dict] = []          # completed CC trades
        self._cc_total_premium: float = 0.0
        self._cc_total_buyback: float = 0.0
        self._cc_indicator_cache: Dict[str, pd.DataFrame] = {}  # pre-computed indicators

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
        sim_start_dt = datetime.strptime(self.config.start_date, '%Y-%m-%d')

        if self._data_preloaded:
            logger.debug("Phase 1: Using preloaded data (skipping DB load)")
        else:
            logger.info("Phase 1: Loading universe and price data...")
            self.universe = load_nifty500()
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

        # Load NIFTYBEES data if idle cash deployment is enabled
        self._niftybees_data = None
        if self.config.idle_cash_to_nifty_etf or self.config.idle_cash_to_debt:
            self._niftybees_data = _load_niftybees_data(
                self.config.start_date, self.config.end_date
            )
            if not self._niftybees_data.empty:
                logger.info(f"  NIFTYBEES data: {len(self._niftybees_data)} days for idle cash deployment")

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

            # Update NIFTYBEES price before portfolio value recording
            if self._niftybees_data is not None and not self._niftybees_data.empty:
                if current_date in self._niftybees_data.index:
                    self.portfolio.update_nifty_etf_price(self._niftybees_data.loc[current_date, 'close'])
                else:
                    prior = self._niftybees_data.index[self._niftybees_data.index <= current_date]
                    if len(prior) > 0:
                        self.portfolio.update_nifty_etf_price(self._niftybees_data.loc[prior[-1], 'close'])

            # Update portfolio prices + accrue debt fund
            self.portfolio.update_prices(prices, current_date)

            # Build initial portfolio after warmup completes
            if not portfolio_built and current_date >= portfolio_start_day:
                logger.info("Phase 3: Initial screening and portfolio construction...")
                self._build_initial_portfolio(current_date)
                portfolio_built = True
                # Pre-compute CC indicators after portfolio is built
                if self.config.cc_enabled:
                    self._precompute_cc_indicators()
                continue  # Skip other checks on construction day

            if not portfolio_built:
                continue  # Warmup period — just tracking equity

            # Check hard stop losses
            self._check_hard_stops(current_date, prices)

            # Check topup-specific stop losses (if configured)
            if self.config.topup_stop_loss_pct > 0:
                reversals = self.portfolio.check_topup_stop_losses(current_date)
                self._topup_sl_log.extend(reversals)

            # Daily ATH drawdown check (if enabled)
            if self.config.daily_ath_drawdown_exit:
                daily_exits = self._check_daily_ath_drawdown(current_date, prices)
                # Immediate replacement: screen & enter new positions for exited stocks
                if self.config.immediate_replacement and daily_exits:
                    self._replace_exited_positions(daily_exits, current_date, prices)

            # Technical exit signals (if enabled)
            if self.config.use_technical_filter:
                self._check_technical_exits(current_date, prices)

            # Semi-annual rebalance check (1st trading day of Jan/Jul)
            if current_date.month in self.config.rebalance_months and current_date.day <= 7:
                prev_day = trading_days[day_idx - 1] if day_idx > 0 else None
                if prev_day is None or prev_day.month != current_date.month:
                    self._run_rebalance(current_date, prices)

            # Consolidation/breakout check (weekly for performance)
            if (current_date - last_consol_check).days >= 5:
                self._check_breakout_topups(current_date)
                last_consol_check = current_date

            # Covered call overlay (if enabled)
            if self.config.cc_enabled:
                self._process_covered_calls(current_date, prices)

            # Idle cash deployment (NIFTYBEES or debt fund)
            if self._niftybees_data is not None and not self._niftybees_data.empty:
                self._deploy_idle_cash(current_date)

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
        # Create technical filter function if enabled
        tech_filter = None
        if self.config.use_technical_filter:
            tech_filter = lambda sym: self._check_technical_entry(sym, start_date)

        candidates = _run_screening(
            self.universe, self.price_data, start_date,
            self.config, quality_scores=self.quality_scores,
            technical_filter_fn=tech_filter,
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
    # Daily ATH Drawdown Exit
    # -------------------------------------------------------------------------

    def _check_daily_ath_drawdown(self, current_date: datetime, prices: Dict[str, float]):
        """Exit positions that breach ATH drawdown threshold (daily check).
        Returns list of exited Trade objects (for immediate replacement logic)."""
        to_exit = []
        for symbol in list(self.portfolio.positions.keys()):
            if self.portfolio.check_ath_drawdown_exit(symbol):
                to_exit.append(symbol)

        exits = []
        for symbol in to_exit:
            price = prices.get(symbol, self.portfolio.positions[symbol].current_price)
            trade = self.portfolio.exit_position(
                symbol, price, current_date, ExitReason.ATH_DRAWDOWN
            )
            if trade:
                exits.append(trade)
        return exits

    def _replace_exited_positions(self, exits: list, current_date: datetime, prices: Dict[str, float]):
        """Immediately screen and enter replacement positions for exited stocks."""
        if not exits:
            return

        current_symbols = set(self.portfolio.positions.keys())
        exited_symbols = {t.symbol for t in exits}

        tech_filter = None
        if self.config.use_technical_filter:
            tech_filter = lambda sym: self._check_technical_entry(sym, current_date)

        candidates = _run_screening(
            self.universe, self.price_data, current_date,
            self.config,
            exclude_symbols=current_symbols | exited_symbols,
            quality_scores=self.quality_scores,
            technical_filter_fn=tech_filter,
        )

        entries = 0
        capital_per_entry = sum(t.exit_value for t in exits) / len(exits) if exits else 0

        for candidate in candidates:
            if entries >= len(exits):
                break

            symbol = candidate['symbol']
            sector = candidate['sector']
            price = candidate['price']

            position = self.portfolio.enter_position(
                symbol, sector, price, current_date,
                capital=capital_per_entry,
            )
            if position:
                entries += 1

        if entries > 0:
            logger.info(
                f"  Immediate replacement {current_date.date()}: "
                f"exited {len(exits)}, replaced {entries}"
            )

    # -------------------------------------------------------------------------
    # Idle Cash Deployment
    # -------------------------------------------------------------------------

    def _deploy_idle_cash(self, current_date: datetime):
        """Deploy idle equity_cash into NIFTYBEES ETF or debt fund based on Nifty 200 SMA."""
        if self._niftybees_data is None or self._niftybees_data.empty:
            return

        # Get NIFTYBEES data for current date
        if current_date not in self._niftybees_data.index:
            # Try nearest prior date
            prior = self._niftybees_data.index[self._niftybees_data.index <= current_date]
            if len(prior) == 0:
                return
            current_date_key = prior[-1]
        else:
            current_date_key = current_date

        row = self._niftybees_data.loc[current_date_key]
        nifty_price = row['close']
        below_sma200 = row['below_sma200']
        consec_below = int(row.get('consec_below', 1))
        consec_above = int(row.get('consec_above', 1))

        # Update NIFTYBEES value with current price
        self.portfolio.update_nifty_etf_price(nifty_price)

        if self.config.idle_cash_to_nifty_etf:
            confirm = self.config.nifty_sma_confirm_days

            if self.config.nifty_etf_above_sma:
                # INVERTED: NiftyBEES when ABOVE 200 SMA, debt when BELOW
                # Use confirmation days for switching
                if consec_above >= confirm:
                    # Confirmed above SMA → deploy to NiftyBEES
                    # First liquidate any NiftyBEES held from wrong side? No, we WANT NiftyBEES here
                    deploy_to_nifty = True
                elif consec_below >= confirm:
                    # Confirmed below SMA → switch to debt, sell NiftyBEES
                    if self.portfolio.nifty_etf_units > 0:
                        self.portfolio.liquidate_nifty_etf(nifty_price)
                    deploy_to_nifty = False
                else:
                    # In transition / not yet confirmed → hold current state, no new deployment
                    return

                if deploy_to_nifty:
                    self.portfolio.deploy_idle_cash(nifty_price, True, current_date)
                else:
                    # Park in debt
                    if self.portfolio.equity_cash > 0:
                        self.portfolio.idle_cash_in_debt += self.portfolio.equity_cash
                        self.portfolio.equity_cash = 0.0
            else:
                # Original logic: NIFTYBEES when below 200 SMA, else debt
                self.portfolio.deploy_idle_cash(nifty_price, below_sma200, current_date)
        elif self.config.idle_cash_to_debt:
            # Simple: sweep all idle cash to debt fund
            if self.portfolio.equity_cash > 0:
                self.portfolio.idle_cash_in_debt += self.portfolio.equity_cash
                self.portfolio.equity_cash = 0.0

    # -------------------------------------------------------------------------
    # Technical Indicator Checks
    # -------------------------------------------------------------------------

    def _check_technical_entry(self, symbol: str, as_of_date: datetime) -> bool:
        """
        Check if technical conditions allow entry for a symbol.
        Returns True if all enabled technical filters pass.
        """
        cfg = self.config
        if not cfg.use_technical_filter:
            return True  # No technical filter enabled

        df = self.price_data.get(symbol)
        if df is None or len(df) < 60:
            return False

        # Get data up to as_of_date
        hist = df[df.index <= as_of_date].copy()
        if len(hist) < 30:
            return False

        # EMA check
        if cfg.use_ema_entry:
            hist = add_ema_signals(hist, cfg.ema_fast, cfg.ema_slow)
            latest = hist.iloc[-1]
            if cfg.ema_entry_type == 'crossover':
                if not latest['ema_bullish']:
                    return False
            elif cfg.ema_entry_type == 'price_above':
                if latest['close'] <= latest['ema_slow']:
                    return False
            elif cfg.ema_entry_type == 'both':
                if not latest['ema_bullish'] or latest['close'] <= latest['ema_slow']:
                    return False

        # RSI check
        if cfg.use_rsi_filter:
            hist = add_rsi_signals(hist, cfg.rsi_period)
            rsi_val = hist.iloc[-1]['rsi']
            if rsi_val < cfg.rsi_min_entry or rsi_val > cfg.rsi_max_entry:
                return False

        # Stochastics check
        if cfg.use_stoch_filter:
            hist = add_stoch_signals(hist, cfg.stoch_k, cfg.stoch_d,
                                     cfg.stoch_overbought, cfg.stoch_oversold)
            if hist.iloc[-1]['stoch_overbought']:
                return False  # Don't enter overbought

        # Ichimoku check
        if cfg.use_ichimoku:
            hist = add_ichimoku_signals(hist, cfg.ichimoku_tenkan, cfg.ichimoku_kijun)
            if cfg.require_above_cloud and not hist.iloc[-1]['price_above_cloud']:
                return False

        # Supertrend check
        if cfg.use_supertrend:
            hist = add_supertrend_signals(hist, cfg.supertrend_atr, cfg.supertrend_mult)
            if cfg.supertrend_entry_bullish and not hist.iloc[-1]['supertrend_bullish']:
                return False

        # MACD check
        if cfg.use_macd:
            hist = add_macd_signals(hist, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
            latest = hist.iloc[-1]
            if cfg.require_macd_positive and latest['macd'] <= 0:
                return False
            if cfg.require_macd_above_signal and not latest['macd_bullish']:
                return False

        # ADX check
        if cfg.use_adx:
            hist = add_adx_signals(hist, cfg.adx_period)
            latest = hist.iloc[-1]
            if latest['adx'] < cfg.adx_min_trend:
                return False  # Weak trend
            if cfg.require_plus_di_above and not latest['adx_bullish']:
                return False  # Not in uptrend

        # Bollinger check
        if cfg.use_bollinger:
            hist = add_bollinger_signals(hist, cfg.bb_period, cfg.bb_std)
            latest = hist.iloc[-1]
            if cfg.bb_entry_squeeze and not latest['bb_squeeze']:
                return False  # Only enter during squeeze

        # Weekly filter (approximated using daily data - 5-day grouped)
        if cfg.use_weekly_filter:
            # Resample daily to weekly (last close of each week)
            if len(hist) >= cfg.weekly_ema_period * 5:
                weekly = hist['close'].resample('W').last().dropna()
                if len(weekly) >= cfg.weekly_ema_period:
                    weekly_ema = calc_ema(weekly, cfg.weekly_ema_period)
                    if cfg.require_weekly_above_ema and weekly.iloc[-1] <= weekly_ema.iloc[-1]:
                        return False

        return True

    def _check_technical_exit(self, symbol: str, as_of_date: datetime) -> bool:
        """
        Check if technical conditions signal an exit for a held position.
        Returns True if any enabled exit signal triggers.
        """
        cfg = self.config
        if not cfg.use_technical_filter:
            return False  # No technical filter enabled

        df = self.price_data.get(symbol)
        if df is None or len(df) < 30:
            return False

        hist = df[df.index <= as_of_date].copy()
        if len(hist) < 20:
            return False

        # EMA exit
        if cfg.use_ema_exit:
            hist = add_ema_signals(hist, cfg.ema_fast, cfg.ema_slow)
            latest = hist.iloc[-1]
            if cfg.ema_exit_type == 'crossover' and latest['ema_cross_down']:
                return True
            if cfg.ema_exit_type == 'price_below' and latest['close'] < latest['ema_slow']:
                return True

        # RSI exit
        if cfg.use_rsi_exit:
            hist = add_rsi_signals(hist, cfg.rsi_period)
            if hist.iloc[-1]['rsi'] >= cfg.rsi_exit_overbought:
                return True

        # Ichimoku exit
        if cfg.use_ichimoku and cfg.exit_below_cloud:
            hist = add_ichimoku_signals(hist, cfg.ichimoku_tenkan, cfg.ichimoku_kijun)
            if hist.iloc[-1]['price_below_cloud']:
                return True

        # Supertrend exit
        if cfg.use_supertrend and cfg.supertrend_exit_flip:
            hist = add_supertrend_signals(hist, cfg.supertrend_atr, cfg.supertrend_mult)
            if hist.iloc[-1]['supertrend_flip_down']:
                return True

        # Bollinger exit
        if cfg.use_bollinger and cfg.bb_exit_overbought:
            hist = add_bollinger_signals(hist, cfg.bb_period, cfg.bb_std)
            if hist.iloc[-1]['bb_above_upper']:
                return True

        return False

    def _check_technical_exits(self, current_date: datetime, prices: Dict[str, float]):
        """Check all positions for technical exit signals."""
        if not self.config.use_technical_filter:
            return

        to_exit = []
        for symbol in list(self.portfolio.positions.keys()):
            if self._check_technical_exit(symbol, current_date):
                to_exit.append(symbol)

        for symbol in to_exit:
            price = prices.get(symbol, self.portfolio.positions[symbol].current_price)
            self.portfolio.exit_position(
                symbol, price, current_date, ExitReason.REBALANCE_REPLACE
            )

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
        tech_filter = None
        if self.config.use_technical_filter:
            tech_filter = lambda sym: self._check_technical_entry(sym, current_date)

        candidates = _run_screening(
            self.universe, self.price_data, current_date,
            self.config, exclude_symbols=current_symbols,
            quality_scores=self.quality_scores,
            technical_filter_fn=tech_filter,
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
        """
        Check portfolio stocks for Darvas Box breakouts → topups.

        Uses Darvas Box theory instead of simple consolidation range:
        1. Stock must have made a new N-day high
        2. Box TOP confirmed after 3 days without new high
        3. Box BOTTOM confirmed after 3 days without new low
        4. BREAKOUT: close > box_top with volume confirmation
        """
        cfg = self.config
        detector = DarvasBoxDetector(
            confirmation_days=3,
            new_high_lookback=20,
            volume_multiplier=cfg.breakout_volume_multiplier,
            min_box_height_pct=1.5,
            max_box_height_pct=20.0,
        )

        for symbol in list(self.portfolio.positions.keys()):
            df = self.price_data.get(symbol)
            if df is None:
                continue

            # Only use data up to current_date
            hist = df[df.index <= current_date]
            if len(hist) < 30:
                continue

            # Run Darvas Box detection on last 120 trading days
            recent = hist.tail(120)
            darvas = detector.detect_boxes(recent, symbol)

            # Log box detection
            if darvas.current_box is not None:
                box = darvas.current_box
                self._consolidation_log.append({
                    'symbol': symbol,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'consol_start': box.formation_start.strftime('%Y-%m-%d') if box.formation_start else '',
                    'days': 0,
                    'range_high': round(box.top, 2),
                    'range_low': round(box.bottom, 2),
                    'range_pct': round(box.height_pct, 2),
                    'midpoint': round(box.midpoint, 2),
                    'status': f'darvas_box_{darvas.state.lower()}',
                    'boxes_found': len(darvas.completed_boxes),
                })

            # Check for breakouts that happened on or near current_date
            for bo in darvas.breakouts:
                # Only act on breakouts within the last 5 trading days
                days_since = (current_date - bo.date).days if hasattr(bo.date, 'day') else 999
                if days_since > 7:
                    continue

                self._consolidation_log.append({
                    'symbol': symbol,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'consol_start': bo.box.formation_start.strftime('%Y-%m-%d') if bo.box.formation_start else '',
                    'days': 0,
                    'range_high': round(bo.box.top, 2),
                    'range_low': round(bo.box.bottom, 2),
                    'range_pct': round(bo.box.height_pct, 2),
                    'midpoint': round(bo.box.midpoint, 2),
                    'status': 'breakout_confirmed',
                    'breakout_price': round(bo.price, 2),
                    'breakout_level': round(bo.box.top, 2),
                    'volume': round(bo.volume, 0),
                    'avg_volume': round(bo.avg_volume, 0),
                    'vol_ratio': round(bo.volume_ratio, 2),
                    'stop_loss': round(bo.stop_loss, 2),
                })

                # Execute topup
                self.portfolio.execute_topup(symbol, bo.price, current_date)

    # -------------------------------------------------------------------------
    # Covered Call Overlay
    # -------------------------------------------------------------------------

    @staticmethod
    def _bs_norm_cdf(x):
        """Standard normal CDF using math.erf."""
        return 0.5 * (1 + _erf(x / _sqrt(2)))

    @staticmethod
    def _bs_call_price(S, K, T, r, sigma):
        """Black-Scholes call option price."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(0.0, S - K)
        d1 = (_log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * _sqrt(T))
        d2 = d1 - sigma * _sqrt(T)
        return S * MQBacktestEngine._bs_norm_cdf(d1) - K * _exp(-r * T) * MQBacktestEngine._bs_norm_cdf(d2)

    def _precompute_cc_indicators(self):
        """Pre-compute technical indicators for CC signal checking (once per run)."""
        cfg = self.config
        sig = cfg.cc_signal_type

        for symbol, df in self.price_data.items():
            if len(df) < 60:
                continue

            if sig in ('ema_cross', 'ema_below'):
                fast = calc_ema(df['close'], cfg.cc_signal_fast)
                slow = calc_ema(df['close'], cfg.cc_signal_slow)
                self._cc_indicator_cache[symbol] = pd.DataFrame(
                    {'fast': fast, 'slow': slow}, index=df.index)

            elif sig == 'rsi_ob':
                rsi = calc_rsi(df['close'], cfg.cc_signal_fast)
                self._cc_indicator_cache[symbol] = pd.DataFrame(
                    {'rsi': rsi}, index=df.index)

            elif sig == 'stoch_ob':
                k, d = calc_stochastics(df, cfg.cc_signal_fast, 3)
                self._cc_indicator_cache[symbol] = pd.DataFrame(
                    {'stoch_k': k}, index=df.index)

            elif sig == 'bb_upper':
                _, upper, _ = calc_bollinger_bands(df, cfg.cc_signal_fast, cfg.cc_signal_threshold)
                self._cc_indicator_cache[symbol] = pd.DataFrame(
                    {'bb_upper': upper, 'close': df['close']}, index=df.index)

            elif sig == 'kc_upper':
                _, upper, _ = calc_keltner_channels(df, cfg.cc_signal_fast, 10, cfg.cc_signal_threshold)
                self._cc_indicator_cache[symbol] = pd.DataFrame(
                    {'kc_upper': upper, 'close': df['close']}, index=df.index)

            elif sig == 'ichimoku_bear':
                ichi = calc_ichimoku(df, cfg.cc_signal_fast, cfg.cc_signal_slow)
                self._cc_indicator_cache[symbol] = pd.DataFrame(
                    {'tenkan': ichi['tenkan_sen'], 'kijun': ichi['kijun_sen']}, index=df.index)

            elif sig == 'supertrend_bear':
                _, direction = calc_supertrend(df, cfg.cc_signal_fast, cfg.cc_signal_threshold)
                self._cc_indicator_cache[symbol] = pd.DataFrame(
                    {'direction': direction}, index=df.index)

            elif sig == 'macd_bear':
                macd_line, signal_line, _ = calc_macd(df['close'], cfg.cc_signal_fast, cfg.cc_signal_slow, 9)
                self._cc_indicator_cache[symbol] = pd.DataFrame(
                    {'macd': macd_line, 'signal': signal_line}, index=df.index)

            elif sig == 'adx_low':
                adx, _, _ = calc_adx(df, cfg.cc_signal_fast)
                self._cc_indicator_cache[symbol] = pd.DataFrame(
                    {'adx': adx}, index=df.index)

            # Pre-compute HV for premium calculation (all signal types)
            returns = df['close'].pct_change()
            hv = returns.rolling(cfg.cc_iv_lookback).std() * np.sqrt(252)
            if symbol not in self._cc_indicator_cache:
                self._cc_indicator_cache[symbol] = pd.DataFrame({'hv': hv}, index=df.index)
            else:
                self._cc_indicator_cache[symbol]['hv'] = hv

    def _check_cc_signal(self, symbol: str, current_date: datetime) -> bool:
        """Check if covered call sell signal fires for this stock today."""
        sig = self.config.cc_signal_type
        threshold = self.config.cc_signal_threshold

        if sig == 'always':
            return True

        cache = self._cc_indicator_cache.get(symbol)
        if cache is None:
            return False

        mask = cache.index <= current_date
        n = mask.sum()
        if n < 2:
            return False

        row = cache[mask].iloc[-1]
        prev = cache[mask].iloc[-2]

        if sig == 'ema_cross':
            return prev['fast'] > prev['slow'] and row['fast'] < row['slow']
        elif sig == 'ema_below':
            return row['fast'] < row['slow']
        elif sig == 'rsi_ob':
            return row['rsi'] > threshold
        elif sig == 'stoch_ob':
            return row['stoch_k'] > threshold
        elif sig == 'bb_upper':
            return row['close'] > row['bb_upper']
        elif sig == 'kc_upper':
            return row['close'] > row['kc_upper']
        elif sig == 'ichimoku_bear':
            return prev['tenkan'] > prev['kijun'] and row['tenkan'] < row['kijun']
        elif sig == 'supertrend_bear':
            return row['direction'] == -1 and prev['direction'] == 1
        elif sig == 'macd_bear':
            return prev['macd'] > prev['signal'] and row['macd'] < row['signal']
        elif sig == 'adx_low':
            return row['adx'] < threshold

        return False

    def _get_hv(self, symbol: str, current_date: datetime) -> float:
        """Get historical volatility for a stock on a date."""
        cache = self._cc_indicator_cache.get(symbol)
        if cache is None or 'hv' not in cache.columns:
            return 0.30  # Default 30%
        mask = cache.index <= current_date
        if not mask.any():
            return 0.30
        val = cache.loc[mask, 'hv'].iloc[-1]
        return val if pd.notna(val) and val > 0 else 0.30

    def _manage_active_ccs(self, current_date: datetime, prices: Dict[str, float]):
        """Manage active covered call positions based on cc_mgmt strategy."""
        cfg = self.config
        mgmt = cfg.cc_mgmt
        to_replace = []  # (symbol, new_strike, reason)

        for symbol, cc in list(self._cc_active.items()):
            if symbol not in self.portfolio.positions:
                continue  # Will be handled in MQ exit step

            stock_price = prices.get(symbol, cc['entry_price'])
            strike = cc['strike']
            remaining_days = max(1, (cc['expiry_date'] - current_date).days)
            T = remaining_days / 365.0
            vol = cc.get('vol', 0.30)

            current_cc_value = self._bs_call_price(
                stock_price, strike, T, cfg.cc_risk_free_rate, vol
            ) * cc['shares']

            if mgmt == 'roll_up' or mgmt == 'roll_defend':
                # Roll up: stock approaching strike → buy back, sell higher
                if stock_price >= strike * (1 - cfg.cc_roll_up_trigger):
                    new_strike = stock_price * (1 + cfg.cc_roll_up_distance)
                    to_replace.append((symbol, new_strike, 'roll_up'))

            if mgmt == 'roll_out':
                # Roll out: near expiry and stock near/above strike → roll to next month
                if remaining_days <= cfg.cc_roll_out_days and stock_price >= strike * 0.98:
                    new_strike = max(strike, stock_price * (1 + cfg.cc_strike_otm_pct))
                    to_replace.append((symbol, new_strike, 'roll_out'))

            if mgmt == 'stop_loss':
                # Stop loss: CC value exceeds X * premium → buy back to limit loss
                if current_cc_value > cc['premium'] * cfg.cc_stop_loss_mult:
                    # Buy back and don't re-sell
                    buyback_total = current_cc_value
                    self.portfolio.equity_cash -= buyback_total
                    self._cc_total_buyback += buyback_total
                    cc_pnl = cc['premium'] - buyback_total
                    self._cc_trades.append({
                        'symbol': symbol, 'type': 'stop_loss',
                        'entry_date': cc['sell_date'], 'expiry_date': current_date,
                        'strike': strike, 'premium': cc['premium'],
                        'buyback_cost': buyback_total, 'pnl': cc_pnl,
                    })
                    del self._cc_active[symbol]
                    continue

            if mgmt == 'roll_defend':
                # Double down: stock dropped significantly → sell additional CC at lower strike
                drop_pct = (cc['entry_price'] - stock_price) / cc['entry_price']
                if drop_pct >= cfg.cc_defend_drop_pct and symbol not in [s for s, _, _ in to_replace]:
                    # Don't roll up and defend simultaneously; defend only if not already rolling
                    pass  # Defend is handled by the signal re-firing naturally after expiry

        # Execute rolls: buy back old CC, sell new one
        for symbol, new_strike, reason in to_replace:
            if symbol not in self._cc_active:
                continue
            cc = self._cc_active[symbol]
            stock_price = prices.get(symbol, cc['entry_price'])
            remaining_days = max(1, (cc['expiry_date'] - current_date).days)
            T_old = remaining_days / 365.0
            vol = cc.get('vol', 0.30)
            pos = self.portfolio.positions.get(symbol)
            if pos is None:
                continue

            # Buy back old CC
            buyback_per_share = self._bs_call_price(
                stock_price, cc['strike'], T_old, cfg.cc_risk_free_rate, vol)
            buyback_total = buyback_per_share * cc['shares']
            self.portfolio.equity_cash -= buyback_total
            self._cc_total_buyback += buyback_total

            # Sell new CC at new strike (use same lot-constrained shares)
            T_new = cfg.cc_expiry_days / 365.0
            new_premium_per_share = self._bs_call_price(
                stock_price, new_strike, T_new, cfg.cc_risk_free_rate, vol)
            new_cc_shares = cc['shares']  # Keep same lot-aligned quantity
            new_premium = new_premium_per_share * new_cc_shares
            self.portfolio.equity_cash += new_premium
            self._cc_total_premium += new_premium

            net_roll = new_premium - buyback_total

            self._cc_trades.append({
                'symbol': symbol, 'type': reason,
                'entry_date': cc['sell_date'], 'expiry_date': current_date,
                'strike': cc['strike'], 'new_strike': new_strike,
                'premium': cc['premium'], 'buyback_cost': buyback_total,
                'new_premium': new_premium, 'roll_net': net_roll,
                'pnl': cc['premium'] - buyback_total,
            })

            # Replace active CC
            self._cc_active[symbol] = {
                'strike': round(new_strike, 2),
                'expiry_date': current_date + timedelta(days=cfg.cc_expiry_days),
                'premium': round(new_premium, 2),
                'shares': new_cc_shares,
                'entry_price': stock_price,
                'sell_date': current_date,
                'vol': vol,
                'lot_size': cc.get('lot_size', 1),
            }

    def _process_covered_calls(self, current_date: datetime, prices: Dict[str, float]):
        """Process covered call overlay: check expiries, sell new CCs, handle MQ exits."""
        cfg = self.config

        # --- 1. Check expiries on active CCs ---
        to_close = []
        for symbol, cc in list(self._cc_active.items()):
            if current_date >= cc['expiry_date']:
                stock_price = prices.get(symbol, cc['entry_price'])
                strike = cc['strike']
                premium = cc['premium']

                if stock_price > strike:
                    # ITM → cash settlement (Indian markets: options are cash-settled)
                    # Pay (stock_price - strike) per covered unit, KEEP the stock
                    shares = cc.get('shares', 0)
                    settlement_cost = (stock_price - strike) * shares
                    self.portfolio.equity_cash -= settlement_cost  # PAY settlement
                    self._cc_total_buyback += settlement_cost
                    cc_pnl = premium - settlement_cost
                    self._cc_trades.append({
                        'symbol': symbol, 'type': 'expired_itm',
                        'entry_date': cc['sell_date'], 'expiry_date': current_date,
                        'strike': strike, 'premium': premium,
                        'settlement_cost': settlement_cost,
                        'stock_price_at_expiry': stock_price, 'pnl': cc_pnl,
                    })
                else:
                    # OTM or stock already exited → call expires worthless, keep premium
                    cc_pnl = premium
                    self._cc_trades.append({
                        'symbol': symbol, 'type': 'expired_otm',
                        'entry_date': cc['sell_date'], 'expiry_date': current_date,
                        'strike': strike, 'premium': premium,
                        'stock_price_at_expiry': stock_price, 'pnl': cc_pnl,
                    })
                to_close.append(symbol)

        for symbol in to_close:
            del self._cc_active[symbol]

        # --- 1b. Active CC position management ---
        if cfg.cc_mgmt != 'none':
            self._manage_active_ccs(current_date, prices)

        # --- 2. Handle MQ exits on stocks with active CCs (buy back the call) ---
        for symbol in list(self._cc_active.keys()):
            if symbol not in self.portfolio.positions:
                cc = self._cc_active[symbol]
                stock_price = prices.get(symbol, cc['entry_price'])
                remaining_days = max(1, (cc['expiry_date'] - current_date).days)
                T = remaining_days / 365.0
                vol = cc.get('vol', 0.30)

                buyback_cost = self._bs_call_price(
                    stock_price, cc['strike'], T, cfg.cc_risk_free_rate, vol)
                buyback_total = buyback_cost * cc['shares']

                self.portfolio.equity_cash -= buyback_total
                self._cc_total_buyback += buyback_total

                cc_pnl = cc['premium'] - buyback_total
                self._cc_trades.append({
                    'symbol': symbol, 'type': 'buyback_mq_exit',
                    'entry_date': cc['sell_date'], 'expiry_date': current_date,
                    'strike': cc['strike'], 'premium': cc['premium'],
                    'buyback_cost': buyback_total, 'pnl': cc_pnl,
                })
                del self._cc_active[symbol]

        # --- 3. Sell new CCs on held stocks where signal fires ---
        for symbol, pos in self.portfolio.positions.items():
            if symbol in self._cc_active:
                continue  # Already have active CC on this stock

            if not self._check_cc_signal(symbol, current_date):
                continue

            stock_price = prices.get(symbol, pos.current_price)
            if stock_price <= 0:
                continue

            # Indian F&O lot sizes: NSE targets ~Rs.5-7.5L per lot (2023-2025)
            lot_size = max(1, round(500_000 / stock_price))
            cc_shares = (pos.total_shares // lot_size) * lot_size
            if cc_shares <= 0:
                continue  # Position too small for even 1 lot

            strike = stock_price * (1 + cfg.cc_strike_otm_pct)
            T = cfg.cc_expiry_days / 365.0
            vol = self._get_hv(symbol, current_date)

            premium_per_share = self._bs_call_price(
                stock_price, strike, T, cfg.cc_risk_free_rate, vol)
            total_premium = premium_per_share * cc_shares

            if total_premium <= 0:
                continue

            # Receive premium into equity cash
            self.portfolio.equity_cash += total_premium
            self._cc_total_premium += total_premium

            expiry_date = current_date + timedelta(days=cfg.cc_expiry_days)

            self._cc_active[symbol] = {
                'strike': round(strike, 2),
                'expiry_date': expiry_date,
                'premium': round(total_premium, 2),
                'shares': cc_shares,
                'entry_price': stock_price,
                'sell_date': current_date,
                'vol': vol,
                'lot_size': lot_size,
            }

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
            consolidation_log=self._consolidation_log,
            topup_sl_log=self._topup_sl_log,
            # Covered call overlay
            cc_trades=self._cc_trades,
            cc_total_premium=round(self._cc_total_premium, 2),
            cc_total_buyback=round(self._cc_total_buyback, 2),
            cc_net_income=round(self._cc_total_premium - self._cc_total_buyback, 2),
            cc_total_calls_sold=len(self._cc_trades),
            cc_expired_otm=sum(1 for t in self._cc_trades if t.get('type') == 'expired_otm'),
            cc_expired_itm=sum(1 for t in self._cc_trades if t.get('type') == 'expired_itm'),
            cc_buybacks=sum(1 for t in self._cc_trades if t.get('type') == 'buyback_mq_exit'),
            cc_rolls=sum(1 for t in self._cc_trades if t.get('type') in ('roll_up', 'roll_out')),
            cc_stop_losses=sum(1 for t in self._cc_trades if t.get('type') == 'stop_loss'),
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
