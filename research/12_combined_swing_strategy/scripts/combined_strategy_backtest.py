"""
Combined Multi-Strategy Portfolio Backtest
==========================================
Combines 3 uncorrelated swing trading edges into a single portfolio:

1. TREND: EMA(20/50) crossover + ADX>25 on 60-min data (long only)
2. MEAN REVERSION: RSI(21)<30 + SMA(200) filter, exit RSI>50 on 60-min (long only)
3. BREAKOUT: NR7 (Narrow Range 7-day) breakout on daily data, hold 3 days (long only)

Capital: Rs 1 Crore, allocated across strategies with per-position sizing.
Costs: 0.05% slippage per side + Rs 40 brokerage per round trip.
Period: 2018-01-01 to 2025-11-07 (~7.8 years)
"""

import sqlite3
import pandas as pd
import numpy as np
import csv
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_strategy_results.csv')

START_DATE = '2018-01-01'
END_DATE = '2025-11-07'
INITIAL_CAPITAL = 10_000_000  # Rs 1 Crore
SLIPPAGE_PCT = 0.0005  # 0.05% per side
BROKERAGE_PER_RT = 40  # Rs per round trip

# FNO stocks (80 liquid stocks)
FNO_STOCKS = [
    'RELIANCE','TCS','HDFCBANK','INFY','ICICIBANK','HINDUNILVR','ITC','SBIN',
    'BHARTIARTL','KOTAKBANK','LT','AXISBANK','ASIANPAINT','MARUTI','TITAN',
    'BAJFINANCE','HCLTECH','SUNPHARMA','ULTRACEMCO','NESTLEIND','WIPRO','ONGC',
    'NTPC','POWERGRID','M&M','TATAMOTORS','TECHM','JSWSTEEL','INDUSINDBK',
    'ADANIPORTS','BAJAJFINSV','HINDALCO','COALINDIA','DIVISLAB','GRASIM',
    'TATACONSUM','DRREDDY','CIPLA','EICHERMOT','BRITANNIA','APOLLOHOSP',
    'HEROMOTOCO','SBILIFE','BPCL','TATASTEEL','BAJAJ-AUTO','HDFCLIFE','SHREECEM',
    'ADANIENT','VEDL','TATAPOWER','GAIL','JINDALSTEL','DLF','GODREJPROP',
    'SIEMENS','HAVELLS','PIDILITIND','DABUR','MARICO','COLPAL','MUTHOOTFIN',
    'CHOLAFIN','BEL','HAL','IOC','IRCTC','COFORGE','PERSISTENT','MCX',
    'CUMMINSIND','VOLTAS','AMBUJACEM','TRENT','ZOMATO','PAYTM','DELHIVERY',
    'PNB','BANKBARODA','FEDERALBNK','IDFCFIRSTB'
]


@dataclass
class StrategyConfig:
    """Configuration for the combined strategy backtest."""
    # Capital allocation per strategy (must sum to 1.0)
    trend_alloc: float = 0.50      # 50% to trend following
    meanrev_alloc: float = 0.30    # 30% to mean reversion
    breakout_alloc: float = 0.20   # 20% to breakout

    # Position sizing
    max_positions_trend: int = 10
    max_positions_meanrev: int = 8
    max_positions_breakout: int = 10
    max_position_pct: float = 0.05  # max 5% of total capital per position

    # Trend strategy params
    ema_fast: int = 20
    ema_slow: int = 50
    adx_threshold: int = 25
    adx_period: int = 14

    # Mean reversion params
    rsi_period: int = 21
    rsi_entry: float = 30.0
    rsi_exit: float = 50.0
    sma_period: int = 200
    meanrev_max_hold: int = 60  # max hold in 60-min bars (~10 trading days)

    # Breakout params
    nr_days: int = 7             # NR7
    breakout_hold_days: int = 3  # hold for 3 trading days

    # Risk
    hard_stop_pct: float = 0.05  # 5% stop loss per position


@dataclass
class Position:
    symbol: str
    strategy: str  # 'trend', 'meanrev', 'breakout'
    entry_date: datetime
    entry_price: float
    shares: int
    capital_used: float
    stop_price: float


@dataclass
class Trade:
    symbol: str
    strategy: str
    direction: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    hold_days: int


# ── Indicators (vectorized) ────────────────────────────────────────────────

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_sma(series, period):
    return series.rolling(period).mean()

def compute_rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx


# ── Signal Generation ──────────────────────────────────────────────────────

def generate_trend_signals(df_60min, config):
    """
    EMA crossover + ADX filter on 60-min data.
    Returns DataFrame with 'signal' column: 1=buy, -1=sell, 0=hold.
    """
    df = df_60min.copy()
    df['ema_fast'] = compute_ema(df['close'], config.ema_fast)
    df['ema_slow'] = compute_ema(df['close'], config.ema_slow)
    df['adx'] = compute_adx(df['high'], df['low'], df['close'], config.adx_period)

    cross_up = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
    cross_dn = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))

    df['signal'] = 0
    df.loc[cross_up & (df['adx'] >= config.adx_threshold), 'signal'] = 1
    df.loc[cross_dn, 'signal'] = -1

    return df[['open', 'high', 'low', 'close', 'volume', 'signal']].copy()


def generate_meanrev_signals(df_60min, config):
    """
    RSI oversold + SMA(200) trend filter on 60-min data.
    Returns DataFrame with 'entry_signal' and 'exit_signal' columns.
    """
    df = df_60min.copy()
    df['rsi'] = compute_rsi(df['close'], config.rsi_period)
    df['sma200'] = compute_sma(df['close'], config.sma_period)

    df['entry_signal'] = (df['rsi'] < config.rsi_entry) & (df['close'] > df['sma200'])
    df['exit_signal'] = df['rsi'] > config.rsi_exit

    return df[['open', 'high', 'low', 'close', 'volume', 'entry_signal', 'exit_signal']].copy()


def generate_breakout_signals(df_daily, config):
    """
    NR7 breakout on daily data.
    Entry: today's range is narrowest in last N days, buy on close.
    Exit: hold for N trading days.
    Returns DataFrame with 'entry_signal' column.
    """
    df = df_daily.copy()
    df['range'] = df['high'] - df['low']
    df['min_range_n'] = df['range'].rolling(config.nr_days).min()
    # NR7: today's range equals the N-day minimum range
    df['entry_signal'] = (df['range'] <= df['min_range_n']) & (df['range'] > 0)

    return df[['open', 'high', 'low', 'close', 'volume', 'range', 'entry_signal']].copy()


# ── Data Loading ────────────────────────────────────────────────────────────

def load_data():
    """Load 60-min and daily data for all FNO stocks."""
    print("Loading data from DB...", flush=True)
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)

    placeholders = ','.join(['?' for _ in FNO_STOCKS])

    # 60-min data
    df_60 = pd.read_sql_query(
        f"SELECT symbol, date, open, high, low, close, volume "
        f"FROM market_data_unified WHERE timeframe='60minute' "
        f"AND symbol IN ({placeholders}) AND date >= ? AND date <= ? "
        f"ORDER BY symbol, date",
        conn, params=FNO_STOCKS + [START_DATE, END_DATE]
    )
    df_60['date'] = pd.to_datetime(df_60['date'])

    # Daily data
    df_daily = pd.read_sql_query(
        f"SELECT symbol, date, open, high, low, close, volume "
        f"FROM market_data_unified WHERE timeframe='day' "
        f"AND symbol IN ({placeholders}) AND date >= ? AND date <= ? "
        f"ORDER BY symbol, date",
        conn, params=FNO_STOCKS + [START_DATE, END_DATE]
    )
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    conn.close()

    # Split into per-symbol dicts
    data_60 = {}
    for sym, grp in df_60.groupby('symbol'):
        sdf = grp.set_index('date').sort_index()
        sdf = sdf[~sdf.index.duplicated(keep='first')]
        sdf = sdf[['open', 'high', 'low', 'close', 'volume']].astype(float)
        if len(sdf) >= 500:
            data_60[sym] = sdf

    data_daily = {}
    for sym, grp in df_daily.groupby('symbol'):
        sdf = grp.set_index('date').sort_index()
        sdf = sdf[~sdf.index.duplicated(keep='first')]
        sdf = sdf[['open', 'high', 'low', 'close', 'volume']].astype(float)
        if len(sdf) >= 200:
            data_daily[sym] = sdf

    elapsed = time.time() - t0
    print(f"  60-min: {len(data_60)} stocks, {len(df_60):,} rows", flush=True)
    print(f"  Daily:  {len(data_daily)} stocks, {len(df_daily):,} rows", flush=True)
    print(f"  Loaded in {elapsed:.1f}s", flush=True)

    return data_60, data_daily


# ── Pre-compute All Signals ────────────────────────────────────────────────

def precompute_all_signals(data_60, data_daily, config):
    """
    Pre-compute signals for all strategies and all stocks.
    Returns dict of signal events sorted by date for portfolio-level simulation.
    """
    print("\nPre-computing signals...", flush=True)
    t0 = time.time()

    # Collect all signal events as a unified timeline
    events = []  # list of (date, symbol, event_type, price, data)

    # ── TREND signals (60-min) ──
    trend_count = 0
    for sym, df in data_60.items():
        sig_df = generate_trend_signals(df, config)
        buys = sig_df[sig_df['signal'] == 1]
        sells = sig_df[sig_df['signal'] == -1]

        for dt, row in buys.iterrows():
            events.append((dt, sym, 'trend_buy', row['close'], {}))
            trend_count += 1
        for dt, row in sells.iterrows():
            events.append((dt, sym, 'trend_sell', row['close'], {}))

    # ── MEAN REVERSION signals (60-min) ──
    meanrev_count = 0
    for sym, df in data_60.items():
        sig_df = generate_meanrev_signals(df, config)
        entries = sig_df[sig_df['entry_signal']]
        exits = sig_df[sig_df['exit_signal']]

        for dt, row in entries.iterrows():
            events.append((dt, sym, 'meanrev_entry', row['close'], {}))
            meanrev_count += 1
        for dt, row in exits.iterrows():
            events.append((dt, sym, 'meanrev_exit', row['close'], {}))

    # ── BREAKOUT signals (daily) ──
    breakout_count = 0
    for sym, df in data_daily.items():
        sig_df = generate_breakout_signals(df, config)
        entries = sig_df[sig_df['entry_signal']]

        for dt, row in entries.iterrows():
            events.append((dt, sym, 'breakout_entry', row['close'], {}))
            breakout_count += 1

    events.sort(key=lambda x: x[0])
    elapsed = time.time() - t0
    print(f"  Trend signals: {trend_count} buys", flush=True)
    print(f"  MeanRev signals: {meanrev_count} entries", flush=True)
    print(f"  Breakout signals: {breakout_count} entries", flush=True)
    print(f"  Total events: {len(events):,} in {elapsed:.1f}s", flush=True)

    return events


# ── Portfolio Simulator ────────────────────────────────────────────────────

def run_portfolio_backtest(events, data_60, data_daily, config):
    """
    Simulate portfolio across all strategies with proper capital management.
    """
    print("\nRunning portfolio simulation...", flush=True)
    t0 = time.time()

    capital = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}  # key = f"{strategy}_{symbol}"
    trades: List[Trade] = []
    equity_curve = {}

    # Capital allocated per strategy
    trend_capital = capital * config.trend_alloc
    meanrev_capital = capital * config.meanrev_alloc
    breakout_capital = capital * config.breakout_alloc

    # Track strategy-level invested amounts
    trend_invested = 0.0
    meanrev_invested = 0.0
    breakout_invested = 0.0

    # Build daily close lookup for breakout exit and stop checks
    daily_close = {}
    for sym, df in data_daily.items():
        for dt, row in df.iterrows():
            daily_close[(sym, dt.date())] = row['close']

    # Build 60-min close lookup for stop checks
    # (too large to store all — we'll check stops at event time)

    def count_positions(strategy):
        return sum(1 for k, p in positions.items() if p.strategy == strategy)

    def get_invested(strategy):
        return sum(p.capital_used for k, p in positions.items() if p.strategy == strategy)

    def close_position(key, exit_price_raw, exit_date, exit_reason):
        nonlocal cash, trend_invested, meanrev_invested, breakout_invested
        pos = positions.pop(key)

        # Apply slippage on exit
        exit_price = exit_price_raw * (1 - SLIPPAGE_PCT)
        pnl = (exit_price - pos.entry_price) * pos.shares - BROKERAGE_PER_RT
        pnl_pct = (exit_price / pos.entry_price - 1) * 100

        cash += pos.capital_used + pnl
        if pos.strategy == 'trend':
            trend_invested -= pos.capital_used
        elif pos.strategy == 'meanrev':
            meanrev_invested -= pos.capital_used
        else:
            breakout_invested -= pos.capital_used

        hold_days = (exit_date - pos.entry_date).days
        trades.append(Trade(
            symbol=pos.symbol, strategy=pos.strategy, direction='long',
            entry_date=pos.entry_date, exit_date=exit_date,
            entry_price=pos.entry_price, exit_price=exit_price,
            shares=pos.shares, pnl=pnl, pnl_pct=pnl_pct,
            exit_reason=exit_reason, hold_days=hold_days
        ))

    def open_position(symbol, strategy, price_raw, date, strategy_capital, max_pos):
        nonlocal cash, trend_invested, meanrev_invested, breakout_invested

        key = f"{strategy}_{symbol}"
        if key in positions:
            return  # already in this position

        if count_positions(strategy) >= max_pos:
            return  # max positions reached

        # Position sizing: equal weight within strategy allocation
        invested = get_invested(strategy)
        available = strategy_capital - invested
        pos_size = min(
            strategy_capital / max_pos,  # equal split
            capital * config.max_position_pct,  # max per position
            available,  # remaining allocation
            cash  # available cash
        )

        if pos_size < 10000:  # minimum Rs 10K per position
            return

        # Apply slippage on entry
        entry_price = price_raw * (1 + SLIPPAGE_PCT)
        shares = int(pos_size / entry_price)
        if shares < 1:
            return

        actual_capital = shares * entry_price
        stop_price = entry_price * (1 - config.hard_stop_pct)

        cash -= actual_capital
        if strategy == 'trend':
            trend_invested += actual_capital
        elif strategy == 'meanrev':
            meanrev_invested += actual_capital
        else:
            breakout_invested += actual_capital

        positions[key] = Position(
            symbol=symbol, strategy=strategy, entry_date=date,
            entry_price=entry_price, shares=shares,
            capital_used=actual_capital, stop_price=stop_price
        )

    # Track last equity curve date to avoid redundant entries
    last_eq_date = None

    # Process events chronologically
    pending_breakout_exits = {}  # key -> exit_date

    for i, (dt, sym, event_type, price, data) in enumerate(events):
        current_date = dt.date() if hasattr(dt, 'date') else dt

        # Check breakout exits (daily resolution)
        keys_to_close = []
        for key, exit_date in list(pending_breakout_exits.items()):
            if current_date >= exit_date and key in positions:
                pos = positions[key]
                # Get exit price from daily data
                exit_price = daily_close.get((pos.symbol, exit_date))
                if exit_price is None:
                    # Find next available close
                    for d_offset in range(5):
                        check_date = exit_date + timedelta(days=d_offset)
                        exit_price = daily_close.get((pos.symbol, check_date))
                        if exit_price:
                            break
                if exit_price:
                    close_position(key, exit_price, datetime.combine(exit_date, datetime.min.time()), 'time_exit')
                    keys_to_close.append(key)
        for key in keys_to_close:
            pending_breakout_exits.pop(key, None)

        # Check stop losses for all positions
        for key in list(positions.keys()):
            pos = positions[key]
            if price <= pos.stop_price and sym == pos.symbol:
                close_position(key, price, dt, 'stop_loss')
                pending_breakout_exits.pop(key, None)

        # Process signal
        if event_type == 'trend_buy':
            open_position(sym, 'trend', price, dt, trend_capital, config.max_positions_trend)

        elif event_type == 'trend_sell':
            key = f"trend_{sym}"
            if key in positions:
                close_position(key, price, dt, 'signal_exit')

        elif event_type == 'meanrev_entry':
            key = f"meanrev_{sym}"
            if key not in positions:
                open_position(sym, 'meanrev', price, dt, meanrev_capital, config.max_positions_meanrev)

        elif event_type == 'meanrev_exit':
            key = f"meanrev_{sym}"
            if key in positions:
                close_position(key, price, dt, 'rsi_exit')

        elif event_type == 'breakout_entry':
            key = f"breakout_{sym}"
            if key not in positions:
                open_position(sym, 'breakout', price, dt, breakout_capital, config.max_positions_breakout)
                if key in positions:
                    # Schedule exit after N trading days
                    exit_date = current_date + timedelta(days=config.breakout_hold_days + 2)  # +2 for weekends
                    pending_breakout_exits[key] = exit_date

        # Record equity curve (once per trading day)
        if current_date != last_eq_date:
            # Mark-to-market all positions
            mtm = cash
            for key, pos in positions.items():
                # Use latest known price for position
                latest_price = daily_close.get((pos.symbol, current_date))
                if latest_price:
                    mtm += latest_price * pos.shares
                else:
                    mtm += pos.capital_used  # fallback to entry value
            equity_curve[current_date] = mtm
            last_eq_date = current_date

    # Close any remaining positions at last known price
    for key in list(positions.keys()):
        pos = positions[key]
        last_dates = sorted(daily_close.keys())
        last_price = None
        for sym_dt in reversed(last_dates):
            if sym_dt[0] == pos.symbol:
                last_price = sym_dt
                break
        if last_price:
            close_position(key, daily_close[last_price],
                          datetime.combine(last_price[1], datetime.min.time()), 'end_of_backtest')
        pending_breakout_exits.pop(key, None)

    elapsed = time.time() - t0
    print(f"  Simulation complete in {elapsed:.1f}s", flush=True)
    print(f"  Total trades: {len(trades)}", flush=True)

    return trades, equity_curve


# ── Metrics Calculation ────────────────────────────────────────────────────

def compute_portfolio_metrics(trades, equity_curve):
    """Compute portfolio-level and per-strategy metrics."""
    if not trades:
        print("No trades generated!")
        return None

    # Sort equity curve
    dates = sorted(equity_curve.keys())
    values = [equity_curve[d] for d in dates]

    # CAGR
    years = (dates[-1] - dates[0]).days / 365.25
    final_value = values[-1]
    cagr = (final_value / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 else 0

    # Max Drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    # Sharpe (daily returns)
    returns = []
    for i in range(1, len(values)):
        returns.append(values[i] / values[i-1] - 1)
    returns = np.array(returns)
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Sortino
    neg_returns = returns[returns < 0]
    downside_std = neg_returns.std() if len(neg_returns) > 0 else 0.001
    sortino = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Calmar
    calmar = cagr / max_dd if max_dd > 0 else 0

    # Trade-level metrics
    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p <= 0]
    win_rate = len(wins) / len(pnl_pcts) * 100 if pnl_pcts else 0
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0

    print(f"\n{'='*80}")
    print(f"COMBINED PORTFOLIO RESULTS ({START_DATE} to {END_DATE})")
    print(f"{'='*80}")
    print(f"  Initial Capital:  Rs {INITIAL_CAPITAL:>15,}")
    print(f"  Final Value:      Rs {final_value:>15,.0f}")
    print(f"  Total Return:     {(final_value/INITIAL_CAPITAL - 1)*100:>14.2f}%")
    print(f"  CAGR:             {cagr*100:>14.2f}%")
    print(f"  Sharpe Ratio:     {sharpe:>14.2f}")
    print(f"  Sortino Ratio:    {sortino:>14.2f}")
    print(f"  Max Drawdown:     {max_dd*100:>14.2f}%")
    print(f"  Calmar Ratio:     {calmar:>14.2f}")
    print(f"  Profit Factor:    {profit_factor:>14.2f}")
    print(f"  Total Trades:     {len(trades):>14,}")
    print(f"  Win Rate:         {win_rate:>14.1f}%")
    print(f"  Avg Win:          {np.mean(wins):>14.2f}%" if wins else "  Avg Win:          N/A")
    print(f"  Avg Loss:         {np.mean(losses):>14.2f}%" if losses else "  Avg Loss:         N/A")
    print(f"  Total PnL:        Rs {sum(pnls):>15,.0f}")

    # Per-strategy breakdown
    print(f"\n{'-'*80}")
    print(f"PER-STRATEGY BREAKDOWN")
    print(f"{'-'*80}")
    for strat in ['trend', 'meanrev', 'breakout']:
        strat_trades = [t for t in trades if t.strategy == strat]
        if not strat_trades:
            print(f"  {strat.upper():12s} | No trades")
            continue
        s_pnls = [t.pnl_pct for t in strat_trades]
        s_wins = [p for p in s_pnls if p > 0]
        s_losses = [p for p in s_pnls if p <= 0]
        s_wr = len(s_wins) / len(s_pnls) * 100
        s_gp = sum(t.pnl for t in strat_trades if t.pnl > 0)
        s_gl = abs(sum(t.pnl for t in strat_trades if t.pnl <= 0))
        s_pf = s_gp / s_gl if s_gl > 0 else 99.0
        s_total_pnl = sum(t.pnl for t in strat_trades)
        s_avg_hold = np.mean([t.hold_days for t in strat_trades])

        print(f"  {strat.upper():12s} | Trades: {len(strat_trades):5d} | "
              f"WR: {s_wr:5.1f}% | PF: {s_pf:5.2f} | "
              f"PnL: Rs {s_total_pnl:>12,.0f} | AvgHold: {s_avg_hold:.1f}d")

    # Exit reason breakdown
    print(f"\n{'-'*80}")
    print(f"EXIT REASONS")
    print(f"{'-'*80}")
    exit_reasons = {}
    for t in trades:
        reason = t.exit_reason
        if reason not in exit_reasons:
            exit_reasons[reason] = {'count': 0, 'pnl': 0}
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['pnl'] += t.pnl
    for reason, stats in sorted(exit_reasons.items(), key=lambda x: -x[1]['count']):
        print(f"  {reason:20s} | {stats['count']:5d} trades | PnL: Rs {stats['pnl']:>12,.0f}")

    # Yearly breakdown
    print(f"\n{'-'*80}")
    print(f"YEARLY BREAKDOWN")
    print(f"{'-'*80}")
    yearly = {}
    for t in trades:
        yr = t.entry_date.year if hasattr(t.entry_date, 'year') else pd.Timestamp(t.entry_date).year
        if yr not in yearly:
            yearly[yr] = {'trades': 0, 'pnl': 0, 'wins': 0}
        yearly[yr]['trades'] += 1
        yearly[yr]['pnl'] += t.pnl
        if t.pnl > 0:
            yearly[yr]['wins'] += 1
    for yr in sorted(yearly.keys()):
        y = yearly[yr]
        wr = y['wins'] / y['trades'] * 100 if y['trades'] > 0 else 0
        print(f"  {yr} | Trades: {y['trades']:5d} | WR: {wr:5.1f}% | PnL: Rs {y['pnl']:>12,.0f}")

    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 2),
        'profit_factor': round(profit_factor, 2),
        'total_trades': len(trades),
        'win_rate': round(win_rate, 1),
        'total_pnl': sum(pnls),
        'final_value': final_value,
        'total_return_pct': round((final_value / INITIAL_CAPITAL - 1) * 100, 2),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def run_config(config, label, data_60, data_daily):
    """Run a single configuration and return metrics."""
    print(f"\n{'#'*80}")
    print(f"# CONFIG: {label}")
    print(f"{'#'*80}")
    print(f"  Allocation: Trend={config.trend_alloc:.0%} MeanRev={config.meanrev_alloc:.0%} "
          f"Breakout={config.breakout_alloc:.0%}")
    print(f"  Max positions: Trend={config.max_positions_trend} MeanRev={config.max_positions_meanrev} "
          f"Breakout={config.max_positions_breakout}")

    events = precompute_all_signals(data_60, data_daily, config)
    trades, equity_curve = run_portfolio_backtest(events, data_60, data_daily, config)
    metrics = compute_portfolio_metrics(trades, equity_curve)

    return metrics, trades, equity_curve


if __name__ == '__main__':
    data_60, data_daily = load_data()

    # ── Baseline config ──
    config = StrategyConfig()
    metrics, trades, eq_curve = run_config(config, "BASELINE", data_60, data_daily)

    if metrics:
        # Save to CSV
        RESULT_FIELDS = ['label', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'calmar',
                         'profit_factor', 'total_trades', 'win_rate', 'total_pnl',
                         'final_value', 'total_return_pct',
                         'trend_alloc', 'meanrev_alloc', 'breakout_alloc',
                         'max_pos_trend', 'max_pos_meanrev', 'max_pos_breakout']

        row = {
            'label': 'BASELINE',
            **metrics,
            'trend_alloc': config.trend_alloc,
            'meanrev_alloc': config.meanrev_alloc,
            'breakout_alloc': config.breakout_alloc,
            'max_pos_trend': config.max_positions_trend,
            'max_pos_meanrev': config.max_positions_meanrev,
            'max_pos_breakout': config.max_positions_breakout,
        }

        with open(OUTPUT_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            w.writeheader()
            w.writerow(row)

        print(f"\nResults saved to {OUTPUT_CSV}")

    print("\nDone.")
