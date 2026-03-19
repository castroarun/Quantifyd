"""
Short Adjusted Sweep: Test if shorts become profitable with:
1. Segregated capital — shorts only use idle capital (capital not deployed in longs)
2. Capital utilization gate — shorts only fire when long utilization < 50%
3. Higher conviction short thresholds — RSI 75/80/85 for MR shorts
4. Tighter short stop losses — 3%/5%/7% (vs 10% baseline)
5. Stricter ADX for trend shorts — ADX 30/35 (vs 20 for longs)

Base config: E25_60_ADX20_R14_30_P20_10 (best long-only from Phase 3)
"""

import csv
import os
import sys
import time
import logging
import numpy as np
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.insert(0, PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'short_adjusted_sweep.csv')

START_DATE = '2018-01-01'
END_DATE = '2025-11-07'
INITIAL_CAPITAL = 10_000_000
SLIPPAGE_PCT = 0.0005
BROKERAGE_PER_RT = 40

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

FIELDNAMES = [
    'label', 'mode', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'calmar',
    'profit_factor', 'total_trades', 'win_rate', 'total_pnl',
    'final_value', 'total_return_pct',
    'trend_long_trades', 'trend_long_pf', 'trend_long_pnl',
    'trend_short_trades', 'trend_short_pf', 'trend_short_pnl',
    'mr_long_trades', 'mr_long_pf', 'mr_long_pnl',
    'mr_short_trades', 'mr_short_pf', 'mr_short_pnl',
    'short_sl', 'rsi_short_entry', 'short_adx_threshold',
    'capital_util_gate', 'shorts_blocked_by_gate',
]


@dataclass
class Config:
    ema_fast: int = 25
    ema_slow: int = 60
    adx_threshold: int = 20       # for longs
    adx_period: int = 14
    rsi_period: int = 14
    rsi_entry: float = 30.0       # long: buy when RSI < this
    rsi_exit: float = 50.0        # long: sell when RSI > this
    rsi_short_entry: float = 70.0 # short: sell when RSI > this
    rsi_short_exit: float = 50.0  # short: cover when RSI < this
    sma_period: int = 200
    max_positions_trend: int = 20
    max_positions_meanrev: int = 10
    max_position_pct: float = 0.05
    hard_stop_pct: float = 0.10   # long stop loss
    short_stop_pct: float = 0.10  # separate short stop loss
    trend_alloc: float = 0.60
    meanrev_alloc: float = 0.40
    enable_trend_short: bool = False
    enable_mr_short: bool = False
    short_adx_threshold: int = 20  # ADX threshold specifically for trend shorts
    capital_util_gate: float = 0.50  # shorts only when long util < this


@dataclass
class Position:
    symbol: str
    strategy: str
    direction: str
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


# ── Indicators ──

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


# ── Signal Generation ──

def generate_trend_signals(df_60min, config):
    """EMA crossover + ADX. Separate ADX thresholds for long vs short."""
    df = df_60min.copy()
    df['ema_fast'] = compute_ema(df['close'], config.ema_fast)
    df['ema_slow'] = compute_ema(df['close'], config.ema_slow)
    df['adx'] = compute_adx(df['high'], df['low'], df['close'], config.adx_period)

    cross_up = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
    cross_dn = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
    strong_trend_long = df['adx'] >= config.adx_threshold
    strong_trend_short = df['adx'] >= config.short_adx_threshold

    df['signal'] = 0
    df.loc[cross_up & strong_trend_long, 'signal'] = 1
    df.loc[cross_dn, 'signal'] = -1

    df['short_signal'] = 0
    if config.enable_trend_short:
        df.loc[cross_dn & strong_trend_short, 'short_signal'] = -1
        df.loc[cross_up, 'short_signal'] = 1

    return df[['open', 'high', 'low', 'close', 'volume', 'signal', 'short_signal']].copy()


def generate_meanrev_signals(df_60min, config):
    """RSI mean reversion with configurable short RSI threshold."""
    df = df_60min.copy()
    df['rsi'] = compute_rsi(df['close'], config.rsi_period)
    df['sma200'] = compute_sma(df['close'], config.sma_period)

    df['entry_signal'] = (df['rsi'] < config.rsi_entry) & (df['close'] > df['sma200'])
    df['exit_signal'] = df['rsi'] > config.rsi_exit

    df['short_entry_signal'] = False
    df['short_exit_signal'] = False
    if config.enable_mr_short:
        df['short_entry_signal'] = (df['rsi'] > config.rsi_short_entry) & (df['close'] < df['sma200'])
        df['short_exit_signal'] = df['rsi'] < config.rsi_short_exit

    return df[['open', 'high', 'low', 'close', 'volume',
               'entry_signal', 'exit_signal',
               'short_entry_signal', 'short_exit_signal']].copy()


# ── Data Loading ──

def load_data():
    print("Loading data from DB...", flush=True)
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    placeholders = ','.join(['?' for _ in FNO_STOCKS])

    df_60 = pd.read_sql_query(
        f"SELECT symbol, date, open, high, low, close, volume "
        f"FROM market_data_unified WHERE timeframe='60minute' "
        f"AND symbol IN ({placeholders}) AND date >= ? AND date <= ? "
        f"ORDER BY symbol, date",
        conn, params=FNO_STOCKS + [START_DATE, END_DATE]
    )
    df_60['date'] = pd.to_datetime(df_60['date'])

    df_daily = pd.read_sql_query(
        f"SELECT symbol, date, open, high, low, close, volume "
        f"FROM market_data_unified WHERE timeframe='day' "
        f"AND symbol IN ({placeholders}) AND date >= ? AND date <= ? "
        f"ORDER BY symbol, date",
        conn, params=FNO_STOCKS + [START_DATE, END_DATE]
    )
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    conn.close()

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

    print(f"  60-min: {len(data_60)} stocks | Daily: {len(data_daily)} stocks | {time.time()-t0:.1f}s", flush=True)
    return data_60, data_daily


# ── Precompute Signals ──

def precompute_all_signals(data_60, data_daily, config):
    events = []

    for sym, df in data_60.items():
        sig_df = generate_trend_signals(df, config)
        for dt, row in sig_df[sig_df['signal'] == 1].iterrows():
            events.append((dt, sym, 'trend_long_buy', row['close']))
        for dt, row in sig_df[sig_df['signal'] == -1].iterrows():
            events.append((dt, sym, 'trend_long_sell', row['close']))
        if config.enable_trend_short:
            for dt, row in sig_df[sig_df['short_signal'] == -1].iterrows():
                events.append((dt, sym, 'trend_short_enter', row['close']))
            for dt, row in sig_df[sig_df['short_signal'] == 1].iterrows():
                events.append((dt, sym, 'trend_short_cover', row['close']))

    for sym, df in data_60.items():
        sig_df = generate_meanrev_signals(df, config)
        for dt, row in sig_df[sig_df['entry_signal']].iterrows():
            events.append((dt, sym, 'mr_long_entry', row['close']))
        for dt, row in sig_df[sig_df['exit_signal']].iterrows():
            events.append((dt, sym, 'mr_long_exit', row['close']))
        if config.enable_mr_short:
            for dt, row in sig_df[sig_df['short_entry_signal']].iterrows():
                events.append((dt, sym, 'mr_short_enter', row['close']))
            for dt, row in sig_df[sig_df['short_exit_signal']].iterrows():
                events.append((dt, sym, 'mr_short_cover', row['close']))

    events.sort(key=lambda x: x[0])
    return events


# ── Portfolio Simulation (Modified: segregated capital + utilization gate) ──

def run_portfolio_backtest(events, data_60, data_daily, config):
    capital = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []
    equity_curve = {}
    shorts_blocked_by_gate = 0

    trend_capital = capital * config.trend_alloc
    meanrev_capital = capital * config.meanrev_alloc

    daily_close = {}
    for sym, df in data_daily.items():
        for dt, row in df.iterrows():
            daily_close[(sym, dt.date())] = row['close']

    def count_positions(strategy, direction=None):
        return sum(1 for k, p in positions.items()
                   if p.strategy == strategy and (direction is None or p.direction == direction))

    def get_invested(strategy=None, direction=None):
        return sum(p.capital_used for k, p in positions.items()
                   if (strategy is None or p.strategy == strategy)
                   and (direction is None or p.direction == direction))

    def get_long_capital_utilization():
        """Calculate what fraction of total capital is deployed in longs."""
        long_invested = get_invested(direction='long')
        return long_invested / capital if capital > 0 else 1.0

    def get_idle_capital():
        """Capital not deployed in longs, available for shorts."""
        long_invested = get_invested(direction='long')
        return max(0, capital - long_invested)  # Use total capital minus long positions

    def close_position(key, exit_price_raw, exit_date, exit_reason):
        nonlocal cash
        pos = positions.pop(key)

        if pos.direction == 'long':
            exit_price = exit_price_raw * (1 - SLIPPAGE_PCT)
            pnl = (exit_price - pos.entry_price) * pos.shares - BROKERAGE_PER_RT
            pnl_pct = (exit_price / pos.entry_price - 1) * 100
        else:
            exit_price = exit_price_raw * (1 + SLIPPAGE_PCT)
            pnl = (pos.entry_price - exit_price) * pos.shares - BROKERAGE_PER_RT
            pnl_pct = (pos.entry_price / exit_price - 1) * 100

        cash += pos.capital_used + pnl
        hold_days = (exit_date - pos.entry_date).days
        trades.append(Trade(
            symbol=pos.symbol, strategy=pos.strategy, direction=pos.direction,
            entry_date=pos.entry_date, exit_date=exit_date,
            entry_price=pos.entry_price, exit_price=exit_price,
            shares=pos.shares, pnl=pnl, pnl_pct=pnl_pct,
            exit_reason=exit_reason, hold_days=hold_days
        ))

    def open_long_position(symbol, strategy, price_raw, date, strategy_capital, max_pos):
        """Open a long position (unchanged from original)."""
        nonlocal cash
        key = f"{strategy}_long_{symbol}"
        if key in positions:
            return
        if count_positions(strategy) >= max_pos:
            return

        invested = get_invested(strategy)
        available = strategy_capital - invested
        pos_size = min(
            strategy_capital / max_pos,
            capital * config.max_position_pct,
            available,
            cash
        )
        if pos_size < 10000:
            return

        entry_price = price_raw * (1 + SLIPPAGE_PCT)
        stop_price = entry_price * (1 - config.hard_stop_pct)
        shares = int(pos_size / entry_price)
        if shares < 1:
            return

        actual_capital = shares * entry_price
        cash -= actual_capital
        positions[key] = Position(
            symbol=symbol, strategy=strategy, direction='long',
            entry_date=date, entry_price=entry_price, shares=shares,
            capital_used=actual_capital, stop_price=stop_price
        )

    def open_short_position(symbol, strategy, price_raw, date, max_pos):
        """Open a short position using SEGREGATED idle capital only.
        Returns True if opened, False if blocked by gate or other reason."""
        nonlocal cash, shorts_blocked_by_gate

        key = f"{strategy}_short_{symbol}"
        if key in positions:
            return False

        # Capital utilization gate: only enter short if long util < threshold
        long_util = get_long_capital_utilization()
        if long_util >= config.capital_util_gate:
            shorts_blocked_by_gate += 1
            return False

        if count_positions(strategy) >= max_pos:
            return False

        # Segregated capital: shorts can only use idle capital
        idle = get_idle_capital()
        short_invested = get_invested(direction='short')
        available_for_shorts = idle - short_invested

        # Size from idle pool, capped at 5% of total
        pos_size = min(
            available_for_shorts / max(1, max_pos - count_positions(strategy, 'short')),
            capital * config.max_position_pct,
            available_for_shorts,
            cash
        )
        if pos_size < 10000:
            return False

        entry_price = price_raw * (1 - SLIPPAGE_PCT)
        stop_price = entry_price * (1 + config.short_stop_pct)  # Use separate short SL
        shares = int(pos_size / entry_price)
        if shares < 1:
            return False

        actual_capital = shares * entry_price
        cash -= actual_capital
        positions[key] = Position(
            symbol=symbol, strategy=strategy, direction='short',
            entry_date=date, entry_price=entry_price, shares=shares,
            capital_used=actual_capital, stop_price=stop_price
        )
        return True

    last_eq_date = None

    for i, (dt, sym, event_type, price) in enumerate(events):
        current_date = dt.date() if hasattr(dt, 'date') else dt

        # Check stop losses
        for key in list(positions.keys()):
            pos = positions[key]
            if sym != pos.symbol:
                continue
            if pos.direction == 'long' and price <= pos.stop_price:
                close_position(key, price, dt, 'stop_loss')
            elif pos.direction == 'short' and price >= pos.stop_price:
                close_position(key, price, dt, 'stop_loss')

        # Process signals
        if event_type == 'trend_long_buy':
            short_key = f"trend_short_{sym}"
            if short_key in positions:
                close_position(short_key, price, dt, 'signal_reversal')
            open_long_position(sym, 'trend', price, dt, trend_capital, config.max_positions_trend)

        elif event_type == 'trend_long_sell':
            key = f"trend_long_{sym}"
            if key in positions:
                close_position(key, price, dt, 'signal_exit')

        elif event_type == 'trend_short_enter':
            long_key = f"trend_long_{sym}"
            if long_key in positions:
                close_position(long_key, price, dt, 'signal_reversal')
            open_short_position(sym, 'trend', price, dt, config.max_positions_trend)

        elif event_type == 'trend_short_cover':
            key = f"trend_short_{sym}"
            if key in positions:
                close_position(key, price, dt, 'signal_exit')

        elif event_type == 'mr_long_entry':
            key = f"meanrev_long_{sym}"
            if key not in positions:
                open_long_position(sym, 'meanrev', price, dt, meanrev_capital, config.max_positions_meanrev)

        elif event_type == 'mr_long_exit':
            key = f"meanrev_long_{sym}"
            if key in positions:
                close_position(key, price, dt, 'rsi_exit')

        elif event_type == 'mr_short_enter':
            key = f"meanrev_short_{sym}"
            if key not in positions:
                open_short_position(sym, 'meanrev', price, dt, config.max_positions_meanrev)

        elif event_type == 'mr_short_cover':
            key = f"meanrev_short_{sym}"
            if key in positions:
                close_position(key, price, dt, 'rsi_exit')

        # Equity curve
        if current_date != last_eq_date:
            mtm = cash
            for key, pos in positions.items():
                latest_price = daily_close.get((pos.symbol, current_date))
                if latest_price:
                    if pos.direction == 'long':
                        mtm += latest_price * pos.shares
                    else:
                        mtm += pos.capital_used + (pos.entry_price - latest_price) * pos.shares
                else:
                    mtm += pos.capital_used
            equity_curve[current_date] = mtm
            last_eq_date = current_date

    # Close remaining positions
    for key in list(positions.keys()):
        pos = positions[key]
        last_price = None
        for sym_dt in sorted(daily_close.keys(), reverse=True):
            if sym_dt[0] == pos.symbol:
                last_price = daily_close[sym_dt]
                break
        if last_price:
            close_position(key, last_price,
                           datetime.combine(sorted(daily_close.keys())[-1][1], datetime.min.time()),
                           'end_of_backtest')

    return trades, equity_curve, shorts_blocked_by_gate


# ── Metrics ──

def compute_metrics(trades, equity_curve):
    if not trades or not equity_curve:
        return None
    dates = sorted(equity_curve.keys())
    values = [equity_curve[d] for d in dates]
    years = (dates[-1] - dates[0]).days / 365.25
    final_value = values[-1]
    cagr = (final_value / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 else 0
    peak = INITIAL_CAPITAL
    max_dd = 0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)
    returns = np.diff(values) / np.array(values[:-1]) if len(values) > 1 else np.array([0])
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    neg_ret = returns[returns < 0]
    ds = neg_ret.std() if len(neg_ret) > 0 else 0.001
    sortino = (returns.mean() / ds * np.sqrt(252)) if ds > 0 else 0
    calmar = cagr / max_dd if max_dd > 0 else 0
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnl_pcts if p > 0]
    win_rate = len(wins) / len(pnl_pcts) * 100 if pnl_pcts else 0
    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    pf = gp / gl if gl > 0 else 99.0

    strat_stats = {}
    for s in ['trend', 'meanrev']:
        for d in ['long', 'short']:
            key = f"{s}_{d}"
            st = [t for t in trades if t.strategy == s and t.direction == d]
            if st:
                sgp = sum(t.pnl for t in st if t.pnl > 0)
                sgl = abs(sum(t.pnl for t in st if t.pnl <= 0))
                strat_stats[key] = {
                    'trades': len(st),
                    'pf': round(sgp/sgl, 2) if sgl > 0 else 99.0,
                    'pnl': round(sum(t.pnl for t in st))
                }
            else:
                strat_stats[key] = {'trades': 0, 'pf': 0, 'pnl': 0}

    return {
        'cagr': round(cagr * 100, 2), 'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2), 'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 2), 'profit_factor': round(pf, 2),
        'total_trades': len(trades), 'win_rate': round(win_rate, 1),
        'total_pnl': round(sum(t.pnl for t in trades)),
        'final_value': round(final_value),
        'total_return_pct': round((final_value / INITIAL_CAPITAL - 1) * 100, 2),
        'trend_long_trades': strat_stats['trend_long']['trades'],
        'trend_long_pf': strat_stats['trend_long']['pf'],
        'trend_long_pnl': strat_stats['trend_long']['pnl'],
        'trend_short_trades': strat_stats['trend_short']['trades'],
        'trend_short_pf': strat_stats['trend_short']['pf'],
        'trend_short_pnl': strat_stats['trend_short']['pnl'],
        'mr_long_trades': strat_stats['meanrev_long']['trades'],
        'mr_long_pf': strat_stats['meanrev_long']['pf'],
        'mr_long_pnl': strat_stats['meanrev_long']['pnl'],
        'mr_short_trades': strat_stats['meanrev_short']['trades'],
        'mr_short_pf': strat_stats['meanrev_short']['pf'],
        'mr_short_pnl': strat_stats['meanrev_short']['pnl'],
    }


def write_row(row):
    exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    with open(OUTPUT_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ── Config definitions ──

# Skip already done
done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {r['label'] for r in csv.DictReader(f)}
    if done:
        print(f"Skipping {len(done)} already done")

print("Loading data...", flush=True)
data_60, data_daily = load_data()

# ── Build config matrix ──
configs = []

# 0. LONG-ONLY BASELINE (no shorts)
configs.append(('BASELINE_LONG_ONLY', 'LONG_ONLY', Config(
    enable_trend_short=False, enable_mr_short=False,
)))

# 1. MR Short adjustments (12 configs)
# RSI short entry: [75, 80, 85] x Short SL: [3%, 5%, 7%, 10%]
for rsi_se in [75, 80, 85]:
    for ssl in [0.03, 0.05, 0.07, 0.10]:
        ssl_pct = int(ssl * 100)
        label = f"MR_SHORT_RSI{rsi_se}_SL{ssl_pct}"
        configs.append((label, 'MR_LS', Config(
            enable_trend_short=False,
            enable_mr_short=True,
            rsi_short_entry=float(rsi_se),
            short_stop_pct=ssl,
            capital_util_gate=0.50,
        )))

# 2. Trend Short adjustments (6 configs)
# ADX threshold for shorts: [30, 35] x Short SL: [3%, 5%, 7%]
for adx_s in [30, 35]:
    for ssl in [0.03, 0.05, 0.07]:
        ssl_pct = int(ssl * 100)
        label = f"TREND_SHORT_ADX{adx_s}_SL{ssl_pct}"
        configs.append((label, 'TREND_LS', Config(
            enable_trend_short=True,
            enable_mr_short=False,
            short_adx_threshold=adx_s,
            short_stop_pct=ssl,
            capital_util_gate=0.50,
        )))

# 3. Combined best combos (test likely best from each category)
# Best MR (RSI 80 + 5% SL) + Best Trend (ADX 30 + 5% SL)
combined_combos = [
    (80, 0.05, 30, 0.05, "COMBINED_MR80_SL5_TADX30_SL5"),
    (85, 0.03, 35, 0.03, "COMBINED_MR85_SL3_TADX35_SL3"),
    (80, 0.07, 30, 0.07, "COMBINED_MR80_SL7_TADX30_SL7"),
]
for rsi_se, mr_ssl, adx_s, t_ssl, label in combined_combos:
    # For combined, use the tighter of the two SLs as the unified short SL
    # (since we can only set one short_stop_pct per config)
    # Use the MR short SL (they share the same field in this implementation)
    configs.append((label, 'BOTH_LS', Config(
        enable_trend_short=True,
        enable_mr_short=True,
        rsi_short_entry=float(rsi_se),
        short_adx_threshold=adx_s,
        short_stop_pct=mr_ssl,  # unified short SL for simplicity
        capital_util_gate=0.50,
    )))

print(f"\nTotal configs: {len(configs)} ({len(done)} done)")
sys.stdout.flush()

signal_cache = {}
total = len(configs)

for i, (label, mode, config) in enumerate(configs, 1):
    if label in done:
        continue

    # Cache key includes all signal-affecting params
    sig_key = (config.ema_fast, config.ema_slow, config.adx_threshold,
               config.short_adx_threshold,
               config.rsi_period, config.rsi_entry,
               config.enable_trend_short, config.enable_mr_short,
               config.rsi_short_entry)

    if sig_key not in signal_cache:
        events = precompute_all_signals(data_60, data_daily, config)
        signal_cache[sig_key] = events
        if len(signal_cache) > 8:
            oldest = next(iter(signal_cache))
            del signal_cache[oldest]
    else:
        events = signal_cache[sig_key]

    t0 = time.time()
    print(f"[{i}/{total}] {label} ...", end='', flush=True)

    try:
        trades, eq_curve, blocked = run_portfolio_backtest(events, data_60, data_daily, config)
        metrics = compute_metrics(trades, eq_curve)

        if metrics:
            row = {
                'label': label, 'mode': mode, **metrics,
                'short_sl': config.short_stop_pct,
                'rsi_short_entry': config.rsi_short_entry,
                'short_adx_threshold': config.short_adx_threshold,
                'capital_util_gate': config.capital_util_gate,
                'shorts_blocked_by_gate': blocked,
            }
            write_row(row)
            elapsed = time.time() - t0
            print(f" {elapsed:.0f}s | CAGR={metrics['cagr']:.2f}% Sharpe={metrics['sharpe']:.2f} "
                  f"MaxDD={metrics['max_drawdown']:.1f}% Calmar={metrics['calmar']:.2f} "
                  f"PF={metrics['profit_factor']:.2f} Trades={metrics['total_trades']} "
                  f"T_S={metrics['trend_short_trades']}(PF={metrics['trend_short_pf']:.2f}) "
                  f"MR_S={metrics['mr_short_trades']}(PF={metrics['mr_short_pf']:.2f}) "
                  f"Blocked={blocked}", flush=True)
        else:
            print(f" {time.time()-t0:.0f}s | NO TRADES", flush=True)
    except Exception as e:
        print(f" {time.time()-t0:.0f}s | ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

print("\nSweep complete!")
print(f"Results in: {OUTPUT_CSV}")
