"""
Long+Short Sweep: Test adding short signals to EMA Trend and RSI Mean Reversion.
Compare long-only vs long+short for the best configs from Phase 3.

Short signals:
  - Trend SHORT: EMA fast crosses BELOW slow + ADX > threshold
  - MeanRev SHORT: RSI > overbought_level + price < SMA(200)  (contradiction of long entry)
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
from dataclasses import dataclass
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.insert(0, PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'longshort_sweep.csv')

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
    'ema_fast', 'ema_slow', 'adx_threshold', 'rsi_period', 'rsi_entry',
    'max_pos_trend', 'max_pos_meanrev', 'hard_stop_pct',
    'trend_long_trades', 'trend_long_pf', 'trend_long_pnl',
    'trend_short_trades', 'trend_short_pf', 'trend_short_pnl',
    'mr_long_trades', 'mr_long_pf', 'mr_long_pnl',
    'mr_short_trades', 'mr_short_pf', 'mr_short_pnl',
]


@dataclass
class Config:
    ema_fast: int = 20
    ema_slow: int = 50
    adx_threshold: int = 25
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
    hard_stop_pct: float = 0.10
    trend_alloc: float = 0.60
    meanrev_alloc: float = 0.40
    enable_trend_short: bool = False
    enable_mr_short: bool = False


@dataclass
class Position:
    symbol: str
    strategy: str
    direction: str  # 'long' or 'short'
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
    """EMA crossover + ADX. Now generates both long and short signals."""
    df = df_60min.copy()
    df['ema_fast'] = compute_ema(df['close'], config.ema_fast)
    df['ema_slow'] = compute_ema(df['close'], config.ema_slow)
    df['adx'] = compute_adx(df['high'], df['low'], df['close'], config.adx_period)

    cross_up = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
    cross_dn = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
    strong_trend = df['adx'] >= config.adx_threshold

    df['signal'] = 0
    # Long signals
    df.loc[cross_up & strong_trend, 'signal'] = 1    # buy long
    df.loc[cross_dn, 'signal'] = -1                   # exit long (always, regardless of ADX)

    # Short signals (only if enabled)
    df['short_signal'] = 0
    if config.enable_trend_short:
        df.loc[cross_dn & strong_trend, 'short_signal'] = -1  # enter short
        df.loc[cross_up, 'short_signal'] = 1                   # exit short (cover)

    return df[['open', 'high', 'low', 'close', 'volume', 'signal', 'short_signal']].copy()


def generate_meanrev_signals(df_60min, config):
    """RSI mean reversion. Long: RSI<30 + price>SMA200. Short: RSI>70 + price<SMA200."""
    df = df_60min.copy()
    df['rsi'] = compute_rsi(df['close'], config.rsi_period)
    df['sma200'] = compute_sma(df['close'], config.sma_period)

    # Long: oversold + uptrend
    df['entry_signal'] = (df['rsi'] < config.rsi_entry) & (df['close'] > df['sma200'])
    df['exit_signal'] = df['rsi'] > config.rsi_exit

    # Short: overbought + downtrend (contradiction of long)
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

        # Long trend signals
        for dt, row in sig_df[sig_df['signal'] == 1].iterrows():
            events.append((dt, sym, 'trend_long_buy', row['close']))
        for dt, row in sig_df[sig_df['signal'] == -1].iterrows():
            events.append((dt, sym, 'trend_long_sell', row['close']))

        # Short trend signals
        if config.enable_trend_short:
            for dt, row in sig_df[sig_df['short_signal'] == -1].iterrows():
                events.append((dt, sym, 'trend_short_enter', row['close']))
            for dt, row in sig_df[sig_df['short_signal'] == 1].iterrows():
                events.append((dt, sym, 'trend_short_cover', row['close']))

    for sym, df in data_60.items():
        sig_df = generate_meanrev_signals(df, config)

        # Long MR signals
        for dt, row in sig_df[sig_df['entry_signal']].iterrows():
            events.append((dt, sym, 'mr_long_entry', row['close']))
        for dt, row in sig_df[sig_df['exit_signal']].iterrows():
            events.append((dt, sym, 'mr_long_exit', row['close']))

        # Short MR signals
        if config.enable_mr_short:
            for dt, row in sig_df[sig_df['short_entry_signal']].iterrows():
                events.append((dt, sym, 'mr_short_enter', row['close']))
            for dt, row in sig_df[sig_df['short_exit_signal']].iterrows():
                events.append((dt, sym, 'mr_short_cover', row['close']))

    events.sort(key=lambda x: x[0])
    return events


# ── Portfolio Simulation ──

def run_portfolio_backtest(events, data_60, data_daily, config):
    capital = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []
    equity_curve = {}

    trend_capital = capital * config.trend_alloc
    meanrev_capital = capital * config.meanrev_alloc

    daily_close = {}
    for sym, df in data_daily.items():
        for dt, row in df.iterrows():
            daily_close[(sym, dt.date())] = row['close']

    def count_positions(strategy, direction=None):
        return sum(1 for k, p in positions.items()
                   if p.strategy == strategy and (direction is None or p.direction == direction))

    def get_invested(strategy):
        return sum(p.capital_used for k, p in positions.items() if p.strategy == strategy)

    def close_position(key, exit_price_raw, exit_date, exit_reason):
        nonlocal cash
        pos = positions.pop(key)

        if pos.direction == 'long':
            exit_price = exit_price_raw * (1 - SLIPPAGE_PCT)
            pnl = (exit_price - pos.entry_price) * pos.shares - BROKERAGE_PER_RT
            pnl_pct = (exit_price / pos.entry_price - 1) * 100
        else:  # short
            exit_price = exit_price_raw * (1 + SLIPPAGE_PCT)  # slippage works against you
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

    def open_position(symbol, strategy, direction, price_raw, date, strategy_capital, max_pos):
        nonlocal cash

        key = f"{strategy}_{direction}_{symbol}"
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

        if direction == 'long':
            entry_price = price_raw * (1 + SLIPPAGE_PCT)
            stop_price = entry_price * (1 - config.hard_stop_pct)
        else:  # short
            entry_price = price_raw * (1 - SLIPPAGE_PCT)  # favorable fill for short
            stop_price = entry_price * (1 + config.hard_stop_pct)  # stop above entry

        shares = int(pos_size / entry_price)
        if shares < 1:
            return

        actual_capital = shares * entry_price
        cash -= actual_capital

        positions[key] = Position(
            symbol=symbol, strategy=strategy, direction=direction,
            entry_date=date, entry_price=entry_price, shares=shares,
            capital_used=actual_capital, stop_price=stop_price
        )

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
            # Also covers any existing short on this symbol
            short_key = f"trend_short_{sym}"
            if short_key in positions:
                close_position(short_key, price, dt, 'signal_reversal')
            open_position(sym, 'trend', 'long', price, dt, trend_capital, config.max_positions_trend)

        elif event_type == 'trend_long_sell':
            key = f"trend_long_{sym}"
            if key in positions:
                close_position(key, price, dt, 'signal_exit')

        elif event_type == 'trend_short_enter':
            # Also close any existing long on this symbol
            long_key = f"trend_long_{sym}"
            if long_key in positions:
                close_position(long_key, price, dt, 'signal_reversal')
            open_position(sym, 'trend', 'short', price, dt, trend_capital, config.max_positions_trend)

        elif event_type == 'trend_short_cover':
            key = f"trend_short_{sym}"
            if key in positions:
                close_position(key, price, dt, 'signal_exit')

        elif event_type == 'mr_long_entry':
            key = f"meanrev_long_{sym}"
            if key not in positions:
                open_position(sym, 'meanrev', 'long', price, dt, meanrev_capital, config.max_positions_meanrev)

        elif event_type == 'mr_long_exit':
            key = f"meanrev_long_{sym}"
            if key in positions:
                close_position(key, price, dt, 'rsi_exit')

        elif event_type == 'mr_short_enter':
            key = f"meanrev_short_{sym}"
            if key not in positions:
                open_position(sym, 'meanrev', 'short', price, dt, meanrev_capital, config.max_positions_meanrev)

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
                    else:  # short: capital + (entry - current) * shares
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

    return trades, equity_curve


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

    # Per strategy+direction breakdown
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

# Test the top configs from Phase 3 in 4 modes each:
# 1. LONG_ONLY (baseline)
# 2. TREND_LS (trend long+short, MR long only)
# 3. MR_LS (trend long only, MR long+short)
# 4. BOTH_LS (both strategies long+short)

BASE_CONFIGS = [
    # (label_base, ema_f, ema_s, adx, rsi_p, rsi_e, mp_t, mp_m)
    ('E20_50_ADX25_R14_30_P20_10', 20, 50, 25, 14, 30, 20, 10),   # Best Calmar
    ('E25_60_ADX20_R21_30_P20_10', 25, 60, 20, 21, 30, 20, 10),   # Best CAGR
    ('E20_50_ADX30_R21_30_P20_10', 20, 50, 30, 21, 30, 20, 10),   # Best PF
    ('E25_60_ADX30_R21_30_P20_10', 25, 60, 30, 21, 30, 20, 10),   # Best Sharpe #2
    ('E25_60_ADX20_R14_30_P20_10', 25, 60, 20, 14, 30, 20, 10),   # Hybrid best
]

MODES = [
    ('LONG_ONLY',  False, False),
    ('TREND_LS',   True,  False),
    ('MR_LS',      False, True),
    ('BOTH_LS',    True,  True),
]

configs = []
for base_label, ef, es, adx, rp, re, mpt, mpm in BASE_CONFIGS:
    for mode_label, trend_short, mr_short in MODES:
        label = f"{base_label}_{mode_label}"
        c = Config(
            ema_fast=ef, ema_slow=es, adx_threshold=adx,
            rsi_period=rp, rsi_entry=re,
            max_positions_trend=mpt, max_positions_meanrev=mpm,
            enable_trend_short=trend_short,
            enable_mr_short=mr_short,
        )
        configs.append((label, mode_label, c))

# Also test RSI short entry levels for MR
for rsi_short_entry in [65, 70, 75, 80]:
    label = f"E20_50_ADX25_R14_30_P20_10_MR_SHORT_RSI{rsi_short_entry}"
    c = Config(
        ema_fast=20, ema_slow=50, adx_threshold=25,
        rsi_period=14, rsi_entry=30,
        rsi_short_entry=float(rsi_short_entry),
        max_positions_trend=20, max_positions_meanrev=10,
        enable_trend_short=False,
        enable_mr_short=True,
    )
    configs.append((label, 'MR_LS', c))

# Test both LS with RSI short levels
for rsi_short_entry in [65, 70, 75, 80]:
    label = f"E20_50_ADX25_R14_30_P20_10_BOTH_RSI{rsi_short_entry}"
    c = Config(
        ema_fast=20, ema_slow=50, adx_threshold=25,
        rsi_period=14, rsi_entry=30,
        rsi_short_entry=float(rsi_short_entry),
        max_positions_trend=20, max_positions_meanrev=10,
        enable_trend_short=True,
        enable_mr_short=True,
    )
    configs.append((label, 'BOTH_LS', c))


print(f"\nTotal configs: {len(configs)} ({len(done)} done)")

signal_cache = {}
total = len(configs)

for i, (label, mode, config) in enumerate(configs, 1):
    if label in done:
        continue

    sig_key = (config.ema_fast, config.ema_slow, config.adx_threshold,
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
        trades, eq_curve = run_portfolio_backtest(events, data_60, data_daily, config)
        metrics = compute_metrics(trades, eq_curve)

        if metrics:
            row = {
                'label': label, 'mode': mode, **metrics,
                'ema_fast': config.ema_fast, 'ema_slow': config.ema_slow,
                'adx_threshold': config.adx_threshold,
                'rsi_period': config.rsi_period, 'rsi_entry': config.rsi_entry,
                'max_pos_trend': config.max_positions_trend,
                'max_pos_meanrev': config.max_positions_meanrev,
                'hard_stop_pct': config.hard_stop_pct,
            }
            write_row(row)
            elapsed = time.time() - t0
            print(f" {elapsed:.0f}s | CAGR={metrics['cagr']:.2f}% Sharpe={metrics['sharpe']:.2f} "
                  f"MaxDD={metrics['max_drawdown']:.1f}% Calmar={metrics['calmar']:.2f} "
                  f"PF={metrics['profit_factor']:.2f} Trades={metrics['total_trades']} "
                  f"T_L={metrics['trend_long_trades']} T_S={metrics['trend_short_trades']} "
                  f"MR_L={metrics['mr_long_trades']} MR_S={metrics['mr_short_trades']}", flush=True)
        else:
            print(f" {time.time()-t0:.0f}s | NO TRADES", flush=True)
    except Exception as e:
        print(f" {time.time()-t0:.0f}s | ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

print("\nSweep complete!")
print(f"Results in: {OUTPUT_CSV}")
