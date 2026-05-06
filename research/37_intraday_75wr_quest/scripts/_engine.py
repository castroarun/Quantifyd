"""Shared utilities for research/37 intraday-75WR quest.

Loads 5-min data from backtest_data/market_data.db, builds session-aware
DataFrames, computes common intraday primitives (VWAP, day-OR, ATR, RSI,
session classification).

Designed for *vectorised* operation across many stocks: load once, cache in
a dict {symbol: df}, reuse across strategy families.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..',
    'backtest_data', 'market_data.db',
)
DB_PATH = os.path.normpath(DB_PATH)

SESSION_START = '09:15'
SESSION_END = '15:15'  # last 5-min candle stamp at 15:25 actually; data uses bar-open stamping


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def list_5min_universe(min_rows: int = 32_000) -> List[str]:
    """Return symbols with at least `min_rows` 5-min candles."""
    with _connect() as con:
        df = pd.read_sql(
            """SELECT symbol, COUNT(*) AS n
               FROM market_data_unified
               WHERE timeframe='5minute'
               GROUP BY symbol
               HAVING n >= ?
               ORDER BY n DESC""",
            con, params=(min_rows,),
        )
    return df['symbol'].tolist()


def load_5min(symbol: str, start: str = '2024-03-18', end: str = '2026-03-25') -> pd.DataFrame:
    """Load 5-min OHLCV for a single symbol, indexed by timestamp."""
    with _connect() as con:
        df = pd.read_sql(
            """SELECT date, open, high, low, close, volume
               FROM market_data_unified
               WHERE timeframe='5minute' AND symbol=? AND date>=? AND date<=?
               ORDER BY date""",
            con, params=(symbol, start, end + ' 23:59:59'),
        )
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['session'] = df.index.normalize()
    df['t'] = df.index.strftime('%H:%M')
    df['bar_idx'] = df.groupby('session').cumcount()
    return df


def load_many_5min(symbols: List[str], start: str = '2024-03-18', end: str = '2026-03-25') -> Dict[str, pd.DataFrame]:
    """Bulk load multiple symbols. Returns {symbol: df}."""
    out: Dict[str, pd.DataFrame] = {}
    with _connect() as con:
        for sym in symbols:
            df = pd.read_sql(
                """SELECT date, open, high, low, close, volume
                   FROM market_data_unified
                   WHERE timeframe='5minute' AND symbol=? AND date>=? AND date<=?
                   ORDER BY date""",
                con, params=(sym, start, end + ' 23:59:59'),
            )
            if df.empty:
                continue
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df['session'] = df.index.normalize()
            df['t'] = df.index.strftime('%H:%M')
            df['bar_idx'] = df.groupby('session').cumcount()
            out[sym] = df
    return out


# ---------------------------------------------------------------------------
# intraday primitives
# ---------------------------------------------------------------------------

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add intraday VWAP (resets each session)."""
    typical = (df['high'] + df['low'] + df['close']) / 3.0
    pv = typical * df['volume']
    cpv = pv.groupby(df['session']).cumsum()
    cv = df['volume'].groupby(df['session']).cumsum().replace(0, np.nan)
    df['vwap'] = (cpv / cv).fillna(typical)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ATR on 5-min bars."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(period, min_periods=1).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder RSI on close."""
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = (100 - 100 / (1 + rs)).fillna(50)
    return df


def add_emas(df: pd.DataFrame, periods: Tuple[int, ...] = (9, 21, 50)) -> pd.DataFrame:
    for p in periods:
        df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()
    return df


def add_session_levels(df: pd.DataFrame, or_minutes: int = 15) -> pd.DataFrame:
    """Add per-session opening range high/low + day OHLC running highs/lows."""
    bars = max(1, or_minutes // 5)
    or_mask = df['bar_idx'] < bars
    or_high = df.loc[or_mask].groupby('session')['high'].max()
    or_low = df.loc[or_mask].groupby('session')['low'].min()
    df['or_high'] = df['session'].map(or_high)
    df['or_low'] = df['session'].map(or_low)
    df['day_high'] = df.groupby('session')['high'].cummax()
    df['day_low'] = df.groupby('session')['low'].cummin()
    df['day_open'] = df.groupby('session')['open'].transform('first')
    return df


def add_prev_close(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-session close (for gap calc)."""
    last_close = df.groupby('session')['close'].last()
    pc = last_close.shift(1).rename('prev_close')
    df = df.merge(pc, left_on='session', right_index=True, how='left')
    return df


def enrich(df: pd.DataFrame, or_minutes: int = 15) -> pd.DataFrame:
    df = add_vwap(df)
    df = add_atr(df)
    df = add_rsi(df)
    df = add_emas(df)
    df = add_session_levels(df, or_minutes=or_minutes)
    df = add_prev_close(df)
    return df


# ---------------------------------------------------------------------------
# session-level summary (one row per session per stock)
# ---------------------------------------------------------------------------

def session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a 5-min df into one row per session with daily OHLC + flags.

    Tolerant of raw (unenriched) input: only includes OR/prev_close columns
    if they were already added via enrich() / add_session_levels().
    """
    g = df.groupby('session')
    cols = {
        'open': g['open'].first(),
        'high': g['high'].max(),
        'low': g['low'].min(),
        'close': g['close'].last(),
        'volume': g['volume'].sum(),
        'bars': g.size(),
    }
    for opt in ('or_high', 'or_low', 'day_open', 'prev_close'):
        if opt in df.columns:
            cols[opt] = g[opt].first()
    out = pd.DataFrame(cols)
    if 'prev_close' not in out.columns:
        out['prev_close'] = out['close'].shift(1)
    out['range'] = out['high'] - out['low']
    out['range_pct'] = out['range'] / out['open'] * 100
    out['gap_pct'] = (out['open'] - out['prev_close']) / out['prev_close'] * 100
    out['close_pos'] = (out['close'] - out['low']) / out['range'].replace(0, np.nan)
    out['open_pos'] = (out['open'] - out['low']) / out['range'].replace(0, np.nan)
    out['ret_pct'] = (out['close'] - out['open']) / out['open'] * 100
    out['open_eq_low'] = out['open'] == out['low']
    out['open_eq_high'] = out['open'] == out['high']
    return out


# ---------------------------------------------------------------------------
# trade simulator (long-only, single entry per session)
# ---------------------------------------------------------------------------

@dataclass
class TradeRules:
    """Simple intraday trade rules: enter on signal, exit on TP / SL / EOD."""
    tp_pct: float = 1.0          # take-profit % above entry
    sl_pct: float = 0.5          # stop-loss % below entry
    max_hold_bars: int = 60      # bars held before EOD market exit (60 * 5min = 5h ≈ session)
    direction: str = 'long'      # 'long' or 'short'


def simulate_signals(df: pd.DataFrame, signal: pd.Series, rules: TradeRules) -> pd.DataFrame:
    """Given a boolean signal Series aligned to df, simulate intraday entries.

    For each signal=True bar, enter at NEXT bar's open. Exit on first of:
    TP hit, SL hit, max_hold_bars, or session end.

    Returns trade-level DataFrame.

    Numpy-vectorised inner loop — per-trade exit search uses array slice + argmax.
    """
    sig_arr = signal.values if hasattr(signal, 'values') else np.asarray(signal)
    sig_idx = np.where(sig_arr)[0]
    if len(sig_idx) == 0:
        return pd.DataFrame()

    n = len(df)
    open_a = df['open'].to_numpy()
    high_a = df['high'].to_numpy()
    low_a = df['low'].to_numpy()
    close_a = df['close'].to_numpy()
    sess_a = df['session'].to_numpy()
    idx_a = df.index.to_numpy()

    # session end position lookup — last bar index of each session
    sess_end_pos = pd.Series(np.arange(n), index=df['session'].values).groupby(level=0).max()

    trades_t = []
    trades_p = []
    trades_x = []
    trades_xt = []
    trades_r = []
    trades_b = []
    trades_xr = []

    is_long = (rules.direction == 'long')
    tp_mult = (1 + rules.tp_pct / 100) if is_long else (1 - rules.tp_pct / 100)
    sl_mult = (1 - rules.sl_pct / 100) if is_long else (1 + rules.sl_pct / 100)

    for i in sig_idx:
        entry_i = i + 1
        if entry_i >= n:
            continue
        sess = sess_a[entry_i]
        if sess != sess_a[i]:
            continue
        entry_price = open_a[entry_i]
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        tp_price = entry_price * tp_mult
        sl_price = entry_price * sl_mult

        try:
            sep = int(sess_end_pos.loc[sess])
        except KeyError:
            sep = n - 1
        max_exit_i = min(entry_i + rules.max_hold_bars, sep, n - 1)

        # vectorised search within the trade window
        hi = high_a[entry_i:max_exit_i + 1]
        lo = low_a[entry_i:max_exit_i + 1]
        if is_long:
            sl_hit = lo <= sl_price
            tp_hit = hi >= tp_price
        else:
            sl_hit = hi >= sl_price
            tp_hit = lo <= tp_price

        sl_first = np.argmax(sl_hit) if sl_hit.any() else -1
        tp_first = np.argmax(tp_hit) if tp_hit.any() else -1

        if sl_first == -1 and tp_first == -1:
            offset = max_exit_i - entry_i
            exit_reason = 'EOD'
            exit_price = close_a[max_exit_i]
        elif sl_first == -1:
            offset = tp_first
            exit_reason = 'TP'
            exit_price = tp_price
        elif tp_first == -1:
            offset = sl_first
            exit_reason = 'SL'
            exit_price = sl_price
        elif sl_first <= tp_first:
            offset = sl_first
            exit_reason = 'SL'
            exit_price = sl_price
        else:
            offset = tp_first
            exit_reason = 'TP'
            exit_price = tp_price

        exit_i = entry_i + offset
        if is_long:
            ret_pct = (exit_price - entry_price) / entry_price * 100
        else:
            ret_pct = (entry_price - exit_price) / entry_price * 100

        trades_t.append(idx_a[entry_i])
        trades_p.append(entry_price)
        trades_x.append(exit_price)
        trades_xt.append(idx_a[exit_i])
        trades_r.append(ret_pct)
        trades_b.append(offset)
        trades_xr.append(exit_reason)

    if not trades_t:
        return pd.DataFrame()

    return pd.DataFrame({
        'entry_time': trades_t,
        'entry_price': trades_p,
        'exit_time': trades_xt,
        'exit_price': trades_x,
        'exit_reason': trades_xr,
        'bars_held': trades_b,
        'ret_pct': trades_r,
    })


def trade_stats(trades: pd.DataFrame, sessions: int = 1) -> dict:
    """Compute summary stats from a trade DataFrame."""
    if trades is None or len(trades) == 0:
        return dict(n_trades=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                    profit_factor=0.0, sharpe=0.0, max_dd_pct=0.0,
                    total_return_pct=0.0, trades_per_year=0.0, aws=0.0)

    wins = trades['ret_pct'] > 0
    losses = trades['ret_pct'] <= 0
    win_rate = wins.mean()
    avg_win = trades.loc[wins, 'ret_pct'].mean() if wins.any() else 0
    avg_loss = trades.loc[losses, 'ret_pct'].mean() if losses.any() else 0
    gp = trades.loc[wins, 'ret_pct'].sum()
    gl = -trades.loc[losses, 'ret_pct'].sum()
    pf = gp / gl if gl > 0 else (np.inf if gp > 0 else 0.0)

    # equity curve in pct units
    eq = trades['ret_pct'].cumsum()
    peak = eq.cummax()
    dd = peak - eq
    max_dd = dd.max() if len(dd) else 0

    # sharpe — daily ret approx
    daily_rets = trades.groupby(trades['entry_time'].dt.date)['ret_pct'].sum()
    if daily_rets.std() > 0:
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
    else:
        sharpe = 0

    total_ret = trades['ret_pct'].sum()
    trades_per_year = len(trades) / max(sessions, 1) * 250

    aws = (
        win_rate
        * np.log1p(len(trades))
        * min(pf / 2, 1.5 if np.isfinite(pf) else 1.5)
        * min(max(sharpe, 0) / 1.5, 1.5)
    )

    return dict(
        n_trades=int(len(trades)),
        win_rate=float(win_rate),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        profit_factor=float(pf if np.isfinite(pf) else 999),
        sharpe=float(sharpe),
        max_dd_pct=float(max_dd),
        total_return_pct=float(total_ret),
        trades_per_year=float(trades_per_year),
        aws=float(aws),
    )


def passes_hard_gates(stats: dict, min_trades: int = 30) -> bool:
    """Hard gates from STATUS doc section 1."""
    return (
        stats['n_trades'] >= min_trades
        and stats['win_rate'] >= 0.75
        and stats['profit_factor'] >= 2.0
        and stats['max_dd_pct'] <= 15.0
        and stats['sharpe'] >= 1.5
    )
