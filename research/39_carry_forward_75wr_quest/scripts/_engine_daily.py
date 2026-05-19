"""Daily-bar engine for research/39 carry-forward quest.

Mirrors research/37 _engine.py patterns but operates on daily bars from
backtest_data/market_data.db, timeframe='day'. Adds multi-day hold simulator,
weekly resampling, and standard indicators (SMA, EMA, RSI, ATR, Donchian,
MACD, Bollinger Bands).

Designed for vectorised loading and per-stock signal sweeping.
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


# ---------------------------------------------------------------------------
# F&O universe (from services/data_manager.py FNO_LOT_SIZES)
# ---------------------------------------------------------------------------

FNO_UNIVERSE: List[str] = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN',
    'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'TITAN',
    'BAJFINANCE', 'HCLTECH', 'SUNPHARMA', 'ULTRACEMCO', 'NESTLEIND', 'WIPRO', 'ONGC',
    'NTPC', 'POWERGRID', 'M&M', 'TATAMOTORS', 'TECHM', 'JSWSTEEL', 'INDUSINDBK',
    'ADANIPORTS', 'BAJAJFINSV', 'HINDALCO', 'COALINDIA', 'DIVISLAB', 'GRASIM',
    'TATACONSUM', 'DRREDDY', 'CIPLA', 'EICHERMOT', 'BRITANNIA', 'APOLLOHOSP',
    'HEROMOTOCO', 'SBILIFE', 'BPCL', 'TATASTEEL', 'BAJAJ-AUTO', 'HDFCLIFE',
    'SHREECEM', 'ADANIENT', 'VEDL', 'TATAPOWER', 'GAIL', 'JINDALSTEL', 'DLF',
    'GODREJPROP', 'SIEMENS', 'HAVELLS', 'PIDILITIND', 'DABUR', 'MARICO', 'COLPAL',
    'MUTHOOTFIN', 'CHOLAFIN', 'BEL', 'HAL', 'IOC', 'IRCTC', 'COFORGE', 'PERSISTENT',
    'MCX', 'CUMMINSIND', 'VOLTAS', 'AMBUJACEM', 'TRENT', 'PAYTM', 'DELHIVERY',
    'PNB', 'BANKBARODA', 'FEDERALBNK', 'IDFCFIRSTB',
]


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_daily(symbol: str,
               start: str = '2015-01-01',
               end: str = '2026-03-31') -> pd.DataFrame:
    """Load daily OHLCV for a symbol, indexed by date."""
    with _connect() as con:
        df = pd.read_sql(
            """SELECT date, open, high, low, close, volume
               FROM market_data_unified
               WHERE timeframe='day' AND symbol=? AND date>=? AND date<=?
               ORDER BY date""",
            con, params=(symbol, start, end),
        )
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


def load_many_daily(symbols: List[str],
                    start: str = '2015-01-01',
                    end: str = '2026-03-31') -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    with _connect() as con:
        for sym in symbols:
            df = pd.read_sql(
                """SELECT date, open, high, low, close, volume
                   FROM market_data_unified
                   WHERE timeframe='day' AND symbol=? AND date>=? AND date<=?
                   ORDER BY date""",
                con, params=(sym, start, end),
            )
            if df.empty:
                continue
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            out[sym] = df
    return out


def load_5min_lasthour(symbol: str,
                       start: str = '2018-01-01',
                       end: str = '2026-03-25') -> pd.DataFrame:
    """Load only the last-hour (14:15-15:25) 5-min bars per session for BTST."""
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
    last_hour = (df['t'] >= '14:15') & (df['t'] <= '15:25')
    return df[last_hour].copy()


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------

def add_sma(df: pd.DataFrame, periods: Tuple[int, ...] = (10, 20, 50, 200)) -> pd.DataFrame:
    for p in periods:
        df[f'sma{p}'] = df['close'].rolling(p, min_periods=p).mean()
    return df


def add_ema(df: pd.DataFrame, periods: Tuple[int, ...] = (9, 21, 50)) -> pd.DataFrame:
    for p in periods:
        df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = (100 - 100 / (1 + rs)).fillna(50)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(period, min_periods=1).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100
    return df


def add_donchian(df: pd.DataFrame, periods: Tuple[int, ...] = (20, 55, 252)) -> pd.DataFrame:
    """Donchian channel — rolling high/low excluding current bar (so signals use prior period)."""
    for p in periods:
        df[f'donch_high_{p}'] = df['high'].rolling(p, min_periods=p).max().shift(1)
        df[f'donch_low_{p}'] = df['low'].rolling(p, min_periods=p).min().shift(1)
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_f = df['close'].ewm(span=fast, adjust=False).mean()
    ema_s = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_f - ema_s
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    mid = df['close'].rolling(period, min_periods=period).mean()
    sd = df['close'].rolling(period, min_periods=period).std()
    df['bb_mid'] = mid
    df['bb_up'] = mid + std_mult * sd
    df['bb_dn'] = mid - std_mult * sd
    df['bb_width_pct'] = (df['bb_up'] - df['bb_dn']) / mid * 100
    return df


def add_volume_avg(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df['vol_sma'] = df['volume'].rolling(period, min_periods=period).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
    return df


def add_close_position(df: pd.DataFrame) -> pd.DataFrame:
    """Where in the daily range did we close? 1 = at high, 0 = at low."""
    rng = (df['high'] - df['low']).replace(0, np.nan)
    df['close_pos'] = (df['close'] - df['low']) / rng
    df['ret_pct'] = df['close'].pct_change() * 100
    df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    return df


def enrich_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_donchian(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_volume_avg(df)
    df = add_close_position(df)
    return df


# ---------------------------------------------------------------------------
# weekly resampling
# ---------------------------------------------------------------------------

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily bars to weekly (Friday-end). Keeps OHLCV semantics."""
    if df.empty:
        return df
    w = pd.DataFrame({
        'open': df['open'].resample('W-FRI').first(),
        'high': df['high'].resample('W-FRI').max(),
        'low': df['low'].resample('W-FRI').min(),
        'close': df['close'].resample('W-FRI').last(),
        'volume': df['volume'].resample('W-FRI').sum(),
    }).dropna()
    return w


def add_weekly_donchian(weekly: pd.DataFrame, period: int = 8) -> pd.DataFrame:
    """Add prior-N-week high/low (excluding current week)."""
    weekly[f'wk_donch_high_{period}'] = weekly['high'].rolling(period, min_periods=period).max().shift(1)
    weekly[f'wk_donch_low_{period}'] = weekly['low'].rolling(period, min_periods=period).min().shift(1)
    weekly[f'wk_prev_low'] = weekly['low'].shift(1)
    weekly[f'wk_prev_close'] = weekly['close'].shift(1)
    return weekly


# ---------------------------------------------------------------------------
# multi-day trade simulator
# ---------------------------------------------------------------------------

@dataclass
class DailyTradeRules:
    """Multi-day trade rules."""
    tp_pct: float = 3.0          # take-profit % above entry
    sl_pct: float = 1.5          # stop-loss % below entry
    max_hold_days: int = 5       # max bars held before time-exit
    direction: str = 'long'      # 'long' or 'short'
    time_exit: str = 'close'     # 'close' = exit at close on max_hold day


def simulate_signals_daily(df: pd.DataFrame,
                           signal: pd.Series,
                           rules: DailyTradeRules,
                           entry_mode: str = 'next_open') -> pd.DataFrame:
    """Multi-day simulator. For each signal=True bar, enter at next-day open
    (or current close if entry_mode='close'). Exit on first of:
    - TP hit (intraday H >= tp_price for long; L <= for short)
    - SL hit (intraday L <= sl_price for long; H >= for short)
    - max_hold_days reached -> exit at that day's close

    BTST special case: max_hold_days=1, time_exit='close' means exit at NEXT day's close.
    Returns trade-level DataFrame.
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
    idx_a = df.index.to_numpy()

    is_long = (rules.direction == 'long')
    tp_mult = (1 + rules.tp_pct / 100) if is_long else (1 - rules.tp_pct / 100)
    sl_mult = (1 - rules.sl_pct / 100) if is_long else (1 + rules.sl_pct / 100)

    rows_t = []
    rows_p = []
    rows_xt = []
    rows_xp = []
    rows_xr = []
    rows_h = []
    rows_r = []

    for i in sig_idx:
        if entry_mode == 'next_open':
            entry_i = i + 1
        else:  # 'close'
            entry_i = i
        if entry_i >= n:
            continue
        entry_price = open_a[entry_i] if entry_mode == 'next_open' else close_a[entry_i]
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        tp_price = entry_price * tp_mult
        sl_price = entry_price * sl_mult

        # exit window: entry_i+1 to entry_i+max_hold_days (so we look at NEXT bars)
        # But for BTST max_hold_days=1, we look at the entry day's H/L too if entry_mode='next_open'?
        # Convention: entry happens at entry_i open. We track H/L from entry_i onward.
        # Day 0 = entry day, day k = k bars after entry.
        last_i = min(entry_i + rules.max_hold_days, n - 1)
        if last_i <= entry_i:
            continue

        # search bars from entry_i+1 (since entry on open, intraday H/L of entry day counts)
        # — convention: track from entry_i (entry day's high/low post-entry)
        offset_found = -1
        exit_price = None
        exit_reason = None

        for k in range(0, last_i - entry_i + 1):
            j = entry_i + k
            if k == 0:
                # entry day: only H/L AFTER entry counts. Approximate by using full day H/L
                # but skip if entry==open (most BTST/swing enter at next-day open).
                # For safety, on entry day we DON'T trigger TP/SL — only from j+1.
                continue
            hi = high_a[j]
            lo = low_a[j]
            if is_long:
                if lo <= sl_price and hi >= tp_price:
                    # Both touched same day — assume SL first (conservative)
                    offset_found = k
                    exit_price = sl_price
                    exit_reason = 'SL'
                    break
                if lo <= sl_price:
                    offset_found = k
                    exit_price = sl_price
                    exit_reason = 'SL'
                    break
                if hi >= tp_price:
                    offset_found = k
                    exit_price = tp_price
                    exit_reason = 'TP'
                    break
            else:
                if hi >= sl_price and lo <= tp_price:
                    offset_found = k
                    exit_price = sl_price
                    exit_reason = 'SL'
                    break
                if hi >= sl_price:
                    offset_found = k
                    exit_price = sl_price
                    exit_reason = 'SL'
                    break
                if lo <= tp_price:
                    offset_found = k
                    exit_price = tp_price
                    exit_reason = 'TP'
                    break

        if offset_found == -1:
            # time exit at last_i close
            offset_found = last_i - entry_i
            exit_price = close_a[last_i]
            exit_reason = 'TIME'

        if not np.isfinite(exit_price) or exit_price <= 0:
            continue

        if is_long:
            ret_pct = (exit_price - entry_price) / entry_price * 100
        else:
            ret_pct = (entry_price - exit_price) / entry_price * 100

        rows_t.append(idx_a[entry_i])
        rows_p.append(entry_price)
        rows_xt.append(idx_a[entry_i + offset_found])
        rows_xp.append(exit_price)
        rows_xr.append(exit_reason)
        rows_h.append(offset_found)
        rows_r.append(ret_pct)

    if not rows_t:
        return pd.DataFrame()

    return pd.DataFrame({
        'entry_time': rows_t,
        'entry_price': rows_p,
        'exit_time': rows_xt,
        'exit_price': rows_xp,
        'exit_reason': rows_xr,
        'days_held': rows_h,
        'ret_pct': rows_r,
    })


def simulate_btst(df: pd.DataFrame, signal: pd.Series, direction: str = 'long') -> pd.DataFrame:
    """BTST simulator: enter at NEXT day's open, exit at NEXT day's close (1-night hold).
    No TP/SL — pure overnight + next-session ride.
    """
    sig_arr = signal.values if hasattr(signal, 'values') else np.asarray(signal)
    sig_idx = np.where(sig_arr)[0]
    if len(sig_idx) == 0:
        return pd.DataFrame()

    n = len(df)
    open_a = df['open'].to_numpy()
    close_a = df['close'].to_numpy()
    idx_a = df.index.to_numpy()
    is_long = (direction == 'long')

    rows_t = []
    rows_p = []
    rows_xt = []
    rows_xp = []
    rows_r = []

    for i in sig_idx:
        entry_i = i + 1
        exit_i = i + 1  # same day's close (since enter at open, exit at close = intraday)
        # Wait — BTST = Buy Today Sell Tomorrow. So enter at day i close, exit at day i+1 close.
        # Re-spec: enter at signal-day close, exit at next-day close.
        entry_i = i
        exit_i = i + 1
        if exit_i >= n:
            continue
        entry_price = close_a[entry_i]
        exit_price = close_a[exit_i]
        if not (np.isfinite(entry_price) and entry_price > 0 and np.isfinite(exit_price) and exit_price > 0):
            continue
        if is_long:
            ret_pct = (exit_price - entry_price) / entry_price * 100
        else:
            ret_pct = (entry_price - exit_price) / entry_price * 100
        rows_t.append(idx_a[entry_i])
        rows_p.append(entry_price)
        rows_xt.append(idx_a[exit_i])
        rows_xp.append(exit_price)
        rows_r.append(ret_pct)

    if not rows_t:
        return pd.DataFrame()
    return pd.DataFrame({
        'entry_time': rows_t,
        'entry_price': rows_p,
        'exit_time': rows_xt,
        'exit_price': rows_xp,
        'exit_reason': ['BTST'] * len(rows_t),
        'days_held': [1] * len(rows_t),
        'ret_pct': rows_r,
    })


def simulate_btst_with_tpsl(df: pd.DataFrame,
                             signal: pd.Series,
                             tp_pct: float,
                             sl_pct: float,
                             direction: str = 'long') -> pd.DataFrame:
    """BTST with TP/SL on the next day's H/L; if neither hit, exit at next day close.
    Entry: signal-day close. Exit: next day's TP/SL hit, else close.
    """
    sig_arr = signal.values if hasattr(signal, 'values') else np.asarray(signal)
    sig_idx = np.where(sig_arr)[0]
    if len(sig_idx) == 0:
        return pd.DataFrame()

    n = len(df)
    high_a = df['high'].to_numpy()
    low_a = df['low'].to_numpy()
    close_a = df['close'].to_numpy()
    idx_a = df.index.to_numpy()
    is_long = (direction == 'long')
    tp_mult = (1 + tp_pct / 100) if is_long else (1 - tp_pct / 100)
    sl_mult = (1 - sl_pct / 100) if is_long else (1 + sl_pct / 100)

    rows_t, rows_p, rows_xt, rows_xp, rows_xr, rows_r = [], [], [], [], [], []
    for i in sig_idx:
        entry_i = i
        next_i = i + 1
        if next_i >= n:
            continue
        entry_price = close_a[entry_i]
        if not (np.isfinite(entry_price) and entry_price > 0):
            continue
        tp_price = entry_price * tp_mult
        sl_price = entry_price * sl_mult
        hi = high_a[next_i]
        lo = low_a[next_i]
        if is_long:
            tp_hit = hi >= tp_price
            sl_hit = lo <= sl_price
        else:
            tp_hit = lo <= tp_price
            sl_hit = hi >= sl_price

        if tp_hit and sl_hit:
            # Conservative: SL first
            exit_price = sl_price
            reason = 'SL'
        elif tp_hit:
            exit_price = tp_price
            reason = 'TP'
        elif sl_hit:
            exit_price = sl_price
            reason = 'SL'
        else:
            exit_price = close_a[next_i]
            reason = 'CLOSE'

        if is_long:
            ret_pct = (exit_price - entry_price) / entry_price * 100
        else:
            ret_pct = (entry_price - exit_price) / entry_price * 100
        rows_t.append(idx_a[entry_i])
        rows_p.append(entry_price)
        rows_xt.append(idx_a[next_i])
        rows_xp.append(exit_price)
        rows_xr.append(reason)
        rows_r.append(ret_pct)

    if not rows_t:
        return pd.DataFrame()
    return pd.DataFrame({
        'entry_time': rows_t,
        'entry_price': rows_p,
        'exit_time': rows_xt,
        'exit_price': rows_xp,
        'exit_reason': rows_xr,
        'days_held': [1] * len(rows_t),
        'ret_pct': rows_r,
    })


# ---------------------------------------------------------------------------
# trade stats
# ---------------------------------------------------------------------------

def trade_stats(trades: pd.DataFrame, cost_pct_round: float = 0.06) -> dict:
    """Compute summary stats. cost_pct_round = round-trip cost in % (subtracted from each trade)."""
    if trades is None or len(trades) == 0:
        return dict(n_trades=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                    profit_factor=0.0, sharpe=0.0, max_dd_pct=0.0,
                    total_return_pct=0.0, avg_days_held=0.0, aws=0.0,
                    cost_pct_round=cost_pct_round)

    net_ret = trades['ret_pct'] - cost_pct_round
    wins_mask = net_ret > 0
    losses_mask = net_ret <= 0
    win_rate = wins_mask.mean()
    avg_win = net_ret[wins_mask].mean() if wins_mask.any() else 0.0
    avg_loss = net_ret[losses_mask].mean() if losses_mask.any() else 0.0
    gp = net_ret[wins_mask].sum()
    gl = -net_ret[losses_mask].sum()
    pf = gp / gl if gl > 0 else (np.inf if gp > 0 else 0.0)

    eq = net_ret.cumsum()
    peak = eq.cummax()
    dd = peak - eq
    max_dd = dd.max() if len(dd) else 0.0

    daily = trades.copy()
    daily['date'] = pd.to_datetime(daily['entry_time']).dt.date
    daily_rets = (daily['ret_pct'] - cost_pct_round).groupby(daily['date']).sum()
    if daily_rets.std() > 0:
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    total_ret = net_ret.sum()
    avg_days = trades['days_held'].mean() if 'days_held' in trades.columns else 0.0

    aws = (
        win_rate
        * np.log1p(len(trades))
        * min((pf if np.isfinite(pf) else 999) / 2, 1.5)
        * min(max(sharpe, 0) / 1.5, 1.5)
    )

    return dict(
        n_trades=int(len(trades)),
        win_rate=float(win_rate),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        profit_factor=float(pf if np.isfinite(pf) else 999.0),
        sharpe=float(sharpe),
        max_dd_pct=float(max_dd),
        total_return_pct=float(total_ret),
        avg_days_held=float(avg_days),
        aws=float(aws),
        cost_pct_round=float(cost_pct_round),
    )


def passes_hard_gates(stats_train: dict, stats_test: dict, min_trades_test: int = 30,
                       min_train_wr: float = 0.75, min_test_wr: float = 0.70,
                       min_test_pf: float = 1.8, max_test_dd: float = 15.0) -> bool:
    return (
        stats_test['n_trades'] >= min_trades_test
        and stats_train['win_rate'] >= min_train_wr
        and stats_test['win_rate'] >= min_test_wr
        and stats_test['profit_factor'] >= min_test_pf
        and stats_test['max_dd_pct'] <= max_test_dd
    )


# ---------------------------------------------------------------------------
# walk-forward split helper
# ---------------------------------------------------------------------------

def split_train_test(trades: pd.DataFrame,
                     train_end: str = '2023-12-31',
                     test_start: str = '2024-01-01') -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades is None or len(trades) == 0:
        return pd.DataFrame(), pd.DataFrame()
    et = pd.to_datetime(trades['entry_time'])
    train = trades[et <= pd.Timestamp(train_end)].copy()
    test = trades[et >= pd.Timestamp(test_start)].copy()
    return train, test
