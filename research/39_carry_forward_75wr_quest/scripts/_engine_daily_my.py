"""Daily-bar engine for research/39 patterns 3 (RSI mean-reversion) & 4
(earnings post-drift). Mirror of research/37 _engine.py but on daily timeframe.

This is a *minimal* local engine intentionally named with a `_my` suffix so it
does not collide with the sister sub-agent's `_engine_daily.py`. Once the
master reconciles, the two engines can be merged.

Usage:
    from _engine_daily_my import (
        list_daily_universe, load_daily, load_many_daily,
        enrich_daily, simulate_signals_daily, trade_stats, TradeRules,
        FNO_UNIVERSE, train_test_split,
    )
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DB_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..',
    'backtest_data', 'market_data.db',
))


# F&O universe (subset of services/data_manager.py FNO_LOT_SIZES) — most-liquid 86 names.
FNO_UNIVERSE = [
    'RELIANCE','TCS','HDFCBANK','INFY','ICICIBANK','HINDUNILVR','ITC','SBIN','BHARTIARTL',
    'KOTAKBANK','LT','AXISBANK','ASIANPAINT','MARUTI','TITAN','BAJFINANCE','HCLTECH',
    'SUNPHARMA','ULTRACEMCO','NESTLEIND','WIPRO','ONGC','NTPC','POWERGRID','M&M',
    'TATAMOTORS','TECHM','JSWSTEEL','INDUSINDBK','ADANIPORTS','BAJAJFINSV','HINDALCO',
    'COALINDIA','DIVISLAB','GRASIM','TATACONSUM','DRREDDY','CIPLA','EICHERMOT',
    'BRITANNIA','APOLLOHOSP','HEROMOTOCO','SBILIFE','BPCL','TATASTEEL','BAJAJ-AUTO',
    'HDFCLIFE','SHREECEM','ADANIENT','VEDL','TATAPOWER','GAIL','JINDALSTEL','DLF',
    'GODREJPROP','SIEMENS','HAVELLS','PIDILITIND','DABUR','MARICO','COLPAL',
    'MUTHOOTFIN','CHOLAFIN','BEL','HAL','IOC','IRCTC','COFORGE','PERSISTENT','MCX',
    'CUMMINSIND','VOLTAS','AMBUJACEM','TRENT','ZOMATO','PAYTM','DELHIVERY','PNB',
    'BANKBARODA','FEDERALBNK','IDFCFIRSTB',
]


# Walk-forward split per charter.
TRAIN_START = '2018-01-01'
TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2026-03-19'


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def list_daily_universe(symbols: Optional[List[str]] = None,
                        start: str = '2018-01-01',
                        end: str = '2026-03-19',
                        min_rows: int = 1500) -> List[str]:
    """Return symbols with at least `min_rows` daily candles in window."""
    with _connect() as con:
        if symbols:
            qmarks = ','.join(['?'] * len(symbols))
            df = pd.read_sql(
                f"""SELECT symbol, COUNT(*) AS n
                    FROM market_data_unified
                    WHERE timeframe='day' AND symbol IN ({qmarks})
                          AND date>=? AND date<=?
                    GROUP BY symbol
                    HAVING n >= ?""",
                con, params=(*symbols, start, end + ' 23:59:59', min_rows),
            )
        else:
            df = pd.read_sql(
                """SELECT symbol, COUNT(*) AS n
                   FROM market_data_unified
                   WHERE timeframe='day' AND date>=? AND date<=?
                   GROUP BY symbol
                   HAVING n >= ?""",
                con, params=(start, end + ' 23:59:59', min_rows),
            )
    return df['symbol'].tolist()


def load_daily(symbol: str, start: str = '2017-01-01', end: str = '2026-03-19') -> pd.DataFrame:
    """Load daily OHLCV for one symbol. Index = date (datetime, tz-naive).

    Note: `start` defaults a year earlier than TRAIN_START so 200-SMA / RSI
    have warm-up history.
    """
    with _connect() as con:
        df = pd.read_sql(
            """SELECT date, open, high, low, close, volume
               FROM market_data_unified
               WHERE timeframe='day' AND symbol=? AND date>=? AND date<=?
               ORDER BY date""",
            con, params=(symbol, start, end + ' 23:59:59'),
        )
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    return df


def load_many_daily(symbols: List[str], start: str = '2017-01-01',
                    end: str = '2026-03-19') -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = load_daily(s, start, end)
        if not df.empty:
            out[s] = df
    return out


# ---------------------------------------------------------------------------
# daily primitives
# ---------------------------------------------------------------------------

def add_sma(df: pd.DataFrame, periods=(20, 50, 200)) -> pd.DataFrame:
    for p in periods:
        df[f'sma{p}'] = df['close'].rolling(p, min_periods=p).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi14'] = (100 - 100 / (1 + rs)).fillna(50)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(period, min_periods=1).mean()
    return df


def add_bbands(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma = df['close'].rolling(period, min_periods=period).mean()
    sd = df['close'].rolling(period, min_periods=period).std()
    df['bb_mid'] = ma
    df['bb_lo'] = ma - k * sd
    df['bb_hi'] = ma + k * sd
    return df


def add_volume_avg(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df['vol_avg20'] = df['volume'].rolling(period, min_periods=1).mean()
    df['vol_ratio'] = df['volume'] / df['vol_avg20'].replace(0, np.nan)
    return df


def add_gap(df: pd.DataFrame) -> pd.DataFrame:
    df['prev_close'] = df['close'].shift(1)
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
    df['ret_pct'] = (df['close'] - df['open']) / df['open'] * 100
    df['close_to_close_pct'] = (df['close'] - df['prev_close']) / df['prev_close'] * 100
    return df


def add_n_day_low_high(df: pd.DataFrame, n: int = 60) -> pd.DataFrame:
    """Multi-month low/high markers. n=60 -> ~3 months."""
    df[f'low_{n}'] = df['close'].rolling(n, min_periods=n).min()
    df[f'high_{n}'] = df['close'].rolling(n, min_periods=n).max()
    df[f'is_{n}d_low'] = df['close'] <= df[f'low_{n}']
    df[f'is_{n}d_high'] = df['close'] >= df[f'high_{n}']
    return df


def enrich_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = add_sma(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_bbands(df)
    df = add_volume_avg(df)
    df = add_gap(df)
    df = add_n_day_low_high(df, n=60)
    return df


# ---------------------------------------------------------------------------
# trade simulator (multi-day hold, daily TP/SL on intraday range, time-stop)
# ---------------------------------------------------------------------------

@dataclass
class TradeRules:
    tp_pct: float = 5.0          # take-profit % from entry
    sl_pct: float = 3.0          # stop-loss % from entry
    max_hold_days: int = 10      # time-stop in trading days
    direction: str = 'long'      # 'long' or 'short'
    rsi_exit: Optional[float] = None  # for mean-rev: exit when RSI crosses this level
    cost_round_trip_pct: float = 0.06  # F&O futures default; 0.21 for CNC delivery


def simulate_signals_daily(df: pd.DataFrame,
                           signal: pd.Series,
                           rules: TradeRules,
                           rsi_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """Daily multi-day hold simulator.

    Entry: at signal-day NEXT bar's open (next trading day open).
    Exit on first of: TP hit (intraday), SL hit (intraday), RSI cross (next-day
    open if rsi_exit set), or time-stop (close on max_hold_days bar).

    Cost (`cost_round_trip_pct`) is subtracted from raw return.
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
    rsi_a = rsi_series.to_numpy() if rsi_series is not None else None

    is_long = (rules.direction == 'long')
    tp_mult = (1 + rules.tp_pct / 100) if is_long else (1 - rules.tp_pct / 100)
    sl_mult = (1 - rules.sl_pct / 100) if is_long else (1 + rules.sl_pct / 100)

    rows = []

    for i in sig_idx:
        entry_i = i + 1
        if entry_i >= n:
            continue
        entry_price = open_a[entry_i]
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        tp_price = entry_price * tp_mult
        sl_price = entry_price * sl_mult
        max_exit_i = min(entry_i + rules.max_hold_days, n - 1)

        hi = high_a[entry_i:max_exit_i + 1]
        lo = low_a[entry_i:max_exit_i + 1]
        if is_long:
            sl_hit = lo <= sl_price
            tp_hit = hi >= tp_price
        else:
            sl_hit = hi >= sl_price
            tp_hit = lo <= tp_price

        sl_first = int(np.argmax(sl_hit)) if sl_hit.any() else -1
        tp_first = int(np.argmax(tp_hit)) if tp_hit.any() else -1

        # RSI exit: trigger once rsi crosses level within window — exit at next-day open
        rsi_exit_offset = -1
        if rules.rsi_exit is not None and rsi_a is not None:
            rsi_window = rsi_a[entry_i:max_exit_i + 1]
            if is_long:
                rsi_cross = rsi_window >= rules.rsi_exit
            else:
                rsi_cross = rsi_window <= rules.rsi_exit
            if rsi_cross.any():
                rsi_exit_offset = int(np.argmax(rsi_cross))

        # Pick the EARLIEST of: SL, TP, RSI-cross, time-stop
        candidates = []
        if sl_first >= 0:
            candidates.append((sl_first, 'SL', sl_price))
        if tp_first >= 0:
            candidates.append((tp_first, 'TP', tp_price))
        if rsi_exit_offset >= 0:
            # exit at next bar's open (or current bar close if last)
            ex_i = entry_i + rsi_exit_offset
            ex_next = ex_i + 1
            if ex_next <= max_exit_i:
                candidates.append((rsi_exit_offset + 1, 'RSI', open_a[ex_next]))
            else:
                candidates.append((rsi_exit_offset, 'RSI', close_a[ex_i]))

        # Time-stop fallback
        time_offset = max_exit_i - entry_i
        candidates.append((time_offset, 'TIME', close_a[max_exit_i]))

        # Pick smallest offset; SL/TP intraday tie -> SL first
        candidates.sort(key=lambda x: (x[0], 0 if x[1] == 'SL' else 1))
        offset, exit_reason, exit_price = candidates[0]
        exit_i = entry_i + offset

        if is_long:
            ret_pct = (exit_price - entry_price) / entry_price * 100
        else:
            ret_pct = (entry_price - exit_price) / entry_price * 100

        # Subtract cost
        ret_pct -= rules.cost_round_trip_pct

        rows.append((
            idx_a[entry_i], entry_price, idx_a[exit_i], exit_price,
            exit_reason, offset, ret_pct,
        ))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[
        'entry_time', 'entry_price', 'exit_time', 'exit_price',
        'exit_reason', 'days_held', 'ret_pct',
    ])


def trade_stats(trades: pd.DataFrame, period_years: float = 1.0) -> dict:
    if trades is None or len(trades) == 0:
        return dict(n_trades=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                    profit_factor=0.0, sharpe=0.0, max_dd_pct=0.0,
                    total_return_pct=0.0, trades_per_year=0.0, aws=0.0,
                    avg_days_held=0.0)
    wins = trades['ret_pct'] > 0
    losses = trades['ret_pct'] <= 0
    win_rate = float(wins.mean())
    avg_win = float(trades.loc[wins, 'ret_pct'].mean()) if wins.any() else 0.0
    avg_loss = float(trades.loc[losses, 'ret_pct'].mean()) if losses.any() else 0.0
    gp = float(trades.loc[wins, 'ret_pct'].sum())
    gl = float(-trades.loc[losses, 'ret_pct'].sum())
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    eq = trades['ret_pct'].cumsum()
    peak = eq.cummax()
    dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0.0

    daily_rets = trades.groupby(trades['entry_time'].dt.date)['ret_pct'].sum()
    if len(daily_rets) > 1 and daily_rets.std() > 0:
        sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252))
    else:
        sharpe = 0.0
    total_ret = float(trades['ret_pct'].sum())
    trades_per_year = len(trades) / max(period_years, 0.1)

    aws = (
        win_rate
        * np.log1p(len(trades))
        * min(pf / 2, 1.5 if np.isfinite(pf) else 1.5)
        * min(max(sharpe, 0) / 1.5, 1.5)
    )

    return dict(
        n_trades=int(len(trades)),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=float(pf if np.isfinite(pf) else 999.0),
        sharpe=sharpe,
        max_dd_pct=max_dd,
        total_return_pct=total_ret,
        trades_per_year=float(trades_per_year),
        aws=float(aws),
        avg_days_held=float(trades['days_held'].mean()),
    )


def passes_walk_forward(train: dict, test: dict,
                        min_train_wr: float = 0.75,
                        min_test_wr: float = 0.70,
                        min_test_pf: float = 1.8,
                        max_test_dd: float = 15.0,
                        min_test_n: int = 30) -> bool:
    if train is None or test is None:
        return False
    return (
        train['n_trades'] >= 30
        and train['win_rate'] >= min_train_wr
        and test['n_trades'] >= min_test_n
        and test['win_rate'] >= min_test_wr
        and test['profit_factor'] >= min_test_pf
        and test['max_dd_pct'] <= max_test_dd
    )


def split_by_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[start:end + ' 23:59:59']


def split_signal_by_period(df: pd.DataFrame, signal: pd.Series,
                           start: str, end: str) -> pd.Series:
    """Mask a signal to only fire within the given window."""
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end + ' 23:59:59'))
    return signal & mask


# ---------------------------------------------------------------------------
# convenience: run a strategy across stocks and return per-stock + pooled stats
# ---------------------------------------------------------------------------

def run_strategy_on_cohort(cache: Dict[str, pd.DataFrame],
                           strategy_fn,
                           rules: TradeRules,
                           period_start: str,
                           period_end: str,
                           rsi_col: str = 'rsi14',
                           **strat_kwargs) -> Tuple[pd.DataFrame, dict]:
    """Run a strategy across multiple stocks. Returns (per-stock df, pooled stats).

    `strategy_fn(df, **strat_kwargs)` -> boolean Series aligned to df.
    Trades fired only on bars within [period_start, period_end].
    """
    per_stock = []
    pooled = []
    for sym, df in cache.items():
        if df.empty or len(df) < 250:
            continue
        try:
            sig = strategy_fn(df, **strat_kwargs)
        except Exception as e:
            continue
        sig = split_signal_by_period(df, sig.fillna(False), period_start, period_end)
        rsi_series = df[rsi_col] if rsi_col in df.columns else None
        trades = simulate_signals_daily(df, sig, rules, rsi_series=rsi_series)
        if len(trades) == 0:
            per_stock.append({'symbol': sym, **trade_stats(trades)})
            continue
        trades['symbol'] = sym
        years = max((pd.Timestamp(period_end) - pd.Timestamp(period_start)).days / 365.25, 0.1)
        s = trade_stats(trades, period_years=years)
        per_stock.append({'symbol': sym, **s})
        pooled.append(trades)

    per_df = pd.DataFrame(per_stock)
    if pooled:
        pooled_trades = pd.concat(pooled, ignore_index=True)
        years = max((pd.Timestamp(period_end) - pd.Timestamp(period_start)).days / 365.25, 0.1)
        pooled_stats = trade_stats(pooled_trades, period_years=years)
    else:
        pooled_stats = trade_stats(pd.DataFrame(), period_years=1.0)
    return per_df, pooled_stats
