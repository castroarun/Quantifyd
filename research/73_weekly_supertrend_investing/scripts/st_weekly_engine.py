"""
Weekly SuperTrend (10,3) trend-following engine — research/73.

Reuses the TradingView-accurate calc_supertrend from services/technical_indicators.
LONG-ONLY. Causal execution: signal read at weekly (W-FRI) CLOSE of week t,
trade fills at OPEN of week t+1. No look-ahead.

Public API:
    load_daily(conn, symbol) -> daily OHLC DataFrame (date index)
    to_weekly(daily_df)      -> weekly W-FRI OHLC DataFrame
    weekly_st_trades(...)    -> list of trade dicts (entry open_{t+1} -> exit open_{t+1})
    buy_hold_return(...)     -> B&H return of the same name over the window
    placebo_trades(...)      -> random-entry, matched-duration placebo trades
"""
import os, sys, sqlite3, random
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
from services.technical_indicators import calc_supertrend  # noqa: E402

DB_PATH = os.path.join(ROOT, 'backtest_data', 'market_data.db')


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_daily(conn, symbol):
    """Daily OHLC for one symbol, ascending by date. Empty df if none."""
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' ORDER BY date",
        conn, params=(symbol,))
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df[df['close'] > 0]
    df = df.set_index('date')
    return df


def to_weekly(daily):
    """Resample daily -> weekly W-FRI. open=first, high=max, low=min, close=last, volume=sum."""
    if daily.empty:
        return daily
    w = daily.resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                     'close': 'last', 'volume': 'sum'})
    w = w.dropna(subset=['open', 'high', 'low', 'close'])
    return w


def weekly_st_trades(daily, symbol, start=None, end=None, atr_period=10, multiplier=3.0,
                     cost_rt=0.0, min_weeks=30):
    """
    Extract long-only weekly-ST trades. Returns list of dicts.
    Signal at weekly close of week t -> fill at open of week t+1.
    cost_rt = round-trip cost fraction (e.g. 0.003 = 0.30%) applied to gross return.
    An open position at the end is closed at the last weekly close (marked).
    """
    w = to_weekly(daily)
    if len(w) < min_weeks:
        return []
    st, direction = calc_supertrend(w[['high', 'low', 'close']].assign(
        open=w['open']), atr_period, multiplier)
    w = w.copy()
    w['dir'] = direction.values
    w['flip_up'] = (w['dir'] == 1) & (w['dir'].shift(1) == -1)
    w['flip_dn'] = (w['dir'] == -1) & (w['dir'].shift(1) == 1)

    dates = w.index
    opens = w['open'].values
    highs = w['high'].values
    closes = w['close'].values
    flip_up = w['flip_up'].values
    flip_dn = w['flip_dn'].values
    n = len(w)

    trades = []
    i = 0
    while i < n - 1:
        if flip_up[i]:
            entry_i = i + 1            # fill next week open
            entry_px = opens[entry_i]
            entry_dt = dates[entry_i]
            # find next flip_dn strictly after entry signal week
            exit_i = None
            for j in range(i + 1, n):
                if flip_dn[j]:
                    exit_i = j + 1 if j + 1 < n else j  # fill next week open, else mark last close
                    break
            if exit_i is None:
                exit_i = n - 1
                exit_px = closes[exit_i]      # still-open -> mark last close
                exit_dt = dates[exit_i]
                open_at_end = True
            else:
                if exit_i < n:
                    exit_px = opens[exit_i] if exit_i < n else closes[n - 1]
                exit_dt = dates[exit_i]
                open_at_end = False
            # window filter on entry date
            if (start is None or entry_dt >= pd.Timestamp(start)) and \
               (end is None or entry_dt <= pd.Timestamp(end)):
                if entry_px and entry_px > 0:
                    gross = exit_px / entry_px - 1.0
                    net = (exit_px * (1 - cost_rt / 2)) / (entry_px * (1 + cost_rt / 2)) - 1.0
                    trades.append(dict(
                        symbol=symbol, entry_dt=entry_dt, exit_dt=exit_dt,
                        entry_px=float(entry_px), exit_px=float(exit_px),
                        weeks=int(exit_i - entry_i), year=entry_dt.year,
                        gross=float(gross), net=float(net), open_at_end=open_at_end))
            # advance past this trade's exit
            i = exit_i if exit_i > i else i + 1
        else:
            i += 1
    return trades


def buy_hold_return(daily, start=None, end=None):
    """B&H return of the name over the (clipped) window; None if insufficient."""
    w = to_weekly(daily)
    if start is not None:
        w = w[w.index >= pd.Timestamp(start)]
    if end is not None:
        w = w[w.index <= pd.Timestamp(end)]
    if len(w) < 2:
        return None
    return float(w['close'].iloc[-1] / w['open'].iloc[0] - 1.0)


def placebo_trades(daily, symbol, real_trades, start=None, end=None, cost_rt=0.0, seed=42):
    """For each real trade, a random-entry weekly trade of the SAME holding length (weeks)."""
    w = to_weekly(daily)
    if start is not None:
        w = w[w.index >= pd.Timestamp(start)]
    if end is not None:
        w = w[w.index <= pd.Timestamp(end)]
    n = len(w)
    if n < 5 or not real_trades:
        return []
    rng = random.Random(hash(symbol) ^ seed)
    opens = w['open'].values
    closes = w['close'].values
    out = []
    for t in real_trades:
        hold = max(1, t['weeks'])
        if hold >= n - 1:
            continue
        ei = rng.randint(0, n - 1 - hold)
        xi = ei + hold
        ep, xp = opens[ei], closes[xi]
        if ep and ep > 0:
            net = (xp * (1 - cost_rt / 2)) / (ep * (1 + cost_rt / 2)) - 1.0
            out.append(dict(symbol=symbol, weeks=hold, net=float(net),
                            gross=float(xp / ep - 1.0)))
    return out
