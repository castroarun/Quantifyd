"""Shared 5-min indicator helpers for live signal evaluation.

Mirrors the research/37 _engine.py logic so live signals match backtest
behaviour. Inputs assume a single-stock 5-min OHLCV DataFrame indexed by
datetime, columns: open, high, low, close, volume.

Adds columns IN-PLACE: bar_idx, session, vwap, rsi, day_open, prev_close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_session(df: pd.DataFrame) -> pd.DataFrame:
    """Add `session` (date) and `bar_idx` (0-based 5-min bar within session)."""
    if 'session' not in df.columns:
        df['session'] = pd.to_datetime(df.index).date
    if 'bar_idx' not in df.columns:
        df['bar_idx'] = df.groupby('session').cumcount()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Intraday VWAP that resets each session."""
    typical = (df['high'] + df['low'] + df['close']) / 3.0
    pv = typical * df['volume']
    cpv = pv.groupby(df['session']).cumsum()
    cv = df['volume'].groupby(df['session']).cumsum().replace(0, np.nan)
    df['vwap'] = (cpv / cv).fillna(typical)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder RSI(14) on close."""
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = (100 - 100 / (1 + rs)).fillna(50)
    return df


def add_ema9(df: pd.DataFrame) -> pd.DataFrame:
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    return df


def add_session_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_open (per session) + prev_close (per session)."""
    df['day_open'] = df.groupby('session')['open'].transform('first')
    last_close = df.groupby('session')['close'].last()
    pc = last_close.shift(1).rename('prev_close')
    if 'prev_close' in df.columns:
        df = df.drop(columns=['prev_close'])
    df = df.merge(pc, left_on='session', right_index=True, how='left')
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """One-shot enrichment: bar_idx + session + vwap + rsi + ema9 + day_open."""
    df = _ensure_session(df)
    df = add_vwap(df)
    df = add_rsi(df)
    df = add_ema9(df)
    df = add_session_levels(df)
    return df
