"""Strategy family library — research/37 Stage 2.

Each strategy is a function (df, **params) -> (signal: pd.Series, direction: str).

`signal` is a boolean Series aligned to df.index where True = enter at the
NEXT bar's open. `direction` is 'long' or 'short'.

Conventions:
- Signals respect time-of-day and skip last 5 bars of session (no late entries).
- Signals require all 5 bars of OR to have happened (bar_idx >= 3).
- One signal per session per stock (collapse to first True per session).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _one_per_session(sig: pd.Series, sessions: pd.Series) -> pd.Series:
    """Reduce signal to first True per session."""
    if not sig.any():
        return sig
    s = sig.copy()
    df = pd.DataFrame({'sig': s, 'session': sessions.values}, index=s.index)
    first = df[df['sig']].groupby('session').head(1).index
    out = pd.Series(False, index=s.index)
    out.loc[first] = True
    return out


def _no_late_entries(sig: pd.Series, df: pd.DataFrame, last_n: int = 5) -> pd.Series:
    """Drop signals in last N bars of session (give time for exit)."""
    bars_per_session = df.groupby('session').size()
    bar_idx = df['bar_idx'].values
    sess = df['session'].values
    max_per_session = pd.Series(bars_per_session, name='m').reindex(df['session']).values
    mask = bar_idx <= (max_per_session - last_n - 1)
    return sig & mask


# ---------------------------------------------------------------------------
# 1. VWAP rejection (long-only) — dip below VWAP then reclaim
# ---------------------------------------------------------------------------
def vwap_rejection(df: pd.DataFrame,
                   max_dip_atr: float = 1.5,
                   reclaim_window: int = 3,
                   rsi_min: float = 40,
                   min_bar_idx: int = 6) -> Tuple[pd.Series, str]:
    """LONG: was below VWAP within last N bars, now closes back above VWAP."""
    above = df['close'] > df['vwap']
    below_recent = (~above).rolling(reclaim_window, min_periods=1).max().astype(bool)
    dip_pct = (df['vwap'] - df['low']) / df['atr'].replace(0, np.nan)
    sig = (
        above
        & below_recent.shift(1).fillna(False)  # was below recently
        & (dip_pct.rolling(reclaim_window, min_periods=1).max() <= max_dip_atr)
        & (df['rsi'] >= rsi_min)
        & (df['bar_idx'] >= min_bar_idx)
    )
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig.fillna(False), 'long'


# ---------------------------------------------------------------------------
# 2. Open Range Breakout (long)
# ---------------------------------------------------------------------------
def or_breakout(df: pd.DataFrame,
                or_minutes: int = 15,
                min_volume_mult: float = 1.0,
                rsi_min: float = 50,
                vwap_align: bool = True) -> Tuple[pd.Series, str]:
    """LONG: close > or_high after OR period ends, with optional vol/vwap/rsi gates."""
    or_bars = max(1, or_minutes // 5)
    after_or = df['bar_idx'] >= or_bars
    breakout = df['close'] > df['or_high']
    vol_avg = df['volume'].rolling(20, min_periods=5).mean()
    vol_ok = df['volume'] >= vol_avg * min_volume_mult
    rsi_ok = df['rsi'] >= rsi_min
    vwap_ok = (df['close'] > df['vwap']) if vwap_align else True
    sig = after_or & breakout & vol_ok & rsi_ok & vwap_ok
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig.fillna(False), 'long'


# ---------------------------------------------------------------------------
# 3. First-candle continuation (Open=Low long)
# ---------------------------------------------------------------------------
def first_candle_open_low(df: pd.DataFrame,
                          confirm_bars: int = 1,
                          rsi_min: float = 55,
                          gap_up_min: float = -0.5) -> Tuple[pd.Series, str]:
    """LONG: first candle has open == low (or very near), confirmed by N bullish bars."""
    g = df.groupby('session')
    first_bar = g.head(1)
    fb = first_bar.copy()
    fb['near_open_low'] = (fb['open'] - fb['low']).abs() / fb['open'] < 0.0005
    sess_ok = set(fb.loc[fb['near_open_low'], 'session'].values.astype('datetime64[ns]').tolist())

    is_first_ok = df['session'].isin(list(sess_ok))
    confirm = (df['close'] > df['open']).rolling(confirm_bars, min_periods=confirm_bars).min().astype(bool)
    if 'prev_close' in df.columns:
        gap_pct = (df['day_open'] - df['prev_close']) / df['prev_close'] * 100
    else:
        gap_pct = pd.Series(0, index=df.index)
    gap_ok = gap_pct >= gap_up_min
    rsi_ok = df['rsi'] >= rsi_min
    sig = is_first_ok & confirm & gap_ok & rsi_ok & (df['bar_idx'] >= confirm_bars) & (df['bar_idx'] <= 6)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig.fillna(False), 'long'


# ---------------------------------------------------------------------------
# 4. EMA9 pullback in trending day
# ---------------------------------------------------------------------------
def ema9_pullback(df: pd.DataFrame,
                  trend_ema: int = 50,
                  pullback_atr: float = 0.5,
                  rsi_min: float = 45,
                  rsi_max: float = 65) -> Tuple[pd.Series, str]:
    """LONG: price > ema_trend, dip to ema9 within pullback_atr, RSI in zone."""
    trend_col = f'ema{trend_ema}'
    if trend_col not in df.columns:
        raise ValueError(f'{trend_col} not in df')
    above_trend = df['close'] > df[trend_col]
    dip = ((df['low'] - df['ema9']).abs() / df['atr'].replace(0, np.nan)) <= pullback_atr
    rsi_ok = (df['rsi'] >= rsi_min) & (df['rsi'] <= rsi_max)
    bullish = df['close'] > df['open']
    sig = above_trend & dip & rsi_ok & bullish & (df['bar_idx'] >= 6)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig.fillna(False), 'long'


# ---------------------------------------------------------------------------
# 5. Time-of-day momentum (09:30-10:00 only)
# ---------------------------------------------------------------------------
def tod_momentum(df: pd.DataFrame,
                 start_bar: int = 3,    # 09:30
                 end_bar: int = 9,      # 10:00
                 rsi_min: float = 55,
                 vwap_align: bool = True,
                 ema_align: bool = True) -> Tuple[pd.Series, str]:
    """LONG: in 09:30-10:00 window, close > VWAP and EMA9 > EMA21 and RSI > min."""
    in_window = (df['bar_idx'] >= start_bar) & (df['bar_idx'] <= end_bar)
    rsi_ok = df['rsi'] >= rsi_min
    vwap_ok = (df['close'] > df['vwap']) if vwap_align else True
    ema_ok = (df['ema9'] > df['ema21']) if ema_align else True
    bullish = df['close'] > df['open']
    sig = in_window & rsi_ok & vwap_ok & ema_ok & bullish
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig.fillna(False), 'long'


# ---------------------------------------------------------------------------
# 6. VWAP mean reversion (extreme deviation -> revert)
# ---------------------------------------------------------------------------
def vwap_revert(df: pd.DataFrame,
                deviation_atr: float = 2.0,
                rsi_max: float = 35,
                direction: str = 'long') -> Tuple[pd.Series, str]:
    """LONG: price > deviation_atr below VWAP and RSI oversold and bullish bar."""
    if direction == 'long':
        far_below = (df['vwap'] - df['close']) / df['atr'].replace(0, np.nan) >= deviation_atr
        rsi_ok = df['rsi'] <= rsi_max
        bullish = df['close'] > df['open']
        sig = far_below & rsi_ok & bullish & (df['bar_idx'] >= 6)
    else:
        far_above = (df['close'] - df['vwap']) / df['atr'].replace(0, np.nan) >= deviation_atr
        rsi_ok = df['rsi'] >= (100 - rsi_max)
        bearish = df['close'] < df['open']
        sig = far_above & rsi_ok & bearish & (df['bar_idx'] >= 6)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig.fillna(False), direction


# ---------------------------------------------------------------------------
# 7. Volume burst + price thrust
# ---------------------------------------------------------------------------
def volume_thrust(df: pd.DataFrame,
                  vol_mult: float = 3.0,
                  thrust_atr: float = 1.0,
                  vwap_align: bool = True) -> Tuple[pd.Series, str]:
    """LONG: volume > N * 20-bar avg AND close - open > thrust_atr * ATR."""
    vol_avg = df['volume'].rolling(20, min_periods=5).mean()
    vol_ok = df['volume'] >= vol_avg * vol_mult
    thrust_ok = (df['close'] - df['open']) >= df['atr'] * thrust_atr
    vwap_ok = (df['close'] > df['vwap']) if vwap_align else True
    sig = vol_ok & thrust_ok & vwap_ok & (df['bar_idx'] >= 3)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig.fillna(False), 'long'


# ---------------------------------------------------------------------------
# 8. RSI extreme (mean reversion)
# ---------------------------------------------------------------------------
def rsi_extreme(df: pd.DataFrame,
                rsi_threshold: float = 25,
                direction: str = 'long',
                require_bullish_bar: bool = True) -> Tuple[pd.Series, str]:
    """LONG: RSI <= threshold then bullish reversal bar."""
    if direction == 'long':
        rsi_low = df['rsi'].shift(1) <= rsi_threshold
        bar = (df['close'] > df['open']) if require_bullish_bar else True
        sig = rsi_low & bar & (df['bar_idx'] >= 6)
    else:
        rsi_hi = df['rsi'].shift(1) >= (100 - rsi_threshold)
        bar = (df['close'] < df['open']) if require_bullish_bar else True
        sig = rsi_hi & bar & (df['bar_idx'] >= 6)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig.fillna(False), direction


# ---------------------------------------------------------------------------
# 9. Inside-bar compression breakout
# ---------------------------------------------------------------------------
def inside_bar_breakout(df: pd.DataFrame,
                        min_inside: int = 2,
                        rsi_min: float = 50,
                        vwap_align: bool = True) -> Tuple[pd.Series, str]:
    """LONG: N consecutive inside bars, then close above the highest of them."""
    is_inside = (df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))
    inside_run = is_inside.rolling(min_inside, min_periods=min_inside).min().astype(bool)
    range_high = df['high'].rolling(min_inside + 1, min_periods=min_inside + 1).max().shift(1)
    breakout = df['close'] > range_high
    rsi_ok = df['rsi'] >= rsi_min
    vwap_ok = (df['close'] > df['vwap']) if vwap_align else True
    sig = inside_run.shift(1).fillna(False) & breakout & rsi_ok & vwap_ok & (df['bar_idx'] >= 6)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig.fillna(False), 'long'


# ---------------------------------------------------------------------------
# 10. Donchian-N breakout
# ---------------------------------------------------------------------------
def donchian_breakout(df: pd.DataFrame,
                      lookback: int = 20,
                      vol_mult: float = 1.0,
                      vwap_align: bool = True) -> Tuple[pd.Series, str]:
    """LONG: close > rolling N-bar high, with volume + vwap filters."""
    dn_high = df['high'].rolling(lookback, min_periods=lookback).max().shift(1)
    breakout = df['close'] > dn_high
    vol_avg = df['volume'].rolling(20, min_periods=5).mean()
    vol_ok = df['volume'] >= vol_avg * vol_mult
    vwap_ok = (df['close'] > df['vwap']) if vwap_align else True
    sig = breakout & vol_ok & vwap_ok & (df['bar_idx'] >= max(lookback, 6))
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig.fillna(False), 'long'


# ---------------------------------------------------------------------------
# 11. EMA9/21 cross with VWAP alignment
# ---------------------------------------------------------------------------
def ema_cross(df: pd.DataFrame,
              fast: int = 9,
              slow: int = 21,
              vwap_align: bool = True,
              rsi_min: float = 50) -> Tuple[pd.Series, str]:
    """LONG: ema_fast crosses above ema_slow, close > vwap, rsi >= min."""
    fast_col, slow_col = f'ema{fast}', f'ema{slow}'
    cross_up = (df[fast_col] > df[slow_col]) & (df[fast_col].shift(1) <= df[slow_col].shift(1))
    vwap_ok = (df['close'] > df['vwap']) if vwap_align else True
    rsi_ok = df['rsi'] >= rsi_min
    sig = cross_up & vwap_ok & rsi_ok & (df['bar_idx'] >= 6)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig.fillna(False), 'long'


# ---------------------------------------------------------------------------
# 12. Gap-and-go (gap up + first hour momentum)
# ---------------------------------------------------------------------------
def gap_and_go(df: pd.DataFrame,
               gap_min_pct: float = 0.5,
               rsi_min: float = 55,
               max_bar_idx: int = 12) -> Tuple[pd.Series, str]:
    """LONG: gap up >= gap_min_pct, in first hour, vwap+ema aligned."""
    if 'prev_close' not in df.columns:
        return pd.Series(False, index=df.index), 'long'
    gap_pct = (df['day_open'] - df['prev_close']) / df['prev_close'] * 100
    gap_up = gap_pct >= gap_min_pct
    rsi_ok = df['rsi'] >= rsi_min
    vwap_ok = df['close'] > df['vwap']
    ema_ok = df['ema9'] > df['ema21']
    bullish = df['close'] > df['open']
    sig = gap_up & rsi_ok & vwap_ok & ema_ok & bullish & (df['bar_idx'] >= 3) & (df['bar_idx'] <= max_bar_idx)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig.fillna(False), 'long'


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
STRATEGIES = {
    'vwap_rejection': vwap_rejection,
    'or_breakout': or_breakout,
    'first_candle_open_low': first_candle_open_low,
    'ema9_pullback': ema9_pullback,
    'tod_momentum': tod_momentum,
    'vwap_revert': vwap_revert,
    'volume_thrust': volume_thrust,
    'rsi_extreme': rsi_extreme,
    'inside_bar_breakout': inside_bar_breakout,
    'donchian_breakout': donchian_breakout,
    'ema_cross': ema_cross,
    'gap_and_go': gap_and_go,
}
