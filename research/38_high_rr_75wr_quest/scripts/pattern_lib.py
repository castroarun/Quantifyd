"""Pattern signal library for research/38 — tight-SL / wide-TP intraday patterns.

Each function returns a boolean pd.Series aligned to df.index where True means
"enter at NEXT bar's open" — the convention used by simulate_signals() in
research/37 _engine.py.

Each pattern targets the structural feature that creates a *tight* invalidation:
 - Inside bar: stop at parent bar low/high (often <0.5%)
 - NR4 / NR7: stop at narrow-range bar low/high
 - Failed breakout: stop at the false-break extreme (small wick = tight stop)
 - VWAP V-reversal: stop at the rejection wick low/high
 - Compression: stop at low-vol cluster low/high
 - Multi-bar confirmation: stop at the swing low/high formed within the pattern

ASCII-only console output (Windows cp1252 safe).
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

# bring in research/37 _engine helpers (one-per-session, no-late-entries)
_R37 = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    '37_intraday_75wr_quest', 'scripts',
))
if _R37 not in sys.path:
    sys.path.insert(0, _R37)
from _strategies import _one_per_session, _no_late_entries  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Inside-bar breakout
# ---------------------------------------------------------------------------

def inside_bar_breakout(df: pd.DataFrame,
                        direction: str = 'long',
                        min_bar_idx: int = 4,
                        max_bar_idx: int = 60,
                        rsi_min: float = 50,
                        rsi_max: float = 75,
                        require_vwap_align: bool = True,
                        require_volume_lift: bool = False,
                        ) -> pd.Series:
    """Inside-bar followed by directional breakout.

    Setup (long):
      - Bar t-1 = inside (high <= prior_high AND low >= prior_low)
      - Bar t closes above prior_high (the breakout)
      - Optional: close > VWAP, RSI in [rsi_min, rsi_max]

    Stop is at the inside bar's low (-> tight, often 0.3%-0.6%).
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    bar_idx = df['bar_idx'].values
    sess = df['session'].values
    rsi = df['rsi'].values
    vwap = df['vwap'].values
    volume = df['volume'].values

    # parent = bar t-2 (the bar BEFORE the inside)
    p_high = pd.Series(high).shift(2).values
    p_low = pd.Series(low).shift(2).values
    # inside = bar t-1
    i_high = pd.Series(high).shift(1).values
    i_low = pd.Series(low).shift(1).values
    # same session
    s2 = pd.Series(sess).shift(2).values
    s1 = pd.Series(sess).shift(1).values
    same_sess = (s2 == sess) & (s1 == sess)

    inside = (i_high <= p_high) & (i_low >= p_low) & same_sess

    if direction == 'long':
        breakout = close > i_high
        vwap_ok = (close > vwap) if require_vwap_align else np.ones_like(close, dtype=bool)
        rsi_ok = (rsi >= rsi_min) & (rsi <= rsi_max)
    else:
        breakout = close < i_low
        vwap_ok = (close < vwap) if require_vwap_align else np.ones_like(close, dtype=bool)
        rsi_ok = (rsi <= rsi_max) & (rsi >= rsi_min)

    if require_volume_lift:
        # current bar volume > 1.2x median(20)
        v20 = pd.Series(volume).rolling(20, min_periods=5).median().values
        vol_ok = volume > 1.2 * v20
    else:
        vol_ok = np.ones_like(close, dtype=bool)

    in_window = (bar_idx >= min_bar_idx) & (bar_idx <= max_bar_idx)
    cond = inside & breakout & vwap_ok & rsi_ok & vol_ok & in_window
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig


# ---------------------------------------------------------------------------
# 2. NR4 / NR7 breakout
# ---------------------------------------------------------------------------

def nr_breakout(df: pd.DataFrame,
                direction: str = 'long',
                nr_len: int = 4,
                min_bar_idx: int = 4,
                max_bar_idx: int = 60,
                rsi_min: float = 50,
                rsi_max: float = 75,
                require_vwap_align: bool = True,
                ) -> pd.Series:
    """NR4 / NR7 — narrowest range bar of the last N bars triggers breakout.

    NR4 = current range is the narrowest of the last 4 bars (NR7 = last 7).

    Setup (long): bar t-1 was an NR4/NR7 bar; bar t closes above NR bar's high.
    Stop = NR bar's low (very tight, often 0.2%-0.5%).
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    bar_idx = df['bar_idx'].values
    sess = df['session'].values
    rsi = df['rsi'].values
    vwap = df['vwap'].values

    rng = high - low
    rng_s = pd.Series(rng)

    # NR bar at t-1: range[t-1] is min of rolling N range ending at t-1
    rng_min = rng_s.shift(1).rolling(nr_len, min_periods=nr_len).min()
    is_nr = rng_s.shift(1) <= rng_min  # equality (since shift().rolling() includes self)

    # parent NR bar values
    nr_high = pd.Series(high).shift(1).values
    nr_low = pd.Series(low).shift(1).values

    # session continuity
    s_back = pd.Series(sess).shift(nr_len - 1).values
    same_sess = s_back == sess

    if direction == 'long':
        breakout = close > nr_high
        vwap_ok = (close > vwap) if require_vwap_align else np.ones_like(close, dtype=bool)
    else:
        breakout = close < nr_low
        vwap_ok = (close < vwap) if require_vwap_align else np.ones_like(close, dtype=bool)

    rsi_ok = (rsi >= rsi_min) & (rsi <= rsi_max)
    in_window = (bar_idx >= min_bar_idx) & (bar_idx <= max_bar_idx)

    cond = is_nr.values & breakout & vwap_ok & rsi_ok & in_window & same_sess
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig


# ---------------------------------------------------------------------------
# 3. Failed-breakout reversal (false break of OR or prior-day high/low)
# ---------------------------------------------------------------------------

def failed_breakout_reversal(df: pd.DataFrame,
                             direction: str = 'short',
                             level: str = 'or_high',
                             min_bar_idx: int = 4,
                             max_bar_idx: int = 60,
                             rsi_max: float = 75,
                             rsi_min: float = 25,
                             require_wick_back: bool = True,
                             ) -> pd.Series:
    """Failed breakout / fake-out fade.

    For SHORT (direction='short'):
      - Within last `look` bars, high pierced level (or_high / day_high / prev_close+x)
      - But current close is back below the level (the trap closed)
      - Stop = the wick high above the level (typically 0.2%-0.5%)
      - Target = VWAP / midline (often 1.0%-2.0% below)

    For LONG: mirrored against or_low / day_low.
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    bar_idx = df['bar_idx'].values
    rsi = df['rsi'].values

    if level == 'or_high':
        lvl = df['or_high'].values
    elif level == 'or_low':
        lvl = df['or_low'].values
    elif level == 'day_high':
        lvl = df['day_high'].values
    elif level == 'day_low':
        lvl = df['day_low'].values
    else:
        raise ValueError(f'unknown level: {level}')

    # has the level been pierced in the last 5 bars?
    if direction == 'short':
        pierced = pd.Series(high > lvl).rolling(5, min_periods=1).max().astype(bool).values
        back_below = close < lvl
        cond = pierced & back_below
    else:
        pierced = pd.Series(low < lvl).rolling(5, min_periods=1).max().astype(bool).values
        back_above = close > lvl
        cond = pierced & back_above

    # require RSI in tradable zone (not extreme exhaustion)
    rsi_ok = (rsi >= rsi_min) & (rsi <= rsi_max)
    in_window = (bar_idx >= min_bar_idx) & (bar_idx <= max_bar_idx)

    if require_wick_back:
        # current bar is closing in lower (short) / upper (long) part of own range
        own_rng = (high - low)
        with np.errstate(invalid='ignore', divide='ignore'):
            close_pos = np.where(own_rng > 0, (close - low) / own_rng, 0.5)
        if direction == 'short':
            wick_ok = close_pos < 0.5
        else:
            wick_ok = close_pos > 0.5
    else:
        wick_ok = np.ones_like(close, dtype=bool)

    cond = cond & rsi_ok & in_window & wick_ok & np.isfinite(lvl)
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig


# ---------------------------------------------------------------------------
# 4. VWAP V-reversal (sharp dip below VWAP, immediate reclaim)
# ---------------------------------------------------------------------------

def vwap_v_reversal(df: pd.DataFrame,
                    direction: str = 'long',
                    extension_atr: float = 1.0,
                    reclaim_window: int = 3,
                    min_bar_idx: int = 6,
                    max_bar_idx: int = 60,
                    rsi_min: float = 30,
                    rsi_max: float = 70,
                    ) -> pd.Series:
    """VWAP V-shape reversal (long version).

    Long setup:
      - Within last `reclaim_window` bars, low went >= extension_atr * ATR
        below VWAP (sharp extension)
      - Current close > VWAP (reclaim)
      - Current bar bullish (close > open)
      - RSI bouncing (not still falling)
    Stop at the V-bottom low (the rejection wick).
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    bar_idx = df['bar_idx'].values
    rsi = df['rsi'].values
    vwap = df['vwap'].values
    atr = df['atr'].values

    if direction == 'long':
        # how far did the low extend BELOW VWAP, in ATRs
        ext = (vwap - low) / np.where(atr > 0, atr, np.nan)
        ext_recent = pd.Series(ext).rolling(reclaim_window, min_periods=1).max().values
        sharp_dip = ext_recent >= extension_atr
        reclaim = close > vwap
        bullish = close > open_
    else:
        ext = (high - vwap) / np.where(atr > 0, atr, np.nan)
        ext_recent = pd.Series(ext).rolling(reclaim_window, min_periods=1).max().values
        sharp_dip = ext_recent >= extension_atr
        reclaim = close < vwap
        bullish = close < open_

    rsi_ok = (rsi >= rsi_min) & (rsi <= rsi_max)
    in_window = (bar_idx >= min_bar_idx) & (bar_idx <= max_bar_idx)

    cond = sharp_dip & reclaim & bullish & rsi_ok & in_window
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig


# ---------------------------------------------------------------------------
# 5. Compression breakout (low-vol cluster -> directional breakout)
# ---------------------------------------------------------------------------

def compression_breakout(df: pd.DataFrame,
                         direction: str = 'long',
                         compression_window: int = 6,
                         compression_pct: float = 0.6,  # cluster range <= 0.6% of price
                         min_bar_idx: int = 6,
                         max_bar_idx: int = 60,
                         rsi_min: float = 50,
                         rsi_max: float = 75,
                         require_vwap_align: bool = True,
                         ) -> pd.Series:
    """Volatility-compression breakout.

    Setup (long):
      - Over the last `compression_window` bars (excl current), the range
        (max-high - min-low) was <= compression_pct% of the cluster's mid price.
      - Current close > the cluster's max-high.
    Stop = the cluster's min-low (often <0.4%).
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    bar_idx = df['bar_idx'].values
    rsi = df['rsi'].values
    vwap = df['vwap'].values
    sess = df['session'].values

    # rolling cluster: bars t-compression_window..t-1
    h_s = pd.Series(high)
    l_s = pd.Series(low)
    cluster_high = h_s.shift(1).rolling(compression_window, min_periods=compression_window).max()
    cluster_low = l_s.shift(1).rolling(compression_window, min_periods=compression_window).min()
    cluster_mid = (cluster_high + cluster_low) / 2.0
    cluster_pct = (cluster_high - cluster_low) / cluster_mid * 100.0

    # session continuity
    s_back = pd.Series(sess).shift(compression_window).values
    same_sess = s_back == sess

    is_compressed = cluster_pct <= compression_pct

    if direction == 'long':
        breakout = close > cluster_high.values
        vwap_ok = (close > vwap) if require_vwap_align else np.ones_like(close, dtype=bool)
    else:
        breakout = close < cluster_low.values
        vwap_ok = (close < vwap) if require_vwap_align else np.ones_like(close, dtype=bool)

    rsi_ok = (rsi >= rsi_min) & (rsi <= rsi_max)
    in_window = (bar_idx >= min_bar_idx) & (bar_idx <= max_bar_idx)

    cond = is_compressed.values & breakout & vwap_ok & rsi_ok & in_window & same_sess
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig


# ---------------------------------------------------------------------------
# 6. Multi-bar confirmation entry (3-bar trend ignition)
# ---------------------------------------------------------------------------

def multi_bar_confirm(df: pd.DataFrame,
                      direction: str = 'long',
                      n_bars: int = 3,
                      min_bar_idx: int = 4,
                      max_bar_idx: int = 60,
                      rsi_min: float = 50,
                      rsi_max: float = 75,
                      require_vwap_align: bool = True,
                      require_higher_lows: bool = True,
                      ) -> pd.Series:
    """3-bar (n-bar) directional confirmation.

    Long setup:
      - Last n bars all bullish (close > open)
      - Each successive bar's low >= prior bar's low (higher-lows) [optional]
      - Current close > VWAP, RSI in [50,75]
    Stop = lowest low of the n bars (tight when bars are small, ~0.4%-0.6%).
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    bar_idx = df['bar_idx'].values
    rsi = df['rsi'].values
    vwap = df['vwap'].values
    sess = df['session'].values

    if direction == 'long':
        bull = close > open_
    else:
        bull = close < open_

    bull_s = pd.Series(bull.astype(int))
    consec_bull = bull_s.rolling(n_bars, min_periods=n_bars).sum() == n_bars

    if require_higher_lows:
        if direction == 'long':
            l_diff = pd.Series(low).diff().values  # low[t] - low[t-1]
            hl_ok = pd.Series(l_diff >= 0).rolling(n_bars, min_periods=n_bars).sum() == n_bars
        else:
            h_diff = pd.Series(high).diff().values
            hl_ok = pd.Series(h_diff <= 0).rolling(n_bars, min_periods=n_bars).sum() == n_bars
    else:
        hl_ok = pd.Series(np.ones(len(df), dtype=bool))

    s_back = pd.Series(sess).shift(n_bars - 1).values
    same_sess = s_back == sess

    if direction == 'long':
        vwap_ok = (close > vwap) if require_vwap_align else np.ones_like(close, dtype=bool)
    else:
        vwap_ok = (close < vwap) if require_vwap_align else np.ones_like(close, dtype=bool)
    rsi_ok = (rsi >= rsi_min) & (rsi <= rsi_max)
    in_window = (bar_idx >= min_bar_idx) & (bar_idx <= max_bar_idx)

    cond = consec_bull.values & hl_ok.values & vwap_ok & rsi_ok & in_window & same_sess
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig


# ---------------------------------------------------------------------------
# 7. Stop-run / liquidity-grab reversal
# ---------------------------------------------------------------------------

def stop_run_reversal(df: pd.DataFrame,
                      direction: str = 'short',
                      lookback: int = 12,
                      pierce_pct: float = 0.05,
                      min_bar_idx: int = 6,
                      max_bar_idx: int = 60,
                      rsi_min: float = 30,
                      rsi_max: float = 80,
                      ) -> pd.Series:
    """Liquidity-grab reversal.

    Short setup:
      - Current high pierced the high of the last `lookback` bars (excl current)
        by at least pierce_pct % (sweep above prior swing-high)
      - But current close is back below the prior swing-high (the sweep failed)
    Stop = current bar's high (the false swing) — tight (~0.3%-0.6%).
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    bar_idx = df['bar_idx'].values
    rsi = df['rsi'].values

    h_s = pd.Series(high)
    l_s = pd.Series(low)
    prior_high = h_s.shift(1).rolling(lookback, min_periods=lookback).max().values
    prior_low = l_s.shift(1).rolling(lookback, min_periods=lookback).min().values

    if direction == 'short':
        # sweep: current high > prior_high * (1 + pierce_pct/100)
        # but close back below prior_high
        sweep = high > prior_high * (1 + pierce_pct / 100.0)
        rejected = close < prior_high
        cond = sweep & rejected & np.isfinite(prior_high)
    else:
        sweep = low < prior_low * (1 - pierce_pct / 100.0)
        rejected = close > prior_low
        cond = sweep & rejected & np.isfinite(prior_low)

    rsi_ok = (rsi >= rsi_min) & (rsi <= rsi_max)
    in_window = (bar_idx >= min_bar_idx) & (bar_idx <= max_bar_idx)

    cond = cond & rsi_ok & in_window
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

PATTERN_FUNCS = {
    'inside_bar':       inside_bar_breakout,
    'nr_breakout':      nr_breakout,
    'failed_breakout':  failed_breakout_reversal,
    'vwap_v':           vwap_v_reversal,
    'compression':      compression_breakout,
    'multi_bar':        multi_bar_confirm,
    'stop_run':         stop_run_reversal,
}
