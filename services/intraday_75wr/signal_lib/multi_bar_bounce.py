"""Live Multi-Bar SHORT Bounce signal (Config C).

Continuous scan during the session. Ported from research/38
scripts/pattern_lib.py multi_bar_confirm + the FINAL_LIVE_SETUP.md spec:

  1. Last N=4 consecutive 5-min bars all bearish (close < open)
  2. Each successive bar has lower high than prior bar (lower_highs)
  3. Close < intraday VWAP (own stock)
  4. RSI(14) <= 55
  5. NIFTY 50 close < its own session VWAP at scan time (broad-market weak)
  6. bar_idx >= 4 and not in last 5 bars of session
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from . import _indicators

logger = logging.getLogger(__name__)


def evaluate(
    df_5m: pd.DataFrame,
    *,
    instrument: str,
    today: pd.Timestamp,
    cfg: dict,
    nifty_ctx: dict,
) -> Optional[dict]:
    if df_5m is None or df_5m.empty:
        return None

    df = _indicators.enrich(df_5m.copy())
    today_date = pd.Timestamp(today).date()
    today_df = df[df['session'] == today_date].copy()
    if today_df.empty:
        return None

    n_bars = int(cfg.get('n_bars_consecutive', 4))
    require_lower_highs = bool(cfg.get('require_lower_highs', True))
    rsi_max = float(cfg.get('rsi_max', 55))
    require_below_vwap = bool(cfg.get('require_below_vwap', True))
    nifty_below_own_vwap = bool(cfg.get('nifty_below_own_vwap_required', True))
    min_bar_idx = int(cfg.get('min_bar_idx', 4))
    last_bars_skip = int(cfg.get('last_bars_skip', 5))

    if len(today_df) < n_bars:
        return None
    last = today_df.iloc[-1]
    bar_idx = int(last['bar_idx'])
    if bar_idx < min_bar_idx:
        return None

    # Skip last 5 bars (assume 75 bars/session of 5-min intervals)
    if bar_idx >= 75 - last_bars_skip:
        return None

    # 1. Last n_bars all bearish
    tail = today_df.tail(n_bars)
    if not (tail['close'] < tail['open']).all():
        return None

    # 2. Lower highs across the n bars
    if require_lower_highs:
        highs = tail['high'].values
        if not all(highs[i + 1] <= highs[i] for i in range(len(highs) - 1)):
            return None

    close_p = float(last['close'])
    open_p = float(last['open'])
    rsi = float(last.get('rsi', 50))
    vwap = float(last.get('vwap', 0))

    # 3. close < own VWAP
    if require_below_vwap and not (close_p < vwap):
        return None

    # 4. RSI <= rsi_max
    if rsi > rsi_max:
        return None

    # 5. NIFTY < its own VWAP
    if nifty_below_own_vwap:
        nifty_close = nifty_ctx.get('current_close')
        nifty_vwap = nifty_ctx.get('current_vwap')
        if nifty_close is None or nifty_vwap is None:
            return None
        if not (float(nifty_close) < float(nifty_vwap)):
            return None

    entry_price = close_p
    sl_pct = float(cfg.get('sl_pct', 1.0))
    tp_pct = float(cfg.get('tp_pct', 1.5))
    sl_price = entry_price * (1 + sl_pct / 100)        # SHORT: SL above entry
    target_price = entry_price * (1 - tp_pct / 100)    # SHORT: TP below entry

    return {
        'fired': True,
        'direction': 'SHORT',
        'entry_price': round(entry_price, 2),
        'sl_price': round(sl_price, 2),
        'target_price': round(target_price, 2),
        'meta': {
            'signal': 'multi_bar_bounce_c',
            'instrument': instrument,
            'rsi': round(rsi, 2),
            'vwap': round(vwap, 2),
            'close_at_signal': round(close_p, 2),
            'bar_idx': bar_idx,
            'n_bars_bearish': n_bars,
            'nifty_close': nifty_ctx.get('current_close'),
            'nifty_vwap': nifty_ctx.get('current_vwap'),
        },
    }
