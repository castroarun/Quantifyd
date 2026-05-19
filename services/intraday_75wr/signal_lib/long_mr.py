"""Live Long Mean-Reversion signal (Config A2 / B2).

Trigger continuously between 11:15-13:15 IST. Ported from research/37
scripts/11c_long_late_reversal.py.

Conditions on the just-closed 5-min bar:
  1. Cumulative drop from day_open <= drop_pct (e.g. -2.0%)
  2. RSI(14) was below rsi_oversold within last rsi_lookback bars
  3. RSI now > rsi_lift
  4. Current bar bullish (close > open)
  5. Current close > 3-bar prior high
  6. NIFTY not crashing (filter)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
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

    drop_pct = float(cfg.get('a2_drop_pct', -2.0))
    rsi_oversold = float(cfg.get('a2_rsi_oversold', 28))
    rsi_lift = float(cfg.get('a2_rsi_lift', 35))
    rsi_lookback = int(cfg.get('a2_rsi_lookback_bars', 6))
    require_3bar_break = bool(cfg.get('a2_require_3bar_break', True))
    require_bullish_bar = bool(cfg.get('a2_require_bullish_bar', True))

    last = today_df.iloc[-1]
    bar_idx = int(last['bar_idx'])

    # Window check (11:15 = bar 24, 13:15 = bar 48)
    if bar_idx < 24 or bar_idx > 48:
        return None
    # Don't enter in last 5 bars of session
    if bar_idx >= 70:
        return None

    close_p = float(last['close'])
    open_p = float(last['open'])
    day_open = float(last['day_open'])
    rsi = float(last.get('rsi', 50))

    # 1. drop from day_open
    drop_now = (close_p - day_open) / day_open * 100.0
    if drop_now > drop_pct:
        return None

    # 2. RSI was oversold in last `rsi_lookback` bars
    rsi_window = today_df['rsi'].tail(rsi_lookback)
    if not (rsi_window < rsi_oversold).any():
        return None

    # 3. RSI now > lift
    if rsi <= rsi_lift:
        return None

    # 4. bullish bar
    if require_bullish_bar and not (close_p > open_p):
        return None

    # 5. 3-bar break - close > rolling 3-bar high of *prior* bars
    if require_3bar_break:
        if len(today_df) < 4:
            return None
        prior_high3 = today_df['high'].iloc[-4:-1].max()
        if not (close_p > prior_high3):
            return None

    # 6. NIFTY not crashing
    if not _nifty_pass(nifty_ctx, cfg.get('a2_nifty_filter', 'b3_not_crashing')):
        return None

    entry_price = close_p
    sl_pct = float(cfg.get('sl_pct', 1.5))
    tp_pct = float(cfg.get('tp_pct', 0.5))
    sl_price = entry_price * (1 - sl_pct / 100)        # LONG: SL below entry
    target_price = entry_price * (1 + tp_pct / 100)    # LONG: TP above entry

    return {
        'fired': True,
        'direction': 'LONG',
        'entry_price': round(entry_price, 2),
        'sl_price': round(sl_price, 2),
        'target_price': round(target_price, 2),
        'meta': {
            'signal': 'long_mr_a2',
            'instrument': instrument,
            'rsi': round(rsi, 2),
            'drop_pct_from_day_open': round(drop_now, 2),
            'bar_idx': bar_idx,
            'close_at_signal': round(close_p, 2),
            'day_open': round(day_open, 2),
            'nifty_filter': cfg.get('a2_nifty_filter'),
        },
    }


def _nifty_pass(ctx: dict, filter_name: str) -> bool:
    if filter_name == 'none' or not ctx:
        return True
    if filter_name == 'b3_not_crashing':
        b3 = ctx.get('b3_change_pct')
        return b3 is None or float(b3) > -0.5
    if filter_name == 'b3_change_neg':
        b3 = ctx.get('b3_change_pct')
        return b3 is not None and float(b3) < -0.1
    if filter_name == 'first_bearish':
        return bool(ctx.get('first_bearish'))
    return True
