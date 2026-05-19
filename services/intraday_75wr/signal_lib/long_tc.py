"""Live Long Trend-Continuation signal (Config A3 / B3).

Continuous scan 09:15-10:30 IST (bars 7-15). Ported from research/37
scripts/11b_long_trend_pullback.py — variant 'vwap_within_0p3'.

Pre-conditions per session:
  - gap_up >= gap_min_pct
  - bar_6 close > day_open
  - first-hour high (bars 0..11) > day_open * (1 + first_hour_strength_pct/100)

Per-bar conditions (bars bar_min..bar_max):
  - pullback (current bar low close to VWAP within 0.3%)
  - bullish bar (close > open)
  - rsi > rsi_floor
  - NIFTY also gap-up + bullish at b6
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

    bar_min = int(cfg.get('a3_bar_min', 7))
    bar_max = int(cfg.get('a3_bar_max', 15))
    last = today_df.iloc[-1]
    bar_idx = int(last['bar_idx'])

    if bar_idx < bar_min or bar_idx > bar_max:
        return None

    gap_min_pct = float(cfg.get('a3_gap_min_pct', 0.5))
    fh_strength_pct = float(cfg.get('a3_first_hour_strength_pct', 0.5))
    pullback_mode = cfg.get('a3_pullback_mode', 'vwap_within_0p3')
    rsi_floor = float(cfg.get('a3_rsi_floor', 45))

    # Pre-conditions
    day_open = float(last['day_open'])
    prev_close = last.get('prev_close')
    if prev_close is None or pd.isna(prev_close):
        return None
    prev_close = float(prev_close)

    gap_pct = (day_open - prev_close) / prev_close * 100.0
    if gap_pct < gap_min_pct:
        return None

    # bar 6 close > day_open
    bar6_row = today_df[today_df['bar_idx'] == 6]
    if bar6_row.empty:
        return None
    if not (float(bar6_row.iloc[0]['close']) > day_open):
        return None

    # first-hour (bars 0..11) high > day_open * (1 + fh_strength/100)
    fh = today_df[today_df['bar_idx'] <= 11]
    if fh.empty:
        return None
    fh_high = float(fh['high'].max())
    if not (fh_high > day_open * (1 + fh_strength_pct / 100)):
        return None

    # Per-bar conditions
    close_p = float(last['close'])
    open_p = float(last['open'])
    low_p = float(last['low'])
    rsi = float(last.get('rsi', 50))
    vwap = float(last.get('vwap', 0))
    ema9 = float(last.get('ema9', 0))

    # Pullback variants
    if pullback_mode == 'vwap_within_0p3':
        pullback_ok = abs(low_p - vwap) / vwap <= 0.003 if vwap > 0 else False
    elif pullback_mode == 'vwap_within_0p5':
        pullback_ok = abs(low_p - vwap) / vwap <= 0.005 if vwap > 0 else False
    elif pullback_mode == 'ema9_touch':
        pullback_ok = (low_p <= ema9) and (close_p >= ema9 * 0.998)
    elif pullback_mode == 'ema9_within_0p2':
        pullback_ok = abs(low_p - ema9) / ema9 <= 0.002 if ema9 > 0 else False
    elif pullback_mode == 'ema9_or_vwap':
        e9_ok = abs(low_p - ema9) / ema9 <= 0.002 if ema9 > 0 else False
        vw_ok = abs(low_p - vwap) / vwap <= 0.003 if vwap > 0 else False
        pullback_ok = e9_ok or vw_ok
    else:
        pullback_ok = False

    if not pullback_ok:
        return None
    if not (close_p > open_p):
        return None
    if rsi <= rsi_floor:
        return None

    # NIFTY filter
    if not _nifty_pass(nifty_ctx, cfg.get('a3_nifty_filter', 'nifty_strong_both')):
        return None

    entry_price = close_p
    sl_pct = float(cfg.get('sl_pct', 1.5))
    tp_pct = float(cfg.get('tp_pct', 0.5))
    sl_price = entry_price * (1 - sl_pct / 100)
    target_price = entry_price * (1 + tp_pct / 100)

    return {
        'fired': True,
        'direction': 'LONG',
        'entry_price': round(entry_price, 2),
        'sl_price': round(sl_price, 2),
        'target_price': round(target_price, 2),
        'meta': {
            'signal': 'long_tc_a3',
            'instrument': instrument,
            'rsi': round(rsi, 2),
            'vwap': round(vwap, 2),
            'gap_pct': round(gap_pct, 3),
            'bar_idx': bar_idx,
            'pullback_mode': pullback_mode,
            'close_at_signal': round(close_p, 2),
            'nifty_filter': cfg.get('a3_nifty_filter'),
        },
    }


def _nifty_pass(ctx: dict, filter_name: str) -> bool:
    if filter_name == 'none' or not ctx:
        return True
    if filter_name == 'nifty_strong_both':
        gap = ctx.get('gap_pct')
        b6 = ctx.get('b6_close')
        d_open = ctx.get('day_open')
        if gap is None or b6 is None or d_open is None:
            return False
        return float(gap) >= 0.1 and float(b6) > float(d_open)
    if filter_name == 'nifty_gap_up':
        gap = ctx.get('gap_pct')
        return gap is not None and float(gap) >= 0.1
    if filter_name == 'first_bullish':
        return bool(ctx.get('first_bullish'))
    return True
