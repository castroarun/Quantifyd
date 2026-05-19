"""Live Diamond Short signal (used by Config A1 and Config B1).

Trigger fires once per session at the 09:45 IST scan (bar 6, after first 30 min).
Mechanics ported from research/37 scripts/08_diamond_short_with_nifty.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, time
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
    """Live Diamond Short signal eval at the 09:45 scan.

    Returns the signal dict if conditions are met right now (current bar
    closed = bar_idx 6 = 09:45). The caller is responsible for calling this
    only at 09:45 +/- the entry window.
    """
    if df_5m is None or df_5m.empty:
        return None

    df = _indicators.enrich(df_5m.copy())

    # Today's session
    today_date = pd.Timestamp(today).date()
    today_df = df[df['session'] == today_date]
    if today_df.empty or len(today_df) < 6:
        return None

    # bar 6 = 09:45 IST close
    bar6 = today_df[today_df['bar_idx'] == 6]
    if bar6.empty:
        # If we have a row at bar_idx 6 use it; if scan tick fired slightly late
        # (>=6) the latest available row is acceptable.
        bar6 = today_df[today_df['bar_idx'] >= 6].head(1)
        if bar6.empty:
            return None
    last = bar6.iloc[0]

    rsi_thr = float(cfg.get('a1_rsi_threshold', 40))
    require_below_vwap = bool(cfg.get('a1_require_below_vwap', True))

    rsi = float(last.get('rsi', 50))
    close_p = float(last['close'])
    vwap = float(last.get('vwap', 0))

    # Per-stock conditions
    if rsi >= rsi_thr:
        return None
    if require_below_vwap and not (close_p < vwap):
        return None

    # NIFTY regime gate (frozen at 09:45 by ctx)
    if not _nifty_pass(nifty_ctx, cfg.get('a1_nifty_filter', 'b3_change_neg')):
        return None

    # Compute entry / SL / TP. Live entry is at NEXT bar's open;
    # we use the just-closed bar's close as the projected entry price.
    entry_price = close_p
    sl_pct = float(cfg.get('sl_pct', 1.5))
    tp_pct = float(cfg.get('tp_pct', 0.5))
    sl_price = entry_price * (1 + sl_pct / 100)         # SHORT: SL above entry
    target_price = entry_price * (1 - tp_pct / 100)     # SHORT: TP below entry

    return {
        'fired': True,
        'direction': 'SHORT',
        'entry_price': round(entry_price, 2),
        'sl_price': round(sl_price, 2),
        'target_price': round(target_price, 2),
        'meta': {
            'signal': 'diamond_short_a1',
            'instrument': instrument,
            'rsi': round(rsi, 2),
            'vwap': round(vwap, 2),
            'close_at_signal': round(close_p, 2),
            'bar_idx': int(last['bar_idx']),
            'nifty_filter': cfg.get('a1_nifty_filter'),
            'nifty_b3_change_pct': nifty_ctx.get('b3_change_pct'),
            'nifty_below_vwap_b3': nifty_ctx.get('below_vwap_b3'),
            'nifty_first_bearish': nifty_ctx.get('first_bearish'),
        },
    }


def _nifty_pass(ctx: dict, filter_name: str) -> bool:
    """Mirror of research/37 build_nifty_regime + filter_by_nifty.
    True when NIFTY regime satisfies the requested filter."""
    if filter_name == 'none' or not ctx:
        return True
    if filter_name == 'below_vwap_b3':
        return bool(ctx.get('below_vwap_b3'))
    if filter_name == 'first_bearish':
        return bool(ctx.get('first_bearish'))
    if filter_name == 'both':
        return bool(ctx.get('below_vwap_b3')) and bool(ctx.get('first_bearish'))
    if filter_name == 'b3_change_neg':
        b3 = ctx.get('b3_change_pct')
        return b3 is not None and float(b3) < -0.1
    if filter_name == 'b3_change_neg_strong':
        b3 = ctx.get('b3_change_pct')
        return b3 is not None and float(b3) < -0.3
    return True
