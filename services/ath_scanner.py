"""
Breakout scanner — new all-time highs, ATH+volume spikes, and consolidation
breakouts, evaluated live, at the last close (EOD), or T-1 / T-2 / T-3.

Design notes
------------
* Two bulk SQL reads instead of a query per symbol: the prior all-time-high
  close per symbol, and the last ~70 daily bars for everyone. The rest is
  pandas, so a 1600-name scan stays fast.
* "All-time" means the highest close in our stored history; the response
  reports that start date so the claim stays honest.
* Volume ratios are reported against BOTH the 10- and 20-day averages, with
  20 as the default filter. A 10-day window is contaminated by the very
  quiet stretch that precedes a breakout, so 1.5x over 10 days is much
  easier to clear than it looks; 20 days is steadier and matches the
  research/71 breakout work.
* LIVE mode pro-rates volume: intraday volume is compared against the 20-day
  average scaled by the fraction of the session elapsed, so a 10:00 spike is
  not judged against a whole day's average.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
MARKET_DB = REPO / 'backtest_data' / 'market_data.db'

SESSION_START = dtime(9, 15)
SESSION_END = dtime(15, 30)
SESSION_MINUTES = 375

DEFAULTS = {
    'vol_mult': 1.5,
    'vol_window': 20,
    'consol_days': 12,
    'consol_max_range_pct': 8.0,
    'min_turnover_cr': 1.0,
    'min_price': 20.0,
}


def _universe(name: str) -> List[str]:
    from services.eod_breakout_scanner import (
        _load_fno_universe, _load_nifty500_universe, _load_smallcap_universe)
    name = (name or 'nifty500').lower()
    if name == 'fno':
        return _load_fno_universe()
    if name == 'smallcap':
        return _load_smallcap_universe()
    if name == 'all':
        con = sqlite3.connect(f'file:{MARKET_DB}?mode=ro', uri=True)
        try:
            rows = con.execute(
                "SELECT DISTINCT symbol FROM market_data_unified "
                "WHERE timeframe='day'").fetchall()
        finally:
            con.close()
        return sorted(r[0] for r in rows)
    return _load_nifty500_universe()


def _trading_dates(limit: int = 12) -> List[str]:
    con = sqlite3.connect(f'file:{MARKET_DB}?mode=ro', uri=True)
    try:
        rows = con.execute(
            "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
            "ORDER BY date DESC LIMIT ?", (limit,)).fetchall()
    finally:
        con.close()
    return [r[0][:10] for r in rows]


def _recent_bars(symbols: List[str], upto: str, days: int = 70) -> pd.DataFrame:
    con = sqlite3.connect(f'file:{MARKET_DB}?mode=ro', uri=True)
    try:
        dates = [r[0][:10] for r in con.execute(
            "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
            "AND date<=? ORDER BY date DESC LIMIT ?", (upto + ' 23:59', days))]
        if not dates:
            return pd.DataFrame()
        floor = min(dates)
        df = pd.read_sql_query(
            "SELECT symbol, date, open, high, low, close, volume "
            "FROM market_data_unified WHERE timeframe='day' "
            "AND date>=? AND date<=? ORDER BY symbol, date",
            con, params=(floor, upto + ' 23:59'))
    finally:
        con.close()
    if df.empty:
        return df
    df['date'] = df['date'].str[:10]
    df = df[df['symbol'].isin(set(symbols))]
    return df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]


def _prior_ath(symbols: List[str], before: str) -> Dict[str, float]:
    con = sqlite3.connect(f'file:{MARKET_DB}?mode=ro', uri=True)
    try:
        rows = con.execute(
            "SELECT symbol, MAX(close) FROM market_data_unified "
            "WHERE timeframe='day' AND date<? GROUP BY symbol",
            (before,)).fetchall()
    finally:
        con.close()
    keep = set(symbols)
    return {s: v for s, v in rows if s in keep and v}


def _history_start() -> str:
    con = sqlite3.connect(f'file:{MARKET_DB}?mode=ro', uri=True)
    try:
        r = con.execute(
            "SELECT MIN(date) FROM market_data_unified "
            "WHERE timeframe='day'").fetchone()
    finally:
        con.close()
    return (r[0] or '')[:10]


def _live_quotes(symbols: List[str]) -> Dict[str, dict]:
    from services.kite_service import get_kite_with_refresh
    kite = get_kite_with_refresh()
    out: Dict[str, dict] = {}
    keys = [f'NSE:{s}' for s in symbols]
    for i in range(0, len(keys), 400):
        try:
            q = kite.quote(keys[i:i + 400])
        except Exception as e:
            logger.warning('[scanner] quote batch failed: %s', e)
            continue
        for k, v in q.items():
            out[k.split(':', 1)[1]] = v
    return out


def _session_fraction(now: Optional[datetime] = None) -> float:
    now = now or datetime.now()
    t = now.time()
    if t <= SESSION_START:
        return 0.02
    if t >= SESSION_END:
        return 1.0
    elapsed = (now.hour * 60 + now.minute) - (9 * 60 + 15)
    return max(0.02, min(1.0, elapsed / SESSION_MINUTES))


def scan(asof: str = 'eod', universe: str = 'nifty500',
         vol_mult: float = None, vol_window: int = None,
         consol_days: int = None, consol_max_range_pct: float = None,
         min_turnover_cr: float = None) -> Dict[str, Any]:
    """asof: live | eod | t1 | t2 | t3   (t1 = one session before the last close)."""
    vol_mult = float(vol_mult or DEFAULTS['vol_mult'])
    vol_window = int(vol_window or DEFAULTS['vol_window'])
    consol_days = int(consol_days or DEFAULTS['consol_days'])
    consol_max = float(consol_max_range_pct or DEFAULTS['consol_max_range_pct'])
    min_turn = float(min_turnover_cr if min_turnover_cr is not None
                     else DEFAULTS['min_turnover_cr'])

    syms = _universe(universe)
    sessions = _trading_dates(12)
    if not sessions:
        return {'error': 'no daily bars in store'}

    live = (asof == 'live')
    offset = {'eod': 0, 't1': 1, 't2': 2, 't3': 3}.get(asof, 0)
    asof_date = sessions[0] if live else sessions[min(offset, len(sessions) - 1)]

    hist_upto = sessions[0] if live else asof_date
    bars = _recent_bars(syms, hist_upto, days=70)
    if bars.empty:
        return {'error': 'no bars for this universe'}

    quotes = _live_quotes(syms) if live else {}
    sess_frac = _session_fraction() if live else 1.0
    prior_ath = _prior_ath(syms, (sessions[0] + ' 23:59') if live else asof_date)

    ath_rows: List[dict] = []
    consol_rows: List[dict] = []

    for sym, g in bars.groupby('symbol', sort=False):
        g = g.sort_values('date')
        if len(g) < max(consol_days + 2, 22):
            continue

        if live:
            q = quotes.get(sym)
            if not q:
                continue
            ohlc = q.get('ohlc') or {}
            cur = float(q.get('last_price') or 0)
            day_high = float(ohlc.get('high') or cur)
            vol_today = float(q.get('volume_traded') or q.get('volume') or 0)
            prev = g
            ref_date = 'LIVE ' + datetime.now().strftime('%H:%M')
        else:
            row = g[g['date'] == asof_date]
            if row.empty:
                continue
            cur = float(row['close'].iloc[0])
            day_high = float(row['high'].iloc[0])
            vol_today = float(row['volume'].iloc[0])
            prev = g[g['date'] < asof_date]
            ref_date = asof_date

        if cur < DEFAULTS['min_price'] or len(prev) < 21:
            continue

        v = prev['volume'].to_numpy(float)
        c = prev['close'].to_numpy(float)
        h = prev['high'].to_numpy(float)
        l = prev['low'].to_numpy(float)

        avg10 = float(np.mean(v[-10:]))
        avg20 = float(np.mean(v[-20:]))
        turnover_cr = float(np.median(c[-20:] * v[-20:])) / 1e7
        if turnover_cr < min_turn:
            continue

        d10 = avg10 * sess_frac if live else avg10
        d20 = avg20 * sess_frac if live else avg20
        r10 = round(vol_today / d10, 2) if d10 > 0 else None
        r20 = round(vol_today / d20, 2) if d20 > 0 else None
        ratio = r20 if vol_window == 20 else r10
        spike = bool(ratio is not None and ratio >= vol_mult)
        prev_close = float(c[-1])

        pa = prior_ath.get(sym)
        if pa and cur > pa:
            ath_rows.append({
                'symbol': sym,
                'price': round(cur, 2),
                'day_high': round(day_high, 2),
                'prior_ath': round(pa, 2),
                'pct_above_prior_ath': round((cur / pa - 1) * 100, 2),
                'chg_pct': round((cur / prev_close - 1) * 100, 2),
                'vol_x_10d': r10,
                'vol_x_20d': r20,
                'vol_spike': spike,
                'turnover_cr': round(turnover_cr, 2),
                'asof': ref_date,
            })

        if len(h) >= consol_days:
            box_h = float(np.max(h[-consol_days:]))
            box_l = float(np.min(l[-consol_days:]))
            base = box_h if box_h > 0 else 1.0
            range_pct = (box_h - box_l) / base * 100
            if cur > box_h and range_pct <= consol_max:
                consol_rows.append({
                    'symbol': sym,
                    'price': round(cur, 2),
                    'box_high': round(box_h, 2),
                    'box_low': round(box_l, 2),
                    'box_days': consol_days,
                    'range_pct': round(range_pct, 2),
                    'pct_above_box': round((cur / box_h - 1) * 100, 2),
                    'chg_pct': round((cur / prev_close - 1) * 100, 2),
                    'vol_x_10d': r10,
                    'vol_x_20d': r20,
                    'vol_spike': spike,
                    'turnover_cr': round(turnover_cr, 2),
                    'asof': ref_date,
                })

    key = 'vol_x_20d' if vol_window == 20 else 'vol_x_10d'
    ath_rows.sort(key=lambda r: -(r[key] or 0))
    consol_rows.sort(key=lambda r: -(r[key] or 0))

    return {
        'asof': asof,
        'asof_date': ('LIVE ' + datetime.now().strftime('%d-%b %H:%M')
                      if live else asof_date),
        'session_fraction': round(sess_frac, 3) if live else 1.0,
        'universe': universe,
        'universe_size': len(syms),
        'scanned': int(bars['symbol'].nunique()),
        'history_since': _history_start(),
        'params': {
            'vol_mult': vol_mult, 'vol_window': vol_window,
            'consol_days': consol_days, 'consol_max_range_pct': consol_max,
            'min_turnover_cr': min_turn,
        },
        'sessions': sessions[:5],
        'ath': ath_rows,
        'ath_with_volume': [r for r in ath_rows if r['vol_spike']],
        'consolidation_breakout': consol_rows,
        'consolidation_with_volume': [r for r in consol_rows if r['vol_spike']],
    }
