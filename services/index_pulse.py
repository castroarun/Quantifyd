"""
Index Pulse — the data layer for the /app/indices page.

Three endpoints' worth of shapes from one module:
  strip()      -> every index: last, day change, 5-day sparkline
  compare()    -> rebased (growth-of-1) series for a window, absolute or
                  relative-to-Nifty50 (RS mode)
  drilldown()  -> for one index: top constituents by 20d turnover (an honest
                  weightage PROXY — real index weights are free-float mcap,
                  which we don't have; labelled as such in the UI) plus
                  leaders/laggards over the window.

Constituent sources: official CSVs in backtest_data/ for Nifty50 / Next50 /
Midcap150 / Smallcap250 / Nifty200, data/nifty500_list.csv for Nifty 500,
and its Industry column for the sector indices.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DB = REPO / 'backtest_data' / 'market_data.db'

INDEX_META = [
    # (symbol, label, group)
    ('NIFTY50',        'Nifty 50',        'broad'),
    ('NIFTYNEXT50',    'Next 50',         'broad'),
    ('NIFTYMIDCAP150', 'Midcap 150',      'broad'),
    ('NIFTYSMLCAP250', 'Smallcap 250',    'broad'),
    ('NIFTY500',       'Nifty 500',       'broad'),
    ('BANKNIFTY',      'Bank',            'sector'),
    ('NIFTYFINSRV',    'Fin Services',    'sector'),
    ('NIFTYIT',        'IT',              'sector'),
    ('NIFTYPHARMA',    'Pharma',          'sector'),
    ('NIFTYAUTO',      'Auto',            'sector'),
    ('NIFTYMETAL',     'Metal',           'sector'),
    ('NIFTYFMCG',      'FMCG',            'sector'),
    ('NIFTYENERGY',    'Energy',          'sector'),
    ('NIFTYREALTY',    'Realty',          'sector'),
    ('NIFTYPSUBANK',   'PSU Bank',        'sector'),
    ('INDIAVIX',       'India VIX',       'vol'),
]

WINDOWS = {'1w': 7, '1m': 22, '3m': 66, '6m': 132, '1y': 252, '3y': 756}

# constituent lists
_LISTS = {
    'NIFTY50':        REPO / 'backtest_data' / 'nifty50_official.csv',
    'NIFTYNEXT50':    REPO / 'backtest_data' / 'niftynext50_official.csv',
    'NIFTYMIDCAP150': REPO / 'backtest_data' / 'niftymidcap150_official.csv',
    'NIFTYSMLCAP250': REPO / 'backtest_data' / 'niftysmallcap250_official.csv',
    'NIFTY500':       REPO / 'data' / 'nifty500_list.csv',
}

# sector index -> Industry values in data/nifty500_list.csv
_SECTOR_MAP = {
    'BANKNIFTY':    ['FINANCIAL SERVICES'],
    'NIFTYFINSRV':  ['FINANCIAL SERVICES'],
    'NIFTYIT':      ['IT'],
    'NIFTYPHARMA':  ['PHARMA'],
    'NIFTYAUTO':    ['AUTOMOBILE'],
    'NIFTYMETAL':   ['METALS'],
    'NIFTYFMCG':    ['CONSUMER GOODS'],
    'NIFTYENERGY':  ['ENERGY'],
    'NIFTYREALTY':  ['CONSTRUCTION'],
    'NIFTYPSUBANK': ['FINANCIAL SERVICES'],
}


def _conn():
    return sqlite3.connect(f'file:{DB}?mode=ro', uri=True)


@lru_cache(maxsize=8)
def _constituents(index_sym: str) -> List[str]:
    p = _LISTS.get(index_sym)
    if p and p.exists():
        out = []
        with p.open() as f:
            for r in csv.DictReader(f):
                sym = (r.get('Symbol') or r.get('symbol') or '').strip()
                if sym:
                    out.append(sym)
        return out
    inds = _SECTOR_MAP.get(index_sym)
    if inds:
        out = []
        with (_LISTS['NIFTY500']).open() as f:
            for r in csv.DictReader(f):
                if (r.get('Industry') or '').strip().upper() in inds:
                    out.append(r['Symbol'].strip())
        return out
    return []


def _index_frame(days: int = 800) -> pd.DataFrame:
    syms = [m[0] for m in INDEX_META]
    con = _conn()
    try:
        df = pd.read_sql_query(
            "SELECT symbol, date, close FROM market_data_unified "
            "WHERE timeframe='day' AND symbol IN (%s) "
            "ORDER BY date" % ','.join('?' * len(syms)),
            con, params=syms)
    finally:
        con.close()
    df['date'] = df['date'].str[:10]
    wide = df.pivot_table(index='date', columns='symbol', values='close',
                          aggfunc='last').sort_index()
    return wide.tail(days)


def strip() -> Dict[str, Any]:
    wide = _index_frame(days=30)
    out = []
    for sym, label, grp in INDEX_META:
        if sym not in wide.columns:
            continue
        s = wide[sym].dropna()
        if len(s) < 2:
            continue
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        spark = [round(float(x), 2) for x in s.tail(6).tolist()]
        out.append({
            'symbol': sym, 'label': label, 'group': grp,
            'last': round(last, 2),
            'chg_pct': round((last / prev - 1) * 100, 2),
            'spark': spark,
            'asof': s.index[-1],
        })
    return {'indices': out, 'generated': datetime.now().isoformat(timespec='seconds')}


def compare(window: str = '3m', mode: str = 'abs',
            symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    n = WINDOWS.get(window, 66)
    wide = _index_frame(days=n + 10)
    wide = wide.tail(n)
    picked = symbols or [m[0] for m in INDEX_META if m[2] != 'vol']
    series = {}
    for sym in picked:
        if sym not in wide.columns:
            continue
        s = wide[sym].dropna()
        if len(s) < 5:
            continue
        if mode == 'rs' and sym != 'NIFTY50' and 'NIFTY50' in wide.columns:
            base = wide['NIFTY50'].reindex(s.index).ffill()
            s = s / base
        s = s / s.iloc[0]
        series[sym] = [round(float(x), 4) for x in s.tolist()]
    dates = [d for d in wide.index]
    # window return per index for the leaderboard
    perf = []
    for sym, vals in series.items():
        label = next((m[1] for m in INDEX_META if m[0] == sym), sym)
        perf.append({'symbol': sym, 'label': label,
                     'ret_pct': round((vals[-1] - 1) * 100, 2)})
    perf.sort(key=lambda x: -x['ret_pct'])
    return {'window': window, 'mode': mode, 'dates': dates,
            'series': series, 'perf': perf}


def drilldown(index_sym: str, window: str = '1m') -> Dict[str, Any]:
    label = next((m[1] for m in INDEX_META if m[0] == index_sym), index_sym)
    cons = _constituents(index_sym)
    if not cons:
        return {'error': f'no constituent list for {index_sym}'}
    n = WINDOWS.get(window, 22)
    con = _conn()
    try:
        df = pd.read_sql_query(
            "SELECT symbol, date, close, volume FROM market_data_unified "
            "WHERE timeframe='day' AND symbol IN (%s) AND date >= date('now','-%d day') "
            "ORDER BY date" % (','.join('?' * len(cons)), int(n * 2.2 + 30)),
            con, params=cons)
    finally:
        con.close()
    if df.empty:
        return {'error': 'no bars for constituents'}
    df['date'] = df['date'].str[:10]
    rows = []
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('date')
        c = g['close'].to_numpy(float)
        v = g['volume'].to_numpy(float)
        if len(c) < 5:
            continue
        lookback = min(n, len(c) - 1)
        turn_cr = float(np.median(c[-20:] * v[-20:])) / 1e7 if len(c) >= 20 else 0.0
        rows.append({
            'symbol': sym,
            'last': round(float(c[-1]), 2),
            'chg_1d_pct': round(float(c[-1] / c[-2] - 1) * 100, 2),
            'ret_window_pct': round(float(c[-1] / c[-1 - lookback] - 1) * 100, 2),
            'turnover_cr': round(float(turn_cr), 1),
        })
    total_turn = sum(r['turnover_cr'] for r in rows) or 1.0
    for r in rows:
        r['weight_proxy_pct'] = round(r['turnover_cr'] / total_turn * 100, 2)
    by_weight = sorted(rows, key=lambda r: -r['turnover_cr'])[:10]
    by_ret = sorted(rows, key=lambda r: -r['ret_window_pct'])
    return {
        'index': index_sym, 'label': label, 'window': window,
        'n_constituents': len(cons), 'n_with_data': len(rows),
        'weight_note': ('weightage shown is a 20-day turnover share PROXY — '
                        'true index weights are free-float market cap, which '
                        'we do not store'),
        'top_by_weight': by_weight,
        'leaders': by_ret[:10],
        'laggards': by_ret[-10:][::-1],
    }
