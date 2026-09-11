# -*- coding: utf-8 -*-
"""research/162 — derived frames the r/160 engine does not carry.

r/160's panel has close / open / tv20 / athc / SMAs / Donchian lows / RS score. It has no
HIGH and no LOW, so it cannot express an ATR-scaled trailing exit — which is exactly the
exit family research/161 found was worth +11.85pp of CAGR on all-time-high entries. This
builds, once, everything research/162 needs on top of that panel, aligned to the SAME date
and symbol axes so the engine can index them interchangeably:

  st_7_3, st_10_3, st_14_4, st_20_3   bool  True on a day the SuperTrend direction is DOWN
                                            (the close-based "get out today" signal). The
                                            recursion is copied verbatim from r/161's
                                            bt_core.supertrend_dir so the two studies are
                                            directly comparable.
  ch_22_2, ch_22_3                    bool  True when close < (22-day highest HIGH minus
                                            k x Wilder ATR14). Rolling form, not the
                                            since-entry form, so the line is path-independent.
  vol60                               f32   sample sigma of daily simple returns over the
                                            last 60 TRADED sessions (inverse-vol weights)
  profit_g3, opm_slope3               f32   point-in-time ranking fields, forward-filled
                                            from the monthly decision dates

Conventions kept identical to the panel builder so nothing new can leak in:
  * phantom placeholder rows (volume == 0 AND high == low, on instruments that carry volume
    at all) are dropped BEFORE any rolling statistic;
  * every statistic is computed on the TRADED rows only and scattered back to the union
    calendar (the dropna-and-reindex recipe) — a missing session can never NaN-poison the
    windows after it;
  * the retroactive split defect is handled the r/160 way: the series is truncated to the
    bars AFTER the last single-day close move <= -40%, so a pre-split price scale can never
    manufacture a trail break. Days before the cut carry NO exit signal (False), which is
    the safe direction;
  * funds/indices (panel.is_fund) are skipped entirely.

READ-ONLY on backtest_data/*.db. Writes one npz into research/162/results/.
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
R160 = ROOT / 'research' / '160_quality_growth_near_ath' / 'results'
OUT_DEFAULT = Path(__file__).resolve().parents[1] / 'results' / 'aux_162.npz'

ST_SPECS = [(7, 3.0), (10, 3.0), (14, 4.0), (20, 3.0)]
CH_SPECS = [(22, 2.0), (22, 3.0)]
SPLIT_DROP = -0.40
MIN_ROWS = 60
VOL_WIN = 60


def supertrend_down(high, low, close, period, mult):
    """r/161 bt_core.supertrend_dir, verbatim; returns True where direction == -1."""
    n = len(close)
    out = np.zeros(n, dtype=bool)
    if n <= period + 2:
        return out
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
    atr = np.full(n, np.nan)
    atr[period - 1] = np.nanmean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    hl2 = (high + low) / 2.0
    up, dn = hl2 + mult * atr, hl2 - mult * atr
    fu, fl = np.copy(up), np.copy(dn)
    d = np.ones(n, dtype=np.int8)
    for i in range(period + 1, n):
        fu[i] = up[i] if (up[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = dn[i] if (dn[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        d[i] = (-1 if close[i] < fl[i] else 1) if d[i - 1] == 1 else (1 if close[i] > fu[i] else -1)
    d[:period + 1] = 1
    return d == -1


def wilder_atr(high, low, close, period):
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False,
                             min_periods=period).mean().to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default=str(R160 / 'panel_2000.npz'))
    ap.add_argument('--out', default=str(OUT_DEFAULT))
    ap.add_argument('--limit', type=int, default=0, help='smoke test: first N symbols')
    a = ap.parse_args()

    t0 = time.time()
    z = np.load(a.panel)
    dates = [str(x)[:10] for x in np.asarray(z['dates'])]
    syms = [str(s) for s in np.asarray(z['syms'])]
    is_fund = np.asarray(z['is_fund']).astype(bool)
    del z
    nd, ns = len(dates), len(syms)
    dpos = {d: i for i, d in enumerate(dates)}
    print('panel axes: %d dates %s..%s, %d symbols (%d funds skipped)'
          % (nd, dates[0], dates[-1], ns, int(is_fund.sum())), flush=True)

    keys_bool = ['st_%d_%g' % (p, m) for p, m in ST_SPECS] + \
                ['ch_%d_%g' % (w, k) for w, k in CH_SPECS]
    B = {k: np.zeros((nd, ns), dtype=bool) for k in keys_bool}
    VOL = np.full((nd, ns), np.nan, dtype=np.float32)

    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    todo = [(j, s) for j, s in enumerate(syms) if not is_fund[j]]
    if a.limit:
        todo = todo[:a.limit]
    n_cut = 0
    for c, (j, s) in enumerate(todo, 1):
        d = pd.read_sql_query(
            'select date, open, high, low, close, volume from market_data_unified '
            "where symbol=? and timeframe='day' order by date", con, params=(s,))
        if d.empty:
            continue
        d['date'] = d['date'].astype(str).str[:10]
        d = d.drop_duplicates('date', keep='last')
        d = d[d['date'].isin(dpos)]
        has_vol = float(pd.to_numeric(d['volume'], errors='coerce').fillna(0).max() or 0) > 0
        if has_vol:
            ph = (pd.to_numeric(d['volume'], errors='coerce').fillna(0) == 0) & \
                 (d['high'] == d['low'])
            d = d[~ph]
        d = d[(d['close'] > 0) & d['high'].notna() & d['low'].notna()]
        if len(d) < MIN_ROWS:
            continue
        cl = d['close'].to_numpy(np.float64)
        # split guard: keep only the bars AFTER the last unadjusted-split collapse
        r1 = np.empty(len(cl))
        r1[0] = 0.0
        r1[1:] = cl[1:] / cl[:-1] - 1.0
        hit = np.nonzero(r1 <= SPLIT_DROP)[0]
        if len(hit):
            d = d.iloc[int(hit.max()):]
            n_cut += 1
            if len(d) < MIN_ROWS:
                continue
            cl = d['close'].to_numpy(np.float64)
        hi = d['high'].to_numpy(np.float64)
        lo = d['low'].to_numpy(np.float64)
        rows = np.fromiter((dpos[x] for x in d['date']), dtype=np.int64, count=len(d))

        for p, m in ST_SPECS:
            B['st_%d_%g' % (p, m)][rows, j] = supertrend_down(hi, lo, cl, p, m)
        atr14 = wilder_atr(hi, lo, cl, 14)
        for w, k in CH_SPECS:
            hh = pd.Series(hi).rolling(w, min_periods=w).max().to_numpy()
            line = hh - k * atr14
            B['ch_%d_%g' % (w, k)][rows, j] = np.where(np.isfinite(line), cl < line, False)
        rr = pd.Series(cl).pct_change()
        VOL[rows, j] = rr.rolling(VOL_WIN, min_periods=int(VOL_WIN * 0.6)).std().to_numpy()

        if c % 250 == 0:
            el = time.time() - t0
            print('  [%d/%d] %.0fs  ETA %.0fs' % (c, len(todo), el, el / c * (len(todo) - c)),
                  flush=True)
    con.close()
    print('price frames done in %.0fs (%d symbols split-truncated)'
          % (time.time() - t0, n_cut), flush=True)

    # ---------------------------------------------------------------- ranking frames ---
    RANK = {}
    src = R160 / 'features_pit_monthly.csv.gz'
    if src.exists():
        p = pd.read_csv(src)
        p = p[p['symbol'].isin(set(syms))]
        mdates = np.array(sorted(p['date'].astype(str).unique()))
        mi = {d: i for i, d in enumerate(mdates)}
        sj = {s: i for i, s in enumerate(syms)}
        r = p['date'].astype(str).map(mi).to_numpy()
        cc = p['symbol'].map(sj).to_numpy()
        row = np.searchsorted(mdates, np.array(dates), side='right') - 1
        have = row >= 0
        for fld in ('profit_g3', 'opm_slope3'):
            g = np.full((len(mdates), ns), np.nan, dtype=np.float32)
            g[r, cc] = pd.to_numeric(p[fld], errors='coerce').to_numpy(np.float32)
            daily = g[np.clip(row, 0, None)]
            daily[~have] = np.nan
            RANK[fld] = daily
            print('rank frame %s: %d monthly rows, %.1f%% finite on the daily frame'
                  % (fld, len(mdates), 100.0 * np.isfinite(daily).mean()), flush=True)
    else:
        print('!! %s missing — ranking frames NOT built; A2 fundamental ranks unavailable'
              % src, flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, dates=np.array(dates), syms=np.array(syms),
                        vol60=VOL, **B, **RANK)
    print('wrote %s (%.0f MB) in %.0fs'
          % (out, out.stat().st_size / 1e6, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
