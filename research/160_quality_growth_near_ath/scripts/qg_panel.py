# -*- coding: utf-8 -*-
"""research/160 — panel cache builder for the quality-growth-near-ATH engine.

Reads `backtest_data/market_data.db` READ-ONLY, one symbol at a time (the VPS has ~3 GB
free RAM and a live trading process on it; a single 6.8M-row pivot is not safe here), and
writes one .npz of aligned float32 [dates x symbols] frames that the engine memory-maps
lazily.

Every data defect this project has been bitten by is handled HERE, once, so the engine
cannot re-introduce it:

  * phantom holiday rows (volume == 0 and high == low — Kite placeholders) are DROPPED
    before any rolling statistic, so a missing bar can never NaN-poison every window after
    it (the failure that silently disabled r/142's SMA-200 gate from Apr-2026);
  * every rolling statistic is computed on the TRADED rows only and then scattered back to
    the union calendar — the dropna-and-reindex recipe, exactly;
  * the retroactive split defect (pre-split rows kept at the old price scale: MCX, HEG,
    NAZARA, CUPID, ...) is guarded by RESTARTING the all-time-high cummax at any single-day
    close move <= -40%, so history at a stale scale can never set the high the near-ATH
    screen compares against. Events are logged to results/panel_split_events.csv;
  * the data start is INDEPENDENT of any trading start (--base-start, default 2000-01-01).
    An ATH is a cummax over the whole history; deriving the data start from the trading
    start turns it into an N-month high (the r/142 trap);
  * funds are excluded by NAME via backtest_data/etf_exclusions.json (a ticker blacklist
    rots — r/158 found 221 gold/silver/index funds inside an equity book), plus r/158's
    ticker regex as the net for delisted funds, plus every symbol whose volume is zero
    throughout (that is how the NIFTYMIDCAP150 / NIFTYSMLCAP250 / sector index series get
    flagged: they are NOT in etf_exclusions.json but they are not companies either);
  * a partial candle (a refresh during market hours storing an intraday price as the daily
    close) is dropped when the last date is today and the IST clock is before 17:45.

Frames written (all float32, [n_dates x n_syms], NaN where the symbol did not trade):
    open, close        raw cleaned prices
    tv20               20-day MEDIAN traded value (close*volume), min 10 obs
    athc               all-time-high CLOSE through t inclusive, split-restarted
    sma20/50/100/200   simple moving average of close, min_periods = window
    donch_low20/50     min close over [t-W, t-1]  (already shifted: "prior W days")
    score              IBD-style 2*r63 + r126 + r189 + r252 (raw; ordering is identical to
                       its cross-sectional percentile, so no rank frame is stored)
Plus: dates (<U10), syms (<U24), is_fund (bool), and build metadata.

Usage:
    venv/bin/python3 qg_panel.py --base-start 2000-01-01 --out results/panel_2000.npz
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():                                   # laptop smoke-runs
    ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
EXCL = ROOT / 'backtest_data' / 'etf_exclusions.json'
STUDY = Path(__file__).resolve().parents[1]

# r/158's regex, kept only as the net for funds that have delisted and are therefore no
# longer in the instrument dump the name-based list is built from.
ETF_RE = re.compile(r'(BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50)')

SMA_WINDOWS = (20, 50, 100, 200)
DONCH_WINDOWS = (20, 50)
SPLIT_DROP = -0.40          # single-day close move treated as an unadjusted split
MIN_ROWS = 60               # a symbol needs at least this many traded days to be kept


def _fund_set():
    if not EXCL.exists():
        raise SystemExit(
            'missing %s — run research/158_oa_arming_width/scripts/build_etf_list.py. '
            'Falling back to the ticker regex alone would reproduce the defect that let '
            '221 gold/silver/index funds into an equity book.' % EXCL)
    return set(json.load(open(EXCL))['symbols'])


def build(base_start: str, out: Path, db: Path = DB, drop_partial: bool = True):
    t0 = time.time()
    con = sqlite3.connect('file:%s?mode=ro' % db, uri=True)      # READ-ONLY. Never writes.

    dates = [str(r[0])[:10] for r in con.execute(
        "select distinct date from market_data_unified where timeframe='day'")]
    dates = sorted(set(d for d in dates if d >= base_start))
    if drop_partial:
        now_ist = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)
        if dates and dates[-1] == now_ist.strftime('%Y-%m-%d') and now_ist.hour * 60 + \
                now_ist.minute < 17 * 60 + 45:
            print('dropping %s — partial candle (IST %s, official close settles ~17:30-18:00)'
                  % (dates[-1], now_ist.strftime('%H:%M')), flush=True)
            dates = dates[:-1]
    dpos = {d: i for i, d in enumerate(dates)}
    nd = len(dates)
    print('calendar: %d sessions %s .. %s' % (nd, dates[0], dates[-1]), flush=True)

    syms = sorted(r[0] for r in con.execute(
        "select symbol from (select symbol, count(*) n from market_data_unified "
        "where timeframe='day' group by symbol) where n >= %d" % MIN_ROWS))
    print('%d candidate symbols' % len(syms), flush=True)

    funds = _fund_set()
    keys = ['open', 'close', 'tv20', 'athc', 'score'] + \
           ['sma%d' % w for w in SMA_WINDOWS] + ['donch_low%d' % w for w in DONCH_WINDOWS]
    F = {k: np.full((nd, len(syms)), np.nan, dtype=np.float32) for k in keys}
    is_fund = np.zeros(len(syms), dtype=bool)

    split_events, n_phantom, kept = [], 0, []
    for j, s in enumerate(syms):
        df = pd.read_sql_query(
            "select date, open, high, low, close, volume from market_data_unified "
            "where symbol=? and timeframe='day' order by date", con, params=(s,))
        if df.empty:
            continue
        df['date'] = df['date'].astype(str).str[:10]
        df = df.drop_duplicates('date', keep='last')
        df = df[df['date'].isin(dpos)]
        # Phantom placeholder rows: no trade happened, the bar is fabricated. The test only
        # applies to instruments that carry volume AT ALL. An INDEX has volume 0 on every
        # row by construction, and several of them (NIFTYMIDCAP150, NIFTYSMLCAP250) are
        # stored O=H=L=C before 2019 — the naive test deleted 1,990 real index sessions
        # each and silently truncated those benchmarks to 2019. Found by self-test 3a.
        has_vol = float(df['volume'].fillna(0).max()) > 0
        if has_vol:
            ph = (df['volume'].fillna(0) == 0) & (df['high'] == df['low'])
            n_phantom += int(ph.sum())
            df = df[~ph]
        df = df[df['close'] > 0]
        if len(df) < MIN_ROWS:
            continue

        rows = np.fromiter((dpos[d] for d in df['date']), dtype=np.int64, count=len(df))
        c = df['close'].to_numpy(dtype=np.float64)
        o = df['open'].to_numpy(dtype=np.float64)
        v = df['volume'].fillna(0).to_numpy(dtype=np.float64)

        # --- split-guarded all-time-high close (restart the cummax at each event) ---
        ret = np.empty(len(c))
        ret[0] = 0.0
        ret[1:] = c[1:] / c[:-1] - 1.0
        ev = np.nonzero(ret <= SPLIT_DROP)[0]
        athc = np.empty_like(c)
        start = 0
        for e in list(ev) + [len(c)]:
            athc[start:e] = np.maximum.accumulate(c[start:e])
            start = e
        for e in ev:
            split_events.append((s, df['date'].iat[e], round(float(ret[e]) * 100, 1),
                                 round(float(c[e - 1]), 2), round(float(c[e]), 2)))

        cs = pd.Series(c)
        tv20 = (cs * v).rolling(20, min_periods=10).median().to_numpy()
        smas = {w: cs.rolling(w, min_periods=w).mean().to_numpy() for w in SMA_WINDOWS}
        donch = {w: cs.rolling(w, min_periods=w).min().shift(1).to_numpy()
                 for w in DONCH_WINDOWS}

        def _r(n):
            out = np.full(len(c), np.nan)
            if len(c) > n:
                out[n:] = c[n:] / c[:-n] - 1.0
            return out
        score = 2 * _r(63) + _r(126) + _r(189) + _r(252)

        F['open'][rows, j] = o
        F['close'][rows, j] = c
        F['tv20'][rows, j] = tv20
        F['athc'][rows, j] = athc
        F['score'][rows, j] = score
        for w in SMA_WINDOWS:
            F['sma%d' % w][rows, j] = smas[w]
        for w in DONCH_WINDOWS:
            F['donch_low%d' % w][rows, j] = donch[w]

        is_fund[j] = (s in funds) or bool(ETF_RE.search(s)) or (v.max() <= 0)
        kept.append(j)
        if (j + 1) % 400 == 0:
            print('  %d/%d symbols (%.0fs)' % (j + 1, len(syms), time.time() - t0), flush=True)

    con.close()
    keep = np.array(kept, dtype=np.int64)
    syms_k = np.array([syms[i] for i in keep], dtype='<U24')
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: F[k][:, keep] for k in keys}
    payload['dates'] = np.array(dates, dtype='<U10')
    payload['syms'] = syms_k
    payload['is_fund'] = is_fund[keep]
    payload['meta'] = np.array([json.dumps(dict(
        base_start=base_start, built=dt.datetime.now().isoformat(timespec='seconds'),
        db=str(db), n_dates=nd, n_syms=int(len(keep)), n_funds=int(is_fund[keep].sum()),
        n_phantom_rows=n_phantom, n_split_events=len(split_events),
        split_drop=SPLIT_DROP, sma_windows=list(SMA_WINDOWS),
        donch_windows=list(DONCH_WINDOWS)))])   # plain unicode: loads without pickle
    np.savez(out, **payload)

    with open(STUDY / 'results' / 'panel_split_events.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['symbol', 'date', 'pct_move', 'prev_close', 'close'])
        w.writerows(sorted(split_events))

    print('\npanel  : %s  (%.0f MB)' % (out, out.stat().st_size / 1e6))
    print('shape  : %d dates x %d symbols (%d flagged as funds/indices, never tradeable)'
          % (nd, len(keep), int(is_fund[keep].sum())))
    print('phantom: %d placeholder rows dropped before any rolling statistic' % n_phantom)
    print('splits : %d ATH-cummax restarts -> results/panel_split_events.csv'
          % len(split_events))
    print('built in %.0fs' % (time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-start', default='2000-01-01',
                    help='price history start. MUST be independent of the trading start: '
                         'the ATH is a cummax over the whole history.')
    ap.add_argument('--out', default=str(STUDY / 'results' / 'panel_2000.npz'))
    ap.add_argument('--db', default=str(DB))
    ap.add_argument('--keep-partial', action='store_true',
                    help='do not drop a same-day partial candle (diagnostics only)')
    a = ap.parse_args()
    build(a.base_start, Path(a.out), Path(a.db), drop_partial=not a.keep_partial)


if __name__ == '__main__':
    main()
