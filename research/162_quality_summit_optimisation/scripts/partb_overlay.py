# -*- coding: utf-8 -*-
"""research/162 Part B — does a point-in-time quality screen help INSIDE Open Alpha · Base Age?

This is research/160's closing recommendation, registered there as a review dated
10-Oct-2026 and brought forward to tonight. r/160 showed the Quality Summit screen is a
poor STANDALONE book; the open question it could not answer is whether the same screen is
useful as an **entry filter** on a book that already works.

The book is r/161's winner, unchanged: a new all-time-high CLOSE whose previous all-time
high is at least 60 bars old and which fell at least 20% below it in between, 16 slots at
6.25% of NAV on Rs 10 lakh, next-open fills, SuperTrend(14,4) close trail, no hard stop,
25 bps a side, after tax, idle cash 5.5%, liquidity >= Rs 2 cr, 60-bar re-arm, 30 selection
seeds. `bt_core162.py` is a byte-identical copy of r/161's engine.

The ONLY change is one line in the event filter: a candidate is dropped unless its symbol
passes the eligibility mask on the SIGNAL day (the mask row in force is the most recent
1st-of-month row at or before that date). Exits, sizing, slot contention and costs are
untouched, and the mask is applied BEFORE the 60-bar re-arm — r/161's own convention,
because which all-time-high close counts as "first" depends on which ones the filter lets
through.

Every cell runs with missing = fail AND missing = pass, because the gap between them is the
coverage bias measured rather than assumed. The 2005-2026 arm is reported but labelled: the
fundamentals panel starts in 2015 and is only usable from Aug-2018, so everything before
that is decided by the missing policy alone.

Resume-safe: a cell already present in results/partB_cells.csv is skipped.
"""
import json
import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt_core162 as B                                                     # noqa: E402

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
R160M = ROOT / 'research/160_quality_growth_near_ath/results/masks_study'
R161 = ROOT / 'research/161_ath_base_age_breakout/results'
RES = Path(__file__).resolve().parents[1] / 'results'
R162M = RES / 'masks162'

SEEDS = list(range(1, 31))
REARM = 60
SPEC = dict(X=60, dep=20.0, K=0.0, sau=0, liq=2.0, ex='ST_14_4', hard=False)
COST = 25.0

WINDOWS = {'full18': ('2018-08-01', '2026-09-30'),
           'W1': ('2018-08-01', '2022-06-30'),
           'W2': ('2022-07-01', '2026-09-30'),
           'long': ('2005-01-03', '2026-09-30')}

MASKS = [('control', None),
         ('b7_g10_qual_mc', R160M / 'b7_g10_qual_mc.npz'),
         ('b3_qual_mc', R160M / 'b3_qual_mc.npz'),
         ('growth_only', R162M / 'growth_only.npz'),
         ('arun_strict', R162M / 'arun_strict.npz')]

FIELDS = ['cell', 'mask', 'missing', 'window', 'n_events', 'seeds', 'cagr_med',
          'cagr_worst', 'cagr_best', 'maxdd_med', 'maxdd_worst', 'calmar_med',
          'sharpe_med', 'trades', 'trades_per_yr', 'win_rate', 'avg_win', 'avg_loss',
          'expectancy', 'max_loss_streak', 'avg_pct_invested']


def rearm(df):
    keep = []
    for _, g in df.sort_values(['symbol', 'hist_bars']).groupby('symbol', sort=False):
        last = -10 ** 9
        for idx, hb in zip(g.index, g['hist_bars'].to_numpy()):
            if hb - last >= REARM:
                keep.append(idx)
                last = hb
    return df.loc[keep]


def mask_lookup(path, missing):
    """-> fn(symbol, 'YYYY-MM-DD') -> bool, under the given missing policy."""
    z = np.load(path)
    md = np.array([str(x)[:10] for x in z['dates']])
    order = np.argsort(md)
    md = md[order]
    mk = np.asarray(z['mask']).astype(bool)[order]
    col = {str(s): i for i, s in enumerate(z['cols'])}
    default = (missing == 'pass')

    def fn(sym, day):
        j = col.get(sym)
        if j is None:
            return default
        r = int(np.searchsorted(md, day, side='right')) - 1
        if r < 0:
            return default
        return bool(mk[r, j])
    return fn


def invested_series(trades, panel, n):
    """bt_core does not record the invested fraction; reconstruct it from the trade list so
    a cell whose screen starves the book cannot quote an idle-cash return as a result."""
    mv = np.zeros(n)
    for t in trades:
        i0, i1 = panel.pos.get(t['entry_date']), panel.pos.get(t['exit_date'])
        if i0 is None:
            continue
        i1 = n - 1 if i1 is None else i1
        c = panel.close[t['symbol']]
        seg = c[i0:i1]
        mv[i0:i1] += np.where(np.isfinite(seg), seg, 0.0) * t['shares']
    return mv


def main():
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    out_cells = RES / 'partB_cells.csv'
    out_pairs = RES / 'partB_paired.csv'
    done = set()
    if out_cells.exists():
        done = set(pd.read_csv(out_cells)['cell'].astype(str))
        print('resuming: %d cells already done' % len(done), flush=True)
    else:
        pd.DataFrame(columns=FIELDS).to_csv(out_cells, index=False)

    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    cal = [d for d in pd.read_sql_query(
        "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
        "AND symbol='NIFTYBEES' ORDER BY date", con)['date'].tolist() if d >= '2005-01-03']
    con.close()
    panel = pickle.load(open(RES / 'panel161.pkl', 'rb'))
    n = panel.n
    print('panel %d symbols, calendar %d sessions %s..%s'
          % (len(panel.close), n, cal[0], cal[-1]), flush=True)

    raw = pd.read_csv(R161 / 'ath_events.csv')
    raw = raw[raw['symbol'].isin(panel.close) & raw['entry_date'].isin(panel.pos)].copy()
    raw['entry_i'] = raw['entry_date'].map(panel.pos)
    base = raw[(raw['tv20_cr'] >= SPEC['liq']) & (raw['x_bars'] >= SPEC['X']) &
               (raw['depth_pct'] >= SPEC['dep'])]
    print('base events after liquidity / base-age / depth: %d' % len(base), flush=True)

    win_idx = {}
    for w, (lo, hi) in WINDOWS.items():
        ix = [i for i, d in enumerate(cal) if lo <= d <= hi]
        win_idx[w] = (ix[0], ix[-1])

    cfg = dict(exit=SPEC['ex'], hard_stop=SPEC['hard'], time_stop=0, cost_bps=COST,
               gate_ok=None)
    per_seed = {}                 # (cell) -> {window: {seed: cagr/dd/calmar}}
    control_navs = None
    all_navs = {}                 # 'mask|policy' -> [30 x n] NAV paths, for the YoY table

    for mname, mpath in MASKS:
        policies = ['-'] if mpath is None else ['fail', 'pass']
        for pol in policies:
            if mpath is None:
                ev_df = base
            else:
                fn = mask_lookup(mpath, pol)
                keep = [fn(s, d) for s, d in zip(base['symbol'], base['trigger_date'])]
                ev_df = base[np.asarray(keep)]
            ev_df = rearm(ev_df)
            events = [dict(symbol=s, entry_i=int(i))
                      for s, i in zip(ev_df['symbol'], ev_df['entry_i'])]
            print('\n%s / missing=%s: %d events (%d before re-arm)'
                  % (mname, pol, len(events), len(ev_df)), flush=True)
            if len(events) < 20:
                print('  skipped: fewer than 20 events', flush=True)
                continue

            navs, trades_by_seed = [], []
            for sd in SEEDS:
                nav, tr = B.simulate(events, panel, cfg, sd)
                navs.append(nav)
                trades_by_seed.append(tr)
            if mname == 'control':
                control_navs = navs
            all_navs['%s|%s' % (mname, pol)] = np.asarray(navs)

            for w, (i0, i1) in win_idx.items():
                cid = '%s|%s|%s' % (mname, pol, w)
                rows_m = []
                for nav, tr in zip(navs, trades_by_seed):
                    sub = nav[i0:i1 + 1]
                    sc = cal[i0:i1 + 1]
                    twin = [t for t in tr if i0 <= panel.pos.get(t['exit_date'], -1) <= i1]
                    m = B.metrics(sub, sc, twin)
                    if not m:
                        continue
                    mv = invested_series(tr, panel, n)[i0:i1 + 1]
                    with np.errstate(invalid='ignore', divide='ignore'):
                        inv = np.nanmean(np.where(sub > 0, mv / sub, np.nan)) * 100.0
                    m['avg_pct_invested'] = float(min(inv, 100.0))
                    rows_m.append(m)
                if not rows_m:
                    continue
                per_seed[cid] = rows_m
                if cid in done:
                    continue
                cg = sorted(m['cagr'] for m in rows_m)
                dd = sorted(m['maxdd'] for m in rows_m)
                row = dict(cell=cid, mask=mname, missing=pol, window=w,
                           n_events=len(events), seeds=len(rows_m),
                           cagr_med=round(float(np.median(cg)), 2), cagr_worst=cg[0],
                           cagr_best=cg[-1],
                           maxdd_med=round(float(np.median(dd)), 2), maxdd_worst=dd[0],
                           calmar_med=round(float(np.median(
                               [m['calmar'] for m in rows_m])), 3),
                           sharpe_med=round(float(np.median(
                               [m['sharpe'] for m in rows_m])), 3))
                for k in ('trades', 'trades_per_yr', 'win_rate', 'avg_win', 'avg_loss',
                          'expectancy', 'max_loss_streak', 'avg_pct_invested'):
                    row[k] = round(float(np.median([m.get(k, np.nan) for m in rows_m])), 2)
                pd.DataFrame([row])[FIELDS].to_csv(out_cells, mode='a', header=False,
                                                   index=False)
                print('  %-28s CAGR %6.2f%% [%.2f..%.2f]  DD %7.2f%%  Calmar %5.3f  '
                      'inv %4.1f%%  tr/yr %5.1f'
                      % (cid, row['cagr_med'], cg[0], cg[-1], row['maxdd_med'],
                         row['calmar_med'], row['avg_pct_invested'],
                         row['trades_per_yr']), flush=True)

    # ---- paired deltas, same seed, against the control -----------------------------
    pairs = []
    for cid, rows_m in per_seed.items():
        mname, pol, w = cid.split('|')
        if mname == 'control':
            continue
        ctrl = per_seed.get('control|-|%s' % w)
        if not ctrl or len(ctrl) != len(rows_m):
            continue
        d_cagr = [a['cagr'] - b['cagr'] for a, b in zip(rows_m, ctrl)]
        d_dd = [a['maxdd'] - b['maxdd'] for a, b in zip(rows_m, ctrl)]
        d_cal = [a['calmar'] - b['calmar'] for a, b in zip(rows_m, ctrl)]
        pairs.append(dict(cell=cid, mask=mname, missing=pol, window=w, n=len(d_cagr),
                          d_cagr_med=round(float(np.median(d_cagr)), 2),
                          cagr_wins=int(sum(1 for x in d_cagr if x > 0)),
                          d_maxdd_med=round(float(np.median(d_dd)), 2),
                          dd_wins=int(sum(1 for x in d_dd if x > 0)),
                          d_calmar_med=round(float(np.median(d_cal)), 3),
                          calmar_wins=int(sum(1 for x in d_cal if x > 0))))
    if pairs:
        pd.DataFrame(pairs).to_csv(out_pairs, index=False)
        print('\nwrote %s (%d paired rows)' % (out_pairs, len(pairs)), flush=True)

    if control_navs is not None:
        df = pd.DataFrame({('seed%d' % s): v for s, v in zip(SEEDS, control_navs)},
                          index=cal)
        df.to_csv(RES / 'baseage_navs30.csv')
        print('wrote %s (%d x %d) - the Base Age ensemble Part C blends on'
              % (RES / 'baseage_navs30.csv', *df.shape), flush=True)
    np.savez_compressed(RES / 'partB_navs.npz', dates=np.array(cal), **all_navs)
    print('wrote %s (%d arms x 30 seeds)' % (RES / 'partB_navs.npz', len(all_navs)),
          flush=True)
    json.dump({k: [dict(m) for m in v] for k, v in per_seed.items()},
              open(RES / 'partB_perseed.json', 'w'), default=float)
    print('\nPART B DONE in %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
