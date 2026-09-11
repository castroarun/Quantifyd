# -*- coding: utf-8 -*-
"""Stage A: Open Alpha's entry economics, swept on entries that can actually be placed.

research/142 swept 680 cells and evaluated every one of them on `close > pivot` filled at
`max(pivot, open)` on the SAME bar - an entry that needs the closing price known at the
open. research/158 measured the cost of that assumption: 40.8% becomes -1.7%. So every
parameter the live book runs is the answer to a question about a mechanic that does not
exist.

This asks the same question of the three mechanics that do exist. 75 cells:

    entry  stop_above_candle | close_same | open_next
    trail  10 | 15 | 20 | 30 | 50
    stop   6% | 8% | 10% | 15% | none

The look-ahead entry is run ONCE as a labelled reference row, never as a candidate.

NOT A HUNT FOR A CELL THAT PRINTS 25%. The deliverable is the shape of the surface: a broad
plateau is a finding, an isolated peak is noise and is reported as such. Pre-registered
criteria (STATUS §2): beat NIFTYBEES 11.5% after cost AND tax, neighbours within a
reasonable band, worst seed not negative.

Efficiency, per the mandatory rules in CLAUDE.md: frames are loaded ONCE per trail value (5
loads, not 75), the trigger is built once per (frames, entry), CSV is appended after every
cell, and finished cells are skipped on restart.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/158_oa_arming_width/scripts'))
sys.path.insert(0, str(ROOT))
import oa_entry_mechanics as em          # noqa: E402

OUT = ROOT / 'research/159_oa_honest_reoptimization/results/stageA.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)
WIN = ('2006-01-01', '2026-08-31')
SEEDS = list(range(1, 31))
COST = 0.0025
SLOTS, SIZE = 16, 0.0625

ENTRIES = ['stop_above_candle', 'close_same', 'open_next', 'open_same_REFERENCE']
TRAILS = [10, 15, 20, 30, 50]
STOPS = [0.06, 0.08, 0.10, 0.15, 9.99]        # 9.99 = effectively no hard stop

FIELDS = ['cell', 'entry', 'trail', 'stop', 'placeable', 'cagr_med', 'cagr_min',
          'cagr_max', 'cagr_worst_seed', 'dd_med', 'dd_worst', 'calmar_med',
          'trades_med', 'win_med', 'n_signals', 'secs']


def done_cells():
    if not OUT.exists():
        return set()
    with open(OUT) as f:
        return {r['cell'] for r in csv.DictReader(f)}


def build(w, entry):
    """-> (trig, pivot_frame). Depends only on the frames and the entry mechanic."""
    close, high, open_, athcp, tv20 = (w[k] for k in
                                       ('close', 'high', 'open', 'athcp', 'tv20'))
    etf = [c for c in close.columns if em.is_etf(c)]
    tv_prev, prev_close = tv20.shift(1), close.shift(1)
    elig = tv_prev >= em.TV_FLOOR
    elig[etf] = False
    r = {n: close / close.shift(n) - 1 for n in (63, 126, 189, 252)}
    rs = ((2 * r[63] + r[126] + r[189] + r[252]).where(elig)
          .rank(axis=1, pct=True) * 100).shift(1)
    setup = (prev_close < athcp) & (prev_close >= 0.8 * athcp) & elig & (rs >= 70.0)
    trig = setup & (close > athcp) & athcp.notna()
    piv = athcp
    if entry == 'open_next':
        pp = athcp.shift(1)
        trig = trig.shift(1).fillna(False).astype(bool) & (high >= pp)
        piv = pp
    elif entry == 'stop_above_candle':
        hp = high.shift(1)
        trig = trig.shift(1).fillna(False).astype(bool) & (high > hp)
        piv = hp
    return trig, piv, rs, tv_prev


def main():
    have = done_cells()
    print('%d cells already done' % len(have), flush=True)
    if not OUT.exists():
        with open(OUT, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    for trail in TRAILS:
        t0 = time.time()
        cells = [(e, s) for e in ENTRIES for s in STOPS
                 if 'cell|%s|%d|%s' % (e, trail, s) not in have]
        if not cells:
            print('trail-%d: all cells done, skipping frame load' % trail, flush=True)
            continue
        print('trail-%d: loading frames for %d cells ...' % (trail, len(cells)), flush=True)
        w = em.load_frames('2005-01-01', trail_sma=trail)
        close, sma = w['close'], w['sma50']
        dates = close.index
        days = np.array([i for i, d in enumerate(dates)
                         if WIN[0] <= str(d.date()) <= WIN[1]])
        built = {}
        for entry, stop in cells:
            key = 'cell|%s|%d|%s' % (entry, trail, stop)
            e_real = 'open_same' if entry.endswith('_REFERENCE') else entry
            if e_real not in built:
                built[e_real] = build(w, e_real)
            trig, piv, rs, tvp = built[e_real]
            fill_close = (e_real == 'close_same')
            c0 = time.time()
            stats = []
            for sd in SEEDS:
                eq, trd, _, _ = em.simulate(
                    sd, 'random', days, dates, close.values, w['high'].values,
                    w['open'].values, piv.values, sma.values, rs.values, tvp.values,
                    trig.fillna(False).values, np.zeros(len(dates), bool),
                    True, COST, stop=stop, slots=SLOTS, size_pct=SIZE,
                    fill_close=fill_close)
                st, _ = em.stats_from(eq, dates[days], trd, em.CAPITAL)
                stats.append(st)
            d = pd.DataFrame(stats)
            row = dict(cell=key, entry=entry, trail=trail,
                       stop=('none' if stop > 1 else stop),
                       placeable=('NO' if entry.endswith('_REFERENCE') else 'yes'),
                       cagr_med=round(d.cagr.median(), 2), cagr_min=round(d.cagr.min(), 2),
                       cagr_max=round(d.cagr.max(), 2),
                       cagr_worst_seed=round(d.cagr.min(), 2),
                       dd_med=round(d.dd.median(), 2), dd_worst=round(d.dd.min(), 2),
                       calmar_med=round((d.cagr / d.dd.abs()).median(), 3),
                       trades_med=int(d.n.median()), win_med=round(d.win.median(), 1),
                       n_signals=int(trig.fillna(False).values[days].sum()),
                       secs=round(time.time() - c0))
            with open(OUT, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print('  %-22s trail%-3d stop%-5s  CAGR %6.2f [%6.2f..%6.2f]  DD %6.2f  '
                  'Cal %5.2f  n%4d  (%ds)'
                  % (entry, trail, row['stop'], row['cagr_med'], row['cagr_min'],
                     row['cagr_max'], row['dd_med'], row['calmar_med'],
                     row['trades_med'], row['secs']), flush=True)
        print('trail-%d done in %.0fs' % (trail, time.time() - t0), flush=True)
    print('STAGE_A_DONE', flush=True)


if __name__ == '__main__':
    main()
