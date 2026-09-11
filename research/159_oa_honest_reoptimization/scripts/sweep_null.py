# -*- coding: utf-8 -*-
"""Stage A2: extend the trail grid, and add the date-matched random-entry null.

Two things stage A left open.

EXTEND. Return was still rising at trail-50, the edge of the grid, so 30 was not an optimum
and 50 may not be either. Trails 75, 100 and 150 settle whether there is a peak or whether
the surface just keeps climbing towards buy-and-hold.

THE NULL, which matters more. As the trail lengthens the book stops being a timing system
and becomes "buy breakout stocks and hold them for months". Trades fall from 5,375 at
trail-10 to 1,466 at trail-50 - about 73 a year across 16 slots. At that point beating a
large-cap index is not evidence of an edge; a long-only smallcap momentum basket does that
on beta alone.

So: on each day the real signal fires N entries, the null fires N entries drawn at RANDOM
from the same eligible universe on the SAME day. Same count, same days, same trail, same
stop, same slots, same costs. Everything identical except which names. If the real signal
cannot beat that, the return belongs to the basket and the universe filter, not to the
all-time-high breakout rule.

This is the control research/142 ran and passed on its own (look-ahead) entry, which proves
nothing about the honest one.
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

OUT = ROOT / 'research/159_oa_honest_reoptimization/results/stageA2.csv'
WIN = ('2006-01-01', '2026-08-31')
SEEDS = list(range(1, 31))
COST, SLOTS, SIZE, STOP = 0.0025, 16, 0.0625, 0.08
TRAILS = [50, 75, 100, 150]
# close_same vs null_random is the comparison; stop_above_candle is carried
# for the trail-extension half of this stage.
ENTRIES = ['stop_above_candle', 'close_same', 'null_random']
FIELDS = ['cell', 'entry', 'trail', 'cagr_med', 'cagr_min', 'cagr_max', 'dd_med',
          'calmar_med', 'trades_med', 'win_med', 'secs']


def base(w):
    close, tv20, athcp = w['close'], w['tv20'], w['athcp']
    etf = [c for c in close.columns if em.is_etf(c)]
    tv_prev, prev_close = tv20.shift(1), close.shift(1)
    elig = tv_prev >= em.TV_FLOOR
    elig[etf] = False
    r = {n: close / close.shift(n) - 1 for n in (63, 126, 189, 252)}
    rs = ((2 * r[63] + r[126] + r[189] + r[252]).where(elig)
          .rank(axis=1, pct=True) * 100).shift(1)
    setup = (prev_close < athcp) & (prev_close >= 0.8 * athcp) & elig & (rs >= 70.0)
    trig = setup & (close > athcp) & athcp.notna()
    return trig, athcp, rs, tv_prev, elig


def make(w, entry, seed=0):
    trig, athcp, rs, tvp, elig = base(w)
    high = w['high']
    if entry == 'stop_above_candle':
        hp = high.shift(1)
        return (trig.shift(1).fillna(False).astype(bool) & (high > hp)), hp, rs, tvp
    if entry == 'close_same':
        return trig, athcp, rs, tvp
    if entry == 'null_random':
        # Date-matched: on each day, draw the SAME NUMBER of names the real signal fired,
        # at random from the names ELIGIBLE that day. Not from the whole panel - drawing
        # from names that fail the liquidity floor would test the floor, not the breakout.
        rng = np.random.default_rng(10_000 + seed)
        T = trig.fillna(False).values
        E = elig.reindex(columns=trig.columns).fillna(False).values
        # A name with no price that day is not a fair draw, it is a missing bar.
        P = np.isfinite(w['close'].values)
        N = np.zeros_like(T)
        for i in range(T.shape[0]):
            k = int(T[i].sum())
            if k == 0:
                continue
            pool = np.nonzero(E[i] & P[i])[0]
            if len(pool) == 0:
                continue
            pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
            N[i, pick] = True
        return pd.DataFrame(N, index=trig.index, columns=trig.columns), athcp, rs, tvp
    raise ValueError(entry)


def done():
    if not OUT.exists():
        return set()
    with open(OUT) as f:
        return {r['cell'] for r in csv.DictReader(f)}


def main():
    have = done()
    if not OUT.exists():
        with open(OUT, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    for trail in TRAILS:
        todo = [e for e in ENTRIES if 'c|%s|%d' % (e, trail) not in have]
        if not todo:
            continue
        print('trail-%d: loading frames for %s' % (trail, todo), flush=True)
        w = em.load_frames('2005-01-01', trail_sma=trail)
        close, sma, dates = w['close'], w['sma50'], w['close'].index
        days = np.array([i for i, d in enumerate(dates)
                         if WIN[0] <= str(d.date()) <= WIN[1]])
        for entry in todo:
            t0 = time.time()
            stats = []
            for sd in SEEDS:
                # the null redraws its names per seed: that IS its path dependence
                trig, piv, rs, tvp = make(w, entry, seed=sd if entry == 'null_random' else 0)
                eq, trd, _, _ = em.simulate(
                    sd, 'random', days, dates, close.values, w['high'].values,
                    w['open'].values, piv.values, sma.values, rs.values, tvp.values,
                    trig.fillna(False).values if hasattr(trig, 'fillna') else trig,
                    np.zeros(len(dates), bool), True, COST, stop=STOP, slots=SLOTS,
                    size_pct=SIZE,
                    # BOTH the real arm and the null buy at that day's close, so
                    # the only difference between them is which names.
                    fill_close=(entry in ('close_same', 'null_random')))
                st, _ = em.stats_from(eq, dates[days], trd, em.CAPITAL)
                stats.append(st)
            d = pd.DataFrame(stats)
            row = dict(cell='c|%s|%d' % (entry, trail), entry=entry, trail=trail,
                       cagr_med=round(d.cagr.median(), 2), cagr_min=round(d.cagr.min(), 2),
                       cagr_max=round(d.cagr.max(), 2), dd_med=round(d.dd.median(), 2),
                       calmar_med=round((d.cagr / d.dd.abs()).median(), 3),
                       trades_med=int(d.n.median()), win_med=round(d.win.median(), 1),
                       secs=round(time.time() - t0))
            with open(OUT, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print('  %-20s trail%-4d CAGR %6.2f [%6.2f..%6.2f] DD %6.2f Cal %5.3f n%5d (%ds)'
                  % (entry, trail, row['cagr_med'], row['cagr_min'], row['cagr_max'],
                     row['dd_med'], row['calmar_med'], row['trades_med'], row['secs']),
                  flush=True)
    print('STAGE_A2_DONE', flush=True)


if __name__ == '__main__':
    main()
