"""research/176 stage 1 — DAILY always-on trend sweep across the whole basket.

One row per (symbol, window, signal, policy, fill). Costs 10/20/40 bps round-trip
are all carried in the same row. Buy-and-hold for the same symbol/window is carried
in every row so the beat-rate can be computed without a second pass.

Resume-safe: symbols already present in a shard are skipped.
Usage:  stage1_daily_basket.py [--workers 3] [--tf day]
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Process

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import (connect, load_bars, all_directions, apply_policy,
                    run_book, buy_and_hold, ST_GRID, EMA_GRID, MST_GRID)

RES = os.path.join(HERE, '..', 'results')

WINDOWS = {
    'full':  ('2006-01-01', '2026-09-15'),
    'h1':    ('2006-01-01', '2015-12-31'),
    'h2':    ('2016-01-01', '2026-09-15'),
}
POLICIES = ['long_flat', 'long_short', 'short_flat']
FILLS = ['next_open', 'signal_close']
COSTS = [10.0, 20.0, 40.0]

FIELDS = ['symbol', 'window', 'family', 'signal', 'policy', 'fill',
          'n_bars', 'years', 'cagr10', 'cagr20', 'cagr40', 'maxdd', 'calmar20',
          'switches_yr', 'n_trades', 'win_rate', 'avg_win', 'avg_loss', 'expectancy',
          'time_in_mkt', 'long_share', 'short_share',
          'bh_cagr', 'bh_maxdd', 'bh_calmar']


def family_of(name):
    return name.split('_')[0]


def worker(shard, symbols, tf):
    out = os.path.join(RES, f'stage1_{tf}_part{shard}.csv')
    done = set()
    if os.path.exists(out):
        try:
            done = set(pd.read_csv(out, usecols=['symbol'])['symbol'].unique())
        except Exception:
            done = set()
    f = open(out, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if f.tell() == 0:
        w.writeheader()
    con = connect()
    t0 = time.time()
    for k, sym in enumerate(symbols):
        if sym in done:
            continue
        try:
            df = load_bars(con, sym, tf)
        except Exception as e:
            print(f'[{shard}] {sym} load fail {e}', flush=True)
            continue
        if df.empty or len(df) < 400:
            continue
        dirs = all_directions(df)
        rows = []
        for wname, (a, b) in WINDOWS.items():
            m = (df.index >= a) & (df.index <= b)
            if m.sum() < 300:
                continue
            sub = df[m]
            bh = {}
            for fill in FILLS:
                bh[fill] = buy_and_hold(sub, fill, 20.0)
            for sname, d in dirs.items():
                dsub = d[m]
                for pol in POLICIES:
                    post = apply_policy(dsub, pol)
                    if (post != 0).sum() == 0:
                        continue
                    for fill in FILLS:
                        r20 = run_book(sub, post, fill, 20.0)
                        r10 = run_book(sub, post, fill, 10.0)
                        r40 = run_book(sub, post, fill, 40.0)
                        rows.append(dict(
                            symbol=sym, window=wname, family=family_of(sname),
                            signal=sname, policy=pol, fill=fill,
                            n_bars=len(sub), years=round(r20['years'], 3),
                            cagr10=round(r10['cagr'], 6), cagr20=round(r20['cagr'], 6),
                            cagr40=round(r40['cagr'], 6),
                            maxdd=round(r20['maxdd'], 6),
                            calmar20=round(r20['calmar'], 4) if np.isfinite(r20['calmar']) else '',
                            switches_yr=round(r20['switches_per_yr'], 2),
                            n_trades=r20['n_trades'],
                            win_rate=round(r20['win_rate'], 4) if r20['n_trades'] else '',
                            avg_win=round(r20['avg_win'], 5) if r20['n_trades'] else '',
                            avg_loss=round(r20['avg_loss'], 5) if r20['n_trades'] else '',
                            expectancy=round(r20['expectancy'], 6) if r20['n_trades'] else '',
                            time_in_mkt=round(r20['time_in_mkt'], 4),
                            long_share=round(r20['long_share'], 4),
                            short_share=round(r20['short_share'], 4),
                            bh_cagr=round(bh[fill]['cagr'], 6),
                            bh_maxdd=round(bh[fill]['maxdd'], 6),
                            bh_calmar=round(bh[fill]['calmar'], 4) if np.isfinite(bh[fill]['calmar']) else '',
                        ))
        w.writerows(rows)
        f.flush()
        el = time.time() - t0
        print(f'[{shard}] {k+1}/{len(symbols)} {sym} rows={len(rows)} '
              f'elapsed={el/60:.1f}m', flush=True)
    con.close()
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--tf', default='day')
    ap.add_argument('--min-tv', type=float, default=25.0)
    a = ap.parse_args()

    uni = pd.read_csv(os.path.join(RES, 'universe.csv'))
    uni = uni[uni.med_tv_cr >= a.min_tv]
    syms = sorted(uni.symbol.tolist())
    print(f'universe after Rs{a.min_tv}cr floor: {len(syms)} symbols', flush=True)

    chunks = [syms[i::a.workers] for i in range(a.workers)]
    ps = [Process(target=worker, args=(i, chunks[i], a.tf)) for i in range(a.workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print('ALL WORKERS DONE', flush=True)


if __name__ == '__main__':
    main()
