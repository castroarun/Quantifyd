"""research/176 stage 2 — INTRADAY always-on trend sweep, bars resampled from 5-min.

This is the direct re-run of research/48's basket kill over 11.5 years instead of
its 2.3-year (2024-03+) window. Same grid, same nulls, same reporting as stage 1.

Usage: stage2_intraday_basket.py --rule 60min --workers 3 --min-tv 10
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
from engine import (connect, load_bars, resample_intraday, all_directions,
                    apply_policy, run_book, buy_and_hold)

RES = os.path.join(HERE, '..', 'results')

WINDOWS = {
    'full': ('2015-02-01', '2026-12-31'),
    'h1':   ('2015-02-01', '2020-12-31'),
    'h2':   ('2021-01-01', '2026-12-31'),
}
POLICIES = ['long_flat', 'long_short', 'short_flat']
FILLS = ['next_open', 'signal_close']

FIELDS = ['symbol', 'window', 'family', 'signal', 'policy', 'fill',
          'n_bars', 'years', 'cagr10', 'cagr20', 'cagr40', 'maxdd', 'calmar20',
          'switches_yr', 'n_trades', 'win_rate', 'avg_win', 'avg_loss', 'expectancy',
          'time_in_mkt', 'long_share', 'short_share',
          'bh_cagr', 'bh_maxdd', 'bh_calmar']

ORIGIN = pd.Timestamp('2015-01-01 09:15:00')


def worker(shard, symbols, rule):
    out = os.path.join(RES, f'stage2_{rule}_part{shard}.csv')
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
            d5 = load_bars(con, sym, '5minute')
            if d5.empty or len(d5) < 50000:
                continue
            df = d5.resample(rule, origin=ORIGIN, label='left', closed='left').agg(
                {'open': 'first', 'high': 'max', 'low': 'min',
                 'close': 'last', 'volume': 'sum'}).dropna(subset=['open', 'close'])
            del d5
        except Exception as e:
            print(f'[{shard}] {sym} load fail {e}', flush=True)
            continue
        if len(df) < 2000:
            continue
        dirs = all_directions(df)
        rows = []
        for wname, (a, b) in WINDOWS.items():
            m = (df.index >= a) & (df.index <= b)
            if m.sum() < 1000:
                continue
            sub = df[m]
            bh = {fl: buy_and_hold(sub, fl, 20.0) for fl in FILLS}
            for sname, dd in dirs.items():
                dsub = dd[m]
                for pol in POLICIES:
                    post = apply_policy(dsub, pol)
                    if (post != 0).sum() == 0:
                        continue
                    for fill in FILLS:
                        r20 = run_book(sub, post, fill, 20.0)
                        r10 = run_book(sub, post, fill, 10.0)
                        r40 = run_book(sub, post, fill, 40.0)
                        rows.append(dict(
                            symbol=sym, window=wname, family=sname.split('_')[0],
                            signal=sname, policy=pol, fill=fill,
                            n_bars=len(sub), years=round(r20['years'], 3),
                            cagr10=round(r10['cagr'], 6), cagr20=round(r20['cagr'], 6),
                            cagr40=round(r40['cagr'], 6), maxdd=round(r20['maxdd'], 6),
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
        print(f'[{shard}] {k+1}/{len(symbols)} {sym} bars={len(df)} rows={len(rows)} '
              f'elapsed={(time.time()-t0)/60:.1f}m', flush=True)
    con.close()
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rule', default='60min')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--min-tv', type=float, default=10.0)
    a = ap.parse_args()
    uni = pd.read_csv(os.path.join(RES, 'universe.csv'))
    uni = uni[uni.med_tv_cr >= a.min_tv]
    syms = sorted(uni.symbol.tolist())
    print(f'{a.rule}: universe {len(syms)} symbols', flush=True)
    chunks = [syms[i::a.workers] for i in range(a.workers)]
    ps = [Process(target=worker, args=(i, chunks[i], a.rule)) for i in range(a.workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print('ALL WORKERS DONE', flush=True)


if __name__ == '__main__':
    main()
