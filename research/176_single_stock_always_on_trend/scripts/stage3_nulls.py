"""research/176 stage 3 — the null control that decides whether the ONE surviving
effect (better Calmar than buy-and-hold) is timing or just being out of the market.

Block-permutation null. For each (symbol, cell) we take the posture series the rule
actually produced, cut it into contiguous spells, and SHUFFLE the spell lengths
within each posture class while keeping the sequence of postures intact. That
preserves, exactly:
  * time in market
  * the number of trades (so the cost bill is identical)
  * the run-length distribution of both long spells and flat spells
and destroys only WHEN the long spells happen. If the rule's Calmar does not beat
this null, the improvement is exposure management, not timing.

Also emits a per-year table for the shortlisted cells.

Usage: stage3_nulls.py --tf day --draws 200
"""
import argparse
import os
import sys
import csv
import time
import numpy as np
import pandas as pd
from multiprocessing import Process

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import (connect, load_bars, all_directions, apply_policy,
                    run_book, buy_and_hold, IDLE_YIELD)

RES = os.path.join(HERE, '..', 'results')
SHORTLIST = ['EMA_9_21', 'EMA_10_30', 'EMA_20_50', 'EMA_50_200',
             'ST_7_3.0', 'ST_7_2.5', 'ST_10_5.0', 'MST_7_5.0_7_2.0', 'MST_7_6.0_7_2.0']
FIELDS = ['symbol', 'tf', 'signal', 'policy', 'cagr', 'calmar', 'maxdd',
          'bh_cagr', 'bh_calmar', 'bh_maxdd',
          'null_cagr_med', 'null_calmar_med', 'null_maxdd_med',
          'pct_cagr', 'pct_calmar', 'n_draws', 'time_in_mkt', 'n_trades']


def spells(held):
    ch = np.flatnonzero(np.diff(held.astype(np.int16))) + 1
    starts = np.concatenate([[0], ch])
    ends = np.concatenate([ch, [len(held)]])
    return held[starts], (ends - starts)


def metrics_from_held(held, ret, one_way, bar_yr, years, idle=IDLE_YIELD):
    gross = held * ret
    idle_leg = (held == 0) * ((1 + idle) ** bar_yr - 1)
    chg = np.abs(np.diff(np.concatenate([[0], held]))).astype(float)
    net = gross + idle_leg - chg * one_way
    eq = np.cumprod(1.0 + net)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    maxdd = float(dd.min())
    total = float(eq[-1])
    cagr = total ** (1.0 / years) - 1.0 if total > 0 else -1.0
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return cagr, calmar, maxdd


def permute(vals, lens, rng):
    out_lens = lens.copy()
    for v in np.unique(vals):
        idx = np.flatnonzero(vals == v)
        out_lens[idx] = rng.permutation(lens[idx])
    return np.repeat(vals, out_lens)


def worker(shard, symbols, tf, draws, rule):
    out = os.path.join(RES, f'stage3_{tf}_part{shard}.csv')
    done = set()
    if os.path.exists(out):
        try:
            d = pd.read_csv(out)
            done = set(zip(d.symbol, d.signal, d.policy))
        except Exception:
            done = set()
    f = open(out, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if f.tell() == 0:
        w.writeheader()
    con = connect()
    t0 = time.time()
    for k, sym in enumerate(symbols):
        try:
            if tf == 'day':
                df = load_bars(con, sym, 'day')
                df = df[(df.index >= '2006-01-01') & (df.index <= '2026-09-15')]
            else:
                d5 = load_bars(con, sym, '5minute')
                df = d5.resample(rule, origin=pd.Timestamp('2015-01-01 09:15'),
                                 label='left', closed='left').agg(
                    {'open': 'first', 'high': 'max', 'low': 'min',
                     'close': 'last', 'volume': 'sum'}).dropna(subset=['open', 'close'])
                del d5
        except Exception as e:
            print(f'[{shard}] {sym} {e}', flush=True)
            continue
        if len(df) < 400:
            continue
        dirs = all_directions(df)
        o = df['open'].values.astype(float)
        ret = np.zeros(len(o))
        ret[:-1] = o[1:] / o[:-1] - 1.0
        years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
        bar_yr = years / max(len(df) - 1, 1)
        one_way = 20.0 / 2.0 / 10000.0
        bh = buy_and_hold(df, 'next_open', 20.0)
        rows = []
        for sname in SHORTLIST:
            if sname not in dirs:
                continue
            for pol in ['long_flat']:
                if (sym, sname, pol) in done:
                    continue
                post = apply_policy(dirs[sname], pol)
                held = np.zeros(len(post), dtype=np.int8)
                held[1:] = post[:-1]
                cagr, calmar, maxdd = metrics_from_held(held, ret, one_way, bar_yr, years)
                vals, lens = spells(held)
                rng = np.random.default_rng(abs(hash((sym, sname))) % (2 ** 32))
                nc, nk, nd = [], [], []
                for _ in range(draws):
                    h2 = permute(vals, lens, rng).astype(np.int8)
                    a, b, c = metrics_from_held(h2, ret, one_way, bar_yr, years)
                    nc.append(a)
                    nk.append(b)
                    nd.append(c)
                nc = np.array(nc)
                nk = np.array(nk, dtype=float)
                nd = np.array(nd)
                rows.append(dict(
                    symbol=sym, tf=tf, signal=sname, policy=pol,
                    cagr=round(cagr, 6), calmar=round(calmar, 4) if np.isfinite(calmar) else '',
                    maxdd=round(maxdd, 6),
                    bh_cagr=round(bh['cagr'], 6),
                    bh_calmar=round(bh['calmar'], 4) if np.isfinite(bh['calmar']) else '',
                    bh_maxdd=round(bh['maxdd'], 6),
                    null_cagr_med=round(float(np.median(nc)), 6),
                    null_calmar_med=round(float(np.nanmedian(nk)), 4),
                    null_maxdd_med=round(float(np.median(nd)), 6),
                    pct_cagr=round(float((nc < cagr).mean()), 4),
                    pct_calmar=round(float((nk < calmar).mean()), 4) if np.isfinite(calmar) else '',
                    n_draws=draws,
                    time_in_mkt=round(float((held != 0).mean()), 4),
                    n_trades=int((vals != 0).sum()),
                ))
        if rows:
            w.writerows(rows)
            f.flush()
        print(f'[{shard}] {k+1}/{len(symbols)} {sym} {(time.time()-t0)/60:.1f}m', flush=True)
    con.close()
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tf', default='day')
    ap.add_argument('--rule', default='60min')
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--min-tv', type=float, default=10.0)
    a = ap.parse_args()
    uni = pd.read_csv(os.path.join(RES, 'universe.csv'))
    uni = uni[uni.med_tv_cr >= a.min_tv]
    syms = sorted(uni.symbol.tolist())
    print(f'stage3 {a.tf}: {len(syms)} symbols x {len(SHORTLIST)} cells x '
          f'{a.draws} draws', flush=True)
    chunks = [syms[i::a.workers] for i in range(a.workers)]
    ps = [Process(target=worker, args=(i, chunks[i], a.tf, a.draws, a.rule))
          for i in range(a.workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print('ALL WORKERS DONE', flush=True)


if __name__ == '__main__':
    main()
