"""research/176 stage 5 — Arun's LITERAL MST machine, with the lot stacking.

Stages 1-4 tested the MST pair reduced to a single-unit posture. Arun's manual
system accumulated: inside a master regime, every child flip in the master's
direction ADDS a lot, up to five; a master reversal closes everything and the
book waits to re-arm. This stage runs that exact accumulation.

Exposure model: 1 lot = 20% of book capital, so five lots = 100%. That makes the
stacked book directly comparable to the single-unit book (same max exposure, so
this is a RAMP not a leverage test).

Also emits: the per-window stability of the one effect stage 3 found real
(better Calmar than buy-and-hold on the daily long/flat arm), and a book-level
cost ladder.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import (connect, load_bars, supertrend_dir, apply_policy,
                    all_directions, IDLE_YIELD)

RES = os.path.join(HERE, '..', 'results')
ARUN3 = ['MARUTI', 'RELIANCE', 'HDFCBANK']
START, END = '2006-01-01', '2026-09-15'


def mst_stacked_exposure(master, child, max_lots=5, long_only=False):
    n = len(master)
    exp = np.zeros(n)
    units = 0
    prev_m = 0
    prev_c = 0
    for i in range(n):
        m = master[i]
        c = child[i]
        if m != prev_m:
            units = 0
            prev_m = m
        if m != 0 and c == m and prev_c != m and units < max_lots:
            units += 1
        prev_c = c
        side = m
        if long_only and side < 0:
            side = 0
        exp[i] = side * units / max_lots
    return exp


def book_from_exposure(df, exp, cost_bps_rt=20.0):
    o = df['open'].values.astype(float)
    ret = np.zeros(len(o))
    ret[:-1] = o[1:] / o[:-1] - 1.0
    held = np.zeros(len(exp))
    held[1:] = exp[:-1]
    bar_yr = 1.0 / 252.0
    idle = (1.0 - np.abs(held)).clip(0) * ((1 + IDLE_YIELD) ** bar_yr - 1)
    chg = np.abs(np.diff(np.concatenate([[0.0], held])))
    net = held * ret + idle - chg * (cost_bps_rt / 2 / 10000.0)
    return pd.Series(net, index=df.index), float(np.count_nonzero(chg))


def stats(r):
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = float((eq / eq.cummax() - 1).min())
    return cagr, dd, cagr / abs(dd), yrs


def main():
    con = connect()
    print('=' * 104)
    print("PART A — Arun's MST machine WITH lot stacking (master 7,5 / child 7,2, max 5 lots,")
    print('          1 lot = 20% of capital), daily bars 2006-2026, next-open fills, 20 bps')
    print('=' * 104)
    print(f"{'name':10s} {'variant':26s} {'CAGR':>8s} {'MaxDD':>8s} {'Calmar':>7s} {'sw/yr':>7s}")
    print('-' * 74)
    legs = {}
    for sym in ARUN3:
        df = load_bars(con, sym, 'day')
        df = df[(df.index >= START) & (df.index <= END)]
        h, l, c = (df['high'].values.astype(float), df['low'].values.astype(float),
                   df['close'].values.astype(float))
        mst = supertrend_dir(h, l, c, 7, 5.0)
        cst = supertrend_dir(h, l, c, 7, 2.0)
        variants = {
            'stacked long/short': mst_stacked_exposure(mst, cst, 5, False),
            'stacked long/flat': mst_stacked_exposure(mst, cst, 5, True),
            'single-unit long/short': all_directions(df)['MST_7_5.0_7_2.0'].astype(float),
            'single-unit long/flat': apply_policy(
                all_directions(df)['MST_7_5.0_7_2.0'], 'long_flat').astype(float),
            'buy and hold': np.ones(len(df)),
        }
        for vn, e in variants.items():
            r, nsw = book_from_exposure(df, e)
            cg, dd, cal, yrs = stats(r)
            legs[(sym, vn)] = r
            print(f'{sym:10s} {vn:26s} {cg:+8.2%} {dd:8.1%} {cal:7.3f} {nsw/yrs:7.1f}')
        print('-' * 74)
    # equal-weight ARUN3 book per variant
    print(f"{'ARUN3':10s} equal-weight book of the three")
    for vn in ['stacked long/short', 'stacked long/flat', 'single-unit long/short',
               'single-unit long/flat', 'buy and hold']:
        M = pd.DataFrame({s: legs[(s, vn)] for s in ARUN3}).fillna(0.0)
        r = M.mean(axis=1)
        cg, dd, cal, _ = stats(r)
        print(f'{"":10s} {vn:26s} {cg:+8.2%} {dd:8.1%} {cal:7.3f}')
    con.close()

    print('\n' + '=' * 104)
    print('PART B — is the ONE real effect stable? Calmar-vs-buy-and-hold beat rate, by window')
    print('         (daily, long/flat, next-open, 20 bps, 146 names)')
    print('=' * 104)
    agg = pd.read_csv(os.path.join(RES, 'agg_day.csv'))
    agg = agg[(agg.fill == 'next_open') & (agg.policy == 'long_flat')]
    keep = ['EMA_9_21', 'EMA_10_30', 'EMA_20_50', 'EMA_50_200', 'ST_7_3.0',
            'MST_7_5.0_7_2.0']
    p = agg[agg.signal.isin(keep)].pivot_table(index='signal', columns='window',
                                               values=['calmar_beat', 'beat_rate20'])
    print(p.to_string(float_format=lambda v: f'{v:.3f}'))

    print('\n' + '=' * 104)
    print('PART C — book-level cost ladder, ARUN3 equal-weight, daily long/flat EMA_9_21')
    print('=' * 104)
    con = connect()
    for cost in (10, 20, 40, 60):
        legs2 = {}
        for sym in ARUN3:
            df = load_bars(con, sym, 'day')
            df = df[(df.index >= START) & (df.index <= END)]
            e = apply_policy(all_directions(df)['EMA_9_21'], 'long_flat').astype(float)
            r, _ = book_from_exposure(df, e, cost)
            legs2[sym] = r
        r = pd.DataFrame(legs2).fillna(0.0).mean(axis=1)
        cg, dd, cal, _ = stats(r)
        print(f'  {cost:3d} bps round trip : CAGR {cg:+.2%}  MaxDD {dd:.1%}  Calmar {cal:.3f}')
    # and B&H for reference
    legs3 = {}
    for sym in ARUN3:
        df = load_bars(con, sym, 'day')
        df = df[(df.index >= START) & (df.index <= END)]
        r, _ = book_from_exposure(df, np.ones(len(df)), 20)
        legs3[sym] = r
    r = pd.DataFrame(legs3).fillna(0.0).mean(axis=1)
    cg, dd, cal, _ = stats(r)
    print(f'  buy and hold      : CAGR {cg:+.2%}  MaxDD {dd:.1%}  Calmar {cal:.3f}')
    con.close()


if __name__ == '__main__':
    main()
