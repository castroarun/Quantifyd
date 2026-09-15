"""research/176 stage 4 — the honest best case, built as a BOOK, and its blend value.

Arun asked for "a single stock / few stocks". Stage 1-3 answered the per-name
question. This stage builds the best-case BOOK from the surviving cell and asks the
only question that decides adoption (research/134's lesson): does it add anything to
the live short-vol book, and does it beat the two nulls that matter —
  * buy-and-hold the same names at the same weight   (r/134: trend timing HURT)
  * plain cash at the same weight                    (the de-levering null)

Books built:
  ARUN3   = MARUTI, RELIANCE, HDFCBANK, equal weight
  LIQ10   = the 10 most liquid names in the panel, equal weight
  ARUN1_* = each of Arun's three names alone
Each with: the best daily cell (EMA_9_21 long/flat), Arun's own cells, and B&H.
"""
import os
import sys
import json
import itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import connect, load_bars, all_directions, apply_policy, IDLE_YIELD

RES = os.path.join(HERE, '..', 'results')
SV = '/home/arun/quantifyd/research/134_directional_diversifier/results/stage_a_monthly.csv'

CELLS = ['EMA_9_21', 'EMA_10_30', 'MST_7_5.0_7_2.0', 'ST_7_3.0', 'EMA_50_200']
ARUN3 = ['MARUTI', 'RELIANCE', 'HDFCBANK']
START, END = '2006-01-01', '2026-09-15'


def daily_book_returns(symbols, cell, policy='long_flat', cost_bps_rt=20.0):
    """Equal-weight, daily-rebalanced book of per-name always-on sleeves.
    Returns a pandas Series of daily net returns indexed by date."""
    con = connect()
    legs = {}
    for s in symbols:
        df = load_bars(con, s, 'day')
        df = df[(df.index >= START) & (df.index <= END)]
        if len(df) < 400:
            continue
        o = df['open'].values.astype(float)
        ret = np.zeros(len(o))
        ret[:-1] = o[1:] / o[:-1] - 1.0
        if cell == 'BH':
            held = np.ones(len(o), dtype=np.int8)
        else:
            d = all_directions(df)[cell]
            post = apply_policy(d, policy)
            held = np.zeros(len(post), dtype=np.int8)
            held[1:] = post[:-1]
        bar_yr = 1.0 / 252.0
        idle = (held == 0) * ((1 + IDLE_YIELD) ** bar_yr - 1)
        chg = np.abs(np.diff(np.concatenate([[0], held]))).astype(float)
        net = held * ret + idle - chg * (cost_bps_rt / 2 / 10000.0)
        legs[s] = pd.Series(net, index=df.index)
    con.close()
    if not legs:
        return pd.Series(dtype=float)
    M = pd.DataFrame(legs)
    # a name with no bar that day sits in cash for the book
    M = M.fillna((1 + IDLE_YIELD) ** (1 / 252.0) - 1)
    return M.mean(axis=1)


def stats(r, periods=252):
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    vol = r.std() * np.sqrt(periods)
    return dict(cagr=cagr, maxdd=float(dd), calmar=cagr / abs(dd),
                vol=vol, sharpe=(cagr - 0.052) / vol if vol else np.nan, years=yrs)


def mstats(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 12.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    return dict(cagr=cagr, maxdd=float(dd), calmar=cagr / abs(dd),
                worst=float(r.min()), n=len(r))


def main():
    print('=' * 108)
    print('PART A — the "few stocks" BOOK, daily, 2006-2026, 20 bps round trip, cash at 5.2%')
    print('=' * 108)
    books = {}
    rows = []
    liq = pd.read_csv(os.path.join(RES, 'universe.csv')).sort_values(
        'med_tv_cr', ascending=False)
    LIQ10 = liq.symbol.head(10).tolist()
    print('LIQ10 =', ', '.join(LIQ10), '\n')
    universes = {'ARUN3': ARUN3, 'LIQ10': LIQ10,
                 'MARUTI': ['MARUTI'], 'RELIANCE': ['RELIANCE'], 'HDFCBANK': ['HDFCBANK']}
    for uname, syms in universes.items():
        for cell in CELLS + ['BH']:
            r = daily_book_returns(syms, cell)
            if r.empty:
                continue
            st = stats(r)
            books[(uname, cell)] = r
            rows.append(dict(book=uname, cell=cell, **st))
    t = pd.DataFrame(rows)
    t['cagr'] = t.cagr.map(lambda v: f'{v:+.2%}')
    t['maxdd'] = t.maxdd.map(lambda v: f'{v:.1%}')
    t['calmar'] = t.calmar.map(lambda v: f'{v:.3f}')
    t['vol'] = t.vol.map(lambda v: f'{v:.1%}')
    t['sharpe'] = t.sharpe.map(lambda v: f'{v:.2f}')
    print(t[['book', 'cell', 'cagr', 'maxdd', 'calmar', 'vol', 'sharpe']].to_string(index=False))

    print('\n' + '=' * 108)
    print('PART B — blend against the LIVE short-vol book (research/134 combined series,')
    print('         C1 stock winged strangles + 45-DTE NIFTY straddle, 75 months 2019-05 to 2026-07)')
    print('=' * 108)
    sv = pd.read_csv(SV)
    sv['month'] = pd.PeriodIndex(sv.month, freq='M')
    sv = sv.set_index('month').sort_index()
    svr = sv['combined_pct'] / 100.0
    base = mstats(svr)
    print(f"short-vol book alone: CAGR {base['cagr']:+.2%}  MaxDD {base['maxdd']:.2%}  "
          f"Calmar {base['calmar']:.2f}  worst month {base['worst']:+.2%}  n={base['n']}")

    cand = {}
    for (uname, cell), r in books.items():
        m = (1 + r).resample('ME').prod() - 1
        m.index = pd.PeriodIndex(m.index, freq='M')
        m = m.reindex(svr.index).dropna()
        if len(m) < 60:
            continue
        cand[f'{uname}/{cell}'] = m
    # cash null at 5.2%
    cand['CASH-NULL'] = pd.Series((1.052) ** (1 / 12) - 1, index=svr.index)

    print(f"\n{'candidate':26s} {'corr':>6s} {'w':>5s} {'CAGR':>8s} {'MaxDD':>8s} "
          f"{'Calmar':>7s} {'worst':>8s}  {'vs base':>9s}")
    print('-' * 92)
    for name, m in cand.items():
        common = svr.index.intersection(m.index)
        a, b = svr.loc[common], m.loc[common]
        c = float(np.corrcoef(a, b)[0, 1])
        for w in (0.10, 0.20, 0.30, 0.40):
            bl = (1 - w) * a + w * b
            s = mstats(bl)
            bs = mstats(a)
            flag = ''
            if s['calmar'] >= bs['calmar'] + 0.10 and s['cagr'] >= bs['cagr']:
                flag = ' <= CLEARS +0.10 CALMAR'
            print(f'{name:26s} {c:+6.2f} {w:5.0%} {s["cagr"]:+8.2%} {s["maxdd"]:8.2%} '
                  f'{s["calmar"]:7.2f} {s["worst"]:+8.2%}  {s["calmar"]-bs["calmar"]:+9.2f}{flag}')
        print('-' * 92)

    print('\n' + '=' * 108)
    print('PART C — where does the trend rule beat buy-and-hold? (is it selectable ex ante?)')
    print('=' * 108)
    import glob
    d = pd.concat([pd.read_csv(f) for f in
                   sorted(glob.glob(os.path.join(RES, 'stage1_day_part*.csv')))],
                  ignore_index=True)
    d = d[(d.window == 'full') & (d.fill == 'next_open') &
          (d.policy == 'long_flat') & (d.signal == 'EMA_9_21')]
    d['excess'] = d.cagr20 - d.bh_cagr
    d['beat'] = d.excess > 0
    print(f'  EMA_9_21 long/flat: beats B&H on {d.beat.sum()}/{len(d)} names')
    for lo, hi, lab in [(-9, 0.0, 'B&H CAGR < 0%'), (0.0, 0.10, '0-10%'),
                        (0.10, 0.20, '10-20%'), (0.20, 9, '>20%')]:
        s = d[(d.bh_cagr >= lo) & (d.bh_cagr < hi)]
        if len(s):
            print(f'    names whose B&H CAGR was {lab:14s} n={len(s):3d}  '
                  f'trend beats B&H on {s.beat.mean():5.1%}  median excess {s.excess.median():+.2%}')
    print('\n  -> if the beat concentrates in the names that did BADLY, the rule is a loss-avoider')
    print('     you can only select after the fact.')


if __name__ == '__main__':
    main()
