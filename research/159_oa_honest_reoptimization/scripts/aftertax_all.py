# -*- coding: utf-8 -*-
"""Re-run the trail surface, the null control and the gate bake-off AFTER TAX.

Arun read the roster page and could not tell which table belonged to which system, because
three of them were pre-tax and v3-only while the rest were after-tax and all-books. Labelling
them helped; putting everything on ONE basis is the actual fix. After tax is the basis that
matters, so every table on the page becomes after tax and the pre-tax versions are dropped
rather than shown alongside.

Three tables, one basis:
    A  trail surface   3 placeable entries + the look-ahead reference, trails 10-150
    B  null control    the rule vs date-matched random names, trails 50-150
    C  gate bake-off   29 gate signals at the adopted trail

All at 20% short-term / 12.5% long-term with Indian financial-year loss netting, 25 bps a
side, 5% on idle cash, 30-seed medians.
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

OUT = ROOT / 'research/159_oa_honest_reoptimization/results/after_tax_tables.csv'
SEEDS = list(range(1, 31))
COST, SLOTS, SIZE, STOP = 0.0025, 16, 0.0625, 0.08
CASH_Y, STCG, LTCG = 0.05, 0.20, 0.125
W_LONG = ('2006-01-01', '2026-08-31')
W_VIX = ('2016-01-01', '2026-08-31')
FIELDS = ['table', 'label', 'trail', 'window', 'cagr_med', 'cagr_min', 'cagr_max',
          'dd_med', 'calmar_med', 'trades_med', 'blocked_pct', 'secs']

PRICE_GATES = (['none'] + ['sma%d' % n for n in (50, 100, 150, 200)]
               + ['ema%d' % n for n in (50, 100, 150, 200)]
               + ['xo20_50', 'xo20_100', 'xo50_100', 'xo50_200']
               + ['dd8', 'dd10', 'dd12', 'dd15'] + ['mom63', 'mom126', 'mom252'])
VIX_GATES = (['none', 'vix15', 'vix18', 'vix20', 'vix25', 'vixsma20', 'vixsma50',
              'vixpct70', 'vixpct80', 'vixpct90'])


def vixseries(index):
    import sqlite3
    con = sqlite3.connect('file:%s?mode=ro' % (ROOT / 'backtest_data/market_data.db'),
                          uri=True)
    d = pd.read_sql_query("select date, close from market_data_unified where "
                          "symbol='INDIAVIX' and timeframe='day' order by date", con)
    con.close()
    d['date'] = pd.to_datetime(d['date'].str[:10])
    return d.drop_duplicates('date').set_index('date')['close'].reindex(index)


def gate_arr(kind, index, nb, vix):
    if kind == 'none':
        return np.zeros(len(index), bool)
    if kind.startswith('vix'):
        s = vix.dropna()
        if kind[3:].isdigit():
            off = s > float(kind[3:])
        elif kind.startswith('vixsma'):
            n = int(kind[6:]); off = s > s.rolling(n, min_periods=n).mean()
        else:
            q = float(kind[6:]) / 100.0
            off = s > s.rolling(252, min_periods=252).quantile(q)
    else:
        s = nb.dropna()
        if kind.startswith('sma'):
            n = int(kind[3:]); off = s < s.rolling(n, min_periods=n).mean()
        elif kind.startswith('ema'):
            n = int(kind[3:]); off = s < s.ewm(span=n, adjust=False).mean()
        elif kind.startswith('xo'):
            f, sl = (int(x) for x in kind[2:].split('_'))
            off = s.rolling(f, min_periods=f).mean() < s.rolling(sl, min_periods=sl).mean()
        elif kind.startswith('dd'):
            x = float(kind[2:]) / 100.0
            off = s < s.rolling(252, min_periods=252).max() * (1 - x)
        else:
            n = int(kind[3:]); off = s < s.shift(n)
    return off.shift(1).reindex(index).ffill().fillna(False).astype(bool).to_numpy()


def build(w, mode, seed=0):
    close, high, athcp, tv20 = w['close'], w['high'], w['athcp'], w['tv20']
    etf = [c for c in close.columns if em.is_etf(c)]
    tvp, prev = tv20.shift(1), close.shift(1)
    elig = tvp >= em.TV_FLOOR
    elig[etf] = False
    r = {n: close / close.shift(n) - 1 for n in (63, 126, 189, 252)}
    rs = ((2 * r[63] + r[126] + r[189] + r[252]).where(elig)
          .rank(axis=1, pct=True) * 100).shift(1)
    setup = (prev < athcp) & (prev >= 0.8 * athcp) & elig & (rs >= 70.0)
    trig = setup & (close > athcp) & athcp.notna()
    piv = athcp
    if mode == 'stop_above_candle':
        hp = high.shift(1)
        trig = trig.shift(1).fillna(False).astype(bool) & (high > hp)
        piv = hp
    elif mode == 'open_next':
        pp = athcp.shift(1)
        trig = trig.shift(1).fillna(False).astype(bool) & (high >= pp)
        piv = pp
    elif mode == 'null_random':
        rng = np.random.default_rng(10_000 + seed)
        T, E = trig.fillna(False).values, elig.reindex(columns=trig.columns).fillna(False).values
        P = np.isfinite(close.values)
        N = np.zeros_like(T)
        for i in range(T.shape[0]):
            k = int(T[i].sum())
            if not k:
                continue
            pool = np.nonzero(E[i] & P[i])[0]
            if len(pool):
                N[i, rng.choice(pool, size=min(k, len(pool)), replace=False)] = True
        trig = pd.DataFrame(N, index=trig.index, columns=trig.columns)
    return trig, piv, rs, tvp


def run(w, mode, trail, win, garr, dates):
    days = np.array([i for i, d in enumerate(dates)
                     if win[0] <= str(d.date()) <= win[1]])
    st = []
    # deterministic modes: build the frames once. Only null_random needs a per-seed draw,
    # because a fresh set of random names IS its path dependence.
    fixed = None if mode == 'null_random' else build(w, mode)
    for sd in SEEDS:
        trig, piv, rs, tvp = fixed if fixed is not None else build(w, mode, seed=sd)
        eq, trd, _, _ = em.simulate(
            sd, 'random', days, dates, w['close'].values, w['high'].values,
            w['open'].values, piv.values, w['sma50'].values, rs.values, tvp.values,
            trig.fillna(False).values if hasattr(trig, 'fillna') else trig,
            garr, True, COST, stop=STOP, slots=SLOTS, size_pct=SIZE,
            fill_close=(mode in ('close_same', 'null_random')),
            cash_yield=CASH_Y, stcg=STCG, ltcg=LTCG)
        s, _ = em.stats_from(eq, dates[days], trd, em.CAPITAL)
        st.append(s)
    return pd.DataFrame(st), days


have = set()
if OUT.exists():
    with open(OUT) as f:
        have = {(r['table'], r['label'], r['trail']) for r in csv.DictReader(f)}
else:
    with open(OUT, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def emit(table, label, trail, win, d, days, garr):
    row = dict(table=table, label=label, trail=trail, window=win[0][:4] + '-2026',
               cagr_med=round(d.cagr.median(), 2), cagr_min=round(d.cagr.min(), 2),
               cagr_max=round(d.cagr.max(), 2), dd_med=round(d.dd.median(), 2),
               calmar_med=round((d.cagr / d.dd.abs()).median(), 3),
               trades_med=int(d.n.median()),
               blocked_pct=round(100.0 * garr[days].mean(), 1), secs=0)
    with open(OUT, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    print('  %-8s %-26s t%-4s %8.2f%%  DD %7.2f%%  Cal %5.3f  blocked %4.1f%%'
          % (table, label, trail, row['cagr_med'], row['dd_med'], row['calmar_med'],
             row['blocked_pct']), flush=True)


MODES = [('close_same', 'buy at breakout close'),
         ('stop_above_candle', 'stop above breakout candle'),
         ('open_next', 'next-day stop at pivot'),
         ('open_same', 'LOOK-AHEAD reference')]

for trail in [10, 15, 20, 30, 50, 75, 100, 150]:
    todo = [(m, lab) for m, lab in MODES if ('A', lab, str(trail)) not in have]
    if trail in (50, 75, 100, 150):
        todo += [(m, lab) for m, lab in [('close_same', 'rule: buy at close'),
                                         ('null_random', 'null: random names')]
                 if ('B', lab, str(trail)) not in have]
    if trail == 75:
        todo += [('GATES', 'GATES')] if ('C', 'none', '75') not in have else []
    if not todo:
        continue
    print('trail-%d ...' % trail, flush=True)
    w = em.load_frames('2005-01-01', trail_sma=trail)
    dates = w['close'].index
    nb = w['close'].get('NIFTYBEES')
    vix = vixseries(dates)
    zero = np.zeros(len(dates), bool)
    for m, lab in todo:
        if m == 'GATES':
            for g in PRICE_GATES:
                if ('C', g, '75') in have:
                    continue
                d, days = run(w, 'close_same', 75, W_LONG, gate_arr(g, dates, nb, vix), dates)
                emit('C', g, '75', W_LONG, d, days, gate_arr(g, dates, nb, vix))
            for g in VIX_GATES:
                if ('C', 'VIX:' + g, '75') in have:
                    continue
                d, days = run(w, 'close_same', 75, W_VIX, gate_arr(g, dates, nb, vix), dates)
                emit('C', 'VIX:' + g, '75', W_VIX, d, days, gate_arr(g, dates, nb, vix))
            continue
        tbl = 'B' if lab.startswith(('rule:', 'null:')) else 'A'
        d, days = run(w, m, trail, W_LONG, zero, dates)
        emit(tbl, lab, str(trail), W_LONG, d, days, zero)
print('AFTERTAX_DONE', flush=True)
