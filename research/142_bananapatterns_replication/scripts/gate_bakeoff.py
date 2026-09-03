"""Gate bake-off for the Open Alpha adopted spec (see GATE_BAKEOFF_DAILY_SWEEP_STATUS.md).

Varies ONLY the weak-market gate: series x construction. All other spec constants
frozen (trail-20, -8% stop, 8 slots, RS>=70, TV floor, 25bps, realistic fills,
10-seed random-selection ensemble).

Gate constructions:
  - price-vs-MA:   entries blocked while close < MA(N) of the gate series (SMA or EMA)
  - MA crossover:  entries blocked while MA(fast) < MA(slow)  (e.g. 50/200 death-cross)
All NaN-robust: computed on the dropna'd series, shift(1) for causality, re-aligned
to the union calendar with ffill.

Windows: W1 2020-01-01 -> last DB day (primary), W2 2016-06-01 -> 2019-12-31
(regime validation; every cell runs both). Incremental CSV; skips completed cells.
"""
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
import bluesky_replay as br

OUT = STUDY / 'results' / 'gate_bakeoff.csv'
FIELDS = ['cell', 'window', 'med_term_x', 'min_x', 'max_x', 'med_maxdd_pct',
          'med_trades', 'blocked_days_pct', 'secs']

SERIES = ['NIFTYBEES', 'NIFTY50', 'NIFTY500', 'NIFTYMIDCAP150', 'NIFTYSMLCAP250']
PRICE_MA = [('SMA', 100), ('SMA', 150), ('SMA', 200),
            ('EMA', 100), ('EMA', 150), ('EMA', 200)]
CROSSES = [('SMA', 50, 200), ('EMA', 20, 100), ('SMA', 20, 100)]
DD_GATES = [0.05, 0.10]        # blocked while close < (1-x) * rolling 252d high
MOM_GATES = [63, 126]          # blocked while N-day return < 0
BREADTH = [(200, 0.40), (200, 0.50), (50, 0.40), (50, 0.50)]  # universe % above own SMA
VOL_GATES = [0.18, 0.25]       # blocked while NIFTY50 20d realized vol (ann.) above x
WINDOWS = [('W1_2020_now', '2020-01-01', None),
           ('W2_2016_2019', '2016-06-01', '2019-12-31')]

print('loading frames (2015-06-01 ->) ...', flush=True)
w = br.load_frames('2015-06-01', trail_sma=20)
close, high, open_, athcp, sma, tv20 = (w[k] for k in
    ('close', 'high', 'open', 'athcp', 'sma50', 'tv20'))
etf = [c for c in close.columns if br.ETF_RE.search(c)]
tv_prev = tv20.shift(1)
prev_close = close.shift(1)
elig = tv_prev >= br.TV_FLOOR
elig[etf] = False
score = 2*(close/close.shift(63)-1) + (close/close.shift(126)-1) \
    + (close/close.shift(189)-1) + (close/close.shift(252)-1)
rs = (score.where(elig).rank(axis=1, pct=True)*100).shift(1)
setup = (prev_close < athcp) & (prev_close >= 0.8*athcp) & elig & (rs >= 70.0)
trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
dates = close.index
C, H, O, ATH, S = close.values, high.values, open_.values, athcp.values, sma.values
RSv, TVv = rs.values, tv_prev.values

for s_ in SERIES:  # sanity print (r/64: some Kite index series are corrupt)
    ser = close[s_].dropna()
    print(f'  {s_}: {ser.index[0].date()} -> {ser.index[-1].date()} '
          f'n={len(ser)} first={ser.iloc[0]:.1f} last={ser.iloc[-1]:.1f}', flush=True)


def ma(s, kind, n):
    return s.ewm(span=n, adjust=False).mean() if kind == 'EMA' else s.rolling(n).mean()


def weak_from(sym, kind, n, fast=None):
    s = close[sym].dropna()
    if fast is None:
        raw = s < ma(s, kind, n)
    else:
        raw = ma(s, kind, fast) < ma(s, kind, n)
    return raw.shift(1).reindex(dates).ffill().fillna(False).astype(bool).values


def weak_dd(sym, x):
    s = close[sym].dropna()
    raw = s < (1 - x) * s.rolling(252).max()
    return raw.shift(1).reindex(dates).ffill().fillna(False).astype(bool).values


def weak_mom(sym, n):
    s = close[sym].dropna()
    raw = s / s.shift(n) - 1 < 0
    return raw.shift(1).reindex(dates).ffill().fillna(False).astype(bool).values


def weak_vol(x):
    s = close['NIFTY50'].dropna()
    vol = np.log(s / s.shift(1)).rolling(20).std() * np.sqrt(252)
    return (vol > x).shift(1).reindex(dates).ffill().fillna(False).astype(bool).values


def weak_breadth(n, thr):
    smaN = close.rolling(n).mean()
    above = (close > smaN).sum(axis=1)
    denom = close.notna().sum(axis=1).clip(lower=1)
    raw = (above / denom) < thr
    return raw.shift(1).reindex(dates).ffill().fillna(False).astype(bool).values


cells = [('gate_OFF', np.zeros(len(dates), dtype=bool))]
for sym in SERIES:
    for kind, n in PRICE_MA:
        cells.append((f'{sym}_{kind}{n}', weak_from(sym, kind, n)))
    for kind, f, sl in CROSSES:
        cells.append((f'{sym}_X_{kind}{f}-{sl}', weak_from(sym, kind, sl, fast=f)))
    for x in DD_GATES:
        cells.append((f'{sym}_DD{int(x*100)}', weak_dd(sym, x)))
    for n in MOM_GATES:
        cells.append((f'{sym}_MOM{n}', weak_mom(sym, n)))
for n, thr in BREADTH:
    cells.append((f'BREADTH_sma{n}_lt{int(thr*100)}', weak_breadth(n, thr)))
for x in VOL_GATES:
    cells.append((f'VOL20_gt{int(x*100)}', weak_vol(x)))

done = set()
if OUT.exists():
    with open(OUT) as fh:
        done = {(r['cell'], r['window']) for r in csv.DictReader(fh)}
    print(f'skipping {len(done)} completed cells', flush=True)
else:
    with open(OUT, 'w', newline='') as fh:
        csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

total = len(cells) * len(WINDOWS)
k = 0
for wname, wstart, wend in WINDOWS:
    days = np.array([i for i, d in enumerate(dates)
                     if str(d.date()) >= wstart and (wend is None or str(d.date()) <= wend)])
    for cell, weak_arr in cells:
        k += 1
        if (cell, wname) in done:
            continue
        t0 = time.time()
        terms, dds, ntr = [], [], []
        for seed in range(1, 11):
            eq, trades, _ = br.simulate(seed, 'random', days, dates, C, H, O, ATH, S,
                                        RSv, TVv, trig, weak_arr, True, 0.0025,
                                        stop=0.08, slots=8)
            eq = np.asarray(eq, dtype=float)
            terms.append(eq[-1] / eq[0])
            dds.append(float((eq / np.maximum.accumulate(eq) - 1).min() * 100))
            ntr.append(len(trades))
        row = dict(cell=cell, window=wname,
                   med_term_x=round(float(np.median(terms)), 2),
                   min_x=round(min(terms), 2), max_x=round(max(terms), 2),
                   med_maxdd_pct=round(float(np.median(dds)), 1),
                   med_trades=int(np.median(ntr)),
                   blocked_days_pct=round(100 * weak_arr[days].mean(), 1),
                   secs=round(time.time() - t0, 1))
        with open(OUT, 'a', newline='') as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow(row)
        print(f'[{k}/{total}] {wname} {cell}: med x{row["med_term_x"]} '
              f'dd {row["med_maxdd_pct"]}% blocked {row["blocked_days_pct"]}% '
              f'({row["secs"]}s)', flush=True)

print('DONE', flush=True)
