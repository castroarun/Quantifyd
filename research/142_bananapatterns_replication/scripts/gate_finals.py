"""Gate bake-off FINALS: full 2006->now window (includes 2008 + 2020 crashes).

Arun's criterion: balance — do not give up bull trades across two decades just to
dodge a once-a-decade crash. So each finalist reports BOTH sides of the trade:
terminal multiple (what the gate costs in upside) and crash-window drawdowns
(what it buys in defense: worst DD inside 2008-2009 and inside 2020).

NIFTYBEES-based / universe-breadth / vol gates cover the full window. Index-series
gates (NIFTY50/500/MIDCAP/SMLCAP start 2011 in our DB) are ffilled-open before 2011
— their 2008 column is meaningless and printed as n/a-2008 (still shown for the
2011+ part of the window; judge those on the two-window sweep + 2011+ behavior).
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
import bluesky_replay as br

OUT = STUDY / 'results' / 'gate_finals.csv'
FIELDS = ['cell', 'med_term_x', 'min_x', 'max_x', 'med_maxdd_pct',
          'dd_2008_pct', 'dd_2020_pct', 'blocked_pct', 'covers_2008']

print('loading frames (2004-06-01 ->) ...', flush=True)
w = br.load_frames('2004-06-01', trail_sma=20)
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
days = np.array([i for i, d in enumerate(dates) if str(d.date()) >= '2006-01-01'])
ddates = dates[days]


def align(raw):
    return raw.shift(1).reindex(dates).ffill().fillna(False).astype(bool).values


def w_ma(sym, kind, n):
    s = close[sym].dropna()
    m = s.ewm(span=n, adjust=False).mean() if kind == 'EMA' else s.rolling(n).mean()
    return align(s < m)


def w_dd(sym, x):
    s = close[sym].dropna()
    return align(s < (1 - x) * s.rolling(252).max())


def w_mom(sym, n):
    s = close[sym].dropna()
    return align(s / s.shift(n) - 1 < 0)


def w_vol(sym, x):
    s = close[sym].dropna()
    v = np.log(s / s.shift(1)).rolling(20).std() * np.sqrt(252)
    return align(v > x)


def w_breadth(n, thr):
    above = (close > close.rolling(n).mean()).sum(axis=1)
    denom = close.notna().sum(axis=1).clip(lower=1)
    return align((above / denom) < thr)


CELLS = [
    ('gate_OFF', np.zeros(len(dates), dtype=bool), True),
    ('NIFTYBEES_SMA200 (incumbent)', w_ma('NIFTYBEES', 'SMA', 200), True),
    ('NIFTYBEES_EMA100', w_ma('NIFTYBEES', 'EMA', 100), True),
    ('NIFTYBEES_DD10', w_dd('NIFTYBEES', 0.10), True),
    ('NIFTYBEES_DD5', w_dd('NIFTYBEES', 0.05), True),
    ('NIFTYBEES_MOM63', w_mom('NIFTYBEES', 63), True),
    ('NIFTYBEES_MOM126', w_mom('NIFTYBEES', 126), True),
    ('BREADTH_sma200_lt40', w_breadth(200, 0.40), True),
    ('BREADTH_sma200_lt50', w_breadth(200, 0.50), True),
    ('NIFTYBEES_VOL20_gt25', w_vol('NIFTYBEES', 0.25), True),
    ('NIFTY50_DD10 (2011+)', w_dd('NIFTY50', 0.10), False),
    ('NIFTYSMLCAP250_MOM63 (2011+)', w_mom('NIFTYSMLCAP250', 63), False),
    ('NIFTYSMLCAP250_DD10 (2011+)', w_dd('NIFTYSMLCAP250', 0.10), False),
]

with open(OUT, 'w', newline='') as fh:
    csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

m08 = (ddates >= '2008-01-01') & (ddates <= '2009-06-30')
m20 = (ddates >= '2020-01-01') & (ddates <= '2020-12-31')

for cell, weak_arr, covers08 in CELLS:
    t0 = time.time()
    terms, dds, d08s, d20s = [], [], [], []
    for seed in range(1, 11):
        eq, trades, _ = br.simulate(seed, 'random', days, dates, C, H, O, ATH, S,
                                    RSv, TVv, trig, weak_arr, True, 0.0025,
                                    stop=0.08, slots=8)
        eq = np.asarray(eq, dtype=float)
        terms.append(eq[-1] / eq[0])
        dd = eq / np.maximum.accumulate(eq) - 1
        dds.append(float(dd.min() * 100))
        d08s.append(float(dd[m08].min() * 100) if m08.any() else np.nan)
        d20s.append(float(dd[m20].min() * 100) if m20.any() else np.nan)
    row = dict(cell=cell, med_term_x=round(float(np.median(terms)), 1),
               min_x=round(min(terms), 1), max_x=round(max(terms), 1),
               med_maxdd_pct=round(float(np.median(dds)), 1),
               dd_2008_pct=round(float(np.median(d08s)), 1),
               dd_2020_pct=round(float(np.median(d20s)), 1),
               blocked_pct=round(100 * weak_arr[days].mean(), 1),
               covers_2008=covers08)
    with open(OUT, 'a', newline='') as fh:
        csv.DictWriter(fh, fieldnames=FIELDS).writerow(row)
    print(f"{cell:32s} x{row['med_term_x']:>8} [{row['min_x']}..{row['max_x']}] "
          f"dd {row['med_maxdd_pct']}% | 2008 {row['dd_2008_pct']}% | 2020 {row['dd_2020_pct']}% "
          f"| blocked {row['blocked_pct']}% ({time.time()-t0:.0f}s)", flush=True)
print('DONE', flush=True)
