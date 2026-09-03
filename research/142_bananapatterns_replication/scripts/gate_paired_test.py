"""Paired-by-seed robustness test: is DD10 bankable, and does 16 slots help?

For each seed the SAME selection path runs under DD10 and under no-gate, so the
difference is purely the gate's doing (seed luck cancels). 30 seeds. Also paired
across sizing (8 slots @18.75% vs 16 @6.25%). Reports per-seed win rates, paired
uplift distribution, and per-seed 2008 protection. 2006->now, pre-tax.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
import bluesky_replay as br

print('loading frames ...', flush=True)
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
nb = close['NIFTYBEES'].dropna()
weak_dd10 = (nb < 0.9*nb.rolling(252).max()).shift(1).reindex(dates)\
    .ffill().fillna(False).astype(bool).values
weak_off = np.zeros(len(dates), dtype=bool)
m08 = (dates[days] >= '2008-01-01') & (dates[days] <= '2009-06-30')

SEEDS = range(1, 31)
n_yrs = len(days) / 247.0


def run(seed, weak, slots, size):
    eq, _, _ = br.simulate(seed, 'random', days, dates, C, H, O, ATH, S,
                           RSv, TVv, trig, weak, True, 0.0025, stop=0.08,
                           slots=slots, size_pct=size)
    eq = np.asarray(eq, dtype=float)
    dd = eq / np.maximum.accumulate(eq) - 1
    return dict(cagr=(eq[-1]/eq[0])**(1/n_yrs)-1,
                maxdd=float(dd.min()*100), dd08=float(dd[m08].min()*100))

rows = []
for seed in SEEDS:
    r = dict(seed=seed)
    for gname, wk in [('dd10', weak_dd10), ('off', weak_off)]:
        for sname, slots, size in [('s8', 8, 0.1875), ('s16', 16, 0.0625)]:
            res = run(seed, wk, slots, size)
            for k, v in res.items():
                r[f'{gname}_{sname}_{k}'] = v
    rows.append(r)
    if seed % 5 == 0:
        print(f'seed {seed}/30 done', flush=True)
df = pd.DataFrame(rows).set_index('seed')
df.to_csv(STUDY / 'results' / 'gate_paired_test.csv')

for sname in ['s8', 's16']:
    d_cagr = (df[f'dd10_{sname}_cagr'] - df[f'off_{sname}_cagr']) * 100
    d_dd08 = df[f'dd10_{sname}_dd08'] - df[f'off_{sname}_dd08']
    d_mdd = df[f'dd10_{sname}_maxdd'] - df[f'off_{sname}_maxdd']
    print(f'\n== PAIRED DD10 minus no-gate, {sname} (30 seeds) ==')
    print(f'CAGR uplift: median {d_cagr.median():+.1f}pp  [{d_cagr.min():+.1f}..{d_cagr.max():+.1f}]  '
          f'gate wins {int((d_cagr > 0).sum())}/30 seeds')
    print(f'2008 DD improvement: median {d_dd08.median():+.1f}pp  '
          f'gate better {int((d_dd08 > 0).sum())}/30 seeds')
    print(f'overall maxDD change: median {d_mdd.median():+.1f}pp  '
          f'gate better {int((d_mdd > 0).sum())}/30 seeds')

for gname in ['dd10', 'off']:
    c8, c16 = df[f'{gname}_s8_cagr']*100, df[f'{gname}_s16_cagr']*100
    print(f'\n== {gname}: 8 slots vs 16 slots (30 seeds) ==')
    print(f'8 slots : CAGR median {c8.median():.1f}%  [{c8.min():.1f}..{c8.max():.1f}]  spread {c8.max()-c8.min():.1f}pp')
    print(f'16 slots: CAGR median {c16.median():.1f}%  [{c16.min():.1f}..{c16.max():.1f}]  spread {c16.max()-c16.min():.1f}pp')
    print(f'worst-seed change 8->16: {c16.min()-c8.min():+.1f}pp; median change {c16.median()-c8.median():+.1f}pp')
print('\nDONE', flush=True)
