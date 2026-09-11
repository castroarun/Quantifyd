"""
research/159 — read the 2,016-cell sweep: is the winner a plateau or a peak, does ANY
cell clear the 20% floor, and what does each axis actually do?

Ranking metric is the pre-registered one: after-tax net CAGR, Calmar as tie-break.
"""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / 'results'
BH_CAGR, BH_DD = 12.29, -59.71          # NIFTYBEES buy-and-hold, same window

df = pd.concat([pd.read_csv(p) for p in sorted(RES.glob('sweep_v3_shard*.csv'))],
               ignore_index=True)
df = df.drop_duplicates('cell')
df.to_csv(RES / 'sweep_v3.csv', index=False)
print('cells completed: %d of 2,016' % len(df))
print('events per cell: %d .. %d\n' % (df.n_events.min(), df.n_events.max()))

print('=== DOES ANY CELL CLEAR THE 20%% FLOOR? ===')
over = df[df.cagr_med >= 20.0]
print('  cells with 10-seed median CAGR >= 20%%: %d of %d' % (len(over), len(df)))
print('  best CAGR in the whole sweep: %.2f%%  (cell %s)'
      % (df.cagr_med.max(), df.loc[df.cagr_med.idxmax(), 'cell']))
print('  best Calmar in the whole sweep: %.3f (cell %s)'
      % (df.calmar_med.max(), df.loc[df.calmar_med.idxmax(), 'cell']))

beat = df[(df.cagr_med > BH_CAGR) & (df.maxdd_med > BH_DD)]
print('\n=== vs NIFTYBEES (%.2f%% / %.2f%%) ===' % (BH_CAGR, BH_DD))
print('  cells beating it on CAGR *and* drawdown: %d (%.0f%%)'
      % (len(beat), 100.0 * len(beat) / len(df)))

print('\n=== TOP 15 BY CAGR (the pre-registered metric) ===')
cols = ['exit', 'hard_stop', 'time_stop', 'shelf_S', 'vol_K', 'ath', 'obv_filter',
        'market_gate', 'n_events', 'cagr_med', 'maxdd_med', 'calmar_med', 'trades']
top = df.nlargest(15, 'cagr_med')
print(top[cols].to_string(index=False))

print('\n=== MARGINAL EFFECT OF EACH AXIS (median CAGR across all other settings) ===')
for ax in ['exit', 'hard_stop', 'time_stop', 'shelf_S', 'vol_K', 'ath', 'obv_filter',
           'market_gate']:
    g = df.groupby(ax).agg(cagr=('cagr_med', 'median'), dd=('maxdd_med', 'median'),
                           calmar=('calmar_med', 'median'), n=('cell', 'size'))
    print('\n  -- %s --' % ax)
    print(g.round(3).to_string())

print('\n=== PLATEAU CHECK around the winner ===')
w = df.loc[df.cagr_med.idxmax()]
print('  winner: %s  CAGR %.2f%%  DD %.2f%%  Calmar %.3f'
      % (w.cell, w.cagr_med, w.maxdd_med, w.calmar_med))
for ax in ['exit', 'hard_stop', 'time_stop', 'shelf_S', 'vol_K', 'ath', 'obv_filter', 'market_gate']:
    nb = df.copy()
    for other in ['exit', 'hard_stop', 'time_stop', 'shelf_S', 'vol_K', 'ath',
                  'obv_filter', 'market_gate']:
        if other != ax:
            nb = nb[nb[other] == w[other]]
    vals = nb.sort_values(ax)[[ax, 'cagr_med', 'maxdd_med', 'calmar_med']]
    print('\n  vary %s (all else at the winner):' % ax)
    print(vals.to_string(index=False))

print('\n=== EXIT FAMILY, best cell in each ===')
best_by_exit = df.loc[df.groupby('exit')['cagr_med'].idxmax()]
print(best_by_exit[cols].sort_values('cagr_med', ascending=False).to_string(index=False))

print('\n=== distribution of cell CAGR ===')
q = df.cagr_med.describe(percentiles=[.1, .25, .5, .75, .9])
print(q.round(2).to_string())
