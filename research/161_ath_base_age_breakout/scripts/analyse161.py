"""
research/161 — read the 864-cell sweep and answer Arun's question directly:
does the AGE of the previous all-time high (X), and VOLUME confirmation (K), add value?
"""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / 'results'
BH_CAGR, BH_DD = 12.29, -59.71

df = pd.concat([pd.read_csv(p) for p in sorted(RES.glob('sweep161_shard*.csv'))],
               ignore_index=True).drop_duplicates('cell')
df.to_csv(RES / 'sweep161.csv', index=False)
print('cells completed: %d of 864 (%d skipped for <20 events)' % (len(df), 864 - len(df)))

OA = df[(df.x_bars == 0) & (df.depth_min == 0) & (df.vol_K == 0) & (df.saucer == 0)
        & (df.exit == 'SMA15') & (df.hard_stop == 1)]
if len(OA):
    o = OA.iloc[0]
    print('\n=== THE HONEST OA-PROXY CELL (plain ATH, X=0, no volume, 15-SMA trail + -8% stop) ===')
    print('  CAGR %.2f%% [%.2f..%.2f]  MaxDD %.2f%%  Calmar %.3f  WR %.1f%%  exp %+.2f%%  '
          'trades/yr %.1f  n_events %d'
          % (o.cagr_med, o.cagr_min, o.cagr_max, o.maxdd_med, o.calmar_med, o.win_rate,
             o.expectancy, o.trades_per_yr, o.n_events))
    OA_CAGR = float(o.cagr_med)
else:
    OA_CAGR = float('nan')

print('\n=== DOES ANY CELL CLEAR 20%% AFTER TAX? ===')
over = df[df.cagr_med >= 20.0]
print('  cells >= 20%%: %d of %d (%.0f%%)' % (len(over), len(df), 100.0 * len(over) / len(df)))
print('  best CAGR %.2f%%  |  best Calmar %.3f' % (df.cagr_med.max(), df.calmar_med.max()))
beat = df[(df.cagr_med > BH_CAGR) & (df.maxdd_med > BH_DD)]
print('  beating NIFTYBEES on CAGR and DD: %d (%.0f%%)' % (len(beat), 100.0 * len(beat) / len(df)))
print('  beating the OA proxy on CAGR: %d (%.0f%%)'
      % ((df.cagr_med > OA_CAGR).sum(), 100.0 * (df.cagr_med > OA_CAGR).mean()))

cols = ['x_bars', 'depth_min', 'vol_K', 'saucer', 'exit', 'hard_stop', 'n_events',
        'cagr_med', 'cagr_min', 'maxdd_med', 'calmar_med', 'win_rate', 'avg_win',
        'avg_loss', 'expectancy', 'max_loss_streak', 'trades_per_yr']
print('\n=== TOP 15 BY CAGR ===')
print(df.nlargest(15, 'cagr_med')[cols].to_string(index=False))
print('\n=== TOP 10 BY CALMAR ===')
print(df.nlargest(10, 'calmar_med')[cols].to_string(index=False))

print('\n=== MARGINAL EFFECT OF EACH AXIS (median across all other settings) ===')
for ax in ['x_bars', 'depth_min', 'vol_K', 'saucer', 'exit', 'hard_stop']:
    g = df.groupby(ax).agg(cagr=('cagr_med', 'median'), dd=('maxdd_med', 'median'),
                           calmar=('calmar_med', 'median'), wr=('win_rate', 'median'),
                           exp=('expectancy', 'median'), tpy=('trades_per_yr', 'median'),
                           n=('cell', 'size'))
    print('\n  -- %s --' % ax)
    print(g.round(3).to_string())

# ---- THE table Arun asked for: X (rows) x K (cols), best exit, saucer off/on ----
best_exit = df.groupby(['exit', 'hard_stop'])['cagr_med'].median().idxmax()
print('\n\n########## X x K AT THE BEST EXIT (%s, stop=%d), depth=any ##########'
      % (best_exit[0], best_exit[1]))
for sau in (0, 1):
    sub = df[(df.exit == best_exit[0]) & (df.hard_stop == best_exit[1])
             & (df.depth_min == 0) & (df.saucer == sau)]
    if not len(sub):
        continue
    print('\n--- saucer requirement: %s ---' % ('ON' if sau else 'OFF'))
    for metric, lab in (('cagr_med', 'CAGR %'), ('win_rate', 'win rate %'),
                        ('expectancy', 'expectancy %/trade'), ('n_events', 'events'),
                        ('maxdd_med', 'MaxDD %'), ('calmar_med', 'Calmar')):
        t = sub.pivot_table(index='x_bars', columns='vol_K', values=metric)
        t.columns = ['K none' if c == 0 else 'K>=%g' % c for c in t.columns]
        print('\n  %s' % lab)
        print(t.round(2).to_string())

print('\n=== PLATEAU: neighbourhood of the best cell ===')
w = df.loc[df.cagr_med.idxmax()]
print('  winner: %s  CAGR %.2f%%  DD %.2f%%  Calmar %.3f  WR %.1f%%'
      % (w.cell, w.cagr_med, w.maxdd_med, w.calmar_med, w.win_rate))
for ax in ['x_bars', 'depth_min', 'vol_K', 'saucer', 'exit', 'hard_stop']:
    nb = df.copy()
    for other in ['x_bars', 'depth_min', 'vol_K', 'saucer', 'exit', 'hard_stop']:
        if other != ax:
            nb = nb[nb[other] == w[other]]
    print('\n  vary %s:' % ax)
    print(nb.sort_values(ax)[[ax, 'n_events', 'cagr_med', 'maxdd_med', 'calmar_med',
                              'win_rate', 'expectancy']].to_string(index=False))

print('\n=== distribution of cell CAGR ===')
print(df.cagr_med.describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
