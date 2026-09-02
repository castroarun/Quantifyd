"""Client tearsheet + multi-benchmark comparison for research/142 p3D (headline config).

- Representative NAV = the seed with MEDIAN terminal value from the 10-seed ensemble
  (stated on the chart; the ensemble range is printed in the caption).
- Tearsheet vs NIFTYBEES (2006->2025) via research/_utilities/tearsheet.py.
- Comparison chart (2011-> common start): strategy vs NIFTYBEES vs NIFTYMIDCAP150
  vs NIFTYSMLCAP250, growth of Rs 100, log scale, CAGRs annotated.
"""
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
RES = STUDY / 'results'
sys.path.insert(0, str(ROOT / 'research' / '_utilities'))
from tearsheet import generate_tearsheet  # noqa: E402

TAG = 'p3D_ext2026'

eq = pd.read_csv(RES / f'replica_{TAG}_equity.csv', index_col=0, parse_dates=True)
term = eq.iloc[-1]
med_seed = (term - term.median()).abs().idxmin()
nav = eq[med_seed]
print(f'representative seed: {med_seed} terminal {term[med_seed]:,.0f} '
      f'(ensemble median {term.median():,.0f}, range {term.min():,.0f}-{term.max():,.0f})')

db = sqlite3.connect(str(ROOT / 'backtest_data' / 'market_data.db'))

def series(sym):
    df = pd.read_sql_query(
        "select date, close from market_data_unified where symbol=? and timeframe='day' "
        "order by date", db, params=(sym,))
    df['date'] = pd.to_datetime(df['date'].str[:10])
    return df.drop_duplicates('date').set_index('date')['close']

nb = series('NIFTYBEES').loc[nav.index[0]:nav.index[-1]]

meta = dict(
    period='2006-01-01 to 2025-12-31',
    config='Blue-sky ATH breakout: close>ATH-close, IBD-RS>=70, TV>=Rs5cr, '
           'mcap>=Rs500cr (PIT proxy), -8% stop, 50-SMA trail, 8 slots, '
           'NIFTYBEES<SMA200 gate, realistic fills, 25bps/side',
    note=f'Median seed of 10-seed selection ensemble (CAGR range 27.9-34.4%). '
         f'Survivorship-biased pre-2015 (Kite lists current instruments only).')
mom_nav = pd.read_csv(ROOT / 'research' / '75_nifty250_momentum_top15' / 'results' / 'nav_armed_spec.csv',
                      index_col=0, parse_dates=True)['nav']
generate_tearsheet(nav, nb, 'BlueSky ATH Breakout (research/142)', meta=meta,
                   out_dir=str(RES),
                   extra_nav=mom_nav, extra_label='Momentum r/75 (net, gated)')
print('tearsheet written to results/')

# ---- multi-benchmark comparison from 2011 ----
mom = pd.read_csv(ROOT / 'research' / '75_nifty250_momentum_top15' / 'results' / 'nav_armed_spec.csv',
                  index_col=0, parse_dates=True)['nav']
bm = {'Momentum r/75 (armed spec, net, index gate ON)': mom,
      'NIFTYBEES (Nifty 50)': nb,
      'NIFTYMIDCAP150': series('NIFTYMIDCAP150'),
      'NIFTYSMLCAP250': series('NIFTYSMLCAP250')}
start = max([s.index[0] for s in bm.values()] + [nav.index[0], pd.Timestamp('2011-01-03')])
end = nav.index[-1]
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=140)
for label, s in [('BlueSky replica (median seed, net)', nav)] + list(bm.items()):
    s = s.loc[start:end].dropna()
    if not len(s):
        continue
    g = s / s.iloc[0] * 100
    yrs = (g.index[-1] - g.index[0]).days / 365.25
    cagr = (g.iloc[-1] / 100) ** (1 / yrs) - 1
    lw, z = (2.2, 5) if 'BlueSky' in label else (1.3, 3)
    ax.plot(g.index, g.values, label=f'{label}  ({cagr*100:.1f}% CAGR)', lw=lw, zorder=z)
ax.set_yscale('log')
ax.set_title('Growth of Rs 100 (log)  -  2011-2025  -  BlueSky replica vs indices\n'
             'net of 25bps/side, realistic fills, mcap>=500cr, weak-market gate ON  -  '
             'median of 10 selection seeds', fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(RES / 'bluesky_vs_indices.png')
print('comparison chart written: results/bluesky_vs_indices.png')

for label, s in bm.items():
    s = s.loc[start:end].dropna()
    y = s.resample('YE').last().pct_change() * 100
    print(label, {d.year: round(v, 1) for d, v in y.dropna().items()})
