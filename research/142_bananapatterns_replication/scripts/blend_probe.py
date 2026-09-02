"""Correlation + 50-50 blend: BlueSky (p5_final, per-seed) x Momentum r/75 armed spec.

Blend = 50/50 capital, rebalanced MONTHLY (monthly leg returns averaged, compounded).
Reports: daily & monthly return correlation, per-seed blend stats (median/range),
standalone stats on the common window, and median per-year blend returns.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]

TAG = sys.argv[1] if len(sys.argv) > 1 else 'p5_final'
blue = pd.read_csv(STUDY / 'results' / f'replica_{TAG}_equity.csv',
                   index_col=0, parse_dates=True)
mom = pd.read_csv(ROOT / 'research' / '75_nifty250_momentum_top15' / 'results' / 'nav_armed_spec.csv',
                  index_col=0, parse_dates=True)['nav']

idx = blue.index.intersection(mom.index)
blue = blue.loc[idx]
mom = mom.loc[idx]
print(f'tag={TAG} common window: {idx[0].date()} -> {idx[-1].date()}  ({len(idx)} days)')

med_seed = (blue.iloc[-1] - blue.iloc[-1].median()).abs().idxmin()
bs = blue[med_seed]
print(f'median seed: {med_seed}')
dr_b, dr_m = bs.pct_change(), mom.pct_change()
print(f'daily return correlation (median seed): {dr_b.corr(dr_m):.3f}')
mb = bs.resample('ME').last().pct_change()
mm = mom.resample('ME').last().pct_change()
print(f'monthly return correlation: {mb.corr(mm):.3f}')


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    return cagr * 100, dd * 100, nav.iloc[-1] / nav.iloc[0]


rows = []
blend_navs = {}
for seed in blue.columns:
    b_m = blue[seed].resample('ME').last().pct_change().fillna(0)
    m_m = mom.resample('ME').last().pct_change().fillna(0)
    blend = (1 + 0.5 * b_m + 0.5 * m_m).cumprod()
    blend_navs[seed] = blend
    c, d, x = stats(blend)
    rows.append(dict(seed=seed, cagr=c, dd=d, x=x))
bdf = pd.DataFrame(rows)

cb, db, xb = stats(bs)
cm, dm, xm = stats(mom)
print(f'\nBlueSky alone (median seed): CAGR {cb:.1f}%  DD {db:.1f}%  {xb:.0f}x')
print(f'Momentum alone:              CAGR {cm:.1f}%  DD {dm:.1f}%  {xm:.0f}x')
print(f'50-50 blend (10 seeds):      CAGR median {bdf.cagr.median():.1f}% '
      f'[{bdf.cagr.min():.1f}..{bdf.cagr.max():.1f}]  '
      f'DD median {bdf.dd.median():.1f}% worst {bdf.dd.min():.1f}%  '
      f'x median {bdf.x.median():.0f}')

med_blend_seed = (bdf.set_index('seed').x - bdf.x.median()).abs().idxmin()
bn = blend_navs[med_blend_seed]
y = bn.resample('YE').last().pct_change() * 100
y.iloc[0] = (bn.resample('YE').last().iloc[0] / 1.0 - 1) * 100
print('\nblend per-year % (median-x seed):',
      {d.year: round(v, 1) for d, v in y.items()})
