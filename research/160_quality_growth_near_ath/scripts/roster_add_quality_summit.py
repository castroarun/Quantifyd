# -*- coding: utf-8 -*-
"""Quality Summit (research/160 Family B) on the honest-entries roster, same window as every
other book.

Reads the 12-offset after-tax equity paths of the Family-B headline cell
(`results/F_Bb7_equity.csv`, 2018-08-01 -> 2026-09-10) and the roster's after-tax curves
(`research/159_oa_honest_reoptimization/results/all_systems_after_tax.csv`, 2016 ->
2026-09-04), rebases everything at 2018-08-01 (the first month with fundamentals coverage,
which is why Quality Summit cannot be drawn earlier) and produces:

  frontend/public/mpf-honest-roster-2026-09-qs.png   log growth-of-100 + drawdown, common window
  results/quality_summit_roster.json                 every number the page needs

House rules honoured: the drawn Quality Summit path is the MEDIAN-CAGR OFFSET, never an
average of the twelve; every per-year drawdown is measured from the running peak of the
FULL curve, not from the year's first bar.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/arun/quantifyd')
R160 = ROOT / 'research/160_quality_growth_near_ath/results'
R159 = ROOT / 'research/159_oa_honest_reoptimization/results'
OUT_PNG = ROOT / 'frontend/public/mpf-honest-roster-2026-09-qs.png'
OUT_JSON = R160 / 'quality_summit_roster.json'

START = pd.Timestamp('2018-08-01')

qs = pd.read_csv(R160 / 'F_Bb7_equity.csv', index_col=0, parse_dates=True)
others = pd.read_csv(R159 / 'all_systems_after_tax.csv', index_col=0, parse_dates=True)
END = min(qs.index[-1], others.index[-1])

qs = qs.loc[START:END]
others = others.loc[START:END]


def cagr(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1


def maxdd(s):
    return (s / s.cummax() - 1).min()


# median-CAGR offset is the drawn path
cg = qs.apply(cagr)
med_col = (cg - cg.median()).abs().idxmin()
qs_path = qs[med_col].rename('Quality Summit')
print('Quality Summit offsets: CAGR median %.2f%% [%.2f .. %.2f], drawn path %s'
      % (cg.median() * 100, cg.min() * 100, cg.max() * 100, med_col))

curves = pd.concat([others, qs_path], axis=1).ffill().dropna()
curves = curves / curves.iloc[0]


def yearly(s):
    out = {}
    peak = s.cummax()
    for y, grp in s.groupby(s.index.year):
        prev = s.loc[:grp.index[0] - pd.Timedelta(days=1)]
        base = prev.iloc[-1] if len(prev) else grp.iloc[0]
        ret = grp.iloc[-1] / base - 1
        dd = (grp / peak.loc[grp.index] - 1).min()
        out[int(y)] = [round(ret * 100, 1), round(dd * 100, 1)]
    return out


summary, yoy = {}, {}
for c in curves.columns:
    s = curves[c]
    summary[c] = {'cagr': round(cagr(s) * 100, 2), 'maxdd': round(maxdd(s) * 100, 2),
                  'calmar': round(cagr(s) / abs(maxdd(s)), 2), 'growth100': round(100 * s.iloc[-1], 0)}
    yoy[c] = yearly(s)

wk = curves.resample('W-FRI').last().pct_change().dropna()
corr = wk.corr().round(2)

res = {'window': [str(curves.index[0].date()), str(curves.index[-1].date())],
       'qs_offsets': {'median': round(cg.median() * 100, 2), 'min': round(cg.min() * 100, 2),
                      'max': round(cg.max() * 100, 2), 'drawn': med_col},
       'summary': summary, 'yoy': yoy, 'weekly_corr': corr.to_dict()}
OUT_JSON.write_text(json.dumps(res, indent=1))
print(json.dumps(res['summary'], indent=1))
print(corr.to_string())

# ---- chart -------------------------------------------------------------------------
BG, PANEL, INK, MUT, GRID = '#0e1116', '#161b22', '#e6edf3', '#8b949e', '#30363d'
COLOR = {'TN incumbent': '#e3b341', 'OA v2': '#3fb950', 'OA v3 (gated)': '#58a6ff',
         'IPO (honest)': '#bc8cff', 'NIFTYBEES': '#8b949e', 'Quality Summit': '#ff7b72'}
LABEL = {'TN incumbent': 'True North (incumbent)',
         'OA v2': 'Open Alpha v2 - base-age breakout (r/161)',
         'OA v3 (gated)': 'Open Alpha v3 - ATH breakout, VIX-gated (r/159)',
         'IPO (honest)': 'IPO Base - honest next-day entry',
         'NIFTYBEES': 'NIFTYBEES (index)',
         'Quality Summit': 'QUALITY SUMMIT - near-ATH + loose quality screen (r/160)'}
ORDER = ['Quality Summit', 'TN incumbent', 'OA v2', 'OA v3 (gated)', 'IPO (honest)', 'NIFTYBEES']
g = 100.0 * curves[[c for c in ORDER if c in curves.columns]]

plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': PANEL, 'savefig.facecolor': BG,
                     'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': MUT,
                     'ytick.color': MUT, 'axes.edgecolor': GRID, 'font.size': 9})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True,
                               gridspec_kw={'height_ratios': [2.4, 1]})
for k in g.columns:
    lw = 2.6 if k == 'Quality Summit' else (1.2 if k == 'NIFTYBEES' else 1.6)
    ax1.plot(g.index, g[k], color=COLOR[k], lw=lw,
             label='%s  -  %.1f%%/yr, %.0f%% worst fall' % (LABEL[k], summary[k]['cagr'], summary[k]['maxdd']))
ax1.set_yscale('log')
ax1.set_yticks([100, 200, 400, 800])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_ylabel('growth of 100 (log scale)')
ax1.grid(True, which='both', color=GRID, lw=0.5, alpha=0.6)
ax1.legend(loc='upper left', frameon=False, fontsize=8.5)
ax1.set_title('Quality Summit beside every book, common window %s to %s, after tax'
              % (g.index[0].date(), g.index[-1].date()), color=INK, fontsize=12, loc='left', pad=12)
for k in g.columns:
    dd = (g[k] / g[k].cummax() - 1) * 100
    lw = 2.0 if k == 'Quality Summit' else (1.0 if k == 'NIFTYBEES' else 1.3)
    ax2.plot(dd.index, dd, color=COLOR[k], lw=lw)
    ax2.fill_between(dd.index, dd, 0, color=COLOR[k], alpha=0.07)
ax2.set_ylabel('drawdown %')
ax2.grid(True, color=GRID, lw=0.5, alpha=0.6)
ax2.axhline(0, color=GRID, lw=0.8)
fig.text(0.012, 0.030,
         'Window starts Aug-2018 because point-in-time fundamentals need four filed fiscal years and '
         'Screener history begins FY2015. Every entry is placeable: close-decided, filled at the next open.',
         color=MUT, fontsize=7.5)
fig.text(0.012, 0.009,
         'Quality Summit = the median-CAGR rebalance offset of 12 (never an average of paths); other books = '
         'their roster curves rebased to Aug-2018. 25 bps a side, Indian tax with FY loss netting, 5%% on idle cash.',
         color=MUT, fontsize=7.5)
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(OUT_PNG, dpi=130)
print('wrote', OUT_PNG)
