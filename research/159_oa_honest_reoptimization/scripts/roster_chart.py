# -*- coding: utf-8 -*-
"""Growth-of-100 and drawdown for every book, honest entries, after tax.

The house rule (CLAUDE.md, roster report) wants one page holding every system side by side
with a log growth chart and a drawdown panel, not a folder of individual studies. This draws
that chart.

LOG SCALE on the growth panel, deliberately. Over a decade a 9x book plotted linearly makes
every other line look flat and hides the 2018 and 2022 drawdowns entirely. On a log axis
equal vertical distances are equal percentage moves, which is what a reader is actually
comparing.

The drawdown panel is the reason the chart exists. Two books ending at the same multiple
with -24% and -36% worst falls are not the same product, and only the lower panel shows it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/159_oa_honest_reoptimization/results'
OUT = ROOT / 'frontend/public/mpf-honest-roster-2026-09.png'

BG, PANEL, INK, MUT, GRID = '#0e1116', '#161b22', '#e6edf3', '#8b949e', '#30363d'
COLOR = {'TN incumbent': '#e3b341',        # gold: the incumbent, the reference line
         'OA v2': '#3fb950',
         'OA v3 (gated)': '#58a6ff',
         'IPO (honest)': '#bc8cff',
         'NIFTYBEES': '#8b949e'}           # grey: the benchmark recedes
LABEL = {'TN incumbent': 'True North (incumbent)',
         'OA v2': 'Open Alpha v2 - base-age breakout (r/161)',
         'OA v3 (gated)': 'Open Alpha v3 - ATH breakout, VIX-gated (r/159)',
         'IPO (honest)': 'IPO Base - honest next-day entry (r/153 corrected)',
         'NIFTYBEES': 'NIFTYBEES (index)'}
ORDER = ['TN incumbent', 'OA v2', 'OA v3 (gated)', 'IPO (honest)', 'NIFTYBEES']

df = pd.read_csv(RES / 'all_systems_after_tax.csv', index_col=0, parse_dates=True)
df = df[[c for c in ORDER if c in df.columns]]
g = 100.0 * df / df.iloc[0]

plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': PANEL,
                     'savefig.facecolor': BG, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUT, 'ytick.color': MUT,
                     'axes.edgecolor': GRID, 'font.size': 9})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True,
                               gridspec_kw={'height_ratios': [2.4, 1]})

for k in g.columns:
    lw = 2.4 if k == 'TN incumbent' else (1.2 if k == 'NIFTYBEES' else 1.8)
    ax1.plot(g.index, g[k], color=COLOR[k], lw=lw,
             label='%s  -  %.1f%%/yr, %.0f%% worst fall'
                   % (LABEL[k],
                      ((g[k].iloc[-1] / 100) ** (365.25 / (g.index[-1] - g.index[0]).days)
                       - 1) * 100,
                      (g[k] / g[k].cummax() - 1).min() * 100))
ax1.set_yscale('log')
ax1.set_yticks([100, 200, 400, 800, 1600])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_ylabel('growth of 100 (log scale)')
ax1.grid(True, which='both', color=GRID, lw=0.5, alpha=0.6)
ax1.legend(loc='upper left', frameon=False, fontsize=8.5)
ax1.set_title('The Momentum Portfolio, honest entries only, after tax   '
              '(%s to %s)' % (g.index[0].date(), g.index[-1].date()),
              color=INK, fontsize=12, loc='left', pad=12)

for k in g.columns:
    dd = (g[k] / g[k].cummax() - 1) * 100
    lw = 2.0 if k == 'TN incumbent' else (1.0 if k == 'NIFTYBEES' else 1.4)
    ax2.plot(dd.index, dd, color=COLOR[k], lw=lw)
    ax2.fill_between(dd.index, dd, 0, color=COLOR[k], alpha=0.07)
ax2.set_ylabel('drawdown %')
ax2.grid(True, color=GRID, lw=0.5, alpha=0.6)
ax2.axhline(0, color=GRID, lw=0.8)

# two lines: a single line of this length runs off the right edge at 12in wide
fig.text(0.012, 0.030,
         'Every entry is one a real order can place: a close-based signal filled at the '
         'next open, or a resting stop. The look-ahead entries behind the published 40.8% '
         '(Open Alpha) and 31.0% (IPO Base) are excluded.',
         color=MUT, fontsize=7.5)
fig.text(0.012, 0.009,
         '30-seed medians, 25 bps a side, Indian tax with financial-year loss netting, '
         '5% on idle cash. Window starts 2016 because the VIX gate needs INDIA VIX, which '
         'begins 2015.',
         color=MUT, fontsize=7.5)
fig.tight_layout(rect=[0, 0.05, 1, 1])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=125)
print('wrote %s (%.0f KB)' % (OUT, OUT.stat().st_size / 1024))
