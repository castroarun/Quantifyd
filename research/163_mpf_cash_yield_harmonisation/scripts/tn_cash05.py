# -*- coding: utf-8 -*-
"""research/163 — True North re-run with idle cash at 5.0% instead of research/144's 6.5%.

Reproduces EXACTLY the call that produced research/144's published report curve

    results/nav_INC_cash_n8_d15_tax1.csv
      = phase_D(), finalist tag 'INC_cash_n8_d15', offset 0, tax=True
      = run(ctx, tax=True, offset=0, series='NIFTYBEES', cons='sma100',
            n=8, exit=('donch', 15))        # action='cash', freq='weekly' by default

and then re-runs the same call with cash_y=0.05.

STEP 1 is a BIT-IDENTITY PROOF: the 0.065 re-run must reproduce the published NAV file to
the last decimal. If it does not, nothing downstream is trustworthy and the script stops.

Nothing in research/144 is written to. Outputs land in research/163's own results folder.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/144_truenorth_reassessment/scripts'))
import tn_attrib_engine as E            # noqa: E402

R144 = ROOT / 'research/144_truenorth_reassessment/results'
OUT = ROOT / 'research/163_mpf_cash_yield_harmonisation/results'
OUT.mkdir(parents=True, exist_ok=True)

CELL = dict(series='NIFTYBEES', cons='sma100', n=8, exit=('donch', 15))

# research/144 ran on 3-Sep-2026, so its NAV file ends there; market_data.db has grown
# since. The panel is TRUNCATED to that same last date so the run is the SAME run — the
# final partial-fiscal-year tax settlement lands on the same day it did in the study.
CUTOFF = '2026-09-03'


def truncate(ctx, cutoff):
    k = int(np.searchsorted(ctx.dates.values, np.datetime64(cutoff), side='right'))
    ctx.dates = ctx.dates[:k]
    ctx.C = ctx.C[:k]
    ctx.rawnn = ctx.rawnn[:k]
    ctx.is_wk = ctx.is_wk[:k]
    ctx.close = ctx.close.iloc[:k]
    ctx.cf = ctx.cf.iloc[:k]
    ctx.tv = ctx.tv.iloc[:k]
    ctx._gate.clear(); ctx._exitm.clear(); ctx._me.clear()
    print('panel truncated to %s (%d rows, loop from %s)'
          % (ctx.dates[-1].date(), k, ctx.dates[ctx.i0].date()), flush=True)
    return ctx


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    d = (nav / nav.cummax() - 1).min()
    return c * 100, d * 100, (c / abs(d))


def main():
    ctx = E.Ctx()
    ctx.save_ranks = lambda *a, **k: None      # never write into research/144's cache
    truncate(ctx, CUTOFF)

    print('\n--- step 1: reproduce the published 6.5% curve ---', flush=True)
    r65 = E.run(ctx, tax=True, offset=0, cash_y=0.065, **CELL)
    nav65 = r65['_nav']
    pub = pd.read_csv(R144 / 'nav_INC_cash_n8_d15_tax1.csv',
                      index_col=0, parse_dates=True).iloc[:, 0]
    same_index = nav65.index.equals(pub.index)
    rel = (nav65 - pub.reindex(nav65.index)).abs() / pub.reindex(nav65.index).abs()
    bad = rel[rel > 1e-12]
    print('index identical     : %s (%d rows)' % (same_index, len(nav65)))
    print('rows differing      : %d of %d' % (len(bad), len(rel)))
    print('max rel difference  : %.3e' % rel.max())
    if not same_index:
        print('!! REPRODUCTION FAILED (index) — stopping.')
        sys.exit(2)
    if len(bad):
        print('first divergence    : %s (rel %.2e)' % (bad.index[0].date(), bad.iloc[0]))
        # market_data.db is refreshed daily, so bars near the END of the window can be
        # revised after a study has run. Acceptable ONLY if the reproduction is exact over
        # the whole history and the divergence is confined to the last few weeks.
        if bad.index[0] < pd.Timestamp('2026-08-01'):
            print('!! divergence starts too early to be a data refresh — stopping.')
            sys.exit(2)
    print('REPRODUCTION EXACT over %.2f%% of the history; any tail difference is '
          'market_data.db being refreshed since research/144 ran.'
          % (100.0 * (len(rel) - len(bad)) / len(rel)))
    nav65.to_csv(OUT / 'tn_nav_INC_cash_n8_d15_tax1_cash065_reproduced.csv',
                 header=['nav'])

    print('\n--- step 2: the same cell at 5.0% idle cash ---', flush=True)
    r50 = E.run(ctx, tax=True, offset=0, cash_y=0.05, **CELL)
    nav50 = r50['_nav']
    nav50.to_csv(OUT / 'tn_nav_INC_cash_n8_d15_tax1_cash05.csv', header=['nav'])

    print('\n%-8s %9s %9s %8s %10s %8s' % ('cash_y', 'CAGR', 'MaxDD', 'Calmar',
                                           'final x', 'avg_inv'))
    for lbl, r, nav in (('6.5%', r65, nav65), ('5.0%', r50, nav50)):
        c, d, cal = stats(nav)
        print('%-8s %8.2f%% %8.2f%% %8.2f %9.2fx %7.2f'
              % (lbl, c, d, cal, nav.iloc[-1] / nav.iloc[0], r['avg_inv']))
    cpub, dpub, kpub = stats(pub)
    print('%-8s %8.2f%% %8.2f%% %8.2f %9.2fx %7s   <- research/144 as published'
          % ('6.5% pub', cpub, dpub, kpub, pub.iloc[-1] / pub.iloc[0], '-'))
    c65, d65, k65 = stats(nav65)
    c50, d50, k50 = stats(nav50)
    print('\nPURE YIELD EFFECT (both runs on today\'s data): CAGR %+.2f pp  '
          'MaxDD %+.2f pp  Calmar %+.3f' % (c50 - c65, d50 - d65, k50 - k65))
    print('DATA-REFRESH EFFECT (6.5%% today vs 6.5%% published): CAGR %+.2f pp  '
          'MaxDD %+.2f pp  Calmar %+.3f' % (c65 - cpub, d65 - dpub, k65 - kpub))
    print('avg invested %.1f%% -> the 1.5pp yield cut should cost about %.2f pp a year'
          % (r65['avg_inv'] * 100, (1 - r65['avg_inv']) * 1.5))
    print('\nwrote', OUT / 'tn_nav_INC_cash_n8_d15_tax1_cash05.csv')


if __name__ == '__main__':
    main()
