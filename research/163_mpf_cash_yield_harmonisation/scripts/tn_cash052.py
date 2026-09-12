# -*- coding: utf-8 -*-
"""research/163 — True North re-run with idle cash at 5.2% (was 5.0% on the page since
12-Sep-2026 18:47, and 6.5% in research/144's own study).

5.2% is the ARBITRAGE-FUND rate after 20% short-term tax at 2025-26 cash-futures spreads.

STEP 1 is a REPRODUCTION GATE: the 0.05 re-run must reproduce
    results/tn_nav_INC_cash_n8_d15_tax1_cash05.csv
which is itself proven bit-exact against research/144's published 6.5% curve over the
history. Only then is the yield moved to 0.052.

Nothing in research/144 is written to. Outputs land in results/cash052/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/144_truenorth_reassessment/scripts'))
import tn_attrib_engine as E            # noqa: E402

R163 = ROOT / 'research/163_mpf_cash_yield_harmonisation/results'
OUT = R163 / 'cash052'
OUT.mkdir(parents=True, exist_ok=True)

CELL = dict(series='NIFTYBEES', cons='sma100', n=8, exit=('donch', 15))
CUTOFF = '2026-09-03'          # research/144's own last bar; see the 12-Sep STATUS 2.2
OLD_Y, NEW_Y = 0.05, 0.052


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
    import time
    t0 = time.time()
    ctx = E.Ctx()
    ctx.save_ranks = lambda *a, **k: None      # never write into research/144's cache
    truncate(ctx, CUTOFF)

    print('\n--- step 1: reproduce the 5.0%% curve this page currently uses ---', flush=True)
    r50 = E.run(ctx, tax=True, offset=0, cash_y=OLD_Y, **CELL)
    nav50 = r50['_nav']
    pub = pd.read_csv(R163 / 'tn_nav_INC_cash_n8_d15_tax1_cash05.csv',
                      index_col=0, parse_dates=True).iloc[:, 0]
    same_index = nav50.index.equals(pub.index)
    rel = (nav50 - pub.reindex(nav50.index)).abs() / pub.reindex(nav50.index).abs()
    bad = rel[rel > 1e-12]
    print('index identical     : %s (%d rows)' % (same_index, len(nav50)))
    print('rows differing      : %d of %d' % (len(bad), len(rel)))
    print('max rel difference  : %.3e' % rel.max())
    if not same_index:
        print('!! REPRODUCTION FAILED (index) — stopping.')
        sys.exit(2)
    if len(bad):
        print('first divergence    : %s (rel %.2e)' % (bad.index[0].date(), bad.iloc[0]))
        if bad.index[0] < pd.Timestamp('2026-08-01'):
            print('!! divergence starts too early to be a data refresh — stopping.')
            sys.exit(2)
    print('REPRODUCTION EXACT over %.2f%% of the history.'
          % (100.0 * (len(rel) - len(bad)) / len(rel)))
    nav50.to_csv(OUT / 'tn_nav_INC_cash_n8_d15_tax1_cash05_reproduced.csv', header=['nav'])

    print('\n--- step 2: the same cell at 5.2%% idle cash ---', flush=True)
    r52 = E.run(ctx, tax=True, offset=0, cash_y=NEW_Y, **CELL)
    nav52 = r52['_nav']
    nav52.to_csv(OUT / 'tn_nav_INC_cash_n8_d15_tax1_cash052.csv', header=['nav'])

    print('\n%-10s %9s %9s %8s %10s %8s' % ('cash_y', 'CAGR', 'MaxDD', 'Calmar',
                                            'final x', 'avg_inv'))
    for lbl, r, nav in (('5.0%', r50, nav50), ('5.2%', r52, nav52)):
        c, d, cal = stats(nav)
        print('%-10s %8.2f%% %8.2f%% %8.2f %9.2fx %7.2f'
              % (lbl, c, d, cal, nav.iloc[-1] / nav.iloc[0], r['avg_inv']))
    cpub, dpub, kpub = stats(pub)
    print('%-10s %8.2f%% %8.2f%% %8.2f %9.2fx %7s   <- the file the page reads today'
          % ('5.0% file', cpub, dpub, kpub, pub.iloc[-1] / pub.iloc[0], '-'))
    c50, d50, k50 = stats(nav50)
    c52, d52, k52 = stats(nav52)
    inv = r50['avg_inv']
    pred = (1 - inv) * 0.2
    print('\nYIELD EFFECT 5.0%% -> 5.2%%: CAGR %+.3f pp  MaxDD %+.3f pp  Calmar %+.4f'
          % (c52 - c50, d52 - d50, k52 - k50))
    print('CONSISTENCY: avg invested %.1f%% -> (1-inv) x 0.2pp = %+.3f pp predicted, '
          '%+.3f pp measured' % (inv * 100, pred, c52 - c50))
    print('\nwrote %s   (%.0fs)' % (OUT / 'tn_nav_INC_cash_n8_d15_tax1_cash052.csv',
                                    time.time() - t0))


if __name__ == '__main__':
    main()
