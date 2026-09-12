# -*- coding: utf-8 -*-
"""research/168 step 2 - True North at 5.2% post-tax idle cash, all 12 rebalance-day offsets.

True North is a DETERMINISTIC rank-based book: it has no seed variance, so its path-dependence
analogue is the 12 rebalance-day offsets (research/144; an offset ensemble once REVERSED the
ranking a single-offset run had produced).

Cost levels. research/144's own basis is rt = 0.003, i.e. 0.3% round trip = 15 bps A SIDE,
whereas Open Alpha and IPO Base are both run at 25 bps a side. That mismatch is inherited from
the published studies, not introduced here, so this script produces BOTH:
    15 bps/side  -> the published basis, used for the main blend so every figure ties to the
                    live mpf report;
    25 bps/side  -> a cost-HARMONISED basis, run so the study can show whether the blend
                    conclusion depends on the mismatch;
    40, 60 bps/side -> the house cost ladder.

STEP 1 IS A BIT-EXACT REPRODUCTION GATE: offset 0 at 15 bps and cash_y = 0.052 must reproduce
research/163's cash052 file exactly. Nothing outside research/168/results/ is written; the
engine's rank cache is disabled so research/144 is never touched.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/144_truenorth_reassessment/scripts'))
import tn_attrib_engine as E                                 # noqa: E402

OUT = ROOT / 'research/168_three_sleeve_blend/results'
OUT.mkdir(parents=True, exist_ok=True)
REF = (ROOT / 'research/163_mpf_cash_yield_harmonisation/results/cash052'
       / 'tn_nav_INC_cash_n8_d15_tax1_cash052.csv')

CELL = dict(series='NIFTYBEES', cons='sma100', n=8, exit=('donch', 15))
CUTOFF = '2026-09-03'
Y = 0.052
RTS = {15: 0.003, 25: 0.005, 40: 0.008, 60: 0.012}
OFFSETS = list(range(12))


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
    print('panel truncated to %s (%d rows)' % (ctx.dates[-1].date(), k), flush=True)
    return ctx


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    d = (nav / nav.cummax() - 1).min()
    return 100 * c, 100 * d, c / abs(d)


def main():
    t0 = time.time()
    ctx = E.Ctx()
    ctx.save_ranks = lambda *a, **k: None          # never write research/144's cache
    truncate(ctx, CUTOFF)

    store, rows, idx_ref = {}, [], None
    for bps, rt in RTS.items():
        for off in OFFSETS:
            r = E.run(ctx, tax=True, offset=off, cash_y=Y, rt=rt, **CELL)
            nav = r['_nav']
            nav = nav / nav.iloc[0]
            if idx_ref is None:
                idx_ref = nav.index
            elif not nav.index.equals(idx_ref):
                print('!! date index changed between cells - stopping'); sys.exit(2)
            store['off%d_%dbps' % (off, bps)] = nav.to_numpy(float)
            c, d, k = stats(nav)
            rows.append(dict(offset=off, cost_bps=bps, cash_y=Y, cagr=round(c, 3),
                             maxdd=round(d, 3), calmar=round(k, 4),
                             avg_inv=round(100 * r['avg_inv'], 2)))
            print('  %2d bps/side  offset %2d   CAGR %6.2f%%  MaxDD %7.2f%%  Calmar %5.3f  '
                  'inv %5.1f%%' % (bps, off, c, d, k, 100 * r['avg_inv']), flush=True)
        sub = [x for x in rows if x['cost_bps'] == bps]
        cg = np.array([x['cagr'] for x in sub]); dd = np.array([x['maxdd'] for x in sub])
        print('%d bps/side ENSEMBLE: CAGR median %.2f%% [%.2f .. %.2f]  MaxDD median %.2f%% '
              '(worst %.2f%%)  Calmar median %.3f'
              % (bps, np.median(cg), cg.min(), cg.max(), np.median(dd), dd.min(),
                 np.median(cg) / abs(np.median(dd))), flush=True)

    # ------------------------------------------------ bit-exact reproduction gate
    print('\n--- REPRODUCTION GATE: offset 0, 15 bps/side, cash 5.2%% vs research/163 ---',
          flush=True)
    pub = pd.read_csv(REF, index_col=0, parse_dates=True).iloc[:, 0]
    pub = pub / pub.iloc[0]
    mine = pd.Series(store['off0_15bps'], index=idx_ref)
    same = bool(mine.index.equals(pub.index))
    rel = ((mine - pub.reindex(mine.index)).abs() / pub.reindex(mine.index).abs())
    bad = rel[rel > 1e-12]
    print('index identical      : %s (%d rows vs %d)' % (same, len(mine), len(pub)))
    print('rows differing       : %d of %d' % (len(bad), len(rel)))
    print('max rel difference   : %.3e' % rel.max())
    if not same or len(bad):
        if len(bad):
            print('first divergence     : %s (rel %.2e)' % (bad.index[0].date(), bad.iloc[0]))
        print('!! REPRODUCTION FAILED - stopping, no npz written.')
        sys.exit(2)
    print('REPRODUCTION BIT-EXACT.', flush=True)

    np.savez_compressed(OUT / 'tn_navs_cash052.npz',
                        dates=np.array([str(d.date()) for d in idx_ref]),
                        offsets=np.array(OFFSETS), **store)
    pd.DataFrame(rows).to_csv(OUT / 'tn_offsets_summary.csv', index=False)
    print('\nwrote %s  (%d arrays, %.0fs)'
          % (OUT / 'tn_navs_cash052.npz', len(store), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
