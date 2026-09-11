# -*- coding: utf-8 -*-
"""research/162 — the one eligibility mask r/160 did not build: growth > 12%.

r/160 built the Family-B ladder at growth > 10 (`b7_g10_qual_mc`, Quality Summit's own
screen), > 15 (`b5_g15_qual_mc`) and > 20 (`arun_strict`, with D/E). The A4 dial sweep needs
the 12% rung between the first two. Recipe, column axis, has_data gating and npz contract
are copied from r/160's `build_study_masks.py` so the engine's loader and every paired
comparison work unchanged; only the growth threshold differs.

Writes into research/162/results/masks162/ — r/160's mask directories are read-only here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
R160 = ROOT / 'research' / '160_quality_growth_near_ath' / 'results'
OUT = Path(__file__).resolve().parents[1] / 'results' / 'masks162'
START, END = '2015-01-01', '2026-09-01'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    uni = pd.read_csv(R160 / 'universe.csv')
    cols = np.array(sorted(uni.symbol.unique()))
    ci = {s: i for i, s in enumerate(cols)}
    dates = pd.date_range(START, END, freq='MS')
    di = {str(d.date()): i for i, d in enumerate(dates)}
    dstr = np.array([str(d.date()) for d in dates])
    shape = (len(dates), len(cols))

    p = pd.read_csv(R160 / 'features_pit_monthly.csv.gz')
    p = p[p.symbol.isin(ci) & p.date.isin(di)].copy()
    r = p.date.map(di).values
    c = p.symbol.map(ci).values

    def grid(series, fill=np.nan):
        g = np.full(shape, fill, dtype=float)
        g[r, c] = pd.to_numeric(series, errors='coerce').values
        return g

    def bgrid(series):
        g = np.zeros(shape, dtype=bool)
        g[r, c] = series.astype(bool).values
        return g

    n_fy = grid(p.n_fy_usable, 0.0)
    sg, pg = grid(p.sales_g3), grid(p.profit_g3)
    roe, roce = grid(p.roe_avg3), grid(p.roce_latest)
    mcap = grid(p.mcap_pit)
    lender, neg = bgrid(p.is_lender), bgrid(p.neg3)

    de = grid(p.de_latest)
    has_data = n_fy >= 4
    ok = lambda a: np.nan_to_num(a, nan=-np.inf)                              # noqa: E731
    # a <= test needs NaN to FAIL, so missing debt data maps to +inf, not -inf. This is why
    # r/160 found the D/E rung removes the whole financial sector: Screener files no
    # Borrowings row for lenders, so their D/E is missing and therefore fails.
    okle = lambda a: np.nan_to_num(a, nan=np.inf)                             # noqa: E731
    quality = ((ok(roe) > 15) & (((ok(roce) > 15) & ~lender) | lender))
    base = (~neg) & (ok(mcap) > 1000) & quality

    rows = []
    # the honest control for any "does the screen add value" claim: the SCREENABLE
    # sub-universe, i.e. every name with four filed fiscal years at that date and nothing
    # else required. r/160 paired its masks against this; the npz was never kept.
    np.savez_compressed(OUT / 'has_data.npz', dates=dstr, cols=cols, mask=has_data)
    rows.append(dict(name='has_data',
                     definition='n_fy_usable >= 4 - the screenable sub-universe, the '
                                'control every screen must beat',
                     mean_n_pass=round(float(has_data.sum(1).mean()), 1),
                     n_pass_2026=int(has_data.sum(1)[-1]), pct_of_has_data=100.0))
    print('has_data: mean %.1f names' % has_data.sum(1).mean())

    # r/160's DATA-leg masks that Part B needs: the npz files were never kept on disk, only
    # their INDEX.csv row. Rebuilt here to the definitions that row states, verbatim.
    extra = {
        'growth_only': ((ok(sg) > 20) & (ok(pg) > 20) & (~neg),
                        'sales_g3>20 & profit_g3>20 & no negatives'),
        'arun_strict': (base & (ok(sg) > 20) & (ok(pg) > 20) & (okle(de) <= 0.2),
                        'THE SCREEN AS ARUN WROTE IT: sales_g3>20 & profit_g3>20 & '
                        'roe_avg3>15 & roce>15 (lenders: ROE only) & de<=0.2 & '
                        'mcap_pit>1000cr & no negatives'),
    }
    for name, (m, dfn) in extra.items():
        m = m & has_data
        np.savez_compressed(OUT / ('%s.npz' % name), dates=dstr, cols=cols, mask=m)
        nn = m.sum(1)
        rows.append(dict(name=name, definition=dfn,
                         mean_n_pass=round(float(nn.mean()), 1), n_pass_2026=int(nn[-1]),
                         pct_of_has_data=round(100.0 * nn.sum() /
                                               max(has_data.sum(), 1), 2)))
        print('%s: mean %.1f names pass, %d in 2026-09 (r/160 INDEX says %s)'
              % (name, nn.mean(), nn[-1],
                 {'growth_only': '162.6 / 276', 'arun_strict': '26.4 / 46'}[name]))
    for t in (12,):
        m = (base & (ok(sg) > t) & (ok(pg) > t)) & has_data
        name = 'b6_g%d_qual_mc' % t
        np.savez_compressed(OUT / ('%s.npz' % name), dates=dstr, cols=cols, mask=m)
        n = m.sum(1)
        rows.append(dict(name=name,
                         definition='profitable & mcap>1000cr & roe_avg3>15 & '
                                    'roce>15 (lenders: ROE only) & sales_g3>%d & '
                                    'profit_g3>%d - the growth rung between r/160 b7 (10) '
                                    'and b5 (15)' % (t, t),
                         mean_n_pass=round(float(n.mean()), 1), n_pass_2026=int(n[-1]),
                         pct_of_has_data=round(100.0 * n.sum() / max(has_data.sum(), 1), 2)))
        print('%s: mean %.1f names pass, %d in 2026-09' % (name, n.mean(), n[-1]))
    pd.DataFrame(rows).to_csv(OUT / 'INDEX.csv', index=False)
    print('wrote %s' % OUT)


if __name__ == '__main__':
    sys.exit(main())
