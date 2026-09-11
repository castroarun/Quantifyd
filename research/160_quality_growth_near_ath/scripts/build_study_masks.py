# -*- coding: utf-8 -*-
"""research/160 STUDY leg - the Family-B masks ("what Arun actually does").

The DATA leg's 31 masks are all descendants of the screen AS WRITTEN: every one of them
keeps the debt/equity test and the 20% growth bars, or drops exactly one of them. But the
replication gate (results/holdings_check.md) says the written screen picks only 8 of the 69
names in his real book, while "near its high and profitable" picks 42, and "growth and D/E
both dropped" picks 40. There is therefore a whole family the DATA leg's grid cannot
express, and it is the family his actual trades live in.

This script builds it, WITHOUT touching anything the DATA leg owns: it reads
results/features_pit_monthly.csv.gz and results/universe.csv read-only and writes into
results/masks_study/, a separate directory. Same npz contract (dates / cols / mask), same
column axis (the whole universe), same rule that every mask is False wherever has_data is
False, so the engine's loader and the has_data pairing work unchanged.

The ladder from Family A to "near-ATH alone", continuous:

    arun_strict          growth>20 + quality + D/E<=0.2 + mcap        (8 of his 69)
    g15_mc1000           growth>15 + quality + D/E<=0.2 + mcap        [DATA leg]
    b5_g15_qual_mc       growth>15 + quality           + mcap         <- Family B headline
    b7_g10_qual_mc       growth>10 + quality           + mcap
    b3_qual_mc                      quality           + mcap          (40 of his 69)
    b4_g15_mc            growth>15                    + mcap
    b2_noneg_mc1000                                     mcap + profitable
    b1_noneg                                                 profitable
    (no mask)            near-ATH alone                                (42 of his 69)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / 'research/160_quality_growth_near_ath/results'
OUT = RES / 'masks_study'
OUT.mkdir(parents=True, exist_ok=True)

START, END = '2015-01-01', '2026-09-01'


def main():
    uni = pd.read_csv(RES / 'universe.csv')
    cols = np.array(sorted(uni.symbol.unique()))
    ci = {s: i for i, s in enumerate(cols)}
    dates = pd.date_range(START, END, freq='MS')
    di = {str(d.date()): i for i, d in enumerate(dates)}
    dstr = np.array([str(d.date()) for d in dates])
    shape = (len(dates), len(cols))

    p = pd.read_csv(RES / 'features_pit_monthly.csv.gz')
    p = p[p.symbol.isin(ci) & p.date.isin(di)].copy()
    r = p.date.map(di).values
    c = p.symbol.map(ci).values
    print('panel %d rows -> grid %d months x %d symbols' % (len(p), *shape), flush=True)

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

    has_data = n_fy >= 4
    ok = lambda a: np.nan_to_num(a, nan=-np.inf)          # NaN never passes a > test  # noqa: E731
    growth = lambda t: (ok(sg) > t) & (ok(pg) > t)        # noqa: E731
    # identical to the DATA leg's: a lender is judged on ROE alone, ROCE n/a not failed
    quality = ((ok(roe) > 15) & (((ok(roce) > 15) & ~lender) | lender))
    no_neg = ~neg
    mc1000 = ok(mcap) > 1000

    masks, defs = {}, {}

    def add(name, m, d):
        masks[name] = m & has_data
        defs[name] = d

    add('b1_noneg', no_neg,
        'no negative Sales or Net Profit in the last 3 filed FY - "profitable", nothing else')
    add('b2_noneg_mc1000', no_neg & mc1000,
        'profitable AND mcap_pit > 1000cr')
    add('b3_qual_mc', no_neg & mc1000 & quality,
        'profitable & mcap>1000cr & roe_avg3>15 & roce>15 (lenders: ROE only) - '
        'i.e. arun_strict with BOTH growth and debt/equity dropped (40 of his 69 names)')
    add('b4_g15_mc', no_neg & mc1000 & growth(15),
        'profitable & mcap>1000cr & sales_g3>15 & profit_g3>15 - relaxed growth, '
        'no debt/equity test, no quality test')
    add('b5_g15_qual_mc', no_neg & mc1000 & quality & growth(15),
        'FAMILY B HEADLINE: profitable & mcap>1000cr & quality & growth>15 - '
        'the written screen with the growth bar relaxed to 15% and D/E dropped')
    add('b7_g10_qual_mc', no_neg & mc1000 & quality & growth(10),
        'b5 with the growth bar at 10% - the softer neighbour of the Family-B headline')

    yrs = np.array([int(s[:4]) for s in dstr])
    hd_n = has_data.sum(1)
    idx = []
    for name, m in masks.items():
        np.savez_compressed(OUT / ('%s.npz' % name), dates=dstr, cols=cols, mask=m)
        n = m.sum(1)
        row = dict(name=name, definition=defs[name],
                   mean_n_pass=round(float(n.mean()), 1), n_pass_2026=int(n[-1]),
                   pct_of_has_data=round(100.0 * n.sum() / max(hd_n.sum(), 1), 2))
        for y in range(2015, 2027):
            sel = yrs == y
            row['y%d' % y] = round(float(n[sel].mean()), 1) if sel.any() else np.nan
        idx.append(row)
    ix = pd.DataFrame(idx)
    ix.to_csv(OUT / 'INDEX.csv', index=False)

    print()
    print('%-18s %7s %7s  %s' % ('mask', 'mean', 'latest', 'mean names passing, by year'))
    for _, rw in ix.iterrows():
        print('%-18s %7.0f %7d  %s' % (
            rw['name'], rw.mean_n_pass, rw.n_pass_2026,
            ' '.join('%d:%.0f' % (y, rw['y%d' % y]) for y in range(2018, 2027))))
    print()
    print('has_data names per month: min %d, median %d, max %d'
          % (hd_n.min(), int(np.median(hd_n)), hd_n.max()))
    print('wrote %d study masks + INDEX.csv to %s' % (len(masks), OUT))


if __name__ == '__main__':
    sys.exit(main())
