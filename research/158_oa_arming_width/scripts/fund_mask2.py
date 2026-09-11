# -*- coding: utf-8 -*-
"""Build Arun's fundamental-eligibility masks from the Screener annual history.

Criteria, as given (11-Sep-2026):

    profit growth over the last three years   > 15%
    sales  growth over the last three years   > 15%
    debt / equity                            <= 0.20
    ROE                                       > 15%
    ROCE                                      > 15%
    no negative figures in the last three years

POINT-IN-TIME. A fiscal year is usable only once filed. Indian year-ends fall on 31-March and
listed companies file audited annuals within about four months, so a year is treated as
knowable from year-end + 4 months. At each decision date only years clearing that test are
read. This is what makes the recent-window test honest, and it is the whole reason the window
starts in Aug-2024 rather than 2006.

Screener carries roughly a decade, so a genuine THREE-year growth rate is available across
the whole window - which Yahoo's four fiscal years could not support (it only reached three
years of growth from mid-2026).

ROCE comes from Screener's own row rather than being computed: its balance sheet lumps
liabilities without splitting out current liabilities, so EBIT/(assets - current liabilities)
off that page would be invention. ROE is computed as net profit / (equity capital + reserves),
which the page does support.

LENDERS. Screener reports no ROCE for banks and NBFCs, because capital employed is not a
meaningful denominator for them. Those names are judged on ROE alone and ROCE is recorded as
not-applicable rather than failed - failing them would quietly exclude the entire financial
sector, which is a sector bet dressed up as a quality filter.

FIVE MASKS, not one. Strict plus four leave-one-out variants, because a single pass rate
cannot say WHICH criterion is doing the work.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/158_oa_arming_width/results'
CACHE = RES / 'screener_cache'
WIN = ('2024-08-01', '2026-09-04')
LAG_MONTHS = 4
GROWTH_MIN, DE_MAX, ROE_MIN, ROCE_MIN = 15.0, 0.20, 15.0, 15.0
MIN_YEARS = 4            # four annual points = a genuine three-year growth rate

VARIANTS = {
    'strict': ('growth', 'de', 'roe', 'roce', 'neg'),
    'no_de': ('growth', 'roe', 'roce', 'neg'),
    'no_roce': ('growth', 'de', 'roe', 'neg'),
    'no_growth': ('de', 'roe', 'roce', 'neg'),
    'growth_only': ('growth', 'neg'),
}


def load():
    recs, src = {}, {}
    for f in sorted(CACHE.glob('*.json')):
        d = json.load(open(f))
        if d.get('fy'):
            recs[d['symbol']] = {pd.Timestamp(k): v for k, v in d['fy'].items()}
            src[d['symbol']] = d.get('source')
    return recs, src


def cagr_pct(a, b, yrs):
    """Growth from a to b over yrs years, in percent. Undefined when the base is not
    positive: a swing from a loss to a profit is not a growth rate. The no-negatives rule
    handles that case instead."""
    if a is None or b is None or yrs <= 0 or a <= 0 or b <= 0:
        return None
    return 100.0 * ((b / a) ** (1.0 / yrs) - 1.0)


def assess(fy_map, asof):
    usable = sorted(k for k in fy_map
                    if k + pd.DateOffset(months=LAG_MONTHS) <= asof)
    if len(usable) < MIN_YEARS:
        return None, None
    use = usable[-MIN_YEARS:]
    span = len(use) - 1                      # 3
    latest = fy_map[use[-1]]
    m = dict(n_fy=len(usable), span=span, fy_latest=str(use[-1].date()))
    m['sales_g'] = cagr_pct(fy_map[use[0]].get('sales'), latest.get('sales'), span)
    m['profit_g'] = cagr_pct(fy_map[use[0]].get('net_profit'),
                             latest.get('net_profit'), span)
    m['roe'] = latest.get('roe_pct')
    m['roce'] = latest.get('roce_pct')
    m['de'] = latest.get('de')
    m['is_lender'] = m['roce'] is None
    # "no negative figures in the last three years" - the three years the growth spans
    last3 = use[-3:]
    m['negatives'] = sum(
        1 for y in last3
        for k in ('sales', 'net_profit')
        if fy_map[y].get(k) is not None and fy_map[y][k] < 0)

    c = {}
    c['growth'] = (m['sales_g'] is not None and m['profit_g'] is not None
                   and m['sales_g'] > GROWTH_MIN and m['profit_g'] > GROWTH_MIN)
    c['de'] = m['de'] is not None and m['de'] <= DE_MAX
    c['roe'] = m['roe'] is not None and m['roe'] > ROE_MIN
    c['roce'] = None if m['is_lender'] else (m['roce'] > ROCE_MIN)
    c['neg'] = m['negatives'] == 0
    return m, c


def main():
    recs, src = load()
    print('%d symbols with Screener annual data' % len(recs))
    yrs = pd.Series({s: len(v) for s, v in recs.items()})
    print('fiscal years per symbol: median %d, min %d, max %d; %d have >= %d'
          % (yrs.median(), yrs.min(), yrs.max(), int((yrs >= MIN_YEARS).sum()), MIN_YEARS))
    months = pd.date_range(WIN[0], WIN[1], freq='MS')
    cols = sorted(recs)
    masks = {k: np.zeros((len(months), len(cols)), dtype=bool) for k in VARIANTS}
    diag = []
    for ci, s in enumerate(cols):
        for mi, d in enumerate(months):
            m, c = assess(recs[s], d)
            if c is None:
                continue
            for vk, crits in VARIANTS.items():
                masks[vk][mi, ci] = all(c[k] for k in crits if c[k] is not None)
            if mi == len(months) - 1:
                diag.append(dict(symbol=s, source=src.get(s),
                                 **{k: (round(v, 2) if isinstance(v, float) else v)
                                    for k, v in m.items()},
                                 **{'pass_' + k: c[k] for k in c}))
    for vk in VARIANTS:
        p = RES / ('fund_mask_%s.npz' % vk)
        np.savez_compressed(p, dates=np.array([str(d.date()) for d in months]),
                            cols=np.array(cols), mask=masks[vk])
        print('  %-12s latest-month pass rate %5.1f%%  (%d of %d)'
              % (vk, 100.0 * masks[vk][-1].mean(), int(masks[vk][-1].sum()), len(cols)))
    dd = pd.DataFrame(diag)
    dd.to_csv(RES / 'fund_diagnostics.csv', index=False)
    print()
    print('per-criterion pass rate, latest month, of %d names assessable:' % len(dd))
    for k in ['growth', 'de', 'roe', 'roce', 'neg']:
        col = dd['pass_' + k]
        print('   %-8s pass %3d   fail %3d   n/a %3d'
              % (k, int((col == True).sum()), int((col == False).sum()),  # noqa: E712
                 int(col.isna().sum())))
    print()
    print('wrote %s' % (RES / 'fund_diagnostics.csv'))


if __name__ == '__main__':
    main()
