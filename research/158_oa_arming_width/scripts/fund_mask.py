# -*- coding: utf-8 -*-
"""Build Arun's fundamental-eligibility masks from the cached annual books.

Criteria, as given (11-Sep-2026):

    profit growth over the last three years   > 15%
    sales  growth over the last three years   > 15%
    debt / equity                            <= 0.20
    ROE                                       > 15%
    ROCE                                      > 15%
    no negative figures in the last three years

POINT-IN-TIME DISCIPLINE. A fiscal year is only usable once it was filed. Indian year ends
run to 31-March and audited annuals are out within roughly four months, so a year end is
treated as knowable from year-end + 4 months. At any decision date only years that clear
that test are read, which is what makes a recent-window test honest where a 2006 one would
not be. The mask is emitted monthly and the engine forward-fills it, because eligibility only
changes when a new annual lands.

ROE and ROCE are computed, not read:

    ROE  = net income / shareholder equity
    ROCE = EBIT / (total assets - total current liabilities)

BANKS AND NBFCs. Capital employed is not meaningful for a lender, and yfinance returns no
EBIT or current-liabilities split for them. Such names are judged on ROE alone and the ROCE
test is recorded as not-applicable rather than failed - failing them would silently exclude
the whole financial sector, which is a sector bet, not a quality filter.

FIVE MASKS, not one. The strict mask plus four leave-one-out variants, because a single
pass/fail number cannot say WHICH criterion is doing the work. If the strict mask kills the
book and dropping debt/equity restores it, that is the finding.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/158_oa_arming_width/results'
CACHE = RES / 'fund_cache'
WIN = ('2024-08-01', '2026-09-04')
LAG_MONTHS = 4
GROWTH_MIN, DE_MAX, ROE_MIN, ROCE_MIN = 0.15, 0.20, 0.15, 0.15

VARIANTS = {
    'strict': dict(growth=True, de=True, roe=True, roce=True, neg=True),
    'no_de': dict(growth=True, de=False, roe=True, roce=True, neg=True),
    'no_roce': dict(growth=True, de=True, roe=True, roce=False, neg=True),
    'no_growth': dict(growth=False, de=True, roe=True, roce=True, neg=True),
    'growth_only': dict(growth=True, de=False, roe=False, roce=False, neg=True),
}


def load():
    recs = {}
    for f in sorted(CACHE.glob('*.json')):
        d = json.load(open(f))
        if d.get('fy'):
            recs[d['symbol']] = {pd.Timestamp(k): v for k, v in d['fy'].items()}
    return recs


def cagr(a, b, yrs):
    """Growth from a to b over yrs years. Undefined if the base is not positive - a swing
    from a loss to a profit is not a growth rate, so it is reported as not-measurable and
    handled by the no-negatives rule instead."""
    if a is None or b is None or yrs <= 0 or a <= 0 or b <= 0:
        return None
    return (b / a) ** (1.0 / yrs) - 1.0


def assess(fy_map, asof):
    """-> (dict of metrics, dict of per-criterion pass/None)."""
    usable = sorted([k for k in fy_map
                     if k + pd.DateOffset(months=LAG_MONTHS) <= asof])
    if len(usable) < 3:
        return None, None
    use = usable[-4:] if len(usable) >= 4 else usable[-3:]
    span = len(use) - 1
    latest = fy_map[use[-1]]
    g = lambda k, y: fy_map[y].get(k)                                    # noqa: E731
    m = dict(n_fy=len(use), span=span, fy_latest=str(use[-1].date()))
    m['sales_g'] = cagr(g('revenue', use[0]), g('revenue', use[-1]), span)
    m['profit_g'] = cagr(g('net_income', use[0]), g('net_income', use[-1]), span)
    eq, ni = latest.get('equity'), latest.get('net_income')
    m['roe'] = (ni / eq) if (eq and ni is not None and eq > 0) else None
    debt = latest.get('debt')
    m['de'] = (debt / eq) if (eq and debt is not None and eq > 0) else None
    ebit, ta, cl = latest.get('ebit'), latest.get('assets'), latest.get('cur_liab')
    cap = (ta - cl) if (ta is not None and cl is not None) else None
    m['roce'] = (ebit / cap) if (ebit is not None and cap and cap > 0) else None
    m['is_lender'] = ebit is None or cl is None
    negs = [y for y in use
            if (g('net_income', y) is not None and g('net_income', y) < 0)
            or (g('revenue', y) is not None and g('revenue', y) < 0)]
    m['negatives'] = len(negs)

    c = {}
    c['growth'] = (m['sales_g'] is not None and m['profit_g'] is not None
                   and m['sales_g'] > GROWTH_MIN and m['profit_g'] > GROWTH_MIN)
    c['de'] = m['de'] is not None and m['de'] <= DE_MAX
    c['roe'] = m['roe'] is not None and m['roe'] > ROE_MIN
    c['roce'] = None if m['is_lender'] else (m['roce'] is not None
                                             and m['roce'] > ROCE_MIN)
    c['neg'] = m['negatives'] == 0
    return m, c


def main():
    recs = load()
    print('%d symbols with annual data in the cache' % len(recs))
    months = pd.date_range(WIN[0], WIN[1], freq='MS')
    cols = sorted(recs)
    diag = []
    masks = {k: np.zeros((len(months), len(cols)), dtype=bool) for k in VARIANTS}
    for ci, s in enumerate(cols):
        for mi, d in enumerate(months):
            m, c = assess(recs[s], d)
            if c is None:
                continue
            for vk, v in VARIANTS.items():
                ok = True
                for crit, want in v.items():
                    if not want:
                        continue
                    val = c[crit]
                    if val is None:           # not applicable (ROCE for a lender)
                        continue
                    ok = ok and val
                masks[vk][mi, ci] = ok
            if mi == len(months) - 1:
                diag.append(dict(symbol=s, **{k: (round(v, 3) if isinstance(v, float)
                                                  else v) for k, v in m.items()},
                                 **{'pass_' + k: c[k] for k in c}))
    for vk in VARIANTS:
        p = RES / ('fund_mask_%s.npz' % vk)
        np.savez_compressed(p, dates=np.array([str(d.date()) for d in months]),
                            cols=np.array(cols), mask=masks[vk])
        rate = 100.0 * masks[vk][-1].mean()
        print('  %-12s latest-month pass rate %5.1f%%  (%d of %d names)  -> %s'
              % (vk, rate, int(masks[vk][-1].sum()), len(cols), p.name))
    dd = pd.DataFrame(diag)
    dd.to_csv(RES / 'fund_diagnostics.csv', index=False)
    print()
    print('per-criterion pass rate at the latest month (of %d names with data):' % len(dd))
    for k in ['growth', 'de', 'roe', 'roce', 'neg']:
        col = dd['pass_' + k]
        na = int(col.isna().sum())
        print('   %-8s pass %3d   fail %3d   n/a %3d'
              % (k, int((col == True).sum()), int((col == False).sum()), na))  # noqa: E712
    print()
    print('wrote %s' % (RES / 'fund_diagnostics.csv'))


if __name__ == '__main__':
    main()
