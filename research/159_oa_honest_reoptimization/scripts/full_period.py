# -*- coding: utf-8 -*-
"""The four live/candidate books over the FULL common period, with cash occupancy.

Arun asked for three things on the table itself: the test period, the average time each book
spends in cash rather than in positions, and the cash-yield assumption stated as a caveat.
And for the comparison to run over the entire common history rather than the clipped decade.

The clip existed only because Open Alpha v3's VIX gate needs INDIA VIX, which starts 2015.
v3 is superseded by v2, so dropping it from the table frees the window: every remaining book
has history from 2006.

    True North      research/144 after-tax NAV        from 2006-04-03
    OA Base Age     research/161 WINNER curve         from 2005-01-03
    IPO Base        honest next-day entry, after tax  from 2006-01-02
    NIFTYBEES       index price series                from 2005-01-03

Common start is therefore True North's, 2006-04-03.

CASH OCCUPANCY matters more than it looks and is measured, not assumed. True North's gate
LIQUIDATES rather than merely blocking new buys, so it holds cash 57% of the time and still
produces the best return in the book - a far stronger result than the CAGR alone says. IPO
Base holds roughly two thirds cash by design, so a meaningful slice of its return is the
sweep rather than the strategy, and quoting its CAGR beside a fully invested index without
saying so would mislead.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/159_oa_honest_reoptimization/results'

navs = {}
z = np.load(ROOT / 'research/161_ath_base_age_breakout/results/curves161.npz',
            allow_pickle=True)
d = pd.to_datetime([str(x)[:10] for x in z['dates']])
navs['Open Alpha - Base Age'] = pd.Series(np.asarray(z['WINNER'], dtype=float), index=d)
navs['True North'] = pd.read_csv(
    ROOT / 'research/144_truenorth_reassessment/results/nav_INC_cash_n8_d15_tax1.csv',
    index_col=0, parse_dates=True).iloc[:, 0]
navs['IPO Base - First Base'] = pd.read_csv(
    RES / 'ipo_honest_curve.csv', index_col=0, parse_dates=True).iloc[:, 0]
nb = pd.read_csv(RES / 'curves_after_tax.csv', index_col=0,
                 parse_dates=True)['NIFTYBEES']

# NIFTYBEES from curves_after_tax.csv starts 2016; pull the full series instead
import sqlite3
con = sqlite3.connect('file:%s?mode=ro' % (ROOT / 'backtest_data/market_data.db'), uri=True)
q = pd.read_sql_query("select date, close from market_data_unified where "
                      "symbol='NIFTYBEES' and timeframe='day' order by date", con)
con.close()
q['date'] = pd.to_datetime(q['date'].str[:10])
navs['NIFTYBEES (index)'] = q.drop_duplicates('date').set_index('date')['close']

spans = {k: (v.dropna().index.min(), v.dropna().index.max()) for k, v in navs.items()}
start = max(s for s, _ in spans.values())
end = min(e for _, e in spans.values())
print('individual spans:')
for k, (a, b) in spans.items():
    print('  %-24s %s -> %s' % (k, a.date(), b.date()))
print()
print('COMMON FULL PERIOD: %s -> %s' % (start.date(), end.date()))

al = pd.DataFrame({k: v for k, v in navs.items()}).loc[start:end].ffill().dropna()
al = al / al.iloc[0]
yrs = (al.index[-1] - al.index[0]).days / 365.25

# measured cash occupancy, from each engine's own reporting
INVESTED = {'True North': 43.0,                 # research/144 phaseA avg_inv 0.43, gate=sma100
            'Open Alpha - Base Age': None,      # to be measured in its own harness
            'IPO Base - First Base': 32.7,      # research/153 g3 'invested 32.7% of NAV'
            'NIFTYBEES (index)': 100.0}

print()
print('%-24s %9s %10s %8s %9s %9s %9s' %
      ('system', 'CAGR', 'maxDD', 'Calmar', 'final', 'invested', 'in cash'))
out = {}
for k in al:
    s = al[k]
    cagr = (s.iloc[-1] ** (1 / yrs) - 1) * 100
    dd = (s / s.cummax() - 1).min() * 100
    inv = INVESTED.get(k)
    out[k] = dict(cagr=round(cagr, 2), dd=round(dd, 2), calmar=round(cagr / abs(dd), 2),
                  final=round(s.iloc[-1], 2), invested=inv,
                  span='%s to %s' % (spans[k][0].date(), spans[k][1].date()))
    print('%-24s %8.2f%% %9.2f%% %8.2f %8.2fx %8s %9s'
          % (k, cagr, dd, cagr / abs(dd), s.iloc[-1],
             ('%.0f%%' % inv) if inv else 'n/m',
             ('%.0f%%' % (100 - inv)) if inv else 'n/m'))

al.to_csv(RES / 'full_period_after_tax.csv')
json.dump(out, open(RES / 'full_period_summary.json', 'w'), indent=1)
print()
print('years: %.1f' % yrs)
print('wrote full_period_after_tax.csv')
