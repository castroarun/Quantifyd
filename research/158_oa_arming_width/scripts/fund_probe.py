# -*- coding: utf-8 -*-
"""Can we actually build Arun's five fundamental criteria, per year, for these names?

The existing service reported ROE for only 13 of 137 cached symbols and never computed ROCE
at all. That may be its extraction logic rather than missing data, so this goes at the raw
annual statements instead and computes both from first principles:

    ROE  = net income / total shareholder equity
    ROCE = EBIT / (total assets - total current liabilities)

and the rest of the criteria:

    profit growth 3y  > 15%      net income CAGR over three full years
    sales growth 3y   > 15%      revenue CAGR over three full years
    debt / equity     <= 0.20
    no negatives in the last three years

This is a FEASIBILITY probe, not the test. It answers one question: for the names this book
actually trades - small and mid caps at a Rs 5 crore a day floor, not the Nifty large caps
already in the cache - does the annual data come back, and how often is each field present?

If coverage is poor, any screen built on it silently selects for data availability, which
favours larger names, and that is a selection effect masquerading as a quality filter. Worth
knowing before building anything.
"""
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import yfinance as yf

# names the live scanner actually surfaced on 10-Sep, plus a few from the site's own
# published trade list - deliberately NOT large caps
SYMS = ['CUPID', 'MOREPENLAB', 'AEROFLEX', 'CYIENTDLM', 'SKYGOLD', 'BODALCHEM',
        'BIRLACABLE', 'TBZ', 'FIEMIND', 'CHOICEIN', 'NSIL', 'ASALCBR',
        'RADICO', 'KARURVYSYA', 'HCLTECH']


def pick(df, names):
    """First matching row of a yfinance statement frame, as a Series over fiscal years."""
    if df is None or df.empty:
        return None
    for n in names:
        for idx in df.index:
            if str(idx).strip().lower() == n.lower():
                return df.loc[idx]
    return None


rows = []
for s in SYMS:
    rec = dict(symbol=s, years=0, revenue=0, net_income=0, equity=0, debt=0,
               ebit=0, assets=0, cur_liab=0, roe='', roce='', de='', err='')
    try:
        t = yf.Ticker(s + '.NS')
        inc = t.income_stmt
        bs = t.balance_sheet
        rev = pick(inc, ['Total Revenue', 'Operating Revenue'])
        ni = pick(inc, ['Net Income', 'Net Income Common Stockholders'])
        ebit = pick(inc, ['EBIT', 'Operating Income'])
        eq = pick(bs, ['Stockholders Equity', 'Total Equity Gross Minority Interest'])
        debt = pick(bs, ['Total Debt'])
        ta = pick(bs, ['Total Assets'])
        cl = pick(bs, ['Current Liabilities', 'Total Current Liabilities'])
        rec['years'] = 0 if rev is None else int(rev.notna().sum())
        for k, v in (('revenue', rev), ('net_income', ni), ('equity', eq),
                     ('debt', debt), ('ebit', ebit), ('assets', ta), ('cur_liab', cl)):
            rec[k] = 0 if v is None else int(v.notna().sum())
        # most recent full year, computed rather than read
        if ni is not None and eq is not None and ni.notna().any() and eq.notna().any():
            y = ni.dropna().index[0]
            if y in eq.index and pd.notna(eq[y]) and eq[y]:
                rec['roe'] = round(100.0 * ni[y] / eq[y], 1)
        if ebit is not None and ta is not None and cl is not None:
            cands = [i for i in ebit.dropna().index if i in ta.index and i in cl.index]
            if cands:
                y = cands[0]
                cap = ta[y] - cl[y]
                if pd.notna(cap) and cap:
                    rec['roce'] = round(100.0 * ebit[y] / cap, 1)
        if debt is not None and eq is not None:
            cands = [i for i in debt.dropna().index if i in eq.index and pd.notna(eq[i])]
            if cands:
                y = cands[0]
                if eq[y]:
                    rec['de'] = round(debt[y] / eq[y], 2)
    except Exception as e:
        rec['err'] = str(e)[:60]
    rows.append(rec)
    time.sleep(1.2)                 # Yahoo throttles; the VPS already saw a 429

df = pd.DataFrame(rows)
print()
print('%-12s %5s %4s %4s %4s %4s %4s %4s %4s  %7s %7s %6s  %s' % (
    'symbol', 'yrs', 'rev', 'ni', 'eq', 'debt', 'ebit', 'TA', 'CL',
    'ROE%', 'ROCE%', 'D/E', 'error'))
for _, r in df.iterrows():
    print('%-12s %5d %4d %4d %4d %4d %4d %4d %4d  %7s %7s %6s  %s' % (
        r.symbol, r.years, r.revenue, r.net_income, r.equity, r.debt, r.ebit,
        r.assets, r.cur_liab, r.roe, r.roce, r.de, r.err))

n = len(df)
print()
print('of %d symbols probed:' % n)
print('  at least 4 annual revenue years : %d' % int((df.years >= 4).sum()))
print('  ROE computable                  : %d' % int((df.roe != '').sum()))
print('  ROCE computable                 : %d' % int((df.roce != '').sum()))
print('  debt/equity computable          : %d' % int((df.de != '').sum()))
print('  hard errors                     : %d' % int((df.err != '').sum()))
