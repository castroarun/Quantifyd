# -*- coding: utf-8 -*-
"""research/160 step 2 - the candidate universe for the fundamental leg.

Every daily-timeframe symbol in market_data.db, minus funds, minus stubs.

FUNDS ARE EXCLUDED BY NAME, NOT BY TICKER. backtest_data/etf_exclusions.json is built from
the broker's long instrument name - every ETF says what it is there, no operating company
does. The old ticker regex is still applied as a second pass, because the two disagree in
both directions and a fund slipping into an equity book is not a tidiness problem: in r/142
gold funds manufactured a fake +32% arm.

STUBS are symbols with fewer than 250 daily rows, or nothing at all after 2015-01-01. A name
with no price history cannot be traded and cannot anchor an all-time high.

SUFFIXES. NSE series suffixes (-BE trade-to-trade, -SM SME, -BZ, ...) are part of the symbol
in market_data.db but not part of a Screener URL. The stripped ticker is carried as its own
column so the mapping is explicit and reversible; the mask files keep the DB spelling.

max_tv20 is the HIGH-WATER MARK of the 20-day median traded value, in rupees crore. It is
deliberately not a current-liquidity measure: it answers "was this name ever liquid enough
to matter", which is the question the survivorship audit needs to ask about dead names.
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/160_quality_growth_near_ath/results'
RES.mkdir(parents=True, exist_ok=True)
DB = ROOT / 'backtest_data/market_data.db'

# the legacy r/142 regex, kept as a second pass only
ETF_RE = re.compile(r'(BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50)')
SUFFIX_RE = re.compile(r'-(BE|BZ|SM|ST|IT|BL|GS|GB|N\d+|RE)$')

MIN_ROWS = 250
ACTIVE_FROM = '2026-08-01'


def screener_ticker(sym):
    return SUFFIX_RE.sub('', sym)


def main():
    excl = set(json.load(open(ROOT / 'backtest_data/etf_exclusions.json'))['symbols'])
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    print('reading daily bars ...', flush=True)
    df = pd.read_sql_query(
        "select symbol, date, close, volume from market_data_unified "
        "where timeframe='day' order by symbol, date", con)
    con.close()
    print('  %d rows, %d symbols' % (len(df), df.symbol.nunique()), flush=True)

    df['tv'] = df.close.astype(float) * df.volume.astype(float) / 1e7   # rupees crore

    rows = []
    for sym, g in df.groupby('symbol', sort=True):
        n = len(g)
        first, last = g.date.iloc[0], g.date.iloc[-1]
        # 20-day MEDIAN traded value, then its running maximum: median resists the single
        # news-day volume spike that a mean would let define a dead name as liquid.
        tv20 = g.tv.rolling(20).median()
        rows.append(dict(symbol=sym, screener_ticker=screener_ticker(sym),
                         first_date=first, last_date=last, n_rows=n,
                         max_tv20=float(np.nanmax(tv20.values)) if n >= 20 else 0.0,
                         active=last >= ACTIVE_FROM))
    u = pd.DataFrame(rows)
    n_all = len(u)

    is_fund = u.symbol.isin(excl) | u.symbol.str.contains(ETF_RE)
    n_fund = int(is_fund.sum())
    n_by_json_only = int((u.symbol.isin(excl) & ~u.symbol.str.contains(ETF_RE)).sum())
    n_by_re_only = int((~u.symbol.isin(excl) & u.symbol.str.contains(ETF_RE)).sum())
    u = u[~is_fund]

    stub = (u.n_rows < MIN_ROWS) | (u.last_date < '2015-01-01')
    n_stub = int(stub.sum())
    u = u[~stub].copy()

    u['max_tv20'] = u.max_tv20.round(3)
    u = u.sort_values('max_tv20', ascending=False).reset_index(drop=True)
    u.to_csv(RES / 'universe.csv', index=False)

    dup = u.screener_ticker.duplicated(keep=False)
    print()
    print('all day symbols            : %d' % n_all)
    print('  funds excluded           : %d  (json-only %d, regex-only %d)'
          % (n_fund, n_by_json_only, n_by_re_only))
    print('  stubs excluded           : %d  (<%d rows or nothing after 2015)'
          % (n_stub, MIN_ROWS))
    print('UNIVERSE                   : %d' % len(u))
    print('  active (last >= %s) : %d' % (ACTIVE_FROM, int(u.active.sum())))
    print('  inactive / delisted      : %d' % int((~u.active).sum()))
    print('  ever tv20 >= 1 cr        : %d' % int((u.max_tv20 >= 1).sum()))
    print('  ever tv20 >= 5 cr        : %d' % int((u.max_tv20 >= 5).sum()))
    print('  suffixed symbols         : %d' % int((u.symbol != u.screener_ticker).sum()))
    if dup.any():
        print('  WARNING duplicate screener tickers: %s'
              % u[dup].sort_values('screener_ticker')[['symbol', 'screener_ticker']]
              .to_dict('records'))
    print()
    print('wrote %s' % (RES / 'universe.csv'))


if __name__ == '__main__':
    sys.exit(main())
