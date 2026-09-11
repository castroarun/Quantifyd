# -*- coding: utf-8 -*-
"""Build a durable ETF exclusion list from what the instrument is CALLED, not its ticker.

The current filter matches tickers: (BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50). It was
written against the ETF names that existed when r/142 was built, and the 2023-2025 wave of
gold and silver funds is named nothing like any of them - EGOLD, GOLD1, GOLDADD, GROWWGOLD,
HDFCGOLD, TATAGOLD, ESILVER, SILVERBETA, MON100, MAFANG, ICICIB22 and the rest all sailed
through into an Indian EQUITY momentum book.

A longer ticker blacklist would work until the next fund launches, which is not a fix. The
durable signal is the instrument's long name in the Kite dump, which the scanner already
downloads for tick sizes:

    HDFCGOLD    EQ   HDFC GOLD ETF                     <- fund
    CHOICEGOLD  EQ   CHOICE GOLD ETF                   <- fund
    SKYGOLD     EQ   SKY GOLD AND DIAMONDS             <- company
    DECNGOLD    EQ   DECCAN GOLD MINES                 <- company
    SILVERTUC   EQ   SILVER TOUCH TECHNO               <- company

Every fund says so in its name; no operating company does. That test needs no maintenance.

Two things it cannot do alone, so both are kept:
  * a fund that has since DELISTED is not in today's dump, and a backtest still meets it in
    the old data. The ticker patterns stay as a second net for those.
  * the dump is a live call. The result is written to a committed JSON so backtests are
    reproducible without network access and without drifting between runs.

Writes: backtest_data/etf_exclusions.json
"""
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT))
from services.oa_real import _kite          # noqa: E402

# what the instrument is CALLED - the durable test
# 'AMC - SYMBOL' is how Kite names a few fund units (AONEAMC - AONESILVER,
# HSBCAMC - HSBCGOLD). The dash is load-bearing: real fund-management COMPANIES
# read 'HDFC AMC', 'UTI ASSET MNGMT CO', 'ADIT BIRL SUN LIF AMC' and must stay.
NAME_RE = re.compile(r'\bETF\b|EXCHANGE TRADED|\bBEES\b|AMC\s+-\s', re.I)
# second net, for funds that have delisted and are absent from today's dump
TICKER_RE = re.compile(
    r'(BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50'
    r'|^GOLD$|^GOLD[0-9]|^EGOLD$|^GOLDADD$|^GOLDAXIS$|^GOLDBETA$|^GOLDCASE$'
    r'|^GROWWGOLD$|^HDFCGOLD$|^LICMFGOLD$|^TATAGOLD$|^AXISGOLD$|^QGOLDHALF$'
    r'|^SILVER$|^SILVER[0-9]|^ESILVER$|^SILVERADD$|^SILVERAG$|^SILVERBETA$'
    r'|^HDFCSILVER$|^SBISILVER$|^TATSILV$|^GROWWSLVR$'
    r'|^MON100$|^MAFANG$|^ICICIB22$|^METAL$|^MODEFENCE$|^HNGSNGBEES$)')

ins = [i for i in _kite().instruments('NSE') if i.get('segment') == 'NSE']
by_name = {i['tradingsymbol']: (i.get('name') or '') for i in ins}
print('NSE instruments in the dump: %d' % len(ins))

con = sqlite3.connect('file:%s?mode=ro' % (ROOT / 'backtest_data/market_data.db'), uri=True)
syms = sorted(r[0] for r in con.execute(
    "select distinct symbol from market_data_unified where timeframe='day'"))
con.close()

by_nm = {s for s in syms if NAME_RE.search(by_name.get(s, ''))}
by_tk = {s for s in syms if TICKER_RE.search(s)}
excl = sorted(by_nm | by_tk)

print('universe symbols          : %d' % len(syms))
print('excluded by long NAME     : %d' % len(by_nm))
print('excluded by ticker only   : %d  (delisted or absent from the dump)'
      % len(by_tk - by_nm))
print('TOTAL excluded            : %d' % len(excl))

# the check that matters: nothing real gets deleted
KEEP = ['SKYGOLD', 'GOLDIAM', 'SILVERTUC', 'SHANTIGOLD', 'DECNGOLD', 'TITAN',
        'KALYANKJIL', 'PCJEWELLER', 'THANGAMAYL', 'RAJESHEXPO', 'VAIBHAVGBL']
present = [s for s in KEEP if s in syms]
wrong = [s for s in present if s in set(excl)]
print()
print('real companies checked : %s' % ', '.join(present))
print('wrongly excluded       : %s' % (', '.join(wrong) if wrong else 'NONE'))

left = [s for s in syms if re.search(r'GOLD|SILVER|SLVR', s) and s not in set(excl)]
print()
print('gold/silver-named symbols still IN the universe (%d):' % len(left))
for s in left:
    print('   %-14s %s' % (s, by_name.get(s, '(not in dump)')))

out = ROOT / 'backtest_data/etf_exclusions.json'
json.dump(dict(built=datetime.now().strftime('%Y-%m-%d %H:%M'),
               n=len(excl),
               by_name=sorted(by_nm), by_ticker_only=sorted(by_tk - by_nm),
               symbols=excl), open(out, 'w'), indent=1)
print()
print('wrote %s' % out)
