# -*- coding: utf-8 -*-
"""Why did IPO Base never take Sri Lotus Developers? Check the spec, condition by condition.

Rather than eyeball a chart, this replays the adopted r/153 rules over every bar the database
holds for LOTUSDEV and reports which condition failed on which day:

    listing      must be in the VETTED listing table (accepted), not the naive first-row proxy
    age band     listed <= 6 months ago AND >= 25 bars of history
    liquidity    20-day median traded value >= Rs 5 crore, measured on the PREVIOUS bar
    base         pivot = highest close of the last 25 bars, shifted 1
    depth        (pivot - lowest low of the base) / pivot <= 30%
    not extended prev_close < pivot
    trigger      close > pivot

A name can fail several at once, so every condition is reported for every bar rather than
short-circuiting at the first failure - otherwise "it failed the age test" hides whether it
would have passed everything else.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
SYM = sys.argv[1] if len(sys.argv) > 1 else 'LOTUSDEV'
TODAY = pd.Timestamp('2026-09-11')

con = sqlite3.connect('file:%s?mode=ro' % (ROOT / 'backtest_data/market_data.db'), uri=True)
df = pd.read_sql_query(
    "select date, open, high, low, close, volume from market_data_unified "
    "where symbol=? and timeframe='day' order by date", con, params=(SYM,))
con.close()
if df.empty:
    sys.exit('%s: no rows in market_data.db' % SYM)
df['date'] = pd.to_datetime(df['date'].str[:10])
df = df.drop_duplicates('date').set_index('date').sort_index()

first, last = df.index[0], df.index[-1]
print('%s' % SYM)
print('  rows in DB        : %d' % len(df))
print('  first bar         : %s' % first.date())
print('  last bar          : %s   (%d days stale as of %s)'
      % (last.date(), (TODAY - last).days, TODAY.date()))
print('  price first/last  : %.2f / %.2f' % (df['close'].iloc[0], df['close'].iloc[-1]))

# vetted listing table
lt = None
for p in (ROOT / 'research/153_ipo_base/data/listing_dates.csv',
          ROOT / 'backtest_data/listing_dates.csv'):
    if p.exists():
        lt = pd.read_csv(p)
        break
if lt is None:
    print('  listing table     : NOT FOUND on disk')
else:
    col = 'symbol' if 'symbol' in lt.columns else lt.columns[0]
    row = lt[lt[col].astype(str).str.upper() == SYM]
    if row.empty:
        print('  listing table     : %s is ABSENT from the vetted table (%d rows)'
              % (SYM, len(lt)))
    else:
        print('  listing table     : %s' % row.iloc[0].to_dict())

# the spec
L, DEPTH, TV_FLOOR, MAX_AGE_M, MIN_BARS = 25, 0.30, 5e7, 6, 25
c, h, lo, v = df['close'], df['high'], df['low'], df['volume']
tv20 = (c * v).rolling(20).median().shift(1)
pivot = c.rolling(L).max().shift(1)
baselow = lo.rolling(L).min().shift(1)
depth = (pivot - baselow) / pivot
bars = pd.Series(np.arange(1, len(df) + 1), index=df.index)
age_days = (df.index - first).days
prev_c = c.shift(1)

ok_age = (age_days <= MAX_AGE_M * 30.44) & (bars >= MIN_BARS)
ok_tv = tv20 >= TV_FLOOR
ok_depth = depth <= DEPTH
ok_notext = prev_c < pivot
ok_trig = c > pivot
allok = ok_age & ok_tv & ok_depth & ok_notext & ok_trig

print()
print('  the 6-month window ran %s -> %s' %
      (first.date(), (first + pd.Timedelta(days=int(MAX_AGE_M * 30.44))).date()))
print('  bars inside it    : %d   (need >= %d for the base to exist)'
      % (int(ok_age.sum()), MIN_BARS))
print()
print('  condition pass-counts over the %d bars in the DB:' % len(df))
for name, s in [('age band (<=6m, >=25 bars)', ok_age), ('liquidity >= Rs5cr', ok_tv),
                ('base depth <= 30%', ok_depth), ('not already extended', ok_notext),
                ('TRIGGER close > pivot', ok_trig)]:
    print('    %-28s %4d of %d' % (name, int(s.fillna(False).sum()), len(df)))
print('    %-28s %4d' % ('ALL AT ONCE', int(allok.fillna(False).sum())))

inwin = df[ok_age.fillna(False)]
if len(inwin):
    print()
    print('  inside the age window, the closest it came:')
    sub = pd.DataFrame({'close': c, 'pivot': pivot, 'tv20_cr': tv20 / 1e7,
                        'depth': depth}).loc[inwin.index].dropna()
    if len(sub):
        sub['gap_to_pivot_%'] = (sub['pivot'] / sub['close'] - 1) * 100
        best = sub.nsmallest(5, 'gap_to_pivot_%')
        print(best[['close', 'pivot', 'gap_to_pivot_%', 'tv20_cr', 'depth']]
              .to_string(float_format=lambda x: '%.2f' % x))
if allok.fillna(False).any():
    print()
    print('  TRIGGERED ON: %s' % [str(d.date()) for d in df.index[allok.fillna(False)]])
