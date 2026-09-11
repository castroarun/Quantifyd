"""
research/159 — scan the WHOLE daily universe for symbols carrying an identical
(date, close) history, e.g. JSWDULUX / AKZOINDIA (renamed company, both tickers kept).

Such pairs double-count one real trade, so any book built on this universe must keep
one symbol per series. Writes results/duplicate_series.csv.
"""
import hashlib
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

DB = Path(__file__).resolve().parents[3] / 'backtest_data' / 'market_data.db'
RES = Path(__file__).resolve().parents[1] / 'results'
con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
syms = [r[0] for r in con.execute(
    "SELECT symbol FROM market_data_unified WHERE timeframe='day' "
    "GROUP BY symbol HAVING COUNT(*)>=90 ORDER BY symbol")]
print('scanning %d symbols for identical close series...' % len(syms))

groups = defaultdict(list)
meta = {}
for i, s in enumerate(syms, 1):
    d = pd.read_sql_query(
        "SELECT date,close,volume FROM market_data_unified WHERE symbol=? AND timeframe='day' "
        "ORDER BY date", con, params=(s,))
    d = d[(d['volume'] > 0) & (d['close'] > 0)]
    if len(d) < 90:
        continue
    h = hashlib.md5((''.join(d['date'].astype(str).str[:10]) +
                     ''.join('%.4f' % x for x in d['close'])).encode()).hexdigest()[:16]
    groups[h].append(s)
    meta[s] = (len(d), d['date'].iloc[0][:10], d['date'].iloc[-1][:10])
    if i % 500 == 0:
        print('  %d/%d' % (i, len(syms)), flush=True)
con.close()

dups = {h: v for h, v in groups.items() if len(v) > 1}
rows = []
print('\n%d duplicate group(s) found:' % len(dups))
for h, v in sorted(dups.items(), key=lambda kv: kv[1][0]):
    keep = sorted(v, key=lambda s: (-meta[s][0], s))[0]
    for s in sorted(v):
        rows.append(dict(series_hash=h, symbol=s, bars=meta[s][0], first=meta[s][1],
                         last=meta[s][2], action='KEEP' if s == keep else 'DROP'))
    print('  %-45s bars=%d  %s -> %s   keep=%s'
          % (' = '.join(sorted(v)), meta[v[0]][0], meta[v[0]][1], meta[v[0]][2], keep))
if rows:
    pd.DataFrame(rows).to_csv(RES / 'duplicate_series.csv', index=False)
    print('\nwrote %s' % (RES / 'duplicate_series.csv'))
else:
    print('  none')
