# -*- coding: utf-8 -*-
"""How often does a resting buy-stop at the pivot actually fill?

Read-only. Nothing is placed, no state is touched, no rule is changed.

The live book arms a buy-stop at the pivot for each FREE SLOT and lets that pending order
hold the slot. The armed names sit 1-13% below their pivot, so the question that decides
whether to arm wider is simply: what fraction of those stops get touched, and how soon?

Measured on the same universe the scanner uses (>=260 daily bars), over the same setup
condition (prev_close below the ATH-close, and at least 0.8x of it), bucketed by how far
below the pivot the name sits when armed - because that is the number the scan prints.
"""
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
DB = ROOT / 'backtest_data' / 'market_data.db'
START = '2015-01-01'

con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
syms = [r[0] for r in con.execute(
    "select symbol from (select symbol, count(*) n from market_data_unified "
    "where timeframe='day' group by symbol) where n >= 260")]
print('%d symbols' % len(syms), flush=True)

BUCKETS = [(0, 1), (1, 2), (2, 4), (4, 7), (7, 11), (11, 20)]
HORIZONS = [1, 2, 3, 5, 10]
hit = {(b, h): 0 for b in range(len(BUCKETS)) for h in HORIZONS}
tot = {b: 0 for b in range(len(BUCKETS))}

t0 = time.time()
for n, s in enumerate(syms):
    df = pd.read_sql_query(
        "select date, high, close from market_data_unified "
        "where symbol=? and timeframe='day' order by date", con, params=(s,))
    if len(df) < 260:
        continue
    df['date'] = df['date'].str[:10]
    df = df.drop_duplicates('date')
    c = df['close'].to_numpy(dtype='float64')
    h = df['high'].to_numpy(dtype='float64')
    # pivot = highest close up to and including the bar we arm from (the scan's definition)
    piv = np.maximum.accumulate(np.where(np.isnan(c), -np.inf, c))
    d = df['date'].to_numpy()
    m = d >= START
    idx = np.nonzero(m)[0]
    for i in idx:
        if i + 1 >= len(c):
            break
        p, pc = piv[i], c[i]
        if not np.isfinite(p) or not np.isfinite(pc) or p <= 0:
            continue
        # the scan's setup: below the pivot, but within the 0.8x base
        if not (pc < p and pc >= 0.8 * p):
            continue
        gap = (p / pc - 1) * 100.0
        b = next((j for j, (lo, hi) in enumerate(BUCKETS) if lo <= gap < hi), None)
        if b is None:
            continue
        tot[b] += 1
        for hz in HORIZONS:
            w = h[i + 1:i + 1 + hz]
            if w.size and np.nanmax(w) >= p:
                hit[(b, hz)] += 1
    if (n + 1) % 600 == 0:
        print('  %d/%d (%.0fs)' % (n + 1, len(syms), time.time() - t0), flush=True)
con.close()

print()
print('Chance a buy-stop at the pivot is touched, by how far below the pivot it was armed')
print('(setups on the scanner universe, %s onward)' % START)
print()
hdr = '  %-12s %9s' % ('gap to pivot', 'setups')
for hz in HORIZONS:
    hdr += '  %6s' % ('%dd' % hz)
print(hdr)
for b, (lo, hi) in enumerate(BUCKETS):
    if not tot[b]:
        continue
    row = '  %-12s %9d' % ('%g-%g%%' % (lo, hi), tot[b])
    for hz in HORIZONS:
        row += '  %5.1f%%' % (100.0 * hit[(b, hz)] / tot[b])
    print(row)

# what the scan armed this morning, for the answer to "how many should we arm"
print()
print('This morning the scan armed these gaps: CUPID 6.64, MOREPENLAB 2.93, '
      'AEROFLEX 1.15, CYIENTDLM 6.13')
one_day = []
for b, (lo, hi) in enumerate(BUCKETS):
    if tot[b]:
        one_day.append((lo, hi, 100.0 * hit[(b, 1)] / tot[b]))
print('Next-day touch rate by bucket: ' + ', '.join(
    '%g-%g%% -> %.0f%%' % x for x in one_day))
