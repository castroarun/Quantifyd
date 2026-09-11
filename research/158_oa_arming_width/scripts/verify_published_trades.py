# -*- coding: utf-8 -*-
"""Are the site's own published trades obtainable with a resting buy-stop?

Arun verified sample trades from the earlier reports by hand and they checked out. That
verification is valid and nothing here contradicts it: the listed trades ARE internally
correct. The stock really did trade through the pivot, so the fill price was genuinely
available, and the exits follow the stated rules.

The claim being tested is about what is NOT in the list. A buy-stop resting at the pivot
fills on EVERY touch, including the touches that close back below it. Those fills cannot
appear in a published trade list, so verifying the listed trades can never surface them.

So for each of their 54 published Blue Sky trades this reconstructs the running
all-time-high-close pivot from our daily bars and asks two questions:

  1. ON THEIR ENTRY DAY, did the close finish ABOVE the pivot? If that is true on
     essentially every trade, their engine only books breakouts that HELD, which is the
     look-ahead: a resting order cannot decline the ones that did not.

  2. BEFORE their entry day, how many times had a resting order at the SAME pivot level
     already been filled and failed? Each of those is a trade the live book takes and
     their list does not show. If the count is high, the published track record is not
     reachable by placing orders.

Read-only. No state, no orders, no engine changes.
"""
import csv
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path('/home/arun/quantifyd')
DB = ROOT / 'backtest_data' / 'market_data.db'
GT = ROOT / 'research/142_bananapatterns_replication/data/trades_groundtruth_bluesky.csv'
OUT = ROOT / 'research/158_oa_arming_width/results/published_trades_audit.csv'
LOOKBACK = 120          # trading days before their entry to inspect for earlier fills
BASE_DEPTH = 0.8        # the screen's "within 20% of the pivot"

con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)


def bars(sym):
    df = pd.read_sql_query(
        "select date, open, high, low, close from market_data_unified "
        "where symbol=? and timeframe='day' order by date", con, params=(sym,))
    if df.empty:
        return df
    df['date'] = df['date'].str[:10]
    df = df.drop_duplicates('date').set_index('date').sort_index()
    # the pivot a resting order sits at today = highest close up to YESTERDAY
    df['pivot'] = df['close'].shift(1).cummax()
    return df


rows = list(csv.DictReader(open(GT)))
print('%d published Blue Sky trades' % len(rows))
print()

out = []
held_yes = held_no = 0
gap_open = 0
missing = scale_bad = 0
total_earlier_fills = 0

for r in rows:
    sym, ed = r['symbol'], r['entry_date']
    try:
        buy = float(r['buy'])
    except (TypeError, ValueError):
        continue
    df = bars(sym)
    if df.empty or ed not in df.index:
        missing += 1
        continue
    d = df.loc[ed]
    # The DB is not retroactively split-adjusted (the study found 72 broken symbols).
    # If their buy price is nowhere near our bar, it is a scale artefact, not evidence.
    if not (d['low'] * 0.9 <= buy <= d['high'] * 1.1):
        scale_bad += 1
        out.append(dict(symbol=sym, entry_date=ed, buy=buy, verdict='SPLIT_SCALE_MISMATCH',
                        o=d['open'], h=d['high'], l=d['low'], c=d['close'],
                        pivot='', held='', open_above='', earlier_fills='',
                        earlier_dates=''))
        continue

    held = d['close'] > buy
    held_yes += bool(held)
    held_no += (not held)
    oa = d['open'] > buy
    gap_open += bool(oa)

    # earlier fills: days in the window where a resting order at a pivot <= their buy was
    # touched, and the close did NOT hold above that pivot
    win = df.loc[:ed].iloc[-(LOOKBACK + 1):-1]
    ef = 0
    ef_dates = []
    for dt, b in win.iterrows():
        piv = b['pivot']
        if pd.isna(piv) or piv <= 0:
            continue
        prev_c = df['close'].shift(1).get(dt)
        if pd.isna(prev_c) or not (prev_c < piv and prev_c >= BASE_DEPTH * piv):
            continue                      # not in setup that day
        if b['high'] >= piv and b['close'] <= piv:
            ef += 1
            ef_dates.append(dt)
    total_earlier_fills += ef
    out.append(dict(symbol=sym, entry_date=ed, buy=round(buy, 2),
                    verdict='held' if held else 'CLOSED_BELOW_PIVOT',
                    o=round(d['open'], 2), h=round(d['high'], 2),
                    l=round(d['low'], 2), c=round(d['close'], 2),
                    pivot=round(d['pivot'], 2) if pd.notna(d['pivot']) else '',
                    held=int(held), open_above=int(oa), earlier_fills=ef,
                    earlier_dates=';'.join(ef_dates[-4:])))

con.close()
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['symbol', 'entry_date', 'buy', 'verdict',
                                      'o', 'h', 'l', 'c', 'pivot', 'held',
                                      'open_above', 'earlier_fills', 'earlier_dates'])
    w.writeheader()
    w.writerows(out)

n = held_yes + held_no
print('=== QUESTION 1: on their entry day, did the close hold above the pivot? ===')
print('  closed ABOVE the pivot : %d of %d  (%.0f%%)' % (held_yes, n, 100.0 * held_yes / n))
print('  closed BELOW the pivot : %d of %d' % (held_no, n))
print('  opened ABOVE their buy : %d of %d  -- a real stop fills at the OPEN here,'
      % (gap_open, n))
print('                            not at the pivot they booked (fill inflation)')
print('  unusable (split scale) : %d   not in DB: %d' % (scale_bad, missing))
print()
print('=== QUESTION 2: fills their list does not show ===')
print('  resting-order fills that FAILED in the %d trading days before their entries:'
      % LOOKBACK)
print('  %d across %d trades  = %.1f per published trade'
      % (total_earlier_fills, n, total_earlier_fills / n if n else 0))
print()
print('  Each is a day a stop resting at the pivot was filled and the close did not hold.')
print('  A live order takes them. Their trade list contains none of them.')
print()
print('wrote %s' % OUT)
