"""Entry forensics: what level is their 'pivot'?

For each ground-truth trade with data, search (window w, gap d) combos:
pivot_candidate = max(high) over the w bars ending d bars before entry.
Report the (w,d) whose candidate is closest to their buy price, plus
prev-day anchors and a split-ratio check (their buy vs our entry-day range).
"""
import csv
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
DB = ROOT / 'backtest_data' / 'market_data.db'

WINDOWS = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 252, 9999]
GAPS = list(range(1, 11))

import sys
GT_NAME = sys.argv[1] if len(sys.argv) > 1 else 'trades_groundtruth.csv'
db = sqlite3.connect(str(DB))
gt = list(csv.DictReader(open(STUDY / 'data' / GT_NAME)))

print(f"{'symbol':12s} {'entry':10s} {'buy':>9s} {'ratio':>6s}  best(w,d)  "
      f"{'pivot':>9s} {'diff%':>7s}   prevH% prevC% dayO% dayH%")

for t in gt:
    s = t['symbol']
    df = pd.read_sql_query(
        "select date, open, high, low, close from market_data_unified "
        "where symbol=? and timeframe='day' order by date", db, params=(s,))
    if df.empty:
        continue
    df['date'] = pd.to_datetime(df['date'].str[:10])
    df = df.drop_duplicates('date').set_index('date').sort_index()
    ed = pd.Timestamp(t['entry_date'])
    if ed not in df.index:
        continue
    buy = float(t['buy'])
    i = df.index.get_loc(ed)
    eday = df.iloc[i]
    ratio = eday['close'] / buy  # split-scale check: ~1 = same scale
    hist = df.iloc[:i]
    best = None
    ath_h = hist['high'].max()
    ath_c = hist['close'].max()
    for d in GAPS:
        if d > len(hist):
            break
        endslice = hist.iloc[:len(hist) - (d - 1)]
        for w in WINDOWS:
            for col in ('high', 'close'):
                seg = endslice[col].tail(w)
                if not len(seg):
                    continue
                piv = seg.max()
                diff = abs(buy - piv) / piv * 100
                if best is None or diff < best[0]:
                    best = (diff, f'{w}{col[0].upper()}', d, piv)
    prev = hist.iloc[-1] if len(hist) else None
    fmt = lambda a, b: f'{(a - b) / b * 100:+.2f}' if b else 'n/a'
    print(f"{s:12s} {t['entry_date']} {buy:9.2f} {ratio:6.2f}  "
          f"w={best[1]:<6s}d={best[2]:<2d} {best[3]:9.2f} {best[0]:7.3f}   "
          f"{fmt(buy, prev['high'])} {fmt(buy, prev['close'])} "
          f"{fmt(buy, eday['open'])} {fmt(buy, eday['high'])}  "
          f"vsATHh={fmt(buy, ath_h)} vsATHc={fmt(buy, ath_c)}")
