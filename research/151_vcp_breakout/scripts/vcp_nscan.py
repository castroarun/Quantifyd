"""P1c — is the VCP pivot a FIXED N-day closing high?

P1b established: for all 37 usable ground-truth trades the buy price is a prior CLOSE
that has not been exceeded on a closing basis since (37/37), the entry day closes above
it (37/37), and there is NO minimum base length or contraction count (pivot_age runs
1..157; 11/37 bases contain zero measurable contractions).

So the pivot behaves like "the highest close of the last N days". This probe finds N.

For each N in a grid we compute, per trade:
  price_ok  max(close[i-N : i]) == buy  (within 0.15%)
  first_ok  close[i] > that max AND close[i-1] <= max(close[i-1-N : i-1])   (first break)
and we also record run_length = trading days since the last close strictly above buy
(= how long the pivot has stood). A fixed-N rule requires N >= max(pivot_age) and
N <= min(run_length); if those two bounds cross, no fixed N can explain their pivots.
"""
import csv
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
DB = ROOT / 'backtest_data' / 'market_data.db'
GT = STUDY / 'data' / 'vcp_trades_groundtruth.csv'
TOL = 0.0015
NS = [10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 120, 150, 180, 200, 252, 400, 750, 100000]


def main():
    db = sqlite3.connect(str(DB))
    gt = list(csv.DictReader(open(GT)))
    per_trade = []
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
        i = df.index.get_loc(ed)
        buy = float(t['buy'])
        C = df['close'].values
        H = df['high'].values
        hist = C[:i]
        if len(hist) < 60:
            continue
        above = np.nonzero(hist > buy * (1 + TOL))[0]
        run_length = (i - 1 - above[-1]) if len(above) else i   # bars since a higher close
        pidx = int(np.argmin(np.abs(hist - buy)))
        # last bar whose close == buy
        eq = np.nonzero(np.abs(hist - buy) / buy <= TOL)[0]
        if len(eq):
            pidx = int(eq[-1])
        pivot_age = i - pidx
        rec = dict(symbol=s, entry=t['entry_date'], buy=buy,
                   pivot_age=pivot_age, run_length=int(run_length))
        for N in NS:
            lo = max(0, i - N)
            mc = float(C[lo:i].max())
            mh = float(H[lo:i].max())
            rec[f'c{N}'] = abs(mc - buy) / buy <= TOL
            rec[f'h{N}'] = abs(mh - buy) / buy <= TOL
            # first-break check on the close-basis rule
            lo2 = max(0, i - 1 - N)
            prev_max = float(C[lo2:i - 1].max()) if i - 1 > lo2 else np.nan
            rec[f'f{N}'] = bool(C[i] > mc and (np.isnan(prev_max) or C[i - 1] <= prev_max))
        per_trade.append(rec)
    db.close()
    d = pd.DataFrame(per_trade)
    d.to_csv(STUDY / 'results' / 'p1c_nscan.csv', index=False)
    n = len(d)
    print(f'=== P1c FIXED-N SCAN — {n} trades ===')
    print(f'pivot_age max = {d.pivot_age.max()}   run_length min = {d.run_length.min()}  '
          f'median = {d.run_length.median():.0f}   run_length>=252: {int((d.run_length>=252).sum())}/{n}')
    print(f'=> a fixed N must satisfy N >= {d.pivot_age.max()} and N <= {d.run_length.min()}: '
          f'{"FEASIBLE" if d.pivot_age.max() <= d.run_length.min() else "IMPOSSIBLE"}')
    print(f'\n{"N":>7s} {"close-high match":>17s} {"high-high match":>16s} {"first-break":>12s}')
    for N in NS:
        print(f'{N:7d} {int(d[f"c{N}"].sum()):10d}/{n:<6d} {int(d[f"h{N}"].sum()):9d}/{n:<6d} '
              f'{int(d[f"f{N}"].sum()):7d}/{n:<4d}')
    print('\nrun_length distribution:')
    print(d.run_length.describe(percentiles=[.05, .1, .25, .5, .75, .9]).round(0).to_string())
    print('\ntightest trades (smallest run_length):')
    print(d.nsmallest(8, 'run_length')[['symbol', 'entry', 'pivot_age', 'run_length']].to_string(index=False))


if __name__ == '__main__':
    main()
