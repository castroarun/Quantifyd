"""Why doesn't the replica take their trades? Per-GT-trade condition autopsy."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
DB = ROOT / 'backtest_data' / 'market_data.db'
TV_FLOOR = 5e7

sys.path.insert(0, str(STUDY / 'scripts'))
from bluesky_replay import load_frames, rs_percentile  # noqa: E402

w = load_frames()
close, high, athcp, tv20 = w['close'], w['high'], w['athcp'], w['tv20']
eligible = tv20.shift(1) >= TV_FLOOR
rs_ibd = rs_percentile(close, eligible, plain=False).shift(1)
rs_pln = rs_percentile(close, eligible, plain=True).shift(1)
prev_close = close.shift(1)

gt = pd.read_csv(STUDY / 'data' / 'trades_groundtruth_bluesky.csv')
print(f"{'symbol':12s} {'entry':10s} {'buy':>9s} {'piv%':>6s} cross prevlt w20 elig rs_ibd rs_pln")
for _, g in gt.iterrows():
    s, d = g.symbol, pd.Timestamp(g.entry_date)
    if s not in close.columns or d not in close.index:
        print(f'{s:12s} {g.entry_date} NO DATA/DATE')
        continue
    piv = athcp.loc[d, s]
    pivpct = (g.buy / piv - 1) * 100 if piv and not np.isnan(piv) else np.nan
    cross = bool(high.loc[d, s] >= piv) if not np.isnan(piv) else False
    prevlt = bool(prev_close.loc[d, s] < piv) if not np.isnan(piv) else False
    w20 = bool(prev_close.loc[d, s] >= 0.8 * piv) if not np.isnan(piv) else False
    el = bool(eligible.loc[d, s]) if not np.isnan(tv20.shift(1).loc[d, s]) else False
    ri = rs_ibd.loc[d, s]
    rp = rs_pln.loc[d, s]
    tv = tv20.shift(1).loc[d, s]
    print(f'{s:12s} {g.entry_date} {g.buy:9.2f} {pivpct:6.2f} {str(cross):5s} '
          f'{str(prevlt):6s} {str(w20):4s} {str(el):4s} {ri:6.1f} {rp:6.1f}  tv={tv/1e7:.1f}cr')
