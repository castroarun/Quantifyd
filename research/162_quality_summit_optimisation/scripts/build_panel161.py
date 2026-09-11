# -*- coding: utf-8 -*-
"""research/162 Part B — rebuild research/161's price / exit-signal panel cache.

r/161's `results/panel161.pkl` is gitignored and no longer on disk, so Part B rebuilds it
from the same inputs with the same code (`bt_core162.py`, a byte-identical copy of r/161's
`bt_core.py`). Calendar and symbol set are r/161's exactly — NIFTYBEES sessions from
2005-01-03, every symbol that appears in `results/ath_events.csv` — so the no-mask control
cell must reproduce r/161's published WINNER row to the second decimal.

READ-ONLY on market_data.db. ~15 minutes, ~500 MB of pickle.
"""
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt_core162 as B                                                   # noqa: E402

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
R161 = ROOT / 'research' / '161_ath_base_age_breakout' / 'results'
OUT = Path(__file__).resolve().parents[1] / 'results' / 'panel161.pkl'


def main():
    t0 = time.time()
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    cal = [d for d in pd.read_sql_query(
        "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
        "AND symbol='NIFTYBEES' ORDER BY date", con)['date'].tolist() if d >= '2005-01-03']
    print('calendar %d sessions %s -> %s' % (len(cal), cal[0], cal[-1]), flush=True)
    ev = pd.read_csv(R161 / 'ath_events.csv')
    syms = sorted(ev['symbol'].unique())
    print('building exit-signal panel for %d symbols...' % len(syms), flush=True)
    panel = B.Panel(con, syms, cal)
    con.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(panel, open(OUT, 'wb'), protocol=4)
    print('panel: %d symbols kept, %.0f MB, %.0fs'
          % (len(panel.close), OUT.stat().st_size / 1e6, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
