"""
research/159 — validate the backtest engine before spending the sweep, and print the
NIFTYBEES buy-and-hold benchmark the whole study is judged against.
"""
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt_core as B

RES = Path(__file__).resolve().parents[1] / 'results'
DB = Path(__file__).resolve().parents[3] / 'backtest_data' / 'market_data.db'

con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
cal = [d for d in pd.read_sql_query(
    "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' AND symbol='NIFTYBEES' "
    "ORDER BY date", con)['date'].tolist() if d >= '2005-01-03']
print('calendar: %d days  %s -> %s' % (len(cal), cal[0], cal[-1]))

# ---------------- NIFTYBEES buy-and-hold benchmark (the adoption bar) -------------
nb = pd.read_sql_query(
    "SELECT date,open,close FROM market_data_unified WHERE timeframe='day' AND symbol='NIFTYBEES' "
    "ORDER BY date", con)
nb = nb[nb['date'].isin(set(cal))].set_index('date').reindex(cal).ffill()
bh = (nb['close'] / nb['close'].iloc[0] * B.START_CAPITAL).to_numpy()
m = B.metrics(bh, cal)
print('\n=== NIFTYBEES BUY & HOLD (the benchmark to beat) ===')
print('   CAGR %.2f%%   MaxDD %.2f%%   Calmar %.3f   Sharpe %.2f   over %.1f years'
      % (m['cagr'], m['maxdd'], m['calmar'], m['sharpe'], m['years']))
for lo, hi, lab in (('2005-01-03', '2015-12-31', 'pre-2016'), ('2016-01-01', '2026-12-31', '2016+')):
    sub = [i for i, d in enumerate(cal) if lo <= d <= hi]
    mm = B.metrics(bh[sub[0]:sub[-1] + 1], cal[sub[0]:sub[-1] + 1])
    print('   %-9s CAGR %.2f%%  MaxDD %.2f%%' % (lab, mm['cagr'], mm['maxdd']))

# ---------------- engine smoke on the base v3 event set --------------------------
ev_df = pd.read_csv(RES / 'rounding_base_events_v3.csv')
ev_df = ev_df[ev_df['symbol'] != 'SILLYMONKS']
ev_df = ev_df.sort_values('pattern_quality', ascending=False).drop_duplicates(['symbol', 'trigger_date'])
syms = sorted(ev_df['symbol'].unique())
print('\nbuilding panel for %d symbols...' % len(syms))
t0 = time.time()
panel = B.Panel(con, syms, cal)
con.close()
print('panel built in %.0fs (%d symbols usable)' % (time.time() - t0, len(panel.close)))

events = []
for r in ev_df.itertuples():
    if not panel.has(r.symbol) or not isinstance(r.entry_date, str):
        continue
    i = panel.pos.get(r.entry_date)
    if i is not None:
        events.append(dict(symbol=r.symbol, entry_i=i))
print('tradeable events on the calendar: %d of %d' % (len(events), len(ev_df)))

print('\n=== ENGINE SMOKE: base v3 entries (S=15 K=3 ATH>=0.90), 25 bps, 3 seeds ===')
print('%-8s %-6s %-6s %8s %8s %8s %7s %7s %6s %7s'
      % ('exit', 'stop8', 'tstop', 'CAGR', 'MaxDD', 'Calmar', 'trades', 'win%', 'exp%', 'sec'))
for ex in B.EXITS:
    for hard in (False, True):
        t1 = time.time()
        res = []
        for sd in (1, 2, 3):
            nav, tr = B.simulate(events, panel, dict(exit=ex, hard_stop=hard, time_stop=0,
                                                     cost_bps=25.0, gate_ok=None), sd)
            res.append(B.metrics(nav, cal, tr))
        med = lambda k: float(np.median([r[k] for r in res if k in r]))
        print('%-8s %-6s %-6s %7.2f%% %7.2f%% %8.3f %7d %6.1f%% %6.2f %6.1f'
              % (ex, hard, 0, med('cagr'), med('maxdd'), med('calmar'),
                 med('trades'), med('win_rate'), med('expectancy'), time.time() - t1))
