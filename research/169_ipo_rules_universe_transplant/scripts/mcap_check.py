# -*- coding: utf-8 -*-
"""research/169 S2b - second, shorter check: size universes by TRUE point-in-time market cap.

The headline universes rank by trailing traded value (a causal proxy). Its overlap with the
mcap_pit ranks is ~80% at top-200/500 but only ~60% at top-50 and ~45% for the mid/small bands.
This re-runs the transplant (Spec A, age band removed) on universes ranked by the PREVIOUS
month's features_pit_monthly.mcap_pit row (no same-day price), window 2018-09-01 -> 2026-09-04,
real + date-matched null, 30 paired seeds, alongside the traded-value proxy on the same window
and Spec A itself on the same window.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/169_ipo_rules_universe_transplant/scripts'))
import xpanel as xp   # noqa: E402
import run169 as r    # noqa: E402

path = xp.RES / 's2b_mcap_universes_2018.csv'
done = r.done_labels(path)
P = xp.default_panel() if hasattr(xp, 'default_panel') else r.default_panel()
ds = np.array([str(d.date()) for d in P.dates])
w18 = np.nonzero((ds >= '2018-09-01') & (ds <= '2026-09-04'))[0]
P.days = {'w2': w18}

con = sqlite3.connect(str(ROOT / 'backtest_data/fundamentals.db'))
f = pd.read_sql_query('select date, symbol, mcap_pit from features_pit_monthly '
                      'where mcap_pit is not null', con)
con.close()
f['m'] = pd.to_datetime(f.date).dt.to_period('M')
colidx = {c: j for j, c in enumerate(P.cols)}
f = f[f.symbol.isin(colidx)]
by_m = {m: g for m, g in f.groupby('m')}
T, N = P.C.shape
MR = np.full((T, N), np.inf, dtype=np.float32)
mon = P.dates.to_period('M')
starts = np.nonzero(np.r_[True, mon[1:] != mon[:-1]])[0]
ends = list(starts[1:]) + [T]
for a, b in zip(starts, ends):
    g = by_m.get(mon[a] - 1)                       # previous month's PIT row
    if g is None:
        continue
    g = g[~g.symbol.map(lambda s: bool(P.FUND[colidx[s]]))]
    g = g.sort_values('mcap_pit', ascending=False)
    row = np.full(N, np.inf)
    row[[colidx[s] for s in g.symbol]] = np.arange(1, len(g) + 1)
    MR[a:b] = row
TVRANK = P.RANK
cells = [('specA_all__le6__2018', 'tv', 'all', 'le6')]
for u in ('top50', 'top100', 'top200', 'top500', 'mid101_250', 'small251_500'):
    cells.append((f'mcap_{u}__none__2018', 'mcap', u, 'none'))
    cells.append((f'tv_{u}__none__2018', 'tv', u, 'none'))
for label, kind, u, a in cells:
    if label in done:
        continue
    P.RANK = MR if kind == 'mcap' else TVRANK
    row = r.cell(P, label, u, a, windows=('w2',))
    row['rank_basis'] = kind
    row['window'] = '2018-09-01..2026-09-04'
    r.append(path, row)
P.RANK = TVRANK
cols = ['label', 'n_signals_w2', 'w2_cagr', 'w2_dd', 'w2_calmar', 'w2_null_cagr', 'w2_edge',
        'w2_wins']
print(pd.read_csv(path)[cols].to_string(index=False), flush=True)
