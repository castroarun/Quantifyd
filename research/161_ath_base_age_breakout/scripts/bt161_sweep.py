"""
research/161 — the pre-registered 864-cell sweep (STATUS section 5).

Grid: X (6) x base depth (3) x volume K (4) x saucer (2) x exit (6) = 864 cells,
10-seed scan each. Ranking metric: after-tax net CAGR, Calmar as tie-break.

The 60-bar re-arm is applied AFTER the filter, because which new-ATH close is "first"
depends on which ones the filter lets through.

Incremental CSV, resumable: a shard skips cells already present in its own file.
"""
import itertools
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
PANEL = RES / 'panel161.pkl'
OUT = RES / 'sweep161.csv'
SHARD_I, SHARD_N = 0, 1

X_VALS = [0, 20, 40, 60, 120, 250]
DEPTH_VALS = [0.0, 10.0, 20.0]
K_VALS = [0.0, 2.0, 3.0, 5.0]
SAUCER_VALS = [0, 1]
EXITS = [('ST_14_4', False), ('ST_14_4', True), ('ST_10_3', False),
         ('SMA15', True), ('DON20', False), ('EMA50', False)]
SEEDS = list(range(1, 11))
COST = 25.0
REARM = 60
LIQ_CR = 2.0

FIELDS = ['cell', 'x_bars', 'depth_min', 'vol_K', 'saucer', 'exit', 'hard_stop',
          'liq_cr', 'n_events', 'seeds', 'cagr_med', 'cagr_min', 'cagr_max',
          'maxdd_med', 'maxdd_worst', 'calmar_med', 'sharpe_med', 'trades',
          'win_rate', 'avg_win', 'avg_loss', 'expectancy', 'max_loss_streak',
          'trades_per_yr', 'final_med']


def rearm(df):
    """Keep the first event per symbol, then only events >= REARM bars later."""
    keep = []
    for _, g in df.sort_values(['symbol', 'hist_bars']).groupby('symbol', sort=False):
        last = -10 ** 9
        for idx, hb in zip(g.index, g['hist_bars'].to_numpy()):
            if hb - last >= REARM:
                keep.append(idx)
                last = hb
    return df.loc[keep]


def main():
    global OUT, SHARD_I, SHARD_N
    liq = LIQ_CR
    tag = ''
    for a in sys.argv[1:]:
        if a.startswith('--shard='):
            i, n = a.split('=', 1)[1].split('/')
            SHARD_I, SHARD_N = int(i), int(n)
        elif a.startswith('--liq='):
            liq = float(a.split('=', 1)[1]); tag = '_liq%g' % liq
    OUT = RES / ('sweep161%s_shard%d.csv' % (tag, SHARD_I))

    t0 = time.time()
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    cal = [d for d in pd.read_sql_query(
        "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
        "AND symbol='NIFTYBEES' ORDER BY date", con)['date'].tolist() if d >= '2005-01-03']
    print('calendar %d days %s -> %s' % (len(cal), cal[0], cal[-1]), flush=True)

    ev = pd.read_csv(RES / 'ath_events.csv')
    ev = ev[ev['tv20_cr'] >= liq]
    ev = ev[ev['entry_date'].notna()]
    print('events after liquidity >= Rs%gcr: %d (%d symbols)'
          % (liq, len(ev), ev['symbol'].nunique()), flush=True)

    if PANEL.exists():
        panel = pickle.load(open(PANEL, 'rb'))
        need = set(ev['symbol']) - set(panel.close)
        print('panel cache: %d symbols, %d missing' % (len(panel.close), len(need)), flush=True)
    else:
        need, panel = set(ev['symbol']), None
    if panel is None or need:
        print('building panel for %d symbols...' % ev['symbol'].nunique(), flush=True)
        panel = B.Panel(con, sorted(ev['symbol'].unique()), cal)
        pickle.dump(panel, open(PANEL, 'wb'), protocol=4)
        print('panel built %d symbols in %.0fs' % (len(panel.close), time.time() - t0), flush=True)
    con.close()

    ev = ev[ev['symbol'].isin(panel.close)]
    ev = ev[ev['entry_date'].isin(panel.pos)]
    ev['entry_i'] = ev['entry_date'].map(panel.pos)
    print('tradeable events on calendar: %d' % len(ev), flush=True)

    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT)['cell'].astype(str))
        print('resuming: %d cells done' % len(done), flush=True)
    else:
        pd.DataFrame(columns=FIELDS).to_csv(OUT, index=False)

    cells = list(itertools.product(X_VALS, DEPTH_VALS, K_VALS, SAUCER_VALS, EXITS))
    total = len(cells)
    if SHARD_N > 1:
        cells = [c for j, c in enumerate(cells) if j % SHARD_N == SHARD_I]
    print('cells %d (shard %d/%d -> %d) x %d seeds'
          % (total, SHARD_I, SHARD_N, len(cells), len(SEEDS)), flush=True)

    fcache = {}
    n_done = 0
    for (X, dep, K, sau, (ex, hard)) in cells:
        cid = 'x%d|d%g|k%g|s%d|%s|h%d|L%g' % (X, dep, K, sau, ex, hard, liq)
        if cid in done:
            continue
        fk = (X, dep, K, sau)
        if fk not in fcache:
            sub = ev
            if X:
                sub = sub[sub['x_bars'] >= X]
            if dep:
                sub = sub[sub['depth_pct'] >= dep]
            if K:
                sub = sub[sub['vol_mult'] >= K]
            if sau:
                sub = sub[sub['saucer_ok'] == 1]
            sub = rearm(sub)
            fcache[fk] = [dict(symbol=s, entry_i=int(i))
                          for s, i in zip(sub['symbol'], sub['entry_i'])]
        events = fcache[fk]
        if len(events) < 20:
            continue
        cfg = dict(exit=ex, hard_stop=hard, time_stop=0, cost_bps=COST, gate_ok=None)
        ms = []
        for sd in SEEDS:
            nav, tr = B.simulate(events, panel, cfg, sd)
            m = B.metrics(nav, cal, tr)
            if m:
                ms.append(m)
        if not ms:
            continue
        cg = sorted(m['cagr'] for m in ms)
        dd = sorted(m['maxdd'] for m in ms)
        row = dict(cell=cid, x_bars=X, depth_min=dep, vol_K=K, saucer=sau, exit=ex,
                   hard_stop=int(hard), liq_cr=liq, n_events=len(events), seeds=len(ms),
                   cagr_med=round(float(np.median(cg)), 2), cagr_min=cg[0], cagr_max=cg[-1],
                   maxdd_med=round(float(np.median(dd)), 2), maxdd_worst=dd[0],
                   calmar_med=round(float(np.median([m['calmar'] for m in ms])), 3),
                   sharpe_med=round(float(np.median([m['sharpe'] for m in ms])), 3),
                   trades=int(np.median([m.get('trades', 0) for m in ms])),
                   win_rate=round(float(np.median([m.get('win_rate', np.nan) for m in ms])), 1),
                   avg_win=round(float(np.median([m.get('avg_win', np.nan) for m in ms])), 2),
                   avg_loss=round(float(np.median([m.get('avg_loss', np.nan) for m in ms])), 2),
                   expectancy=round(float(np.median([m.get('expectancy', np.nan) for m in ms])), 3),
                   max_loss_streak=int(np.median([m.get('max_loss_streak', 0) for m in ms])),
                   trades_per_yr=round(float(np.median([m.get('trades_per_yr', np.nan) for m in ms])), 1),
                   final_med=round(float(np.median([m['final'] for m in ms])), 0))
        pd.DataFrame([row])[FIELDS].to_csv(OUT, mode='a', header=False, index=False)
        n_done += 1
        if n_done % 20 == 0:
            el = time.time() - t0
            print('[%d cells] %.0fs  last %s  n=%d cagr=%.2f%% dd=%.1f%% wr=%.1f%%'
                  % (n_done, el, cid, len(events), row['cagr_med'], row['maxdd_med'],
                     row['win_rate']), flush=True)
    print('SHARD %d DONE: %d cells in %.0fs' % (SHARD_I, n_done, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
