"""
research/159 — the pre-registered sweep (STATUS 11.3).

Grid: exit(7) x hard-stop(2) x time-stop(2) x shelf S(2) x volume K(3)
      x ATH proximity(3) x OBV filter(2) x market gate(2) = 2,016 cells.
Each cell on a 10-seed scan; survivors are re-run on 30 seeds elsewhere.

Ranking metric (pre-registered): after-tax net CAGR, Calmar as tie-break.
Writes one row per completed cell, immediately, and skips cells already present.
"""
import itertools
import os
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
PANEL_PKL = RES / 'panel.pkl'
OUT = RES / 'sweep_v3.csv'
SHARD_I, SHARD_N = 0, 1          # set by --shard=i/n

SHELF_VALS = [15, 20]
K_VALS = [2, 3, 5]
ATH_VALS = ['0.90', '0.95', '1.00']
EXIT_VALS = B.EXITS
HARD_VALS = [False, True]
TIME_VALS = [0, 120]
OBV_VALS = [False, True]
GATE_VALS = [False, True]
SCAN_SEEDS = list(range(1, 11))
COST_BPS = 25.0

FIELDS = ['cell', 'exit', 'hard_stop', 'time_stop', 'shelf_S', 'vol_K', 'ath',
          'obv_filter', 'market_gate', 'n_events', 'seeds',
          'cagr_med', 'cagr_min', 'cagr_max', 'maxdd_med', 'maxdd_worst',
          'calmar_med', 'sharpe_med', 'trades', 'win_rate', 'avg_win', 'avg_loss',
          'expectancy', 'max_loss_streak', 'trades_per_yr', 'final_med']


def variant_path(S, K, A):
    return RES / ('rounding_base_events_v3_s%d_k%d_a%s.csv' % (S, K, A))


def build_calendar(con):
    d = pd.read_sql_query(
        "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
        "AND symbol='NIFTYBEES' ORDER BY date", con)
    return [x for x in d['date'].tolist() if x >= '2005-01-03']


def load_events(S, K, A, panel, obv_filter):
    p = variant_path(S, K, A)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df['symbol'] != 'SILLYMONKS']                     # duplicate of CRESTO
    df = df.sort_values('pattern_quality', ascending=False)
    df = df.drop_duplicates(['symbol', 'trigger_date'])        # one per symbol+trigger
    if obv_filter:
        df = df[df['obv_filter_pass'] == 1]
    df = df[df['entry_date'].notna() & (df['entry_date'] != '')]
    ev = []
    for r in df.itertuples():
        if not panel.has(r.symbol):
            continue
        i = panel.pos.get(r.entry_date)
        if i is None:
            continue
        ev.append(dict(symbol=r.symbol, entry_i=i))
    return ev


def main():
    global OUT, SHARD_I, SHARD_N
    for arg in sys.argv[1:]:
        if arg.startswith('--shard='):
            a, b = arg.split('=', 1)[1].split('/')
            SHARD_I, SHARD_N = int(a), int(b)
            OUT = RES / ('sweep_v3_shard%d.csv' % SHARD_I)
    t0 = time.time()
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    cal = build_calendar(con)
    print('calendar: %d trading days %s -> %s' % (len(cal), cal[0], cal[-1]), flush=True)

    # ---- symbols needed across every variant -------------------------------------
    need = set()
    have_variants = []
    for S, K, A in itertools.product(SHELF_VALS, K_VALS, ATH_VALS):
        p = variant_path(S, K, A)
        if p.exists():
            have_variants.append((S, K, A))
            need |= set(pd.read_csv(p, usecols=['symbol'])['symbol'].unique())
    need.discard('SILLYMONKS')
    print('variants present: %d/18   symbols needed: %d' % (len(have_variants), len(need)), flush=True)
    if len(have_variants) < 18:
        print('WARNING: not all 18 entry variants exist yet', flush=True)

    # ---- panel (cached) ----------------------------------------------------------
    if PANEL_PKL.exists():
        with open(PANEL_PKL, 'rb') as f:
            panel = pickle.load(f)
        missing = need - set(panel.close)
        print('panel loaded from cache (%d symbols, %d missing)' % (len(panel.close), len(missing)), flush=True)
    else:
        missing = need
        panel = None
    if panel is None or missing:
        print('building panel for %d symbols...' % len(need), flush=True)
        panel = B.Panel(con, sorted(need), cal)
        with open(PANEL_PKL, 'wb') as f:
            pickle.dump(panel, f, protocol=4)
        print('panel built: %d symbols in %.0fs' % (len(panel.close), time.time() - t0), flush=True)

    # ---- market gate: NIFTYBEES close > its own 100-day SMA -----------------------
    nb = pd.read_sql_query(
        "SELECT date,close FROM market_data_unified WHERE timeframe='day' AND symbol='NIFTYBEES' "
        "ORDER BY date", con)
    nb = nb[nb['date'].isin(set(cal))].set_index('date')['close'].reindex(cal).ffill()
    gate = (nb > nb.rolling(100, min_periods=100).mean()).to_numpy()
    gate[:100] = True
    con.close()

    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT)['cell'].astype(str))
        print('resuming: %d cells already done' % len(done), flush=True)
    else:
        pd.DataFrame(columns=FIELDS).to_csv(OUT, index=False)

    cells = list(itertools.product(EXIT_VALS, HARD_VALS, TIME_VALS, SHELF_VALS,
                                   K_VALS, ATH_VALS, OBV_VALS, GATE_VALS))
    total = len(cells)
    if SHARD_N > 1:
        cells = [c for j, c in enumerate(cells) if j % SHARD_N == SHARD_I]
    print('total cells: %d  (shard %d/%d -> %d cells, x %d seeds)'
          % (total, SHARD_I, SHARD_N, len(cells), len(SCAN_SEEDS)), flush=True)

    ev_cache = {}
    n_done = 0
    for (ex, hard, tstop, S, K, A, obv, gt) in cells:
        cid = '%s|h%d|t%d|s%d|k%d|a%s|o%d|g%d' % (ex, hard, tstop, S, K, A, obv, gt)
        if cid in done:
            continue
        if (S, K, A) not in have_variants:
            continue
        ck = (S, K, A, obv)
        if ck not in ev_cache:
            ev_cache[ck] = load_events(S, K, A, panel, obv)
        events = ev_cache[ck]
        if not events:
            continue
        cfg = dict(exit=ex, hard_stop=hard, time_stop=tstop, cost_bps=COST_BPS,
                   gate_ok=(gate if gt else None))
        navs, allm = [], []
        for sd in SCAN_SEEDS:
            nav, tr = B.simulate(events, panel, cfg, sd)
            m = B.metrics(nav, cal, tr)
            if m:
                navs.append(m); allm.append(m)
        if not allm:
            continue
        cg = sorted(m['cagr'] for m in allm)
        dd = sorted(m['maxdd'] for m in allm)
        row = dict(cell=cid, exit=ex, hard_stop=int(hard), time_stop=tstop, shelf_S=S,
                   vol_K=K, ath=A, obv_filter=int(obv), market_gate=int(gt),
                   n_events=len(events), seeds=len(allm),
                   cagr_med=round(float(np.median(cg)), 2), cagr_min=cg[0], cagr_max=cg[-1],
                   maxdd_med=round(float(np.median(dd)), 2), maxdd_worst=dd[0],
                   calmar_med=round(float(np.median([m['calmar'] for m in allm])), 3),
                   sharpe_med=round(float(np.median([m['sharpe'] for m in allm])), 3),
                   trades=int(np.median([m.get('trades', 0) for m in allm])),
                   win_rate=round(float(np.median([m.get('win_rate', np.nan) for m in allm])), 1),
                   avg_win=round(float(np.median([m.get('avg_win', np.nan) for m in allm])), 2),
                   avg_loss=round(float(np.median([m.get('avg_loss', np.nan) for m in allm])), 2),
                   expectancy=round(float(np.median([m.get('expectancy', np.nan) for m in allm])), 3),
                   max_loss_streak=int(np.median([m.get('max_loss_streak', 0) for m in allm])),
                   trades_per_yr=round(float(np.median([m.get('trades_per_yr', np.nan) for m in allm])), 1),
                   final_med=round(float(np.median([m['final'] for m in allm])), 0))
        pd.DataFrame([row])[FIELDS].to_csv(OUT, mode='a', header=False, index=False)
        n_done += 1
        if n_done % 25 == 0:
            el = time.time() - t0
            print('[%d cells] %.0fs  last: %s cagr_med=%.1f%% dd=%.1f%% calmar=%.2f'
                  % (n_done, el, cid, row['cagr_med'], row['maxdd_med'], row['calmar_med']),
                  flush=True)
    print('SWEEP DONE: %d new cells in %.0fs' % (n_done, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
