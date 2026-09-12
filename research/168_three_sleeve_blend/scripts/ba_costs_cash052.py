# -*- coding: utf-8 -*-
"""research/168 step 2b - Open Alpha . Base Age at 5.2% idle cash, 40 and 60 bps a side.

The 25 bps ensemble already exists, all 30 seeds, at 5.2%:
    research/163 .../results/cash052/ba_navs_30seed_cash052.npz
This script adds ONLY the two extra cost levels the house cost ladder needs, using research/163's
own engine copy (`ba_cash05.simulate_inv`, which is research/161's `bt_core.simulate` verbatim
plus a daily invested-fraction array). Nothing in research/161 or research/163 is written to.

STEP 1 IS A BIT-EXACT REPRODUCTION GATE: the 25 bps re-run must reproduce all 30 paths of
research/163's npz exactly.
"""
from __future__ import annotations

import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/163_mpf_cash_yield_harmonisation/scripts'))
import ba_cash05 as BA5                        # noqa: E402
import bt_core as B                            # noqa: E402

R161 = ROOT / 'research/161_ath_base_age_breakout/results'
R163 = ROOT / 'research/163_mpf_cash_yield_harmonisation/results'
OUT = ROOT / 'research/168_three_sleeve_blend/results'
OUT.mkdir(parents=True, exist_ok=True)
DB = ROOT / 'backtest_data/market_data.db'
PANEL = R163 / 'panel163.pkl'

SEEDS = list(range(1, 31))
CFG = dict(X=60, dep=20.0, K=0.0, sau=0, ex='ST_14_4', hard=False, liq=2.0)
Y = 0.052
COSTS = (25.0, 40.0, 60.0)


def main():
    t0 = time.time()
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    cal = [d for d in pd.read_sql_query(
        "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
        "AND symbol='NIFTYBEES' ORDER BY date", con)['date'].tolist() if d >= '2005-01-03']
    zz = np.load(R161 / 'curves161.npz', allow_pickle=True)
    CUT = str(zz['dates'][-1])[:10]
    cal = [d for d in cal if d <= CUT]
    print('calendar cut at research/161 last date %s -> %d days (%s .. %s)'
          % (CUT, len(cal), cal[0], cal[-1]), flush=True)
    raw = pd.read_csv(R161 / 'ath_events.csv')
    raw_liq = raw[raw['tv20_cr'] >= CFG['liq']]
    if PANEL.exists():
        panel = pickle.load(open(PANEL, 'rb'))
        print('panel cache: %d symbols' % len(panel.close), flush=True)
    else:
        panel = B.Panel(con, sorted(raw_liq['symbol'].unique()), cal)
        print('panel built in %.0fs (NOT cached - research/163 is read-only here)'
              % (time.time() - t0), flush=True)
    con.close()

    ev_df = raw[raw['symbol'].isin(panel.close) & raw['entry_date'].isin(panel.pos)].copy()
    ev_df['entry_i'] = ev_df['entry_date'].map(panel.pos)
    s = ev_df[(ev_df['tv20_cr'] >= CFG['liq']) & (ev_df['x_bars'] >= CFG['X'])
              & (ev_df['depth_pct'] >= CFG['dep'])]
    s = BA5.rearm(s)
    events = [dict(symbol=a, entry_i=int(b)) for a, b in zip(s['symbol'], s['entry_i'])]
    print('WINNER events: %d' % len(events), flush=True)

    store, rows = {}, []
    for cb in COSTS:
        cfg = dict(exit=CFG['ex'], hard_stop=CFG['hard'], time_stop=0, cost_bps=cb,
                   gate_ok=None, idle_yield=Y)
        navs, ms = [], []
        for sd in SEEDS:
            nav, tr, inv = BA5.simulate_inv(events, panel, cfg, sd)
            m = B.metrics(nav, cal, tr)
            navs.append(np.asarray(nav, float))
            ms.append(m)
            rows.append(dict(cost_bps=cb, seed=sd, cagr=m['cagr'], maxdd=m['maxdd'],
                             calmar=m['calmar'],
                             invested_pct=round(100 * float(np.nanmean(inv)), 2)))
        M = np.vstack(navs)
        M = M / M[:, :1]
        store['ba_%dbps' % int(cb)] = M
        cg = np.array([m['cagr'] for m in ms]); dd = np.array([m['maxdd'] for m in ms])
        print('%3d bps/side  CAGR median %6.2f%% [%.2f .. %.2f]  MaxDD median %7.2f%% '
              '(worst %7.2f%%)  Calmar %5.3f'
              % (cb, np.median(cg), cg.min(), cg.max(), np.median(dd), dd.min(),
                 np.median(cg) / abs(np.median(dd))), flush=True)

    print('\n--- REPRODUCTION GATE: 25 bps, 5.2%%, all 30 paths vs research/163 npz ---',
          flush=True)
    zold = np.load(R163 / 'cash052/ba_navs_30seed_cash052.npz', allow_pickle=True)
    ref = zold['navs']
    ref = ref / ref[:, :1]
    mine = store['ba_25bps']
    if mine.shape != ref.shape:
        print('!! shape mismatch %s vs %s - stopping' % (mine.shape, ref.shape)); sys.exit(2)
    mx = float(np.abs(mine - ref).max())
    print('max abs diff over 30 x %d normalised path values: %.3e' % (mine.shape[1], mx))
    if mx > 1e-10:
        print('!! REPRODUCTION FAILED - stopping, no npz written.'); sys.exit(2)
    print('REPRODUCTION BIT-EXACT.', flush=True)

    np.savez_compressed(OUT / 'ba_navs_cash052.npz',
                        dates=np.array([str(d)[:10] for d in cal]),
                        seeds=np.array(SEEDS), **store)
    pd.DataFrame(rows).to_csv(OUT / 'ba_costs_summary.csv', index=False)
    print('\nwrote %s  (%.0fs)' % (OUT / 'ba_navs_cash052.npz', time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
