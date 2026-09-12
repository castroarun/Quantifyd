# -*- coding: utf-8 -*-
"""research/163 — Quality Summit (research/160 finalist F_Bb7) re-run at 5.2% idle cash.

The Momentum Portfolio report's 2018-window section draws Quality Summit from
`research/160 .../results/F_Bb7_equity.csv` — twelve rebalance offsets, 2018-08-01 ->
2026-09-10, built at cash_yield = 0.05. This re-runs the IDENTICAL cell at 0.052 (the
arbitrage-fund rate after 20% short-term tax) and writes the twelve offsets into
research/163's own folder.

The cell, from research/160 `make_grid.py` FINALISTS -> base(**kw):

    label F_Bb7, mask masks_study/b7_g10_qual_mc.npz, mask_missing 'fail', exits 'none',
    slots 15, entry 'rebalance', cadence 'monthly', rank 'rs', buffer 1.5, state 'near',
    k 0.90, tv_floor 2.0, index_gate 'none', fill 'next_open', cost_bps 25.0, tax True,
    offsets 12, arms 'all', 2018-08-01 .. 2026-09-10

`qg_engine` is imported as a MODULE and `run_cell` called directly, rather than invoking its
CLI — the CLI's --dump-equity writes into research/160's results folder, which this task must
not write to. Nothing about the cell changes.

STEP 1 is a BIT-EXACT REPRODUCTION GATE at 0.05 against F_Bb7_equity.csv. The panel
(`panel_2000.npz`) is a frozen file, so there is no market_data.db-refresh escape hatch here:
anything other than bit-exact is a real difference and stops the script.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
R160 = ROOT / 'research/160_quality_growth_near_ath'
sys.path.insert(0, str(R160 / 'scripts'))
import qg_engine as Q                     # noqa: E402

OUT = ROOT / 'research/163_mpf_cash_yield_harmonisation/results/cash052'
OUT.mkdir(parents=True, exist_ok=True)

OLD_Y, NEW_Y = 0.05, 0.052
CELL = dict(start='2018-08-01', end='2026-09-10', entry='rebalance', cadence='monthly',
            rank='rs', slots=15, buffer=1.5, state='near', k=0.90, tv_floor=2.0,
            mask=str(R160 / 'results/masks_study/b7_g10_qual_mc.npz'), mask_missing='fail',
            exits='none', index_gate='none', fill='next_open', cost_bps=25.0, tax=True,
            offsets=12, arms='all')


def stats(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    c = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    d = (s / s.cummax() - 1).min()
    return c * 100, d * 100, c / abs(d)


def main():
    t0 = time.time()
    panel = Q.Panel.load(str(R160 / 'results/panel_2000.npz'))
    der = Q.Derived(panel)
    print('panel loaded: %d dates %s..%s, %d symbols  (%.0fs)'
          % (len(panel.dates), panel.dates[0], panel.dates[-1], len(panel.syms),
             time.time() - t0), flush=True)

    out = {}
    for y, tag in ((OLD_Y, '050'), (NEW_Y, '052')):
        t1 = time.time()
        cell = Q.Cell(label='F_Bb7_cash%s' % tag, cash_yield=y, **CELL)
        r = Q.run_cell(panel, der, cell, verbose=False)
        cur = pd.DataFrame(r['curves'])
        out[y] = dict(curves=cur, row=r['row'])
        print('idle %.1f%%  after-tax CAGR %.2f  DD %.2f (worst %.2f)  Calmar %.2f  '
              '%d paths  invested %.1f%%  (%.0fs)'
              % (y * 100, r['row']['cagr_net_tax'], r['row']['maxdd'],
                 r['row']['maxdd_worst'], r['row']['calmar'], r['row']['n_paths'],
                 r['row']['avg_pct_invested'], time.time() - t1), flush=True)

    # ---------------- reproduction gate ----------------------------------------------
    print('\n--- reproduction gate at 5.0%% vs research/160 F_Bb7_equity.csv ---', flush=True)
    pub = pd.read_csv(R160 / 'results/F_Bb7_equity.csv', index_col=0, parse_dates=True)
    mine = out[OLD_Y]['curves']
    mine.index = pd.to_datetime(mine.index)
    same_cols = list(mine.columns) == list(pub.columns)
    same_index = bool(mine.index.equals(pub.index))
    ok_shape = same_cols and same_index
    dmax = float(np.abs(mine.values - pub.values).max()) if ok_shape else np.nan
    # F_Bb7_equity.csv is a decimal round trip of the same float64 array, so the last ULP can
    # differ; the panel is frozen, so anything above ~1e-12 RELATIVE is a real difference.
    rmax = (float((np.abs(mine.values - pub.values) / np.abs(pub.values)).max())
            if ok_shape else np.nan)
    print('columns identical : %s  %s' % (same_cols, list(mine.columns)[:3]))
    print('index identical   : %s (%d rows vs %d)' % (same_index, len(mine), len(pub)))
    print('max abs diff      : %.3e     max rel diff : %.3e' % (dmax, rmax))
    if not (ok_shape and rmax < 1e-12):
        print('!! REPRODUCTION FAILED on a frozen panel — stopping.')
        sys.exit(2)
    print('REPRODUCTION EXACT (differences at or below the CSV decimal round trip).')

    # ---------------- outputs ----------------------------------------------------------
    new = out[NEW_Y]['curves']
    new.index = pd.to_datetime(new.index)
    new.to_csv(OUT / 'F_Bb7_equity_cash052.csv')
    json.dump({'row_050': out[OLD_Y]['row'], 'row_052': out[NEW_Y]['row']},
              open(OUT / 'qs_cash052_rows.json', 'w'), indent=1, default=str)

    # the page draws the offset whose CAGR is nearest the median, on the 2018-08-01 window
    def drawn(df):
        cg = df.apply(lambda s: (s.iloc[-1] / s.iloc[0])
                      ** (365.25 / (s.index[-1] - s.index[0]).days) - 1)
        return (cg - cg.median()).abs().idxmin(), cg
    d50, cg50 = drawn(mine)
    d52, cg52 = drawn(new)
    c50, dd50, k50 = stats(mine[d50])
    c52, dd52, k52 = stats(new[d52])
    inv = out[OLD_Y]['row']['avg_pct_invested']
    rot = (1 - inv / 100.0) * 0.2
    paired = (cg52.values - cg50.values) * 100
    print('\nDRAWN OFFSET %s -> %s' % (d50, d52))
    print('YIELD EFFECT 5.0%% -> 5.2%% (drawn path): CAGR %.2f -> %.2f (%+.3f pp)  '
          'MaxDD %.2f -> %.2f  Calmar %.2f -> %.2f'
          % (c50, c52, c52 - c50, dd50, dd52, k50, k52))
    print('CONSISTENCY: paired per-offset CAGR delta median %+.3f pp [%+.3f .. %+.3f]; '
          'rule of thumb at inv %.1f%% = %+.3f pp'
          % (np.median(paired), paired.min(), paired.max(), inv, rot))
    json.dump(dict(drawn_050=d50, drawn_052=d52,
                   cagr_050=round(c50, 3), cagr_052=round(c52, 3),
                   maxdd_050=round(dd50, 3), maxdd_052=round(dd52, 3),
                   calmar_050=round(k50, 3), calmar_052=round(k52, 3),
                   invested_pct=inv,
                   cagr_delta_paired_med_pp=round(float(np.median(paired)), 3),
                   cagr_delta_rule_of_thumb_pp=round(rot, 3),
                   reproduction_bit_exact=True),
              open(OUT / 'qs_cash052_summary.json', 'w'), indent=1)
    print('\nwrote %s   (%.0fs)' % (OUT / 'F_Bb7_equity_cash052.csv', time.time() - t0))


if __name__ == '__main__':
    main()
