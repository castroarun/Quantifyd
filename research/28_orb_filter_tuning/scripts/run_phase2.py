"""ORB Filter Tuning — Phase 2: signal_drift sweep.

Run only AFTER phase 1 is complete and a winning RSI/age combo has been
identified. This script reads the Phase 1 winner from results/summary.csv
(passed via --winner-label CLI arg) and runs the 4-cell drift grid.

Append rows to the same summary.csv (incremental write, header preserved).
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.orb_backtest_engine import ORBBacktestEngine
from run_filter_sweep import (
    UNIVERSE, PERIOD_START, PERIOD_END, FIELDNAMES,
    FilterORBConfig, ORBFilterEngine,
    make_base_config, filter_data_by_period,
    run_cell, append_csv_row, load_existing,
    composite_score,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rsi-long', type=float, required=True)
    ap.add_argument('--rsi-short', type=float, required=True)
    ap.add_argument('--age', type=int, required=True)
    args = ap.parse_args()

    OUT = Path(__file__).resolve().parents[1] / 'results'
    out_csv = OUT / 'summary.csv'

    print('=' * 70)
    print('ORB Filter Tuning Sweep — Phase 2 (signal_drift)')
    print('=' * 70)
    print(f'Holding rsi_l={args.rsi_long} rsi_s={args.rsi_short} age={args.age}')

    base = make_base_config()
    winner = replace(base,
                     rsi_long_threshold=args.rsi_long,
                     rsi_short_threshold=args.rsi_short,
                     signal_age_max_mins=args.age)

    cells = []
    for drift in (0.005, 0.0085, 0.012, 0.015):
        label = (f'p2_drift{drift:.4f}_rsi_l{int(args.rsi_long)}'
                 f'_rsi_s{int(args.rsi_short)}_age{args.age}')
        cfg = replace(winner, signal_drift_max_pct=drift)
        cells.append((label, cfg))

    done = load_existing(out_csv)
    print(f'Existing cells in CSV: {len(done)}')

    t_start = time.time()
    print(f'\nPreloading data...')
    raw = ORBBacktestEngine.preload_data(UNIVERSE)
    data_by_sym = {s: filter_data_by_period(raw[s], PERIOD_START, PERIOD_END)
                   for s in UNIVERSE if s in raw}
    print(f'Preload {time.time()-t_start:.1f}s\n')

    rows = []
    for idx, (label, cfg) in enumerate(cells, 1):
        if label in done:
            print(f'[{idx}/{len(cells)}] {label} SKIP (already done)')
            continue
        t0 = time.time()
        print(f'[{idx}/{len(cells)}] {label} (drift={cfg.signal_drift_max_pct}) ...',
              end='', flush=True)
        row = run_cell(label, cfg, data_by_sym)
        print(f' {time.time()-t0:.0f}s | trades={row["trades"]} '
              f'PF={row["profit_factor"]:.2f} DD={row["max_dd_pct"]:.2f} '
              f'NetPnl={row["net_pnl"]:.0f}', flush=True)
        append_csv_row(out_csv, row)
        rows.append(row)

    print(f'\nPhase 2 done. Total runtime: {time.time()-t_start:.1f}s')

    # Quick ranking
    if rows:
        ranked = sorted(rows, key=composite_score, reverse=True)
        print('\nPhase 2 Ranking:')
        for r in ranked:
            print(f'  {str(r["cell_label"]):<45} trades={int(r["trades"])} '
                  f'PF={float(r["profit_factor"]):.2f} '
                  f'DD={float(r["max_dd_pct"]):.2f} '
                  f'NetPnl={float(r["net_pnl"]):.0f}')


if __name__ == '__main__':
    main()
