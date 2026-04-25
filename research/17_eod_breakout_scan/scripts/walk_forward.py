"""Walk-forward validation — top 3 variants from Phase 1.

Train: 2018-01-01 → 2022-12-31 (5 years, in-sample)
Test:  2023-01-01 → 2025-12-31 (3 years, out-of-sample)

Pass criteria for "real edge":
  OOS Sharpe >= 0.8
  OOS PF >= 1.2
  OOS MaxDD <= 25%

Reuses run_eod_breakout's engine + variant configs (imports it).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eod_breakout import (
    load_universe, load_all_bars, run_variant, compute_metrics, make_variant,
    OUT, START_DATE, END_DATE,
)

TRAIN_START = '2018-01-01'
TRAIN_END   = '2022-12-31'
TEST_START  = '2023-01-01'
TEST_END    = '2025-12-31'

# Top 3 from Phase 1 (by Sharpe)
TOP_VARIANTS = [
    make_variant('D1_fixed_25_8', exit_kind='fixed', exit_target_pct=0.25, exit_stop_pct=0.08),
    make_variant('D4_donch_20', exit_kind='donch', exit_donch_n=20),
    make_variant('baseline_50_2x_200_d10'),  # all defaults
]


def main():
    t_start = time.time()
    print('Loading universe + bars...')
    uni = load_universe()
    bars = load_all_bars(uni)
    print(f'Loaded {len(bars)} stocks. Period: {START_DATE} -> {END_DATE}')
    print(f'Train: {TRAIN_START} -> {TRAIN_END}')
    print(f'Test:  {TEST_START} -> {TEST_END}')
    print()

    rows = []
    keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
            'avg_days_held','profit_factor','net_pnl','cagr_pct','sharpe',
            'max_dd_pct','calmar','final_equity']

    print(f'{"Variant":>25} {"Phase":>5} {"Trades":>7} {"WR%":>6} {"PF":>6} {"CAGR%":>7} {"Sharpe":>7} {"MaxDD%":>7} {"Calmar":>7}')
    print('-' * 95)

    for cfg in TOP_VARIANTS:
        # IS run
        is_trades, is_eq = run_variant(bars, cfg, TRAIN_START, TRAIN_END)
        is_m = compute_metrics(is_trades, is_eq)
        rows.append({'variant': cfg['name'], 'phase': 'IS', **is_m})
        print(f'{cfg["name"]:>25} {"IS":>5} {is_m["trades"]:>7} {is_m["win_rate_pct"]:>6.1f} '
              f'{is_m["profit_factor"]:>6.2f} {is_m["cagr_pct"]:>+6.2f} {is_m["sharpe"]:>7.2f} '
              f'{is_m["max_dd_pct"]:>7.2f} {is_m["calmar"]:>7.2f}', flush=True)

        # OOS run
        oos_trades, oos_eq = run_variant(bars, cfg, TEST_START, TEST_END)
        oos_m = compute_metrics(oos_trades, oos_eq)
        rows.append({'variant': cfg['name'], 'phase': 'OOS', **oos_m})
        print(f'{cfg["name"]:>25} {"OOS":>5} {oos_m["trades"]:>7} {oos_m["win_rate_pct"]:>6.1f} '
              f'{oos_m["profit_factor"]:>6.2f} {oos_m["cagr_pct"]:>+6.2f} {oos_m["sharpe"]:>7.2f} '
              f'{oos_m["max_dd_pct"]:>7.2f} {oos_m["calmar"]:>7.2f}', flush=True)

        # Pass/fail
        pass_sharpe = oos_m['sharpe'] >= 0.8
        pass_pf     = oos_m['profit_factor'] >= 1.2
        pass_dd     = oos_m['max_dd_pct'] <= 25
        verdict = 'PASS' if (pass_sharpe and pass_pf and pass_dd) else 'FAIL'
        gates = []
        gates.append(f'Sharpe={oos_m["sharpe"]:.2f}{" PASS" if pass_sharpe else " fail"}')
        gates.append(f'PF={oos_m["profit_factor"]:.2f}{" PASS" if pass_pf else " fail"}')
        gates.append(f'MaxDD={oos_m["max_dd_pct"]:.1f}%{" PASS" if pass_dd else " fail"}')
        print(f'  -> OOS verdict: {verdict}  ({" ".join(gates)})')
        print()

    # Save
    keys_full = ['variant','phase'] + keys
    with (OUT / 'walk_forward_summary.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(keys_full)
        for r in rows:
            w.writerow([r['variant'], r['phase']] + [r.get(k, '') for k in keys])

    print(f'Total runtime: {time.time()-t_start:.1f}s')


if __name__ == '__main__':
    sys.exit(main())
