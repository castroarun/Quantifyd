"""Walk-forward validation — best variant from the 6-variant sweep.

Train: 2018-01-01 → 2022-12-31 (5 years, in-sample)
Test:  2023-01-01 → 2025-12-31 (3 years, out-of-sample)

Pass criteria for "real edge":
  OOS Profit Factor >= 1.20
  OOS Sharpe >= 0.8
  OOS MaxDD <= 30%

Reuses run_smallcap_daily_backtest's engine (imports it).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_smallcap_daily_backtest import (   # type: ignore
    load_universe, load_all_bars, run_variant, compute_metrics, make_variant,
    RES, START_DATE, END_DATE,
)

TRAIN_START = '2018-01-01'
TRAIN_END   = '2022-12-31'
TEST_START  = '2023-01-01'
TEST_END    = '2025-12-31'

# Best variant from Phase 1 (best Sharpe with PF>=1.0): vol_3x
# (target_30pct is a virtual tie; we report the chosen winner)
TOP_VARIANTS = [
    make_variant('vol_3x', vol_mult=3.0),
    make_variant('target_30pct', target_pct=0.30),
    make_variant('baseline_252_25pct_8pct'),
]


def main():
    t_start = time.time()
    print('Loading universe + bars...', flush=True)
    uni = load_universe()
    bars = load_all_bars(uni)
    print(f'Loaded {len(bars)} stocks. Period: {START_DATE} -> {END_DATE}', flush=True)
    print(f'Train: {TRAIN_START} -> {TRAIN_END}', flush=True)
    print(f'Test:  {TEST_START} -> {TEST_END}', flush=True)
    print()

    rows = []
    keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
            'avg_days_held','profit_factor','net_pnl','cagr_pct','sharpe',
            'max_dd_pct','calmar','final_equity']

    print(f'{"Variant":>26} {"Phase":>5} {"Trades":>7} {"WR%":>6} {"PF":>6} '
          f'{"CAGR%":>7} {"Sharpe":>7} {"MaxDD%":>7} {"Calmar":>7}', flush=True)
    print('-' * 95, flush=True)

    for cfg in TOP_VARIANTS:
        # IS run
        is_trades, is_eq = run_variant(bars, cfg, TRAIN_START, TRAIN_END)
        is_m = compute_metrics(is_trades, is_eq)
        rows.append({'variant': cfg['name'], 'phase': 'IS', **is_m})
        print(f'{cfg["name"]:>26} {"IS":>5} {is_m["trades"]:>7} {is_m["win_rate_pct"]:>6.1f} '
              f'{is_m["profit_factor"]:>6.2f} {is_m["cagr_pct"]:>+6.2f} {is_m["sharpe"]:>7.2f} '
              f'{is_m["max_dd_pct"]:>7.2f} {is_m["calmar"]:>7.2f}', flush=True)

        # OOS run
        oos_trades, oos_eq = run_variant(bars, cfg, TEST_START, TEST_END)
        oos_m = compute_metrics(oos_trades, oos_eq)
        rows.append({'variant': cfg['name'], 'phase': 'OOS', **oos_m})
        print(f'{cfg["name"]:>26} {"OOS":>5} {oos_m["trades"]:>7} {oos_m["win_rate_pct"]:>6.1f} '
              f'{oos_m["profit_factor"]:>6.2f} {oos_m["cagr_pct"]:>+6.2f} {oos_m["sharpe"]:>7.2f} '
              f'{oos_m["max_dd_pct"]:>7.2f} {oos_m["calmar"]:>7.2f}', flush=True)

        # Pass/fail vs spec gates
        pass_pf     = oos_m['profit_factor'] >= 1.20
        pass_sharpe = oos_m['sharpe'] >= 0.8
        pass_dd     = oos_m['max_dd_pct'] <= 30
        verdict = 'PASS' if (pass_pf and pass_sharpe and pass_dd) else 'FAIL'
        gates = []
        gates.append(f'PF={oos_m["profit_factor"]:.2f}{" OK" if pass_pf else " FAIL"}')
        gates.append(f'Sharpe={oos_m["sharpe"]:.2f}{" OK" if pass_sharpe else " FAIL"}')
        gates.append(f'MaxDD={oos_m["max_dd_pct"]:.1f}%{" OK" if pass_dd else " FAIL"}')
        print(f'  -> OOS verdict: {verdict}  ({" | ".join(gates)})')
        print()

    keys_full = ['variant','phase'] + keys
    out = RES / 'daily_walk_forward.csv'
    with out.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(keys_full)
        for r in rows:
            w.writerow([r['variant'], r['phase']] + [r.get(k, '') for k in keys])

    print(f'Saved: {out}', flush=True)
    print(f'Total runtime: {time.time()-t_start:.1f}s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
