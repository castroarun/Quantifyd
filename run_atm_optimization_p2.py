"""
NAS ATM Straddle Optimization - Phase 2
========================================
Corrected IV=0.20 (matches real market premiums ~30% higher than IV=0.15).
Tests later entry times, squeeze occurrence filter, and low re-entries.

Grid: 2 symbols x 4 SL% x 2 RE x 3 entry_times x 2 squeeze_occ = 96 configs
"""

import csv
import os
import sys
import io
import time
import logging
from itertools import product

logging.disable(logging.WARNING)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from services.nas_atm_backtest_engine import run_backtest

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_atm_p2.csv')

FIELDNAMES = [
    'label', 'symbol', 'sl_pct', 'max_reentries', 'min_squeeze_bars',
    'entry_start', 'squeeze_occurrence', 'iv',
    'total_pnl', 'avg_daily_pnl', 'win_rate', 'profit_factor',
    'sharpe', 'sortino', 'max_drawdown', 'best_day', 'worst_day',
    'entry_days', 'squeeze_days', 'total_strangles', 'total_sl_hits',
    'total_reentries', 'avg_reentries_per_day', 'avg_premium',
]

# Skip already completed
done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {row['label'] for row in csv.DictReader(f)}
    print(f'Skipping {len(done)} already-completed configs')
else:
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

# Phase 2 grid - corrected IV=0.20
symbols = ['NIFTY50', 'BANKNIFTY']
sl_pcts = [0.15, 0.20, 0.25, 0.30]
max_reentries_list = [0, 1]
entry_starts = ['10:00', '11:00', '12:00']
squeeze_occs = [1, 2]
iv = 0.20

configs = []
for sym, sl, mre, est, socc in product(symbols, sl_pcts, max_reentries_list, entry_starts, squeeze_occs):
    label = f'{sym}_SL{int(sl*100)}_RE{mre}_E{est.replace(":", "")}_OCC{socc}_IV20'
    if label not in done:
        configs.append((label, sym, sl, mre, est, socc))

total = len(configs)
print(f'Total configs to run: {total}')
print(f'Estimated time: ~{total * 15 / 60:.0f} minutes')
print()

start_time = time.time()

for i, (label, sym, sl, mre, est, socc) in enumerate(configs, 1):
    t0 = time.time()
    print(f'[{i}/{total}] {label} ...', end='', flush=True)

    try:
        r = run_backtest(
            sym,
            start_date='2024-04-01',
            end_date='2026-03-25',
            leg_sl_pct=sl,
            max_reentries=mre,
            min_squeeze_bars=1,
            entry_start_time=est,
            squeeze_occurrence=socc,
            default_iv=iv,
        )

        row = {
            'label': label,
            'symbol': sym,
            'sl_pct': sl,
            'max_reentries': mre,
            'min_squeeze_bars': 1,
            'entry_start': est,
            'squeeze_occurrence': socc,
            'iv': iv,
            'total_pnl': round(r.total_pnl, 0),
            'avg_daily_pnl': round(r.avg_daily_pnl, 0),
            'win_rate': round(r.win_rate, 1),
            'profit_factor': round(r.profit_factor, 2),
            'sharpe': round(r.sharpe_ratio, 2),
            'sortino': round(r.sortino_ratio, 2),
            'max_drawdown': round(r.max_drawdown, 0),
            'best_day': round(r.best_day, 0),
            'worst_day': round(r.worst_day, 0),
            'entry_days': r.entry_days,
            'squeeze_days': r.squeeze_days,
            'total_strangles': r.total_strangles,
            'total_sl_hits': r.total_sl_hits,
            'total_reentries': r.total_reentries,
            'avg_reentries_per_day': round(r.avg_reentries_per_day, 1),
            'avg_premium': round(r.avg_premium_collected, 1),
        }

        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        elapsed = time.time() - t0
        print(f' {elapsed:.0f}s | PnL={r.total_pnl:>10,.0f} WR={r.win_rate:.0f}% PF={r.profit_factor:.2f} Sharpe={r.sharpe_ratio:.2f}')
        sys.stdout.flush()

    except Exception as e:
        import traceback
        print(f' ERROR: {e}')
        traceback.print_exc()
        sys.stdout.flush()

total_time = time.time() - start_time
print(f'\nDone! {total} configs in {total_time/60:.1f} minutes')
print(f'Results saved to: {OUTPUT_CSV}')
