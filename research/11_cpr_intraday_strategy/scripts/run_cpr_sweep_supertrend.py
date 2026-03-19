"""
CPR Intraday Strategy — SuperTrend Parameter Sweep
===================================================
Sweeps st_period x st_multiplier with wide CPR filters.
Preloads data once, writes CSV incrementally after each config.
"""

import sys
import os
import csv
import time
import json
import logging

# Suppress logging
logging.disable(logging.WARNING)

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.cpr_intraday_engine import CPRIntradayEngine, CPRIntradayConfig

# ── Constants ──────────────────────────────────────────────────────────────
SYMBOLS = [
    'BHARTIARTL', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK', 'INFY',
    'ITC', 'KOTAKBANK', 'RELIANCE', 'SBIN', 'TCS',
]

START_DATE = '2024-01-01'
END_DATE = '2025-10-27'

# Wide CPR filters to maximise trade count
NARROW_CPR_THRESHOLD = 3.0
CPR_PROXIMITY_PCT = 3.0
MAX_WICK_PCT = 30.0

# Sweep grid
ST_PERIODS = [5, 7, 10, 14, 20]
ST_MULTIPLIERS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'cpr_sweep_supertrend.csv')

FIELDNAMES = [
    'label', 'st_period', 'st_mult',
    'narrow_cpr_pct', 'proximity_pct', 'max_wick_pct',
    'total_trades', 'win_rate', 'total_pnl', 'pnl_pct',
    'profit_factor', 'max_drawdown', 'sharpe',
    'avg_trades_per_day', 'exit_reasons',
]


def build_configs():
    """Generate all (label, config_kwargs) pairs."""
    configs = []
    for period in ST_PERIODS:
        for mult in ST_MULTIPLIERS:
            label = f'ST{period}_M{mult}'
            configs.append((label, {
                'st_period': period,
                'st_multiplier': mult,
            }))
    return configs


def load_done_labels():
    """Return set of labels already completed in the CSV."""
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline='') as f:
            for row in csv.DictReader(f):
                done.add(row['label'])
    return done


def write_header_if_needed():
    """Write CSV header if the file doesn't exist or is empty."""
    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def append_row(row):
    """Append a single result row to CSV."""
    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def run_batch(configs_batch, daily_data, five_min_data):
    """Run a batch of configs using preloaded data."""
    done = load_done_labels()
    total = len(configs_batch)

    for i, (label, params) in enumerate(configs_batch, 1):
        if label in done:
            print(f'[{i}/{total}] {label} — SKIPPED (already done)', flush=True)
            continue

        config = CPRIntradayConfig(
            symbols=SYMBOLS,
            start_date=START_DATE,
            end_date=END_DATE,
            narrow_cpr_threshold=NARROW_CPR_THRESHOLD,
            cpr_proximity_pct=CPR_PROXIMITY_PCT,
            max_wick_pct=MAX_WICK_PCT,
            st_period=params['st_period'],
            st_multiplier=params['st_multiplier'],
        )

        engine = CPRIntradayEngine(
            config,
            preloaded_daily=daily_data,
            preloaded_5min=five_min_data,
        )

        t0 = time.time()
        print(f'[{i}/{total}] {label} ...', end='', flush=True)

        try:
            result = engine.run()
        except Exception as e:
            print(f' ERROR: {e}', flush=True)
            continue

        elapsed = time.time() - t0

        exit_reasons_str = json.dumps(result.exit_reason_counts) if result.exit_reason_counts else '{}'

        row = {
            'label': label,
            'st_period': params['st_period'],
            'st_mult': params['st_multiplier'],
            'narrow_cpr_pct': NARROW_CPR_THRESHOLD,
            'proximity_pct': CPR_PROXIMITY_PCT,
            'max_wick_pct': MAX_WICK_PCT,
            'total_trades': result.total_trades,
            'win_rate': round(result.win_rate, 2),
            'total_pnl': round(result.total_pnl, 2),
            'pnl_pct': round(result.total_pnl_pct, 2),
            'profit_factor': round(result.profit_factor, 4),
            'max_drawdown': round(result.max_drawdown, 2),
            'sharpe': round(result.sharpe_ratio, 4),
            'avg_trades_per_day': round(result.avg_trades_per_day, 4),
            'exit_reasons': exit_reasons_str,
        }

        append_row(row)
        print(
            f' {elapsed:.0f}s | Trades={result.total_trades} '
            f'WR={result.win_rate:.1f}% PnL={result.total_pnl:,.0f} '
            f'PF={result.profit_factor:.2f} Sharpe={result.sharpe_ratio:.2f}',
            flush=True,
        )


def main():
    """Entry point — preload data, then run whichever batch is requested."""
    batch_start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    all_configs = build_configs()
    batch = all_configs[batch_start:batch_start + batch_size]

    if not batch:
        print('No configs to run in this batch.', flush=True)
        return

    print(f'=== CPR SuperTrend Sweep — batch [{batch_start}:{batch_start+batch_size}] '
          f'({len(batch)} configs) ===', flush=True)
    print(f'CPR filters: narrow={NARROW_CPR_THRESHOLD}% proximity={CPR_PROXIMITY_PCT}% '
          f'wick={MAX_WICK_PCT}%', flush=True)

    write_header_if_needed()

    # Preload data once
    print('Preloading data ...', flush=True)
    t0 = time.time()
    daily_data, five_min_data = CPRIntradayEngine.preload_data(
        SYMBOLS, START_DATE, END_DATE,
    )
    print(f'Data loaded in {time.time()-t0:.1f}s — '
          f'{len(daily_data)} daily, {len(five_min_data)} 5-min symbols', flush=True)

    run_batch(batch, daily_data, five_min_data)
    print('=== Batch complete ===', flush=True)


if __name__ == '__main__':
    main()
