#!/usr/bin/env python3
"""Agent 1: CPR Baseline Parameter Sweep

Sweeps narrow_cpr_threshold, cpr_proximity_pct, and max_wick_pct
across 108 configurations (6 x 6 x 3) on 10 F&O stocks.
Period: 2024-06-01 to 2024-12-31 (6 months, faster runs).
"""
import csv, os, sys, time, logging, io
from contextlib import redirect_stdout
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.cpr_intraday_engine import CPRIntradayEngine, CPRIntradayConfig

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpr_sweep_baseline.csv')
FIELDNAMES = ['label', 'narrow_cpr_pct', 'proximity_pct', 'max_wick_pct', 'st_period', 'st_mult',
              'total_trades', 'win_rate', 'total_pnl', 'pnl_pct', 'profit_factor',
              'max_drawdown', 'sharpe', 'avg_trades_per_day', 'exit_reasons']

# Symbols with 5-min data
SYMBOLS = ['BHARTIARTL', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK', 'INFY',
           'ITC', 'KOTAKBANK', 'RELIANCE', 'SBIN', 'TCS']

START_DATE = '2024-01-01'
END_DATE = '2025-10-27'

# Write header if file doesn't exist
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

# Skip completed
done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {row['label'] for row in csv.DictReader(f)}
    if done:
        print(f'Skipping {len(done)} already-completed configs')

# SWEEP CONFIGS: Vary narrow_cpr_threshold, cpr_proximity_pct, max_wick_pct
configs = []
for narrow_cpr in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    for proximity in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for max_wick in [15.0, 25.0, 40.0]:
            label = f'CPR{narrow_cpr}_PROX{proximity}_WICK{max_wick}'
            configs.append((label, {
                'narrow_cpr_threshold': narrow_cpr,
                'cpr_proximity_pct': proximity,
                'max_wick_pct': max_wick,
                'st_period': 7,
                'st_multiplier': 3.0,
            }))

# Filter out already done
pending = [(l, p) for l, p in configs if l not in done]
print(f'Total configs: {len(configs)}, Pending: {len(pending)}')

if not pending:
    print('All configs already completed!')
    sys.exit(0)

# Preload data once (suppress engine prints)
print(f'Loading data for {len(SYMBOLS)} symbols ({START_DATE} to {END_DATE})...')
t0 = time.time()
with redirect_stdout(io.StringIO()):
    daily_data, five_min_data = CPRIntradayEngine.preload_data(SYMBOLS, START_DATE, END_DATE)
print(f'Data loaded in {time.time()-t0:.1f}s — {len(daily_data)} daily, {len(five_min_data)} 5min symbols')

total = len(configs)
completed_count = len(done)

for label, params in pending:
    completed_count += 1
    print(f'[{completed_count}/{total}] {label} ...', end='', flush=True)
    t1 = time.time()

    try:
        config = CPRIntradayConfig(
            symbols=SYMBOLS,
            start_date=START_DATE,
            end_date=END_DATE,
            initial_capital=1_000_000,
            narrow_cpr_threshold=params['narrow_cpr_threshold'],
            cpr_proximity_pct=params['cpr_proximity_pct'],
            max_wick_pct=params['max_wick_pct'],
            st_period=params['st_period'],
            st_multiplier=params['st_multiplier'],
        )

        engine = CPRIntradayEngine(config, preloaded_daily=daily_data, preloaded_5min=five_min_data)

        # Suppress engine's verbose output
        with redirect_stdout(io.StringIO()):
            result = engine.run()

        row = {
            'label': label,
            'narrow_cpr_pct': params['narrow_cpr_threshold'],
            'proximity_pct': params['cpr_proximity_pct'],
            'max_wick_pct': params['max_wick_pct'],
            'st_period': params['st_period'],
            'st_mult': params['st_multiplier'],
            'total_trades': result.total_trades,
            'win_rate': round(result.win_rate, 2),
            'total_pnl': round(result.total_pnl, 0),
            'pnl_pct': round(result.total_pnl_pct, 2),
            'profit_factor': round(result.profit_factor, 2),
            'max_drawdown': round(result.max_drawdown, 2),
            'sharpe': round(result.sharpe_ratio, 2),
            'avg_trades_per_day': round(result.avg_trades_per_day, 2),
            'exit_reasons': str(result.exit_reason_counts),
        }

        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        elapsed = time.time() - t1
        print(f' {elapsed:.0f}s | Trades={result.total_trades} WR={result.win_rate:.1f}% PF={result.profit_factor:.2f} PnL={result.total_pnl:+,.0f}')
        sys.stdout.flush()

    except Exception as e:
        import traceback
        print(f' ERROR: {e}')
        traceback.print_exc()
        sys.stdout.flush()

print(f'\nDone! Results in {OUTPUT_CSV}')
