#!/usr/bin/env python3
"""CPR V3 Sweep — 79 F&O Stocks (Focused)
==========================================
~20 min per config with 79 stocks, so we test only the most promising.

Configs based on findings from 10-stock sweep:
  - Top PF configs with 79 stocks
  - Relaxed wick for more trades
  - Out-of-sample validation

Usage:
  python run_cpr_v3_79stocks.py              # Run next pending config
  python run_cpr_v3_79stocks.py --oos        # Run out-of-sample configs
  python run_cpr_v3_79stocks.py --batch N    # Start from config index N
  python run_cpr_v3_79stocks.py --size N     # Run N configs (default 1)
"""
import csv, os, sys, time, json, logging, io, argparse
from contextlib import redirect_stdout

logging.disable(logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.cpr_intraday_engine import (
    CPRIntradayEngine, CPRIntradayConfig, INTRADAY_SYMBOLS,
)

FULL_CSV = os.path.join(PROJECT_ROOT, 'cpr_v3_79stocks_full.csv')
OOS_CSV = os.path.join(PROJECT_ROOT, 'cpr_v3_79stocks_oos.csv')

FIELDNAMES = [
    'label', 'period', 'n_symbols',
    'narrow_cpr_pct', 'proximity_pct', 'max_wick_pct',
    'st_period', 'st_mult',
    'total_trades', 'win_rate', 'total_pnl', 'pnl_pct',
    'profit_factor', 'max_drawdown', 'sharpe', 'sortino',
    'avg_trades_per_day', 'avg_holding_min',
    'days_with_trades', 'total_trading_days',
    'exit_reasons',
]


def build_full_period_configs():
    """12 focused configs — each takes ~20 min."""
    configs = [
        # Already run: CPR0.5_PROX2.0_WICK25_ST7_M4.0 → 22 trades, PF 1.34
        # --- Vary ST multiplier (best from 10-stock sweep) ---
        ('CPR0.5_PROX2.0_WICK25_ST5_M4.0', dict(narrow_cpr_threshold=0.5, cpr_proximity_pct=2.0, max_wick_pct=25.0, st_period=5, st_multiplier=4.0)),
        ('CPR0.5_PROX2.0_WICK25_ST7_M3.5', dict(narrow_cpr_threshold=0.5, cpr_proximity_pct=2.0, max_wick_pct=25.0, st_period=7, st_multiplier=3.5)),
        ('CPR0.5_PROX2.0_WICK25_ST5_M3.5', dict(narrow_cpr_threshold=0.5, cpr_proximity_pct=2.0, max_wick_pct=25.0, st_period=5, st_multiplier=3.5)),
        # --- Vary proximity (more trades) ---
        ('CPR0.5_PROX1.5_WICK25_ST7_M4.0', dict(narrow_cpr_threshold=0.5, cpr_proximity_pct=1.5, max_wick_pct=25.0, st_period=7, st_multiplier=4.0)),
        ('CPR0.5_PROX3.0_WICK25_ST7_M4.0', dict(narrow_cpr_threshold=0.5, cpr_proximity_pct=3.0, max_wick_pct=25.0, st_period=7, st_multiplier=4.0)),
        # --- Relaxed wick (more trades) ---
        ('CPR0.5_PROX2.0_WICK40_ST7_M4.0', dict(narrow_cpr_threshold=0.5, cpr_proximity_pct=2.0, max_wick_pct=40.0, st_period=7, st_multiplier=4.0)),
        ('CPR0.5_PROX2.0_WICK50_ST7_M4.0', dict(narrow_cpr_threshold=0.5, cpr_proximity_pct=2.0, max_wick_pct=50.0, st_period=7, st_multiplier=4.0)),
        ('CPR0.5_PROX2.0_WICK60_ST7_M4.0', dict(narrow_cpr_threshold=0.5, cpr_proximity_pct=2.0, max_wick_pct=60.0, st_period=7, st_multiplier=4.0)),
        # --- Wider CPR (more trades) ---
        ('CPR1.0_PROX2.0_WICK30_ST7_M4.0', dict(narrow_cpr_threshold=1.0, cpr_proximity_pct=2.0, max_wick_pct=30.0, st_period=7, st_multiplier=4.0)),
        ('CPR1.0_PROX2.0_WICK30_ST7_M3.5', dict(narrow_cpr_threshold=1.0, cpr_proximity_pct=2.0, max_wick_pct=30.0, st_period=7, st_multiplier=3.5)),
        # --- Best combo: tight CPR + high ST + slightly relaxed wick ---
        ('CPR0.5_PROX2.0_WICK30_ST7_M4.0', dict(narrow_cpr_threshold=0.5, cpr_proximity_pct=2.0, max_wick_pct=30.0, st_period=7, st_multiplier=4.0)),
        ('CPR0.5_PROX2.0_WICK30_ST10_M4.0', dict(narrow_cpr_threshold=0.5, cpr_proximity_pct=2.0, max_wick_pct=30.0, st_period=10, st_multiplier=4.0)),
    ]
    return configs


def build_oos_configs():
    """Out-of-sample: top 4 configs split by year."""
    top_combos = [
        (0.5, 2.0, 25.0, 7, 4.0),
        (0.5, 2.0, 25.0, 5, 4.0),
        (0.5, 2.0, 25.0, 7, 3.5),
        (0.5, 2.0, 40.0, 7, 4.0),
    ]
    configs = []
    for cpr, prox, wick, st_p, st_m in top_combos:
        for period_label, start, end in [
            ('2024', '2024-01-01', '2024-12-31'),
            ('2025', '2025-01-01', '2025-10-27'),
        ]:
            label = f'{period_label}_CPR{cpr}_PROX{prox}_WICK{wick}_ST{st_p}_M{st_m}'
            configs.append((label, {
                'narrow_cpr_threshold': cpr,
                'cpr_proximity_pct': prox,
                'max_wick_pct': wick,
                'st_period': st_p,
                'st_multiplier': st_m,
                'start_date': start,
                'end_date': end,
            }))
    return configs


def load_done(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            done = {row['label'] for row in csv.DictReader(f)}
    return done


def ensure_header(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def run_configs(configs, csv_path, default_start='2024-01-01', default_end='2025-10-27'):
    ensure_header(csv_path)
    done = load_done(csv_path)
    pending = [(l, p) for l, p in configs if l not in done]
    print(f'Total: {len(configs)} | Done: {len(done)} | Pending: {len(pending)}')

    if not pending:
        print('All configs already completed!')
        return

    symbols = INTRADAY_SYMBOLS
    print(f'Universe: {len(symbols)} F&O stocks')

    # Group by date range
    date_groups = {}
    for label, params in pending:
        key = (params.get('start_date', default_start), params.get('end_date', default_end))
        date_groups.setdefault(key, []).append((label, params))

    for (start, end), group in date_groups.items():
        print(f'\n--- Period: {start} to {end} ({len(group)} configs) ---')
        print('Preloading data...', flush=True)
        t0 = time.time()
        daily_data, five_min_data = CPRIntradayEngine.preload_data(symbols, start, end)
        print(f'Data loaded in {time.time()-t0:.1f}s — {len(daily_data)} daily, {len(five_min_data)} 5-min', flush=True)

        for i, (label, params) in enumerate(group, 1):
            print(f'\n[{i}/{len(group)}] {label}', flush=True)
            t1 = time.time()

            try:
                config = CPRIntradayConfig(
                    symbols=symbols,
                    start_date=params.get('start_date', default_start),
                    end_date=params.get('end_date', default_end),
                    initial_capital=1_000_000,
                    narrow_cpr_threshold=params['narrow_cpr_threshold'],
                    cpr_proximity_pct=params['cpr_proximity_pct'],
                    max_wick_pct=params['max_wick_pct'],
                    st_period=params['st_period'],
                    st_multiplier=params['st_multiplier'],
                )

                engine = CPRIntradayEngine(config,
                    preloaded_daily=daily_data, preloaded_5min=five_min_data)

                with redirect_stdout(io.StringIO()):
                    result = engine.run()

                row = {
                    'label': label,
                    'period': f"{config.start_date} to {config.end_date}",
                    'n_symbols': len(symbols),
                    'narrow_cpr_pct': params['narrow_cpr_threshold'],
                    'proximity_pct': params['cpr_proximity_pct'],
                    'max_wick_pct': params['max_wick_pct'],
                    'st_period': params['st_period'],
                    'st_mult': params['st_multiplier'],
                    'total_trades': result.total_trades,
                    'win_rate': round(result.win_rate, 2),
                    'total_pnl': round(result.total_pnl, 0),
                    'pnl_pct': round(result.total_pnl_pct, 2),
                    'profit_factor': round(result.profit_factor, 4),
                    'max_drawdown': round(result.max_drawdown, 4),
                    'sharpe': round(result.sharpe_ratio, 4),
                    'sortino': round(result.sortino_ratio, 4),
                    'avg_trades_per_day': round(result.avg_trades_per_day, 4),
                    'avg_holding_min': round(result.avg_holding_minutes, 1),
                    'days_with_trades': result.days_with_trades,
                    'total_trading_days': result.total_trading_days,
                    'exit_reasons': json.dumps(result.exit_reason_counts),
                }

                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

                elapsed = time.time() - t1
                print(f'  {elapsed:.0f}s | Trades={result.total_trades} '
                      f'WR={result.win_rate:.1f}% PF={result.profit_factor:.2f} '
                      f'PnL={result.total_pnl:+,.0f} MaxDD={result.max_drawdown:.2f}%',
                      flush=True)

            except Exception as e:
                import traceback
                print(f'  ERROR: {e}', flush=True)
                traceback.print_exc()

    print(f'\nDone! Results in {csv_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oos', action='store_true', help='Out-of-sample mode')
    parser.add_argument('--batch', type=int, default=0, help='Start index')
    parser.add_argument('--size', type=int, default=1, help='Configs per run (default 1)')
    args = parser.parse_args()

    if args.oos:
        all_configs = build_oos_configs()
        csv_path = OOS_CSV
        print(f'=== CPR V3 OOS Sweep — {len(all_configs)} configs ===')
    else:
        all_configs = build_full_period_configs()
        csv_path = FULL_CSV
        print(f'=== CPR V3 Full Sweep — {len(all_configs)} configs ===')

    batch = all_configs[args.batch:args.batch + args.size]
    if not batch:
        print('No configs in this batch.')
        return

    print(f'Batch [{args.batch}:{args.batch + args.size}] — {len(batch)} configs (~{len(batch)*20} min)')
    run_configs(batch, csv_path)


if __name__ == '__main__':
    main()
