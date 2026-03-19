"""
CPR Intraday Sweep: RSI & Stochastic Indicator Experiments
==========================================================

Sweeps RSI, Stochastic, and combined RSI+Stochastic filters on top of
the CPR intraday strategy with wide base settings.

Groups:
  A) RSI only — 12 configs
  B) Stochastic only — 6 configs
  C) RSI + Stochastic combined — 4 configs
  + 1 baseline (no indicators)

Total: ~23 configs
"""

import csv
import json
import logging
import os
import sys
import time

# Suppress noisy logs
logging.disable(logging.WARNING)

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.cpr_intraday_engine import CPRIntradayEngine, CPRIntradayConfig

# --- Output file ---
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpr_sweep_rsi_stoch.csv')

FIELDNAMES = [
    'label', 'use_rsi', 'rsi_period', 'rsi_ob', 'rsi_os',
    'use_stoch', 'stoch_k', 'stoch_d',
    'total_trades', 'win_rate', 'total_pnl', 'pnl_pct',
    'profit_factor', 'max_drawdown', 'sharpe', 'avg_trades_per_day',
    'exit_reasons',
]

SYMBOLS = [
    'BHARTIARTL', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK', 'INFY',
    'ITC', 'KOTAKBANK', 'RELIANCE', 'SBIN', 'TCS',
]

# --- Base CPR settings (wide to get trades) ---
BASE = dict(
    symbols=SYMBOLS,
    start_date='2024-01-01',
    end_date='2025-10-27',
    initial_capital=1_000_000,
    narrow_cpr_threshold=3.0,
    cpr_proximity_pct=3.0,
    max_wick_pct=30.0,
    st_period=7,
    st_multiplier=3.0,
)


def build_configs():
    """Build all experiment configurations."""
    configs = []

    # 0) BASELINE — no indicators
    configs.append(('BASELINE', {}))

    # ---------------------------------------------------------------
    # Group A: RSI variations (12 configs)
    # ---------------------------------------------------------------
    # Representative combos across period x overbought x oversold
    rsi_combos = [
        # period 7 — fast RSI
        (7, 65, 35),
        (7, 70, 30),
        (7, 75, 25),
        (7, 70, 25),
        # period 14 — standard RSI
        (14, 65, 35),
        (14, 70, 30),
        (14, 75, 25),
        (14, 65, 25),
        # period 21 — slow RSI
        (21, 65, 35),
        (21, 70, 30),
        (21, 75, 25),
        (21, 70, 25),
    ]
    for rsi_p, rsi_ob, rsi_os in rsi_combos:
        label = f'RSI_{rsi_p}_OB{rsi_ob}_OS{rsi_os}'
        params = dict(
            use_rsi=True,
            rsi_period=rsi_p,
            rsi_overbought=float(rsi_ob),
            rsi_oversold=float(rsi_os),
        )
        configs.append((label, params))

    # ---------------------------------------------------------------
    # Group B: Stochastic variations (6 configs)
    # ---------------------------------------------------------------
    stoch_combos = [
        (5, 3),
        (5, 5),
        (14, 3),
        (14, 5),
        (21, 3),
        (21, 5),
    ]
    for sk, sd in stoch_combos:
        label = f'STOCH_K{sk}_D{sd}'
        params = dict(
            use_stochastic=True,
            stoch_k=sk,
            stoch_d=sd,
        )
        configs.append((label, params))

    # ---------------------------------------------------------------
    # Group C: RSI + Stochastic combined (4 configs)
    # Best RSI settings paired with best Stochastic settings
    # ---------------------------------------------------------------
    combined = [
        # Fast RSI + fast Stoch
        (7, 70, 30, 5, 3),
        # Standard RSI + standard Stoch
        (14, 70, 30, 14, 3),
        # Fast RSI + standard Stoch
        (7, 75, 25, 14, 3),
        # Standard RSI + fast Stoch
        (14, 65, 35, 5, 3),
    ]
    for rsi_p, rsi_ob, rsi_os, sk, sd in combined:
        label = f'RSI{rsi_p}_OB{rsi_ob}_OS{rsi_os}_STOCH_K{sk}_D{sd}'
        params = dict(
            use_rsi=True,
            rsi_period=rsi_p,
            rsi_overbought=float(rsi_ob),
            rsi_oversold=float(rsi_os),
            use_stochastic=True,
            stoch_k=sk,
            stoch_d=sd,
        )
        configs.append((label, params))

    return configs


def run_one(label, params, daily_data, five_min_data):
    """Run a single CPR backtest config and return a result dict."""
    cfg_args = {**BASE, **params}
    config = CPRIntradayConfig(**cfg_args)

    engine = CPRIntradayEngine(
        config,
        preloaded_daily=daily_data,
        preloaded_5min=five_min_data,
    )
    result = engine.run()

    exit_reasons = json.dumps(result.exit_reason_counts) if result.exit_reason_counts else '{}'

    return {
        'label': label,
        'use_rsi': params.get('use_rsi', False),
        'rsi_period': params.get('rsi_period', ''),
        'rsi_ob': params.get('rsi_overbought', ''),
        'rsi_os': params.get('rsi_oversold', ''),
        'use_stoch': params.get('use_stochastic', False),
        'stoch_k': params.get('stoch_k', ''),
        'stoch_d': params.get('stoch_d', ''),
        'total_trades': result.total_trades,
        'win_rate': round(result.win_rate, 2),
        'total_pnl': round(result.total_pnl, 2),
        'pnl_pct': round(result.total_pnl_pct, 2),
        'profit_factor': round(result.profit_factor, 2),
        'max_drawdown': round(result.max_drawdown, 2),
        'sharpe': round(result.sharpe_ratio, 2),
        'avg_trades_per_day': round(result.avg_trades_per_day, 2),
        'exit_reasons': exit_reasons,
    }


def main():
    configs = build_configs()
    total = len(configs)
    print(f'CPR RSI/Stochastic sweep: {total} configs', flush=True)

    # --- Skip already-completed configs ---
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'r') as f:
            done = {row['label'] for row in csv.DictReader(f)}
        if done:
            print(f'Skipping {len(done)} already-completed configs', flush=True)

    remaining = [(l, p) for l, p in configs if l not in done]
    if not remaining:
        print('All configs already completed!', flush=True)
        return

    # --- Write CSV header if file doesn't exist ---
    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # --- Preload data once ---
    print('Preloading data...', flush=True)
    t0 = time.time()
    daily_data, five_min_data = CPRIntradayEngine.preload_data(
        SYMBOLS, BASE['start_date'], BASE['end_date']
    )
    load_time = time.time() - t0
    print(f'Data loaded in {load_time:.1f}s ({len(daily_data)} daily, {len(five_min_data)} 5min)', flush=True)

    # --- Run each config ---
    for i, (label, params) in enumerate(remaining, 1):
        print(f'[{i}/{len(remaining)}] {label} ...', end='', flush=True)
        t1 = time.time()

        try:
            row = run_one(label, params, daily_data, five_min_data)
            elapsed = time.time() - t1

            # Write incrementally
            with open(OUTPUT_CSV, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

            print(
                f' {elapsed:.0f}s | trades={row["total_trades"]} '
                f'WR={row["win_rate"]}% PnL={row["total_pnl"]:+,.0f} '
                f'PF={row["profit_factor"]} Sharpe={row["sharpe"]}',
                flush=True,
            )
        except Exception as e:
            elapsed = time.time() - t1
            print(f' ERROR ({elapsed:.0f}s): {e}', flush=True)

    print('\nSweep complete! Results in:', OUTPUT_CSV, flush=True)


if __name__ == '__main__':
    main()
