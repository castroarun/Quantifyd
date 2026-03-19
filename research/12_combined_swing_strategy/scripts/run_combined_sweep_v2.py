"""
Fast sweep: precompute signals ONCE, then vary allocation/sizing/stops only.
"""

import csv
import os
import sys
import time
import logging
import numpy as np
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from combined_strategy_backtest import (
    StrategyConfig, load_data, precompute_all_signals,
    run_portfolio_backtest, INITIAL_CAPITAL
)

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_sweep_v2.csv')

FIELDNAMES = [
    'label', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'calmar',
    'profit_factor', 'total_trades', 'win_rate', 'total_pnl',
    'final_value', 'total_return_pct',
    'trend_alloc', 'meanrev_alloc', 'breakout_alloc',
    'max_pos_trend', 'max_pos_meanrev', 'max_pos_breakout',
    'hard_stop_pct', 'max_position_pct',
    'trend_trades', 'trend_pf', 'trend_pnl',
    'meanrev_trades', 'meanrev_pf', 'meanrev_pnl',
    'breakout_trades', 'breakout_pf', 'breakout_pnl',
]


def compute_metrics(trades, equity_curve):
    if not trades or not equity_curve:
        return None
    dates = sorted(equity_curve.keys())
    values = [equity_curve[d] for d in dates]
    years = (dates[-1] - dates[0]).days / 365.25
    final_value = values[-1]
    cagr = (final_value / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 else 0

    peak = INITIAL_CAPITAL
    max_dd = 0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    returns = np.diff(values) / values[:-1] if len(values) > 1 else np.array([0])
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    neg_ret = returns[returns < 0]
    ds = neg_ret.std() if len(neg_ret) > 0 else 0.001
    sortino = (returns.mean() / ds * np.sqrt(252)) if ds > 0 else 0
    calmar = cagr / max_dd if max_dd > 0 else 0

    pnl_pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnl_pcts if p > 0]
    win_rate = len(wins) / len(pnl_pcts) * 100 if pnl_pcts else 0
    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    pf = gp / gl if gl > 0 else 99.0

    strat_stats = {}
    for s in ['trend', 'meanrev', 'breakout']:
        st = [t for t in trades if t.strategy == s]
        if st:
            sgp = sum(t.pnl for t in st if t.pnl > 0)
            sgl = abs(sum(t.pnl for t in st if t.pnl <= 0))
            strat_stats[s] = {'trades': len(st), 'pf': round(sgp/sgl, 2) if sgl > 0 else 99.0,
                              'pnl': round(sum(t.pnl for t in st))}
        else:
            strat_stats[s] = {'trades': 0, 'pf': 0, 'pnl': 0}

    return {
        'cagr': round(cagr * 100, 2), 'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2), 'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 2), 'profit_factor': round(pf, 2),
        'total_trades': len(trades), 'win_rate': round(win_rate, 1),
        'total_pnl': round(sum(t.pnl for t in trades)),
        'final_value': round(final_value),
        'total_return_pct': round((final_value / INITIAL_CAPITAL - 1) * 100, 2),
        'trend_trades': strat_stats['trend']['trades'],
        'trend_pf': strat_stats['trend']['pf'],
        'trend_pnl': strat_stats['trend']['pnl'],
        'meanrev_trades': strat_stats['meanrev']['trades'],
        'meanrev_pf': strat_stats['meanrev']['pf'],
        'meanrev_pnl': strat_stats['meanrev']['pnl'],
        'breakout_trades': strat_stats['breakout']['trades'],
        'breakout_pf': strat_stats['breakout']['pf'],
        'breakout_pnl': strat_stats['breakout']['pnl'],
    }


def write_row(row):
    exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    with open(OUTPUT_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerow(row)


# Skip done
done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {r['label'] for r in csv.DictReader(f)}
    if done:
        print(f"Skipping {len(done)} already done")

# Load data once
print("Loading data...", flush=True)
data_60, data_daily = load_data()

# Precompute signals once with default strategy params
print("\nPrecomputing signals (default params)...", flush=True)
default_config = StrategyConfig()
events = precompute_all_signals(data_60, data_daily, default_config)
print(f"Events: {len(events):,}")

# ── Define configs: only vary allocation/sizing/risk params ──
configs = []

# Allocation sweeps
for t, m, b, label in [
    (0.50, 0.30, 0.20, "A_50_30_20"),
    (0.60, 0.25, 0.15, "A_60_25_15"),
    (0.70, 0.20, 0.10, "A_70_20_10"),
    (0.80, 0.15, 0.05, "A_80_15_05"),
    (0.50, 0.50, 0.00, "A_50_50_00"),
    (0.60, 0.40, 0.00, "A_60_40_00"),
    (0.70, 0.30, 0.00, "A_70_30_00"),
    (0.80, 0.20, 0.00, "A_80_20_00"),
    (0.40, 0.40, 0.20, "A_40_40_20"),
    (0.30, 0.50, 0.20, "A_30_50_20"),
    (0.50, 0.20, 0.30, "A_50_20_30"),
    (0.40, 0.30, 0.30, "A_40_30_30"),
    (1.00, 0.00, 0.00, "TREND_ONLY"),
    (0.00, 1.00, 0.00, "MR_ONLY"),
    (0.00, 0.00, 1.00, "BO_ONLY"),
]:
    c = StrategyConfig(trend_alloc=t, meanrev_alloc=m, breakout_alloc=b)
    configs.append((label, c))

# Position count sweeps (best allocation from above will be refined)
for mp_t, mp_m, mp_b, label in [
    (5, 5, 5, "P5_5_5"),
    (8, 5, 8, "P8_5_8"),
    (10, 8, 10, "P10_8_10"),
    (15, 10, 10, "P15_10_10"),
    (15, 10, 15, "P15_10_15"),
    (20, 10, 10, "P20_10_10"),
    (3, 3, 3, "P3_3_3"),
    (5, 3, 0, "P5_3_0_noBO"),
    (8, 5, 0, "P8_5_0_noBO"),
    (10, 8, 0, "P10_8_0_noBO"),
    (15, 10, 0, "P15_10_0_noBO"),
    (20, 10, 0, "P20_10_0_noBO"),
]:
    # Use no-breakout allocation for _noBO labels
    if 'noBO' in label:
        c = StrategyConfig(trend_alloc=0.60, meanrev_alloc=0.40, breakout_alloc=0.00,
                           max_positions_trend=mp_t, max_positions_meanrev=mp_m, max_positions_breakout=mp_b)
    else:
        c = StrategyConfig(trend_alloc=0.60, meanrev_alloc=0.25, breakout_alloc=0.15,
                           max_positions_trend=mp_t, max_positions_meanrev=mp_m, max_positions_breakout=mp_b)
    configs.append((label, c))

# Stop loss sweeps
for sl, label in [
    (0.02, "SL2"),
    (0.03, "SL3"),
    (0.05, "SL5"),
    (0.08, "SL8"),
    (0.10, "SL10"),
    (0.15, "SL15"),
    (0.20, "SL20"),
    (0.50, "SL50_noSL"),
]:
    c = StrategyConfig(trend_alloc=0.60, meanrev_alloc=0.40, breakout_alloc=0.00,
                       hard_stop_pct=sl)
    configs.append((label, c))

# Max position size sweeps
for mps, label in [
    (0.02, "MPS2"),
    (0.03, "MPS3"),
    (0.05, "MPS5"),
    (0.08, "MPS8"),
    (0.10, "MPS10"),
    (0.15, "MPS15"),
]:
    c = StrategyConfig(trend_alloc=0.60, meanrev_alloc=0.40, breakout_alloc=0.00,
                       max_position_pct=mps)
    configs.append((label, c))

print(f"\nTotal configs: {len(configs)} ({len(done)} done)")

for i, (label, config) in enumerate(configs, 1):
    if label in done:
        continue
    t0 = time.time()
    print(f"[{i}/{len(configs)}] {label} ...", end='', flush=True)

    try:
        trades, eq_curve = run_portfolio_backtest(events, data_60, data_daily, config)
        metrics = compute_metrics(trades, eq_curve)

        if metrics:
            row = {
                'label': label,
                **metrics,
                'trend_alloc': config.trend_alloc,
                'meanrev_alloc': config.meanrev_alloc,
                'breakout_alloc': config.breakout_alloc,
                'max_pos_trend': config.max_positions_trend,
                'max_pos_meanrev': config.max_positions_meanrev,
                'max_pos_breakout': config.max_positions_breakout,
                'hard_stop_pct': config.hard_stop_pct,
                'max_position_pct': config.max_position_pct,
            }
            write_row(row)
            elapsed = time.time() - t0
            print(f" {elapsed:.0f}s | CAGR={metrics['cagr']:.2f}% Sharpe={metrics['sharpe']:.2f} "
                  f"MaxDD={metrics['max_drawdown']:.1f}% Calmar={metrics['calmar']:.2f} "
                  f"PF={metrics['profit_factor']:.2f}", flush=True)
        else:
            print(f" {time.time()-t0:.0f}s | NO TRADES", flush=True)
    except Exception as e:
        print(f" {time.time()-t0:.0f}s | ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

print("\nSweep complete!")
