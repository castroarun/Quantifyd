"""
Phase 2 sweep: vary strategy parameters on best allocation (Trend+MR, no breakout).
Precompute signals per unique param set.
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

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_sweep_v3.csv')

FIELDNAMES = [
    'label', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'calmar',
    'profit_factor', 'total_trades', 'win_rate', 'total_pnl',
    'final_value', 'total_return_pct',
    'ema_fast', 'ema_slow', 'adx_threshold', 'rsi_period', 'rsi_entry',
    'max_pos_trend', 'max_pos_meanrev', 'hard_stop_pct',
    'trend_trades', 'trend_pf', 'trend_pnl',
    'meanrev_trades', 'meanrev_pf', 'meanrev_pnl',
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
    for s in ['trend', 'meanrev']:
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
    }


def write_row(row):
    exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    with open(OUTPUT_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerow(row)


done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {r['label'] for r in csv.DictReader(f)}
    if done:
        print(f"Skipping {len(done)} already done")

print("Loading data...", flush=True)
data_60, data_daily = load_data()

# Define configs grouped by signal params (to minimize recomputation)
# Each group shares the same strategy params = same signals
configs = []

# Base allocation: 60/40 trend/MR, no breakout
BASE = dict(
    trend_alloc=0.60, meanrev_alloc=0.40, breakout_alloc=0.00,
    hard_stop_pct=0.10,
)

# Group 1: EMA period sweep x ADX threshold
for ema_f, ema_s in [(9,21), (10,30), (13,34), (15,40), (20,50), (25,60)]:
    for adx_t in [20, 25, 30]:
        for rsi_p, rsi_e in [(21, 30)]:  # fix MR params
            for mp_t, mp_m in [(8,5), (15,10), (20,10)]:
                label = f"E{ema_f}_{ema_s}_ADX{adx_t}_R{rsi_p}_{int(rsi_e)}_P{mp_t}_{mp_m}"
                c = StrategyConfig(
                    **BASE,
                    ema_fast=ema_f, ema_slow=ema_s, adx_threshold=adx_t,
                    rsi_period=rsi_p, rsi_entry=rsi_e,
                    max_positions_trend=mp_t, max_positions_meanrev=mp_m,
                    max_positions_breakout=0,
                )
                configs.append((label, c))

# Group 2: RSI sweep (fix best EMA from above = 20/50 ADX25)
for rsi_p, rsi_e in [(7, 20), (7, 25), (14, 25), (14, 30), (21, 25), (21, 30), (21, 35)]:
    for mp_t, mp_m in [(8,5), (15,10), (20,10)]:
        label = f"E20_50_ADX25_R{rsi_p}_{int(rsi_e)}_P{mp_t}_{mp_m}"
        c = StrategyConfig(
            **BASE,
            ema_fast=20, ema_slow=50, adx_threshold=25,
            rsi_period=rsi_p, rsi_entry=rsi_e,
            max_positions_trend=mp_t, max_positions_meanrev=mp_m,
            max_positions_breakout=0,
        )
        configs.append((label, c))

# Deduplicate
seen = set()
unique_configs = []
for label, c in configs:
    if label not in seen:
        seen.add(label)
        unique_configs.append((label, c))
configs = unique_configs

print(f"Total configs: {len(configs)} ({len(done)} done)")

# Group by signal params to avoid redundant precomputation
signal_cache = {}
total = len(configs)

for i, (label, config) in enumerate(configs, 1):
    if label in done:
        continue

    sig_key = (config.ema_fast, config.ema_slow, config.adx_threshold,
               config.rsi_period, config.rsi_entry)

    if sig_key not in signal_cache:
        events = precompute_all_signals(data_60, data_daily, config)
        signal_cache[sig_key] = events
        # Keep cache bounded
        if len(signal_cache) > 5:
            oldest = next(iter(signal_cache))
            del signal_cache[oldest]
    else:
        events = signal_cache[sig_key]

    t0 = time.time()
    print(f"[{i}/{total}] {label} ...", end='', flush=True)

    try:
        trades, eq_curve = run_portfolio_backtest(events, data_60, data_daily, config)
        metrics = compute_metrics(trades, eq_curve)

        if metrics:
            row = {
                'label': label, **metrics,
                'ema_fast': config.ema_fast, 'ema_slow': config.ema_slow,
                'adx_threshold': config.adx_threshold,
                'rsi_period': config.rsi_period, 'rsi_entry': config.rsi_entry,
                'max_pos_trend': config.max_positions_trend,
                'max_pos_meanrev': config.max_positions_meanrev,
                'hard_stop_pct': config.hard_stop_pct,
            }
            write_row(row)
            elapsed = time.time() - t0
            print(f" {elapsed:.0f}s | CAGR={metrics['cagr']:.2f}% Sharpe={metrics['sharpe']:.2f} "
                  f"MaxDD={metrics['max_drawdown']:.1f}% Calmar={metrics['calmar']:.2f} "
                  f"PF={metrics['profit_factor']:.2f} Trades={metrics['total_trades']}", flush=True)
        else:
            print(f" {time.time()-t0:.0f}s | NO TRADES", flush=True)
    except Exception as e:
        print(f" {time.time()-t0:.0f}s | ERROR: {e}", flush=True)

print("\nSweep complete!")
print(f"Results in: {OUTPUT_CSV}")
