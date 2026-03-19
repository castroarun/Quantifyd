"""
Sweep allocation & sizing parameters for combined multi-strategy backtest.
Tests different capital splits, position counts, and strategy parameter variants.
"""

import csv
import os
import sys
import time
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from combined_strategy_backtest import (
    StrategyConfig, load_data, precompute_all_signals,
    run_portfolio_backtest, INITIAL_CAPITAL, START_DATE, END_DATE
)
import numpy as np

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_sweep_results.csv')

FIELDNAMES = [
    'label', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'calmar',
    'profit_factor', 'total_trades', 'win_rate', 'total_pnl',
    'final_value', 'total_return_pct',
    'trend_alloc', 'meanrev_alloc', 'breakout_alloc',
    'max_pos_trend', 'max_pos_meanrev', 'max_pos_breakout',
    'trend_trades', 'trend_pf', 'meanrev_trades', 'meanrev_pf',
    'breakout_trades', 'breakout_pf',
]


def compute_metrics_quick(trades, equity_curve):
    """Quick metrics without printing."""
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

    returns = []
    for i in range(1, len(values)):
        returns.append(values[i] / values[i-1] - 1)
    returns = np.array(returns)
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 0 and returns.std() > 0 else 0
    neg_returns = returns[returns < 0]
    downside_std = neg_returns.std() if len(neg_returns) > 0 else 0.001
    sortino = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    calmar = cagr / max_dd if max_dd > 0 else 0

    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p <= 0]
    win_rate = len(wins) / len(pnl_pcts) * 100 if pnl_pcts else 0
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0

    # Per-strategy
    strat_stats = {}
    for strat in ['trend', 'meanrev', 'breakout']:
        st = [t for t in trades if t.strategy == strat]
        if st:
            gp = sum(t.pnl for t in st if t.pnl > 0)
            gl = abs(sum(t.pnl for t in st if t.pnl <= 0))
            strat_stats[strat] = {'trades': len(st), 'pf': round(gp / gl, 2) if gl > 0 else 99.0}
        else:
            strat_stats[strat] = {'trades': 0, 'pf': 0}

    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 2),
        'profit_factor': round(profit_factor, 2),
        'total_trades': len(trades),
        'win_rate': round(win_rate, 1),
        'total_pnl': round(sum(pnls)),
        'final_value': round(final_value),
        'total_return_pct': round((final_value / INITIAL_CAPITAL - 1) * 100, 2),
        'trend_trades': strat_stats['trend']['trades'],
        'trend_pf': strat_stats['trend']['pf'],
        'meanrev_trades': strat_stats['meanrev']['trades'],
        'meanrev_pf': strat_stats['meanrev']['pf'],
        'breakout_trades': strat_stats['breakout']['trades'],
        'breakout_pf': strat_stats['breakout']['pf'],
    }


def write_row(row):
    file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    with open(OUTPUT_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# Skip already completed
done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {row['label'] for row in csv.DictReader(f)}
    if done:
        print(f"Skipping {len(done)} already-completed configs")


# ── Define sweep configs ───────────────────────────────────────────────────

configs = []

# 1. Allocation sweeps (keep default position counts)
for t_a, m_a, b_a, label in [
    (0.50, 0.30, 0.20, "BASELINE"),
    (0.70, 0.20, 0.10, "TREND_HEAVY"),
    (0.60, 0.25, 0.15, "TREND_60"),
    (0.40, 0.40, 0.20, "MEANREV_HEAVY"),
    (0.80, 0.10, 0.10, "TREND_80"),
    (0.50, 0.50, 0.00, "NO_BREAKOUT"),
    (0.70, 0.30, 0.00, "TREND_MR_ONLY"),
    (1.00, 0.00, 0.00, "TREND_ONLY"),
    (0.00, 1.00, 0.00, "MEANREV_ONLY"),
    (0.00, 0.00, 1.00, "BREAKOUT_ONLY"),
]:
    c = StrategyConfig(trend_alloc=t_a, meanrev_alloc=m_a, breakout_alloc=b_a)
    configs.append((label, c))

# 2. Position count sweeps (with trend-heavy allocation)
for max_t, max_m, max_b, label in [
    (5, 5, 5, "POS_5_5_5"),
    (15, 10, 15, "POS_15_10_15"),
    (20, 10, 20, "POS_20_10_20"),
    (5, 3, 5, "CONCENTRATED"),
    (3, 3, 3, "VERY_CONCENTRATED"),
]:
    c = StrategyConfig(
        trend_alloc=0.60, meanrev_alloc=0.25, breakout_alloc=0.15,
        max_positions_trend=max_t, max_positions_meanrev=max_m, max_positions_breakout=max_b
    )
    configs.append((label, c))

# 3. Strategy param variants
# Trend: different EMA periods
for fast, slow, label in [
    (13, 34, "EMA13_34"),
    (9, 21, "EMA9_21"),
    (20, 50, "EMA20_50"),  # same as baseline
    (10, 30, "EMA10_30"),
]:
    c = StrategyConfig(
        trend_alloc=0.60, meanrev_alloc=0.25, breakout_alloc=0.15,
        ema_fast=fast, ema_slow=slow
    )
    configs.append((label, c))

# ADX threshold variants
for adx_t, label in [
    (20, "ADX20"),
    (25, "ADX25"),  # baseline
    (30, "ADX30"),
]:
    c = StrategyConfig(
        trend_alloc=0.60, meanrev_alloc=0.25, breakout_alloc=0.15,
        adx_threshold=adx_t
    )
    configs.append((label, c))

# Mean reversion RSI variants
for rsi_p, rsi_e, label in [
    (14, 25, "RSI14_25"),
    (14, 30, "RSI14_30"),
    (21, 30, "RSI21_30"),  # baseline
    (7, 20, "RSI7_20"),
]:
    c = StrategyConfig(
        trend_alloc=0.60, meanrev_alloc=0.25, breakout_alloc=0.15,
        rsi_period=rsi_p, rsi_entry=rsi_e
    )
    configs.append((label, c))

# Breakout hold period
for hold_d, label in [
    (3, "HOLD3"),  # baseline
    (5, "HOLD5"),
    (7, "HOLD7"),
    (10, "HOLD10"),
]:
    c = StrategyConfig(
        trend_alloc=0.60, meanrev_alloc=0.25, breakout_alloc=0.15,
        breakout_hold_days=hold_d
    )
    configs.append((label, c))

# Stop loss variants
for sl, label in [
    (0.03, "SL3"),
    (0.05, "SL5"),  # baseline
    (0.08, "SL8"),
    (0.10, "SL10"),
    (0.15, "SL15"),
]:
    c = StrategyConfig(
        trend_alloc=0.60, meanrev_alloc=0.25, breakout_alloc=0.15,
        hard_stop_pct=sl
    )
    configs.append((label, c))

# Max position size variants
for mps, label in [
    (0.03, "MAXPOS3"),
    (0.05, "MAXPOS5"),  # baseline
    (0.08, "MAXPOS8"),
    (0.10, "MAXPOS10"),
]:
    c = StrategyConfig(
        trend_alloc=0.60, meanrev_alloc=0.25, breakout_alloc=0.15,
        max_position_pct=mps
    )
    configs.append((label, c))


# ── Run sweep ──────────────────────────────────────────────────────────────

print(f"Total configs to sweep: {len(configs)} ({len(done)} already done)")
print("Loading data...", flush=True)
data_60, data_daily = load_data()

total = len(configs)
for i, (label, config) in enumerate(configs, 1):
    if label in done:
        continue

    print(f"\n[{i}/{total}] {label} ...", end='', flush=True)
    t0 = time.time()

    try:
        events = precompute_all_signals(data_60, data_daily, config)
        trades, eq_curve = run_portfolio_backtest(events, data_60, data_daily, config)
        metrics = compute_metrics_quick(trades, eq_curve)

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
            }
            write_row(row)

            elapsed = time.time() - t0
            print(f" {elapsed:.0f}s | CAGR={metrics['cagr']:.2f}% Sharpe={metrics['sharpe']:.2f} "
                  f"MaxDD={metrics['max_drawdown']:.1f}% PF={metrics['profit_factor']:.2f} "
                  f"Calmar={metrics['calmar']:.2f}", flush=True)
        else:
            elapsed = time.time() - t0
            print(f" {elapsed:.0f}s | NO TRADES", flush=True)

    except Exception as e:
        elapsed = time.time() - t0
        print(f" {elapsed:.0f}s | ERROR: {e}", flush=True)

    sys.stdout.flush()

print("\n\nSweep complete!")
print(f"Results in: {OUTPUT_CSV}")
