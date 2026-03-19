"""
Sweep 3: Technical Indicators Optimization
Tests 27 configs: EMA, RSI, SuperTrend, MACD, ADX, Weekly filter, and combos.
Base: PS30, HSL50, ATH20, EQ95 (proven optimal baseline).
Uses preloaded data + incremental CSV writes.
"""
import logging
logging.disable(logging.WARNING)

import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_sweep3_technical.csv')
FIELDNAMES = ['label', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'calmar',
              'total_trades', 'win_rate', 'final_value', 'total_return_pct', 'topups']

# ── Correct base params (proven optimal for PS30) ──
# NOTE: Do NOT set debt_reserve_pct explicitly — default 0.20 is correct
# (it funds Darvas topups which drive CAGR from 26% to 32%)
base = dict(
    portfolio_size=30,
    equity_allocation_pct=0.95,
    hard_stop_loss=0.50,
    rebalance_ath_drawdown=0.20,
)

# ── Build 27 configs ──
configs = []

# 1. Baseline: no technical filter
configs.append(('BASELINE_NO_TECH', {**base}))

# 2. EMA filters (6 configs)
for fast, slow in [(9, 21), (10, 30), (20, 50)]:
    for exit_type in ['crossover', 'price_below']:
        configs.append((f'EMA_{fast}_{slow}_exit_{exit_type}', {
            **base, 'use_technical_filter': True,
            'use_ema_entry': True, 'use_ema_exit': True,
            'ema_fast': fast, 'ema_slow': slow, 'ema_exit_type': exit_type,
        }))

# 3. RSI filters (6 configs)
for period in [7, 14]:
    for overbought in [75, 80, 85]:
        configs.append((f'RSI_{period}_ob{overbought}', {
            **base, 'use_technical_filter': True,
            'use_rsi_filter': True, 'use_rsi_exit': True,
            'rsi_period': period, 'rsi_exit_overbought': overbought,
            'rsi_min_entry': 40, 'rsi_max_entry': 70,
        }))

# 4. SuperTrend filters (6 configs)
for atr_period in [7, 10, 14]:
    for mult in [2.0, 3.0]:
        configs.append((f'STREND_atr{atr_period}_m{mult}', {
            **base, 'use_technical_filter': True,
            'use_supertrend': True, 'supertrend_atr': atr_period,
            'supertrend_mult': mult, 'supertrend_entry_bullish': True,
            'supertrend_exit_flip': True,
        }))

# 5. MACD (1 config)
configs.append(('MACD_12_26_9', {
    **base, 'use_technical_filter': True,
    'use_macd': True, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
    'require_macd_positive': True, 'require_macd_above_signal': True,
}))

# 6. ADX (3 configs)
for min_trend in [20, 25, 30]:
    configs.append((f'ADX_min{min_trend}', {
        **base, 'use_technical_filter': True,
        'use_adx': True, 'adx_period': 14, 'adx_min_trend': min_trend,
        'require_plus_di_above': True,
    }))

# 7. Weekly filter (1 config)
configs.append(('WEEKLY_EMA20', {
    **base, 'use_technical_filter': True,
    'use_weekly_filter': True, 'weekly_ema_period': 20,
    'require_weekly_above_ema': True,
}))

# 8. Combos (3 configs)
configs.append(('COMBO_EMA20_50_RSI14', {
    **base, 'use_technical_filter': True,
    'use_ema_entry': True, 'use_ema_exit': True,
    'ema_fast': 20, 'ema_slow': 50, 'ema_exit_type': 'price_below',
    'use_rsi_filter': True, 'rsi_period': 14, 'rsi_min_entry': 40, 'rsi_max_entry': 70,
}))

configs.append(('COMBO_EMA9_21_STREND', {
    **base, 'use_technical_filter': True,
    'use_ema_entry': True, 'use_ema_exit': True, 'ema_fast': 9, 'ema_slow': 21,
    'use_supertrend': True, 'supertrend_atr': 10, 'supertrend_mult': 3.0,
    'supertrend_exit_flip': True,
}))

configs.append(('COMBO_MACD_RSI_WEEKLY', {
    **base, 'use_technical_filter': True,
    'use_macd': True, 'use_rsi_filter': True, 'use_weekly_filter': True,
    'rsi_period': 14, 'rsi_min_entry': 40, 'rsi_max_entry': 70,
}))


if __name__ == '__main__':
    total = len(configs)
    print(f'=== Sweep 3: Technical Indicators | {total} configs ===')
    print(f'Base: PS30, HSL50, ATH20, EQ95')
    print(f'Output: {OUTPUT_CSV}')
    sys.stdout.flush()

    # Skip already-completed configs
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}
        print(f'Skipping {len(done)} already-completed configs')
    else:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Preload data once
    print('Preloading universe + price data...', end='', flush=True)
    t_load = time.time()
    universe, price_data = MQBacktestEngine.preload_data(MQBacktestConfig())
    print(f' {time.time() - t_load:.0f}s')
    sys.stdout.flush()

    t0 = time.time()
    completed = 0
    for i, (label, params) in enumerate(configs, 1):
        if label in done:
            print(f'[{i:2d}/{total}] {label} ... SKIP (already done)')
            continue

        t1 = time.time()
        print(f'[{i:2d}/{total}] {label} ...', end='', flush=True)

        try:
            cfg = MQBacktestConfig()
            for k, v in params.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

            engine = MQBacktestEngine(cfg,
                preloaded_universe=universe,
                preloaded_price_data=price_data)
            r = engine.run()

            row = dict(
                label=label,
                cagr=round(r.cagr, 2),
                sharpe=round(r.sharpe_ratio, 2),
                sortino=round(r.sortino_ratio, 2),
                max_drawdown=round(r.max_drawdown, 2),
                calmar=round(r.calmar_ratio, 2),
                total_trades=r.total_trades,
                win_rate=round(r.win_rate, 1),
                final_value=round(r.final_value, 2),
                total_return_pct=round(r.total_return_pct, 2),
                topups=r.total_topups,
            )

            # Write immediately
            with open(OUTPUT_CSV, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

            elapsed = time.time() - t1
            completed += 1
            print(f' {elapsed:.0f}s | CAGR={row["cagr"]:.2f}% Sharpe={row["sharpe"]:.2f} MaxDD={row["max_drawdown"]:.2f}%')

        except Exception as e:
            elapsed = time.time() - t1
            print(f' ERROR {elapsed:.0f}s: {e}')

        sys.stdout.flush()

    tt = time.time() - t0
    print(f'=== Done: {completed} configs in {tt:.0f}s ({tt/60:.1f}min) ===')
    print(f'Results: {OUTPUT_CSV}')
