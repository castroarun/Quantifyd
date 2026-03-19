"""
Sweep 5B: Daily ATH Drawdown with Immediate Replacement
Compare daily ATH exit (capital sits idle) vs daily ATH exit + immediate replacement
"""
import sys, os, csv, time, logging

logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_sweep5b_replacement.csv')
FIELDNAMES = ['label','cagr','sharpe','sortino','max_drawdown','calmar',
              'total_trades','win_rate','final_value','total_return_pct','topups']

# Skip already-done configs
done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {row['label'] for row in csv.DictReader(f)}
    print(f'Skipping {len(done)} already-completed configs')
else:
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

# Base params (PS30 baseline)
base = dict(
    portfolio_size=30,
    equity_allocation_pct=0.95,
    hard_stop_loss=0.50,
    rebalance_ath_drawdown=0.20,
)

configs = [
    # Baseline: 6-month rebalance only (no daily ATH, no replacement)
    ('BASELINE_6M_ONLY', {}),
    # Daily ATH at various thresholds WITHOUT replacement
    ('DAILY_ATH15_NO_REPLACE', {'daily_ath_drawdown_exit': True, 'rebalance_ath_drawdown': 0.15}),
    ('DAILY_ATH20_NO_REPLACE', {'daily_ath_drawdown_exit': True, 'rebalance_ath_drawdown': 0.20}),
    ('DAILY_ATH25_NO_REPLACE', {'daily_ath_drawdown_exit': True, 'rebalance_ath_drawdown': 0.25}),
    ('DAILY_ATH30_NO_REPLACE', {'daily_ath_drawdown_exit': True, 'rebalance_ath_drawdown': 0.30}),
    # Daily ATH at various thresholds WITH immediate replacement
    ('DAILY_ATH15_REPLACE', {'daily_ath_drawdown_exit': True, 'rebalance_ath_drawdown': 0.15, 'immediate_replacement': True}),
    ('DAILY_ATH20_REPLACE', {'daily_ath_drawdown_exit': True, 'rebalance_ath_drawdown': 0.20, 'immediate_replacement': True}),
    ('DAILY_ATH25_REPLACE', {'daily_ath_drawdown_exit': True, 'rebalance_ath_drawdown': 0.25, 'immediate_replacement': True}),
    ('DAILY_ATH30_REPLACE', {'daily_ath_drawdown_exit': True, 'rebalance_ath_drawdown': 0.30, 'immediate_replacement': True}),
]

print(f'=== Sweep 5B: Daily ATH + Immediate Replacement | {len(configs)} configs ===')
print(f'Base: PS30, HSL50, EQ95')
print(f'Output: {OUTPUT_CSV}')

# Preload data once
print('Preloading universe + price data...', end='', flush=True)
t0 = time.time()
universe, price_data = MQBacktestEngine.preload_data(MQBacktestConfig())
print(f' {time.time()-t0:.0f}s')

total = len(configs)
for i, (label, overrides) in enumerate(configs, 1):
    if label in done:
        print(f'[{i}/{total}] {label} ... SKIPPED (already done)')
        continue

    print(f'[{i}/{total}] {label} ...', end='', flush=True)
    t1 = time.time()

    params = {**base, **overrides}
    config = MQBacktestConfig(**params)
    engine = MQBacktestEngine(config, preloaded_universe=universe, preloaded_price_data=price_data)
    result = engine.run()

    elapsed = time.time() - t1

    row = {
        'label': label,
        'cagr': result.cagr,
        'sharpe': result.sharpe_ratio,
        'sortino': result.sortino_ratio,
        'max_drawdown': result.max_drawdown,
        'calmar': result.calmar_ratio,
        'total_trades': result.total_trades,
        'win_rate': result.win_rate,
        'final_value': round(result.final_value, 2),
        'total_return_pct': result.total_return_pct,
        'topups': result.total_topups,
    }

    # Write incrementally
    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    print(f' {elapsed:.0f}s | CAGR={row["cagr"]:.2f}% Sharpe={row["sharpe"]:.2f} MaxDD={row["max_drawdown"]:.2f}% Trades={row["total_trades"]}')
    sys.stdout.flush()

print('\n=== DONE ===')
print(f'Results saved to: {OUTPUT_CSV}')
