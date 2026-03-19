"""EMA Crossover Grid Sweep — 48 configs (8 pairs x 6 exit variants)
Period: 2005-2025 (20 years), PS25, monthly rebalance.
"""
import csv, os, sys, time, logging, functools
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, load_enriched_cache,
    StrategyConfig, StrategyExplorer,
    entry_ema_crossover,
    rank_momentum_12m, rank_composite_momentum,
    exit_trailing_stop, exit_ath_drawdown, exit_fixed_stop_loss,
    exit_take_profit, exit_time_based,
    make_combined_exit,
)
from services.technical_indicators import calc_ema
import pandas as pd

START, END = '2005-01-01', '2025-12-31'

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exploration_ema_crossover.csv')
FIELDNAMES = ['name', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'calmar',
              'profit_factor', 'total_trades', 'win_rate', 'avg_win_pct',
              'avg_loss_pct', 'final_value', 'total_return_pct',
              'top3_pnl_pct', 'top3_symbols', 'cagr_ex_top3', 'exit_reason_counts']

# Skip already done
done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {row['name'] for row in csv.DictReader(f)}
    print(f'Skipping {len(done)} already-completed configs')
else:
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

# Load data
print('Loading data...', flush=True)
enriched = load_enriched_cache(START, END)
if enriched is None:
    universe, price_data = preload_exploration_data(START, END)
    from services.strategy_backtest import enrich_with_indicators, save_enriched_cache
    enriched = enrich_with_indicators(price_data)
    save_enriched_cache(enriched, START, END)
    del price_data
else:
    universe, _ = preload_exploration_data(START, END)

# Add missing EMA periods for our sweep
print('Adding missing EMA columns...', flush=True)
EXTRA_EMAS = [5, 8, 13, 34]
EMA_PAIRS = [(5,20), (8,21), (9,21), (10,30), (13,34), (20,50), (50,200), (10,50)]
for sym, df in enriched.items():
    for period in EXTRA_EMAS:
        col = f'ema_{period}'
        if col not in df.columns:
            df[col] = calc_ema(df['close'], period)
    for fast, slow in EMA_PAIRS:
        col_up = f'ema_cross_up_{fast}_{slow}'
        if col_up not in df.columns:
            f_col, s_col = f'ema_{fast}', f'ema_{slow}'
            if f_col in df.columns and s_col in df.columns:
                bull = df[f_col] > df[s_col]
                df[col_up] = bull & ~bull.shift(1).fillna(False)
                df[f'ema_cross_down_{fast}_{slow}'] = ~bull & bull.shift(1).fillna(True)
                df[f'ema_bullish_{fast}_{slow}'] = bull
print('EMA columns ready', flush=True)

# Entry function factory
def make_ema_entry(fast, slow):
    def fn(df, idx):
        return entry_ema_crossover(df, idx, fast=fast, slow=slow)
    fn.__name__ = f'entry_ema_{fast}_{slow}'
    return fn

# Exit variants
EXIT_VARIANTS = {
    'Trail10':  make_combined_exit([(exit_trailing_stop, {'pct': 10}), (exit_time_based, {'max_days': 252})]),
    'Trail15':  make_combined_exit([(exit_trailing_stop, {'pct': 15}), (exit_time_based, {'max_days': 252})]),
    'Trail20':  make_combined_exit([(exit_trailing_stop, {'pct': 20}), (exit_time_based, {'max_days': 252})]),
    'ATH15':    make_combined_exit([(exit_ath_drawdown, {'pct': 15}), (exit_time_based, {'max_days': 365})]),
    'ATH20':    make_combined_exit([(exit_ath_drawdown, {'pct': 20}), (exit_time_based, {'max_days': 365})]),
    'SL5_TP30': make_combined_exit([(exit_fixed_stop_loss, {'pct': 5}), (exit_take_profit, {'pct': 30}), (exit_time_based, {'max_days': 180})]),
}

# Build configs: 8 pairs x 6 exits = 48
configs = []
for fast, slow in EMA_PAIRS:
    entry_fn = make_ema_entry(fast, slow)
    for exit_name, exit_fn in EXIT_VARIANTS.items():
        name = f'EMA_{fast}_{slow}_{exit_name}'
        configs.append({
            'name': name,
            'entry_fn': entry_fn,
            'rank_fn': rank_momentum_12m,
            'exit_fn': exit_fn,
            'config': StrategyConfig(name=name, start_date=START, end_date=END, portfolio_size=25),
        })

total = len([c for c in configs if c['name'] not in done])
print(f'\nRunning {total} configs ({len(done)} already done)...\n', flush=True)

i = 0
for strat in configs:
    if strat['name'] in done:
        continue
    i += 1
    t0 = time.time()
    print(f'[{i}/{total}] {strat["name"]} ...', end='', flush=True)

    explorer = StrategyExplorer(universe, enriched, strat['config'])
    result = explorer.run(strat['entry_fn'], strat['rank_fn'], strat['exit_fn'])

    row = {
        'name': result.strategy_name, 'cagr': round(result.cagr, 2),
        'sharpe': round(result.sharpe, 2), 'sortino': round(result.sortino, 2),
        'max_drawdown': round(result.max_drawdown, 2), 'calmar': round(result.calmar, 2),
        'profit_factor': round(result.profit_factor, 2),
        'total_trades': result.total_trades, 'win_rate': round(result.win_rate, 1),
        'avg_win_pct': round(result.avg_win_pct, 1), 'avg_loss_pct': round(result.avg_loss_pct, 1),
        'final_value': round(result.final_value), 'total_return_pct': round(result.total_return_pct, 1),
        'top3_pnl_pct': round(result.top3_pnl_pct, 1), 'top3_symbols': str(result.top3_symbols),
        'cagr_ex_top3': round(result.cagr_ex_top3, 2),
        'exit_reason_counts': str(result.exit_reason_counts),
    }
    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    elapsed = time.time() - t0
    print(f' {elapsed:.0f}s | CAGR={row["cagr"]:.2f}% Calmar={row["calmar"]:.2f} DD={row["max_drawdown"]:.1f}%')
    sys.stdout.flush()

print(f'\nDone! Results in {OUTPUT_CSV}')
