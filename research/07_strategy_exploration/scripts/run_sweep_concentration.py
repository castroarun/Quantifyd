"""
Concentration Sweep: Top 3 entry signals x PS5/10/15 x best exits
Tests whether smaller portfolios boost CAGR like the MQ system showed.
"""
import csv, os, sys, time, logging
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, enrich_with_indicators,
    load_enriched_cache, StrategyConfig, StrategyExplorer,
    entry_stoch_oversold, entry_rsi_oversold, entry_ema_crossover,
    entry_cci_oversold,
    rank_momentum_12m, rank_composite_momentum,
    exit_trailing_stop, exit_ath_drawdown, exit_time_based,
    make_combined_exit
)

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exploration_concentration.csv')
FIELDNAMES = ['name','cagr','sharpe','sortino','max_drawdown','calmar',
              'profit_factor','total_trades','win_rate','avg_win_pct',
              'avg_loss_pct','final_value','total_return_pct',
              'top3_pnl_pct','top3_symbols','cagr_ex_top3','exit_reason_counts']

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
START, END = '2005-01-01', '2025-12-31'
print('Loading data...')
enriched = load_enriched_cache(START, END)
if enriched is None:
    universe, price_data = preload_exploration_data(START, END)
    enriched = enrich_with_indicators(price_data)
    del price_data
else:
    universe, _ = preload_exploration_data(START, END)
print(f'Loaded {len(enriched)} stocks with data')
print('Data ready\n')

# Define exit combos - make_combined_exit expects list of (fn, kwargs_dict) tuples
def make_trail10():
    return make_combined_exit([
        (exit_trailing_stop, {'pct': 0.10}),
        (exit_time_based, {'max_days': 252}),
    ])

def make_trail15():
    return make_combined_exit([
        (exit_trailing_stop, {'pct': 0.15}),
        (exit_time_based, {'max_days': 252}),
    ])

def make_ath20():
    return make_combined_exit([
        (exit_ath_drawdown, {'pct': 0.20}),
        (exit_time_based, {'max_days': 365}),
    ])

# Build configs: 3 entries x 3 exits x 3 portfolio sizes = 27 configs
configs = []

for ps in [5, 10, 15]:
    # --- Stoch20 entry ---
    for exit_name, exit_maker in [('Trail15', make_trail15), ('ATH20', make_ath20)]:
        configs.append({
            'name': f'Stoch20_{exit_name}_PS{ps}',
            'entry_fn': lambda df, idx: entry_stoch_oversold(df, idx, threshold=20),
            'rank_fn': rank_momentum_12m,
            'exit_fn': exit_maker(),
            'config': StrategyConfig(
                name=f'Stoch20_{exit_name}_PS{ps}',
                start_date='2005-01-01', end_date='2025-12-31',
                portfolio_size=ps,
            ),
        })

    # --- RSI2 < 10 entry ---
    for exit_name, exit_maker in [('Trail15', make_trail15), ('ATH20', make_ath20)]:
        configs.append({
            'name': f'RSI2_10_{exit_name}_PS{ps}',
            'entry_fn': lambda df, idx: entry_rsi_oversold(df, idx, col='rsi_2', threshold=10),
            'rank_fn': rank_momentum_12m,
            'exit_fn': exit_maker(),
            'config': StrategyConfig(
                name=f'RSI2_10_{exit_name}_PS{ps}',
                start_date='2005-01-01', end_date='2025-12-31',
                portfolio_size=ps,
            ),
        })

    # --- EMA 20/50 crossover entry ---
    for exit_name, exit_maker in [('Trail10', make_trail10), ('Trail15', make_trail15), ('ATH20', make_ath20)]:
        configs.append({
            'name': f'EMA2050_{exit_name}_PS{ps}',
            'entry_fn': lambda df, idx: entry_ema_crossover(df, idx, fast=20, slow=50),
            'rank_fn': rank_momentum_12m,
            'exit_fn': exit_maker(),
            'config': StrategyConfig(
                name=f'EMA2050_{exit_name}_PS{ps}',
                start_date='2005-01-01', end_date='2025-12-31',
                portfolio_size=ps,
            ),
        })

    # --- CCI < -100 entry (best profit factor) ---
    for exit_name, exit_maker in [('ATH20', make_ath20)]:
        configs.append({
            'name': f'CCI_neg100_{exit_name}_PS{ps}',
            'entry_fn': lambda df, idx: entry_cci_oversold(df, idx, threshold=-100),
            'rank_fn': rank_momentum_12m,
            'exit_fn': exit_maker(),
            'config': StrategyConfig(
                name=f'CCI_neg100_{exit_name}_PS{ps}',
                start_date='2005-01-01', end_date='2025-12-31',
                portfolio_size=ps,
            ),
        })

todo = [c for c in configs if c['name'] not in done]
total = len(todo)
print(f'Running {total} configs ({len(done)} already done)...\n')

for i, strat in enumerate(todo):
    t0 = time.time()
    print(f'[{i+1}/{total}] {strat["name"]} ...', end='', flush=True)

    explorer = StrategyExplorer(universe, enriched, strat['config'])
    result = explorer.run(strat['entry_fn'], strat['rank_fn'], strat['exit_fn'])

    row = {
        'name': result.strategy_name,
        'cagr': round(result.cagr, 2),
        'sharpe': round(result.sharpe, 2),
        'sortino': round(result.sortino, 2),
        'max_drawdown': round(result.max_drawdown, 2),
        'calmar': round(result.calmar, 2),
        'profit_factor': round(result.profit_factor, 2),
        'total_trades': result.total_trades,
        'win_rate': round(result.win_rate, 1),
        'avg_win_pct': round(result.avg_win_pct, 1),
        'avg_loss_pct': round(result.avg_loss_pct, 1),
        'final_value': int(result.final_value),
        'total_return_pct': round(result.total_return_pct, 1),
        'top3_pnl_pct': round(result.top3_pnl_pct, 1),
        'top3_symbols': str(result.top3_symbols),
        'cagr_ex_top3': round(result.cagr_ex_top3, 2),
        'exit_reason_counts': str(result.exit_reason_counts),
    }
    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    elapsed = time.time() - t0
    print(f' {elapsed:.0f}s | CAGR={row["cagr"]:.2f}% MaxDD={row["max_drawdown"]:.1f}% Calmar={row["calmar"]:.2f} PF={row["profit_factor"]:.2f}')
    sys.stdout.flush()

print(f'\nDone! Results in {OUTPUT_CSV}')
