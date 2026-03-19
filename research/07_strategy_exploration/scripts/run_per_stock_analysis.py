"""Per-Stock Signal Analysis — Find stocks giving best returns across all signals.
Period: 2005-2025 (20 years).
Runs top strategies and aggregates P/L by stock to find consistent winners.
"""
import csv, os, sys, time, logging, json
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, load_enriched_cache,
    StrategyConfig, StrategyExplorer,
    entry_ema_crossover, entry_macd_crossover, entry_adx_trending,
    entry_supertrend_bullish, entry_donchian_breakout,
    entry_rsi_oversold, entry_bollinger_lower,
    entry_price_above_sma,
    rank_momentum_12m,
    exit_trailing_stop, exit_ath_drawdown, exit_time_based,
    make_combined_exit,
)
from services.technical_indicators import calc_ema
from collections import defaultdict
import pandas as pd

START, END = '2005-01-01', '2025-12-31'

OUTPUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'per_stock_analysis.json')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'per_stock_analysis.csv')

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

# Add missing EMAs
for sym, df in enriched.items():
    for period in [5, 8, 13, 34]:
        col = f'ema_{period}'
        if col not in df.columns:
            df[col] = calc_ema(df['close'], period)
    for fast, slow in [(9,21), (20,50)]:
        col_up = f'ema_cross_up_{fast}_{slow}'
        if col_up not in df.columns:
            f_col, s_col = f'ema_{fast}', f'ema_{slow}'
            if f_col in df.columns and s_col in df.columns:
                bull = df[f_col] > df[s_col]
                df[col_up] = bull & ~bull.shift(1).fillna(False)
print('Data ready', flush=True)

# ---- Entry functions ----
def entry_ema_20_50(df, idx): return entry_ema_crossover(df, idx, fast=20, slow=50)
def entry_ema_9_21(df, idx): return entry_ema_crossover(df, idx, fast=9, slow=21)
def entry_macd(df, idx): return entry_macd_crossover(df, idx)
def entry_adx25(df, idx): return entry_adx_trending(df, idx, threshold=25)
def entry_st(df, idx): return entry_supertrend_bullish(df, idx)
def entry_dc50(df, idx): return entry_donchian_breakout(df, idx, period=50)
def entry_rsi30(df, idx): return entry_rsi_oversold(df, idx, threshold=30)
def entry_sma200(df, idx): return entry_price_above_sma(df, idx, 200)

def entry_macd_sma200(df, idx):
    if 'macd_cross_up' not in df.columns or 'sma_200' not in df.columns:
        return False
    sma = df['sma_200'].iloc[idx]
    if pd.isna(sma):
        return False
    return bool(df['macd_cross_up'].iloc[idx]) and df['close'].iloc[idx] > sma

# Common exits
EXIT_TRAIL15 = make_combined_exit([(exit_trailing_stop, {'pct': 15}), (exit_time_based, {'max_days': 252})])
EXIT_ATH20 = make_combined_exit([(exit_ath_drawdown, {'pct': 20}), (exit_time_based, {'max_days': 365})])

# ---- Strategies to analyze ----
strategies = [
    ('EMA_20_50', entry_ema_20_50, EXIT_TRAIL15),
    ('EMA_9_21', entry_ema_9_21, EXIT_TRAIL15),
    ('MACD', entry_macd, EXIT_TRAIL15),
    ('MACD_SMA200', entry_macd_sma200, EXIT_TRAIL15),
    ('ADX25', entry_adx25, EXIT_TRAIL15),
    ('SuperTrend', entry_st, EXIT_TRAIL15),
    ('Donchian50', entry_dc50, EXIT_TRAIL15),
    ('RSI30', entry_rsi30, EXIT_TRAIL15),
    ('SMA200_Mom', entry_sma200, EXIT_ATH20),
]

# Aggregate: per stock, per strategy -> trades + P/L
stock_data = defaultdict(lambda: {
    'total_trades': 0, 'wins': 0, 'total_pnl_pct': 0,
    'strategies': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl_pct': 0}),
})

print(f'\nRunning {len(strategies)} strategies for per-stock analysis...\n', flush=True)

for strat_name, entry_fn, exit_fn in strategies:
    t0 = time.time()
    print(f'  {strat_name} ...', end='', flush=True)

    config = StrategyConfig(name=strat_name, start_date=START, end_date=END, portfolio_size=25)
    explorer = StrategyExplorer(universe, enriched, config)
    result = explorer.run(entry_fn, rank_momentum_12m, exit_fn)

    for trade in result.trade_list:
        sym = trade['symbol']
        pnl = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100
        is_win = pnl > 0

        stock_data[sym]['total_trades'] += 1
        stock_data[sym]['wins'] += int(is_win)
        stock_data[sym]['total_pnl_pct'] += pnl
        stock_data[sym]['strategies'][strat_name]['trades'] += 1
        stock_data[sym]['strategies'][strat_name]['wins'] += int(is_win)
        stock_data[sym]['strategies'][strat_name]['pnl_pct'] += pnl

    elapsed = time.time() - t0
    print(f' {elapsed:.0f}s | {result.total_trades} trades | CAGR={result.cagr:.1f}%')
    sys.stdout.flush()

# ---- Build output ----
print('\nComputing rankings...', flush=True)

results = []
for sym, data in stock_data.items():
    if data['total_trades'] < 5:
        continue  # skip stocks with too few trades

    win_rate = (data['wins'] / data['total_trades'] * 100) if data['total_trades'] > 0 else 0
    avg_pnl = data['total_pnl_pct'] / data['total_trades'] if data['total_trades'] > 0 else 0
    strat_count = len(data['strategies'])

    # Best and worst strategy for this stock
    best_strat = max(data['strategies'].items(), key=lambda x: x[1]['pnl_pct'])
    worst_strat = min(data['strategies'].items(), key=lambda x: x[1]['pnl_pct'])

    results.append({
        'symbol': sym,
        'total_trades': data['total_trades'],
        'wins': data['wins'],
        'win_rate': round(win_rate, 1),
        'total_pnl_pct': round(data['total_pnl_pct'], 1),
        'avg_pnl_pct': round(avg_pnl, 1),
        'strategies_active': strat_count,
        'best_strategy': best_strat[0],
        'best_strategy_pnl': round(best_strat[1]['pnl_pct'], 1),
        'worst_strategy': worst_strat[0],
        'worst_strategy_pnl': round(worst_strat[1]['pnl_pct'], 1),
        'per_strategy': {k: {
            'trades': v['trades'],
            'wins': v['wins'],
            'pnl_pct': round(v['pnl_pct'], 1),
        } for k, v in data['strategies'].items()},
    })

# Sort by total P/L descending
results.sort(key=lambda x: x['total_pnl_pct'], reverse=True)

# Save JSON (full data)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'Saved full data: {OUTPUT_JSON}')

# Save CSV (summary)
csv_fields = ['symbol', 'total_trades', 'wins', 'win_rate', 'total_pnl_pct',
              'avg_pnl_pct', 'strategies_active', 'best_strategy', 'best_strategy_pnl']
with open(OUTPUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
    w.writeheader()
    for r in results:
        w.writerow(r)
print(f'Saved summary: {OUTPUT_CSV}')

# Print top 20
print(f'\n{"="*80}')
print(f'TOP 20 STOCKS BY TOTAL P/L (across {len(strategies)} strategies, {START} to {END})')
print(f'{"="*80}')
print(f'{"Rank":<5} {"Symbol":<15} {"Trades":<8} {"WR%":<7} {"Total P/L%":<12} {"Avg P/L%":<10} {"Strategies":<10} {"Best Strategy":<20}')
print('-' * 87)
for i, r in enumerate(results[:20], 1):
    print(f'{i:<5} {r["symbol"]:<15} {r["total_trades"]:<8} {r["win_rate"]:<7} {r["total_pnl_pct"]:<12} {r["avg_pnl_pct"]:<10} {r["strategies_active"]:<10} {r["best_strategy"]:<20}')

# Bottom 10
print(f'\n{"="*80}')
print(f'BOTTOM 10 STOCKS BY TOTAL P/L')
print(f'{"="*80}')
for i, r in enumerate(results[-10:], len(results) - 9):
    print(f'{i:<5} {r["symbol"]:<15} {r["total_trades"]:<8} {r["win_rate"]:<7} {r["total_pnl_pct"]:<12} {r["avg_pnl_pct"]:<10} {r["strategies_active"]:<10}')

print(f'\nDone! {len(results)} stocks analyzed across {len(strategies)} strategies.')
