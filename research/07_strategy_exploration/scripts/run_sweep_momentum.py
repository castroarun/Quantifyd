"""Phase 1 Category 1: Trend Following / Breakout strategies (12 configs)."""
import csv, os, sys, time, logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, enrich_with_indicators,
    load_enriched_cache, save_enriched_cache,
    StrategyConfig, StrategyExplorer,
    # Entry functions
    entry_ema_crossover, entry_price_above_sma,
    entry_supertrend_bullish, entry_donchian_breakout,
    entry_macd_crossover, entry_adx_trending,
    entry_volume_breakout, entry_52w_high_proximity,
    entry_bollinger_squeeze_breakout,
    # Ranking functions
    rank_momentum_12m, rank_momentum_6m, rank_rsi_strength,
    rank_distance_from_high, rank_composite_momentum,
    # Exit functions
    exit_trailing_stop, exit_ath_drawdown, exit_time_based,
    exit_indicator_reversal, exit_fixed_stop_loss,
    make_combined_exit,
)

START, END = '2015-01-01', '2025-12-31'
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exploration_momentum.csv')
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
print('Loading data...', flush=True)
t0 = time.time()
enriched = load_enriched_cache(START, END)
if enriched is None:
    print('ERROR: No enriched cache found. Run run_enrichment.py first.', flush=True)
    sys.exit(1)
universe, _ = preload_exploration_data(START, END)
print(f'Data ready in {time.time()-t0:.0f}s', flush=True)

# Standard exit: 15% trailing stop + 180d time limit
std_exit = make_combined_exit([
    (exit_trailing_stop, {'pct': 15}),
    (exit_time_based, {'max_days': 180}),
])

# ATH drawdown exit: 20% from peak + 180d
ath_exit = make_combined_exit([
    (exit_ath_drawdown, {'pct': 20}),
    (exit_time_based, {'max_days': 180}),
])

# Tight trailing: 10% + 120d
tight_exit = make_combined_exit([
    (exit_trailing_stop, {'pct': 10}),
    (exit_time_based, {'max_days': 120}),
])

# Wide trailing: 20% + 252d (1yr hold max)
wide_exit = make_combined_exit([
    (exit_trailing_stop, {'pct': 20}),
    (exit_time_based, {'max_days': 252}),
])

def base_config(name, ps=25, rebal='monthly'):
    return StrategyConfig(name=name, start_date=START, end_date=END,
                          portfolio_size=ps, rebalance_freq=rebal)

# ===== 12 Momentum/Trend Strategies =====
configs = [
    # 1. EMA 20/50 crossover + momentum ranking + trailing stop
    {
        'name': 'MOM_EMA2050_Trail15',
        'entry_fn': lambda df, idx: entry_ema_crossover(df, idx, fast=20, slow=50),
        'rank_fn': rank_momentum_12m,
        'exit_fn': std_exit,
        'config': base_config('MOM_EMA2050_Trail15'),
    },
    # 2. EMA 10/30 crossover (faster) + composite momentum + ATH exit
    {
        'name': 'MOM_EMA1030_ATH20',
        'entry_fn': lambda df, idx: entry_ema_crossover(df, idx, fast=10, slow=30),
        'rank_fn': rank_composite_momentum,
        'exit_fn': ath_exit,
        'config': base_config('MOM_EMA1030_ATH20'),
    },
    # 3. EMA 50/200 golden cross + 12m mom + wide trailing
    {
        'name': 'MOM_GoldenCross_Wide20',
        'entry_fn': lambda df, idx: entry_ema_crossover(df, idx, fast=50, slow=200),
        'rank_fn': rank_momentum_12m,
        'exit_fn': wide_exit,
        'config': base_config('MOM_GoldenCross_Wide20', rebal='quarterly'),
    },
    # 4. SuperTrend flip + momentum + trailing
    {
        'name': 'MOM_ST10_3_Trail15',
        'entry_fn': lambda df, idx: entry_supertrend_bullish(df, idx, 10, 3),
        'rank_fn': rank_momentum_12m,
        'exit_fn': std_exit,
        'config': base_config('MOM_ST10_3_Trail15'),
    },
    # 5. SuperTrend(7,2) faster + 6m momentum + tight trailing
    {
        'name': 'MOM_ST7_2_Tight10',
        'entry_fn': lambda df, idx: entry_supertrend_bullish(df, idx, 7, 2),
        'rank_fn': rank_momentum_6m,
        'exit_fn': tight_exit,
        'config': base_config('MOM_ST7_2_Tight10'),
    },
    # 6. Donchian 50-day breakout + 12m momentum + trailing
    {
        'name': 'MOM_DC50_Trail15',
        'entry_fn': lambda df, idx: entry_donchian_breakout(df, idx, 50),
        'rank_fn': rank_momentum_12m,
        'exit_fn': std_exit,
        'config': base_config('MOM_DC50_Trail15'),
    },
    # 7. Donchian 20-day breakout + composite mom + ATH exit
    {
        'name': 'MOM_DC20_ATH20',
        'entry_fn': lambda df, idx: entry_donchian_breakout(df, idx, 20),
        'rank_fn': rank_composite_momentum,
        'exit_fn': ath_exit,
        'config': base_config('MOM_DC20_ATH20'),
    },
    # 8. MACD crossover + RSI strength ranking + trailing
    {
        'name': 'MOM_MACD_RSIRank_Trail15',
        'entry_fn': entry_macd_crossover,
        'rank_fn': rank_rsi_strength,
        'exit_fn': std_exit,
        'config': base_config('MOM_MACD_RSIRank_Trail15'),
    },
    # 9. ADX trending (>25, DI+ > DI-) + momentum + trailing
    {
        'name': 'MOM_ADX25_Trail15',
        'entry_fn': lambda df, idx: entry_adx_trending(df, idx, 25),
        'rank_fn': rank_momentum_12m,
        'exit_fn': std_exit,
        'config': base_config('MOM_ADX25_Trail15'),
    },
    # 10. Volume breakout (2x avg) + 12m momentum + wide trailing
    {
        'name': 'MOM_VolBreak2x_Wide20',
        'entry_fn': lambda df, idx: entry_volume_breakout(df, idx, 2.0),
        'rank_fn': rank_momentum_12m,
        'exit_fn': wide_exit,
        'config': base_config('MOM_VolBreak2x_Wide20'),
    },
    # 11. 52-week high proximity (>90%) + momentum + trailing
    {
        'name': 'MOM_52WHigh90_Trail15',
        'entry_fn': lambda df, idx: entry_52w_high_proximity(df, idx, 0.90),
        'rank_fn': rank_momentum_12m,
        'exit_fn': std_exit,
        'config': base_config('MOM_52WHigh90_Trail15'),
    },
    # 12. Bollinger squeeze breakout + momentum + ATH exit
    {
        'name': 'MOM_BBSqueeze_ATH20',
        'entry_fn': entry_bollinger_squeeze_breakout,
        'rank_fn': rank_momentum_12m,
        'exit_fn': ath_exit,
        'config': base_config('MOM_BBSqueeze_ATH20'),
    },
]

# Run strategies
remaining = [c for c in configs if c['name'] not in done]
total = len(remaining)
print(f'\nRunning {total} momentum strategies...', flush=True)

for i, strat in enumerate(remaining):
    t1 = time.time()
    print(f"[{i+1}/{total}] {strat['name']} ...", end='', flush=True)

    explorer = StrategyExplorer(universe, enriched, strat['config'])
    result = explorer.run(strat['entry_fn'], strat['rank_fn'], strat['exit_fn'])

    row = {
        'name': result.strategy_name,
        'cagr': f'{result.cagr:.2f}',
        'sharpe': f'{result.sharpe:.2f}',
        'sortino': f'{result.sortino:.2f}',
        'max_drawdown': f'{result.max_drawdown:.2f}',
        'calmar': f'{result.calmar:.2f}',
        'profit_factor': f'{result.profit_factor:.2f}',
        'total_trades': result.total_trades,
        'win_rate': f'{result.win_rate:.1f}',
        'avg_win_pct': f'{result.avg_win_pct:.1f}',
        'avg_loss_pct': f'{result.avg_loss_pct:.1f}',
        'final_value': f'{result.final_value:.0f}',
        'total_return_pct': f'{result.total_return_pct:.1f}',
        'top3_pnl_pct': f'{result.top3_pnl_pct:.1f}',
        'top3_symbols': str(result.top3_symbols),
        'cagr_ex_top3': f'{result.cagr_ex_top3:.2f}',
        'exit_reason_counts': str(result.exit_reason_counts),
    }

    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    elapsed = time.time() - t1
    print(f' {elapsed:.0f}s | CAGR={result.cagr:.2f}% MaxDD={result.max_drawdown:.1f}% Calmar={result.calmar:.2f} PF={result.profit_factor:.2f}', flush=True)

print(f'\nDone! Results in {OUTPUT_CSV}')
print(f'Total sweep time: {time.time()-t0:.0f}s', flush=True)
