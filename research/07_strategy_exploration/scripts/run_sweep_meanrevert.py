"""Phase 1 Category 2: Mean Reversion in Uptrend strategies (12 configs)."""
import csv, os, sys, time, logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, enrich_with_indicators,
    load_enriched_cache, save_enriched_cache,
    StrategyConfig, StrategyExplorer,
    # Entry functions
    entry_rsi_oversold, entry_bollinger_lower, entry_keltner_lower,
    entry_stoch_oversold, entry_williams_r_oversold,
    entry_cci_oversold, entry_mfi_oversold,
    entry_weekly_trend_daily_pullback,
    # Ranking functions
    rank_momentum_12m, rank_momentum_6m, rank_rsi_strength,
    rank_distance_from_high, rank_composite_momentum,
    # Exit functions
    exit_trailing_stop, exit_ath_drawdown, exit_time_based,
    exit_take_profit, exit_fixed_stop_loss,
    exit_rsi_overbought, exit_rsi2_overbought,
    exit_bollinger_mid, exit_bollinger_upper,
    exit_keltner_mid, exit_cci_overbought,
    exit_stoch_overbought, exit_williams_r_overbought,
    exit_mfi_overbought,
    make_combined_exit,
)

START, END = '2015-01-01', '2025-12-31'
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exploration_meanrevert.csv')
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

# Mean reversion exits: bounce to middle band / take profit / time limit
def mr_exit_bb():
    """Exit at BB mid + trailing SL + time limit."""
    return make_combined_exit([
        (exit_bollinger_mid, {}),
        (exit_fixed_stop_loss, {'pct': 8}),
        (exit_time_based, {'max_days': 60}),
    ])

def mr_exit_kc():
    """Exit at KC mid + SL + time."""
    return make_combined_exit([
        (exit_keltner_mid, {}),
        (exit_fixed_stop_loss, {'pct': 8}),
        (exit_time_based, {'max_days': 60}),
    ])

def mr_exit_rsi():
    """Exit when RSI goes overbought + SL + time."""
    return make_combined_exit([
        (exit_rsi_overbought, {'threshold': 70}),
        (exit_fixed_stop_loss, {'pct': 8}),
        (exit_time_based, {'max_days': 60}),
    ])

def mr_exit_rsi2():
    """Exit when RSI(2) goes overbought + SL + time."""
    return make_combined_exit([
        (exit_rsi2_overbought, {'threshold': 90}),
        (exit_fixed_stop_loss, {'pct': 5}),
        (exit_time_based, {'max_days': 15}),
    ])

def mr_exit_tp():
    """Exit at take profit + SL + time."""
    return make_combined_exit([
        (exit_take_profit, {'pct': 15}),
        (exit_fixed_stop_loss, {'pct': 5}),
        (exit_time_based, {'max_days': 30}),
    ])

def base_config(name, ps=25, rebal='monthly'):
    return StrategyConfig(name=name, start_date=START, end_date=END,
                          portfolio_size=ps, rebalance_freq=rebal)

# ===== 12 Mean Reversion Strategies =====
configs = [
    # 1. RSI(14) < 30 in uptrend + exit RSI > 70
    {
        'name': 'MR_RSI30_ExitRSI70',
        'entry_fn': lambda df, idx: entry_rsi_oversold(df, idx, 'rsi_14', 30),
        'rank_fn': rank_momentum_12m,
        'exit_fn': mr_exit_rsi(),
        'config': base_config('MR_RSI30_ExitRSI70'),
    },
    # 2. RSI(2) < 10 (aggressive) + exit RSI(2) > 90
    {
        'name': 'MR_RSI2_10_Exit90',
        'entry_fn': lambda df, idx: entry_rsi_oversold(df, idx, 'rsi_2', 10),
        'rank_fn': rank_momentum_12m,
        'exit_fn': mr_exit_rsi2(),
        'config': base_config('MR_RSI2_10_Exit90'),
    },
    # 3. Bollinger lower band + exit at BB mid
    {
        'name': 'MR_BB_Lower_ExitMid',
        'entry_fn': entry_bollinger_lower,
        'rank_fn': rank_momentum_12m,
        'exit_fn': mr_exit_bb(),
        'config': base_config('MR_BB_Lower_ExitMid'),
    },
    # 4. Keltner lower + exit at KC mid
    {
        'name': 'MR_KC_Lower_ExitMid',
        'entry_fn': entry_keltner_lower,
        'rank_fn': rank_momentum_12m,
        'exit_fn': mr_exit_kc(),
        'config': base_config('MR_KC_Lower_ExitMid'),
    },
    # 5. Stochastic < 20 + exit Stoch > 80
    {
        'name': 'MR_Stoch20_Exit80',
        'entry_fn': lambda df, idx: entry_stoch_oversold(df, idx, 20),
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_stoch_overbought, {'threshold': 80}),
            (exit_fixed_stop_loss, {'pct': 8}),
            (exit_time_based, {'max_days': 60}),
        ]),
        'config': base_config('MR_Stoch20_Exit80'),
    },
    # 6. Williams %R < -80 + exit WR > -20
    {
        'name': 'MR_WR80_Exit20',
        'entry_fn': lambda df, idx: entry_williams_r_oversold(df, idx, -80),
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_williams_r_overbought, {'threshold': -20}),
            (exit_fixed_stop_loss, {'pct': 8}),
            (exit_time_based, {'max_days': 60}),
        ]),
        'config': base_config('MR_WR80_Exit20'),
    },
    # 7. CCI < -100 + exit CCI > 100
    {
        'name': 'MR_CCI100_Exit100',
        'entry_fn': lambda df, idx: entry_cci_oversold(df, idx, -100),
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_cci_overbought, {'threshold': 100}),
            (exit_fixed_stop_loss, {'pct': 8}),
            (exit_time_based, {'max_days': 60}),
        ]),
        'config': base_config('MR_CCI100_Exit100'),
    },
    # 8. MFI < 20 + exit MFI > 80
    {
        'name': 'MR_MFI20_Exit80',
        'entry_fn': lambda df, idx: entry_mfi_oversold(df, idx, 20),
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_mfi_overbought, {'threshold': 80}),
            (exit_fixed_stop_loss, {'pct': 8}),
            (exit_time_based, {'max_days': 60}),
        ]),
        'config': base_config('MR_MFI20_Exit80'),
    },
    # 9. Weekly uptrend + daily RSI pullback + trailing stop exit
    {
        'name': 'MR_WeeklyUp_DailyPull_Trail15',
        'entry_fn': lambda df, idx: entry_weekly_trend_daily_pullback(df, idx, 40),
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_trailing_stop, {'pct': 15}),
            (exit_time_based, {'max_days': 120}),
        ]),
        'config': base_config('MR_WeeklyUp_DailyPull_Trail15'),
    },
    # 10. Bollinger lower + take profit 15% (swing trade style)
    {
        'name': 'MR_BB_Lower_TP15',
        'entry_fn': entry_bollinger_lower,
        'rank_fn': rank_momentum_12m,
        'exit_fn': mr_exit_tp(),
        'config': base_config('MR_BB_Lower_TP15'),
    },
    # 11. RSI(14) < 25 (deeper oversold) + composite ranking + BB upper exit
    {
        'name': 'MR_RSI25_BBUpper',
        'entry_fn': lambda df, idx: entry_rsi_oversold(df, idx, 'rsi_14', 25),
        'rank_fn': rank_composite_momentum,
        'exit_fn': make_combined_exit([
            (exit_bollinger_upper, {}),
            (exit_fixed_stop_loss, {'pct': 10}),
            (exit_time_based, {'max_days': 90}),
        ]),
        'config': base_config('MR_RSI25_BBUpper'),
    },
    # 12. Keltner lower + distance-from-high ranking + trailing stop
    {
        'name': 'MR_KC_DistRank_Trail10',
        'entry_fn': entry_keltner_lower,
        'rank_fn': rank_distance_from_high,
        'exit_fn': make_combined_exit([
            (exit_trailing_stop, {'pct': 10}),
            (exit_time_based, {'max_days': 90}),
        ]),
        'config': base_config('MR_KC_DistRank_Trail10'),
    },
]

# Run strategies
remaining = [c for c in configs if c['name'] not in done]
total = len(remaining)
print(f'\nRunning {total} mean reversion strategies...', flush=True)

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
