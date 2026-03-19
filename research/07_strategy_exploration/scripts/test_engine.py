"""Quick test of the strategy backtest engine."""
import sys, os, time, logging
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, enrich_with_indicators,
    StrategyConfig, StrategyExplorer,
    entry_ema_crossover, rank_momentum_12m,
    exit_trailing_stop, make_combined_exit, exit_time_based,
)

# Use 5 years instead of 10 for faster test
t0 = time.time()
print('Loading data...', flush=True)
universe, price_data = preload_exploration_data('2020-01-01', '2025-12-31')
print(f'Data loaded: {len(price_data)} stocks in {time.time()-t0:.0f}s', flush=True)

t1 = time.time()
print('Computing indicators...', flush=True)
enriched = enrich_with_indicators(price_data)
print(f'Indicators computed: {len(enriched)} stocks in {time.time()-t1:.0f}s', flush=True)

config = StrategyConfig(
    name='TEST_EMA_20_50',
    start_date='2020-01-01',
    end_date='2025-12-31',
    portfolio_size=25,
    rebalance_freq='monthly',
)

exit_fn = make_combined_exit([
    (exit_trailing_stop, {'pct': 15}),
    (exit_time_based, {'max_days': 180}),
])

t2 = time.time()
print('Running backtest...', flush=True)
explorer = StrategyExplorer(universe, enriched, config)
result = explorer.run(
    entry_fn=lambda df, idx: entry_ema_crossover(df, idx, fast=20, slow=50),
    rank_fn=rank_momentum_12m,
    exit_fn=exit_fn,
)
print(f'Backtest complete in {time.time()-t2:.0f}s', flush=True)
print(f'CAGR={result.cagr:.2f}% MaxDD={result.max_drawdown:.2f}% Calmar={result.calmar:.2f}', flush=True)
print(f'Sharpe={result.sharpe:.2f} Sortino={result.sortino:.2f} PF={result.profit_factor:.2f}', flush=True)
print(f'Trades={result.total_trades} WinRate={result.win_rate:.1f}% AvgWin={result.avg_win_pct:.1f}% AvgLoss={result.avg_loss_pct:.1f}%', flush=True)
print(f'Top3 P/L share: {result.top3_pnl_pct:.1f}% ({result.top3_symbols})', flush=True)
print(f'CAGR ex top3: {result.cagr_ex_top3:.2f}%', flush=True)
print(f'Exit reasons: {result.exit_reason_counts}', flush=True)
print(f'Total time: {time.time()-t0:.0f}s', flush=True)
