"""Extract sample trades from top 5 strategies for final report."""
import json, os, sys, time, logging, random
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, load_enriched_cache,
    StrategyConfig, StrategyExplorer,
    entry_macd_crossover, entry_adx_trending,
    entry_price_above_sma,
    rank_momentum_12m, rank_composite_momentum, rank_rsi_strength,
    exit_trailing_stop, exit_ath_drawdown, exit_time_based,
    exit_fixed_stop_loss,
    exit_indicator_reversal,
    make_combined_exit,
)
import pandas as pd

START, END = '2015-01-01', '2025-12-31'

# Load data
print('Loading data...', flush=True)
enriched = load_enriched_cache(START, END)
universe, _ = preload_exploration_data(START, END)
print('Data ready', flush=True)

# Entry functions
def entry_macd_sma200(df, idx):
    if 'macd_cross_up' not in df.columns or 'sma_200' not in df.columns:
        return False
    sma = df['sma_200'].iloc[idx]
    if pd.isna(sma):
        return False
    return bool(df['macd_cross_up'].iloc[idx]) and df['close'].iloc[idx] > sma

def entry_adx25(df, idx):
    return entry_adx_trending(df, idx, 25)

def entry_sma200(df, idx):
    return entry_price_above_sma(df, idx, 200)

def exit_adx_weak(pos, df, idx):
    if 'adx' not in df.columns:
        return False, '', 0
    adx_val = df['adx'].iloc[idx]
    if pd.isna(adx_val):
        return False, '', 0
    if adx_val < 20:
        return True, 'ADX_Weak', df['close'].iloc[idx]
    return False, '', 0

# Top 5 strategies to extract trades from
strategies = [
    {
        'name': 'P3_MACD200_T20_PS15_Q',
        'entry_fn': entry_macd_sma200,
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 20}), (exit_time_based, {'max_days': 252})]),
        'config': StrategyConfig(name='P3_MACD200_T20_PS15_Q', start_date=START, end_date=END, portfolio_size=15, rebalance_freq='quarterly'),
    },
    {
        'name': 'HYB_MACD_SMA200_STExit',
        'entry_fn': entry_macd_sma200,
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_indicator_reversal, {'indicator': 'supertrend_10_3'}),
            (exit_fixed_stop_loss, {'pct': 12}),
            (exit_time_based, {'max_days': 180}),
        ]),
        'config': StrategyConfig(name='HYB_MACD_SMA200_STExit', start_date=START, end_date=END, portfolio_size=25),
    },
    {
        'name': 'P2_MACD200_Trail20_252d',
        'entry_fn': entry_macd_sma200,
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 20}), (exit_time_based, {'max_days': 252})]),
        'config': StrategyConfig(name='P2_MACD200_Trail20_252d', start_date=START, end_date=END, portfolio_size=25),
    },
    {
        'name': 'P2_ADX25_ADXWeakExit',
        'entry_fn': entry_adx25,
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_trailing_stop, {'pct': 15}),
            (exit_adx_weak, {}),
            (exit_time_based, {'max_days': 252}),
        ]),
        'config': StrategyConfig(name='P2_ADX25_ADXWeakExit', start_date=START, end_date=END, portfolio_size=25),
    },
    {
        'name': 'P3_SMA200_Comp_PS10',
        'entry_fn': entry_sma200,
        'rank_fn': rank_composite_momentum,
        'exit_fn': make_combined_exit([(exit_ath_drawdown, {'pct': 20}), (exit_time_based, {'max_days': 365})]),
        'config': StrategyConfig(name='P3_SMA200_Comp_PS10', start_date=START, end_date=END, portfolio_size=10),
    },
]

output = {}
for strat in strategies:
    name = strat['name']
    print(f'Running {name}...', flush=True)
    t0 = time.time()
    explorer = StrategyExplorer(universe, enriched, strat['config'])
    result = explorer.run(strat['entry_fn'], strat['rank_fn'], strat['exit_fn'])
    print(f'  {time.time()-t0:.0f}s | {result.total_trades} trades', flush=True)

    trades = result.trade_list
    if not trades:
        output[name] = []
        continue

    # Sort trades by P/L %
    for t in trades:
        t['pnl_pct'] = ((t['exit_price'] - t['entry_price']) / t['entry_price']) * 100

    winners = [t for t in trades if t['pnl_pct'] > 2]
    losers = [t for t in trades if t['pnl_pct'] < -2]
    breakeven = [t for t in trades if -2 <= t['pnl_pct'] <= 2]

    random.seed(42)
    sample = []
    sample += random.sample(winners, min(5, len(winners)))
    sample += random.sample(losers, min(3, len(losers)))
    sample += random.sample(breakeven, min(2, len(breakeven)))

    # Format for output
    formatted = []
    for t in sample:
        entry_date = t['entry_date']
        exit_date = t['exit_date']
        if hasattr(entry_date, 'strftime'):
            entry_date = entry_date.strftime('%Y-%m-%d')
        if hasattr(exit_date, 'strftime'):
            exit_date = exit_date.strftime('%Y-%m-%d')
        formatted.append({
            'symbol': t['symbol'],
            'entry_date': str(entry_date),
            'entry_price': round(t['entry_price'], 2),
            'exit_date': str(exit_date),
            'exit_price': round(t['exit_price'], 2),
            'exit_reason': t['exit_reason'],
            'days': t.get('holding_days', ''),
            'pnl_pct': round(t['pnl_pct'], 1),
        })
    output[name] = formatted

# Save
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_trades.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f'\nSaved to {out_path}')
