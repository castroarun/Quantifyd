"""Phase 1 Category 3: Multi-factor / Hybrid strategies (12 configs)."""
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
    entry_rsi_oversold, entry_bollinger_lower,
    entry_weekly_trend_daily_pullback,
    entry_bollinger_squeeze_breakout,
    # Ranking functions
    rank_momentum_12m, rank_momentum_6m, rank_rsi_strength,
    rank_distance_from_high, rank_composite_momentum,
    # Exit functions
    exit_trailing_stop, exit_ath_drawdown, exit_time_based,
    exit_take_profit, exit_fixed_stop_loss,
    exit_indicator_reversal, exit_donchian_low,
    exit_adx_weakening,
    make_combined_exit,
)
import pandas as pd
import numpy as np

START, END = '2015-01-01', '2025-12-31'
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exploration_hybrid.csv')
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

def base_config(name, ps=25, rebal='monthly'):
    return StrategyConfig(name=name, start_date=START, end_date=END,
                          portfolio_size=ps, rebalance_freq=rebal)

# === Combo entry functions (multiple conditions) ===
def entry_ema_plus_adx(df, idx):
    """EMA 20/50 bullish AND ADX > 25 (trending)."""
    if 'ema_bullish_20_50' not in df.columns or 'adx' not in df.columns:
        return False
    if not bool(df['ema_bullish_20_50'].iloc[idx]):
        return False
    adx_val = df['adx'].iloc[idx]
    if pd.isna(adx_val):
        return False
    return adx_val > 25 and df['plus_di'].iloc[idx] > df['minus_di'].iloc[idx]

def entry_st_plus_volume(df, idx):
    """SuperTrend bullish AND volume > 1.5x average."""
    if 'st_bullish_10_3' not in df.columns or 'vol_ratio' not in df.columns:
        return False
    if not bool(df['st_bullish_10_3'].iloc[idx]):
        return False
    vr = df['vol_ratio'].iloc[idx]
    return not pd.isna(vr) and vr > 1.5

def entry_rsi_plus_bb(df, idx):
    """RSI < 30 AND close < lower Bollinger (double oversold)."""
    if 'rsi_14' not in df.columns or 'bb_below_lower' not in df.columns:
        return False
    if 'sma_200' not in df.columns:
        return False
    rsi = df['rsi_14'].iloc[idx]
    sma = df['sma_200'].iloc[idx]
    if pd.isna(rsi) or pd.isna(sma):
        return False
    return rsi < 30 and bool(df['bb_below_lower'].iloc[idx]) and df['close'].iloc[idx] > sma

def entry_macd_plus_sma200(df, idx):
    """MACD cross up AND price > SMA200."""
    if 'macd_cross_up' not in df.columns or 'sma_200' not in df.columns:
        return False
    sma = df['sma_200'].iloc[idx]
    if pd.isna(sma):
        return False
    return bool(df['macd_cross_up'].iloc[idx]) and df['close'].iloc[idx] > sma

def entry_dc50_plus_adx(df, idx):
    """Donchian 50 breakout AND ADX > 20."""
    if 'dc_breakout_up_50' not in df.columns or 'adx' not in df.columns:
        return False
    if not bool(df['dc_breakout_up_50'].iloc[idx]):
        return False
    adx_val = df['adx'].iloc[idx]
    return not pd.isna(adx_val) and adx_val > 20

def entry_52w_high_plus_volume(df, idx):
    """Near 52-week high (>95%) AND volume > 1.5x average."""
    if 'dist_from_52w' not in df.columns or 'vol_ratio' not in df.columns:
        return False
    d52 = df['dist_from_52w'].iloc[idx]
    vr = df['vol_ratio'].iloc[idx]
    if pd.isna(d52) or pd.isna(vr):
        return False
    return d52 >= 0.95 and vr > 1.5

def entry_weekly_trend_plus_macd(df, idx):
    """Weekly uptrend AND daily MACD bullish."""
    if 'weekly_trend_up' not in df.columns or 'macd_bullish' not in df.columns:
        return False
    return bool(df['weekly_trend_up'].iloc[idx]) and bool(df['macd_bullish'].iloc[idx])

def entry_golden_cross_plus_rsi(df, idx):
    """EMA 50/200 bullish AND RSI > 50 (confirming momentum)."""
    if 'ema_bullish_50_200' not in df.columns or 'rsi_14' not in df.columns:
        return False
    if not bool(df['ema_bullish_50_200'].iloc[idx]):
        return False
    rsi = df['rsi_14'].iloc[idx]
    return not pd.isna(rsi) and rsi > 50

# Smaller portfolio sizes for concentrated bets
def config_small(name, ps=20):
    return StrategyConfig(name=name, start_date=START, end_date=END,
                          portfolio_size=ps, rebalance_freq='monthly')

# ===== 12 Hybrid/Multi-Factor Strategies =====
configs = [
    # 1. EMA bullish + ADX trending (momentum confirmation) + trailing
    {
        'name': 'HYB_EMA_ADX_Trail15',
        'entry_fn': entry_ema_plus_adx,
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_trailing_stop, {'pct': 15}),
            (exit_time_based, {'max_days': 180}),
        ]),
        'config': base_config('HYB_EMA_ADX_Trail15'),
    },
    # 2. SuperTrend + Volume confirmation + ATH exit
    {
        'name': 'HYB_ST_Vol_ATH20',
        'entry_fn': entry_st_plus_volume,
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_ath_drawdown, {'pct': 20}),
            (exit_time_based, {'max_days': 180}),
        ]),
        'config': base_config('HYB_ST_Vol_ATH20'),
    },
    # 3. Double oversold (RSI + BB) + BB mid exit (mean reversion confluence)
    {
        'name': 'HYB_RSI_BB_Confluence',
        'entry_fn': entry_rsi_plus_bb,
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_take_profit, {'pct': 15}),
            (exit_fixed_stop_loss, {'pct': 8}),
            (exit_time_based, {'max_days': 60}),
        ]),
        'config': base_config('HYB_RSI_BB_Confluence'),
    },
    # 4. MACD cross + SMA200 filter + SuperTrend exit
    {
        'name': 'HYB_MACD_SMA200_STExit',
        'entry_fn': entry_macd_plus_sma200,
        'rank_fn': rank_composite_momentum,
        'exit_fn': make_combined_exit([
            (exit_indicator_reversal, {'indicator': 'supertrend_10_3'}),
            (exit_fixed_stop_loss, {'pct': 12}),
            (exit_time_based, {'max_days': 180}),
        ]),
        'config': base_config('HYB_MACD_SMA200_STExit'),
    },
    # 5. Donchian breakout + ADX confirmation + Donchian low exit
    {
        'name': 'HYB_DC50_ADX_DCExit20',
        'entry_fn': entry_dc50_plus_adx,
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_donchian_low, {'period': 20}),
            (exit_time_based, {'max_days': 180}),
        ]),
        'config': base_config('HYB_DC50_ADX_DCExit20'),
    },
    # 6. 52W high + volume (breakout momentum) + tight trailing
    {
        'name': 'HYB_52W_Vol_Trail10',
        'entry_fn': entry_52w_high_plus_volume,
        'rank_fn': rank_distance_from_high,
        'exit_fn': make_combined_exit([
            (exit_trailing_stop, {'pct': 10}),
            (exit_time_based, {'max_days': 120}),
        ]),
        'config': config_small('HYB_52W_Vol_Trail10'),
    },
    # 7. Weekly trend + MACD (multi-timeframe) + trailing
    {
        'name': 'HYB_Weekly_MACD_Trail15',
        'entry_fn': entry_weekly_trend_plus_macd,
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_trailing_stop, {'pct': 15}),
            (exit_time_based, {'max_days': 180}),
        ]),
        'config': base_config('HYB_Weekly_MACD_Trail15'),
    },
    # 8. Golden cross + RSI confirmation + wide trailing
    {
        'name': 'HYB_GoldenRSI_Wide20',
        'entry_fn': entry_golden_cross_plus_rsi,
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_trailing_stop, {'pct': 20}),
            (exit_time_based, {'max_days': 252}),
        ]),
        'config': StrategyConfig(name='HYB_GoldenRSI_Wide20', start_date=START, end_date=END,
                                  portfolio_size=25, rebalance_freq='quarterly'),
    },
    # 9. Weekly pullback + RSI(2) exit (fast mean reversion in uptrend)
    {
        'name': 'HYB_WeeklyPull_RSI2Exit',
        'entry_fn': lambda df, idx: entry_weekly_trend_daily_pullback(df, idx, 35),
        'rank_fn': rank_momentum_6m,
        'exit_fn': make_combined_exit([
            (exit_take_profit, {'pct': 10}),
            (exit_fixed_stop_loss, {'pct': 5}),
            (exit_time_based, {'max_days': 20}),
        ]),
        'config': base_config('HYB_WeeklyPull_RSI2Exit'),
    },
    # 10. EMA + ADX + concentrated portfolio (20 stocks)
    {
        'name': 'HYB_EMA_ADX_PS20',
        'entry_fn': entry_ema_plus_adx,
        'rank_fn': rank_composite_momentum,
        'exit_fn': make_combined_exit([
            (exit_ath_drawdown, {'pct': 20}),
            (exit_time_based, {'max_days': 180}),
        ]),
        'config': config_small('HYB_EMA_ADX_PS20'),
    },
    # 11. SuperTrend + MACD double confirm + ADX weakening exit
    {
        'name': 'HYB_ST_MACD_ADXExit',
        'entry_fn': lambda df, idx: (
            entry_supertrend_bullish(df, idx, 10, 3) and
            'macd_bullish' in df.columns and bool(df['macd_bullish'].iloc[idx])
        ),
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_adx_weakening, {'threshold': 20}),
            (exit_trailing_stop, {'pct': 15}),
            (exit_time_based, {'max_days': 180}),
        ]),
        'config': base_config('HYB_ST_MACD_ADXExit'),
    },
    # 12. Momentum + near-high ranking (pure momentum portfolio) + semi-annual rebal
    {
        'name': 'HYB_SMA200_HighRank_SemiAnn',
        'entry_fn': lambda df, idx: entry_price_above_sma(df, idx, 200),
        'rank_fn': rank_distance_from_high,
        'exit_fn': make_combined_exit([
            (exit_ath_drawdown, {'pct': 20}),
            (exit_time_based, {'max_days': 365}),
        ]),
        'config': StrategyConfig(name='HYB_SMA200_HighRank_SemiAnn', start_date=START, end_date=END,
                                  portfolio_size=25, rebalance_freq='semi_annual'),
    },
]

# Run strategies
remaining = [c for c in configs if c['name'] not in done]
total = len(remaining)
print(f'\nRunning {total} hybrid strategies...', flush=True)

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
