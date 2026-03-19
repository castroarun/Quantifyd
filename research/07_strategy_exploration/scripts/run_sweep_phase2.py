"""Phase 2: Exit optimization + portfolio size tuning for top 3 Phase 1 winners."""
import csv, os, sys, time, logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, load_enriched_cache,
    StrategyConfig, StrategyExplorer,
    entry_macd_crossover, entry_adx_trending,
    entry_52w_high_proximity, entry_volume_breakout,
    entry_price_above_sma,
    rank_momentum_12m, rank_composite_momentum,
    rank_distance_from_high,
    exit_trailing_stop, exit_ath_drawdown, exit_time_based,
    exit_fixed_stop_loss, exit_take_profit,
    exit_indicator_reversal,
    make_combined_exit,
)
import pandas as pd

START, END = '2015-01-01', '2025-12-31'
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exploration_phase2.csv')
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
    print('ERROR: No enriched cache found.', flush=True)
    sys.exit(1)
universe, _ = preload_exploration_data(START, END)
print(f'Data ready in {time.time()-t0:.0f}s', flush=True)

# === Entry functions for top 3 winners ===
def entry_macd_sma200(df, idx):
    """MACD cross up AND price > SMA200."""
    if 'macd_cross_up' not in df.columns or 'sma_200' not in df.columns:
        return False
    sma = df['sma_200'].iloc[idx]
    if pd.isna(sma):
        return False
    return bool(df['macd_cross_up'].iloc[idx]) and df['close'].iloc[idx] > sma

def entry_52w_vol(df, idx):
    """Near 52-week high (>95%) AND volume > 1.5x average."""
    if 'dist_from_52w' not in df.columns or 'vol_ratio' not in df.columns:
        return False
    d52 = df['dist_from_52w'].iloc[idx]
    vr = df['vol_ratio'].iloc[idx]
    if pd.isna(d52) or pd.isna(vr):
        return False
    return d52 >= 0.95 and vr > 1.5

def cfg(name, ps=25, rebal='monthly'):
    return StrategyConfig(name=name, start_date=START, end_date=END,
                          portfolio_size=ps, rebalance_freq=rebal)

# ===== Phase 2 Configs (24 total) =====
configs = [
    # --- Winner 1: MACD + SMA200 + various exits ---
    # Exit: tighter trailing 10% + 120d
    {'name': 'P2_MACD200_Trail10_120d', 'entry_fn': entry_macd_sma200,
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 10}), (exit_time_based, {'max_days': 120})]),
     'config': cfg('P2_MACD200_Trail10_120d')},
    # Exit: trailing 12% + 150d
    {'name': 'P2_MACD200_Trail12_150d', 'entry_fn': entry_macd_sma200,
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 12}), (exit_time_based, {'max_days': 150})]),
     'config': cfg('P2_MACD200_Trail12_150d')},
    # Exit: wide trailing 20% + 252d
    {'name': 'P2_MACD200_Trail20_252d', 'entry_fn': entry_macd_sma200,
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 20}), (exit_time_based, {'max_days': 252})]),
     'config': cfg('P2_MACD200_Trail20_252d')},
    # Exit: ATH 15% + 180d
    {'name': 'P2_MACD200_ATH15_180d', 'entry_fn': entry_macd_sma200,
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_ath_drawdown, {'pct': 15}), (exit_time_based, {'max_days': 180})]),
     'config': cfg('P2_MACD200_ATH15_180d')},
    # Exit: ATH 25% + 365d (very wide)
    {'name': 'P2_MACD200_ATH25_365d', 'entry_fn': entry_macd_sma200,
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_ath_drawdown, {'pct': 25}), (exit_time_based, {'max_days': 365})]),
     'config': cfg('P2_MACD200_ATH25_365d')},
    # Exit: MACD flip + SL 15% + time 180d
    {'name': 'P2_MACD200_MACDFlip_SL15', 'entry_fn': entry_macd_sma200,
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_indicator_reversal, {'indicator': 'macd'}), (exit_fixed_stop_loss, {'pct': 15}), (exit_time_based, {'max_days': 180})]),
     'config': cfg('P2_MACD200_MACDFlip_SL15')},
    # PS20 concentrated
    {'name': 'P2_MACD200_STExit_PS20', 'entry_fn': entry_macd_sma200,
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_indicator_reversal, {'indicator': 'supertrend_10_3'}), (exit_fixed_stop_loss, {'pct': 12}), (exit_time_based, {'max_days': 180})]),
     'config': cfg('P2_MACD200_STExit_PS20', ps=20)},
    # PS15 concentrated
    {'name': 'P2_MACD200_STExit_PS15', 'entry_fn': entry_macd_sma200,
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_indicator_reversal, {'indicator': 'supertrend_10_3'}), (exit_fixed_stop_loss, {'pct': 12}), (exit_time_based, {'max_days': 180})]),
     'config': cfg('P2_MACD200_STExit_PS15', ps=15)},

    # --- Winner 2: 52W High + Volume + various exits ---
    # Trailing 8% + 90d (tighter)
    {'name': 'P2_52WVol_Trail8_90d', 'entry_fn': entry_52w_vol,
     'rank_fn': rank_distance_from_high,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 8}), (exit_time_based, {'max_days': 90})]),
     'config': cfg('P2_52WVol_Trail8_90d', ps=20)},
    # Trailing 12% + 120d
    {'name': 'P2_52WVol_Trail12_120d', 'entry_fn': entry_52w_vol,
     'rank_fn': rank_distance_from_high,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 12}), (exit_time_based, {'max_days': 120})]),
     'config': cfg('P2_52WVol_Trail12_120d', ps=20)},
    # ATH 15% + 180d
    {'name': 'P2_52WVol_ATH15_180d', 'entry_fn': entry_52w_vol,
     'rank_fn': rank_distance_from_high,
     'exit_fn': make_combined_exit([(exit_ath_drawdown, {'pct': 15}), (exit_time_based, {'max_days': 180})]),
     'config': cfg('P2_52WVol_ATH15_180d', ps=20)},
    # TP 30% + SL 10% + 180d
    {'name': 'P2_52WVol_TP30_SL10', 'entry_fn': entry_52w_vol,
     'rank_fn': rank_distance_from_high,
     'exit_fn': make_combined_exit([(exit_take_profit, {'pct': 30}), (exit_fixed_stop_loss, {'pct': 10}), (exit_time_based, {'max_days': 180})]),
     'config': cfg('P2_52WVol_TP30_SL10', ps=20)},
    # PS15 with 10% trailing + 120d
    {'name': 'P2_52WVol_Trail10_PS15', 'entry_fn': entry_52w_vol,
     'rank_fn': rank_distance_from_high,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 10}), (exit_time_based, {'max_days': 120})]),
     'config': cfg('P2_52WVol_Trail10_PS15', ps=15)},
    # PS25 + momentum ranking instead
    {'name': 'P2_52WVol_Trail10_MomRank', 'entry_fn': entry_52w_vol,
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 10}), (exit_time_based, {'max_days': 120})]),
     'config': cfg('P2_52WVol_Trail10_MomRank', ps=20)},

    # --- Winner 3: ADX25 + various exits ---
    # ATH 20% + 180d (instead of trailing)
    {'name': 'P2_ADX25_ATH20_180d', 'entry_fn': lambda df, idx: entry_adx_trending(df, idx, 25),
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_ath_drawdown, {'pct': 20}), (exit_time_based, {'max_days': 180})]),
     'config': cfg('P2_ADX25_ATH20_180d')},
    # Trailing 20% + 252d (wider)
    {'name': 'P2_ADX25_Trail20_252d', 'entry_fn': lambda df, idx: entry_adx_trending(df, idx, 25),
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 20}), (exit_time_based, {'max_days': 252})]),
     'config': cfg('P2_ADX25_Trail20_252d')},
    # Trail 10% + 120d (tighter)
    {'name': 'P2_ADX25_Trail10_120d', 'entry_fn': lambda df, idx: entry_adx_trending(df, idx, 25),
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 10}), (exit_time_based, {'max_days': 120})]),
     'config': cfg('P2_ADX25_Trail10_120d')},
    # ADX weakening exit
    {'name': 'P2_ADX25_ADXWeakExit', 'entry_fn': lambda df, idx: entry_adx_trending(df, idx, 25),
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 15}), (exit_time_based, {'max_days': 252})]),
     'config': cfg('P2_ADX25_ADXWeakExit')},
    # PS20 concentrated
    {'name': 'P2_ADX25_Trail15_PS20', 'entry_fn': lambda df, idx: entry_adx_trending(df, idx, 25),
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 15}), (exit_time_based, {'max_days': 180})]),
     'config': cfg('P2_ADX25_Trail15_PS20', ps=20)},
    # PS15 concentrated
    {'name': 'P2_ADX25_Trail15_PS15', 'entry_fn': lambda df, idx: entry_adx_trending(df, idx, 25),
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 15}), (exit_time_based, {'max_days': 180})]),
     'config': cfg('P2_ADX25_Trail15_PS15', ps=15)},
    # Composite momentum ranking instead of 12m
    {'name': 'P2_ADX25_Trail15_CompRank', 'entry_fn': lambda df, idx: entry_adx_trending(df, idx, 25),
     'rank_fn': rank_composite_momentum,
     'exit_fn': make_combined_exit([(exit_trailing_stop, {'pct': 15}), (exit_time_based, {'max_days': 180})]),
     'config': cfg('P2_ADX25_Trail15_CompRank')},

    # --- Bonus: price>SMA200 pure momentum with small portfolio ---
    {'name': 'P2_SMA200_HighRank_PS20', 'entry_fn': lambda df, idx: entry_price_above_sma(df, idx, 200),
     'rank_fn': rank_distance_from_high,
     'exit_fn': make_combined_exit([(exit_ath_drawdown, {'pct': 20}), (exit_time_based, {'max_days': 365})]),
     'config': cfg('P2_SMA200_HighRank_PS20', ps=20, rebal='semi_annual')},
    {'name': 'P2_SMA200_HighRank_PS15', 'entry_fn': lambda df, idx: entry_price_above_sma(df, idx, 200),
     'rank_fn': rank_distance_from_high,
     'exit_fn': make_combined_exit([(exit_ath_drawdown, {'pct': 20}), (exit_time_based, {'max_days': 365})]),
     'config': cfg('P2_SMA200_HighRank_PS15', ps=15, rebal='semi_annual')},
    {'name': 'P2_SMA200_MomRank_PS20', 'entry_fn': lambda df, idx: entry_price_above_sma(df, idx, 200),
     'rank_fn': rank_momentum_12m,
     'exit_fn': make_combined_exit([(exit_ath_drawdown, {'pct': 20}), (exit_time_based, {'max_days': 365})]),
     'config': cfg('P2_SMA200_MomRank_PS20', ps=20, rebal='semi_annual')},
]

# Run strategies
remaining = [c for c in configs if c['name'] not in done]
total = len(remaining)
print(f'\nRunning {total} Phase 2 strategies...', flush=True)

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
