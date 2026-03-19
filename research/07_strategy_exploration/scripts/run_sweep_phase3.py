"""Phase 3: Portfolio construction optimization – concentration, rebalance, sector limits."""
import csv, os, sys, time, logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, load_enriched_cache,
    StrategyConfig, StrategyExplorer,
    entry_macd_crossover, entry_adx_trending,
    entry_price_above_sma,
    rank_momentum_12m, rank_composite_momentum,
    rank_distance_from_high, rank_rsi_strength,
    exit_trailing_stop, exit_ath_drawdown, exit_time_based,
    exit_fixed_stop_loss,
    make_combined_exit,
)
import pandas as pd

START, END = '2015-01-01', '2025-12-31'
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exploration_phase3.csv')
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

# ======== Entry functions ========

def entry_macd_sma200(df, idx):
    """MACD cross up AND price > SMA200."""
    if 'macd_cross_up' not in df.columns or 'sma_200' not in df.columns:
        return False
    sma = df['sma_200'].iloc[idx]
    if pd.isna(sma):
        return False
    return bool(df['macd_cross_up'].iloc[idx]) and df['close'].iloc[idx] > sma

def entry_adx25(df, idx):
    """ADX > 25 and +DI > -DI (strong trend)."""
    return entry_adx_trending(df, idx, 25)

def entry_sma200_price(df, idx):
    """Price above SMA200 - pure trend filter."""
    return entry_price_above_sma(df, idx, 200)

# ======== Exit combos ========

def exit_macd200_best():
    """Best Phase 2 exit: Trail 20%, time 252d."""
    return make_combined_exit([
        (exit_trailing_stop, {'pct': 20}),
        (exit_time_based, {'max_days': 252}),
    ])

def exit_adx_weak(pos, df, idx):
    """Exit if ADX drops below 20 (trend weakening)."""
    if 'adx' not in df.columns:
        return False, '', 0
    adx_val = df['adx'].iloc[idx]
    if pd.isna(adx_val):
        return False, '', 0
    if adx_val < 20:
        return True, 'ADX_Weak', df['close'].iloc[idx]
    return False, '', 0

def exit_adx_best():
    """Best Phase 2 ADX exit: Trail 15% + ADX weak + time 252d."""
    return make_combined_exit([
        (exit_trailing_stop, {'pct': 15}),
        (exit_adx_weak, {}),
        (exit_time_based, {'max_days': 252}),
    ])

def exit_sma200_best():
    """Best SMA200 exit: ATH 20% + time 365d."""
    return make_combined_exit([
        (exit_ath_drawdown, {'pct': 20}),
        (exit_time_based, {'max_days': 365}),
    ])

# ======== Config builder ========

def cfg(name, ps=25, rebal='monthly', max_sect=0.25, max_per_sect=8):
    return StrategyConfig(
        name=name, start_date=START, end_date=END,
        portfolio_size=ps, rebalance_freq=rebal,
        max_sector_pct=max_sect,
        max_stocks_per_sector=max_per_sect,
    )

# ======== 24 Phase 3 configs ========
configs = [
    # === Group A: MACD+SMA200 base (best Calmar) with portfolio construction variants ===
    # A1: PS10 (aggressive concentration)
    {'name': 'P3_MACD200_T20_PS10',
     'entry_fn': entry_macd_sma200, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_macd200_best(), 'config': cfg('P3_MACD200_T20_PS10', ps=10)},
    # A2: PS10 quarterly rebalance
    {'name': 'P3_MACD200_T20_PS10_Q',
     'entry_fn': entry_macd_sma200, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_macd200_best(), 'config': cfg('P3_MACD200_T20_PS10_Q', ps=10, rebal='quarterly')},
    # A3: PS10 semi-annual rebalance
    {'name': 'P3_MACD200_T20_PS10_SA',
     'entry_fn': entry_macd_sma200, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_macd200_best(), 'config': cfg('P3_MACD200_T20_PS10_SA', ps=10, rebal='semi_annual')},
    # A4: PS15 quarterly
    {'name': 'P3_MACD200_T20_PS15_Q',
     'entry_fn': entry_macd_sma200, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_macd200_best(), 'config': cfg('P3_MACD200_T20_PS15_Q', ps=15, rebal='quarterly')},
    # A5: PS20 with relaxed sectors (40%, max_per_sect=10)
    {'name': 'P3_MACD200_T20_PS20_SECT40',
     'entry_fn': entry_macd_sma200, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_macd200_best(), 'config': cfg('P3_MACD200_T20_PS20_SECT40', ps=20, max_sect=0.40, max_per_sect=10)},
    # A6: PS10 with relaxed sectors + higher pos size
    {'name': 'P3_MACD200_T20_PS10_CONC',
     'entry_fn': entry_macd_sma200, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_macd200_best(), 'config': cfg('P3_MACD200_T20_PS10_CONC', ps=10, max_sect=0.50, max_per_sect=5)},
    # A7: PS10 composite momentum rank
    {'name': 'P3_MACD200_T20_PS10_COMP',
     'entry_fn': entry_macd_sma200, 'rank_fn': rank_composite_momentum,
     'exit_fn': exit_macd200_best(), 'config': cfg('P3_MACD200_T20_PS10_COMP', ps=10)},
    # A8: PS10 RSI strength rank
    {'name': 'P3_MACD200_T20_PS10_RSI',
     'entry_fn': entry_macd_sma200, 'rank_fn': rank_rsi_strength,
     'exit_fn': exit_macd200_best(), 'config': cfg('P3_MACD200_T20_PS10_RSI', ps=10)},

    # === Group B: ADX25 base (best CAGR) with portfolio construction variants ===
    # B1: PS10
    {'name': 'P3_ADX25_ADXExit_PS10',
     'entry_fn': entry_adx25, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_adx_best(), 'config': cfg('P3_ADX25_ADXExit_PS10', ps=10)},
    # B2: PS10 quarterly
    {'name': 'P3_ADX25_ADXExit_PS10_Q',
     'entry_fn': entry_adx25, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_adx_best(), 'config': cfg('P3_ADX25_ADXExit_PS10_Q', ps=10, rebal='quarterly')},
    # B3: PS10 semi-annual
    {'name': 'P3_ADX25_ADXExit_PS10_SA',
     'entry_fn': entry_adx25, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_adx_best(), 'config': cfg('P3_ADX25_ADXExit_PS10_SA', ps=10, rebal='semi_annual')},
    # B4: PS15 quarterly
    {'name': 'P3_ADX25_ADXExit_PS15_Q',
     'entry_fn': entry_adx25, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_adx_best(), 'config': cfg('P3_ADX25_ADXExit_PS15_Q', ps=15, rebal='quarterly')},
    # B5: PS20 relaxed sectors
    {'name': 'P3_ADX25_ADXExit_PS20_SECT40',
     'entry_fn': entry_adx25, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_adx_best(), 'config': cfg('P3_ADX25_ADXExit_PS20_SECT40', ps=20, max_sect=0.40, max_per_sect=10)},
    # B6: PS10 concentrated
    {'name': 'P3_ADX25_ADXExit_PS10_CONC',
     'entry_fn': entry_adx25, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_adx_best(), 'config': cfg('P3_ADX25_ADXExit_PS10_CONC', ps=10, max_sect=0.50, max_per_sect=5)},
    # B7: PS10 composite rank
    {'name': 'P3_ADX25_ADXExit_PS10_COMP',
     'entry_fn': entry_adx25, 'rank_fn': rank_composite_momentum,
     'exit_fn': exit_adx_best(), 'config': cfg('P3_ADX25_ADXExit_PS10_COMP', ps=10)},
    # B8: PS10 distance-from-high rank (buy dips in strong trends)
    {'name': 'P3_ADX25_ADXExit_PS10_DIST',
     'entry_fn': entry_adx25, 'rank_fn': rank_distance_from_high,
     'exit_fn': exit_adx_best(), 'config': cfg('P3_ADX25_ADXExit_PS10_DIST', ps=10)},

    # === Group C: SMA200 pure momentum (best PF) with portfolio construction ===
    # C1: PS10
    {'name': 'P3_SMA200_Mom_PS10',
     'entry_fn': entry_sma200_price, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_sma200_best(), 'config': cfg('P3_SMA200_Mom_PS10', ps=10)},
    # C2: PS10 quarterly
    {'name': 'P3_SMA200_Mom_PS10_Q',
     'entry_fn': entry_sma200_price, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_sma200_best(), 'config': cfg('P3_SMA200_Mom_PS10_Q', ps=10, rebal='quarterly')},
    # C3: PS10 semi-annual
    {'name': 'P3_SMA200_Mom_PS10_SA',
     'entry_fn': entry_sma200_price, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_sma200_best(), 'config': cfg('P3_SMA200_Mom_PS10_SA', ps=10, rebal='semi_annual')},
    # C4: PS10 concentrated sectors
    {'name': 'P3_SMA200_Mom_PS10_CONC',
     'entry_fn': entry_sma200_price, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_sma200_best(), 'config': cfg('P3_SMA200_Mom_PS10_CONC', ps=10, max_sect=0.50, max_per_sect=5)},
    # C5: PS10 composite rank
    {'name': 'P3_SMA200_Comp_PS10',
     'entry_fn': entry_sma200_price, 'rank_fn': rank_composite_momentum,
     'exit_fn': exit_sma200_best(), 'config': cfg('P3_SMA200_Comp_PS10', ps=10)},
    # C6: PS15 quarterly
    {'name': 'P3_SMA200_Mom_PS15_Q',
     'entry_fn': entry_sma200_price, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_sma200_best(), 'config': cfg('P3_SMA200_Mom_PS15_Q', ps=15, rebal='quarterly')},
    # C7: PS20 quarterly
    {'name': 'P3_SMA200_Mom_PS20_Q',
     'entry_fn': entry_sma200_price, 'rank_fn': rank_momentum_12m,
     'exit_fn': exit_sma200_best(), 'config': cfg('P3_SMA200_Mom_PS20_Q', ps=20, rebal='quarterly')},
    # C8: PS10 with RSI rank
    {'name': 'P3_SMA200_RSI_PS10',
     'entry_fn': entry_sma200_price, 'rank_fn': rank_rsi_strength,
     'exit_fn': exit_sma200_best(), 'config': cfg('P3_SMA200_RSI_PS10', ps=10)},
]

# Run strategies
remaining = [c for c in configs if c['name'] not in done]
total = len(remaining)
print(f'\nRunning {total} Phase 3 strategies...', flush=True)

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
