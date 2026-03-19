"""Price Action Indicators Sweep — 36 configs (12 entries x 3 exit variants)
Period: 2005-2025 (20 years), PS25, monthly rebalance.
Tests each indicator independently to find best single-indicator systems.
"""
import csv, os, sys, time, logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, load_enriched_cache,
    StrategyConfig, StrategyExplorer,
    # Entry signals
    entry_rsi_oversold, entry_macd_crossover, entry_adx_trending,
    entry_supertrend_bullish, entry_donchian_breakout,
    entry_bollinger_lower, entry_keltner_lower,
    entry_stoch_oversold, entry_cci_oversold,
    entry_williams_r_oversold, entry_mfi_oversold,
    entry_bollinger_squeeze_breakout, entry_volume_breakout,
    entry_52w_high_proximity,
    # Ranking
    rank_momentum_12m, rank_composite_momentum,
    # Exits
    exit_trailing_stop, exit_ath_drawdown, exit_fixed_stop_loss,
    exit_take_profit, exit_time_based,
    exit_rsi_overbought, exit_bollinger_upper, exit_stoch_overbought,
    exit_cci_overbought, exit_williams_r_overbought, exit_mfi_overbought,
    exit_indicator_reversal, exit_adx_weakening,
    make_combined_exit,
)
import pandas as pd

START, END = '2005-01-01', '2025-12-31'

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exploration_price_action.csv')
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
print('Data ready', flush=True)

# ---- ENTRY SIGNALS ----
# Each entry_fn with its matching "natural" indicator exit
def entry_rsi14_30(df, idx): return entry_rsi_oversold(df, idx, col='rsi_14', threshold=30)
def entry_rsi2_10(df, idx): return entry_rsi_oversold(df, idx, col='rsi_2', threshold=10)
def entry_macd(df, idx): return entry_macd_crossover(df, idx)
def entry_adx30(df, idx): return entry_adx_trending(df, idx, threshold=30)
def entry_st_bull(df, idx): return entry_supertrend_bullish(df, idx, period=10, mult=3)
def entry_dc50(df, idx): return entry_donchian_breakout(df, idx, period=50)
def entry_bb_low(df, idx): return entry_bollinger_lower(df, idx)
def entry_stoch20(df, idx): return entry_stoch_oversold(df, idx, threshold=20)
def entry_cci_neg100(df, idx): return entry_cci_oversold(df, idx, threshold=-100)
def entry_wr_neg80(df, idx): return entry_williams_r_oversold(df, idx, threshold=-80)
def entry_mfi20(df, idx): return entry_mfi_oversold(df, idx, threshold=20)
def entry_bb_squeeze(df, idx): return entry_bollinger_squeeze_breakout(df, idx)

# ---- EXIT VARIANTS ----
# Generic trailing exits
EXIT_TRAIL15 = make_combined_exit([(exit_trailing_stop, {'pct': 15}), (exit_time_based, {'max_days': 252})])
EXIT_ATH20 = make_combined_exit([(exit_ath_drawdown, {'pct': 20}), (exit_time_based, {'max_days': 365})])

# Indicator-matched exits
EXIT_RSI_OB = make_combined_exit([(exit_rsi_overbought, {'threshold': 70}), (exit_trailing_stop, {'pct': 20}), (exit_time_based, {'max_days': 126})])
EXIT_BB_UP = make_combined_exit([(exit_bollinger_upper, {}), (exit_trailing_stop, {'pct': 20}), (exit_time_based, {'max_days': 126})])
EXIT_STOCH_OB = make_combined_exit([(exit_stoch_overbought, {'threshold': 80}), (exit_trailing_stop, {'pct': 20}), (exit_time_based, {'max_days': 126})])
EXIT_CCI_OB = make_combined_exit([(exit_cci_overbought, {'threshold': 100}), (exit_trailing_stop, {'pct': 20}), (exit_time_based, {'max_days': 126})])
EXIT_WR_OB = make_combined_exit([(exit_williams_r_overbought, {'threshold': -20}), (exit_trailing_stop, {'pct': 20}), (exit_time_based, {'max_days': 126})])
EXIT_MFI_OB = make_combined_exit([(exit_mfi_overbought, {'threshold': 80}), (exit_trailing_stop, {'pct': 20}), (exit_time_based, {'max_days': 126})])
EXIT_ST_REV = make_combined_exit([(exit_indicator_reversal, {'indicator': 'supertrend_10_3'}), (exit_fixed_stop_loss, {'pct': 12}), (exit_time_based, {'max_days': 180})])
EXIT_ADX_WEAK = make_combined_exit([(exit_adx_weakening, {'threshold': 20}), (exit_trailing_stop, {'pct': 15}), (exit_time_based, {'max_days': 252})])

# ---- CONFIGS ----
# 12 entries x 3 exits each = 36 configs
# Each entry gets: Trail15, ATH20, and its natural indicator exit
entries = [
    ('RSI14_30', entry_rsi14_30, EXIT_RSI_OB),
    ('RSI2_10', entry_rsi2_10, EXIT_RSI_OB),
    ('MACD', entry_macd, EXIT_TRAIL15),  # MACD has no natural overbought exit
    ('ADX30', entry_adx30, EXIT_ADX_WEAK),
    ('SuperTrend', entry_st_bull, EXIT_ST_REV),
    ('Donchian50', entry_dc50, EXIT_TRAIL15),
    ('BB_Lower', entry_bb_low, EXIT_BB_UP),
    ('Stoch20', entry_stoch20, EXIT_STOCH_OB),
    ('CCI_neg100', entry_cci_neg100, EXIT_CCI_OB),
    ('WR_neg80', entry_wr_neg80, EXIT_WR_OB),
    ('MFI20', entry_mfi20, EXIT_MFI_OB),
    ('BB_Squeeze', entry_bb_squeeze, EXIT_TRAIL15),
]

configs = []
for label, entry_fn, natural_exit in entries:
    # Variant 1: Trail15 + Time 252d
    name1 = f'PA_{label}_Trail15'
    configs.append({'name': name1, 'entry_fn': entry_fn, 'rank_fn': rank_momentum_12m,
                    'exit_fn': EXIT_TRAIL15,
                    'config': StrategyConfig(name=name1, start_date=START, end_date=END, portfolio_size=25)})
    # Variant 2: ATH20 + Time 365d
    name2 = f'PA_{label}_ATH20'
    configs.append({'name': name2, 'entry_fn': entry_fn, 'rank_fn': rank_momentum_12m,
                    'exit_fn': EXIT_ATH20,
                    'config': StrategyConfig(name=name2, start_date=START, end_date=END, portfolio_size=25)})
    # Variant 3: Natural indicator exit
    name3 = f'PA_{label}_IndicatorExit'
    configs.append({'name': name3, 'entry_fn': entry_fn, 'rank_fn': rank_momentum_12m,
                    'exit_fn': natural_exit,
                    'config': StrategyConfig(name=name3, start_date=START, end_date=END, portfolio_size=25)})

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
    print(f' {elapsed:.0f}s | CAGR={row["cagr"]:.2f}% Calmar={row["calmar"]:.2f} PF={row["profit_factor"]:.2f}')
    sys.stdout.flush()

print(f'\nDone! Results in {OUTPUT_CSV}')
