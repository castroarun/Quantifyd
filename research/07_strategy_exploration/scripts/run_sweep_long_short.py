"""Long + Short Systems Sweep — 24 configs
Period: 2005-2025 (20 years).
Uses run_long_short() method for hedged portfolios.
"""
import csv, os, sys, time, logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, load_enriched_cache,
    StrategyConfig, StrategyExplorer,
    entry_ema_crossover, entry_macd_crossover, entry_adx_trending,
    entry_supertrend_bullish, entry_rsi_oversold,
    entry_price_above_sma,
    rank_momentum_12m, rank_composite_momentum,
    exit_trailing_stop, exit_ath_drawdown, exit_fixed_stop_loss,
    exit_take_profit, exit_time_based, exit_indicator_reversal,
    exit_adx_weakening,
    make_combined_exit,
)
from services.technical_indicators import calc_ema
import pandas as pd
import numpy as np

START, END = '2005-01-01', '2025-12-31'

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exploration_long_short.csv')
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

# Add missing EMAs
for sym, df in enriched.items():
    for period in [5, 8, 13, 34]:
        col = f'ema_{period}'
        if col not in df.columns:
            df[col] = calc_ema(df['close'], period)
    for fast, slow in [(9,21), (10,30), (20,50), (50,200)]:
        col_up = f'ema_cross_up_{fast}_{slow}'
        col_dn = f'ema_cross_down_{fast}_{slow}'
        if col_up not in df.columns:
            f_col, s_col = f'ema_{fast}', f'ema_{slow}'
            if f_col in df.columns and s_col in df.columns:
                bull = df[f_col] > df[s_col]
                df[col_up] = bull & ~bull.shift(1).fillna(False)
                df[col_dn] = ~bull & bull.shift(1).fillna(True)
                df[f'ema_bullish_{fast}_{slow}'] = bull
print('Data ready', flush=True)

# ---- Long Entry Functions ----
def long_ema_20_50(df, idx):
    return entry_ema_crossover(df, idx, fast=20, slow=50)

def long_ema_9_21(df, idx):
    return entry_ema_crossover(df, idx, fast=9, slow=21)

def long_macd(df, idx):
    return entry_macd_crossover(df, idx)

def long_adx30(df, idx):
    return entry_adx_trending(df, idx, threshold=30)

def long_macd_sma200(df, idx):
    """MACD cross up AND price > SMA200."""
    if 'macd_cross_up' not in df.columns or 'sma_200' not in df.columns:
        return False
    sma = df['sma_200'].iloc[idx]
    if pd.isna(sma):
        return False
    return bool(df['macd_cross_up'].iloc[idx]) and df['close'].iloc[idx] > sma

# ---- Short Entry Functions (reversal of long signals) ----
def short_ema_20_50(df, idx):
    """EMA 20 crosses BELOW EMA 50 (bearish crossover)."""
    col = 'ema_cross_down_20_50'
    if col in df.columns:
        return bool(df[col].iloc[idx])
    return False

def short_ema_9_21(df, idx):
    col = 'ema_cross_down_9_21'
    if col in df.columns:
        return bool(df[col].iloc[idx])
    return False

def short_macd(df, idx):
    """MACD cross down (bearish)."""
    if 'macd_cross_down' not in df.columns:
        return False
    return bool(df['macd_cross_down'].iloc[idx])

def short_supertrend_bearish(df, idx):
    """SuperTrend flips to bearish."""
    col = 'st_flip_down_10_3'
    if col not in df.columns:
        return False
    return bool(df[col].iloc[idx])

def short_rsi_overbought(df, idx):
    """RSI > 70 AND price < SMA200 (downtrend confirm)."""
    if 'rsi_14' not in df.columns or 'sma_200' not in df.columns:
        return False
    rsi = df['rsi_14'].iloc[idx]
    sma = df['sma_200'].iloc[idx]
    if pd.isna(rsi) or pd.isna(sma):
        return False
    return rsi > 70 and df['close'].iloc[idx] < sma

# ---- Short Exit Functions ----
def exit_short_trailing(pos, df, idx, pct=15):
    """Trail stop for short: exit if price rises pct% from trough."""
    trough = pos.get('trough_price', pos['entry_price'])
    current = df['close'].iloc[idx]
    if current < trough:
        pos['trough_price'] = current
        trough = current
    if trough > 0 and (current / trough - 1) * 100 >= pct:
        return True, f'ShortTrail_{pct}pct', current
    return False, '', 0

def exit_short_fixed_sl(pos, df, idx, pct=10):
    """Fixed stop for short: exit if price rises pct% from entry."""
    if (df['close'].iloc[idx] / pos['entry_price'] - 1) * 100 >= pct:
        return True, f'ShortSL_{pct}pct', df['close'].iloc[idx]
    return False, '', 0

def exit_short_take_profit(pos, df, idx, pct=20):
    """Take profit for short: exit if price drops pct% from entry."""
    if (1 - df['close'].iloc[idx] / pos['entry_price']) * 100 >= pct:
        return True, f'ShortTP_{pct}pct', df['close'].iloc[idx]
    return False, '', 0

def exit_short_time(pos, df, idx, max_days=90):
    """Time-based exit for shorts."""
    entry_date = pos['entry_date']
    current_date = df.index[idx] if hasattr(df.index[idx], 'date') else df.index[idx]
    if (current_date - entry_date).days >= max_days:
        return True, f'ShortTime_{max_days}d', df['close'].iloc[idx]
    return False, '', 0

# Combined short exits
SHORT_EXIT_TRAIL15 = make_combined_exit([
    (exit_short_trailing, {'pct': 15}),
    (exit_short_fixed_sl, {'pct': 10}),
    (exit_short_time, {'max_days': 126}),
])

SHORT_EXIT_TRAIL10 = make_combined_exit([
    (exit_short_trailing, {'pct': 10}),
    (exit_short_fixed_sl, {'pct': 8}),
    (exit_short_time, {'max_days': 90}),
])

# Long exits
LONG_EXIT_TRAIL15 = make_combined_exit([(exit_trailing_stop, {'pct': 15}), (exit_time_based, {'max_days': 252})])
LONG_EXIT_TRAIL20 = make_combined_exit([(exit_trailing_stop, {'pct': 20}), (exit_time_based, {'max_days': 252})])
LONG_EXIT_ATH20 = make_combined_exit([(exit_ath_drawdown, {'pct': 20}), (exit_time_based, {'max_days': 365})])

# ---- CONFIGS: 24 total ----
configs = []

# Group 1: EMA 20/50 L+S (6 configs: 3 PS x 2 exit variants)
for ps in [20, 30, 40]:
    for exit_label, long_exit, short_exit in [
        ('Trail15', LONG_EXIT_TRAIL15, SHORT_EXIT_TRAIL15),
        ('Trail20', LONG_EXIT_TRAIL20, SHORT_EXIT_TRAIL10),
    ]:
        name = f'LS_EMA2050_{exit_label}_PS{ps}'
        configs.append({
            'name': name,
            'long_entry_fn': long_ema_20_50,
            'long_exit_fn': long_exit,
            'short_entry_fn': short_ema_20_50,
            'short_exit_fn': short_exit,
            'rank_fn': rank_momentum_12m,
            'config': StrategyConfig(name=name, start_date=START, end_date=END,
                                     portfolio_size=ps, allow_short=True),
        })

# Group 2: EMA 9/21 L+S (4 configs)
for ps in [20, 30]:
    for exit_label, long_exit, short_exit in [
        ('Trail15', LONG_EXIT_TRAIL15, SHORT_EXIT_TRAIL15),
        ('ATH20', LONG_EXIT_ATH20, SHORT_EXIT_TRAIL10),
    ]:
        name = f'LS_EMA921_{exit_label}_PS{ps}'
        configs.append({
            'name': name,
            'long_entry_fn': long_ema_9_21,
            'long_exit_fn': long_exit,
            'short_entry_fn': short_ema_9_21,
            'short_exit_fn': short_exit,
            'rank_fn': rank_momentum_12m,
            'config': StrategyConfig(name=name, start_date=START, end_date=END,
                                     portfolio_size=ps, allow_short=True),
        })

# Group 3: MACD L+S (4 configs)
for ps in [20, 30]:
    for exit_label, long_exit, short_exit in [
        ('Trail15', LONG_EXIT_TRAIL15, SHORT_EXIT_TRAIL15),
        ('Trail20', LONG_EXIT_TRAIL20, SHORT_EXIT_TRAIL10),
    ]:
        name = f'LS_MACD_{exit_label}_PS{ps}'
        configs.append({
            'name': name,
            'long_entry_fn': long_macd,
            'long_exit_fn': long_exit,
            'short_entry_fn': short_macd,
            'short_exit_fn': short_exit,
            'rank_fn': rank_momentum_12m,
            'config': StrategyConfig(name=name, start_date=START, end_date=END,
                                     portfolio_size=ps, allow_short=True),
        })

# Group 4: MACD+SMA200 long / SuperTrend short (4 configs)
for ps in [20, 30]:
    for exit_label, long_exit, short_exit in [
        ('Trail15', LONG_EXIT_TRAIL15, SHORT_EXIT_TRAIL15),
        ('ATH20', LONG_EXIT_ATH20, SHORT_EXIT_TRAIL10),
    ]:
        name = f'LS_MACDSMA_ST_{exit_label}_PS{ps}'
        configs.append({
            'name': name,
            'long_entry_fn': long_macd_sma200,
            'long_exit_fn': long_exit,
            'short_entry_fn': short_supertrend_bearish,
            'short_exit_fn': short_exit,
            'rank_fn': rank_momentum_12m,
            'config': StrategyConfig(name=name, start_date=START, end_date=END,
                                     portfolio_size=ps, allow_short=True),
        })

# Group 5: ADX long / RSI overbought short (4 configs)
for ps in [20, 30]:
    for exit_label, long_exit, short_exit in [
        ('Trail15', LONG_EXIT_TRAIL15, SHORT_EXIT_TRAIL15),
        ('Trail20', LONG_EXIT_TRAIL20, SHORT_EXIT_TRAIL10),
    ]:
        name = f'LS_ADX_RSI_{exit_label}_PS{ps}'
        configs.append({
            'name': name,
            'long_entry_fn': long_adx30,
            'long_exit_fn': long_exit,
            'short_entry_fn': short_rsi_overbought,
            'short_exit_fn': short_exit,
            'rank_fn': rank_momentum_12m,
            'config': StrategyConfig(name=name, start_date=START, end_date=END,
                                     portfolio_size=ps, allow_short=True),
        })

# Group 6: Mixed — Composite momentum ranking (2 configs)
for ps in [20, 30]:
    name = f'LS_EMA2050_CompRank_PS{ps}'
    configs.append({
        'name': name,
        'long_entry_fn': long_ema_20_50,
        'long_exit_fn': LONG_EXIT_ATH20,
        'short_entry_fn': short_ema_20_50,
        'short_exit_fn': SHORT_EXIT_TRAIL15,
        'rank_fn': rank_composite_momentum,
        'config': StrategyConfig(name=name, start_date=START, end_date=END,
                                 portfolio_size=ps, allow_short=True),
    })

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
    result = explorer.run_long_short(
        long_entry_fn=strat['long_entry_fn'],
        long_exit_fn=strat['long_exit_fn'],
        short_entry_fn=strat['short_entry_fn'],
        short_exit_fn=strat['short_exit_fn'],
        rank_fn=strat['rank_fn'],
    )

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
    print(f' {elapsed:.0f}s | CAGR={row["cagr"]:.2f}% MaxDD={row["max_drawdown"]:.1f}% Calmar={row["calmar"]:.2f}')
    sys.stdout.flush()

print(f'\nDone! Results in {OUTPUT_CSV}')
