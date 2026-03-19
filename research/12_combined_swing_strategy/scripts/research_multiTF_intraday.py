"""
Multi-Timeframe Intraday Strategy Research (Vectorized)
=======================================================
Tests 5 intraday strategies on 5-min data with daily trend filters.
All trades close by 15:20. Costs: 0.05% slippage/side + Rs 40 brokerage.

Fully vectorized with pandas — NO per-row Python iteration.
"""

import sqlite3
import pandas as pd
import numpy as np
import csv
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research_results_multiTF.csv')

FIELDNAMES = [
    'strategy', 'params', 'symbol', 'total_trades', 'wins', 'losses',
    'win_rate', 'profit_factor', 'avg_win_pct', 'avg_loss_pct',
    'total_pnl_pct', 'max_win_pct', 'max_loss_pct',
    'avg_hold_bars', 'best_entry_hour', 'direction'
]

SLIPPAGE_PCT = 0.0005  # 0.05% per side
BROKERAGE_RS = 40
ASSUMED_CAPITAL_PER_TRADE = 500_000
BROKERAGE_PCT = (BROKERAGE_RS / ASSUMED_CAPITAL_PER_TRADE) * 100  # ~0.008%

START_DATE = '2024-03-18'
END_DATE = '2026-03-12'

TOP_20 = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN',
    'BHARTIARTL', 'ITC', 'KOTAKBANK', 'HINDUNILVR',
    'BAJFINANCE', 'AXISBANK', 'LT', 'TITAN',
    'HCLTECH', 'WIPRO', 'SUNPHARMA', 'MARUTI', 'ADANIENT',
    'TATAMOTORS',  # will be skipped if missing
]


def load_data():
    """Load 5-min and daily data for top 20 symbols."""
    conn = sqlite3.connect(DB_PATH)

    print("Loading 5-min data...", flush=True)
    placeholders = ','.join(['?'] * len(TOP_20))
    q5 = f"""SELECT symbol, date, open, high, low, close, volume
             FROM market_data_unified
             WHERE timeframe='5minute' AND symbol IN ({placeholders})
             AND date >= ? AND date <= ?
             ORDER BY symbol, date"""
    df5 = pd.read_sql(q5, conn, params=TOP_20 + [START_DATE, END_DATE + ' 23:59:59'])
    df5['date'] = pd.to_datetime(df5['date'])
    df5['trade_date'] = df5['date'].dt.date
    df5['time_str'] = df5['date'].dt.strftime('%H:%M')
    df5['hour'] = df5['date'].dt.hour
    print(f"  Loaded {len(df5):,} 5-min rows for {df5['symbol'].nunique()} symbols", flush=True)

    print("Loading daily data...", flush=True)
    warmup_start = '2024-01-01'
    qd = f"""SELECT symbol, date, open, high, low, close, volume
             FROM market_data_unified
             WHERE timeframe='day' AND symbol IN ({placeholders})
             AND date >= ? AND date <= ?
             ORDER BY symbol, date"""
    dfd = pd.read_sql(qd, conn, params=TOP_20 + [warmup_start, END_DATE + ' 23:59:59'])
    dfd['date'] = pd.to_datetime(dfd['date'])
    print(f"  Loaded {len(dfd):,} daily rows for {dfd['symbol'].nunique()} symbols", flush=True)

    conn.close()
    return df5, dfd


def compute_daily_features(dfd):
    """Compute daily EMA20, prev day high/low, etc. Returns dict of {symbol: DataFrame indexed by trade_date}."""
    daily_map = {}
    for sym, gdf in dfd.groupby('symbol'):
        gdf = gdf.sort_values('date').copy()
        gdf['ema20'] = gdf['close'].ewm(span=20, adjust=False).mean()
        gdf['prev_high'] = gdf['high'].shift(1)
        gdf['prev_low'] = gdf['low'].shift(1)
        gdf['prev_close'] = gdf['close'].shift(1)
        gdf['trade_date'] = gdf['date'].dt.date
        gdf['daily_trend_up'] = gdf['close'] > gdf['ema20']
        daily_map[sym] = gdf.set_index('trade_date')[['ema20', 'prev_high', 'prev_low', 'prev_close', 'daily_trend_up', 'close']].rename(columns={'close': 'daily_close'})
    return daily_map


def apply_costs_vec(entry_prices, exit_prices, direction):
    """Vectorized cost application. direction: 'long' or 'short'."""
    if direction == 'long':
        eff_entry = entry_prices * (1 + SLIPPAGE_PCT)
        eff_exit = exit_prices * (1 - SLIPPAGE_PCT)
        gross = (eff_exit - eff_entry) / eff_entry * 100
    else:
        eff_entry = entry_prices * (1 - SLIPPAGE_PCT)
        eff_exit = exit_prices * (1 + SLIPPAGE_PCT)
        gross = (eff_entry - eff_exit) / eff_entry * 100
    return gross - BROKERAGE_PCT


def summarize_trades_from_df(trades_df, strategy_name, params_str, symbol):
    """Summarize from a DataFrame with columns: pnl_pct, entry_hour, hold_bars, direction."""
    if trades_df is None or len(trades_df) == 0:
        return None

    n = len(trades_df)
    pnls = trades_df['pnl_pct'].values
    wins_mask = pnls > 0
    n_wins = int(wins_mask.sum())
    n_losses = n - n_wins

    gross_wins = float(pnls[wins_mask].sum()) if n_wins > 0 else 0
    gross_losses = float(abs(pnls[~wins_mask].sum())) if n_losses > 0 else 0.001

    hour_pnl = trades_df.groupby('entry_hour')['pnl_pct'].sum()
    best_hour = int(hour_pnl.idxmax()) if len(hour_pnl) > 0 else 0

    dirs = trades_df['direction'].unique()
    dir_str = 'both' if len(dirs) > 1 else (dirs[0] if len(dirs) == 1 else 'both')

    return {
        'strategy': strategy_name,
        'params': params_str,
        'symbol': symbol,
        'total_trades': n,
        'wins': n_wins,
        'losses': n_losses,
        'win_rate': round(n_wins / n * 100, 2),
        'profit_factor': round(gross_wins / gross_losses, 3) if gross_losses > 0 else 999,
        'avg_win_pct': round(float(pnls[wins_mask].mean()), 4) if n_wins > 0 else 0,
        'avg_loss_pct': round(float(pnls[~wins_mask].mean()), 4) if n_losses > 0 else 0,
        'total_pnl_pct': round(float(pnls.sum()), 4),
        'max_win_pct': round(float(pnls.max()), 4),
        'max_loss_pct': round(float(pnls.min()), 4),
        'avg_hold_bars': round(float(trades_df['hold_bars'].mean()), 1) if 'hold_bars' in trades_df.columns else 0,
        'best_entry_hour': best_hour,
        'direction': dir_str,
    }


def write_result(row):
    """Append one row to CSV."""
    file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _add_day_bar_number(df5_sym):
    """Add bar_num (0-based within each trade_date)."""
    df = df5_sym.copy()
    df['bar_num'] = df.groupby('trade_date').cumcount()
    return df


def _compute_pnl(exits, long_mask_col='is_long'):
    """Compute PnL for a exits DataFrame with entry_price, exit_price, is_long columns."""
    long_mask = exits[long_mask_col]
    exits = exits.copy()
    exits['pnl_pct'] = 0.0
    if long_mask.any():
        exits.loc[long_mask, 'pnl_pct'] = apply_costs_vec(
            exits.loc[long_mask, 'entry_price'], exits.loc[long_mask, 'exit_price'], 'long')
    if (~long_mask).any():
        exits.loc[~long_mask, 'pnl_pct'] = apply_costs_vec(
            exits.loc[~long_mask, 'entry_price'], exits.loc[~long_mask, 'exit_price'], 'short')
    exits['direction'] = np.where(exits[long_mask_col], 'long', 'short')
    return exits


# ============================================================
# STRATEGY 1: Opening Range Breakout (ORB) - Vectorized
# ============================================================
def strategy_orb_vec(df5_sym, daily_map_sym, orb_minutes, target_mult, use_trend_filter):
    """Vectorized ORB. One trade per day max."""
    orb_bars = orb_minutes // 5
    df = _add_day_bar_number(df5_sym)

    # ORB high/low per day
    orb_data = df[df['bar_num'] < orb_bars].groupby('trade_date').agg(
        orb_high=('high', 'max'), orb_low=('low', 'min'))
    orb_data['orb_range'] = orb_data['orb_high'] - orb_data['orb_low']
    orb_data = orb_data[orb_data['orb_range'] > 0]

    df = df.merge(orb_data, on='trade_date', how='inner')

    # Entry candidates: after ORB, before 14:00
    post_orb = df[(df['bar_num'] >= orb_bars) & (df['time_str'] <= '14:00')].copy()
    post_orb['long_break'] = post_orb['close'] > post_orb['orb_high']
    post_orb['short_break'] = post_orb['close'] < post_orb['orb_low']

    if use_trend_filter and daily_map_sym is not None:
        trend_df = daily_map_sym[['daily_trend_up']].copy()
        trend_df.index.name = 'trade_date'
        post_orb = post_orb.merge(trend_df, on='trade_date', how='left')
        post_orb['long_break'] = post_orb['long_break'] & (post_orb['daily_trend_up'] == True)
        post_orb['short_break'] = post_orb['short_break'] & (post_orb['daily_trend_up'] == False)

    post_orb['any_break'] = post_orb['long_break'] | post_orb['short_break']
    first_breaks = post_orb[post_orb['any_break']].groupby('trade_date').first().reset_index()
    if len(first_breaks) == 0:
        return pd.DataFrame()

    first_breaks['entry_price'] = first_breaks['close']
    first_breaks['entry_bar_num'] = first_breaks['bar_num']
    first_breaks['entry_hour'] = first_breaks['hour']
    first_breaks['is_long'] = first_breaks['long_break']
    first_breaks['sl_long'] = first_breaks['orb_low']
    first_breaks['tgt_long'] = first_breaks['entry_price'] + target_mult * first_breaks['orb_range']
    first_breaks['sl_short'] = first_breaks['orb_high']
    first_breaks['tgt_short'] = first_breaks['entry_price'] - target_mult * first_breaks['orb_range']

    entry_cols = ['trade_date', 'entry_price', 'entry_bar_num', 'entry_hour',
                  'is_long', 'sl_long', 'tgt_long', 'sl_short', 'tgt_short']
    exit_scan = df.merge(first_breaks[entry_cols], on='trade_date', how='inner')
    exit_scan = exit_scan[exit_scan['bar_num'] > exit_scan['entry_bar_num']]
    if len(exit_scan) == 0:
        return pd.DataFrame()

    exit_scan['hit_sl'] = np.where(exit_scan['is_long'],
        exit_scan['low'] <= exit_scan['sl_long'], exit_scan['high'] >= exit_scan['sl_short'])
    exit_scan['hit_tgt'] = np.where(exit_scan['is_long'],
        exit_scan['high'] >= exit_scan['tgt_long'], exit_scan['low'] <= exit_scan['tgt_short'])
    exit_scan['hit_eod'] = exit_scan['time_str'] >= '15:20'
    exit_scan['hit_any'] = exit_scan['hit_sl'] | exit_scan['hit_tgt'] | exit_scan['hit_eod']

    exits = exit_scan[exit_scan['hit_any']].groupby('trade_date').first().reset_index()
    if len(exits) == 0:
        return pd.DataFrame()

    exits['exit_price'] = np.where(exits['hit_sl'],
        np.where(exits['is_long'], exits['sl_long'], exits['sl_short']),
        np.where(exits['hit_tgt'],
                 np.where(exits['is_long'], exits['tgt_long'], exits['tgt_short']),
                 exits['close']))
    exits['hold_bars'] = exits['bar_num'] - exits['entry_bar_num']
    exits = _compute_pnl(exits)
    return exits[['trade_date', 'pnl_pct', 'entry_hour', 'hold_bars', 'direction']]


# ============================================================
# STRATEGY 2: Daily Trend + 5-min EMA Crossover - Vectorized
# ============================================================
def strategy_ema_cross_vec(df5_sym, daily_map_sym, fast_period=9, slow_period=21):
    """Daily close > EMA(20) = only longs. 5-min EMA cross for entry. One trade/day."""
    df = _add_day_bar_number(df5_sym)

    def _ema_per_day(group):
        g = group.sort_values('bar_num')
        g['ema_fast'] = g['close'].ewm(span=fast_period, adjust=False).mean()
        g['ema_slow'] = g['close'].ewm(span=slow_period, adjust=False).mean()
        return g

    df = df.groupby('trade_date', group_keys=False).apply(_ema_per_day)

    if daily_map_sym is not None:
        trend_df = daily_map_sym[['daily_trend_up']].copy()
        trend_df.index.name = 'trade_date'
        df = df.merge(trend_df, on='trade_date', how='left')
    else:
        df['daily_trend_up'] = True

    df['fast_above'] = df['ema_fast'] > df['ema_slow']
    df['cross_up'] = df['fast_above'] & ~df.groupby('trade_date')['fast_above'].shift(1, fill_value=False)
    df['cross_dn'] = ~df['fast_above'] & df.groupby('trade_date')['fast_above'].shift(1, fill_value=True)

    min_bar = max(fast_period, slow_period) + 2
    entry_zone = df[(df['bar_num'] >= min_bar) & (df['time_str'] <= '14:45')].copy()
    entry_zone['entry_long'] = entry_zone['cross_up'] & (entry_zone['daily_trend_up'] == True)
    entry_zone['entry_short'] = entry_zone['cross_dn'] & (entry_zone['daily_trend_up'] == False)
    entry_zone['any_entry'] = entry_zone['entry_long'] | entry_zone['entry_short']

    first_entries = entry_zone[entry_zone['any_entry']].groupby('trade_date').first().reset_index()
    if len(first_entries) == 0:
        return pd.DataFrame()

    first_entries['entry_price'] = first_entries['close']
    first_entries['entry_bar_num'] = first_entries['bar_num']
    first_entries['is_long'] = first_entries['entry_long']

    entry_info = first_entries[['trade_date', 'entry_price', 'entry_bar_num', 'hour', 'is_long']].rename(
        columns={'hour': 'entry_hour'})

    exit_scan = df.merge(entry_info, on='trade_date', how='inner')
    exit_scan = exit_scan[exit_scan['bar_num'] > exit_scan['entry_bar_num']]
    if len(exit_scan) == 0:
        return pd.DataFrame()

    exit_scan['exit_signal'] = np.where(exit_scan['is_long'], exit_scan['cross_dn'], exit_scan['cross_up'])
    exit_scan['hit_eod'] = exit_scan['time_str'] >= '15:20'
    exit_scan['hit_any'] = exit_scan['exit_signal'] | exit_scan['hit_eod']

    exits = exit_scan[exit_scan['hit_any']].groupby('trade_date').first().reset_index()
    if len(exits) == 0:
        return pd.DataFrame()

    exits['exit_price'] = exits['close']
    exits['hold_bars'] = exits['bar_num'] - exits['entry_bar_num']
    exits = _compute_pnl(exits)
    return exits[['trade_date', 'pnl_pct', 'entry_hour', 'hold_bars', 'direction']]


# ============================================================
# STRATEGY 3: Previous Day High/Low Breakout - Vectorized
# ============================================================
def strategy_pdhl_vec(df5_sym, daily_map_sym, target_pct, sl_pct):
    """Entry: First close above PDH / below PDL before 11:15. One trade/day."""
    if daily_map_sym is None:
        return pd.DataFrame()

    df = _add_day_bar_number(df5_sym)
    pdhl = daily_map_sym[['prev_high', 'prev_low']].copy()
    pdhl.index.name = 'trade_date'
    df = df.merge(pdhl, on='trade_date', how='inner')
    df = df.dropna(subset=['prev_high', 'prev_low'])

    entry_zone = df[(df['bar_num'] >= 1) & (df['time_str'] <= '11:15')].copy()
    entry_zone['long_break'] = entry_zone['close'] > entry_zone['prev_high']
    entry_zone['short_break'] = entry_zone['close'] < entry_zone['prev_low']
    entry_zone['any_break'] = entry_zone['long_break'] | entry_zone['short_break']

    first_breaks = entry_zone[entry_zone['any_break']].groupby('trade_date').first().reset_index()
    if len(first_breaks) == 0:
        return pd.DataFrame()

    first_breaks['entry_price'] = first_breaks['close']
    first_breaks['entry_bar_num'] = first_breaks['bar_num']
    first_breaks['entry_hour'] = first_breaks['hour']
    first_breaks['is_long'] = first_breaks['long_break']
    first_breaks['sl_long'] = first_breaks['entry_price'] * (1 - sl_pct / 100)
    first_breaks['tgt_long'] = first_breaks['entry_price'] * (1 + target_pct / 100)
    first_breaks['sl_short'] = first_breaks['entry_price'] * (1 + sl_pct / 100)
    first_breaks['tgt_short'] = first_breaks['entry_price'] * (1 - target_pct / 100)

    entry_cols = ['trade_date', 'entry_price', 'entry_bar_num', 'entry_hour',
                  'is_long', 'sl_long', 'tgt_long', 'sl_short', 'tgt_short']
    exit_scan = df.merge(first_breaks[entry_cols], on='trade_date', how='inner')
    exit_scan = exit_scan[exit_scan['bar_num'] > exit_scan['entry_bar_num']]
    if len(exit_scan) == 0:
        return pd.DataFrame()

    exit_scan['hit_sl'] = np.where(exit_scan['is_long'],
        exit_scan['low'] <= exit_scan['sl_long'], exit_scan['high'] >= exit_scan['sl_short'])
    exit_scan['hit_tgt'] = np.where(exit_scan['is_long'],
        exit_scan['high'] >= exit_scan['tgt_long'], exit_scan['low'] <= exit_scan['tgt_short'])
    exit_scan['hit_eod'] = exit_scan['time_str'] >= '15:20'
    exit_scan['hit_any'] = exit_scan['hit_sl'] | exit_scan['hit_tgt'] | exit_scan['hit_eod']

    exits = exit_scan[exit_scan['hit_any']].groupby('trade_date').first().reset_index()
    if len(exits) == 0:
        return pd.DataFrame()

    exits['exit_price'] = np.where(exits['hit_sl'],
        np.where(exits['is_long'], exits['sl_long'], exits['sl_short']),
        np.where(exits['hit_tgt'],
                 np.where(exits['is_long'], exits['tgt_long'], exits['tgt_short']),
                 exits['close']))
    exits['hold_bars'] = exits['bar_num'] - exits['entry_bar_num']
    exits = _compute_pnl(exits)
    return exits[['trade_date', 'pnl_pct', 'entry_hour', 'hold_bars', 'direction']]


# ============================================================
# STRATEGY 4: VWAP + EMA - Vectorized
# ============================================================
def strategy_vwap_vec(df5_sym, daily_map_sym, ema_period):
    """Price crosses above VWAP + close > EMA = long. Exit on cross back. One trade/day."""
    df = _add_day_bar_number(df5_sym)

    def _vwap_per_day(group):
        g = group.sort_values('bar_num')
        tp = (g['high'] + g['low'] + g['close']) / 3
        g['vwap'] = (tp * g['volume']).cumsum() / g['volume'].cumsum().replace(0, np.nan)
        g['ema_sig'] = g['close'].ewm(span=ema_period, adjust=False).mean()
        return g

    df = df.groupby('trade_date', group_keys=False).apply(_vwap_per_day)

    df['above_vwap'] = df['close'] > df['vwap']
    df['cross_up'] = df['above_vwap'] & ~df.groupby('trade_date')['above_vwap'].shift(1, fill_value=False)
    df['cross_dn'] = ~df['above_vwap'] & df.groupby('trade_date')['above_vwap'].shift(1, fill_value=True)

    min_bar = ema_period + 2
    entry_zone = df[(df['bar_num'] >= min_bar) & (df['time_str'] <= '14:45')].copy()
    entry_zone['entry_long'] = entry_zone['cross_up'] & (entry_zone['close'] > entry_zone['ema_sig'])
    entry_zone['entry_short'] = entry_zone['cross_dn'] & (entry_zone['close'] < entry_zone['ema_sig'])
    entry_zone['any_entry'] = entry_zone['entry_long'] | entry_zone['entry_short']

    first_entries = entry_zone[entry_zone['any_entry']].groupby('trade_date').first().reset_index()
    if len(first_entries) == 0:
        return pd.DataFrame()

    first_entries['entry_price'] = first_entries['close']
    first_entries['entry_bar_num'] = first_entries['bar_num']
    first_entries['entry_hour'] = first_entries['hour']
    first_entries['is_long'] = first_entries['entry_long']

    entry_info = first_entries[['trade_date', 'entry_price', 'entry_bar_num', 'hour', 'is_long']].rename(
        columns={'hour': 'entry_hour'})

    exit_scan = df.merge(entry_info, on='trade_date', how='inner')
    exit_scan = exit_scan[exit_scan['bar_num'] > exit_scan['entry_bar_num']]
    if len(exit_scan) == 0:
        return pd.DataFrame()

    exit_scan['exit_signal'] = np.where(exit_scan['is_long'],
        exit_scan['close'] < exit_scan['vwap'], exit_scan['close'] > exit_scan['vwap'])
    exit_scan['hit_eod'] = exit_scan['time_str'] >= '15:20'
    exit_scan['hit_any'] = exit_scan['exit_signal'] | exit_scan['hit_eod']

    exits = exit_scan[exit_scan['hit_any']].groupby('trade_date').first().reset_index()
    if len(exits) == 0:
        return pd.DataFrame()

    exits['exit_price'] = exits['close']
    exits['hold_bars'] = exits['bar_num'] - exits['entry_bar_num']
    exits = _compute_pnl(exits)
    return exits[['trade_date', 'pnl_pct', 'entry_hour', 'hold_bars', 'direction']]


# ============================================================
# STRATEGY 5: EMA Ribbon (8/13/21/34) - Vectorized
# ============================================================
def strategy_ema_ribbon_vec(df5_sym, daily_map_sym, vol_mult=None):
    """EMAs 8,13,21,34 aligned = entry. Exit on 8/13 cross. One trade/day."""
    df = _add_day_bar_number(df5_sym)

    def _ribbon_per_day(group):
        g = group.sort_values('bar_num')
        g['ema8'] = g['close'].ewm(span=8, adjust=False).mean()
        g['ema13'] = g['close'].ewm(span=13, adjust=False).mean()
        g['ema21'] = g['close'].ewm(span=21, adjust=False).mean()
        g['ema34'] = g['close'].ewm(span=34, adjust=False).mean()
        g['vol_avg'] = g['volume'].rolling(20, min_periods=1).mean()
        return g

    df = df.groupby('trade_date', group_keys=False).apply(_ribbon_per_day)

    df['bullish_aligned'] = (df['ema8'] > df['ema13']) & (df['ema13'] > df['ema21']) & (df['ema21'] > df['ema34'])
    df['bearish_aligned'] = (df['ema8'] < df['ema13']) & (df['ema13'] < df['ema21']) & (df['ema21'] < df['ema34'])

    if vol_mult is not None and vol_mult > 0:
        df['vol_surge'] = df['volume'] > vol_mult * df['vol_avg']
    else:
        df['vol_surge'] = True

    entry_zone = df[(df['bar_num'] >= 35) & (df['time_str'] <= '14:45') & df['vol_surge']].copy()
    entry_zone['entry_long'] = entry_zone['bullish_aligned']
    entry_zone['entry_short'] = entry_zone['bearish_aligned']
    entry_zone['any_entry'] = entry_zone['entry_long'] | entry_zone['entry_short']

    first_entries = entry_zone[entry_zone['any_entry']].groupby('trade_date').first().reset_index()
    if len(first_entries) == 0:
        return pd.DataFrame()

    first_entries['entry_price'] = first_entries['close']
    first_entries['entry_bar_num'] = first_entries['bar_num']
    first_entries['is_long'] = first_entries['entry_long']

    entry_info = first_entries[['trade_date', 'entry_price', 'entry_bar_num', 'hour', 'is_long']].rename(
        columns={'hour': 'entry_hour'})

    exit_scan = df.merge(entry_info, on='trade_date', how='inner')
    exit_scan = exit_scan[exit_scan['bar_num'] > exit_scan['entry_bar_num']]
    if len(exit_scan) == 0:
        return pd.DataFrame()

    exit_scan['ema8_below_13'] = exit_scan['ema8'] < exit_scan['ema13']
    exit_scan['ema8_above_13'] = exit_scan['ema8'] > exit_scan['ema13']
    exit_scan['exit_signal'] = np.where(exit_scan['is_long'],
        exit_scan['ema8_below_13'], exit_scan['ema8_above_13'])
    exit_scan['hit_eod'] = exit_scan['time_str'] >= '15:20'
    exit_scan['hit_any'] = exit_scan['exit_signal'] | exit_scan['hit_eod']

    exits = exit_scan[exit_scan['hit_any']].groupby('trade_date').first().reset_index()
    if len(exits) == 0:
        return pd.DataFrame()

    exits['exit_price'] = exits['close']
    exits['hold_bars'] = exits['bar_num'] - exits['entry_bar_num']
    exits = _compute_pnl(exits)
    return exits[['trade_date', 'pnl_pct', 'entry_hour', 'hold_bars', 'direction']]


# ============================================================
# Config builder
# ============================================================
def get_all_configs():
    """Return list of (strategy_name, params_str, func, kwargs)."""
    configs = []

    # ORB variants: 3 timeframes x 3 targets x 2 filters = 18
    for orb_min in [15, 30, 60]:
        for tgt_mult in [1.0, 1.5, 2.0]:
            for trend_filt in [False, True]:
                tf_label = 'TF' if trend_filt else 'NF'
                label = f"ORB{orb_min}_T{tgt_mult}_{tf_label}"
                configs.append(('ORB', label, strategy_orb_vec,
                               {'orb_minutes': orb_min, 'target_mult': tgt_mult, 'use_trend_filter': trend_filt}))

    # EMA crossover variants: 4 combos
    for fast, slow in [(9, 21), (5, 13), (8, 21), (13, 34)]:
        label = f"EMAx_{fast}_{slow}"
        configs.append(('EMA_Cross', label, strategy_ema_cross_vec,
                       {'fast_period': fast, 'slow_period': slow}))

    # PDHL variants: 4 targets x 3 SLs = 12
    for tgt_pct in [0.5, 1.0, 1.5, 2.0]:
        for sl_pct in [0.3, 0.5, 1.0]:
            label = f"PDHL_T{tgt_pct}_SL{sl_pct}"
            configs.append(('PDHL', label, strategy_pdhl_vec,
                           {'target_pct': tgt_pct, 'sl_pct': sl_pct}))

    # VWAP variants: 4 EMA periods
    for ema_p in [10, 15, 20, 30]:
        label = f"VWAP_EMA{ema_p}"
        configs.append(('VWAP', label, strategy_vwap_vec, {'ema_period': ema_p}))

    # EMA Ribbon variants: 3 volume filters
    for vol_m in [None, 1.5, 2.0]:
        vol_label = 'NOVOL' if vol_m is None else f'VOL{vol_m}'
        label = f"RIBBON_{vol_label}"
        configs.append(('EMA_Ribbon', label, strategy_ema_ribbon_vec, {'vol_mult': vol_m}))

    return configs


def main():
    t0 = time.time()

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    df5, dfd = load_data()
    daily_map = compute_daily_features(dfd)
    del dfd

    sym_groups = {sym: g.copy() for sym, g in df5.groupby('symbol')}
    available_symbols = sorted(sym_groups.keys())
    del df5
    print(f"\nAvailable symbols ({len(available_symbols)}): {', '.join(available_symbols)}", flush=True)

    configs = get_all_configs()
    total_configs = len(configs)
    print(f"\n{'='*80}")
    print(f"Running {total_configs} strategy configs across {len(available_symbols)} symbols")
    print(f"{'='*80}\n", flush=True)

    results_summary = []

    for ci, (strat_name, params_str, strat_func, kwargs) in enumerate(configs):
        t1 = time.time()
        all_trades = []

        for sym in available_symbols:
            df5_sym = sym_groups[sym]
            daily_sym = daily_map.get(sym)

            try:
                trades_df = strat_func(df5_sym, daily_sym, **kwargs)
            except Exception as e:
                print(f"  ERROR {sym} {params_str}: {e}", flush=True)
                continue

            if trades_df is not None and len(trades_df) > 0:
                trades_df = trades_df.copy()
                trades_df['symbol'] = sym
                all_trades.append(trades_df)

                if len(trades_df) >= 5:
                    row = summarize_trades_from_df(trades_df, strat_name, params_str, sym)
                    if row:
                        write_result(row)

        if all_trades:
            agg_df = pd.concat(all_trades, ignore_index=True)
            agg_row = summarize_trades_from_df(agg_df, strat_name, params_str, 'ALL')
            if agg_row:
                write_result(agg_row)
                results_summary.append(agg_row)

        elapsed = time.time() - t1
        if results_summary and results_summary[-1]['params'] == params_str:
            r = results_summary[-1]
            print(f"[{ci+1}/{total_configs}] {strat_name:12s} {params_str:25s} | "
                  f"Trades={r['total_trades']:5d} WR={r['win_rate']:5.1f}% PF={r['profit_factor']:5.2f} "
                  f"PnL={r['total_pnl_pct']:8.2f}% | {elapsed:.1f}s", flush=True)
        else:
            print(f"[{ci+1}/{total_configs}] {strat_name:12s} {params_str:25s} | NO TRADES | {elapsed:.1f}s", flush=True)

    total_time = time.time() - t0

    print(f"\n{'='*80}")
    print(f"COMPLETED in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Results written to: {OUTPUT_CSV}")
    print(f"{'='*80}\n")

    if not results_summary:
        print("No strategies produced trades.")
        return

    df_res = pd.DataFrame(results_summary)
    df_viable = df_res[df_res['total_trades'] >= 100].sort_values('profit_factor', ascending=False)

    print(f"\n{'='*80}")
    print(f"VIABLE STRATEGIES (100+ trades, sorted by Profit Factor)")
    print(f"{'='*80}")
    print(f"{'Strategy':12s} {'Params':25s} {'Trades':>7s} {'WR%':>6s} {'PF':>6s} {'AvgW%':>7s} {'AvgL%':>8s} {'TotPnL%':>9s}")
    print("-" * 90)
    for _, r in df_viable.iterrows():
        marker = ' ***' if r['profit_factor'] > 1.1 else ''
        print(f"{r['strategy']:12s} {r['params']:25s} {r['total_trades']:7d} {r['win_rate']:5.1f}% "
              f"{r['profit_factor']:6.3f} {r['avg_win_pct']:6.4f}% {r['avg_loss_pct']:7.4f}% "
              f"{r['total_pnl_pct']:8.2f}%{marker}")

    print(f"\n\nBEST PER STRATEGY TYPE (100+ trades):")
    print("-" * 90)
    for strat in df_viable['strategy'].unique():
        best = df_viable[df_viable['strategy'] == strat].iloc[0]
        marker = ' ***' if best['profit_factor'] > 1.1 else ''
        print(f"{best['strategy']:12s} {best['params']:25s} {best['total_trades']:7d} {best['win_rate']:5.1f}% "
              f"{best['profit_factor']:6.3f} {best['avg_win_pct']:6.4f}% {best['avg_loss_pct']:7.4f}% "
              f"{best['total_pnl_pct']:8.2f}%{marker}")

    pf_good = df_viable[df_viable['profit_factor'] > 1.1]
    if len(pf_good) > 0:
        print(f"\n\n*** STRATEGIES WITH PF > 1.1 AND 100+ TRADES: {len(pf_good)} found ***")
        print("-" * 90)
        for _, r in pf_good.iterrows():
            print(f"  {r['strategy']:12s} {r['params']:25s} Trades={r['total_trades']:5d} WR={r['win_rate']:5.1f}% "
                  f"PF={r['profit_factor']:6.3f} TotPnL={r['total_pnl_pct']:8.2f}%")
    else:
        print("\n\nNo strategies achieved PF > 1.1 with 100+ trades.")

    df_few = df_res[(df_res['total_trades'] >= 30) & (df_res['total_trades'] < 100) & (df_res['profit_factor'] > 1.2)]
    if len(df_few) > 0:
        df_few = df_few.sort_values('profit_factor', ascending=False)
        print(f"\n\nHIGH PF WITH FEWER TRADES (30-99 trades, PF > 1.2):")
        print("-" * 90)
        for _, r in df_few.iterrows():
            print(f"  {r['strategy']:12s} {r['params']:25s} Trades={r['total_trades']:5d} WR={r['win_rate']:5.1f}% "
                  f"PF={r['profit_factor']:6.3f} TotPnL={r['total_pnl_pct']:8.2f}%")


if __name__ == '__main__':
    main()
