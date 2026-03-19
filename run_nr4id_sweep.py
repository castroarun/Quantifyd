"""
NR4/ID Breakout Strategy Sweep
================================
Tests NR4 (Narrow Range 4), Inside Day, and NR4+ID combo breakout strategies
on 50 F&O stocks, daily data 2018-2025.

Signal: Detected at end of NR4/ID day. Entry next bar on breakout.
"""

import sys
import os
import csv
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.intraday_backtest_engine import (
    IntradayBacktestEngine, TradeSignal, Direction, BacktestResult,
    load_data_from_db
)
from services.technical_indicators import calc_atr

# ============================================================================
# Config
# ============================================================================

SYMBOLS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL',
    'ITC', 'KOTAKBANK', 'HINDUNILVR', 'LT', 'AXISBANK', 'BAJFINANCE', 'MARUTI',
    'HCLTECH', 'SUNPHARMA', 'TATAMOTORS', 'NTPC', 'POWERGRID', 'TITAN',
    'WIPRO', 'ULTRACEMCO', 'ADANIENT', 'ADANIPORTS', 'NESTLEIND', 'JSWSTEEL',
    'TATASTEEL', 'TECHM', 'M&M', 'COALINDIA', 'ONGC', 'GRASIM', 'BAJAJFINSV',
    'APOLLOHOSP', 'DIVISLAB', 'DRREDDY', 'CIPLA', 'HEROMOTOCO', 'EICHERMOT',
    'BPCL', 'INDUSINDBK', 'TATACONSUM', 'SHRIRAMFIN', 'BRITANNIA', 'HINDALCO',
    'ASIANPAINT', 'SBILIFE', 'HDFCLIFE', 'PIDILITIND', 'DABUR'
]

INITIAL_CAPITAL = 10_000_000  # Rs 1 Crore
POSITION_SIZE_PCT = 0.10       # 10% per position
MAX_POSITIONS = 10
COMMISSION_PCT = 0.001         # 0.1%
SLIPPAGE_PCT = 0.001           # 0.1%

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nr4id_sweep_results.csv')

FIELDNAMES = [
    'label', 'pattern', 'exit_type', 'risk_reward', 'max_hold',
    'total_trades', 'long_trades', 'short_trades',
    'win_rate', 'long_wr', 'short_wr',
    'profit_factor', 'cagr', 'sharpe', 'sortino', 'calmar',
    'max_drawdown', 'avg_win', 'avg_loss', 'avg_rr', 'avg_bars_held',
    'total_pnl', 'total_pnl_pct',
    'exit_reasons',
    # Year-by-year columns
    'y2018_trades', 'y2018_wr', 'y2018_pf', 'y2018_pnl_pct',
    'y2019_trades', 'y2019_wr', 'y2019_pf', 'y2019_pnl_pct',
    'y2020_trades', 'y2020_wr', 'y2020_pf', 'y2020_pnl_pct',
    'y2021_trades', 'y2021_wr', 'y2021_pf', 'y2021_pnl_pct',
    'y2022_trades', 'y2022_wr', 'y2022_pf', 'y2022_pnl_pct',
    'y2023_trades', 'y2023_wr', 'y2023_pf', 'y2023_pnl_pct',
    'y2024_trades', 'y2024_wr', 'y2024_pf', 'y2024_pnl_pct',
    'y2025_trades', 'y2025_wr', 'y2025_pf', 'y2025_pnl_pct',
]

# ============================================================================
# Pattern detection helpers
# ============================================================================

def detect_nr4(df: pd.DataFrame) -> pd.Series:
    """NR4: Today's range is narrowest of last 4 days."""
    ranges = df['high'] - df['low']
    nr4 = pd.Series(False, index=df.index)
    for i in range(3, len(df)):
        today_range = ranges.iloc[i]
        if today_range < ranges.iloc[i-1] and today_range < ranges.iloc[i-2] and today_range < ranges.iloc[i-3]:
            nr4.iloc[i] = True
    return nr4


def detect_inside_day(df: pd.DataFrame) -> pd.Series:
    """Inside Day: Today's high < yesterday's high AND today's low > yesterday's low."""
    inside = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    return inside.fillna(False)


def precompute_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add NR4, ID, NR4+ID columns and ATR to dataframe."""
    df = df.copy()
    df['nr4'] = detect_nr4(df)
    df['inside_day'] = detect_inside_day(df)
    df['nr4id'] = df['nr4'] & df['inside_day']
    df['atr14'] = calc_atr(df, 14)
    return df


# ============================================================================
# Strategy configs
# ============================================================================

CONFIGS = [
    # (label, pattern_col, exit_type, risk_reward, max_hold, atr_trail_mult)
    ('NR4_BREAK_RR2_MH5',    'nr4',    'fixed', 2.0, 5,  None),
    ('NR4_BREAK_RR3_MH7',    'nr4',    'fixed', 3.0, 7,  None),
    ('ID_BREAK_RR2_MH5',     'inside_day', 'fixed', 2.0, 5,  None),
    ('ID_BREAK_RR3_MH7',     'inside_day', 'fixed', 3.0, 7,  None),
    ('NR4ID_BREAK_RR2_MH5',  'nr4id',  'fixed', 2.0, 5,  None),
    ('NR4ID_BREAK_RR3_MH7',  'nr4id',  'fixed', 3.0, 7,  None),
    ('NR4_ATR_TRAIL_MH10',   'nr4',    'atr_trail', None, 10, 2.0),
    ('NR4ID_ATR_TRAIL_MH10', 'nr4id',  'atr_trail', None, 10, 2.0),
]


# ============================================================================
# Run one config across all symbols
# ============================================================================

def run_config(
    label: str,
    pattern_col: str,
    exit_type: str,
    risk_reward: Optional[float],
    max_hold: int,
    atr_trail_mult: Optional[float],
    all_data: Dict[str, pd.DataFrame],
    prebuilt: dict = None,
) -> dict:
    """Run a single config across all symbols. Uses prebuilt arrays for speed."""
    from services.intraday_backtest_engine import ExitType

    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        position_size_pct=POSITION_SIZE_PCT,
        max_positions=MAX_POSITIONS,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        mode='cash',
        fixed_sizing=True,
    )

    all_dates = prebuilt['all_dates']
    sym_date_idx = prebuilt['sym_date_idx']  # sym -> {date_str: row_index}
    sym_arrays = prebuilt['sym_arrays']      # sym -> {open, high, low, close, atr14, dates, pattern_cols}

    pending_signals: Dict[str, dict] = {}

    bar_counter = 0
    for date_str in all_dates:
        bar_counter += 1
        close_prices = {}

        for sym in prebuilt['symbols']:
            didx = sym_date_idx[sym]
            if date_str not in didx:
                continue

            ri = didx[date_str]
            sa = sym_arrays[sym]
            o = sa['open'][ri]
            h = sa['high'][ri]
            l = sa['low'][ri]
            c = sa['close'][ri]
            atr_val = sa['atr14'][ri]
            if np.isnan(atr_val):
                atr_val = None
            close_prices[sym] = c

            # --- Check exits first ---
            if sym in engine.positions:
                engine.check_exits(sym, bar_counter, date_str, h, l, c, atr=atr_val)

            # --- Check pending signal ---
            if sym in pending_signals and sym not in engine.positions:
                ps = pending_signals[sym]
                pattern_high = ps['high']
                pattern_low = ps['low']

                long_entry = h > pattern_high
                short_entry = l < pattern_low

                if long_entry and short_entry:
                    del pending_signals[sym]
                elif long_entry:
                    entry_price = pattern_high
                    sl = pattern_low
                    if exit_type == 'fixed':
                        risk = entry_price - sl
                        tp = entry_price + risk * risk_reward
                        signal = TradeSignal(Direction.LONG, entry_price, sl, target=tp, max_hold_bars=max_hold)
                    else:
                        signal = TradeSignal(Direction.LONG, entry_price, sl, trailing_atr_mult=atr_trail_mult, max_hold_bars=max_hold)

                    if h >= entry_price and l <= entry_price:
                        engine.open_position(sym, signal, bar_counter, date_str)
                    elif o > pattern_high:
                        signal.entry_price = o
                        if exit_type == 'fixed':
                            risk = signal.entry_price - sl
                            signal.target = signal.entry_price + risk * risk_reward
                        engine.open_position(sym, signal, bar_counter, date_str)
                    del pending_signals[sym]
                elif short_entry:
                    entry_price = pattern_low
                    sl = pattern_high
                    if exit_type == 'fixed':
                        risk = sl - entry_price
                        tp = entry_price - risk * risk_reward
                        signal = TradeSignal(Direction.SHORT, entry_price, sl, target=tp, max_hold_bars=max_hold)
                    else:
                        signal = TradeSignal(Direction.SHORT, entry_price, sl, trailing_atr_mult=atr_trail_mult, max_hold_bars=max_hold)

                    if l <= entry_price and h >= entry_price:
                        engine.open_position(sym, signal, bar_counter, date_str)
                    elif o < pattern_low:
                        signal.entry_price = o
                        if exit_type == 'fixed':
                            risk = sl - signal.entry_price
                            signal.target = signal.entry_price - risk * risk_reward
                        engine.open_position(sym, signal, bar_counter, date_str)
                    del pending_signals[sym]
                else:
                    del pending_signals[sym]

            # --- Detect new pattern ---
            if sa[pattern_col][ri] and sym not in engine.positions:
                pending_signals[sym] = {'high': h, 'low': l, 'atr': atr_val}

        engine.update_equity(date_str, close_prices)

    # Close remaining positions
    for sym in list(engine.positions.keys()):
        if sym in all_data:
            df = all_data[sym]
            last_row = df.iloc[-1]
            engine.close_position(sym, last_row['close'], bar_counter, last_row['date_str'], ExitType.EOD)

    result = engine.get_results()
    return result


def result_to_row(label: str, config_tuple, result: BacktestResult) -> dict:
    """Convert result to CSV row."""
    _, pattern_col, exit_type, rr, mh, atr_mult = config_tuple

    row = {
        'label': label,
        'pattern': pattern_col,
        'exit_type': exit_type,
        'risk_reward': rr if rr else f'ATR{atr_mult}',
        'max_hold': mh,
        'total_trades': result.total_trades,
        'long_trades': result.long_trades,
        'short_trades': result.short_trades,
        'win_rate': round(result.win_rate, 2),
        'long_wr': round(result.long_win_rate, 2),
        'short_wr': round(result.short_win_rate, 2),
        'profit_factor': round(result.profit_factor, 2),
        'cagr': round(result.cagr, 2),
        'sharpe': round(result.sharpe_ratio, 2),
        'sortino': round(result.sortino_ratio, 2),
        'calmar': round(result.calmar_ratio, 2),
        'max_drawdown': round(result.max_drawdown, 2),
        'avg_win': round(result.avg_win, 2),
        'avg_loss': round(result.avg_loss, 2),
        'avg_rr': round(result.avg_rr, 2),
        'avg_bars_held': round(result.avg_bars_held, 1),
        'total_pnl': round(result.total_pnl, 0),
        'total_pnl_pct': round(result.total_pnl_pct, 2),
        'exit_reasons': str(result.exit_reasons),
    }

    # Year-by-year
    for y in range(2018, 2026):
        if y in result.yearly_stats:
            ys = result.yearly_stats[y]
            row[f'y{y}_trades'] = ys['trades']
            row[f'y{y}_wr'] = round(ys['win_rate'], 1)
            row[f'y{y}_pf'] = round(ys['profit_factor'], 2)
            row[f'y{y}_pnl_pct'] = round(ys['pnl_pct'], 2)
        else:
            row[f'y{y}_trades'] = 0
            row[f'y{y}_wr'] = 0
            row[f'y{y}_pf'] = 0
            row[f'y{y}_pnl_pct'] = 0

    return row


# ============================================================================
# Main
# ============================================================================

def main():
    print('=' * 70)
    print('NR4/ID Breakout Strategy Sweep')
    print(f'Symbols: {len(SYMBOLS)} | Capital: {INITIAL_CAPITAL:,.0f}')
    print(f'Configs: {len(CONFIGS)}')
    print('=' * 70)

    # Check already done
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}
        if done:
            print(f'Skipping {len(done)} already-completed configs')

    remaining = [(l, *rest) for l, *rest in CONFIGS if l not in done]
    if not remaining:
        print('All configs already completed!')
        # Print existing results
        with open(OUTPUT_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                print(f"  {row['label']:30s} Trades={row['total_trades']:>5s} WR={row['win_rate']:>6s}% "
                      f"PF={row['profit_factor']:>5s} CAGR={row['cagr']:>7s}% MaxDD={row['max_drawdown']:>6s}%")
        return

    # Load data once
    t0 = time.time()
    print(f'\nLoading daily data for {len(SYMBOLS)} symbols...', end='', flush=True)
    all_data_raw = load_data_from_db(SYMBOLS, 'day', '2018-01-01', '2025-12-31')
    print(f' {time.time()-t0:.1f}s | {len(all_data_raw)} symbols loaded')

    # Precompute patterns for all symbols
    print('Precomputing NR4/ID patterns and ATR...', end='', flush=True)
    t1 = time.time()
    all_data = {}
    for sym, df in all_data_raw.items():
        all_data[sym] = precompute_patterns(df)
    print(f' {time.time()-t1:.1f}s')

    # Pre-build numpy arrays and date index for fast lookups
    print('Building fast lookup arrays...', end='', flush=True)
    t2 = time.time()
    all_dates_set = set()
    sym_date_idx = {}
    sym_arrays = {}
    symbols_list = list(all_data.keys())

    for sym, df in all_data.items():
        dates = df['date_str'].values
        all_dates_set.update(dates.tolist())
        sym_date_idx[sym] = {d: i for i, d in enumerate(dates)}
        sym_arrays[sym] = {
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'atr14': df['atr14'].values,
            'nr4': df['nr4'].values,
            'inside_day': df['inside_day'].values,
            'nr4id': df['nr4id'].values,
        }

    all_dates_sorted = sorted(all_dates_set)
    prebuilt = {
        'all_dates': all_dates_sorted,
        'sym_date_idx': sym_date_idx,
        'sym_arrays': sym_arrays,
        'symbols': symbols_list,
    }
    print(f' {time.time()-t2:.1f}s | {len(all_dates_sorted)} dates')

    # Write header if needed
    if not done:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Run configs
    total = len(remaining)
    for i, (label, pattern_col, exit_type, rr, mh, atr_mult) in enumerate(remaining):
        t_start = time.time()
        print(f'\n[{i+1}/{total}] {label} ...', end='', flush=True)

        result = run_config(label, pattern_col, exit_type, rr, mh, atr_mult, all_data, prebuilt)
        elapsed = time.time() - t_start

        row = result_to_row(label, (label, pattern_col, exit_type, rr, mh, atr_mult), result)

        # Write incrementally
        with open(OUTPUT_CSV, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(row)

        print(f' {elapsed:.1f}s | Trades={result.total_trades} WR={result.win_rate:.1f}% '
              f'PF={result.profit_factor:.2f} CAGR={result.cagr:.2f}% MaxDD={result.max_drawdown:.1f}%')

        # Print year-by-year
        if result.yearly_stats:
            print(f'    {"Year":>6} {"Trades":>7} {"WR":>6} {"PF":>6} {"PnL%":>8} {"L/S":>7}')
            for year, ys in sorted(result.yearly_stats.items()):
                print(f'    {year:>6} {ys["trades"]:>7} {ys["win_rate"]:>5.1f}% '
                      f'{ys["profit_factor"]:>5.2f} {ys["pnl_pct"]:>+7.2f}% '
                      f'{ys["long_trades"]}/{ys["short_trades"]:>3}')

        print(f'    Exits: {result.exit_reasons}')
        sys.stdout.flush()

    # Final summary
    print('\n' + '=' * 70)
    print('FINAL RESULTS SUMMARY')
    print('=' * 70)
    print(f'{"Label":30s} {"Trades":>6} {"WR":>6} {"PF":>6} {"CAGR":>7} {"MaxDD":>6} {"Sharpe":>6} {"AvgRR":>5}')
    print('-' * 70)
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            print(f'{row["label"]:30s} {row["total_trades"]:>6s} {row["win_rate"]:>5s}% '
                  f'{row["profit_factor"]:>5s} {row["cagr"]:>6s}% {row["max_drawdown"]:>5s}% '
                  f'{row["sharpe"]:>6s} {row["avg_rr"]:>5s}')
    print('=' * 70)


if __name__ == '__main__':
    main()
