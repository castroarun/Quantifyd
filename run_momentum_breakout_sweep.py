"""
Momentum-Filtered Breakout Strategy Sweep
==========================================
Tests Donchian breakout with momentum (ROC) and volume filters on daily F&O stocks.
Addresses the #1 problem of no stock selection by filtering for strong momentum.

Variants:
  a. ROC_5_VOL1.5_TR2.5_MH15  (base case)
  b. ROC_10_VOL2.0_TR3.0_MH20 (higher thresholds)
  c. ROC_3_VOL1.2_TR2.0_MH10  (lower thresholds)
  d. ROC_5_VOL1.5_TR2.5_MH15_NIFTY (with Nifty regime filter)
"""

import os
import sys
import csv
import time
import logging
import numpy as np
import pandas as pd

logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.intraday_backtest_engine import (
    IntradayBacktestEngine, TradeSignal, Direction, ExitType,
    load_data_from_db, get_all_symbols_for_timeframe
)
from services.technical_indicators import calc_atr, calc_donchian_channels, calc_ema
from services.data_manager import FNO_LOT_SIZES

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'momentum_breakout_results.csv')

FIELDNAMES = [
    'strategy', 'params', 'timeframe', 'n_symbols', 'period',
    'total_trades', 'long_trades', 'short_trades',
    'win_rate', 'long_wr', 'short_wr',
    'total_pnl', 'pnl_pct', 'profit_factor',
    'cagr', 'max_drawdown', 'sharpe', 'sortino', 'calmar',
    'avg_win', 'avg_loss', 'avg_rr', 'avg_bars_held',
    'exit_reasons',
]
# Add year-by-year columns
for y in range(2018, 2026):
    FIELDNAMES.extend([f'y{y}_trades', f'y{y}_wr', f'y{y}_pf', f'y{y}_pnl_pct'])


def compute_indicators(df, donchian_period=20, atr_period=14, roc_period=20, vol_avg_period=20):
    """Pre-compute all indicators for a stock."""
    df = df.copy()
    # Donchian channels (use shifted to avoid lookahead)
    _, dc_upper, dc_lower = calc_donchian_channels(df, donchian_period)
    df['dc_upper_prev'] = dc_upper.shift(1)
    df['dc_lower_prev'] = dc_lower.shift(1)

    # ATR
    df['atr'] = calc_atr(df, atr_period)

    # ROC (Rate of Change)
    df['roc'] = (df['close'] / df['close'].shift(roc_period) - 1) * 100

    # Volume average
    df['vol_avg'] = df['volume'].rolling(vol_avg_period).mean()
    df['vol_ratio'] = df['volume'] / df['vol_avg']

    return df


def run_strategy(
    all_data,          # Dict[symbol, DataFrame with indicators]
    nifty_data,        # DataFrame for NIFTYBEES or None
    roc_thresh,        # ROC threshold for momentum filter
    vol_mult,          # Volume multiplier threshold
    trail_atr_mult,    # ATR trailing stop multiplier
    max_hold,          # Max hold days
    use_nifty_filter,  # Whether to use Nifty regime filter
    label,
):
    """Run momentum breakout strategy across all symbols."""
    engine = IntradayBacktestEngine(
        initial_capital=10_000_000,
        position_size_pct=0.10,
        max_positions=10,
        commission_pct=0.001,
        slippage_pct=0.001,
        mode='cash',
        fixed_sizing=True,
    )

    # Precompute Nifty regime (above 200 SMA = bull)
    nifty_bull = {}
    if nifty_data is not None and use_nifty_filter:
        nifty_sma200 = calc_ema(nifty_data['close'], 200)  # Use EMA as SMA proxy
        for i in range(len(nifty_data)):
            date_str = nifty_data['date_str'].iloc[i]
            nifty_bull[date_str] = nifty_data['close'].iloc[i] > nifty_sma200.iloc[i]

    # Build a combined date-indexed structure for all symbols
    # Collect all unique dates
    all_dates = set()
    for sym, df in all_data.items():
        all_dates.update(df['date_str'].values)
    all_dates = sorted(all_dates)

    # For each date, process exits first, then entries
    for date_str in all_dates:
        prices_today = {}

        # Phase 1: Check exits for open positions
        for sym in list(engine.positions.keys()):
            if sym not in all_data:
                continue
            df = all_data[sym]
            mask = df['date_str'] == date_str
            if not mask.any():
                continue
            idx = mask.values.nonzero()[0][0]
            row = df.iloc[idx]

            prices_today[sym] = row['close']
            atr_val = row['atr'] if not pd.isna(row['atr']) else None

            engine.check_exits(
                symbol=sym,
                bar_idx=idx,
                bar_date=date_str,
                high=row['high'],
                low=row['low'],
                close=row['close'],
                atr=atr_val,
            )

        # Phase 2: Check entries
        # Nifty regime filter
        if use_nifty_filter and nifty_data is not None:
            is_bull = nifty_bull.get(date_str, True)
        else:
            is_bull = True  # No filter = always allow

        for sym, df in all_data.items():
            if sym in engine.positions:
                continue
            mask = df['date_str'] == date_str
            if not mask.any():
                continue
            idx = mask.values.nonzero()[0][0]
            row = df.iloc[idx]

            # Skip if indicators not ready
            if pd.isna(row.get('dc_upper_prev')) or pd.isna(row.get('atr')) or pd.isna(row.get('roc')) or pd.isna(row.get('vol_avg')):
                continue

            atr_val = row['atr']
            close = row['close']
            vol_ratio = row['vol_ratio'] if not pd.isna(row['vol_ratio']) else 0

            # LONG entry: close > prev Donchian upper + ROC > thresh + volume > mult * avg
            if (is_bull and
                close > row['dc_upper_prev'] and
                row['roc'] > roc_thresh and
                vol_ratio > vol_mult and
                atr_val > 0):

                sl = close - 2.0 * atr_val
                signal = TradeSignal(
                    direction=Direction.LONG,
                    entry_price=close,
                    stop_loss=sl,
                    target=None,
                    trailing_atr_mult=trail_atr_mult,
                    max_hold_bars=max_hold,
                )
                engine.open_position(sym, signal, idx, date_str)
                prices_today[sym] = close

            # SHORT entry: close < prev Donchian lower + volume + NOT bull regime
            elif (not is_bull and
                  close < row['dc_lower_prev'] and
                  vol_ratio > vol_mult and
                  row['roc'] < -roc_thresh and
                  atr_val > 0):

                sl = close + 2.0 * atr_val
                signal = TradeSignal(
                    direction=Direction.SHORT,
                    entry_price=close,
                    stop_loss=sl,
                    target=None,
                    trailing_atr_mult=trail_atr_mult,
                    max_hold_bars=max_hold,
                )
                engine.open_position(sym, signal, idx, date_str)
                prices_today[sym] = close

        # Update equity
        for sym, df in all_data.items():
            if sym not in prices_today:
                mask = df['date_str'] == date_str
                if mask.any():
                    prices_today[sym] = df.loc[mask, 'close'].iloc[0]

        engine.update_equity(date_str, prices_today)

    return engine.get_results()


def result_to_row(label, params_str, result, n_symbols):
    """Convert BacktestResult to CSV row dict."""
    row = {
        'strategy': f'D_MOM_BREAKOUT_{label}',
        'params': params_str,
        'timeframe': 'day',
        'n_symbols': n_symbols,
        'period': '2018-01 to 2025-12',
        'total_trades': result.total_trades,
        'long_trades': result.long_trades,
        'short_trades': result.short_trades,
        'win_rate': round(result.win_rate, 2),
        'long_wr': round(result.long_win_rate, 2),
        'short_wr': round(result.short_win_rate, 2),
        'total_pnl': round(result.total_pnl, 0),
        'pnl_pct': round(result.total_pnl_pct, 2),
        'profit_factor': round(result.profit_factor, 4),
        'cagr': round(result.cagr, 2),
        'max_drawdown': round(result.max_drawdown, 2),
        'sharpe': round(result.sharpe_ratio, 4),
        'sortino': round(result.sortino_ratio, 4),
        'calmar': round(result.calmar_ratio, 4),
        'avg_win': round(result.avg_win, 2),
        'avg_loss': round(result.avg_loss, 2),
        'avg_rr': round(result.avg_rr, 2),
        'avg_bars_held': round(result.avg_bars_held, 1),
        'exit_reasons': str(result.exit_reasons),
    }

    for y in range(2018, 2026):
        ys = result.yearly_stats.get(y, {})
        row[f'y{y}_trades'] = ys.get('trades', 0)
        row[f'y{y}_wr'] = round(ys.get('win_rate', 0), 2)
        row[f'y{y}_pf'] = round(ys.get('profit_factor', 0), 4)
        row[f'y{y}_pnl_pct'] = round(ys.get('pnl_pct', 0), 2)

    return row


def main():
    t0 = time.time()
    print("=" * 70)
    print("  Momentum-Filtered Breakout Strategy Sweep (Daily, F&O stocks)")
    print("=" * 70)

    # Get F&O symbols with sufficient daily data
    all_daily = get_all_symbols_for_timeframe('day', min_rows=1000)
    fno_symbols = sorted([s for s in FNO_LOT_SIZES.keys() if s in all_daily])[:50]
    print(f"\nF&O symbols with daily data: {len(fno_symbols)}")
    print(f"Symbols: {', '.join(fno_symbols[:10])}... (first 10)")

    # Load data
    print("\nLoading daily data for all symbols...", flush=True)
    t_load = time.time()
    all_raw = load_data_from_db(fno_symbols, 'day', '2018-01-01', '2025-12-31')
    print(f"  Loaded {len(all_raw)} symbols in {time.time()-t_load:.1f}s")

    # Load NIFTYBEES for regime filter
    nifty_raw = load_data_from_db(['NIFTYBEES'], 'day', '2017-01-01', '2025-12-31')
    nifty_data = nifty_raw.get('NIFTYBEES')
    if nifty_data is not None:
        print(f"  NIFTYBEES data: {len(nifty_data)} bars (for regime filter)")
    else:
        print("  WARNING: No NIFTYBEES data found, regime filter will be skipped")

    # Pre-compute indicators for all stocks
    print("Computing indicators...", flush=True)
    t_ind = time.time()
    all_data = {}
    for sym, df in all_raw.items():
        if len(df) < 50:
            continue
        all_data[sym] = compute_indicators(df)
    print(f"  Indicators computed for {len(all_data)} symbols in {time.time()-t_ind:.1f}s")

    n_symbols = len(all_data)

    # Define sweep configs
    configs = [
        ('ROC5_VOL1.5_TR2.5_MH15',      5,  1.5, 2.5, 15, False),
        ('ROC10_VOL2.0_TR3.0_MH20',      10, 2.0, 3.0, 20, False),
        ('ROC3_VOL1.2_TR2.0_MH10',       3,  1.2, 2.0, 10, False),
        ('ROC5_VOL1.5_TR2.5_MH15_NIFTY', 5,  1.5, 2.5, 15, True),
    ]

    # Check already done
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'r') as f:
            done = {row['strategy'] for row in csv.DictReader(f)}
        print(f"\nSkipping {len(done)} already-completed configs")

    # Write header if new file
    if not done:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Run sweep
    print(f"\n{'='*70}")
    print(f"  Running {len(configs)} configurations...")
    print(f"{'='*70}\n")

    for i, (label, roc_thresh, vol_mult, trail_atr, max_hold, use_nifty) in enumerate(configs, 1):
        full_label = f'D_MOM_BREAKOUT_{label}'
        if full_label in done:
            print(f"[{i}/{len(configs)}] {label} — SKIPPED (already done)")
            continue

        params_str = (f'{{"roc_thresh": {roc_thresh}, "vol_mult": {vol_mult}, '
                      f'"trail_atr": {trail_atr}, "max_hold": {max_hold}, '
                      f'"nifty_filter": {use_nifty}}}')

        print(f"[{i}/{len(configs)}] {label} ...", end='', flush=True)
        t1 = time.time()

        result = run_strategy(
            all_data=all_data,
            nifty_data=nifty_data,
            roc_thresh=roc_thresh,
            vol_mult=vol_mult,
            trail_atr_mult=trail_atr,
            max_hold=max_hold,
            use_nifty_filter=use_nifty,
            label=label,
        )

        elapsed = time.time() - t1
        row = result_to_row(label, params_str, result, n_symbols)

        # Write incrementally
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        print(f' {elapsed:.0f}s | Trades={result.total_trades} CAGR={result.cagr:.2f}% '
              f'WR={result.win_rate:.1f}% PF={result.profit_factor:.2f} '
              f'MaxDD={result.max_drawdown:.2f}% Sharpe={result.sharpe_ratio:.2f}')
        sys.stdout.flush()

        # Print detailed results
        IntradayBacktestEngine.print_results(result, label)

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
