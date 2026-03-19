"""
Fast Range Detection Backtest (Vectorized)
==========================================
Uses numpy vectorization instead of Python loops for 50x speedup.
"""

import os, sys, csv, time
import pandas as pd
import numpy as np
import logging

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.range_detection_engine import (
    get_all_timeframes, generate_signals, calc_atr
)

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'range_detection_results.csv')

SYMBOLS = ['NIFTY50', 'BANKNIFTY']
HOLD_PERIODS = [1, 3, 5, 7, 10, 15, 20]
ZONE_WIDTHS = [1.0, 1.5, 2.0, 2.5]

FIELDNAMES = [
    'symbol', 'timeframe', 'signal', 'category', 'zone_width_atr',
    'hold_period', 'zone_hold_rate', 'avg_zone_duration',
    'avg_break_magnitude', 'max_break_magnitude',
    'signal_frequency', 'total_signals', 'total_bars', 'score',
]


def compute_score(hold_rate, freq, break_mag, duration, hold_period):
    hold_score = hold_rate / 100
    freq_score = min(freq / 50, 1.0)
    break_score = max(0, 1 - break_mag / 5)
    duration_score = min(duration / hold_period, 1.0) if hold_period > 0 else 0
    return 0.40 * hold_score + 0.20 * freq_score + 0.20 * break_score + 0.20 * duration_score


def vectorized_backtest(close_arr, high_arr, low_arr, atr_arr, signal_arr,
                        zone_width_mult, hold_periods):
    """Fully vectorized backtest using numpy."""
    n = len(close_arr)
    signal_indices = np.where(signal_arr == 1)[0]

    if len(signal_indices) == 0:
        return {}

    results = {}
    for hp in hold_periods:
        # Filter signals that have enough forward data
        valid_mask = signal_indices + hp < n
        valid_signals = signal_indices[valid_mask]

        if len(valid_signals) == 0:
            continue

        entry_closes = close_arr[valid_signals]
        entry_atrs = atr_arr[valid_signals]

        # Skip signals with zero/nan ATR
        valid_atr_mask = (entry_atrs > 0) & (~np.isnan(entry_atrs))
        valid_signals = valid_signals[valid_atr_mask]
        entry_closes = entry_closes[valid_atr_mask]
        entry_atrs = entry_atrs[valid_atr_mask]

        if len(valid_signals) == 0:
            continue

        zone_widths = zone_width_mult * entry_atrs
        zone_uppers = entry_closes + zone_widths
        zone_lowers = entry_closes - zone_widths

        # For each signal, compute max high and min low over forward period
        max_highs = np.zeros(len(valid_signals))
        min_lows = np.zeros(len(valid_signals))
        first_break_bars = np.full(len(valid_signals), hp, dtype=float)

        for j in range(1, hp + 1):
            fwd_indices = valid_signals + j
            fwd_highs = high_arr[fwd_indices]
            fwd_lows = low_arr[fwd_indices]

            max_highs = np.maximum(max_highs, fwd_highs)
            min_lows = np.where(min_lows == 0, fwd_lows, np.minimum(min_lows, fwd_lows))

            # Track first break bar
            broke_up = fwd_highs > zone_uppers
            broke_down = fwd_lows < zone_lowers
            broke = broke_up | broke_down
            first_break_bars = np.where(
                (first_break_bars == hp) & broke,
                j - 1,  # 0-indexed bar when it broke
                first_break_bars
            )

        # Did price stay in zone?
        stayed = (max_highs <= zone_uppers) & (min_lows >= zone_lowers)

        # Break magnitudes
        upside_breaks = np.maximum(0, (max_highs - zone_uppers) / entry_closes * 100)
        downside_breaks = np.maximum(0, (zone_lowers - min_lows) / entry_closes * 100)
        break_mags = np.maximum(upside_breaks, downside_breaks)
        break_mags_failed = break_mags[~stayed]

        total = len(valid_signals)
        in_zone = stayed.sum()

        results[hp] = {
            'zone_hold_rate': in_zone / total * 100,
            'avg_zone_duration': float(np.mean(first_break_bars)),
            'avg_break_magnitude': float(np.mean(break_mags_failed)) if len(break_mags_failed) > 0 else 0,
            'max_break_magnitude': float(np.max(break_mags_failed)) if len(break_mags_failed) > 0 else 0,
            'total_tested': total,
        }

    return results


def main():
    start_time = time.time()

    # Write header
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    total_rows = 0

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")

        all_timeframes = get_all_timeframes(symbol)

        for tf_name, df in all_timeframes.items():
            if len(df) < 60:
                print(f"  Skipping {tf_name}: only {len(df)} bars")
                continue

            tf_start = time.time()
            print(f"  {tf_name} ({len(df)} bars)...", end='', flush=True)

            # Pre-compute arrays
            atr = calc_atr(df, 14)
            close_arr = df['close'].values.astype(float)
            high_arr = df['high'].values.astype(float)
            low_arr = df['low'].values.astype(float)
            atr_arr = atr.values.astype(float)

            # Generate all signals
            signals = generate_signals(df)
            total_bars = len(df)

            rows_batch = []

            for signal in signals:
                sig_series = signal.signal_series
                if sig_series is None or sig_series.empty:
                    continue

                signal_arr = sig_series.reindex(df.index).fillna(0).values.astype(int)
                total_signals = signal_arr.sum()
                signal_freq = total_signals / total_bars * 100

                if total_signals < 5:
                    continue

                for zw in ZONE_WIDTHS:
                    results = vectorized_backtest(
                        close_arr, high_arr, low_arr, atr_arr, signal_arr,
                        zw, HOLD_PERIODS
                    )

                    for hp, metrics in results.items():
                        score = compute_score(
                            metrics['zone_hold_rate'],
                            signal_freq,
                            metrics['avg_break_magnitude'],
                            metrics['avg_zone_duration'],
                            hp
                        )

                        rows_batch.append({
                            'symbol': symbol,
                            'timeframe': tf_name,
                            'signal': signal.name,
                            'category': signal.category,
                            'zone_width_atr': zw,
                            'hold_period': hp,
                            'zone_hold_rate': round(metrics['zone_hold_rate'], 2),
                            'avg_zone_duration': round(metrics['avg_zone_duration'], 2),
                            'avg_break_magnitude': round(metrics['avg_break_magnitude'], 2),
                            'max_break_magnitude': round(metrics['max_break_magnitude'], 2),
                            'signal_frequency': round(signal_freq, 2),
                            'total_signals': total_signals,
                            'total_bars': total_bars,
                            'score': round(score, 4),
                        })

            # Write batch
            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerows(rows_batch)

            total_rows += len(rows_batch)
            tf_elapsed = time.time() - tf_start
            print(f" {len(rows_batch)} rows in {tf_elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE! {total_rows} results in {elapsed:.0f}s")
    print(f"Output: {OUTPUT_CSV}")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    df_all = pd.read_csv(OUTPUT_CSV)

    # --- TOP SIGNALS FOR OPTIONS STRATEGY TIMEFRAMES ---
    # For weekly options: hold_period 5 (daily) is key
    # For bi-weekly: hold_period 10
    # For monthly: hold_period 15-20

    for options_window, hp_filter in [('Weekly', 5), ('Bi-Weekly', 10), ('Monthly', 20)]:
        print(f"\n{'='*60}")
        print(f"TOP 15 SIGNALS FOR {options_window.upper()} OPTIONS (hold_period={hp_filter})")
        print(f"{'='*60}")

        mask = df_all['hold_period'] == hp_filter
        top = df_all[mask].nlargest(15, 'score')

        print(f"{'Symbol':<10} {'TF':<8} {'Signal':<22} {'ZW':<5} {'Hold%':<7} "
              f"{'Freq%':<7} {'AvgBrk':<7} {'MaxBrk':<7} {'Score':<6}")
        print("-" * 90)

        for _, row in top.iterrows():
            print(f"{row['symbol']:<10} {row['timeframe']:<8} {row['signal']:<22} "
                  f"{row['zone_width_atr']:<5.1f} {row['zone_hold_rate']:<7.1f} "
                  f"{row['signal_frequency']:<7.1f} {row['avg_break_magnitude']:<7.2f} "
                  f"{row['max_break_magnitude']:<7.2f} {row['score']:<6.4f}")

    # --- TIMEFRAME RANKING ---
    print(f"\n{'='*60}")
    print("TIMEFRAME RANKING (avg score across all signals, HP >= 5)")
    print(f"{'='*60}")

    mask = df_all['hold_period'] >= 5
    tf_rank = df_all[mask].groupby(['symbol', 'timeframe']).agg({
        'score': ['mean', 'max', 'std'],
        'zone_hold_rate': ['mean', 'max'],
        'avg_break_magnitude': 'mean',
        'signal_frequency': 'mean',
    }).round(3)
    tf_rank.columns = ['_'.join(col) for col in tf_rank.columns]
    tf_rank = tf_rank.sort_values('score_mean', ascending=False)
    print(tf_rank.to_string())

    # --- BEST SIGNAL PER TIMEFRAME ---
    print(f"\n{'='*60}")
    print("BEST SIGNAL PER TIMEFRAME (HP=5, highest score)")
    print(f"{'='*60}")

    mask = df_all['hold_period'] == 5
    for sym in SYMBOLS:
        print(f"\n  {sym}:")
        sym_mask = mask & (df_all['symbol'] == sym)
        best = df_all[sym_mask].sort_values('score', ascending=False).groupby('timeframe').first()
        best = best[['signal', 'zone_width_atr', 'zone_hold_rate', 'signal_frequency',
                      'avg_break_magnitude', 'score']].sort_values('score', ascending=False)
        print(best.to_string())

    # --- NIFTY vs BANKNIFTY COMPARISON ---
    print(f"\n{'='*60}")
    print("NIFTY vs BANKNIFTY COMPARISON (avg across all)")
    print(f"{'='*60}")

    comparison = df_all.groupby('symbol').agg({
        'score': 'mean',
        'zone_hold_rate': 'mean',
        'avg_break_magnitude': 'mean',
    }).round(3)
    print(comparison.to_string())

    # --- CATEGORY RANKING ---
    print(f"\n{'='*60}")
    print("SIGNAL CATEGORY RANKING (HP >= 5)")
    print(f"{'='*60}")

    mask = df_all['hold_period'] >= 5
    cat_rank = df_all[mask].groupby(['symbol', 'category']).agg({
        'score': ['mean', 'max'],
        'zone_hold_rate': 'mean',
    }).round(3)
    cat_rank.columns = ['_'.join(col) for col in cat_rank.columns]
    cat_rank = cat_rank.sort_values('score_mean', ascending=False)
    print(cat_rank.to_string())

    # --- ZONE WIDTH ANALYSIS ---
    print(f"\n{'='*60}")
    print("ZONE WIDTH ANALYSIS (which ATR multiple works best?)")
    print(f"{'='*60}")

    zw_analysis = df_all.groupby(['symbol', 'zone_width_atr']).agg({
        'score': 'mean',
        'zone_hold_rate': 'mean',
        'avg_break_magnitude': 'mean',
    }).round(3)
    print(zw_analysis.to_string())


if __name__ == '__main__':
    main()
