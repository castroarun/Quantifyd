"""
Run Range Detection Backtest
============================

Runs all range-detection signals across all available timeframes for NIFTY50
and BANKNIFTY. Tests multiple zone widths and holding periods.

Output: CSV with comprehensive results for analysis.
"""

import os
import sys
import csv
import time
import logging
import pandas as pd
import numpy as np

logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.range_detection_engine import (
    get_all_timeframes, generate_signals, backtest_signal,
    results_to_dataframe, calc_atr
)

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'range_detection_results.csv')

SYMBOLS = ['NIFTY50', 'BANKNIFTY']

# Hold periods in bars (interpretation depends on timeframe)
# For daily: 1,3,5,7,10,15,20 days
# For 60min: these become hours, etc.
HOLD_PERIODS = [1, 3, 5, 7, 10, 15, 20]

# Zone widths to test (multiples of ATR)
ZONE_WIDTHS = [1.0, 1.5, 2.0, 2.5]

FIELDNAMES = [
    'symbol', 'timeframe', 'signal', 'category', 'zone_width_atr',
    'hold_period', 'zone_hold_rate', 'avg_zone_duration',
    'avg_break_magnitude', 'max_break_magnitude',
    'signal_frequency', 'total_signals', 'total_bars',
    # Composite score
    'score',
]


def compute_score(row):
    """
    Composite score combining:
    - Zone hold rate (higher = better, weight 40%)
    - Signal frequency (need enough signals, weight 20%)
    - Break magnitude (lower = better, weight 20%)
    - Zone duration (longer = better, weight 20%)
    """
    hold_rate = row.get('zone_hold_rate', 0)
    freq = row.get('signal_frequency', 0)
    break_mag = row.get('avg_break_magnitude', 0)
    duration = row.get('avg_zone_duration', 0)
    hold_period = row.get('hold_period', 1)

    # Normalize components
    hold_score = hold_rate / 100  # 0-1
    freq_score = min(freq / 50, 1.0)  # Cap at 50% frequency
    break_score = max(0, 1 - break_mag / 5)  # 5% break = 0 score
    duration_score = min(duration / hold_period, 1.0) if hold_period > 0 else 0

    return (0.40 * hold_score +
            0.20 * freq_score +
            0.20 * break_score +
            0.20 * duration_score)


def main():
    start_time = time.time()

    # Write header
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

    total_configs = 0

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")

        all_timeframes = get_all_timeframes(symbol)
        print(f"Available timeframes: {list(all_timeframes.keys())}")

        for tf_name, df in all_timeframes.items():
            if len(df) < 60:
                print(f"\n  Skipping {tf_name}: only {len(df)} bars")
                continue

            print(f"\n  {tf_name} ({len(df)} bars, {df.index.min().date()} to {df.index.max().date()})...")

            # Generate all signals for this timeframe
            signals = generate_signals(df)
            print(f"    Generated {len(signals)} signals")

            for zone_width in ZONE_WIDTHS:
                for signal in signals:
                    result = backtest_signal(
                        df, signal, symbol, tf_name,
                        hold_periods=HOLD_PERIODS,
                        zone_width_atr_mult=zone_width,
                    )

                    # Write results for each hold period
                    for hp in HOLD_PERIODS:
                        if hp not in result.zone_hold_rates:
                            continue

                        row = {
                            'symbol': symbol,
                            'timeframe': tf_name,
                            'signal': signal.name,
                            'category': signal.category,
                            'zone_width_atr': zone_width,
                            'hold_period': hp,
                            'zone_hold_rate': round(result.zone_hold_rates.get(hp, 0), 2),
                            'avg_zone_duration': round(result.avg_zone_durations.get(hp, 0), 2),
                            'avg_break_magnitude': round(result.avg_break_magnitudes.get(hp, 0), 2),
                            'max_break_magnitude': round(result.max_break_magnitudes.get(hp, 0), 2),
                            'signal_frequency': round(result.signal_frequency, 2),
                            'total_signals': result.total_signals,
                            'total_bars': result.total_bars,
                        }
                        row['score'] = round(compute_score(row), 4)

                        with open(OUTPUT_CSV, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                            writer.writerow(row)

                        total_configs += 1

            print(f"    Completed {tf_name} ({total_configs} rows so far)")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE! {total_configs} results in {elapsed:.0f}s")
    print(f"Output: {OUTPUT_CSV}")

    # Quick summary
    df_results = pd.read_csv(OUTPUT_CSV)
    print(f"\n{'='*60}")
    print("TOP 20 SIGNAL COMBINATIONS (by composite score)")
    print(f"{'='*60}")

    # For options strategies, focus on hold periods 5-20
    mask = df_results['hold_period'] >= 5
    top = df_results[mask].nlargest(20, 'score')

    print(f"\n{'Symbol':<10} {'TF':<8} {'Signal':<22} {'ZoneW':<6} {'HP':<4} "
          f"{'Hold%':<7} {'Freq%':<7} {'AvgBrk':<7} {'Score':<6}")
    print("-" * 85)

    for _, row in top.iterrows():
        print(f"{row['symbol']:<10} {row['timeframe']:<8} {row['signal']:<22} "
              f"{row['zone_width_atr']:<6.1f} {row['hold_period']:<4} "
              f"{row['zone_hold_rate']:<7.1f} {row['signal_frequency']:<7.1f} "
              f"{row['avg_break_magnitude']:<7.2f} {row['score']:<6.4f}")

    # Timeframe comparison
    print(f"\n{'='*60}")
    print("TIMEFRAME COMPARISON (avg score, hold_period >= 5)")
    print(f"{'='*60}")

    tf_summary = df_results[mask].groupby(['symbol', 'timeframe']).agg({
        'score': 'mean',
        'zone_hold_rate': 'mean',
        'signal_frequency': 'mean',
        'avg_break_magnitude': 'mean',
    }).round(3)

    print(tf_summary.to_string())

    # Category comparison
    print(f"\n{'='*60}")
    print("SIGNAL CATEGORY COMPARISON (avg score, hold_period >= 5)")
    print(f"{'='*60}")

    cat_summary = df_results[mask].groupby(['symbol', 'category']).agg({
        'score': 'mean',
        'zone_hold_rate': 'mean',
        'signal_frequency': 'mean',
    }).round(3)

    print(cat_summary.to_string())


if __name__ == '__main__':
    main()
