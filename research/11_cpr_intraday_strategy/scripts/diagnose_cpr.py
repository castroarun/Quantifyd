"""
CPR Strategy Diagnostic
=======================

Diagnoses why no trades are being generated in the CPR backtest.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.intraday_data_bridge import get_intraday_bridge
from services.data_manager import get_data_manager
import pandas as pd

def diagnose():
    print("=" * 70)
    print("CPR STRATEGY DIAGNOSTIC")
    print("=" * 70)

    # Load data
    bridge = get_intraday_bridge()
    data_manager = get_data_manager()

    symbol = 'RELIANCE'

    # Get date range
    min_date, max_date = bridge.get_date_range(symbol)
    print(f"\n{symbol} 30-min data range: {min_date} to {max_date}")

    # Load intraday data
    intraday_df = bridge.load_30min_data(symbol, min_date, max_date)
    print(f"Loaded {len(intraday_df)} 30-min candles")

    # Load daily data
    daily_df = data_manager.load_data(symbol, 'day', min_date, max_date)
    print(f"Loaded {len(daily_df)} daily candles")

    if intraday_df.empty:
        print("ERROR: No intraday data!")
        return

    # Show sample of intraday data
    print("\n--- Sample 30-min data (first 10 rows) ---")
    print(intraday_df.head(10))

    # Get unique dates in intraday data
    intraday_df['date_only'] = intraday_df.index.date
    unique_dates = intraday_df['date_only'].unique()
    print(f"\nUnique trading days in intraday data: {len(unique_dates)}")
    print(f"First: {unique_dates[0]}, Last: {unique_dates[-1]}")

    # Calculate weekly CPR for each week
    print("\n--- Weekly CPR Analysis ---")

    # Group by week
    intraday_df['week'] = intraday_df.index.isocalendar().week
    intraday_df['year'] = intraday_df.index.year

    weeks = intraday_df.groupby(['year', 'week']).agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first'
    })

    print(f"\nWeeks in data: {len(weeks)}")
    print(weeks)

    # For each week, calculate CPR from PREVIOUS week
    print("\n--- CPR Calculation Check ---")

    prev_high, prev_low, prev_close = None, None, None

    for idx, (week_key, week_data) in enumerate(weeks.iterrows()):
        year, week_num = week_key

        if prev_high is not None:
            # Calculate CPR from previous week
            pivot = (prev_high + prev_low + prev_close) / 3
            bc = (prev_high + prev_low) / 2
            tc = 2 * pivot - bc
            cpr_width = abs(tc - bc)

            # Get first candle of current week
            week_mask = (intraday_df['year'] == year) & (intraday_df['week'] == week_num)
            week_candles = intraday_df[week_mask]

            if not week_candles.empty:
                first_candle = week_candles.iloc[0]
                first_close = first_candle['close']
                first_time = week_candles.index[0]

                # Check entry condition
                cpr_middle = (tc + bc) / 2
                cpr_width_pct = (cpr_width / cpr_middle) * 100
                is_below_cpr = first_close < min(tc, bc)
                is_narrow = cpr_width_pct < 0.5

                print(f"\nWeek {week_num} ({first_time.date()}):")
                print(f"  Previous Week: H={prev_high:.2f}, L={prev_low:.2f}, C={prev_close:.2f}")
                print(f"  CPR: Pivot={pivot:.2f}, BC={bc:.2f}, TC={tc:.2f}")
                print(f"  CPR Width: {cpr_width:.2f} ({cpr_width_pct:.3f}%)")
                print(f"  First Candle: {first_time} Close={first_close:.2f}")
                print(f"  Entry Check: Below CPR={is_below_cpr}, Narrow={is_narrow}")

                if is_narrow:
                    print(f"  --> SKIP: Narrow CPR (< 0.5%)")
                elif not is_below_cpr:
                    print(f"  --> NO ENTRY: First candle ({first_close:.2f}) NOT below CPR ({min(tc, bc):.2f})")
                else:
                    print(f"  --> VALID ENTRY SIGNAL!")

        # Save for next iteration
        prev_high = week_data['high']
        prev_low = week_data['low']
        prev_close = week_data['close']

    # Also check if the daily data has the right date alignment
    print("\n--- Daily Data Sample ---")
    if daily_df is not None and not daily_df.empty:
        print(daily_df.head(10))
    else:
        print("No daily data!")


if __name__ == '__main__':
    diagnose()