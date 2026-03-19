"""
Debug CPR Backtest - Step by step tracing
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.intraday_data_bridge import get_intraday_bridge
from services.data_manager import get_data_manager
from services.cpr_covered_call_service import CPRCoveredCallEngine, CPRBacktestConfig
import pandas as pd

def debug_backtest():
    print("=" * 70)
    print("DEBUG CPR BACKTEST")
    print("=" * 70)

    # Load data
    bridge = get_intraday_bridge()
    data_manager = get_data_manager()

    symbols = ['RELIANCE', 'TCS']
    min_date, max_date = bridge.get_date_range(symbols[0])

    print(f"\nData range: {min_date} to {max_date}")

    # Load stock data
    stock_data = {}
    intraday_data = {}

    for symbol in symbols:
        daily_df = data_manager.load_data(symbol, 'day', min_date - timedelta(days=30), max_date)
        if daily_df is not None and not daily_df.empty:
            stock_data[symbol] = daily_df
            print(f"\n{symbol} Daily Data:")
            print(f"  Index type: {type(daily_df.index)}")
            print(f"  Date range: {daily_df.index.min()} to {daily_df.index.max()}")
            print(f"  Sample index: {daily_df.index[:3].tolist()}")

        intra_df = bridge.load_30min_data(symbol, min_date - timedelta(days=14), max_date)
        if not intra_df.empty:
            intraday_data[symbol] = intra_df
            print(f"  Intraday candles: {len(intra_df)}")

    # Create config with very lenient thresholds
    config = CPRBacktestConfig(
        symbols=symbols,
        start_date=min_date,
        end_date=max_date,
        initial_capital=1000000.0,
        dte_min=20,  # Very lenient DTE
        dte_max=60,  # Very lenient DTE
        narrow_cpr_threshold=0.01,  # Almost no filter
        otm_strike_pct=5.0,
        enable_premium_rollout=False,
        premium_erosion_target=50.0,
        dte_exit_threshold=5,
        enable_r1_exit=False,
    )

    print(f"\nConfig: {config}")

    engine = CPRCoveredCallEngine(config)

    # Pre-calculate CPR
    engine._precalculate_weekly_cpr(stock_data, intraday_data)

    print("\n--- Cached Weekly CPR ---")
    for symbol in symbols:
        cpr_cache = engine.weekly_cpr_cache.get(symbol, {})
        print(f"\n{symbol}: {len(cpr_cache)} weeks cached")
        for monday, cpr_data in sorted(cpr_cache.items())[:5]:
            print(f"  {monday.date()}: Pivot={cpr_data.pivot:.2f}, BC={cpr_data.bc:.2f}, TC={cpr_data.tc:.2f}")
            print(f"           CPR Width: {cpr_data.cpr_width_pct:.3f}%, Narrow={cpr_data.is_narrow}")
            print(f"           First Candle: {cpr_data.first_candle_close}, Time: {cpr_data.first_candle_time}")
            print(f"           CPR Bottom: {cpr_data.cpr_bottom:.2f}, Entry Valid: {cpr_data.first_candle_close and cpr_data.first_candle_close < cpr_data.cpr_bottom}")

    # Get trading days
    all_dates = set()
    for symbol, df in stock_data.items():
        dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
        all_dates.update(dates.to_pydatetime())

    trading_days = sorted([d for d in all_dates if config.start_date <= d <= config.end_date])

    print(f"\n--- Trading Days ---")
    print(f"Total: {len(trading_days)}")
    print(f"First: {trading_days[0] if trading_days else 'None'}")
    print(f"Last: {trading_days[-1] if trading_days else 'None'}")

    # Find Mondays
    mondays = [d for d in trading_days if d.weekday() == 0]
    print(f"\nMondays in range: {len(mondays)}")
    for m in mondays[:5]:
        print(f"  {m.date()} (weekday={m.weekday()})")

    # Manual entry check on first Monday
    if mondays:
        test_monday = mondays[0]
        print(f"\n--- Manual Entry Check for {test_monday.date()} ---")

        for symbol in symbols:
            print(f"\n{symbol}:")

            # Get CPR
            cpr_data = engine._get_week_cpr(symbol, test_monday)
            if not cpr_data:
                print("  No CPR data!")
                continue

            print(f"  CPR: Pivot={cpr_data.pivot:.2f}, BC={cpr_data.bc:.2f}, TC={cpr_data.tc:.2f}")
            print(f"  CPR Width: {cpr_data.cpr_width_pct:.3f}%")
            print(f"  Is Narrow (< {config.narrow_cpr_threshold}%): {cpr_data.is_narrow}")
            print(f"  CPR Bottom: {cpr_data.cpr_bottom:.2f}")
            print(f"  First Candle Close: {cpr_data.first_candle_close}")

            if cpr_data.first_candle_close:
                entry_valid = cpr_data.first_candle_close < cpr_data.cpr_bottom
                print(f"  Entry Condition (candle < CPR bottom): {entry_valid}")
                print(f"    {cpr_data.first_candle_close:.2f} < {cpr_data.cpr_bottom:.2f} = {entry_valid}")

    # Now run full backtest
    print("\n" + "=" * 70)
    print("RUNNING FULL BACKTEST")
    print("=" * 70)

    results = engine.run_backtest(stock_data, intraday_data)

    print(f"\n--- Results ---")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(f"Total Return: {results.get('total_return', 0):.2f}%")

    print(f"\n--- Engine Stats ---")
    for key, value in engine.stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    debug_backtest()