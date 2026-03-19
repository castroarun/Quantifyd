"""
CPR Strategy Optimization Runner
================================

Runs grid search optimization on CPR covered call strategy to find best parameters.
"""

import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from services.intraday_data_bridge import get_intraday_bridge
from services.data_manager import get_data_manager
from services.cpr_strategy_optimizer import CPRStrategyOptimizer, QUICK_OPTIMIZATION_GRID, DEFAULT_OPTIMIZATION_GRID

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def progress_callback(pct: float, msg: str):
    """Print progress updates."""
    print(f"[{pct:5.1f}%] {msg}")


def run_optimization(mode: str = 'medium'):
    """
    Run CPR strategy optimization.

    Args:
        mode: 'quick' (small grid), 'medium' (balanced), 'full' (exhaustive)
    """
    print("=" * 70)
    print("CPR COVERED CALL STRATEGY OPTIMIZER")
    print("=" * 70)

    # Check intraday data availability
    bridge = get_intraday_bridge()
    if not bridge.is_available():
        print(f"\nERROR: Intraday database not found at {bridge.db_path}")
        print("Please ensure quantflow project has 30-minute data available.")
        return None

    # Get data summary
    summary = bridge.get_data_summary()
    print(f"\nIntraday Data Status: {summary.get('status', 'unknown')}")
    if '30min_date_range' in summary:
        print(f"  Date Range: {summary['30min_date_range']['start']} to {summary['30min_date_range']['end']}")

    # Get available symbols
    available_symbols = bridge.get_available_symbols()
    print(f"  Available Symbols: {len(available_symbols)}")

    # Select symbols for optimization (top liquid stocks)
    target_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    symbols = [s for s in target_symbols if s in available_symbols]

    if not symbols:
        print("\nERROR: None of the target symbols have intraday data.")
        print(f"Available: {available_symbols[:10]}...")
        return None

    print(f"\nOptimizing for symbols: {symbols}")

    # Determine date range based on available data
    min_date, max_date = None, None
    for symbol in symbols:
        s_min, s_max = bridge.get_date_range(symbol)
        if s_min and s_max:
            if min_date is None or s_min > min_date:
                min_date = s_min
            if max_date is None or s_max < max_date:
                max_date = s_max

    if not min_date or not max_date:
        print("\nERROR: Could not determine date range from data.")
        return None

    # Use last 1 year of data for optimization
    end_date = max_date
    start_date = max(min_date, end_date - timedelta(days=365))

    print(f"Backtest Period: {start_date.date()} to {end_date.date()}")

    # Load daily stock data
    print("\nLoading daily stock data...")
    data_manager = get_data_manager()
    stock_data = {}

    for symbol in symbols:
        try:
            df = data_manager.load_data(symbol, 'day', start_date, end_date)
            if df is not None and not df.empty:
                stock_data[symbol] = df
                print(f"  {symbol}: {len(df)} daily candles")
        except Exception as e:
            print(f"  {symbol}: Failed to load - {e}")

    if not stock_data:
        print("\nERROR: No daily stock data available.")
        return None

    # Load intraday data
    print("\nLoading 30-minute intraday data...")
    intraday_data = {}

    for symbol in stock_data.keys():
        try:
            df = bridge.load_30min_data(symbol, start_date, end_date)
            if not df.empty:
                intraday_data[symbol] = df
                print(f"  {symbol}: {len(df)} 30-min candles")
        except Exception as e:
            print(f"  {symbol}: Failed to load - {e}")

    if not intraday_data:
        print("\nERROR: No intraday data available.")
        return None

    # Determine grid based on mode
    if mode == 'quick':
        grid = QUICK_OPTIMIZATION_GRID
        max_combos = 30
    elif mode == 'medium':
        grid = {
            'narrow_cpr_threshold': [0.05, 0.1, 0.2, 0.3],  # Lower thresholds for real market data
            'otm_strike_pct': [3.0, 5.0, 7.0],
            'dte_min': [25, 30],
            'dte_max': [35, 40],
            'premium_double_threshold': [1.5, 2.0, 2.5],
            'premium_erosion_target': [70.0, 75.0, 80.0],
            'dte_exit_threshold': [7, 10],
            'enable_r1_exit': [True],
            'use_closer_r1': [True],
        }
        max_combos = 200
    else:  # full
        grid = DEFAULT_OPTIMIZATION_GRID
        max_combos = 500

    print(f"\nOptimization Mode: {mode.upper()}")
    print(f"Max Combinations: {max_combos}")

    # Initialize optimizer
    optimizer = CPRStrategyOptimizer(
        symbols=list(stock_data.keys()),
        start_date=start_date,
        end_date=end_date,
        stock_data=stock_data,
        intraday_data=intraday_data,
        initial_capital=1000000.0
    )

    print("\n" + "-" * 70)
    print("STARTING OPTIMIZATION...")
    print("-" * 70)

    # Run optimization
    results = optimizer.run_optimization(
        grid=grid,
        max_combinations=max_combos,
        progress_callback=progress_callback
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)

    print(f"\nTotal Tested: {results['total_tested']}")
    print(f"Successful: {results['successful_tests']}")

    if results['top_10']:
        print(f"\nBest Sharpe Ratio: {results['best_sharpe']:.3f}")
        print(f"Best Total Return: {results['best_return']:.2f}%")

        print("\n" + "-" * 70)
        print("TOP 10 PARAMETER COMBINATIONS")
        print("-" * 70)

        for i, r in enumerate(results['top_10'], 1):
            print(f"\n#{i} - Composite Score: {r['composite_score']:.1f}")
            print(f"    Return: {r['total_return']:.2f}% | Ann. Return: {r['annualized_return']:.2f}%")
            print(f"    Sharpe: {r['sharpe_ratio']:.3f} | Max DD: {r['max_drawdown']:.2f}%")
            print(f"    Win Rate: {r['win_rate']:.1f}% | Trades: {r['total_trades']}")
            print(f"    Parameters:")
            params = r['params']
            print(f"      - CPR Threshold: {params.get('narrow_cpr_threshold', 0.5)}%")
            print(f"      - OTM Strike: {params.get('otm_strike_pct', 5.0)}%")
            print(f"      - DTE Range: {params.get('dte_min', 30)}-{params.get('dte_max', 35)}")
            print(f"      - Premium Double: {params.get('premium_double_threshold', 2.0)}x")
            print(f"      - Erosion Target: {params.get('premium_erosion_target', 75.0)}%")
            print(f"      - DTE Exit: {params.get('dte_exit_threshold', 10)}")

    # Save results to JSON
    output_file = Path(__file__).parent / f"{mode}_optimization_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Export full results to CSV
    df_results = optimizer.export_results()
    if not df_results.empty:
        csv_file = Path(__file__).parent / f"{mode}_optimization_results.csv"
        df_results.to_csv(csv_file, index=False)
        print(f"Full results exported to: {csv_file}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run CPR Strategy Optimization')
    parser.add_argument('--mode', choices=['quick', 'medium', 'full'],
                        default='medium', help='Optimization mode')
    args = parser.parse_args()

    run_optimization(args.mode)