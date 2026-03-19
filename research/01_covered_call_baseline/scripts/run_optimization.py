"""
Covered Call Strategy Optimizer - Runner Script
================================================

Run this script to find the best covered call strategies.

Usage:
    python run_optimization.py [--quick] [--full]

Options:
    --quick : Quick test with 2 stocks, 30 strategies (~2 min)
    --full  : Full optimization with 10 stocks, 300 strategies (~30 min)
    (default): Medium test with 5 stocks, 100 strategies (~10 min)
"""

import sys
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)

from datetime import datetime
from strategy_optimizer import StrategyOptimizer


def run_quick_test():
    """Quick proof-of-concept with minimal settings"""
    print("=" * 60)
    print("QUICK OPTIMIZATION TEST")
    print("=" * 60)
    print("Stocks: 2 | Strategies: 30 | Time: ~2 minutes")
    print()

    optimizer = StrategyOptimizer(
        symbols=['RELIANCE', 'TCS'],
        start_date=datetime(2023, 6, 1),
        end_date=datetime(2024, 6, 30),
        initial_capital=1000000
    )

    results = optimizer.run_optimization(sample_size=30)
    if results:
        optimizer.print_results(top_n=10)
        optimizer.save_results('quick_optimization_results.json')
    return results


def run_medium_test():
    """Medium optimization for reasonable results"""
    print("=" * 60)
    print("MEDIUM OPTIMIZATION")
    print("=" * 60)
    print("Stocks: 5 | Strategies: 100 | Time: ~10 minutes")
    print()

    optimizer = StrategyOptimizer(
        symbols=['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK'],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 12, 31),
        initial_capital=1000000
    )

    results = optimizer.run_optimization(sample_size=100)
    if results:
        optimizer.print_results(top_n=15)
        optimizer.save_results('medium_optimization_results.json')
    return results


def run_full_optimization():
    """Full optimization with all stocks and strategies"""
    print("=" * 60)
    print("FULL OPTIMIZATION")
    print("=" * 60)
    print("Stocks: 10 | Strategies: 300 | Time: ~30 minutes")
    print()

    optimizer = StrategyOptimizer(
        symbols=['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
                 'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'ITC', 'AXISBANK'],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 12, 31),
        initial_capital=1000000
    )

    results = optimizer.run_optimization(sample_size=300)
    if results:
        optimizer.print_results(top_n=20)
        optimizer.save_results('full_optimization_results.json')
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--medium"

    if mode == "--quick":
        run_quick_test()
    elif mode == "--full":
        run_full_optimization()
    else:
        run_medium_test()
