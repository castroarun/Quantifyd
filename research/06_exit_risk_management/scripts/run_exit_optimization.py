"""
Comprehensive Exit Rule Optimization - Agent 1
Tests ALL combinations of exit parameters with portfolio size 30+.
Total: 2*2*4*3*2*2 = 192 combinations
"""
import logging
logging.disable(logging.WARNING)

import sys
import os
import time
import csv
from itertools import product
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine, BacktestResult
from services.mq_portfolio import MQBacktestConfig

# Parameter grid
PARAMS = {
    'portfolio_size': [30, 40],
    'equity_allocation_pct': [0.90, 0.95],
    'hard_stop_loss': [0.15, 0.20, 0.25, 0.30],
    'rebalance_ath_drawdown': [0.10, 0.15, 0.20],
    'trailing_stop_loss': [True, False],
    'daily_ath_drawdown_exit': [True, False],
}

def make_label(combo):
    ps, eq, hsl, ath, trail, daily = combo
    parts = [
        f"HSL{int(hsl*100)}",
        f"ATH{int(ath*100)}",
        f"PS{ps}",
        f"EQ{int(eq*100)}",
    ]
    if trail:
        parts.append("TRAIL")
    if daily:
        parts.append("DAILY")
    return "_".join(parts)

def main():
    # Build all combinations
    keys = list(PARAMS.keys())
    values = list(PARAMS.values())
    combos = list(product(*values))
    total = len(combos)
    print(f"=== Exit Rule Optimization: {total} combinations ===\n")

    results = []
    start_all = time.time()

    for i, combo in enumerate(combos):
        ps, eq, hsl, ath, trail, daily = combo
        label = make_label(combo)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_all
            eta = (elapsed / max(i, 1)) * (total - i)
            print(f"[{i+1}/{total}] Running {label}... (elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)")

        try:
            config = MQBacktestConfig(
                portfolio_size=ps,
                equity_allocation_pct=eq,
                debt_reserve_pct=round(1.0 - eq, 4),
                hard_stop_loss=hsl,
                rebalance_ath_drawdown=ath,
                trailing_stop_loss=trail,
                daily_ath_drawdown_exit=daily,
            )
            engine = MQBacktestEngine(config)
            result = engine.run()

            results.append({
                'label': label,
                'portfolio_size': ps,
                'equity_allocation_pct': eq,
                'hard_stop_loss': hsl,
                'rebalance_ath_drawdown': ath,
                'trailing_stop_loss': trail,
                'daily_ath_drawdown_exit': daily,
                'cagr': result.cagr,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': result.max_drawdown,
                'calmar_ratio': result.calmar_ratio,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'final_value': result.final_value,
                'total_return_pct': result.total_return_pct,
                'avg_win_pct': result.avg_win_pct,
                'avg_loss_pct': result.avg_loss_pct,
            })
        except Exception as e:
            print(f"  ERROR on {label}: {e}")
            results.append({
                'label': label,
                'portfolio_size': ps,
                'equity_allocation_pct': eq,
                'hard_stop_loss': hsl,
                'rebalance_ath_drawdown': ath,
                'trailing_stop_loss': trail,
                'daily_ath_drawdown_exit': daily,
                'cagr': None,
                'sharpe_ratio': None,
                'sortino_ratio': None,
                'max_drawdown': None,
                'calmar_ratio': None,
                'total_trades': None,
                'win_rate': None,
                'final_value': None,
                'total_return_pct': None,
                'avg_win_pct': None,
                'avg_loss_pct': None,
            })

    total_time = time.time() - start_all
    print(f"\n=== Completed {total} backtests in {total_time:.1f}s ({total_time/total:.1f}s avg) ===\n")

    # Filter valid results
    valid = [r for r in results if r['cagr'] is not None]

    # Sort by CAGR descending
    by_cagr = sorted(valid, key=lambda x: x['cagr'], reverse=True)
    print("=" * 120)
    print(f"{'TOP 20 BY CAGR':^120}")
    print("=" * 120)
    header = f"{'#':>3} {'Label':<40} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8} {'Trades':>7} {'WinRate':>8} {'FinalVal':>14}"
    print(header)
    print("-" * 120)
    for rank, r in enumerate(by_cagr[:20], 1):
        print(f"{rank:>3} {r['label']:<40} {r['cagr']:>7.2%} {r['sharpe_ratio']:>8.3f} {r['max_drawdown']:>7.2%} {r['calmar_ratio']:>8.3f} {r['total_trades']:>7} {r['win_rate']:>7.2%} {r['final_value']:>14,.0f}")

    # Sort by Calmar ratio descending
    by_calmar = sorted(valid, key=lambda x: x['calmar_ratio'], reverse=True)
    print("\n" + "=" * 120)
    print(f"{'TOP 10 BY CALMAR RATIO':^120}")
    print("=" * 120)
    print(header)
    print("-" * 120)
    for rank, r in enumerate(by_calmar[:10], 1):
        print(f"{rank:>3} {r['label']:<40} {r['cagr']:>7.2%} {r['sharpe_ratio']:>8.3f} {r['max_drawdown']:>7.2%} {r['calmar_ratio']:>8.3f} {r['total_trades']:>7} {r['win_rate']:>7.2%} {r['final_value']:>14,.0f}")

    # Save CSV
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimization_agent1_exits.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {csv_path}")
    print(f"Total valid results: {len(valid)}/{len(results)}")

if __name__ == "__main__":
    main()
