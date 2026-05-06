"""Stage 2 — strategy family sweep.

Run a single family's parameter grid across the universe, with a fixed
TP/SL/exit rule grid. Output: results/02_<family>_ranking.csv with one
row per (family, params, tp, sl, hold) aggregated across all stocks
(portfolio-wide stats), plus per-stock detail in 02_<family>_perstock.csv.

Each portfolio-row is a candidate "system": its WR / PF / DD are computed
across ALL trades from ALL stocks for that param combo.

Usage:
    python 02_strategy_battery.py --family vwap_rejection
    python 02_strategy_battery.py --family or_breakout --max-stocks 50
    python 02_strategy_battery.py --family all --max-stocks 30
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import (  # type: ignore
    list_5min_universe, load_5min, enrich,
    simulate_signals, trade_stats, TradeRules,
)
from _strategies import STRATEGIES  # type: ignore


RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
)
LOG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
)

PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    'vwap_rejection': dict(
        max_dip_atr=[1.0, 1.5, 2.0],
        reclaim_window=[2, 3, 5],
        rsi_min=[40, 50],
    ),
    'or_breakout': dict(
        or_minutes=[5, 15, 30],
        min_volume_mult=[1.0, 1.5],
        rsi_min=[50, 55, 60],
        vwap_align=[True],
    ),
    'first_candle_open_low': dict(
        confirm_bars=[1, 2, 3],
        rsi_min=[50, 55],
        gap_up_min=[-1.0, -0.3, 0.0],
    ),
    'ema9_pullback': dict(
        trend_ema=[21, 50],
        pullback_atr=[0.3, 0.5, 0.8],
        rsi_min=[40, 45, 50],
        rsi_max=[60, 65, 70],
    ),
    'tod_momentum': dict(
        start_bar=[3, 6],
        end_bar=[9, 12],
        rsi_min=[55, 60],
        vwap_align=[True],
        ema_align=[True, False],
    ),
    'vwap_revert': dict(
        deviation_atr=[1.5, 2.0, 2.5],
        rsi_max=[25, 30, 35],
        direction=['long'],
    ),
    'volume_thrust': dict(
        vol_mult=[2.5, 3.0, 4.0],
        thrust_atr=[0.8, 1.0, 1.5],
        vwap_align=[True],
    ),
    'rsi_extreme': dict(
        rsi_threshold=[20, 25, 30],
        direction=['long'],
        require_bullish_bar=[True],
    ),
    'inside_bar_breakout': dict(
        min_inside=[2, 3],
        rsi_min=[50, 55],
        vwap_align=[True],
    ),
    'donchian_breakout': dict(
        lookback=[12, 20, 30],
        vol_mult=[1.0, 1.5],
        vwap_align=[True],
    ),
    'ema_cross': dict(
        fast=[9],
        slow=[21],
        vwap_align=[True],
        rsi_min=[50, 55, 60],
    ),
    'gap_and_go': dict(
        gap_min_pct=[0.3, 0.5, 1.0],
        rsi_min=[55, 60],
        max_bar_idx=[12, 18],
    ),
}

EXIT_GRID = [
    # (tp_pct, sl_pct, max_hold_bars)
    (0.5, 0.3, 12),
    (0.8, 0.4, 24),
    (1.0, 0.5, 36),
    (1.5, 0.7, 48),
    (2.0, 1.0, 60),
]


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_family(family: str, stocks: List[str], min_trades: int = 30) -> int:
    """Run one strategy family across stocks × param grid × exit grid."""
    if family not in STRATEGIES:
        print(f'unknown family {family}')
        return 1
    strat_fn = STRATEGIES[family]
    grid = PARAM_GRIDS.get(family, {})
    param_combos = expand_grid(grid)
    exit_combos = EXIT_GRID
    n_systems = len(param_combos) * len(exit_combos)
    print(f'[{family}] {len(stocks)} stocks × {len(param_combos)} params × {len(exit_combos)} exits = {n_systems} systems')

    # cache enriched data — load once per stock, reuse across all combos
    print('  loading + enriching data ...', end='', flush=True)
    t0 = time.time()
    cache: Dict[str, pd.DataFrame] = {}
    for sym in stocks:
        df = load_5min(sym)
        if df.empty or len(df) < 1000:
            continue
        try:
            cache[sym] = enrich(df, or_minutes=15)
        except Exception as e:
            print(f'  enrich fail {sym}: {e}')
    print(f' done {len(cache)} stocks in {time.time()-t0:.0f}s')

    out_path = os.path.join(RESULTS_DIR, f'02_{family}_ranking.csv')
    fields = [
        'family', 'params', 'tp_pct', 'sl_pct', 'max_hold_bars',
        'n_stocks', 'n_trades', 'win_rate', 'avg_win', 'avg_loss',
        'profit_factor', 'sharpe', 'max_dd_pct', 'total_return_pct',
        'trades_per_stock_year', 'aws', 'passes_gates',
    ]
    done = set()
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                done.add((row['params'], row['tp_pct'], row['sl_pct'], row['max_hold_bars']))
        mode = 'a'
    else:
        mode = 'w'

    with open(out_path, mode, newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == 'w':
            w.writeheader()

        cell = 0
        t_run = time.time()
        for params in param_combos:
            for tp, sl, hold in exit_combos:
                cell += 1
                key = (str(params), str(tp), str(sl), str(hold))
                if key in done:
                    continue
                # collect trades across all stocks for this system
                all_trades = []
                stock_sessions = 0
                stocks_with_trades = 0
                for sym, df in cache.items():
                    try:
                        sig, direction = strat_fn(df, **params)
                    except Exception:
                        continue
                    rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold, direction=direction)
                    trades = simulate_signals(df, sig, rules)
                    if len(trades) == 0:
                        continue
                    trades['symbol'] = sym
                    all_trades.append(trades)
                    stock_sessions += df['session'].nunique()
                    stocks_with_trades += 1
                if not all_trades:
                    continue
                t = pd.concat(all_trades, ignore_index=True)
                # avg sessions per stock for trades_per_year math
                avg_sessions = stock_sessions / max(stocks_with_trades, 1)
                stats = trade_stats(t, sessions=int(avg_sessions))
                trades_per_stock_year = stats['trades_per_year'] / max(stocks_with_trades, 1)
                passes = (
                    stats['n_trades'] >= min_trades
                    and stats['win_rate'] >= 0.60   # write any with WR >= 60%
                )
                row = dict(
                    family=family,
                    params=str(params),
                    tp_pct=tp, sl_pct=sl, max_hold_bars=hold,
                    n_stocks=stocks_with_trades,
                    n_trades=stats['n_trades'],
                    win_rate=round(stats['win_rate'], 4),
                    avg_win=round(stats['avg_win'], 4),
                    avg_loss=round(stats['avg_loss'], 4),
                    profit_factor=round(stats['profit_factor'], 3),
                    sharpe=round(stats['sharpe'], 3),
                    max_dd_pct=round(stats['max_dd_pct'], 3),
                    total_return_pct=round(stats['total_return_pct'], 2),
                    trades_per_stock_year=round(trades_per_stock_year, 1),
                    aws=round(stats['aws'], 4),
                    passes_gates=int(passes),
                )
                w.writerow(row)
                f.flush()
                if stats['win_rate'] >= 0.65 and stats['n_trades'] >= min_trades:
                    print(f'  [{cell}/{n_systems}] {family} p={params} tp={tp} sl={sl} hold={hold} '
                          f'→ WR={stats["win_rate"]:.1%} n={stats["n_trades"]} '
                          f'PF={stats["profit_factor"]:.2f} DD={stats["max_dd_pct"]:.1f}%')
                if cell % 10 == 0:
                    elapsed = time.time() - t_run
                    rate = cell / max(elapsed, 1)
                    eta = (n_systems - cell) / max(rate, 0.001)
                    print(f'  progress: {cell}/{n_systems} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)', flush=True)
    print(f'[{family}] done — output {out_path}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True,
                    help='strategy family or "all"')
    ap.add_argument('--max-stocks', type=int, default=30,
                    help='cap on number of stocks (sorted by 5-min row count)')
    ap.add_argument('--min-trades', type=int, default=30)
    args = ap.parse_args()

    universe = list_5min_universe(min_rows=32_000)
    stocks = universe[:args.max_stocks]
    print(f'Universe: {len(stocks)}/{len(universe)} stocks')

    families = list(STRATEGIES.keys()) if args.family == 'all' else [args.family]
    for fam in families:
        run_family(fam, stocks, min_trades=args.min_trades)
    return 0


if __name__ == '__main__':
    sys.exit(main())
