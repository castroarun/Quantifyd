"""Stage B — full per-pattern sweep with NIFTY regime filter.

For one pattern at a time:
1. Read its perstock screen results, pick top stocks by win-rate (n>=20).
2. Build a parameter grid (RSI/window/level options) x exit-grid (TP > SL).
3. For each cell: aggregate across cohort, compute portfolio WR/PF/DD.
4. Write incrementally to results/02_<pattern>_ranking.csv.

Pass criterion (loose for ranking, tight for walk-forward):
  WR >= 0.70, PF >= 2.0, n >= 30 -> "passes_gates"

ASCII-only console (Windows cp1252 safe). Incremental CSV writes.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_R37 = os.path.normpath(os.path.join(_HERE, '..', '..', '37_intraday_75wr_quest', 'scripts'))
if _R37 not in sys.path:
    sys.path.insert(0, _R37)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _engine import (  # type: ignore
    load_5min, enrich, simulate_signals, trade_stats, TradeRules,
)
from pattern_lib import PATTERN_FUNCS  # type: ignore
import importlib
mod_8 = importlib.import_module('08_diamond_short_with_nifty')
build_nifty_regime = mod_8.build_nifty_regime


RESULTS_DIR = os.path.normpath(os.path.join(_HERE, '..', 'results'))
LOGS_DIR = os.path.normpath(os.path.join(_HERE, '..', 'logs'))


# Pattern-specific param grids for Stage B
# (keep total grid sane — total cells = grid * exit_grid * cohort_size)
PARAM_GRIDS = {
    'inside_bar': dict(
        min_bar_idx=[3, 6, 12],
        max_bar_idx=[36, 60],
        rsi_min=[40, 50, 55],
        rsi_max=[75, 80],
        require_vwap_align=[True, False],
        require_volume_lift=[False, True],
    ),
    'nr_breakout': dict(
        nr_len=[4, 7],
        min_bar_idx=[3, 6, 12],
        max_bar_idx=[36, 60],
        rsi_min=[40, 50, 55],
        rsi_max=[75, 80],
        require_vwap_align=[True, False],
    ),
    'failed_breakout': dict(
        level=['or_high', 'or_low', 'day_high', 'day_low'],
        min_bar_idx=[3, 6, 12],
        max_bar_idx=[48, 60],
        rsi_min=[25, 35],
        rsi_max=[70, 80],
        require_wick_back=[True, False],
    ),
    'vwap_v': dict(
        extension_atr=[0.7, 1.0, 1.5],
        reclaim_window=[2, 3, 5],
        min_bar_idx=[6, 12],
        max_bar_idx=[48, 60],
        rsi_min=[25, 35],
        rsi_max=[65, 75],
    ),
    'compression': dict(
        compression_window=[4, 6, 8],
        compression_pct=[0.4, 0.6, 0.8],
        min_bar_idx=[6, 12],
        max_bar_idx=[48, 60],
        rsi_min=[40, 50],
        rsi_max=[75, 80],
        require_vwap_align=[True, False],
    ),
    'multi_bar': dict(
        n_bars=[2, 3, 4],
        min_bar_idx=[3, 6, 12],
        max_bar_idx=[48, 60],
        rsi_min=[45, 55],
        rsi_max=[75, 80],
        require_vwap_align=[True, False],
        require_higher_lows=[True, False],
    ),
    'stop_run': dict(
        lookback=[8, 12, 18],
        pierce_pct=[0.05, 0.1, 0.2],
        min_bar_idx=[6, 12],
        max_bar_idx=[48, 60],
        rsi_min=[30, 40],
        rsi_max=[70, 80],
    ),
}

# All patterns share this exit grid — TP > SL by design (favorable RR)
EXIT_GRID = [
    (1.0, 0.4, 60),    # 2.5:1
    (1.5, 0.5, 60),    # 3.0:1
    (1.5, 0.6, 60),    # 2.5:1
    (2.0, 0.7, 60),    # 2.86:1
    (1.0, 0.5, 60),    # 2.0:1 (sanity)
]

# NIFTY regime filters available
NIFTY_FILTERS = {
    'long':  ['none', 'not_below_vwap_b3', 'first_bullish', 'b3_change_pos', 'b3_not_crashing'],
    'short': ['none', 'below_vwap_b3', 'first_bearish', 'b3_change_neg', 'b3_change_neg_strong'],
}


def filter_by_nifty(sig: pd.Series, df: pd.DataFrame, nifty: pd.DataFrame, nifty_filter: str) -> pd.Series:
    if nifty_filter == 'none' or nifty.empty:
        return sig

    # build session set per filter
    if nifty_filter == 'below_vwap_b3':
        ok = nifty.loc[nifty['n_below_vwap_b3'] == 1, 'session']
    elif nifty_filter == 'first_bearish':
        ok = nifty.loc[nifty['n_first_bearish'] == 1, 'session']
    elif nifty_filter == 'b3_change_neg':
        ok = nifty.loc[nifty['n_b3_change_pct'] < -0.1, 'session']
    elif nifty_filter == 'b3_change_neg_strong':
        ok = nifty.loc[nifty['n_b3_change_pct'] < -0.3, 'session']
    elif nifty_filter == 'not_below_vwap_b3':
        ok = nifty.loc[nifty['n_below_vwap_b3'] == 0, 'session']
    elif nifty_filter == 'first_bullish':
        ok = nifty.loc[nifty['n_first_bearish'] == 0, 'session']
    elif nifty_filter == 'b3_change_pos':
        ok = nifty.loc[nifty['n_b3_change_pct'] > 0.1, 'session']
    elif nifty_filter == 'b3_not_crashing':
        ok = nifty.loc[nifty['n_b3_change_pct'] > -0.5, 'session']
    else:
        return sig

    ok_set = set(pd.to_datetime(ok).values.astype('datetime64[ns]'))
    mask = df['session'].isin(list(ok_set))
    return sig & mask


def get_cohort(pattern: str, direction: str, top_n: int = 30,
               min_n: int = 20, min_wr: float = 0.55) -> list[str]:
    """Read perstock screen, return top stocks for this pattern+direction."""
    path = os.path.join(RESULTS_DIR, f'01_{pattern}_perstock.csv')
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    sub = df[(df['direction'] == direction) & (df['n_trades'] >= min_n) &
             (df['win_rate'] >= min_wr)]
    sub = sub.sort_values('win_rate', ascending=False)
    return sub.head(top_n)['symbol'].tolist()


def run_pattern_sweep(pattern: str, direction: str,
                      top_cohort: int = 30,
                      min_screen_wr: float = 0.55,
                      log=print) -> int:
    func = PATTERN_FUNCS[pattern]
    grid = PARAM_GRIDS[pattern]

    cohort = get_cohort(pattern, direction, top_n=top_cohort, min_wr=min_screen_wr)
    if len(cohort) < 5:
        log(f'  [{pattern}/{direction}] cohort too small ({len(cohort)}) — skip')
        return 0

    log(f'  [{pattern}/{direction}] cohort = {len(cohort)} stocks')
    log(f'  cohort: {cohort[:10]}{"..." if len(cohort) > 10 else ""}')

    # cartesian product of param grid
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    param_combos = list(itertools.product(*values))
    nifty_filters = NIFTY_FILTERS[direction]

    n_total = len(param_combos) * len(EXIT_GRID) * len(nifty_filters)
    log(f'  grid: {len(param_combos)} param x {len(EXIT_GRID)} exits x {len(nifty_filters)} nifty = {n_total} cells')

    # preload data
    log('  preloading data ...')
    cache = {}
    t0 = time.time()
    for sym in cohort:
        try:
            df = load_5min(sym)
            if df.empty or len(df) < 1500:
                continue
            cache[sym] = enrich(df, or_minutes=15)
        except Exception as e:
            log(f'    {sym} load/enrich fail: {e}')
    log(f'  loaded {len(cache)} stocks in {time.time() - t0:.0f}s')

    # build NIFTY regime
    log('  building NIFTY regime ...')
    nifty = build_nifty_regime()

    out_path = os.path.join(RESULTS_DIR, f'02_{pattern}_{direction}_ranking.csv')
    fields = ['family', 'direction', 'params', 'nifty_filter',
              'tp_pct', 'sl_pct', 'max_hold_bars',
              'n_stocks', 'n_trades', 'win_rate', 'avg_win', 'avg_loss',
              'profit_factor', 'sharpe', 'max_dd_pct', 'total_return_pct',
              'trades_per_stock_year', 'aws', 'passes_gates']

    done_keys = set()
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                done_keys.add((r['params'], r['nifty_filter'], r['tp_pct'], r['sl_pct'], r['max_hold_bars']))
        log(f'  resume: {len(done_keys)} cells already done')

    mode = 'a' if os.path.exists(out_path) else 'w'
    with open(out_path, mode, newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == 'w':
            w.writeheader()

        cell = 0
        n_passes = 0
        n_print = 0
        t_run = time.time()
        for combo in param_combos:
            params = dict(zip(keys, combo))
            for nifty_filter in nifty_filters:
                for tp, sl, hold in EXIT_GRID:
                    cell += 1
                    key = (str(params), nifty_filter, str(tp), str(sl), str(hold))
                    if key in done_keys:
                        continue
                    all_trades = []
                    stocks_with_trades = 0
                    stock_sessions = 0
                    for sym, df in cache.items():
                        try:
                            sig = func(df, direction=direction, **params)
                        except Exception:
                            continue
                        sig = filter_by_nifty(sig, df, nifty, nifty_filter)
                        rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold,
                                           direction=direction)
                        trades = simulate_signals(df, sig, rules)
                        if len(trades) == 0:
                            continue
                        trades['symbol'] = sym
                        all_trades.append(trades)
                        stocks_with_trades += 1
                        stock_sessions += df['session'].nunique()
                    if not all_trades:
                        continue
                    t = pd.concat(all_trades, ignore_index=True)
                    avg_sessions = stock_sessions / max(stocks_with_trades, 1)
                    stats = trade_stats(t, sessions=int(avg_sessions))
                    trades_per_stock_year = stats['trades_per_year'] / max(stocks_with_trades, 1)
                    passes = (stats['n_trades'] >= 30 and stats['win_rate'] >= 0.70
                              and stats['profit_factor'] >= 2.0)
                    row = dict(
                        family=pattern, direction=direction,
                        params=str(params), nifty_filter=nifty_filter,
                        tp_pct=tp, sl_pct=sl, max_hold_bars=hold,
                        n_stocks=stocks_with_trades, n_trades=stats['n_trades'],
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
                    if passes:
                        n_passes += 1
                    if (stats['win_rate'] >= 0.70 and stats['n_trades'] >= 30 and
                        stats['profit_factor'] >= 2.0 and n_print < 30):
                        log(f'    [{cell}/{n_total}] {params} nifty={nifty_filter} tp={tp} sl={sl} '
                            f'-> WR={stats["win_rate"]:.1%} n={stats["n_trades"]} '
                            f'PF={stats["profit_factor"]:.2f} DD={stats["max_dd_pct"]:.1f}%')
                        n_print += 1
                    if cell % 200 == 0:
                        log(f'    ... cell {cell}/{n_total} elapsed {time.time() - t_run:.0f}s passes={n_passes}')

    log(f'  Done [{pattern}/{direction}] in {time.time() - t_run:.0f}s. Passes: {n_passes}')
    return n_passes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', required=True, choices=list(PARAM_GRIDS.keys()))
    ap.add_argument('--directions', nargs='+', default=['long', 'short'])
    ap.add_argument('--cohort-size', type=int, default=30)
    ap.add_argument('--min-screen-wr', type=float, default=0.55)
    args = ap.parse_args()

    log_path = os.path.join(LOGS_DIR, f'02_{args.pattern}_sweep.log')
    log_f = open(log_path, 'a', encoding='utf-8', buffering=1)

    def log(msg: str) -> None:
        sys.stdout.write(msg + '\n')
        sys.stdout.flush()
        log_f.write(msg + '\n')
        log_f.flush()

    log('=' * 70)
    log(f'Stage B: pattern sweep [{args.pattern}] | {time.strftime("%Y-%m-%d %H:%M:%S")}')
    log('=' * 70)

    for d in args.directions:
        log(f'\n--- {args.pattern} / {d} ---')
        run_pattern_sweep(args.pattern, d, top_cohort=args.cohort_size,
                          min_screen_wr=args.min_screen_wr, log=log)
    log_f.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
