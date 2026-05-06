"""Stage 4 — confluence stacking based on Stage 1 signatures.

The reverse-engineer found two dominant morning signatures on big-move days:

LONG SIGNATURE (UP days, n=2112):
  - Slight gap DOWN (-0.14% avg)
  - First candle: Open == Low frequency 29.5% (vs 5% baseline)
  - First candle bullish: 75%
  - Bar 3 close > VWAP: 72%
  - RSI at bar 3: 60+
  - EMA9 > EMA21 at bar 6: 74%

SHORT SIGNATURE (DOWN days, n=2328):
  - Slight gap UP (+0.32% avg)
  - First candle: Open == High frequency 33.8% (vs 7% baseline)
  - First candle bearish: 81%
  - Bar 3 close < VWAP: 79%
  - RSI at bar 3: ~39
  - EMA9 < EMA21 at bar 6: 78%

This script tests a single confluence strategy that requires several of
these conditions to be true simultaneously. Sweeps the strictness gates.

Output: results/04_confluence_ranking.csv (incremental).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import (  # type: ignore
    list_5min_universe, load_5min, enrich,
    simulate_signals, trade_stats, TradeRules,
)
from _strategies import _one_per_session, _no_late_entries  # type: ignore


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))


def confluence_long(df: pd.DataFrame,
                    open_eq_low_tol: float = 0.001,    # 0.1%
                    require_first_bullish: bool = True,
                    rsi_bar3_min: float = 55,
                    rsi_bar3_max: float = 75,
                    require_above_vwap: bool = True,
                    require_ema9_above_21: bool = True,
                    gap_max: float = 0.5,             # gap pct must be <= +X%
                    entry_bar: int = 3) -> pd.Series:
    """Build LONG signal at `entry_bar` based on first-candle signature.

    Signal fires on the bar at index entry_bar of each session if:
      - first candle had open == low within tolerance
      - (optional) first candle was bullish
      - close at entry_bar > VWAP at entry_bar
      - RSI at entry_bar in [min, max]
      - EMA9 > EMA21 at entry_bar
      - day's gap_pct <= gap_max
    """
    g_first = df.groupby('session').head(1)
    fb = pd.DataFrame({
        'fopen': g_first['open'].values,
        'fhigh': g_first['high'].values,
        'flow': g_first['low'].values,
        'fclose': g_first['close'].values,
        'frange': g_first['high'].values - g_first['low'].values,
    }, index=g_first['session'].values)
    fb['open_eq_low'] = (fb['fopen'] - fb['flow']).abs() / fb['fopen'] <= open_eq_low_tol
    fb['first_bullish'] = fb['fclose'] > fb['fopen']

    # gap pct from prev_close (df already has prev_close from enrich)
    g_open = df.groupby('session')['day_open'].first()
    g_pc = df.groupby('session')['prev_close'].first()
    gap_pct = ((g_open - g_pc) / g_pc * 100).reindex(fb.index)
    fb['gap_pct'] = gap_pct.values

    sess_arr = df['session'].values
    is_open_eq_low = pd.Series(fb['open_eq_low'].values, index=fb.index).reindex(sess_arr).values
    is_first_bullish = pd.Series(fb['first_bullish'].values, index=fb.index).reindex(sess_arr).values
    sess_gap = pd.Series(fb['gap_pct'].values, index=fb.index).reindex(sess_arr).values

    cond = (
        (df['bar_idx'].values == entry_bar)
        & is_open_eq_low
        & (is_first_bullish if require_first_bullish else True)
        & ((df['close'].values > df['vwap'].values) if require_above_vwap else True)
        & (df['rsi'].values >= rsi_bar3_min)
        & (df['rsi'].values <= rsi_bar3_max)
        & ((df['ema9'].values > df['ema21'].values) if require_ema9_above_21 else True)
        & (sess_gap <= gap_max)
    )
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig


def confluence_short(df: pd.DataFrame,
                     open_eq_high_tol: float = 0.001,
                     require_first_bearish: bool = True,
                     rsi_bar3_min: float = 25,
                     rsi_bar3_max: float = 45,
                     require_below_vwap: bool = True,
                     require_ema9_below_21: bool = True,
                     gap_min: float = -0.5,            # gap pct must be >= X%
                     entry_bar: int = 3) -> pd.Series:
    g_first = df.groupby('session').head(1)
    fb = pd.DataFrame({
        'fopen': g_first['open'].values,
        'fhigh': g_first['high'].values,
        'flow': g_first['low'].values,
        'fclose': g_first['close'].values,
    }, index=g_first['session'].values)
    fb['open_eq_high'] = (fb['fopen'] - fb['fhigh']).abs() / fb['fopen'] <= open_eq_high_tol
    fb['first_bearish'] = fb['fclose'] < fb['fopen']

    g_open = df.groupby('session')['day_open'].first()
    g_pc = df.groupby('session')['prev_close'].first()
    gap_pct = ((g_open - g_pc) / g_pc * 100).reindex(fb.index)
    fb['gap_pct'] = gap_pct.values

    sess_arr = df['session'].values
    is_open_eq_high = pd.Series(fb['open_eq_high'].values, index=fb.index).reindex(sess_arr).values
    is_first_bearish = pd.Series(fb['first_bearish'].values, index=fb.index).reindex(sess_arr).values
    sess_gap = pd.Series(fb['gap_pct'].values, index=fb.index).reindex(sess_arr).values

    cond = (
        (df['bar_idx'].values == entry_bar)
        & is_open_eq_high
        & (is_first_bearish if require_first_bearish else True)
        & ((df['close'].values < df['vwap'].values) if require_below_vwap else True)
        & (df['rsi'].values >= rsi_bar3_min)
        & (df['rsi'].values <= rsi_bar3_max)
        & ((df['ema9'].values < df['ema21'].values) if require_ema9_below_21 else True)
        & (sess_gap >= gap_min)
    )
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

LONG_GRID = dict(
    rsi_bar3_min=[55, 60, 65],
    require_first_bullish=[True],
    require_above_vwap=[True],
    require_ema9_above_21=[True, False],
    gap_max=[0.5, 1.0, 99.0],
    entry_bar=[2, 3, 4],
)

SHORT_GRID = dict(
    rsi_bar3_max=[35, 40, 45],
    require_first_bearish=[True],
    require_below_vwap=[True],
    require_ema9_below_21=[True, False],
    gap_min=[-99.0, -1.0, -0.5],
    entry_bar=[2, 3, 4],
)

EXIT_GRID = [
    (0.5, 0.3, 12),
    (0.7, 0.4, 24),
    (1.0, 0.5, 36),
    (1.5, 0.7, 48),
    (1.0, 0.7, 48),    # asymmetric: 1.4:1 RR
    (0.6, 0.4, 18),
]


def expand(grid):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    return [dict(zip(keys, c)) for c in itertools.product(*vals)]


def run_side(side: str, stocks: list[str], min_trades: int) -> int:
    out = os.path.join(RESULTS_DIR, f'04_confluence_{side}_ranking.csv')
    if side == 'long':
        grid = LONG_GRID
        sig_fn = confluence_long
        direction = 'long'
    else:
        grid = SHORT_GRID
        sig_fn = confluence_short
        direction = 'short'

    combos = expand(grid)
    n_systems = len(combos) * len(EXIT_GRID)
    print(f'[{side}] {len(stocks)} stocks × {len(combos)} params × {len(EXIT_GRID)} exits = {n_systems} systems')

    print('  loading + enriching ...', end='', flush=True)
    t0 = time.time()
    cache = {}
    for sym in stocks:
        df = load_5min(sym)
        if df.empty or len(df) < 1000:
            continue
        try:
            cache[sym] = enrich(df, or_minutes=15)
        except Exception:
            continue
    print(f' done {len(cache)} stocks in {time.time()-t0:.0f}s')

    fields = [
        'side', 'params', 'tp_pct', 'sl_pct', 'max_hold_bars',
        'n_stocks', 'n_trades', 'win_rate', 'avg_win', 'avg_loss',
        'profit_factor', 'sharpe', 'max_dd_pct', 'total_return_pct',
        'trades_per_stock_year', 'aws', 'passes_gates',
    ]
    done = set()
    if os.path.exists(out):
        with open(out, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                done.add((row['params'], row['tp_pct'], row['sl_pct'], row['max_hold_bars']))
        mode = 'a'
    else:
        mode = 'w'

    cell = 0
    t_run = time.time()
    with open(out, mode, newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == 'w':
            w.writeheader()
        for params in combos:
            for tp, sl, hold in EXIT_GRID:
                cell += 1
                key = (str(params), str(tp), str(sl), str(hold))
                if key in done:
                    continue
                all_trades = []
                stocks_with_trades = 0
                stock_sessions = 0
                for sym, df in cache.items():
                    try:
                        sig = sig_fn(df, **params)
                    except Exception:
                        continue
                    rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold, direction=direction)
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
                passes = (
                    stats['n_trades'] >= min_trades
                    and stats['win_rate'] >= 0.60
                )
                row = dict(
                    side=side, params=str(params),
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
                if stats['win_rate'] >= 0.65 and stats['n_trades'] >= min_trades:
                    print(f'  [{cell}/{n_systems}] {side} {params} tp={tp} sl={sl} hold={hold} '
                          f'→ WR={stats["win_rate"]:.1%} n={stats["n_trades"]} '
                          f'PF={stats["profit_factor"]:.2f} DD={stats["max_dd_pct"]:.1f}%')
                if cell % 20 == 0:
                    elapsed = time.time() - t_run
                    eta = (n_systems - cell) * elapsed / max(cell, 1)
                    print(f'  progress: {cell}/{n_systems} ({elapsed:.0f}s, ETA {eta:.0f}s)', flush=True)
    print(f'[{side}] done — output {out}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--side', choices=['long', 'short', 'both'], default='both')
    ap.add_argument('--max-stocks', type=int, default=50)
    ap.add_argument('--min-trades', type=int, default=30)
    args = ap.parse_args()

    universe = list_5min_universe(min_rows=32_000)
    stocks = universe[:args.max_stocks]
    print(f'Universe: {len(stocks)}/{len(universe)} stocks')

    if args.side in ('long', 'both'):
        run_side('long', stocks, args.min_trades)
    if args.side in ('short', 'both'):
        run_side('short', stocks, args.min_trades)
    return 0


if __name__ == '__main__':
    sys.exit(main())
