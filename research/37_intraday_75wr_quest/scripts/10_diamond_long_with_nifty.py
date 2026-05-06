"""Stage 10 — Diamond-LONG mirror of Stage 8.

Take the 25 long-diamond stocks (top L_b3_strong_wr) and layer NIFTY-UP
regime filter to lift WR from baseline ~54% to (hopefully) 75%+.

Filter: stock above VWAP + RSI > threshold at entry_bar
NIFTY filter: NIFTY up (various definitions)

Output: results/10_diamond_long_ranking.csv
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
    load_5min, enrich, simulate_signals, trade_stats, TradeRules,
)
from _strategies import _one_per_session, _no_late_entries  # type: ignore

import importlib
mod_8 = importlib.import_module('08_diamond_short_with_nifty')
build_nifty_regime = mod_8.build_nifty_regime


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
DIAMONDS_PATH = os.path.join(RESULTS_DIR, '10_long_diamonds.txt')
OUT_PATH = os.path.join(RESULTS_DIR, '10_diamond_long_ranking.csv')


def diamond_long(df: pd.DataFrame,
                 rsi_threshold: float = 60,
                 entry_bar: int = 6,
                 require_above_vwap: bool = True) -> pd.Series:
    """LONG signal at entry_bar: above VWAP + RSI > threshold."""
    cond = (
        (df['bar_idx'].values == entry_bar)
        & (df['rsi'].values > rsi_threshold)
        & ((df['close'].values > df['vwap'].values) if require_above_vwap else True)
    )
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig


def filter_by_nifty_up(sig: pd.Series, df: pd.DataFrame, nifty: pd.DataFrame,
                       nifty_filter: str) -> pd.Series:
    """Apply NIFTY-UP regime filter (mirror of short-side filter)."""
    if nifty_filter == 'none':
        return sig
    if nifty.empty:
        return sig

    if nifty_filter == 'above_vwap_b3':
        ok_sessions = nifty.loc[nifty['n_below_vwap_b3'] == 0, 'session']
    elif nifty_filter == 'first_bullish':
        ok_sessions = nifty.loc[nifty['n_first_bearish'] == 0, 'session']
    elif nifty_filter == 'both':
        ok_sessions = nifty.loc[
            (nifty['n_below_vwap_b3'] == 0) & (nifty['n_first_bearish'] == 0),
            'session',
        ]
    elif nifty_filter == 'b3_change_pos':
        ok_sessions = nifty.loc[nifty['n_b3_change_pct'] > 0.1, 'session']
    elif nifty_filter == 'b3_change_pos_strong':
        ok_sessions = nifty.loc[nifty['n_b3_change_pct'] > 0.3, 'session']
    else:
        return sig

    ok_set = set(pd.to_datetime(ok_sessions).values.astype('datetime64[ns]'))
    mask = df['session'].isin(list(ok_set))
    return sig & mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-trades', type=int, default=30)
    args = ap.parse_args()

    with open(DIAMONDS_PATH) as f:
        diamonds = [s.strip() for s in f if s.strip()]
    print(f'Loaded {len(diamonds)} long-diamond stocks')

    print('Loading NIFTY regime ...', end='', flush=True)
    t0 = time.time()
    nifty = build_nifty_regime()
    print(f' {len(nifty)} sessions in {time.time()-t0:.0f}s')

    print('Loading + enriching diamond data ...')
    cache = {}
    t0 = time.time()
    for sym in diamonds:
        df = load_5min(sym)
        if df.empty or len(df) < 1000:
            continue
        try:
            cache[sym] = enrich(df, or_minutes=15)
        except Exception:
            continue
    print(f'  done {len(cache)} stocks in {time.time()-t0:.0f}s')

    PARAM_GRID = list(itertools.product(
        [3, 6, 9, 12],                  # entry_bar (try later entries too)
        [55, 60, 65, 70],               # rsi_threshold
        ['none', 'above_vwap_b3', 'first_bullish', 'both',
         'b3_change_pos', 'b3_change_pos_strong'],
    ))
    EXIT_GRID = [
        # Mirror of short-side: small TP, wider SL
        (0.5, 1.5, 60),
        (0.7, 1.5, 60),
        (1.0, 1.5, 60),
        (0.5, 1.0, 60),
        # also try long-side higher RR
        (1.5, 0.5, 60),
        (2.0, 0.7, 60),
        (1.0, 0.7, 48),
        (5.0, 2.0, 60),    # hold-to-EOD with wide SL
    ]
    n_total = len(PARAM_GRID) * len(EXIT_GRID)
    print(f'Sweep: {len(PARAM_GRID)} param combos x {len(EXIT_GRID)} exits = {n_total} systems')

    fields = [
        'family', 'params', 'tp_pct', 'sl_pct', 'max_hold_bars',
        'n_stocks', 'n_trades', 'win_rate', 'avg_win', 'avg_loss',
        'profit_factor', 'sharpe', 'max_dd_pct', 'total_return_pct',
        'trades_per_stock_year', 'aws', 'passes_gates',
    ]
    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        cell = 0
        t_run = time.time()
        for entry_bar, rsi_thr, nifty_filter in PARAM_GRID:
            params = dict(entry_bar=entry_bar, rsi_threshold=rsi_thr, nifty_filter=nifty_filter)
            for tp, sl, hold in EXIT_GRID:
                cell += 1
                all_trades = []
                stocks_with_trades = 0
                stock_sessions = 0
                for sym, df in cache.items():
                    sig = diamond_long(df, rsi_threshold=rsi_thr, entry_bar=entry_bar)
                    sig = filter_by_nifty_up(sig, df, nifty, nifty_filter)
                    rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold, direction='long')
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
                passes = (stats['n_trades'] >= args.min_trades and stats['win_rate'] >= 0.65)
                row = dict(
                    family='diamond_long', params=str(params),
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
                if stats['win_rate'] >= 0.65 and stats['n_trades'] >= args.min_trades:
                    print(f'  [{cell}/{n_total}] {params} tp={tp} sl={sl} hold={hold} '
                          f'-> WR={stats["win_rate"]:.1%} n={stats["n_trades"]} '
                          f'PF={stats["profit_factor"]:.2f} DD={stats["max_dd_pct"]:.1f}%')

        print(f'Done in {time.time()-t_run:.0f}s -> {OUT_PATH}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
