"""Stage 8 — Diamond-Short strategy with NIFTY regime filter.

Building on Stage 7 finding: 25 stocks have 60–72% WR on the simple
"below VWAP + RSI < 40 at bar 3, hold to session close" SHORT setup.
ZEEL alone is at 72.4%.

To push from 64% portfolio mean to 75%, layer one more filter: only
short when NIFTY itself is showing weakness in the first 15–30 min.

Sweep:
- per-stock universe: top 25 short-diamond stocks
- bar3 vs bar6 entry
- RSI threshold: < 35, < 40, < 45
- NIFTY filter variants:
    none / NIFTY below VWAP at bar3 / NIFTY first-bar bearish / both
- TP/SL: full-day hold OR (1.5% TP, 2.0% SL) OR (1.0% TP, 1.0% SL)

Output: results/08_diamond_short_ranking.csv
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


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
DIAMONDS_PATH = os.path.join(RESULTS_DIR, '07_short_diamonds.txt')
OUT_PATH = os.path.join(RESULTS_DIR, '08_diamond_short_ranking.csv')


def load_diamonds() -> list[str]:
    with open(DIAMONDS_PATH) as f:
        return [s.strip() for s in f if s.strip()]


def build_nifty_regime() -> pd.DataFrame:
    """Return per-session NIFTY regime tags."""
    df = load_5min('NIFTY50')
    if df.empty:
        return pd.DataFrame()
    df = enrich(df, or_minutes=15)

    # bar 3 (09:30) and bar 6 (09:45) features
    bar3 = df[df['bar_idx'] == 3].set_index('session')[['close', 'vwap', 'rsi', 'open']]
    bar6 = df[df['bar_idx'] == 6].set_index('session')[['close', 'vwap', 'rsi']]
    bar3.columns = [f'n_b3_{c}' for c in bar3.columns]
    bar6.columns = [f'n_b6_{c}' for c in bar6.columns]
    first_bar = df[df['bar_idx'] == 0].set_index('session')[['open', 'close']]
    first_bar.columns = ['n_b0_open', 'n_b0_close']
    sess_open = df.groupby('session')['day_open'].first().rename('n_day_open')
    sess_pc = df.groupby('session')['prev_close'].first().rename('n_prev_close')

    out = pd.concat([bar3, bar6, first_bar, sess_open, sess_pc], axis=1)
    out['n_below_vwap_b3'] = (out['n_b3_close'] < out['n_b3_vwap']).astype(int)
    out['n_below_vwap_b6'] = (out['n_b6_close'] < out['n_b6_vwap']).astype(int)
    out['n_first_bearish'] = (out['n_b0_close'] < out['n_b0_open']).astype(int)
    out['n_gap_pct'] = (out['n_day_open'] - out['n_prev_close']) / out['n_prev_close'] * 100
    out['n_b3_change_pct'] = (out['n_b3_close'] - out['n_day_open']) / out['n_day_open'] * 100
    return out.reset_index()


def diamond_short(df: pd.DataFrame,
                  rsi_threshold: float = 40,
                  entry_bar: int = 3,
                  require_below_vwap: bool = True) -> pd.Series:
    """Short signal at entry_bar: below VWAP + RSI < threshold."""
    cond = (
        (df['bar_idx'].values == entry_bar)
        & (df['rsi'].values < rsi_threshold)
        & ((df['close'].values < df['vwap'].values) if require_below_vwap else True)
    )
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig


def filter_by_nifty(sig: pd.Series, df: pd.DataFrame, nifty: pd.DataFrame,
                    nifty_filter: str) -> pd.Series:
    """Apply NIFTY-regime filter to the per-stock signal."""
    if nifty_filter == 'none':
        return sig
    if nifty.empty:
        return sig

    # Build mask for sessions matching the filter
    if nifty_filter == 'below_vwap_b3':
        ok_sessions = nifty.loc[nifty['n_below_vwap_b3'] == 1, 'session']
    elif nifty_filter == 'first_bearish':
        ok_sessions = nifty.loc[nifty['n_first_bearish'] == 1, 'session']
    elif nifty_filter == 'both':
        ok_sessions = nifty.loc[
            (nifty['n_below_vwap_b3'] == 1) & (nifty['n_first_bearish'] == 1),
            'session',
        ]
    elif nifty_filter == 'b3_change_neg':
        ok_sessions = nifty.loc[nifty['n_b3_change_pct'] < -0.1, 'session']
    elif nifty_filter == 'b3_change_neg_strong':
        ok_sessions = nifty.loc[nifty['n_b3_change_pct'] < -0.3, 'session']
    else:
        return sig

    ok_set = set(pd.to_datetime(ok_sessions).values.astype('datetime64[ns]'))
    mask = df['session'].isin(list(ok_set))
    return sig & mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-trades', type=int, default=30)
    args = ap.parse_args()

    diamonds = load_diamonds()
    print(f'Loaded {len(diamonds)} diamond stocks')

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
        except Exception as e:
            print(f'  enrich fail {sym}: {e}')
    print(f'  done {len(cache)} stocks in {time.time()-t0:.0f}s')

    # parameter grid
    PARAM_GRID = list(itertools.product(
        [3, 6],                          # entry_bar
        [35, 40, 45],                    # rsi_threshold
        ['none', 'below_vwap_b3', 'first_bearish', 'both',
         'b3_change_neg', 'b3_change_neg_strong'],   # nifty_filter
    ))
    EXIT_GRID = [
        (0.5, 1.5, 60),     # 0.5% TP, 1.5% SL, full-day hold
        (1.0, 1.5, 60),
        (1.5, 2.0, 60),
        (0.7, 1.0, 48),
        (5.0, 2.0, 60),     # ~hold-to-EOD with wide SL only
        (3.0, 1.5, 60),
        (2.0, 99.0, 60),    # pure hold-to-EOD with TP only (very wide SL)
    ]
    print(f'Sweep size: {len(PARAM_GRID)} param combos × {len(EXIT_GRID)} exits = {len(PARAM_GRID)*len(EXIT_GRID)} systems')

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
        n_total = len(PARAM_GRID) * len(EXIT_GRID)
        t_run = time.time()
        for entry_bar, rsi_thr, nifty_filter in PARAM_GRID:
            params = dict(entry_bar=entry_bar, rsi_threshold=rsi_thr, nifty_filter=nifty_filter)
            for tp, sl, hold in EXIT_GRID:
                cell += 1
                all_trades = []
                stocks_with_trades = 0
                stock_sessions = 0
                for sym, df in cache.items():
                    sig = diamond_short(df, rsi_threshold=rsi_thr, entry_bar=entry_bar)
                    sig = filter_by_nifty(sig, df, nifty, nifty_filter)
                    rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold, direction='short')
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
                    family='diamond_short',
                    params=str(params),
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

        elapsed = time.time() - t_run
        print(f'Done in {elapsed:.0f}s — output {OUT_PATH}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
