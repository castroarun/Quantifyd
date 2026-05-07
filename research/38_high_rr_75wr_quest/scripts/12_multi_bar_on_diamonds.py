"""Stage 12 — multi_bar SHORT signal on the EXISTING 25-stock short-diamond
cohort from research/37 + NIFTY filter.

Hypothesis: stocks already known to have structural short-bias should
respond best to a tight-stop SHORT signal pattern. Combine the cohort
strength (Stage 7 short-diamonds) with the tight-stop signal pattern
(multi_bar SHORT from Stage A) and NIFTY-regime filtering.
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

R37 = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    '37_intraday_75wr_quest', 'scripts',
))
sys.path.insert(0, R37)
from _engine import load_5min, enrich, simulate_signals, trade_stats, TradeRules  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pattern_lib import multi_bar_confirm  # type: ignore  # noqa: E402

spec = importlib.util.spec_from_file_location('mod_8',
    os.path.join(R37, '08_diamond_short_with_nifty.py'))
mod_8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_8)


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
OUT = os.path.join(RESULTS_DIR, '12_multi_bar_on_diamonds.csv')

DIAMONDS_PATH = os.path.normpath(os.path.join(
    R37, '..', 'results', '07_short_diamonds.txt',
))


def main() -> int:
    cohort = open(DIAMONDS_PATH).read().split()
    print(f'25 short-diamond cohort: {cohort}', flush=True)
    print()

    print('Loading + enriching ...', flush=True)
    t0 = time.time()
    cache = {}
    for sym in cohort:
        df = load_5min(sym)
        if df.empty or len(df) < 1000:
            continue
        try:
            cache[sym] = enrich(df, or_minutes=15)
        except Exception:
            continue
    print(f'  loaded {len(cache)}/{len(cohort)} stocks in {time.time()-t0:.0f}s', flush=True)

    nifty = mod_8.build_nifty_regime()
    print(f'  NIFTY: {len(nifty)} sessions', flush=True)

    PARAM_GRID = list(itertools.product(
        [3, 4],            # n_bars
        [55, 60, 65, 70],  # rsi_max
        ['none', 'b3_change_neg', 'b3_change_neg_strong', 'both', 'first_bearish', 'below_vwap_b3'],
    ))
    EXIT_GRID = [
        # Mostly favorable-RR (TP > SL) per the 38_quest charter
        (0.7, 0.4, 60),    # RR 1.75
        (1.0, 0.5, 60),    # RR 2.0
        (1.0, 0.6, 60),    # RR 1.67
        (1.0, 0.7, 60),    # RR 1.43
        (1.0, 0.9, 60),    # RR 1.11
        (1.5, 0.7, 60),    # RR 2.14
        (1.5, 1.0, 60),    # RR 1.5
        (1.5, 1.3, 60),    # RR 1.15
        (2.0, 1.0, 60),    # RR 2.0
        (0.5, 0.3, 60),    # RR 1.67 (tight stop)
        (0.7, 0.6, 60),    # RR 1.17
    ]
    n_total = len(PARAM_GRID) * len(EXIT_GRID)
    print(f'  Sweep size: {n_total}', flush=True)

    fields = [
        'cohort_size', 'n_bars', 'rsi_max', 'nifty_filter',
        'tp_pct', 'sl_pct', 'rr', 'hold_bars',
        'n_trades', 'win_rate', 'profit_factor', 'max_dd_pct',
        'sharpe', 'avg_win', 'avg_loss', 'total_return_pct',
        'trades_per_stock_year', 'passes',
    ]
    cell = 0
    t_run = time.time()
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for n_bars, rsi_max, nifty_filter in PARAM_GRID:
            for tp, sl, hold in EXIT_GRID:
                cell += 1
                all_trades = []
                stocks_with_trades = 0
                stock_sessions = 0
                for sym, df in cache.items():
                    try:
                        sig = multi_bar_confirm(
                            df, direction='short',
                            n_bars=n_bars,
                            rsi_max=rsi_max,
                        )
                    except Exception as e:
                        print(f'    sig fail {sym}: {e}', flush=True)
                        continue
                    if nifty_filter != 'none':
                        sig = mod_8.filter_by_nifty(sig, df, nifty, nifty_filter)
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
                rr = round(tp / sl, 2)
                trades_per_stock_year = stats['trades_per_year'] / max(stocks_with_trades, 1)
                passes = (
                    stats['win_rate'] >= 0.75
                    and rr >= 1.1
                    and stats['n_trades'] >= 30
                    and stats['profit_factor'] >= 2.0
                    and stats['max_dd_pct'] <= 12.0
                )
                w.writerow(dict(
                    cohort_size=stocks_with_trades,
                    n_bars=n_bars, rsi_max=rsi_max, nifty_filter=nifty_filter,
                    tp_pct=tp, sl_pct=sl, rr=rr, hold_bars=hold,
                    n_trades=stats['n_trades'],
                    win_rate=round(stats['win_rate'], 4),
                    profit_factor=round(stats['profit_factor'], 3),
                    max_dd_pct=round(stats['max_dd_pct'], 3),
                    sharpe=round(stats['sharpe'], 3),
                    avg_win=round(stats['avg_win'], 4),
                    avg_loss=round(stats['avg_loss'], 4),
                    total_return_pct=round(stats['total_return_pct'], 2),
                    trades_per_stock_year=round(trades_per_stock_year, 1),
                    passes=int(passes),
                ))
                f.flush()
                if stats['win_rate'] >= 0.65 and stats['n_trades'] >= 30:
                    marker = '[PASS]' if passes else '[high-WR]'
                    print(f'  {marker} [{cell}/{n_total}] n_bars={n_bars} rsi<={rsi_max} '
                          f'nifty={nifty_filter} tp={tp} sl={sl} RR={rr:.2f} -> '
                          f'WR={stats["win_rate"]:.1%} n={stats["n_trades"]} '
                          f'PF={stats["profit_factor"]:.2f} DD={stats["max_dd_pct"]:.1f}% '
                          f'Sharpe={stats["sharpe"]:.2f}',
                          flush=True)
                if cell % 30 == 0:
                    elapsed = time.time() - t_run
                    print(f'  progress: {cell}/{n_total} ({elapsed:.0f}s)', flush=True)

    print(f'\nDone. Wrote {OUT}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
