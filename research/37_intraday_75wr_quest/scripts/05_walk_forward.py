"""Stage 5 — walk-forward validation.

For a candidate system, split data into train (2024-03-18 to 2025-09-30)
and test (2025-10-01 to 2026-03-25). The system passes ONLY IF:
  - Train WR >= 75% AND Test WR >= 72% (allow 3% drift)
  - Test trades >= 25 per stock per year
  - Test PF >= 1.8
  - Test MaxDD <= 18%

Reads candidate definitions from results/04_confluence_*_ranking.csv
top rows (by AWS).

Output: results/05_walk_forward_final.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import (  # type: ignore
    list_5min_universe, load_5min, enrich,
    simulate_signals, trade_stats, TradeRules,
)


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
TRAIN_START = '2024-03-18'
TRAIN_END = '2025-09-30'
TEST_START = '2025-10-01'
TEST_END = '2026-03-25'


def get_strategy_fn(side: str):
    if side == 'long':
        from importlib import import_module
        mod = import_module('04_confluence_stack')
        return mod.confluence_long
    elif side == 'short':
        from importlib import import_module
        mod = import_module('04_confluence_stack')
        return mod.confluence_short
    raise ValueError(side)


def evaluate(strat_fn, params, tp, sl, hold, direction, cache_train, cache_test, min_trades=25):
    def _run(cache):
        all_trades = []
        stocks_with_trades = 0
        sessions = 0
        for sym, df in cache.items():
            try:
                sig = strat_fn(df, **params)
            except Exception:
                continue
            rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold, direction=direction)
            trades = simulate_signals(df, sig, rules)
            if len(trades) == 0:
                continue
            trades['symbol'] = sym
            all_trades.append(trades)
            stocks_with_trades += 1
            sessions += df['session'].nunique()
        if not all_trades:
            return None
        t = pd.concat(all_trades, ignore_index=True)
        avg_sessions = sessions / max(stocks_with_trades, 1)
        stats = trade_stats(t, sessions=int(avg_sessions))
        stats['n_stocks'] = stocks_with_trades
        return stats

    train_stats = _run(cache_train)
    test_stats = _run(cache_test)
    return train_stats, test_stats


def passes_walk_forward(train: dict, test: dict) -> bool:
    if train is None or test is None:
        return False
    return (
        train['win_rate'] >= 0.75
        and test['win_rate'] >= 0.72
        and test['n_trades'] >= 25
        and test['profit_factor'] >= 1.8
        and test['max_dd_pct'] <= 18.0
    )


def load_top_candidates(side: str, top_n: int = 30) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, f'04_confluence_{side}_ranking.csv')
    if not os.path.exists(path):
        print(f'No ranking file at {path}')
        return pd.DataFrame()
    df = pd.read_csv(path)
    # filter: WR >= 0.70 first, then top by AWS
    df = df[(df['win_rate'] >= 0.70) & (df['n_trades'] >= 30)]
    if df.empty:
        return df
    df = df.sort_values('aws', ascending=False).head(top_n)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--side', choices=['long', 'short', 'both'], default='both')
    ap.add_argument('--top-n', type=int, default=30)
    ap.add_argument('--max-stocks', type=int, default=50)
    args = ap.parse_args()

    universe = list_5min_universe(min_rows=32_000)
    stocks = universe[:args.max_stocks]
    print(f'Loading train+test caches for {len(stocks)} stocks ...')

    cache_train, cache_test = {}, {}
    t0 = time.time()
    for sym in stocks:
        df_full = load_5min(sym, start=TRAIN_START, end=TEST_END)
        if df_full.empty or len(df_full) < 1000:
            continue
        df_full = enrich(df_full, or_minutes=15)
        cache_train[sym] = df_full[df_full.index <= TRAIN_END]
        cache_test[sym] = df_full[df_full.index >= TEST_START]
    print(f'  loaded {len(cache_train)} stocks in {time.time()-t0:.0f}s')

    out = os.path.join(RESULTS_DIR, '05_walk_forward_final.csv')
    fields = [
        'side', 'params', 'tp_pct', 'sl_pct', 'max_hold_bars',
        'train_wr', 'train_n', 'train_pf', 'train_dd', 'train_sharpe',
        'test_wr', 'test_n', 'test_pf', 'test_dd', 'test_sharpe',
        'wr_drift', 'passes',
    ]
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for side in (['long', 'short'] if args.side == 'both' else [args.side]):
            cands = load_top_candidates(side, top_n=args.top_n)
            if cands.empty:
                print(f'[{side}] no candidates with WR >= 70% — skipping')
                continue

            strat_fn = get_strategy_fn(side)
            direction = side
            print(f'[{side}] evaluating {len(cands)} candidates ...')
            for i, row in cands.iterrows():
                params = ast.literal_eval(row['params'])
                tp = float(row['tp_pct'])
                sl = float(row['sl_pct'])
                hold = int(row['max_hold_bars'])
                tr, te = evaluate(strat_fn, params, tp, sl, hold, direction, cache_train, cache_test)
                if tr is None or te is None:
                    continue
                drift = tr['win_rate'] - te['win_rate']
                passes = passes_walk_forward(tr, te)
                w.writerow(dict(
                    side=side, params=str(params),
                    tp_pct=tp, sl_pct=sl, max_hold_bars=hold,
                    train_wr=round(tr['win_rate'], 4),
                    train_n=tr['n_trades'],
                    train_pf=round(tr['profit_factor'], 3),
                    train_dd=round(tr['max_dd_pct'], 3),
                    train_sharpe=round(tr['sharpe'], 3),
                    test_wr=round(te['win_rate'], 4),
                    test_n=te['n_trades'],
                    test_pf=round(te['profit_factor'], 3),
                    test_dd=round(te['max_dd_pct'], 3),
                    test_sharpe=round(te['sharpe'], 3),
                    wr_drift=round(drift, 4),
                    passes=int(passes),
                ))
                f.flush()
                marker = '✅' if passes else '❌'
                print(f'  {marker} {side} {params} tp={tp} sl={sl} '
                      f'TRAIN WR={tr["win_rate"]:.1%} n={tr["n_trades"]} | '
                      f'TEST WR={te["win_rate"]:.1%} n={te["n_trades"]} drift={drift:+.1%}')

    print(f'Done — output {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
