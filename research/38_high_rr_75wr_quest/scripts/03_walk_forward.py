"""Stage D — walk-forward validation of top per-pattern systems.

Train: 2024-03-18 to 2025-09-30 (~18 months)
Test:  2025-10-01 to 2026-03-25 (~6 months)

Pass criteria (the user's spec):
  - Train WR >= 0.72 (drift tolerance)
  - Test WR >= 0.75
  - Test PF >= 2.5
  - Test MaxDD <= 10%
  - Test n >= 30 trades

Reads top-N systems from results/02_<pattern>_<direction>_ranking.csv
(filtered by passes_gates and n_trades >= 60 for stability), re-runs
on train + test windows separately, writes results/03_walk_forward.csv.

ASCII-only console (Windows cp1252 safe).
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
import time

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
mod_02 = importlib.import_module('02_pattern_sweep')
get_cohort = mod_02.get_cohort
filter_by_nifty = mod_02.filter_by_nifty


RESULTS_DIR = os.path.normpath(os.path.join(_HERE, '..', 'results'))
LOGS_DIR = os.path.normpath(os.path.join(_HERE, '..', 'logs'))

TRAIN_START = '2024-03-18'
TRAIN_END = '2025-09-30'
TEST_START = '2025-10-01'
TEST_END = '2026-03-25'

OUT_PATH = os.path.join(RESULTS_DIR, '03_walk_forward.csv')


def load_split_data(symbols, log=print):
    cache_train, cache_test = {}, {}
    t0 = time.time()
    for sym in symbols:
        try:
            df = load_5min(sym, start=TRAIN_START, end=TEST_END)
            if df.empty or len(df) < 500:
                continue
            df = enrich(df, or_minutes=15)
            cache_train[sym] = df[df.index <= TRAIN_END]
            cache_test[sym] = df[df.index >= TEST_START]
        except Exception as e:
            log(f'  {sym} load fail: {e}')
    log(f'  loaded {len(cache_train)} stocks (train+test) in {time.time() - t0:.0f}s')
    return cache_train, cache_test


def evaluate_system(pattern: str, direction: str, params: dict,
                    nifty_filter: str, tp: float, sl: float, hold: int,
                    cache, nifty) -> dict | None:
    func = PATTERN_FUNCS[pattern]
    all_trades = []
    stocks_with_trades = 0
    stock_sessions = 0
    for sym, df in cache.items():
        try:
            sig = func(df, direction=direction, **params)
        except Exception:
            continue
        sig = filter_by_nifty(sig, df, nifty, nifty_filter)
        rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold, direction=direction)
        trades = simulate_signals(df, sig, rules)
        if len(trades) == 0:
            continue
        trades['symbol'] = sym
        all_trades.append(trades)
        stocks_with_trades += 1
        stock_sessions += df['session'].nunique()
    if not all_trades:
        return None
    t = pd.concat(all_trades, ignore_index=True)
    avg_sessions = stock_sessions / max(stocks_with_trades, 1)
    stats = trade_stats(t, sessions=int(avg_sessions))
    stats['n_stocks'] = stocks_with_trades
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--patterns', nargs='*', default=None)
    ap.add_argument('--top-per-pattern', type=int, default=10,
                    help='top-N systems per pattern+direction (by WR with min n=50)')
    ap.add_argument('--cohort-size', type=int, default=30)
    ap.add_argument('--min-rank-wr', type=float, default=0.70)
    args = ap.parse_args()

    log_path = os.path.join(LOGS_DIR, '03_walk_forward.log')
    log_f = open(log_path, 'a', encoding='utf-8', buffering=1)

    def log(msg: str):
        sys.stdout.write(msg + '\n')
        sys.stdout.flush()
        log_f.write(msg + '\n')
        log_f.flush()

    patterns = args.patterns if args.patterns else list(PATTERN_FUNCS.keys())

    # gather candidates from all ranking CSVs
    candidates = []
    for p in patterns:
        for d in ('long', 'short'):
            path = os.path.join(RESULTS_DIR, f'02_{p}_{d}_ranking.csv')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            sub = df[(df['n_trades'] >= 60) & (df['win_rate'] >= args.min_rank_wr)
                     & (df['profit_factor'] >= 2.0)]
            sub = sub.sort_values(['win_rate', 'profit_factor'], ascending=False)
            top = sub.head(args.top_per_pattern)
            for _, r in top.iterrows():
                candidates.append(dict(
                    pattern=p, direction=d,
                    params=ast.literal_eval(r['params']),
                    nifty_filter=r['nifty_filter'],
                    tp=r['tp_pct'], sl=r['sl_pct'], hold=int(r['max_hold_bars']),
                    full_wr=r['win_rate'], full_pf=r['profit_factor'],
                    full_n=r['n_trades'],
                ))

    log(f'Loaded {len(candidates)} candidate systems for walk-forward')
    if not candidates:
        log('No candidates — Stage B may not have run yet')
        return 1

    # Build NIFTY regime once for full window, split
    log('Building NIFTY regime ...')
    nifty_full = build_nifty_regime()
    nifty_full['session'] = pd.to_datetime(nifty_full['session'])
    nifty_train = nifty_full[nifty_full['session'] <= TRAIN_END]
    nifty_test = nifty_full[nifty_full['session'] >= TEST_START]
    log(f'  NIFTY: {len(nifty_train)} train sessions, {len(nifty_test)} test sessions')

    # Load all unique cohorts (union of all patterns+dirs)
    all_syms = set()
    cohorts = {}
    for p in patterns:
        for d in ('long', 'short'):
            cohort = get_cohort(p, d, top_n=args.cohort_size)
            cohorts[(p, d)] = cohort
            all_syms.update(cohort)
    all_syms = sorted(all_syms)
    log(f'Total unique cohort union: {len(all_syms)} stocks')

    log('Loading + enriching split data ...')
    cache_train, cache_test = load_split_data(all_syms, log=log)

    fields = ['pattern', 'direction', 'params', 'nifty_filter',
              'tp', 'sl', 'hold',
              'full_wr', 'full_pf', 'full_n',
              'train_wr', 'train_n', 'train_pf', 'train_dd', 'train_sharpe',
              'test_wr', 'test_n', 'test_pf', 'test_dd', 'test_sharpe',
              'wr_drift', 'passes']

    n_pass = 0
    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for i, c in enumerate(candidates, 1):
            cohort = cohorts[(c['pattern'], c['direction'])]
            cache_tr_sub = {s: cache_train[s] for s in cohort if s in cache_train}
            cache_te_sub = {s: cache_test[s] for s in cohort if s in cache_test}

            tr = evaluate_system(c['pattern'], c['direction'], c['params'],
                                 c['nifty_filter'], c['tp'], c['sl'], c['hold'],
                                 cache_tr_sub, nifty_train)
            te = evaluate_system(c['pattern'], c['direction'], c['params'],
                                 c['nifty_filter'], c['tp'], c['sl'], c['hold'],
                                 cache_te_sub, nifty_test)
            if tr is None or te is None:
                log(f'  [{i}/{len(candidates)}] {c["pattern"]}/{c["direction"]} no trades — skip')
                continue

            drift = tr['win_rate'] - te['win_rate']
            passes = (
                tr['win_rate'] >= 0.72
                and te['win_rate'] >= 0.75
                and te['profit_factor'] >= 2.5
                and te['max_dd_pct'] <= 10.0
                and te['n_trades'] >= 30
            )
            marker = '[PASS]' if passes else '[FAIL]'
            log(f'  [{i}/{len(candidates)}] {marker} {c["pattern"]}/{c["direction"]} '
                f'tp={c["tp"]} sl={c["sl"]} nifty={c["nifty_filter"]} '
                f'TR WR={tr["win_rate"]:.1%} n={tr["n_trades"]} | '
                f'TE WR={te["win_rate"]:.1%} n={te["n_trades"]} PF={te["profit_factor"]:.2f} '
                f'DD={te["max_dd_pct"]:.1f}%')
            if passes:
                n_pass += 1

            row = dict(
                pattern=c['pattern'], direction=c['direction'],
                params=str(c['params']), nifty_filter=c['nifty_filter'],
                tp=c['tp'], sl=c['sl'], hold=c['hold'],
                full_wr=round(c['full_wr'], 4), full_pf=round(c['full_pf'], 3),
                full_n=c['full_n'],
                train_wr=round(tr['win_rate'], 4), train_n=tr['n_trades'],
                train_pf=round(tr['profit_factor'], 3),
                train_dd=round(tr['max_dd_pct'], 3),
                train_sharpe=round(tr['sharpe'], 3),
                test_wr=round(te['win_rate'], 4), test_n=te['n_trades'],
                test_pf=round(te['profit_factor'], 3),
                test_dd=round(te['max_dd_pct'], 3),
                test_sharpe=round(te['sharpe'], 3),
                wr_drift=round(drift, 4),
                passes=int(passes),
            )
            w.writerow(row)
            f.flush()

    log(f'\nDone. {n_pass}/{len(candidates)} systems pass walk-forward.')
    log(f'Output: {OUT_PATH}')
    log_f.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
