"""Stage 9 — walk-forward validation of top diamond-short candidates.

Train: 2024-03-18 to 2025-09-30 (~18 months)
Test:  2025-10-01 to 2026-03-25 (~6 months held out)

Pass criteria:
  - Train WR >= 75%
  - Test WR >= 70% (allow 5% drift)
  - Test PF >= 1.5
  - Test MaxDD <= 18%

Output: results/09_walk_forward_final.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import (  # type: ignore
    load_5min, enrich, simulate_signals, trade_stats, TradeRules,
)
from _strategies import _one_per_session, _no_late_entries  # type: ignore
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
mod = importlib.import_module('08_diamond_short_with_nifty')
build_nifty_regime = mod.build_nifty_regime
diamond_short = mod.diamond_short
filter_by_nifty = mod.filter_by_nifty


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
DIAMONDS_PATH = os.path.join(RESULTS_DIR, '07_short_diamonds.txt')
OUT = os.path.join(RESULTS_DIR, '09_walk_forward_final.csv')

TRAIN_START = '2024-03-18'
TRAIN_END = '2025-09-30'
TEST_START = '2025-10-01'
TEST_END = '2026-03-25'


# Top 3 candidates from Stage 8 (extracted from CSV inspection)
CANDIDATES = [
    dict(label='top1_strict', entry_bar=6, rsi_threshold=35, nifty_filter='b3_change_neg_strong',
         tp=0.5, sl=1.5, hold=60),
    dict(label='top2_both',   entry_bar=6, rsi_threshold=35, nifty_filter='both',
         tp=0.5, sl=1.5, hold=60),
    dict(label='top3_neg',    entry_bar=6, rsi_threshold=35, nifty_filter='b3_change_neg',
         tp=0.5, sl=1.5, hold=60),
    dict(label='loose_rsi40', entry_bar=6, rsi_threshold=40, nifty_filter='none',
         tp=0.5, sl=1.5, hold=60),
    dict(label='loose_rsi40_neg', entry_bar=6, rsi_threshold=40, nifty_filter='b3_change_neg',
         tp=0.5, sl=1.5, hold=60),
    dict(label='loose_rsi40_first_bear', entry_bar=6, rsi_threshold=40, nifty_filter='first_bearish',
         tp=0.5, sl=1.5, hold=60),
]


def evaluate(cands, cache_train, cache_test, nifty_train, nifty_test):
    rows = []
    for c in cands:
        def _run(cache, nifty):
            all_trades = []
            stocks_with_trades = 0
            stock_sessions = 0
            for sym, df in cache.items():
                sig = diamond_short(df, rsi_threshold=c['rsi_threshold'], entry_bar=c['entry_bar'])
                sig = filter_by_nifty(sig, df, nifty, c['nifty_filter'])
                rules = TradeRules(tp_pct=c['tp'], sl_pct=c['sl'], max_hold_bars=c['hold'], direction='short')
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

        tr = _run(cache_train, nifty_train)
        te = _run(cache_test, nifty_test)
        if tr is None or te is None:
            print(f'  [{c["label"]}] no trades — skip')
            continue
        drift = tr['win_rate'] - te['win_rate']
        passes = (
            tr['win_rate'] >= 0.75
            and te['win_rate'] >= 0.70
            and te['profit_factor'] >= 1.5
            and te['max_dd_pct'] <= 18.0
            and te['n_trades'] >= 20
        )
        marker = '[PASS]' if passes else '[FAIL]'
        print(f'  {marker} {c["label"]}: TRAIN WR={tr["win_rate"]:.1%} n={tr["n_trades"]} '
              f'PF={tr["profit_factor"]:.2f} | TEST WR={te["win_rate"]:.1%} n={te["n_trades"]} '
              f'PF={te["profit_factor"]:.2f} DD={te["max_dd_pct"]:.1f}% drift={drift:+.1%}')
        rows.append(dict(
            label=c['label'], params=str({k: v for k, v in c.items() if k != 'label'}),
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
    return rows


def main() -> int:
    with open(DIAMONDS_PATH) as f:
        diamonds = [s.strip() for s in f if s.strip()]
    print(f'{len(diamonds)} diamond stocks')

    # build NIFTY regimes
    print('Building NIFTY regime tables ...')
    nifty_full = build_nifty_regime()
    nifty_full['session'] = pd.to_datetime(nifty_full['session'])
    nifty_train = nifty_full[nifty_full['session'] <= TRAIN_END]
    nifty_test = nifty_full[nifty_full['session'] >= TEST_START]
    print(f'  NIFTY train sessions: {len(nifty_train)}, test sessions: {len(nifty_test)}')

    print('Loading + enriching diamond data ...')
    cache_train, cache_test = {}, {}
    t0 = time.time()
    for sym in diamonds:
        df = load_5min(sym, start=TRAIN_START, end=TEST_END)
        if df.empty or len(df) < 500:
            continue
        df = enrich(df, or_minutes=15)
        cache_train[sym] = df[df.index <= TRAIN_END]
        cache_test[sym] = df[df.index >= TEST_START]
    print(f'  loaded {len(cache_train)} stocks in {time.time()-t0:.0f}s')
    print()

    print('=== Walk-forward results ===')
    rows = evaluate(CANDIDATES, cache_train, cache_test, nifty_train, nifty_test)

    fields = list(rows[0].keys()) if rows else ['label']
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\nDone — output {OUT}')
    n_pass = sum(r['passes'] for r in rows)
    print(f'{n_pass}/{len(rows)} candidates pass walk-forward')
    return 0


if __name__ == '__main__':
    sys.exit(main())
