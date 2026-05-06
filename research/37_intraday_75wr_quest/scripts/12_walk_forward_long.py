"""Stage 12 — walk-forward validation for long-side candidates.

Tests the top WR-passing candidates from Stage 10 (and any from Stage 11
when those sub-agents complete) on the held-out 2025-10 to 2026-03 window.

Pass criteria (matching short-side):
  - Train WR >= 75%
  - Test WR >= 70% (allow 5% drift)
  - Test PF >= 1.4 (relaxed from 1.5 because long-side has tighter math)
  - Test MaxDD <= 12%
  - Test n_trades >= 20
"""

from __future__ import annotations

import csv
import importlib
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import load_5min, enrich, simulate_signals, trade_stats, TradeRules  # type: ignore
mod_8 = importlib.import_module('08_diamond_short_with_nifty')
mod_10 = importlib.import_module('10_diamond_long_with_nifty')
build_nifty_regime = mod_8.build_nifty_regime
diamond_long = mod_10.diamond_long
filter_by_nifty_up = mod_10.filter_by_nifty_up


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
DIAMONDS_PATH = os.path.join(RESULTS_DIR, '10_long_diamonds.txt')
OUT = os.path.join(RESULTS_DIR, '12_walk_forward_long.csv')

TRAIN_START = '2024-03-18'
TRAIN_END = '2025-09-30'
TEST_START = '2025-10-01'
TEST_END = '2026-03-25'

# Top WR-passing candidates from Stage 10 (>= 73% WR, varying PF)
CANDIDATES = [
    dict(label='top_wr_75', entry_bar=6, rsi_threshold=70, nifty_filter='both',
         tp=0.5, sl=1.5, hold=60),
    dict(label='top_wr_75_first_bull', entry_bar=6, rsi_threshold=70, nifty_filter='first_bullish',
         tp=0.5, sl=1.5, hold=60),
    dict(label='top_wr_75_b9', entry_bar=9, rsi_threshold=70, nifty_filter='both',
         tp=0.5, sl=1.5, hold=60),
    dict(label='best_pf_60', entry_bar=9, rsi_threshold=70, nifty_filter='both',
         tp=5.0, sl=2.0, hold=60),
    dict(label='best_pf_strict', entry_bar=9, rsi_threshold=70, nifty_filter='b3_change_pos_strong',
         tp=5.0, sl=2.0, hold=60),
    dict(label='hi_freq_72', entry_bar=3, rsi_threshold=65, nifty_filter='above_vwap_b3',
         tp=0.5, sl=1.5, hold=60),
]


def main() -> int:
    with open(DIAMONDS_PATH) as f:
        diamonds = [s.strip() for s in f if s.strip()]
    print(f'{len(diamonds)} long-diamond stocks')

    print('Building NIFTY regime ...')
    nifty_full = build_nifty_regime()
    nifty_full['session'] = pd.to_datetime(nifty_full['session'])
    nifty_train = nifty_full[nifty_full['session'] <= TRAIN_END]
    nifty_test = nifty_full[nifty_full['session'] >= TEST_START]

    print('Loading + enriching ...')
    cache_train, cache_test = {}, {}
    t0 = time.time()
    for sym in diamonds:
        df = load_5min(sym, start=TRAIN_START, end=TEST_END)
        if df.empty or len(df) < 500:
            continue
        df = enrich(df, or_minutes=15)
        cache_train[sym] = df[df.index <= TRAIN_END]
        cache_test[sym] = df[df.index >= TEST_START]
    print(f'  {len(cache_train)} stocks in {time.time()-t0:.0f}s')
    print()

    fields = [
        'label', 'params',
        'train_wr', 'train_n', 'train_pf', 'train_dd', 'train_sharpe',
        'test_wr', 'test_n', 'test_pf', 'test_dd', 'test_sharpe',
        'wr_drift', 'passes',
    ]
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        print('=== Walk-forward results (LONG side) ===')
        for c in CANDIDATES:
            def _run(cache, nifty):
                all_trades = []
                stocks_with_trades = 0
                stock_sessions = 0
                for sym, df in cache.items():
                    sig = diamond_long(df, rsi_threshold=c['rsi_threshold'], entry_bar=c['entry_bar'])
                    sig = filter_by_nifty_up(sig, df, nifty, c['nifty_filter'])
                    rules = TradeRules(tp_pct=c['tp'], sl_pct=c['sl'], max_hold_bars=c['hold'], direction='long')
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
                and te['profit_factor'] >= 1.4
                and te['max_dd_pct'] <= 12.0
                and te['n_trades'] >= 20
            )
            marker = '[PASS]' if passes else '[FAIL]'
            print(f'  {marker} {c["label"]}: TRAIN WR={tr["win_rate"]:.1%} n={tr["n_trades"]} '
                  f'PF={tr["profit_factor"]:.2f} | TEST WR={te["win_rate"]:.1%} n={te["n_trades"]} '
                  f'PF={te["profit_factor"]:.2f} DD={te["max_dd_pct"]:.1f}% drift={drift:+.1%}')
            w.writerow(dict(
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

    print(f'\nDone — output {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
