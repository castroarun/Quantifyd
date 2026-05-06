"""Stage 11c walk-forward — validate top long-reversal candidates.

Train: 2024-03-18 to 2025-09-30 (~18 months)
Test:  2025-10-01 to 2026-03-25 (~6 months held out)

Pass criteria:
  - Train WR >= 75%
  - Test WR >= 70% (5% drift allowed)
  - Test PF >= 1.5
  - Test MaxDD <= 18%
  - Test n_trades >= 20

Output: results/11c_walk_forward_long_reversal.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import (  # type: ignore
    load_5min, enrich, simulate_signals, trade_stats, TradeRules,
)
import importlib
mod_11c = importlib.import_module('11c_long_late_reversal')
mod_8 = importlib.import_module('08_diamond_short_with_nifty')
build_nifty_regime = mod_8.build_nifty_regime
late_reversal_signal = mod_11c.late_reversal_signal
filter_by_nifty_for_reversal = mod_11c.filter_by_nifty_for_reversal


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
DIAMONDS_PATH = os.path.join(RESULTS_DIR, '11c_long_reversal_diamonds.txt')
RANK_CSV = os.path.join(RESULTS_DIR, '11c_long_late_reversal_ranking.csv')
OUT = os.path.join(RESULTS_DIR, '11c_walk_forward_long_reversal.csv')

TRAIN_START = '2024-03-18'
TRAIN_END = '2025-09-30'
TEST_START = '2025-10-01'
TEST_END = '2026-03-25'


def parse_params(s: str) -> dict:
    """Parse the params dict from the CSV (eval-safe)."""
    import ast
    return ast.literal_eval(s)


def evaluate(cands, cache_train, cache_test, nifty_train, nifty_test):
    rows = []
    for c in cands:
        p = c['parsed']

        def _run(cache, nifty):
            all_trades = []
            stocks_with_trades = 0
            stock_sessions = 0
            for sym, df in cache.items():
                sig = late_reversal_signal(
                    df,
                    drop_pct=p['drop_pct'],
                    entry_window_start=p['window'][0],
                    entry_window_end=p['window'][1],
                    rsi_oversold=p['rsi_oversold'],
                    rsi_lift=p['rsi_lift'],
                )
                sig = filter_by_nifty_for_reversal(sig, df, nifty, p['nifty_filter'])
                rules = TradeRules(tp_pct=c['tp'], sl_pct=c['sl'],
                                   max_hold_bars=c['hold'], direction='long')
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
            label=c['label'], params=str(p),
            tp=c['tp'], sl=c['sl'], hold=c['hold'],
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=8,
                    help='top-N systems by WR to walk-forward (with min n=30)')
    args = ap.parse_args()

    if not os.path.exists(RANK_CSV):
        print(f'Missing {RANK_CSV} — run 11c_long_late_reversal.py first')
        return 1
    rank = pd.read_csv(RANK_CSV)
    # filter to passes and high WR with n>=30
    sub = rank[(rank['n_trades'] >= 30) & (rank['win_rate'] >= 0.72)]
    sub = sub.sort_values('win_rate', ascending=False)
    print(f'Picking top {args.top} candidates from {len(sub)} qualifying rows')
    top = sub.head(args.top)
    cands = []
    for i, r in enumerate(top.itertuples(), 1):
        try:
            parsed = parse_params(r.params)
        except Exception as e:
            print(f'  parse fail: {r.params} -> {e}')
            continue
        cands.append(dict(
            label=f'top{i}',
            parsed=parsed,
            tp=r.tp_pct,
            sl=r.sl_pct,
            hold=int(r.max_hold_bars),
            wr=r.win_rate,
            n=r.n_trades,
            pf=r.profit_factor,
        ))
        print(f'  top{i}: WR={r.win_rate:.1%} n={r.n_trades} PF={r.profit_factor:.2f} '
              f'params={parsed} tp={r.tp_pct} sl={r.sl_pct} hold={r.max_hold_bars}')

    if not cands:
        print('No candidates to walk-forward.')
        return 1

    with open(DIAMONDS_PATH) as f:
        diamonds = [s.strip() for s in f if s.strip()]
    print(f'\n{len(diamonds)} diamond stocks')

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
    print(f'  loaded {len(cache_train)} stocks in {time.time() - t0:.0f}s')
    print()

    print('=== Walk-forward results ===')
    rows = evaluate(cands, cache_train, cache_test, nifty_train, nifty_test)

    fields = list(rows[0].keys()) if rows else ['label']
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\nDone. Output: {OUT}')
    n_pass = sum(r['passes'] for r in rows)
    print(f'{n_pass}/{len(rows)} candidates pass walk-forward')
    return 0


if __name__ == '__main__':
    sys.exit(main())
