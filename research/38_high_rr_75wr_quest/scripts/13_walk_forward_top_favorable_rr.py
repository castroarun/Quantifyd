"""Stage 13 — walk-forward validate the top favorable-RR candidates.

Train: 2024-03-18 to 2025-09-30
Test:  2025-10-01 to 2026-03-25

Pass criteria:
  - Train PF >= 1.4
  - Test PF >= 1.3
  - Test n >= 25
  - Test MaxDD <= 12%
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
import time

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
DIAMONDS_PATH = os.path.normpath(os.path.join(
    R37, '..', 'results', '07_short_diamonds.txt',
))
OUT = os.path.join(RESULTS_DIR, '13_walk_forward_favorable_rr.csv')

TRAIN_START = '2024-03-18'
TRAIN_END = '2025-09-30'
TEST_START = '2025-10-01'
TEST_END = '2026-03-25'

# Top favorable-RR candidates from Stage 12
CANDIDATES = [
    dict(label='top1_4bar_55_belowvwap_15_10', n_bars=4, rsi_max=55,
         nifty='below_vwap_b3', tp=1.5, sl=1.0, hold=60),
    dict(label='top2_3bar_70_both_10_05',     n_bars=3, rsi_max=70,
         nifty='both', tp=1.0, sl=0.5, hold=60),
    dict(label='top3_3bar_55_b3neg_07_06',    n_bars=3, rsi_max=55,
         nifty='b3_change_neg', tp=0.7, sl=0.6, hold=60),
    dict(label='top4_4bar_55_belowvwap_15_07', n_bars=4, rsi_max=55,
         nifty='below_vwap_b3', tp=1.5, sl=0.7, hold=60),
    dict(label='top5_3bar_55_both_10_07',     n_bars=3, rsi_max=55,
         nifty='both', tp=1.0, sl=0.7, hold=60),
    dict(label='top6_3bar_55_b3negstrong_07_04', n_bars=3, rsi_max=55,
         nifty='b3_change_neg_strong', tp=0.7, sl=0.4, hold=60),
]


def main() -> int:
    cohort = open(DIAMONDS_PATH).read().split()
    print(f'Cohort: {len(cohort)} stocks', flush=True)

    print('Loading + enriching ...', flush=True)
    cache_train, cache_test = {}, {}
    t0 = time.time()
    for sym in cohort:
        df = load_5min(sym, start=TRAIN_START, end=TEST_END)
        if df.empty or len(df) < 500:
            continue
        df = enrich(df, or_minutes=15)
        cache_train[sym] = df[df.index <= TRAIN_END]
        cache_test[sym] = df[df.index >= TEST_START]
    print(f'  loaded {len(cache_train)} stocks in {time.time()-t0:.0f}s', flush=True)

    nifty = mod_8.build_nifty_regime()
    nifty['session'] = pd.to_datetime(nifty['session'])
    nifty_train = nifty[nifty['session'] <= TRAIN_END]
    nifty_test = nifty[nifty['session'] >= TEST_START]

    fields = ['label', 'params',
              'train_n', 'train_wr', 'train_pf', 'train_dd', 'train_sharpe',
              'test_n', 'test_wr', 'test_pf', 'test_dd', 'test_sharpe',
              'wr_drift', 'pf_drift', 'passes']

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        print('\n=== Walk-forward results ===', flush=True)
        for c in CANDIDATES:
            def _run(cache, nfilt):
                all_t = []
                ses = 0
                stocks_with = 0
                for sym, df in cache.items():
                    sig = multi_bar_confirm(df, direction='short',
                                            n_bars=c['n_bars'], rsi_max=c['rsi_max'])
                    if c['nifty'] != 'none':
                        sig = mod_8.filter_by_nifty(sig, df, nfilt, c['nifty'])
                    rules = TradeRules(tp_pct=c['tp'], sl_pct=c['sl'],
                                       max_hold_bars=c['hold'], direction='short')
                    trades = simulate_signals(df, sig, rules)
                    if len(trades) == 0:
                        continue
                    trades['symbol'] = sym
                    all_t.append(trades)
                    stocks_with += 1
                    ses += df['session'].nunique()
                if not all_t:
                    return None
                t = pd.concat(all_t, ignore_index=True)
                avg_ses = ses / max(stocks_with, 1)
                return trade_stats(t, sessions=int(avg_ses))

            tr = _run(cache_train, nifty_train)
            te = _run(cache_test, nifty_test)
            if tr is None or te is None:
                print(f'  [SKIP] {c["label"]} - no trades', flush=True)
                continue
            wr_drift = tr['win_rate'] - te['win_rate']
            pf_drift = tr['profit_factor'] - te['profit_factor']
            passes = (
                tr['profit_factor'] >= 1.4
                and te['profit_factor'] >= 1.3
                and te['n_trades'] >= 25
                and te['max_dd_pct'] <= 12
            )
            mark = '[PASS]' if passes else '[FAIL]'
            print(f'  {mark} {c["label"]}: TRAIN WR={tr["win_rate"]:.1%} '
                  f'n={tr["n_trades"]} PF={tr["profit_factor"]:.2f} | '
                  f'TEST WR={te["win_rate"]:.1%} n={te["n_trades"]} '
                  f'PF={te["profit_factor"]:.2f} DD={te["max_dd_pct"]:.1f}% '
                  f'wr_drift={wr_drift:+.1%}',
                  flush=True)
            w.writerow(dict(
                label=c['label'],
                params=str({k: v for k, v in c.items() if k != 'label'}),
                train_n=tr['n_trades'], train_wr=round(tr['win_rate'], 4),
                train_pf=round(tr['profit_factor'], 3), train_dd=round(tr['max_dd_pct'], 3),
                train_sharpe=round(tr['sharpe'], 3),
                test_n=te['n_trades'], test_wr=round(te['win_rate'], 4),
                test_pf=round(te['profit_factor'], 3), test_dd=round(te['max_dd_pct'], 3),
                test_sharpe=round(te['sharpe'], 3),
                wr_drift=round(wr_drift, 4), pf_drift=round(pf_drift, 3),
                passes=int(passes),
            ))

    print(f'\nWrote {OUT}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
