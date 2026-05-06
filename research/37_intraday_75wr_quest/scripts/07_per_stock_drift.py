"""Stage 7 — per-stock drift screen.

For every stock, compute the WR of the simplest possible intraday bias:

  L1: BUY at bar 3 close (09:30), SELL at session close
  L2: BUY at bar 6 close (09:45), SELL at session close
  S1: SHORT at bar 3 close, COVER at session close
  S2: SHORT at bar 6 close, COVER at session close

For each, also compute the WR conditional on simple regime markers:
  - Long-only above EMA200 daily
  - Long-only when prev-close was in lower half of prev-day's range
  - Short-only when prev-close in upper half

Goal: find stocks where ANY simple rule gives >= 70% WR. These are
"diamond stocks" worth running confluence systems on.

Output: results/07_per_stock_drift.csv
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import list_5min_universe, load_5min, enrich  # type: ignore


OUT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results', '07_per_stock_drift.csv'
))


def evaluate_stock(symbol: str) -> dict | None:
    df = load_5min(symbol)
    if df.empty or len(df) < 1000:
        return None
    df = enrich(df, or_minutes=15)

    # bar 3 close → session close
    bar3 = df[df['bar_idx'] == 3].copy()
    bar6 = df[df['bar_idx'] == 6].copy()
    if len(bar3) < 30:
        return None

    sess_close = df.groupby('session')['close'].last().reset_index()
    sess_close.columns = ['session', 'sess_close']

    # for each session, get bar3 close and session close
    bar3_eod = bar3.merge(sess_close, on='session', how='left')
    bar6_eod = bar6.merge(sess_close, on='session', how='left')

    bar3_eod['ret'] = (bar3_eod['sess_close'] - bar3_eod['close']) / bar3_eod['close']
    bar6_eod['ret'] = (bar6_eod['sess_close'] - bar6_eod['close']) / bar6_eod['close']

    bar3_eod = bar3_eod.dropna(subset=['ret'])
    bar6_eod = bar6_eod.dropna(subset=['ret'])

    def wr_stats(returns: pd.Series, direction: str = 'long') -> dict:
        if len(returns) < 30:
            return dict(wr=np.nan, n=len(returns), avg_ret=np.nan)
        if direction == 'long':
            wins = returns > 0
        else:
            returns = -returns
            wins = returns > 0
        return dict(
            wr=float(wins.mean()),
            n=len(returns),
            avg_ret=float(returns.mean()),
        )

    out = dict(symbol=symbol, n_sessions=df['session'].nunique())

    # unconditional drift
    s = wr_stats(bar3_eod['ret'], 'long');  out.update({f'L_b3_{k}': v for k, v in s.items()})
    s = wr_stats(bar6_eod['ret'], 'long');  out.update({f'L_b6_{k}': v for k, v in s.items()})
    s = wr_stats(bar3_eod['ret'], 'short'); out.update({f'S_b3_{k}': v for k, v in s.items()})
    s = wr_stats(bar6_eod['ret'], 'short'); out.update({f'S_b6_{k}': v for k, v in s.items()})

    # conditional: above-VWAP at bar 3
    avwp3 = bar3_eod[bar3_eod['close'] > bar3_eod['vwap']]
    if len(avwp3) >= 30:
        s = wr_stats(avwp3['ret'], 'long')
        out.update({f'L_b3_aboveVWAP_{k}': v for k, v in s.items()})
    else:
        out.update({f'L_b3_aboveVWAP_{k}': np.nan for k in ('wr', 'n', 'avg_ret')})

    bvwp3 = bar3_eod[bar3_eod['close'] < bar3_eod['vwap']]
    if len(bvwp3) >= 30:
        s = wr_stats(bvwp3['ret'], 'short')
        out.update({f'S_b3_belowVWAP_{k}': v for k, v in s.items()})
    else:
        out.update({f'S_b3_belowVWAP_{k}': np.nan for k in ('wr', 'n', 'avg_ret')})

    # conditional: RSI > 60 at bar 3
    rsi_high = bar3_eod[(bar3_eod['close'] > bar3_eod['vwap']) & (bar3_eod['rsi'] > 60)]
    if len(rsi_high) >= 30:
        s = wr_stats(rsi_high['ret'], 'long')
        out.update({f'L_b3_strong_{k}': v for k, v in s.items()})
    else:
        out.update({f'L_b3_strong_{k}': np.nan for k in ('wr', 'n', 'avg_ret')})

    rsi_low = bar3_eod[(bar3_eod['close'] < bar3_eod['vwap']) & (bar3_eod['rsi'] < 40)]
    if len(rsi_low) >= 30:
        s = wr_stats(rsi_low['ret'], 'short')
        out.update({f'S_b3_weak_{k}': v for k, v in s.items()})
    else:
        out.update({f'S_b3_weak_{k}': np.nan for k in ('wr', 'n', 'avg_ret')})

    return out


def main() -> int:
    universe = list_5min_universe(min_rows=32_000)
    print(f'Per-stock drift screen on {len(universe)} stocks')

    fields = None
    rows = []
    t0 = time.time()
    for i, sym in enumerate(universe, 1):
        try:
            r = evaluate_stock(sym)
        except Exception as e:
            print(f'  [{i}] {sym} error {e}')
            continue
        if r is None:
            continue
        rows.append(r)
        if i % 25 == 0:
            elapsed = time.time() - t0
            print(f'  [{i}/{len(universe)}] {sym} elapsed {elapsed:.0f}s')

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f'Wrote {len(df)} rows to {OUT}')

    # quick analysis
    print('\n=== Top 20 LONG drift (b3, unconditional) ===')
    print(df.nlargest(20, 'L_b3_wr')[['symbol', 'L_b3_wr', 'L_b3_n', 'L_b3_avg_ret']].to_string(index=False))
    print('\n=== Top 20 SHORT drift (b3) ===')
    print(df.nlargest(20, 'S_b3_wr')[['symbol', 'S_b3_wr', 'S_b3_n', 'S_b3_avg_ret']].to_string(index=False))
    print('\n=== Top 20 LONG strong-confluence (above VWAP + RSI > 60) ===')
    sub = df[df['L_b3_strong_n'] >= 50]
    print(sub.nlargest(20, 'L_b3_strong_wr')[['symbol', 'L_b3_strong_wr', 'L_b3_strong_n', 'L_b3_strong_avg_ret']].to_string(index=False))
    print('\n=== Top 20 SHORT weak-confluence (below VWAP + RSI < 40) ===')
    sub = df[df['S_b3_weak_n'] >= 50]
    print(sub.nlargest(20, 'S_b3_weak_wr')[['symbol', 'S_b3_weak_wr', 'S_b3_weak_n', 'S_b3_weak_avg_ret']].to_string(index=False))

    return 0


if __name__ == '__main__':
    sys.exit(main())
