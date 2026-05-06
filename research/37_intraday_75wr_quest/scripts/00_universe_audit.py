"""Stage 0 — universe audit.

For every stock in the 5-min universe (rows >= 32_000), produce a one-row
characterisation: avg range, ATR, gap %, open=low / open=high frequency,
trending vs ranging day frequency, etc.

Output: ../results/00_universe_audit.csv (incremental, --skip-done)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import list_5min_universe, load_5min, session_summary  # type: ignore

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results', '00_universe_audit.csv'
)
OUT = os.path.normpath(OUT)


def audit_one(symbol: str) -> dict | None:
    df = load_5min(symbol)
    if df.empty:
        return None
    sess = session_summary(df)
    if len(sess) < 30:
        return None

    # robust range stats
    range_pct = sess['range_pct'].dropna()
    gap_pct = sess['gap_pct'].dropna()
    ret_pct = sess['ret_pct'].dropna()
    close_pos = sess['close_pos'].dropna()
    open_pos = sess['open_pos'].dropna()

    # day "trending" = closes in top 20% or bottom 20% of session range
    trending_up = (close_pos >= 0.8) & (ret_pct > 0)
    trending_dn = (close_pos <= 0.2) & (ret_pct < 0)
    ranging = ((close_pos > 0.2) & (close_pos < 0.8))

    # open=low / open=high (Path E from research/29)
    open_eq_low = sess['open_eq_low'].mean()
    open_eq_high = sess['open_eq_high'].mean()

    # gap behavior
    gap_up_3 = (gap_pct > 0.3).mean()
    gap_dn_3 = (gap_pct < -0.3).mean()

    # avg daily range
    avg_range_pct = range_pct.mean()
    median_range_pct = range_pct.median()

    return dict(
        symbol=symbol,
        n_sessions=int(len(sess)),
        first_session=str(sess.index[0].date()),
        last_session=str(sess.index[-1].date()),
        avg_range_pct=round(avg_range_pct, 3),
        median_range_pct=round(median_range_pct, 3),
        avg_abs_ret_pct=round(ret_pct.abs().mean(), 3),
        trending_up_freq=round(trending_up.mean(), 3),
        trending_dn_freq=round(trending_dn.mean(), 3),
        trending_freq=round((trending_up | trending_dn).mean(), 3),
        ranging_freq=round(ranging.mean(), 3),
        open_eq_low_freq=round(open_eq_low, 3),
        open_eq_high_freq=round(open_eq_high, 3),
        gap_up_3_freq=round(gap_up_3, 3),
        gap_dn_3_freq=round(gap_dn_3, 3),
        avg_volume=int(sess['volume'].mean()),
        avg_close=round(sess['close'].mean(), 2),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-done', action='store_true', default=True)
    ap.add_argument('--max-symbols', type=int, default=None)
    args = ap.parse_args()

    universe = list_5min_universe(min_rows=32_000)
    print(f'Universe: {len(universe)} stocks with >= 32k 5-min rows')

    if args.max_symbols:
        universe = universe[:args.max_symbols]

    fields = [
        'symbol', 'n_sessions', 'first_session', 'last_session',
        'avg_range_pct', 'median_range_pct', 'avg_abs_ret_pct',
        'trending_up_freq', 'trending_dn_freq', 'trending_freq',
        'ranging_freq', 'open_eq_low_freq', 'open_eq_high_freq',
        'gap_up_3_freq', 'gap_dn_3_freq', 'avg_volume', 'avg_close',
    ]

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(prev['symbol'].tolist())
        print(f'Resuming — {len(done)} symbols already audited')
    else:
        with open(OUT, 'w', encoding='utf-8') as f:
            f.write(','.join(fields) + '\n')

    pending = [s for s in universe if s not in done]
    t0 = time.time()
    for i, sym in enumerate(pending, 1):
        try:
            row = audit_one(sym)
        except Exception as e:
            print(f'  [{i}/{len(pending)}] {sym} — ERROR {e}')
            continue
        if row is None:
            continue
        with open(OUT, 'a', encoding='utf-8') as f:
            f.write(','.join(str(row.get(k, '')) for k in fields) + '\n')
        if i % 25 == 0 or i == len(pending):
            elapsed = time.time() - t0
            print(f'  [{i}/{len(pending)}] {sym} — '
                  f'range={row["avg_range_pct"]}% trend={row["trending_freq"]} '
                  f'OL/OH={row["open_eq_low_freq"]}/{row["open_eq_high_freq"]} '
                  f'({elapsed:.0f}s)')
    print(f'Done. Output: {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
