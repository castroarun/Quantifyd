"""Stage 1 — reverse-engineer big-move days.

For every stock × session, mine the days with intraday range >= 2x ATR
(or top 20% by absolute return). Then for each "winner" day compute the
morning signature:
  - Open vs prev_close (gap_pct)
  - First-candle pattern (open=low / open=high / inside)
  - VWAP relationship at bar 3, 6, 9 (close > vwap?)
  - RSI(14) at bar 3, 6, 9
  - EMA9 vs EMA21 at bar 6
  - Volume bar 0..2 vs 20-day avg
  - Position of intraday high/low (bar number)

Output: results/01_winner_signatures.csv — one row per (stock, session)
that was a "winner" + summary CSV with frequency tables.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import list_5min_universe, load_5min, enrich, session_summary  # type: ignore

OUT_RAW = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results', '01_winner_days.csv'
))
OUT_SUMMARY = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results', '01_winner_signatures.csv'
))


def winner_threshold(sess: pd.DataFrame) -> float:
    """Top 20% of |ret_pct| sessions are winners."""
    return float(sess['ret_pct'].abs().quantile(0.80))


def signature_for_session(df_sess: pd.DataFrame, sess_row: pd.Series) -> dict:
    """Compute morning signature for a single session."""
    if len(df_sess) < 10:
        return {}
    bar = lambda i: df_sess.iloc[i] if i < len(df_sess) else df_sess.iloc[-1]
    b0, b1, b2 = bar(0), bar(1), bar(2)
    b3, b6, b9 = bar(3), bar(6), bar(9)

    # bar position of session high / low
    hi_idx = int(np.argmax(df_sess['high'].values))
    lo_idx = int(np.argmin(df_sess['low'].values))

    # vol vs avg (use last 20 sessions of same stock pre-loaded? expensive — skip for now)
    vol_first3 = df_sess.iloc[:3]['volume'].sum()

    # vwap position at key bars
    vwap_at = lambda b: 1 if b['close'] > b['vwap'] else (-1 if b['close'] < b['vwap'] else 0)

    return dict(
        gap_pct=round(sess_row.get('gap_pct', np.nan), 3),
        first_open_eq_low=int(abs(b0['open'] - b0['low']) / b0['open'] < 0.0005),
        first_open_eq_high=int(abs(b0['open'] - b0['high']) / b0['open'] < 0.0005),
        first_bullish=int(b0['close'] > b0['open']),
        first_bearish=int(b0['close'] < b0['open']),
        bar3_above_vwap=vwap_at(b3),
        bar6_above_vwap=vwap_at(b6),
        bar9_above_vwap=vwap_at(b9),
        rsi_at_bar3=round(b3['rsi'], 1),
        rsi_at_bar6=round(b6['rsi'], 1),
        rsi_at_bar9=round(b9['rsi'], 1),
        ema9_above_ema21_bar6=int(b6['ema9'] > b6['ema21']),
        vol_first3=int(vol_first3),
        hi_bar=hi_idx,
        lo_bar=lo_idx,
        bars=len(df_sess),
    )


def process_stock(symbol: str, top_pct: float = 0.20) -> pd.DataFrame:
    df = load_5min(symbol)
    if df.empty or len(df) < 1000:
        return pd.DataFrame()
    df = enrich(df, or_minutes=15)
    sess = session_summary(df)
    if len(sess) < 30:
        return pd.DataFrame()

    # threshold = top top_pct of |ret_pct|
    thr = float(sess['ret_pct'].abs().quantile(1 - top_pct))
    winner_mask = sess['ret_pct'].abs() >= thr
    winners_idx = sess.index[winner_mask]

    rows = []
    for sess_date in winners_idx:
        sess_row = sess.loc[sess_date]
        df_sess = df[df['session'] == sess_date]
        sig = signature_for_session(df_sess, sess_row)
        if not sig:
            continue
        sig['symbol'] = symbol
        sig['session'] = str(sess_date.date())
        sig['ret_pct'] = round(sess_row['ret_pct'], 3)
        sig['range_pct'] = round(sess_row['range_pct'], 3)
        sig['direction'] = 'up' if sess_row['ret_pct'] > 0 else 'dn'
        rows.append(sig)
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Frequency table of signature columns split by up vs dn winners."""
    if rows.empty:
        return pd.DataFrame()
    summary = []
    for direction in ['up', 'dn']:
        sub = rows[rows['direction'] == direction]
        if sub.empty:
            continue
        summary.append(dict(
            direction=direction,
            n=len(sub),
            avg_gap_pct=round(sub['gap_pct'].mean(), 3),
            pct_open_eq_low=round(sub['first_open_eq_low'].mean(), 3),
            pct_open_eq_high=round(sub['first_open_eq_high'].mean(), 3),
            pct_first_bullish=round(sub['first_bullish'].mean(), 3),
            pct_first_bearish=round(sub['first_bearish'].mean(), 3),
            pct_bar3_above_vwap=round((sub['bar3_above_vwap'] == 1).mean(), 3),
            pct_bar6_above_vwap=round((sub['bar6_above_vwap'] == 1).mean(), 3),
            pct_bar9_above_vwap=round((sub['bar9_above_vwap'] == 1).mean(), 3),
            avg_rsi_bar3=round(sub['rsi_at_bar3'].mean(), 1),
            avg_rsi_bar6=round(sub['rsi_at_bar6'].mean(), 1),
            avg_rsi_bar9=round(sub['rsi_at_bar9'].mean(), 1),
            pct_ema9_above_ema21_bar6=round(sub['ema9_above_ema21_bar6'].mean(), 3),
            avg_hi_bar=round(sub['hi_bar'].mean(), 1),
            avg_lo_bar=round(sub['lo_bar'].mean(), 1),
        ))
    return pd.DataFrame(summary)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-pct', type=float, default=0.20, help='top N% sessions are "winners"')
    ap.add_argument('--max-stocks', type=int, default=None)
    args = ap.parse_args()

    universe = list_5min_universe(min_rows=32_000)
    if args.max_stocks:
        universe = universe[:args.max_stocks]
    print(f'Reverse-engineering {len(universe)} stocks; top {args.top_pct*100:.0f}% sessions = winners')

    if os.path.exists(OUT_RAW):
        os.remove(OUT_RAW)

    all_rows = []
    t0 = time.time()
    for i, sym in enumerate(universe, 1):
        try:
            rows = process_stock(sym, top_pct=args.top_pct)
        except Exception as e:
            print(f'  [{i}/{len(universe)}] {sym} ERROR {e}')
            continue
        if rows.empty:
            continue
        rows.to_csv(OUT_RAW, mode='a', header=not os.path.exists(OUT_RAW) or i == 1 and len(all_rows) == 0,
                    index=False)
        all_rows.append(rows)
        if i % 20 == 0 or i == len(universe):
            elapsed = time.time() - t0
            print(f'  [{i}/{len(universe)}] {sym} ({elapsed:.0f}s)')

    if not all_rows:
        print('No winner days found.')
        return 1

    combined = pd.concat(all_rows, ignore_index=True)
    summary = summarize(combined)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f'Done. Raw rows: {OUT_RAW}, summary: {OUT_SUMMARY}')
    print(summary.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
