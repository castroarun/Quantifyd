"""Small/Micro-Cap Daily Universe Builder.

Filters the daily-bar database down to a quality-filtered small/micro-cap
universe for the Daily ATH/52w Breakout backtest.

Pipeline:
  Step 1 — Base: all daily-bar symbols with >=1500 bars since 2018-01-01.
  Step 2 — Exclude all Nifty 500 names (focus on outside-the-index exposure).
  Step 3 — Turnover band: avg trailing-60-day daily turnover (close*volume)
            in [Rs 5 Cr, Rs 100 Cr]. Liquidity floor + small/micro cap cap.
  Step 4 — Circuit / freeze filter: <4 days in last 60 trading days where
            |today_close - prev_close| / prev_close >= 10%.
  Step 5 — Volatility floor: trailing 30-day avg of ATR(14) / close >= 1%.
  Step 6 — EQ-series filter: SKIPPED (no series metadata in DB) — flagged.

Outputs:
  results/daily_universe_selection.csv  — full diagnostics per candidate
  results/daily_universe.csv            — final filtered list (one symbol/line)
"""
from __future__ import annotations

import csv
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
NIFTY500 = ROOT / 'data' / 'nifty500_list.csv'
OUT = Path(__file__).resolve().parents[1] / 'results'
LOGS = Path(__file__).resolve().parents[1] / 'logs'
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
HEARTBEAT = LOGS / 'universe_heartbeat.txt'

START_DATE = '2018-01-01'
MIN_BARS = 1500

# Filter constants
TURNOVER_MIN_RS = 5_00_00_000        # Rs 5 Cr (50 million)
TURNOVER_MAX_RS = 100_00_00_000      # Rs 100 Cr (1 billion)
TURNOVER_LOOKBACK = 60               # trailing 60 trading days
CIRCUIT_THRESHOLD_PCT = 0.10         # 10% daily move
CIRCUIT_LOOKBACK = 60
CIRCUIT_MAX_HITS = 4                 # exclude if >=4 such days
ATR_PERIOD = 14
ATR_LOOKBACK = 30                    # avg ATR%/close over trailing 30 days
ATR_PCT_MIN = 0.01                   # 1% minimum

OUT_DETAIL = OUT / 'daily_universe_selection.csv'
OUT_FINAL = OUT / 'daily_universe.csv'


def heartbeat(msg: str) -> None:
    with HEARTBEAT.open('w') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} | {msg}\n')


def load_nifty500() -> set[str]:
    syms = set()
    with NIFTY500.open() as f:
        for r in csv.DictReader(f):
            syms.add(r['Symbol'].strip().upper())
    return syms


def load_base_universe() -> list[str]:
    """All daily-bar symbols with >=MIN_BARS since START_DATE."""
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        """SELECT symbol, COUNT(*) AS c FROM market_data_unified
            WHERE timeframe='day' AND date>=?
            GROUP BY symbol HAVING c>=?
            ORDER BY symbol""",
        (START_DATE, MIN_BARS),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def load_bars(conn: sqlite3.Connection, sym: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        """SELECT date, open, high, low, close, volume FROM market_data_unified
            WHERE symbol=? AND timeframe='day' AND date>=?
            ORDER BY date""",
        conn, params=(sym, START_DATE),
    )
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def compute_diagnostics(df: pd.DataFrame) -> dict:
    """Compute trailing 60-day turnover, circuit count, ATR%."""
    bars = len(df)
    if bars < max(TURNOVER_LOOKBACK, CIRCUIT_LOOKBACK, ATR_LOOKBACK + ATR_PERIOD):
        return {
            'bars_count': bars,
            'avg_turnover_cr': 0.0,
            'circuit_count': -1,
            'atr_pct': 0.0,
            'reason_short_history': True,
        }

    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    # Turnover (last TURNOVER_LOOKBACK days)
    turnover = (close * volume).iloc[-TURNOVER_LOOKBACK:]
    avg_turnover_rs = float(turnover.mean()) if not turnover.empty else 0.0
    avg_turnover_cr = avg_turnover_rs / 1e7  # in Rs Cr

    # Circuit / freeze count (last CIRCUIT_LOOKBACK days)
    daily_move = (close / close.shift(1) - 1.0).abs()
    circuit_window = daily_move.iloc[-CIRCUIT_LOOKBACK:].dropna()
    circuit_count = int((circuit_window >= CIRCUIT_THRESHOLD_PCT).sum())

    # ATR(14) (Wilder)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()
    atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan).dropna()
    atr_pct_window = atr_pct.iloc[-ATR_LOOKBACK:]
    avg_atr_pct = float(atr_pct_window.mean()) if not atr_pct_window.empty else 0.0

    return {
        'bars_count': bars,
        'avg_turnover_cr': round(avg_turnover_rs / 1e7, 4),
        'circuit_count': circuit_count,
        'atr_pct': round(avg_atr_pct, 5),
        'reason_short_history': False,
    }


def classify(d: dict) -> tuple[bool, str]:
    """Apply filters to diagnostics; return (included, exclusion_reason)."""
    if d.get('reason_short_history'):
        return False, 'SHORT_HISTORY'
    avg_t = d['avg_turnover_cr']
    if avg_t * 1e7 < TURNOVER_MIN_RS:
        return False, 'TURNOVER_TOO_LOW'
    if avg_t * 1e7 > TURNOVER_MAX_RS:
        return False, 'TURNOVER_TOO_HIGH'
    if d['circuit_count'] >= CIRCUIT_MAX_HITS:
        return False, 'CIRCUIT_PRONE'
    if d['atr_pct'] < ATR_PCT_MIN:
        return False, 'LOW_VOL'
    return True, ''


def main():
    t0 = time.time()
    heartbeat('start: loading nifty500 + base universe')
    n500 = load_nifty500()
    base = load_base_universe()
    print(f'Base daily-bar universe (>={MIN_BARS} bars since {START_DATE}): {len(base)}', flush=True)

    after_n500 = [s for s in base if s.upper() not in n500]
    print(f'After excluding Nifty 500: {len(after_n500)}', flush=True)
    print(f'Now scanning {len(after_n500)} symbols for turnover / circuit / volatility...', flush=True)

    # Counters for stage funnel (turnover band, circuit, volatility)
    fail_turnover_low = 0
    fail_turnover_high = 0
    fail_circuit = 0
    fail_lowvol = 0
    fail_short = 0
    included = 0

    rows = []
    conn = sqlite3.connect(DB)
    for i, sym in enumerate(after_n500, start=1):
        if i % 25 == 0 or i == len(after_n500):
            elapsed = time.time() - t0
            heartbeat(f'{i}/{len(after_n500)} processed | included={included} | elapsed={elapsed:.0f}s')
            print(f'  [{i}/{len(after_n500)}] in={included} elapsed={elapsed:.0f}s', flush=True)

        df = load_bars(conn, sym)
        if df.empty:
            continue
        d = compute_diagnostics(df)
        ok, reason = classify(d)
        if ok:
            included += 1
        else:
            if reason == 'SHORT_HISTORY':
                fail_short += 1
            elif reason == 'TURNOVER_TOO_LOW':
                fail_turnover_low += 1
            elif reason == 'TURNOVER_TOO_HIGH':
                fail_turnover_high += 1
            elif reason == 'CIRCUIT_PRONE':
                fail_circuit += 1
            elif reason == 'LOW_VOL':
                fail_lowvol += 1

        rows.append({
            'symbol': sym,
            'bars_count': d['bars_count'],
            'avg_turnover_cr': d['avg_turnover_cr'],
            'circuit_count': d['circuit_count'],
            'atr_pct': d['atr_pct'],
            'included': 'Y' if ok else 'N',
            'exclusion_reason': reason,
        })
    conn.close()

    # Write detail CSV (all candidates with diagnostics)
    fields = ['symbol', 'bars_count', 'avg_turnover_cr', 'circuit_count',
              'atr_pct', 'included', 'exclusion_reason']
    with OUT_DETAIL.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write final list (one symbol per line)
    final = [r['symbol'] for r in rows if r['included'] == 'Y']
    final.sort()
    with OUT_FINAL.open('w', newline='') as f:
        for s in final:
            f.write(s + '\n')

    after_turnover = len(after_n500) - fail_turnover_low - fail_turnover_high - fail_short
    after_circuit = after_turnover - fail_circuit
    after_vol = after_circuit - fail_lowvol

    print()
    print('=== UNIVERSE FUNNEL ===', flush=True)
    print(f'  Base daily-bar (>={MIN_BARS} bars):       {len(base)}', flush=True)
    print(f'  After Nifty500 exclusion:               {len(after_n500)}', flush=True)
    print(f'  After turnover band [5-100 Cr]:         {after_turnover}    '
          f'(low={fail_turnover_low}, high={fail_turnover_high}, short_hist={fail_short})', flush=True)
    print(f'  After circuit filter (<{CIRCUIT_MAX_HITS} hits/60d):       {after_circuit}    '
          f'(circuit_prone={fail_circuit})', flush=True)
    print(f'  After volatility floor (>=1% ATR%):     {after_vol}    '
          f'(low_vol={fail_lowvol})', flush=True)
    print(f'  FINAL:                                  {len(final)}', flush=True)
    print()
    print(f'Detail: {OUT_DETAIL}', flush=True)
    print(f'Final list: {OUT_FINAL}', flush=True)
    print(f'Total runtime: {time.time()-t0:.1f}s', flush=True)
    heartbeat(f'DONE: final={len(final)}')


if __name__ == '__main__':
    sys.exit(main())
