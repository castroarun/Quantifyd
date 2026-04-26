"""Universe builder for the small/micro-cap intraday breakout system.

Implements steps 1-3 of the spec in `data/future_plans.json` plan id
`smallcap-intraday-orb`:

  Step 1 -- candidates: stocks with >=500 bars of 5-min data since 2024-03-18,
           excluding ORB universe and index tickers. Per the spec the base
           should be Nifty Smallcap 250 + Microcap 250, but local DB only
           has 5-min bars for ~81 large/mid-caps. See SMALLCAP-STATUS.md.

  Step 2 -- turnover filter: avg daily turnover (close*volume) over last 30
           trading days >= Rs 50 Cr.

  Step 3 -- circuit-history filter: count of daily |move| >= 5% over last 60
           trading days < 3.

Outputs:
  results/universe_selection.csv  -- full diagnostics per candidate
  results/universe.csv            -- final filtered symbols, one per line

Usage:
  python build_universe.py                 # default run
  python build_universe.py --no-mid-proxy  # strict (excludes Nifty 500)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

logging.disable(logging.WARNING)

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
N500_CSV = ROOT / 'data' / 'nifty500_list.csv'

OUT = Path(__file__).resolve().parents[1] / 'results'
LOGS = Path(__file__).resolve().parents[1] / 'logs'
OUT.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

START_DATE = '2024-03-18'
END_DATE   = '2026-03-12'

MIN_5MIN_BARS         = 500
MIN_AVG_TURNOVER_INR  = 50_00_00_000   # Rs 50 Cr
TURNOVER_LOOKBACK_DAYS = 30
CIRCUIT_PCT_THRESHOLD  = 0.05          # 5% daily move proxy for circuit hit
CIRCUIT_LOOKBACK_DAYS  = 60
MAX_CIRCUIT_DAYS       = 3             # exclude if >= this many circuit days

# ORB universe (cannot be reused in small-cap system)
ORB_UNIVERSE = {
    'ADANIENT', 'TATASTEEL', 'BEL', 'VEDL', 'BPCL', 'M&M', 'BAJFINANCE',
    'TRENT', 'HAL', 'IRCTC', 'GRASIM', 'GODREJPROP', 'RELIANCE',
    'AXISBANK', 'APOLLOHOSP',
}
INDEX_TICKERS = {'NIFTY50', 'BANKNIFTY', 'NIFTY', 'NIFTY100', 'NIFTYIT', 'FINNIFTY'}


@dataclass
class CandidateRow:
    symbol: str
    bars_5min: int
    avg_daily_turnover_inr: float
    avg_daily_turnover_cr: float
    daily_bars_window: int
    circuit_days: int
    in_nifty500: bool
    is_mid_proxy: bool
    included: bool
    exclusion_reason: str


def load_nifty500() -> set[str]:
    syms: set[str] = set()
    with N500_CSV.open() as f:
        for row in csv.DictReader(f):
            syms.add(row['Symbol'].strip())
    return syms


def fetch_5min_symbol_bars(conn) -> dict[str, int]:
    """Return {symbol: bar_count} for all symbols with any 5-min data
    in the (START_DATE, END_DATE] window."""
    rows = conn.execute(
        """SELECT symbol, COUNT(*) as bars
           FROM market_data_unified
           WHERE timeframe='5minute' AND date>=? AND date<=?
           GROUP BY symbol""",
        (START_DATE, END_DATE + ' 23:59:59'),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def fetch_avg_turnover(conn, symbol: str) -> tuple[float, int]:
    """Avg daily turnover (Rs) over last TURNOVER_LOOKBACK_DAYS daily bars
    inside the test window.

    Returns (avg_turnover_inr, bars_used)."""
    rows = conn.execute(
        """SELECT close, volume FROM market_data_unified
           WHERE symbol=? AND timeframe='day' AND date>=? AND date<=?
           ORDER BY date DESC LIMIT ?""",
        (symbol, START_DATE, END_DATE + ' 23:59:59', TURNOVER_LOOKBACK_DAYS),
    ).fetchall()
    if not rows:
        return 0.0, 0
    turnovers = [float(c) * float(v) for c, v in rows if c is not None and v is not None]
    if not turnovers:
        return 0.0, 0
    return sum(turnovers) / len(turnovers), len(turnovers)


def fetch_circuit_days(conn, symbol: str) -> int:
    """Count of daily bars in the last CIRCUIT_LOOKBACK_DAYS where
    |close - prev_close| / prev_close >= CIRCUIT_PCT_THRESHOLD."""
    rows = conn.execute(
        """SELECT date, close FROM market_data_unified
           WHERE symbol=? AND timeframe='day' AND date>=? AND date<=?
           ORDER BY date DESC LIMIT ?""",
        (symbol, START_DATE, END_DATE + ' 23:59:59', CIRCUIT_LOOKBACK_DAYS + 1),
    ).fetchall()
    if len(rows) < 2:
        return 0
    closes = [float(r[1]) for r in rows[::-1]]   # chronological
    count = 0
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0:
            continue
        move = abs(closes[i] - closes[i - 1]) / closes[i - 1]
        if move >= CIRCUIT_PCT_THRESHOLD:
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--no-mid-proxy', action='store_true',
        help='Strict mode: drop Nifty-500 names (yields ~4 candidates only). '
             'Default keeps non-ORB Nifty-500 5-min names as a mid-cap proxy '
             'because the local DB has no true small-cap 5-min data.',
    )
    args = ap.parse_args()

    print(f'Universe builder -- small/micro-cap intraday system')
    print(f'  DB: {DB}')
    print(f'  Window: {START_DATE} to {END_DATE}')
    print(f'  Mid-cap proxy mode: {not args.no_mid_proxy}')
    print()

    n500 = load_nifty500()
    conn = sqlite3.connect(str(DB))

    # Step 1 -- candidates
    bar_counts = fetch_5min_symbol_bars(conn)
    print(f'5-min symbols in DB (in window): {len(bar_counts)}')

    candidates: list[str] = []
    excluded_step1: list[tuple[str, str]] = []
    for sym, bars in bar_counts.items():
        if sym in INDEX_TICKERS:
            excluded_step1.append((sym, 'index_ticker'))
            continue
        if sym in ORB_UNIVERSE:
            excluded_step1.append((sym, 'in_orb_universe'))
            continue
        if bars < MIN_5MIN_BARS:
            excluded_step1.append((sym, f'bars<{MIN_5MIN_BARS}'))
            continue
        in_n500 = sym in n500
        if args.no_mid_proxy and in_n500:
            excluded_step1.append((sym, 'in_nifty500_strict_mode'))
            continue
        candidates.append(sym)

    print(f'After Step 1 (bars + exclusions): {len(candidates)} candidates')
    print()

    # Steps 2 + 3 -- per-candidate diagnostics
    rows: list[CandidateRow] = []
    print(f'Computing turnover + circuit history for {len(candidates)} candidates...')
    for i, sym in enumerate(sorted(candidates), 1):
        avg_turn, daily_bars = fetch_avg_turnover(conn, sym)
        circuits = fetch_circuit_days(conn, sym)
        in_n500 = sym in n500
        is_mid_proxy = in_n500 and not args.no_mid_proxy

        passes_turn = avg_turn >= MIN_AVG_TURNOVER_INR
        passes_circ = circuits < MAX_CIRCUIT_DAYS

        if not passes_turn:
            reason = f'turnover<{MIN_AVG_TURNOVER_INR/1e7:.0f}Cr'
            included = False
        elif not passes_circ:
            reason = f'circuit_days>={MAX_CIRCUIT_DAYS}'
            included = False
        else:
            reason = ''
            included = True

        rows.append(CandidateRow(
            symbol=sym, bars_5min=bar_counts[sym],
            avg_daily_turnover_inr=round(avg_turn, 0),
            avg_daily_turnover_cr=round(avg_turn / 1e7, 2),
            daily_bars_window=daily_bars,
            circuit_days=circuits,
            in_nifty500=in_n500,
            is_mid_proxy=is_mid_proxy,
            included=included,
            exclusion_reason=reason,
        ))
        if i % 20 == 0 or i == len(candidates):
            print(f'  {i}/{len(candidates)} done', flush=True)
    conn.close()

    # Add Step-1 excluded rows for completeness
    for sym, reason in excluded_step1:
        rows.append(CandidateRow(
            symbol=sym, bars_5min=bar_counts.get(sym, 0),
            avg_daily_turnover_inr=0.0, avg_daily_turnover_cr=0.0,
            daily_bars_window=0, circuit_days=0,
            in_nifty500=(sym in n500), is_mid_proxy=False,
            included=False, exclusion_reason=reason,
        ))

    # Sort: included first, then by turnover desc
    rows.sort(key=lambda r: (not r.included, -r.avg_daily_turnover_cr, r.symbol))

    # Write diagnostic CSV
    diag_path = OUT / 'universe_selection.csv'
    with diag_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    # Write final universe (just symbols, one per line)
    final = [r.symbol for r in rows if r.included]
    final.sort()
    uni_path = OUT / 'universe.csv'
    with uni_path.open('w', newline='') as f:
        f.write('symbol\n')
        for s in final:
            f.write(f'{s}\n')

    # Summary
    n_after_step1 = len(candidates)
    n_after_turn = sum(1 for r in rows if r.exclusion_reason in
                       ('', f'circuit_days>={MAX_CIRCUIT_DAYS}'))
    n_after_circ = len(final)

    print()
    print('=' * 72)
    print(f'Universe reduced from {n_after_step1} candidates ->')
    print(f'  {n_after_turn} after turnover filter (>= Rs {MIN_AVG_TURNOVER_INR/1e7:.0f} Cr) ->')
    print(f'  {n_after_circ} final after circuit filter (< {MAX_CIRCUIT_DAYS} days >=5% in last {CIRCUIT_LOOKBACK_DAYS}d)')
    print('=' * 72)
    print(f'Diagnostics: {diag_path}')
    print(f'Final list:  {uni_path}  ({n_after_circ} symbols)')

    if n_after_circ < 20:
        print()
        print('!! WARNING: final universe is very small. The local DB likely')
        print('   does not have 5-min bars for true small/microcap stocks.')
        print('   See SMALLCAP-STATUS.md for the spec deviation note.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
