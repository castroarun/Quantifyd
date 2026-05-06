"""Stage A — per-stock per-pattern screen with TIGHT-SL / WIDE-TP exit.

For each (stock, pattern, direction) we measure unconditional WR / PF / n
on a baseline exit of TP=1.0% / SL=0.4% (RR 2.5:1).

Outputs one CSV per pattern in results/01_<pattern>_perstock.csv.

The sweep then identifies "diamonds" — stocks where the pattern produces
WR >= 60% with n >= 20 trades. These cohorts feed Stage B.

ASCII-only console (Windows cp1252 safe). Incremental CSV writes.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

# import research/37 _engine + our pattern_lib
_HERE = os.path.dirname(os.path.abspath(__file__))
_R37 = os.path.normpath(os.path.join(_HERE, '..', '..', '37_intraday_75wr_quest', 'scripts'))
if _R37 not in sys.path:
    sys.path.insert(0, _R37)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _engine import (  # type: ignore
    list_5min_universe, load_5min, enrich,
    simulate_signals, trade_stats, TradeRules,
)
from pattern_lib import PATTERN_FUNCS  # type: ignore


RESULTS_DIR = os.path.normpath(os.path.join(_HERE, '..', 'results'))
LOGS_DIR = os.path.normpath(os.path.join(_HERE, '..', 'logs'))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# baseline params per pattern (intentionally lax — the screen is a filter)
PATTERN_PARAMS = {
    'inside_bar':       dict(min_bar_idx=4, max_bar_idx=60, rsi_min=45, rsi_max=80,
                             require_vwap_align=True),
    'nr_breakout':      dict(nr_len=4, min_bar_idx=4, max_bar_idx=60, rsi_min=45, rsi_max=80,
                             require_vwap_align=True),
    'failed_breakout':  dict(level='or_high', min_bar_idx=4, max_bar_idx=60, rsi_min=25, rsi_max=80,
                             require_wick_back=True),
    'vwap_v':           dict(extension_atr=1.0, reclaim_window=3, min_bar_idx=6, max_bar_idx=60,
                             rsi_min=30, rsi_max=70),
    'compression':      dict(compression_window=6, compression_pct=0.6, min_bar_idx=6, max_bar_idx=60,
                             rsi_min=45, rsi_max=80, require_vwap_align=True),
    'multi_bar':        dict(n_bars=3, min_bar_idx=4, max_bar_idx=60, rsi_min=50, rsi_max=80,
                             require_vwap_align=True, require_higher_lows=True),
    'stop_run':         dict(lookback=12, pierce_pct=0.05, min_bar_idx=6, max_bar_idx=60,
                             rsi_min=30, rsi_max=80),
}

# direction defaults: long-bias / short-bias / both
PATTERN_DIRS = {
    'inside_bar':      ['long', 'short'],
    'nr_breakout':     ['long', 'short'],
    'failed_breakout': ['short', 'long'],   # fade fake breakouts
    'vwap_v':          ['long', 'short'],
    'compression':     ['long', 'short'],
    'multi_bar':       ['long', 'short'],
    'stop_run':        ['short', 'long'],
}

# screen exit (tight SL, wide TP)
SCREEN_TP = 1.0
SCREEN_SL = 0.4
SCREEN_HOLD = 60


def screen_one(symbol: str, df: pd.DataFrame, pattern_name: str, direction: str) -> dict | None:
    func = PATTERN_FUNCS[pattern_name]
    params = dict(PATTERN_PARAMS[pattern_name])
    params['direction'] = direction
    try:
        sig = func(df, **params)
    except Exception as e:
        return dict(symbol=symbol, pattern=pattern_name, direction=direction,
                    n_trades=0, win_rate=0, profit_factor=0, error=str(e)[:80])
    rules = TradeRules(tp_pct=SCREEN_TP, sl_pct=SCREEN_SL, max_hold_bars=SCREEN_HOLD,
                       direction=direction)
    trades = simulate_signals(df, sig, rules)
    stats = trade_stats(trades, sessions=df['session'].nunique())
    return dict(
        symbol=symbol, pattern=pattern_name, direction=direction,
        n_sessions=int(df['session'].nunique()),
        n_trades=stats['n_trades'],
        win_rate=round(stats['win_rate'], 4),
        avg_win=round(stats['avg_win'], 4),
        avg_loss=round(stats['avg_loss'], 4),
        profit_factor=round(stats['profit_factor'], 3),
        max_dd_pct=round(stats['max_dd_pct'], 3),
        sharpe=round(stats['sharpe'], 3),
        total_return_pct=round(stats['total_return_pct'], 2),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--patterns', nargs='*', default=None,
                    help='subset of patterns to run (default: all)')
    ap.add_argument('--max-stocks', type=int, default=320)
    ap.add_argument('--min-rows', type=int, default=32_000,  # research/37 cutoff (308 stocks)
                    help='min 5-min rows to include a stock')
    args = ap.parse_args()

    log_path = os.path.join(LOGS_DIR, '01_pattern_screen.log')
    log_f = open(log_path, 'a', encoding='utf-8', buffering=1)

    def log(msg: str) -> None:
        sys.stdout.write(msg + '\n')
        sys.stdout.flush()
        log_f.write(msg + '\n')
        log_f.flush()

    patterns = args.patterns if args.patterns else list(PATTERN_FUNCS.keys())
    log('=' * 70)
    log(f'Stage A: per-stock pattern screen | {time.strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'Patterns: {patterns}')
    log(f'Exit (screen): TP={SCREEN_TP}% SL={SCREEN_SL}% (RR {SCREEN_TP/SCREEN_SL:.2f}:1)')
    log('=' * 70)

    log('Loading 5-min universe ...')
    universe = list_5min_universe(min_rows=args.min_rows)
    universe = [s for s in universe if s not in ('NIFTY50', 'BANKNIFTY')]
    universe = universe[: args.max_stocks]
    log(f'  {len(universe)} stocks')

    fields = ['symbol', 'pattern', 'direction', 'n_sessions', 'n_trades',
              'win_rate', 'avg_win', 'avg_loss', 'profit_factor', 'max_dd_pct',
              'sharpe', 'total_return_pct']

    # one CSV per pattern
    csv_paths = {}
    done_sets = {}
    for p in patterns:
        path = os.path.join(RESULTS_DIR, f'01_{p}_perstock.csv')
        csv_paths[p] = path
        done = set()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    done.add((r['symbol'], r['direction']))
            log(f'  resume {p}: {len(done)} (symbol, dir) pairs already done')
        done_sets[p] = done
        if not os.path.exists(path):
            with open(path, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    t0 = time.time()
    last_log = t0

    for i, sym in enumerate(universe, 1):
        # check if all (pattern, direction) for this stock already done -> skip load
        all_done = True
        for p in patterns:
            for d in PATTERN_DIRS[p]:
                if (sym, d) not in done_sets[p]:
                    all_done = False
                    break
            if not all_done:
                break
        if all_done:
            continue

        try:
            df = load_5min(sym)
        except Exception as e:
            log(f'  [{i}/{len(universe)}] {sym} load err {e}')
            continue
        if df.empty or len(df) < 1500:
            continue
        try:
            df = enrich(df, or_minutes=15)
        except Exception as e:
            log(f'  [{i}/{len(universe)}] {sym} enrich err {e}')
            continue

        for p in patterns:
            for d in PATTERN_DIRS[p]:
                if (sym, d) in done_sets[p]:
                    continue
                row = screen_one(sym, df, p, d)
                if row is None:
                    continue
                # write incrementally
                row_to_write = {k: row.get(k, '') for k in fields}
                with open(csv_paths[p], 'a', newline='', encoding='utf-8') as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row_to_write)
                done_sets[p].add((sym, d))

        if i % 10 == 0 or (time.time() - last_log) > 60:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1)
            eta = (len(universe) - i) / rate
            log(f'  [{i}/{len(universe)}] {sym} elapsed {elapsed:.0f}s rate {rate:.2f}/s ETA {eta:.0f}s')
            last_log = time.time()

    log(f'\nDone Stage A in {time.time() - t0:.0f}s')
    for p in patterns:
        path = csv_paths[p]
        if not os.path.exists(path):
            continue
        try:
            df_p = pd.read_csv(path)
            qual = df_p[(df_p['n_trades'] >= 20) & (df_p['win_rate'] >= 0.60)]
            log(f'  {p}: {len(df_p)} rows | {len(qual)} (sym,dir) pass screen (n>=20, WR>=60%)')
        except Exception:
            pass

    log_f.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
