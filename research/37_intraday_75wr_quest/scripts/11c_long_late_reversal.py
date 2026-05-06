"""Stage 11c — Long LATE-DAY REVERSAL discovery.

Hypothesis: stocks that sell off heavily in the morning often see late-day
buying (NSE has a documented "afternoon mean reversion" effect on certain
names — a counterpart to the morning-drift pattern that the winning short
system exploits).

The signal (LONG):
- By bar 24 (11:15 IST) or bar 36 (12:15 IST), stock is DOWN at least N%
  from today's open (N in {-1.0%, -1.5%, -2.0%}).
- AND stock now showing reversal: RSI(14) was below 30 within last 6 bars
  AND now is rising (RSI > 35); current bar bullish; 3-bar low broken to
  upside (close > rolling 3-bar high of prior bars).
- Entry at NEXT bar's open after the trigger bar (between 11:15 and 14:00,
  or last hour 13:15-14:45 — variant-controlled).
- Per-stock screen (top 100 stocks by row count) with simplest variant
  to find which stocks have this pattern with >= 55% WR.
- Sweep then runs full param grid on the surviving cohort, with optional
  NIFTY regime filter.

Output:
  results/11c_long_late_reversal_perstock.csv  (per-stock screen)
  results/11c_long_reversal_diamonds.txt       (cohort)
  results/11c_long_late_reversal_ranking.csv   (full sweep ranking)
  logs/11c_long_late_reversal.log              (run log)

NOTE: ASCII-only console output (no unicode arrows) — Windows cp1252 safe.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import (  # type: ignore
    list_5min_universe, load_5min, enrich,
    simulate_signals, trade_stats, TradeRules,
)
from _strategies import _one_per_session, _no_late_entries  # type: ignore

import importlib
mod_8 = importlib.import_module('08_diamond_short_with_nifty')
build_nifty_regime = mod_8.build_nifty_regime


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
LOGS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'logs',
))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

PERSTOCK_CSV = os.path.join(RESULTS_DIR, '11c_long_late_reversal_perstock.csv')
DIAMONDS_TXT = os.path.join(RESULTS_DIR, '11c_long_reversal_diamonds.txt')
RANK_CSV = os.path.join(RESULTS_DIR, '11c_long_late_reversal_ranking.csv')
LOG_PATH = os.path.join(LOGS_DIR, '11c_long_late_reversal.log')


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
class TeeLog:
    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8', buffering=1)

    def write(self, msg):
        sys.__stdout__.write(msg)
        sys.__stdout__.flush()
        self.f.write(msg)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


# ---------------------------------------------------------------------------
# late-day reversal signal
# ---------------------------------------------------------------------------

def late_reversal_signal(df: pd.DataFrame,
                         drop_pct: float = -1.5,
                         entry_window_start: int = 24,   # bar idx 24 = 11:15
                         entry_window_end: int = 48,     # bar idx 48 = 13:15
                         rsi_oversold: float = 30,
                         rsi_lift: float = 35,
                         rsi_lookback: int = 6,
                         require_3bar_break: bool = True,
                         require_bullish_bar: bool = True,
                         ) -> pd.Series:
    """Late-day reversal LONG signal.

    Triggers at bars in [entry_window_start, entry_window_end] when:
      1. Cumulative drop from day_open <= drop_pct (e.g. -1.5%)
      2. RSI(14) was below `rsi_oversold` within last `rsi_lookback` bars
      3. RSI(14) now > `rsi_lift` (lift from oversold)
      4. Current bar bullish (close > open)         [if require_bullish_bar]
      5. Current close > 3-bar prior high (break)   [if require_3bar_break]

    Signal collapsed to first True per session.
    """
    bar_idx = df['bar_idx'].values
    close = df['close'].values
    open_ = df['open'].values
    day_open = df['day_open'].values
    rsi = df['rsi'].values

    # 1. drop from day_open
    drop = (close - day_open) / day_open * 100.0  # in %
    drop_ok = drop <= drop_pct

    # 2. RSI was oversold in last `rsi_lookback` bars
    rsi_s = pd.Series(rsi, index=df.index)
    rsi_was_oversold = (rsi_s < rsi_oversold).rolling(rsi_lookback, min_periods=1).max().astype(bool).values

    # 3. RSI now above lift threshold (out of oversold)
    rsi_now_lift = rsi > rsi_lift

    # 4. bullish bar
    bullish = close > open_ if require_bullish_bar else np.ones(len(df), dtype=bool)

    # 5. 3-bar break — close > rolling 3-bar high of *prior* bars
    if require_3bar_break:
        prior_high3 = pd.Series(df['high'].values, index=df.index).shift(1).rolling(3, min_periods=1).max().values
        break_ok = close > prior_high3
    else:
        break_ok = np.ones(len(df), dtype=bool)

    # 6. window
    in_window = (bar_idx >= entry_window_start) & (bar_idx <= entry_window_end)

    cond = drop_ok & rsi_was_oversold & rsi_now_lift & bullish & break_ok & in_window
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig


# ---------------------------------------------------------------------------
# per-stock screen (simplest variant)
# ---------------------------------------------------------------------------

def perstock_screen(symbols: list[str], drop_pct: float = -1.5,
                    entry_window=(24, 48), tp=1.0, sl=1.0, hold=48,
                    log=print) -> pd.DataFrame:
    """Run the simplest reversal variant on each stock; record WR + n.

    Writes incrementally to PERSTOCK_CSV.
    """
    fields = [
        'symbol', 'n_sessions', 'n_trades', 'win_rate',
        'avg_win', 'avg_loss', 'profit_factor', 'max_dd_pct',
        'sharpe', 'total_return_pct',
    ]

    done = set()
    if os.path.exists(PERSTOCK_CSV):
        with open(PERSTOCK_CSV, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                done.add(r['symbol'])
        log(f'  resume: {len(done)} symbols already screened')

    mode = 'a' if os.path.exists(PERSTOCK_CSV) else 'w'
    with open(PERSTOCK_CSV, mode, newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == 'w':
            w.writeheader()
        t0 = time.time()
        rows = []
        for i, sym in enumerate(symbols, 1):
            if sym in done:
                continue
            try:
                df = load_5min(sym)
            except Exception as e:
                log(f'  [{i}/{len(symbols)}] {sym} load err {e}')
                continue
            if df.empty or len(df) < 1500:
                continue
            try:
                df = enrich(df, or_minutes=15)
            except Exception as e:
                log(f'  [{i}/{len(symbols)}] {sym} enrich err {e}')
                continue

            sig = late_reversal_signal(
                df, drop_pct=drop_pct,
                entry_window_start=entry_window[0],
                entry_window_end=entry_window[1],
            )
            rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold, direction='long')
            trades = simulate_signals(df, sig, rules)
            stats = trade_stats(trades, sessions=df['session'].nunique())
            row = dict(
                symbol=sym, n_sessions=int(df['session'].nunique()),
                n_trades=stats['n_trades'],
                win_rate=round(stats['win_rate'], 4),
                avg_win=round(stats['avg_win'], 4),
                avg_loss=round(stats['avg_loss'], 4),
                profit_factor=round(stats['profit_factor'], 3),
                max_dd_pct=round(stats['max_dd_pct'], 3),
                sharpe=round(stats['sharpe'], 3),
                total_return_pct=round(stats['total_return_pct'], 2),
            )
            w.writerow(row)
            f.flush()
            rows.append(row)
            if i % 25 == 0:
                elapsed = time.time() - t0
                log(f'  [{i}/{len(symbols)}] {sym} elapsed {elapsed:.0f}s')

    return pd.read_csv(PERSTOCK_CSV)


# ---------------------------------------------------------------------------
# NIFTY filter (mirror of stage 8/10, but for LONG reversal context)
# ---------------------------------------------------------------------------

def filter_by_nifty_for_reversal(sig: pd.Series, df: pd.DataFrame,
                                 nifty: pd.DataFrame,
                                 nifty_filter: str) -> pd.Series:
    """For long late-reversal we want NIFTY *not* crashing further.

    Filters tested:
    - 'none'                 — no filter
    - 'not_below_vwap_b3'    — NIFTY ABOVE VWAP at bar 3 (broad strength)
    - 'first_bullish'        — NIFTY first 5-min bar bullish
    - 'b3_not_crashing'      — NIFTY's first-30-min change > -0.5% (not a crash)
    - 'b3_change_pos'        — NIFTY actually up at bar 3 (>0.1%)
    """
    if nifty_filter == 'none':
        return sig
    if nifty.empty:
        return sig

    if nifty_filter == 'not_below_vwap_b3':
        ok_sessions = nifty.loc[nifty['n_below_vwap_b3'] == 0, 'session']
    elif nifty_filter == 'first_bullish':
        ok_sessions = nifty.loc[nifty['n_first_bearish'] == 0, 'session']
    elif nifty_filter == 'b3_not_crashing':
        ok_sessions = nifty.loc[nifty['n_b3_change_pct'] > -0.5, 'session']
    elif nifty_filter == 'b3_change_pos':
        ok_sessions = nifty.loc[nifty['n_b3_change_pct'] > 0.1, 'session']
    else:
        return sig

    ok_set = set(pd.to_datetime(ok_sessions).values.astype('datetime64[ns]'))
    mask = df['session'].isin(list(ok_set))
    return sig & mask


# ---------------------------------------------------------------------------
# main sweep
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--screen-top', type=int, default=100,
                    help='top-N stocks to per-stock-screen')
    ap.add_argument('--diamond-top', type=int, default=30,
                    help='top-N stocks (after screen) for full sweep')
    ap.add_argument('--diamond-min-trades', type=int, default=20,
                    help='min trades in screen to qualify as diamond')
    ap.add_argument('--diamond-min-wr', type=float, default=0.55,
                    help='min WR in screen to qualify as diamond')
    ap.add_argument('--min-trades', type=int, default=30)
    ap.add_argument('--skip-screen', action='store_true',
                    help='skip per-stock screen, use existing diamond list')
    args = ap.parse_args()

    log_file = TeeLog(LOG_PATH)
    sys.stdout = log_file

    print('=' * 70)
    print(f'Stage 11c — Late-Day Reversal LONG | {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    # universe — top N by row count
    print('Loading 5-min universe ...', flush=True)
    universe = list_5min_universe(min_rows=32_000)
    # exclude indices
    universe = [s for s in universe if s not in ('NIFTY50', 'BANKNIFTY')]
    print(f'  {len(universe)} stocks total; screening top {args.screen_top}')
    screen_syms = universe[:args.screen_top]

    # Build NIFTY regime up front (used in main sweep)
    print('Loading NIFTY regime ...', end='', flush=True)
    t0 = time.time()
    nifty = build_nifty_regime()
    print(f' {len(nifty)} sessions in {time.time() - t0:.0f}s')

    # ---- per-stock screen ----
    if args.skip_screen and os.path.exists(DIAMONDS_TXT):
        with open(DIAMONDS_TXT) as f:
            diamonds = [s.strip() for s in f if s.strip()]
        print(f'Skipping screen — using existing {len(diamonds)} diamonds')
    else:
        print('\n--- Per-stock screen (drop -1.5%, window 24-48, TP 1% / SL 1% / hold 48) ---')
        screen_df = perstock_screen(screen_syms,
                                    drop_pct=-1.5,
                                    entry_window=(24, 48),
                                    tp=1.0, sl=1.0, hold=48,
                                    log=print)
        # rank by WR with min trades
        qual = screen_df[(screen_df['n_trades'] >= args.diamond_min_trades) &
                         (screen_df['win_rate'] >= args.diamond_min_wr)]
        print(f'  qualifying ({args.diamond_min_trades}+ trades, {args.diamond_min_wr*100:.0f}+ WR): {len(qual)}')
        diamonds = qual.nlargest(args.diamond_top, 'win_rate')['symbol'].tolist()
        # always also include top by AWS-ish (WR * sqrt(n))
        if len(diamonds) < 5:
            # fallback: top 25 by WR with at least 10 trades
            relax = screen_df[screen_df['n_trades'] >= 10].nlargest(25, 'win_rate')
            diamonds = relax['symbol'].tolist()
            print(f'  too few qualifiers — using fallback top-25 by WR (n>=10): {len(diamonds)}')
        with open(DIAMONDS_TXT, 'w', encoding='utf-8') as f:
            for s in diamonds:
                f.write(s + '\n')
        print(f'  wrote {len(diamonds)} diamonds to {DIAMONDS_TXT}')

    if not diamonds:
        print('No diamonds found — aborting sweep.')
        return 1

    print('\nDiamond cohort:')
    for s in diamonds:
        print(f'  {s}')

    # ---- preload diamond data ----
    print('\nLoading + enriching diamond data ...', flush=True)
    cache = {}
    t0 = time.time()
    for sym in diamonds:
        df = load_5min(sym)
        if df.empty or len(df) < 1500:
            continue
        try:
            cache[sym] = enrich(df, or_minutes=15)
        except Exception as e:
            print(f'  enrich fail {sym}: {e}')
    print(f'  done {len(cache)} stocks in {time.time() - t0:.0f}s')

    # ---- main parameter sweep ----
    PARAM_GRID = list(itertools.product(
        [-1.0, -1.5, -2.0],                                      # drop_pct
        [(24, 48), (36, 60), (48, 66)],                          # window: afternoon, midday-late, last-hour
        [(30, 35), (30, 40), (28, 35)],                          # (rsi_oversold, rsi_lift)
        ['none', 'not_below_vwap_b3', 'first_bullish',
         'b3_not_crashing', 'b3_change_pos'],                    # nifty filters
    ))
    EXIT_GRID = [
        (0.5, 1.0, 60),
        (0.5, 1.5, 60),
        (1.0, 1.5, 60),
        (0.7, 1.0, 48),
    ]
    n_total = len(PARAM_GRID) * len(EXIT_GRID)
    print(f'\nSweep size: {len(PARAM_GRID)} param combos x {len(EXIT_GRID)} exits = {n_total} systems')

    fields = [
        'family', 'params', 'tp_pct', 'sl_pct', 'max_hold_bars',
        'n_stocks', 'n_trades', 'win_rate', 'avg_win', 'avg_loss',
        'profit_factor', 'sharpe', 'max_dd_pct', 'total_return_pct',
        'trades_per_stock_year', 'aws', 'passes_gates',
    ]

    # incremental write
    done_keys = set()
    if os.path.exists(RANK_CSV):
        with open(RANK_CSV, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                key = (r['params'], r['tp_pct'], r['sl_pct'], r['max_hold_bars'])
                done_keys.add(key)
        print(f'  resume: {len(done_keys)} cells already done')

    mode = 'a' if os.path.exists(RANK_CSV) else 'w'
    with open(RANK_CSV, mode, newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == 'w':
            w.writeheader()

        cell = 0
        n_passes = 0
        t_run = time.time()
        for drop_pct, (ws, we), (rsi_os, rsi_lift), nifty_filter in PARAM_GRID:
            params = dict(drop_pct=drop_pct, window=(ws, we),
                          rsi_oversold=rsi_os, rsi_lift=rsi_lift,
                          nifty_filter=nifty_filter)
            for tp, sl, hold in EXIT_GRID:
                cell += 1
                key = (str(params), str(tp), str(sl), str(hold))
                if key in done_keys:
                    continue

                all_trades = []
                stocks_with_trades = 0
                stock_sessions = 0
                for sym, df in cache.items():
                    sig = late_reversal_signal(
                        df, drop_pct=drop_pct,
                        entry_window_start=ws,
                        entry_window_end=we,
                        rsi_oversold=rsi_os,
                        rsi_lift=rsi_lift,
                    )
                    sig = filter_by_nifty_for_reversal(sig, df, nifty, nifty_filter)
                    rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold,
                                       direction='long')
                    trades = simulate_signals(df, sig, rules)
                    if len(trades) == 0:
                        continue
                    trades['symbol'] = sym
                    all_trades.append(trades)
                    stocks_with_trades += 1
                    stock_sessions += df['session'].nunique()
                if not all_trades:
                    continue
                t = pd.concat(all_trades, ignore_index=True)
                avg_sessions = stock_sessions / max(stocks_with_trades, 1)
                stats = trade_stats(t, sessions=int(avg_sessions))
                trades_per_stock_year = stats['trades_per_year'] / max(stocks_with_trades, 1)
                passes = (stats['n_trades'] >= args.min_trades and
                          stats['win_rate'] >= 0.75 and
                          stats['profit_factor'] >= 2.0)
                row = dict(
                    family='long_late_reversal',
                    params=str(params),
                    tp_pct=tp, sl_pct=sl, max_hold_bars=hold,
                    n_stocks=stocks_with_trades, n_trades=stats['n_trades'],
                    win_rate=round(stats['win_rate'], 4),
                    avg_win=round(stats['avg_win'], 4),
                    avg_loss=round(stats['avg_loss'], 4),
                    profit_factor=round(stats['profit_factor'], 3),
                    sharpe=round(stats['sharpe'], 3),
                    max_dd_pct=round(stats['max_dd_pct'], 3),
                    total_return_pct=round(stats['total_return_pct'], 2),
                    trades_per_stock_year=round(trades_per_stock_year, 1),
                    aws=round(stats['aws'], 4),
                    passes_gates=int(passes),
                )
                w.writerow(row)
                f.flush()
                if passes:
                    n_passes += 1
                if stats['win_rate'] >= 0.65 and stats['n_trades'] >= args.min_trades:
                    print(f'  [{cell}/{n_total}] drop={drop_pct} win={ws}-{we} '
                          f'rsi={rsi_os}/{rsi_lift} nifty={nifty_filter} '
                          f'tp={tp} sl={sl} hold={hold} '
                          f'-> WR={stats["win_rate"]:.1%} n={stats["n_trades"]} '
                          f'PF={stats["profit_factor"]:.2f} DD={stats["max_dd_pct"]:.1f}%')
                if cell % 50 == 0:
                    print(f'  ... cell {cell}/{n_total}, {n_passes} passing, '
                          f'elapsed {time.time() - t_run:.0f}s')

        print(f'\nDone. {n_passes}/{n_total} systems passing hard gates. '
              f'Output: {RANK_CSV}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
