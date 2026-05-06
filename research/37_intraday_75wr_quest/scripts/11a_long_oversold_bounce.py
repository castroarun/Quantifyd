"""Stage 11a — LONG mean-reversion at OVERSOLD EXTREMES.

Hypothesis: stocks that get extremely oversold intraday (RSI < 25-30
AND price 1.5-2.5x ATR below VWAP) often bounce. Looking for stocks
where this pattern hits 75%+ WR after layering NIFTY regime.

Pipeline:
  Stage A (per-stock screen)
    - Default rule: entry_bar=12, RSI<25, dev>=2.0 ATR below VWAP,
      bullish bar; TP 0.7%, SL 1.0%, hold 36 bars.
    - Run on all stocks with >= 32k rows (the 310-universe).
    - Keep stocks with WR >= 60% AND n >= 15 trades.
    - Top 25-30 by WR = the "bounce diamonds".

  Stage B (full sweep on bounce diamonds)
    - Grid: entry_bar in {6,9,12,18,24} x RSI<{20,25,30}
            x dev_atr in {1.5,2.0,2.5}
            x exits in {(0.5,1.5,60),(1.0,1.5,60),(0.7,1.0,36)}
            x NIFTY filter in {none, above_vwap_b3, first_bullish, both,
                                b3_change_pos, b3_change_pos_strong,
                                bouncing}.
    - 'bouncing' = NIFTY itself was below VWAP at b3 but turned bullish
       by b6 (mean-reversion regime).

Output:
  results/11a_long_oversold_bounce_screen.csv  (per-stock Stage A)
  results/11a_long_oversold_bounce_diamonds.txt
  results/11a_long_oversold_bounce_ranking.csv (Stage B sweep)
  logs/11a_long_oversold_bounce.log
"""

from __future__ import annotations

import argparse
import csv
import importlib
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

# reuse build_nifty_regime from Stage 8
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

SCREEN_PATH = os.path.join(RESULTS_DIR, '11a_long_oversold_bounce_screen.csv')
DIAMONDS_PATH = os.path.join(RESULTS_DIR, '11a_long_oversold_bounce_diamonds.txt')
RANKING_PATH = os.path.join(RESULTS_DIR, '11a_long_oversold_bounce_ranking.csv')
LOG_PATH = os.path.join(LOGS_DIR, '11a_long_oversold_bounce.log')


def log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    except Exception:
        pass


# ---------------------------------------------------------------------------
# signal: long mean-reversion at oversold extreme
# ---------------------------------------------------------------------------

def oversold_bounce_signal(df: pd.DataFrame,
                           entry_bar: int = 12,
                           rsi_max: float = 25.0,
                           dev_atr: float = 2.0,
                           require_bullish_bar: bool = True) -> pd.Series:
    """LONG signal at entry_bar:
       - RSI(14) < rsi_max
       - close is at least dev_atr * ATR below VWAP
       - current bar is bullish (close > open)  (optional)
       Single signal per session; signal-bar must be >= entry_bar.
       Trade enters at NEXT bar's open (engine convention).
    """
    bar_idx = df['bar_idx'].values
    rsi = df['rsi'].values
    close = df['close'].values
    open_ = df['open'].values
    vwap = df['vwap'].values
    atr = df['atr'].values
    atr_safe = np.where(atr > 0, atr, np.nan)

    dev = (vwap - close) / atr_safe  # positive when below VWAP

    cond = (
        (bar_idx >= entry_bar)
        & (rsi < rsi_max)
        & (dev >= dev_atr)
    )
    if require_bullish_bar:
        cond = cond & (close > open_)

    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig


# ---------------------------------------------------------------------------
# NIFTY regime filter — long-side variants + bouncing variant
# ---------------------------------------------------------------------------

def filter_by_nifty(sig: pd.Series, df: pd.DataFrame, nifty: pd.DataFrame,
                    nifty_filter: str) -> pd.Series:
    if nifty_filter == 'none' or nifty.empty:
        return sig

    if nifty_filter == 'above_vwap_b3':
        ok_sessions = nifty.loc[nifty['n_below_vwap_b3'] == 0, 'session']
    elif nifty_filter == 'first_bullish':
        ok_sessions = nifty.loc[nifty['n_first_bearish'] == 0, 'session']
    elif nifty_filter == 'both':
        ok_sessions = nifty.loc[
            (nifty['n_below_vwap_b3'] == 0) & (nifty['n_first_bearish'] == 0),
            'session',
        ]
    elif nifty_filter == 'b3_change_pos':
        ok_sessions = nifty.loc[nifty['n_b3_change_pct'] > 0.1, 'session']
    elif nifty_filter == 'b3_change_pos_strong':
        ok_sessions = nifty.loc[nifty['n_b3_change_pct'] > 0.3, 'session']
    elif nifty_filter == 'not_crashing':
        # NIFTY first-30min change > -0.3% (avoid days NIFTY itself is dumping)
        ok_sessions = nifty.loc[nifty['n_b3_change_pct'] > -0.3, 'session']
    elif nifty_filter == 'bouncing':
        # NIFTY was below VWAP at b3 but turned bullish in next 15min
        ok_sessions = nifty.loc[
            (nifty['n_below_vwap_b3'] == 1)
            & (nifty['n_b6_close'] > nifty['n_b6_vwap']),
            'session',
        ]
    else:
        return sig

    ok_set = set(pd.to_datetime(ok_sessions).values.astype('datetime64[ns]'))
    mask = df['session'].isin(list(ok_set))
    return sig & mask


# ---------------------------------------------------------------------------
# Stage A — per-stock screen with one default param combo
# ---------------------------------------------------------------------------

DEFAULT_SCREEN = dict(
    entry_bar=12,
    rsi_max=25.0,
    dev_atr=2.0,
    tp=0.7,
    sl=1.0,
    hold=36,
)


def run_screen(universe: list[str], min_rows: int = 1000,
               min_trades: int = 15) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Per-stock screen. Returns (screen_df, cache_of_enriched_dfs)."""
    rows = []
    cache: dict[str, pd.DataFrame] = {}
    fields = [
        'symbol', 'n_sessions', 'n_trades',
        'win_rate', 'avg_win', 'avg_loss', 'profit_factor',
        'sharpe', 'max_dd_pct', 'total_return_pct',
    ]
    # write screen CSV incrementally
    with open(SCREEN_PATH, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()

        rules = TradeRules(
            tp_pct=DEFAULT_SCREEN['tp'],
            sl_pct=DEFAULT_SCREEN['sl'],
            max_hold_bars=DEFAULT_SCREEN['hold'],
            direction='long',
        )

        t0 = time.time()
        for i, sym in enumerate(universe, 1):
            try:
                df = load_5min(sym)
                if df.empty or len(df) < min_rows:
                    continue
                df = enrich(df, or_minutes=15)
            except Exception as e:
                log(f'  [{i}] {sym} enrich fail: {e}')
                continue

            sig = oversold_bounce_signal(
                df,
                entry_bar=DEFAULT_SCREEN['entry_bar'],
                rsi_max=DEFAULT_SCREEN['rsi_max'],
                dev_atr=DEFAULT_SCREEN['dev_atr'],
            )
            trades = simulate_signals(df, sig, rules)
            if len(trades) == 0:
                cache[sym] = df  # still cache for Stage B if it becomes diamond via other params
                row = dict(symbol=sym, n_sessions=df['session'].nunique(),
                           n_trades=0, win_rate=np.nan, avg_win=np.nan,
                           avg_loss=np.nan, profit_factor=np.nan,
                           sharpe=np.nan, max_dd_pct=np.nan,
                           total_return_pct=np.nan)
            else:
                stats = trade_stats(trades, sessions=df['session'].nunique())
                row = dict(
                    symbol=sym, n_sessions=df['session'].nunique(),
                    n_trades=stats['n_trades'],
                    win_rate=round(stats['win_rate'], 4),
                    avg_win=round(stats['avg_win'], 4),
                    avg_loss=round(stats['avg_loss'], 4),
                    profit_factor=round(stats['profit_factor'], 3),
                    sharpe=round(stats['sharpe'], 3),
                    max_dd_pct=round(stats['max_dd_pct'], 3),
                    total_return_pct=round(stats['total_return_pct'], 2),
                )
                cache[sym] = df

            w.writerow(row)
            fh.flush()
            rows.append(row)

            if i % 25 == 0:
                elapsed = time.time() - t0
                log(f'  screen [{i}/{len(universe)}] elapsed={elapsed:.0f}s '
                    f'cache_size={len(cache)}')

    df_screen = pd.DataFrame(rows)
    return df_screen, cache


# ---------------------------------------------------------------------------
# Stage B — full sweep on bounce diamonds
# ---------------------------------------------------------------------------

PARAM_GRID = list(itertools.product(
    [6, 9, 12, 18, 24],          # entry_bar
    [20.0, 25.0, 30.0],          # rsi_max
    [1.5, 2.0, 2.5],             # dev_atr
    ['none', 'above_vwap_b3', 'first_bullish', 'both',
     'b3_change_pos', 'b3_change_pos_strong', 'not_crashing', 'bouncing'],
))
EXIT_GRID = [
    (0.5, 1.5, 60),
    (1.0, 1.5, 60),
    (0.7, 1.0, 36),
]


def run_sweep(diamonds: list[str], cache: dict[str, pd.DataFrame],
              nifty: pd.DataFrame, min_trades: int = 30) -> None:
    """Full sweep on bounce-diamond cohort."""
    fields = [
        'family', 'params', 'tp_pct', 'sl_pct', 'max_hold_bars',
        'n_stocks', 'n_trades', 'win_rate', 'avg_win', 'avg_loss',
        'profit_factor', 'sharpe', 'max_dd_pct', 'total_return_pct',
        'trades_per_stock_year', 'aws', 'passes_gates',
    ]
    n_total = len(PARAM_GRID) * len(EXIT_GRID)
    log(f'Stage B sweep: {len(PARAM_GRID)} param combos x {len(EXIT_GRID)} '
        f'exits = {n_total} systems on {len(diamonds)} stocks')

    with open(RANKING_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        cell = 0
        t_run = time.time()
        for entry_bar, rsi_thr, dev_atr, nifty_filter in PARAM_GRID:
            params = dict(entry_bar=entry_bar, rsi_max=rsi_thr,
                          dev_atr=dev_atr, nifty_filter=nifty_filter)
            for tp, sl, hold in EXIT_GRID:
                cell += 1
                all_trades = []
                stocks_with_trades = 0
                stock_sessions = 0
                for sym in diamonds:
                    df = cache.get(sym)
                    if df is None:
                        continue
                    sig = oversold_bounce_signal(
                        df,
                        entry_bar=entry_bar,
                        rsi_max=rsi_thr,
                        dev_atr=dev_atr,
                    )
                    sig = filter_by_nifty(sig, df, nifty, nifty_filter)
                    rules = TradeRules(tp_pct=tp, sl_pct=sl,
                                       max_hold_bars=hold, direction='long')
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
                trades_per_stock_year = (
                    stats['trades_per_year'] / max(stocks_with_trades, 1)
                )
                passes = (
                    stats['n_trades'] >= min_trades
                    and stats['win_rate'] >= 0.75
                    and stats['profit_factor'] >= 2.0
                    and trades_per_stock_year >= 8.0
                )
                row = dict(
                    family='oversold_bounce_long',
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
                if (stats['win_rate'] >= 0.70 and
                        stats['n_trades'] >= min_trades):
                    log(f'  [{cell}/{n_total}] {params} tp={tp} sl={sl} '
                        f'hold={hold} -> WR={stats["win_rate"]:.1%} '
                        f'n={stats["n_trades"]} '
                        f'PF={stats["profit_factor"]:.2f} '
                        f'DD={stats["max_dd_pct"]:.1f}%')

        log(f'Stage B done in {time.time()-t_run:.0f}s -> {RANKING_PATH}')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-screen-trades', type=int, default=15,
                    help='Minimum n trades per stock in screen to be eligible')
    ap.add_argument('--min-screen-wr', type=float, default=0.60,
                    help='Minimum WR per stock in screen to qualify as diamond')
    ap.add_argument('--top-n-diamonds', type=int, default=30,
                    help='Cap the diamond cohort size')
    ap.add_argument('--min-sweep-trades', type=int, default=30,
                    help='Minimum total trades to pass hard gates')
    ap.add_argument('--max-stocks', type=int, default=0,
                    help='If >0, limit universe size (for fast iteration)')
    ap.add_argument('--skip-stage-a', action='store_true',
                    help='Skip Stage A and reuse existing diamonds list')
    args = ap.parse_args()

    # fresh log
    open(LOG_PATH, 'w', encoding='utf-8').close()
    log(f'Stage 11a — long oversold bounce — start')
    log(f'  default screen params: {DEFAULT_SCREEN}')
    log(f'  args: {vars(args)}')

    log('Loading NIFTY regime ...')
    t0 = time.time()
    nifty = build_nifty_regime()
    log(f'  NIFTY {len(nifty)} sessions in {time.time()-t0:.0f}s')

    if args.skip_stage_a and os.path.exists(DIAMONDS_PATH):
        with open(DIAMONDS_PATH) as fh:
            diamonds = [s.strip() for s in fh if s.strip()]
        log(f'Reusing {len(diamonds)} diamonds from {DIAMONDS_PATH}')
        # need to rebuild cache for Stage B
        log('Rebuilding cache for diamonds ...')
        cache: dict[str, pd.DataFrame] = {}
        t0 = time.time()
        for sym in diamonds:
            df = load_5min(sym)
            if df.empty or len(df) < 1000:
                continue
            try:
                cache[sym] = enrich(df, or_minutes=15)
            except Exception:
                continue
        log(f'  cache built {len(cache)} stocks in {time.time()-t0:.0f}s')
    else:
        universe = list_5min_universe(min_rows=32_000)
        if args.max_stocks > 0:
            universe = universe[:args.max_stocks]
        log(f'Stage A — per-stock screen on {len(universe)} stocks')

        t0 = time.time()
        df_screen, cache = run_screen(universe, min_trades=args.min_screen_trades)
        log(f'Stage A done in {time.time()-t0:.0f}s. Cached {len(cache)} stocks.')

        # rank stocks
        elig = df_screen.dropna(subset=['win_rate'])
        elig = elig[elig['n_trades'] >= args.min_screen_trades].copy()
        elig.sort_values('win_rate', ascending=False, inplace=True)
        diamonds_df = elig[elig['win_rate'] >= args.min_screen_wr].head(args.top_n_diamonds)
        diamonds = diamonds_df['symbol'].tolist()

        log(f'  eligible (n>= {args.min_screen_trades}): {len(elig)}')
        log(f'  diamonds (WR >= {args.min_screen_wr:.0%}): {len(diamonds)}')
        if len(diamonds) > 0:
            log('  top picks:')
            for _, r in diamonds_df.head(15).iterrows():
                log(f'    {r["symbol"]:<14} WR={r["win_rate"]:.1%} '
                    f'n={int(r["n_trades"])} PF={r["profit_factor"]} '
                    f'sharpe={r["sharpe"]}')

        if len(diamonds) < 5:
            # Fall back: take top 25 by WR ignoring 60% gate (still need n>=15)
            log('  WARNING: fewer than 5 diamonds at 60% gate; '
                'falling back to top 25 by WR among eligible.')
            diamonds = elig.head(25)['symbol'].tolist()

        with open(DIAMONDS_PATH, 'w', encoding='utf-8') as fh:
            for s in diamonds:
                fh.write(s + '\n')
        log(f'  wrote diamonds to {DIAMONDS_PATH}')

    if not diamonds:
        log('No bounce diamonds found — aborting Stage B.')
        return 1

    run_sweep(diamonds, cache, nifty, min_trades=args.min_sweep_trades)
    log(f'Done. ranking={RANKING_PATH}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
