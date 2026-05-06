"""Stage 6 — high-conviction empirical strategies.

After the Stage-4 confluence stack failed (WR ~36% — survivor-bias trap
where signature correlates with big-move days but doesn't predict the
NEXT 12 bars), pivot to strategies that are empirically known to have
high WR on intraday data:

(A) FIRST-HOUR FADE — when the first hour produces a wide one-direction
    move, fade back to VWAP / mid. Mean-reversion has 65–75% WR
    historically because most wide first-hour moves get bought/sold-back.

(B) DAY-SIGNATURE BET (full-day hold) — use the Open=Low/Open=High
    morning signature but exit at session END, not on TP/SL. This bets
    the day-direction prediction directly.

(C) PRIOR-DAY-CLOSE BREAK — break and hold above (long) / below (short)
    yesterday's close in the first 30 min. Decent WR base.

(D) VWAP MAGNET — once price is > 0.8% from VWAP and reversing toward
    it, take a small fade trade.

Output: results/06_<strategy>_ranking.csv
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


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))


# ---------------------------------------------------------------------------
# (A) FIRST-HOUR FADE
# ---------------------------------------------------------------------------
def first_hour_fade(df: pd.DataFrame,
                    fh_bars: int = 12,         # 60 min = 12 bars
                    min_fh_range_pct: float = 1.0,
                    direction: str = 'short',  # 'short' to fade up, 'long' to fade down
                    rsi_extreme: float = 70,   # require RSI > 70 (for short) or < 30 (for long)
                    extension_atr: float = 1.5  # close must be >= extension_atr * ATR from VWAP
                    ) -> tuple[pd.Series, str]:
    """Fade a wide one-direction first-hour move.

    SHORT signal: by end of first hour, price is up >= min_fh_range_pct from
    open AND RSI > 70 AND price is >= extension_atr above VWAP.
    LONG signal: mirror.

    Entry on bar fh_bars+1 (start of 2nd hour). Holds until TP/SL/EOD.
    """
    fh_close = df.groupby('session')['close'].transform(
        lambda x: x.iloc[fh_bars - 1] if len(x) >= fh_bars else np.nan
    )
    fh_high = df.groupby('session')['high'].transform(
        lambda x: x.iloc[:fh_bars].max() if len(x) >= fh_bars else np.nan
    )
    fh_low = df.groupby('session')['low'].transform(
        lambda x: x.iloc[:fh_bars].min() if len(x) >= fh_bars else np.nan
    )
    day_open = df['day_open']

    fh_pct_up = (fh_high - day_open) / day_open * 100
    fh_pct_dn = (day_open - fh_low) / day_open * 100

    # entry only at fh_bars (start of 2nd hour)
    at_entry = df['bar_idx'] == fh_bars
    above_vwap_atr = (df['close'] - df['vwap']) / df['atr'].replace(0, np.nan)

    if direction == 'short':
        cond = (
            at_entry
            & (fh_pct_up >= min_fh_range_pct)
            & (df['rsi'] >= rsi_extreme)
            & (above_vwap_atr >= extension_atr)
        )
    else:
        cond = (
            at_entry
            & (fh_pct_dn >= min_fh_range_pct)
            & (df['rsi'] <= (100 - rsi_extreme))
            & (above_vwap_atr <= -extension_atr)
        )

    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=10)
    return sig, direction


# ---------------------------------------------------------------------------
# (B) DAY-SIGNATURE BET — use morning signature, hold till EOD
# ---------------------------------------------------------------------------
def day_signature_bet(df: pd.DataFrame,
                      side: str = 'long',
                      open_eq_tol: float = 0.0005,
                      rsi_min_long: float = 55,
                      rsi_max_short: float = 45,
                      gap_max_long: float = 0.3,
                      gap_min_short: float = -0.3,
                      entry_bar: int = 1) -> tuple[pd.Series, str]:
    """Place a single end-of-day directional bet at bar 1 based on morning signature."""
    g_first = df.groupby('session').head(1)
    fb_open = pd.Series(g_first['open'].values, index=g_first['session'].values)
    fb_high = pd.Series(g_first['high'].values, index=g_first['session'].values)
    fb_low = pd.Series(g_first['low'].values, index=g_first['session'].values)
    fb_close = pd.Series(g_first['close'].values, index=g_first['session'].values)

    open_eq_low = (fb_open - fb_low).abs() / fb_open <= open_eq_tol
    open_eq_high = (fb_open - fb_high).abs() / fb_open <= open_eq_tol
    first_bullish = fb_close > fb_open
    first_bearish = fb_close < fb_open

    sess_arr = df['session'].values
    is_oel = open_eq_low.reindex(sess_arr).values
    is_oeh = open_eq_high.reindex(sess_arr).values
    is_fb_bull = first_bullish.reindex(sess_arr).values
    is_fb_bear = first_bearish.reindex(sess_arr).values

    g_open = df.groupby('session')['day_open'].first()
    g_pc = df.groupby('session')['prev_close'].first()
    gap_pct = ((g_open - g_pc) / g_pc * 100).reindex(sess_arr).values

    if side == 'long':
        cond = (
            (df['bar_idx'].values == entry_bar)
            & is_oel
            & is_fb_bull
            & (df['rsi'].values >= rsi_min_long)
            & (gap_pct <= gap_max_long)
        )
    else:
        cond = (
            (df['bar_idx'].values == entry_bar)
            & is_oeh
            & is_fb_bear
            & (df['rsi'].values <= rsi_max_short)
            & (gap_pct >= gap_min_short)
        )
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig, side


# ---------------------------------------------------------------------------
# (C) PRIOR-DAY-CLOSE BREAK
# ---------------------------------------------------------------------------
def prev_close_break(df: pd.DataFrame,
                     side: str = 'long',
                     hold_for_break_bars: int = 2,
                     max_entry_bar: int = 6,
                     rsi_min: float = 50) -> tuple[pd.Series, str]:
    """LONG: close > prev_close for hold_for_break_bars consecutive bars within
    first max_entry_bar bars; close > VWAP."""
    if 'prev_close' not in df.columns:
        return pd.Series(False, index=df.index), side
    if side == 'long':
        held = (df['close'] > df['prev_close']).rolling(
            hold_for_break_bars, min_periods=hold_for_break_bars
        ).min().astype(bool)
        cond = (
            held
            & (df['close'] > df['vwap'])
            & (df['rsi'] >= rsi_min)
            & (df['bar_idx'] >= hold_for_break_bars)
            & (df['bar_idx'] <= max_entry_bar)
        )
    else:
        held = (df['close'] < df['prev_close']).rolling(
            hold_for_break_bars, min_periods=hold_for_break_bars
        ).min().astype(bool)
        cond = (
            held
            & (df['close'] < df['vwap'])
            & (df['rsi'] <= (100 - rsi_min))
            & (df['bar_idx'] >= hold_for_break_bars)
            & (df['bar_idx'] <= max_entry_bar)
        )
    sig = cond.fillna(False)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig, side


# ---------------------------------------------------------------------------
# (D) VWAP MAGNET — fade extension reverting to VWAP
# ---------------------------------------------------------------------------
def vwap_magnet(df: pd.DataFrame,
                side: str = 'short',
                extension_atr: float = 1.5,
                rsi_extreme: float = 65,
                require_reversal_bar: bool = True) -> tuple[pd.Series, str]:
    """SHORT: price >= extension_atr above VWAP, RSI > 65, current bar is a
    bearish reversal (close < open AND high > prev high)."""
    dist = (df['close'] - df['vwap']) / df['atr'].replace(0, np.nan)
    if side == 'short':
        far = dist >= extension_atr
        rsi_ok = df['rsi'] >= rsi_extreme
        if require_reversal_bar:
            rev = (df['close'] < df['open']) & (df['high'] > df['high'].shift(1))
        else:
            rev = pd.Series(True, index=df.index)
    else:
        far = dist <= -extension_atr
        rsi_ok = df['rsi'] <= (100 - rsi_extreme)
        if require_reversal_bar:
            rev = (df['close'] > df['open']) & (df['low'] < df['low'].shift(1))
        else:
            rev = pd.Series(True, index=df.index)
    cond = far & rsi_ok & rev & (df['bar_idx'] >= 6)
    sig = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df)
    return sig, side


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
GRIDS = {
    'first_hour_fade': dict(
        fh_bars=[6, 9, 12],
        min_fh_range_pct=[0.8, 1.0, 1.5, 2.0],
        direction=['short', 'long'],
        rsi_extreme=[65, 70, 75],
        extension_atr=[1.0, 1.5, 2.0],
    ),
    'day_signature_bet': dict(
        side=['long', 'short'],
        rsi_min_long=[50, 55, 60],
        rsi_max_short=[40, 45, 50],
        gap_max_long=[0.3, 0.8, 99.0],
        gap_min_short=[-99.0, -0.8, -0.3],
        entry_bar=[1, 2],
    ),
    'prev_close_break': dict(
        side=['long', 'short'],
        hold_for_break_bars=[1, 2, 3],
        max_entry_bar=[3, 6, 12],
        rsi_min=[50, 55, 60],
    ),
    'vwap_magnet': dict(
        side=['short', 'long'],
        extension_atr=[1.0, 1.5, 2.0],
        rsi_extreme=[60, 65, 70],
        require_reversal_bar=[True],
    ),
}

STRATEGIES = {
    'first_hour_fade': first_hour_fade,
    'day_signature_bet': day_signature_bet,
    'prev_close_break': prev_close_break,
    'vwap_magnet': vwap_magnet,
}

EXIT_GRID = [
    (0.4, 0.3, 12),
    (0.6, 0.4, 18),
    (0.8, 0.4, 24),
    (1.0, 0.5, 36),
    (0.5, 0.5, 30),    # 1:1 RR — for fades
    (1.0, 0.7, 48),
    # full-day hold variants
    (5.0, 1.0, 60),    # essentially exit-on-stop or EOD
    (3.0, 0.8, 60),
]


def expand(grid):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    return [dict(zip(keys, c)) for c in itertools.product(*vals)]


def run_strategy(strat: str, stocks: list[str], min_trades: int) -> int:
    fn = STRATEGIES[strat]
    grid = GRIDS[strat]
    combos = expand(grid)

    print(f'[{strat}] {len(stocks)} stocks × {len(combos)} params × {len(EXIT_GRID)} exits')

    print('  loading + enriching ...', end='', flush=True)
    t0 = time.time()
    cache = {}
    for sym in stocks:
        df = load_5min(sym)
        if df.empty or len(df) < 1000:
            continue
        try:
            cache[sym] = enrich(df, or_minutes=15)
        except Exception:
            continue
    print(f' done {len(cache)} stocks in {time.time()-t0:.0f}s')

    out = os.path.join(RESULTS_DIR, f'06_{strat}_ranking.csv')
    fields = [
        'family', 'params', 'tp_pct', 'sl_pct', 'max_hold_bars',
        'n_stocks', 'n_trades', 'win_rate', 'avg_win', 'avg_loss',
        'profit_factor', 'sharpe', 'max_dd_pct', 'total_return_pct',
        'trades_per_stock_year', 'aws', 'passes_gates',
    ]
    done = set()
    if os.path.exists(out):
        with open(out, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                done.add((row['params'], row['tp_pct'], row['sl_pct'], row['max_hold_bars']))
        mode = 'a'
    else:
        mode = 'w'

    cell = 0
    n_total = len(combos) * len(EXIT_GRID)
    t_run = time.time()
    with open(out, mode, newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == 'w':
            w.writeheader()
        for params in combos:
            for tp, sl, hold in EXIT_GRID:
                cell += 1
                key = (str(params), str(tp), str(sl), str(hold))
                if key in done:
                    continue
                all_trades = []
                stocks_with_trades = 0
                stock_sessions = 0
                direction = None
                for sym, df in cache.items():
                    try:
                        sig, direction = fn(df, **params)
                    except Exception:
                        continue
                    rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold, direction=direction)
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
                passes = (stats['n_trades'] >= min_trades and stats['win_rate'] >= 0.60)
                row = dict(
                    family=strat, params=str(params),
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
                if stats['win_rate'] >= 0.65 and stats['n_trades'] >= min_trades:
                    print(f'  [{cell}/{n_total}] {strat} {params} tp={tp} sl={sl} hold={hold} '
                          f'→ WR={stats["win_rate"]:.1%} n={stats["n_trades"]} '
                          f'PF={stats["profit_factor"]:.2f} DD={stats["max_dd_pct"]:.1f}%')
                if cell % 25 == 0:
                    elapsed = time.time() - t_run
                    eta = (n_total - cell) * elapsed / max(cell, 1)
                    print(f'  progress: {cell}/{n_total} ({elapsed:.0f}s, ETA {eta:.0f}s)', flush=True)
    print(f'[{strat}] done — output {out}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', required=True,
                    help='one of first_hour_fade, day_signature_bet, prev_close_break, vwap_magnet, all')
    ap.add_argument('--max-stocks', type=int, default=50)
    ap.add_argument('--min-trades', type=int, default=30)
    ap.add_argument('--cohort', choices=['top', 'natural'], default='top',
                    help='top = top-N by row count; natural = natural confluence cohort')
    args = ap.parse_args()

    if args.cohort == 'natural':
        cohort_path = os.path.join(RESULTS_DIR, '00_natural_cohort.txt')
        with open(cohort_path) as f:
            stocks = [line.strip() for line in f if line.strip()][:args.max_stocks]
    else:
        universe = list_5min_universe(min_rows=32_000)
        stocks = universe[:args.max_stocks]
    print(f'Universe ({args.cohort}): {len(stocks)} stocks')

    strats = list(STRATEGIES.keys()) if args.strategy == 'all' else [args.strategy]
    for s in strats:
        run_strategy(s, stocks, args.min_trades)
    return 0


if __name__ == '__main__':
    sys.exit(main())
