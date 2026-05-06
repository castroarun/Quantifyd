"""Stage 11b - LONG Trend-Pullback in confirmed-uptrend day.

Hypothesis: in stocks that GAP UP + show first-hour strength, the *first
pullback* to a moving average (EMA9 or VWAP) is a high-probability long
entry. Strong-open day + bullish bounce confirmation = trend continuation.

Signal (LONG):
- Stock GAPPED UP (>= 0.3% / 0.5% / 1.0% above prev_close)
- First 30 min bullish: bar 6 close > day_open AND first-hour high > day_open + 0.5%
- Pullback: current bar low touches EMA9 OR within 0.3% of VWAP
- Current bar IS bullish (close > open) - bounce confirmation
- RSI(14) > rsi_floor (45/50/55)
- entry window bars 7-30 (10:00-12:30)
- NIFTY filter optional: NIFTY also gap-up + bullish at bar 6

Two-pass design:
  Pass 1: Per-stock screen with default params on universe -> identify "trend-pullback diamonds"
  Pass 2: Full sweep on diamond cohort + NIFTY filters

Output: results/11b_long_trend_pullback_ranking.csv
Diamonds: results/11b_trend_pullback_diamonds.txt
Log: logs/11b_long_trend_pullback.log
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import (  # type: ignore
    load_5min, enrich, simulate_signals, trade_stats, TradeRules,
    list_5min_universe,
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

OUT_CSV = os.path.join(RESULTS_DIR, '11b_long_trend_pullback_ranking.csv')
DIAMONDS_OUT = os.path.join(RESULTS_DIR, '11b_trend_pullback_diamonds.txt')
SCREEN_CSV = os.path.join(RESULTS_DIR, '11b_per_stock_screen.csv')
LOG_PATH = os.path.join(LOGS_DIR, '11b_long_trend_pullback.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[logging.FileHandler(LOG_PATH, mode='a', encoding='utf-8'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('11b')


# ---------------------------------------------------------------------------
# signal builder
# ---------------------------------------------------------------------------

def trend_pullback_signal(
    df: pd.DataFrame,
    gap_min_pct: float = 0.3,
    first_hour_strength_pct: float = 0.5,
    pullback_mode: str = 'ema9_touch',  # 'ema9_touch' | 'ema9_within_0p2' | 'vwap_within_0p3' | 'vwap_within_0p5'
    rsi_floor: float = 45.0,
    bar_min: int = 7,
    bar_max: int = 30,
) -> pd.Series:
    """Trend-pullback LONG signal.

    Pre-conditions captured per session:
    - gap_up >= gap_min_pct
    - bar_6 close > day_open
    - first-hour high (bars 0..11 high) > day_open * (1 + first_hour_strength_pct/100)

    Per-bar conditions (bars bar_min..bar_max):
    - pullback condition (one of pullback_mode variants)
    - bullish bar (close > open)
    - rsi > rsi_floor
    """
    if 'prev_close' not in df.columns:
        return pd.Series(False, index=df.index)
    if 'ema9' not in df.columns or 'vwap' not in df.columns:
        return pd.Series(False, index=df.index)

    # Per-session pre-conditions
    sess = df['session']
    day_open = df.groupby(sess)['open'].transform('first')
    prev_close = df['prev_close']
    gap_pct = (day_open - prev_close) / prev_close * 100
    gap_ok = gap_pct >= gap_min_pct

    # bar_6 close > day_open
    bar6_mask = (df['bar_idx'] == 6)
    bar6_close = df.loc[bar6_mask].groupby(sess)['close'].first()
    sess_b6_close = df['session'].map(bar6_close)
    bar6_ok = sess_b6_close > day_open

    # first-hour high (bars 0..11)
    fh_mask = (df['bar_idx'] >= 0) & (df['bar_idx'] <= 11)
    # cumulative max of high within session up to bar 11
    fh_high_per_sess = df.loc[fh_mask].groupby(sess)['high'].max()
    sess_fh_high = df['session'].map(fh_high_per_sess)
    strength_ok = sess_fh_high > day_open * (1 + first_hour_strength_pct / 100)

    # Per-bar conditions
    if pullback_mode == 'ema9_touch':
        pullback_ok = (df['low'] <= df['ema9']) & (df['close'] >= df['ema9'] * 0.998)
    elif pullback_mode == 'ema9_within_0p2':
        pullback_ok = ((df['low'] - df['ema9']).abs() / df['ema9']) <= 0.002
    elif pullback_mode == 'vwap_within_0p3':
        pullback_ok = ((df['low'] - df['vwap']).abs() / df['vwap']) <= 0.003
    elif pullback_mode == 'vwap_within_0p5':
        pullback_ok = ((df['low'] - df['vwap']).abs() / df['vwap']) <= 0.005
    elif pullback_mode == 'ema9_or_vwap':
        e9 = ((df['low'] - df['ema9']).abs() / df['ema9']) <= 0.002
        vw = ((df['low'] - df['vwap']).abs() / df['vwap']) <= 0.003
        pullback_ok = e9 | vw
    else:
        pullback_ok = pd.Series(False, index=df.index)

    bullish = df['close'] > df['open']
    rsi_ok = df['rsi'] > rsi_floor

    in_window = (df['bar_idx'] >= bar_min) & (df['bar_idx'] <= bar_max)

    sig = (
        gap_ok.fillna(False)
        & bar6_ok.fillna(False)
        & strength_ok.fillna(False)
        & pullback_ok.fillna(False)
        & bullish
        & rsi_ok
        & in_window
    )
    sig = sig.astype(bool)
    sig = _one_per_session(sig, df['session'])
    sig = _no_late_entries(sig, df, last_n=5)
    return sig.fillna(False)


# ---------------------------------------------------------------------------
# NIFTY filter (long-side specialised)
# ---------------------------------------------------------------------------

def filter_by_nifty_long(sig: pd.Series, df: pd.DataFrame, nifty: pd.DataFrame,
                         nifty_filter: str) -> pd.Series:
    """Apply NIFTY-UP regime filter for long entries.

    Filters:
    - 'none'                : no filter
    - 'nifty_gap_up'        : NIFTY gap_pct >= 0.1%
    - 'nifty_bullish_b6'    : NIFTY bar 6 close > NIFTY day_open
    - 'nifty_strong_both'   : both gap-up AND bullish at b6
    - 'nifty_b3_change_pos' : NIFTY first 15-min change > 0.1%
    """
    if nifty_filter == 'none' or nifty.empty:
        return sig

    n = nifty.copy()
    # add bullish_b6 column
    n['n_bullish_b6'] = (n['n_b6_close'] > n['n_day_open']).astype(int)
    n['n_gap_up'] = (n['n_gap_pct'] >= 0.1).astype(int)

    if nifty_filter == 'nifty_gap_up':
        ok = n.loc[n['n_gap_up'] == 1, 'session']
    elif nifty_filter == 'nifty_bullish_b6':
        ok = n.loc[n['n_bullish_b6'] == 1, 'session']
    elif nifty_filter == 'nifty_strong_both':
        ok = n.loc[(n['n_gap_up'] == 1) & (n['n_bullish_b6'] == 1), 'session']
    elif nifty_filter == 'nifty_b3_change_pos':
        ok = n.loc[n['n_b3_change_pct'] > 0.1, 'session']
    else:
        return sig

    ok_set = set(pd.to_datetime(ok).values.astype('datetime64[ns]'))
    mask = df['session'].isin(list(ok_set))
    return sig & mask


# ---------------------------------------------------------------------------
# data load + cache
# ---------------------------------------------------------------------------

def build_cache(symbols, train_end=None) -> dict:
    cache = {}
    for i, sym in enumerate(symbols):
        try:
            df = load_5min(sym)
            if df.empty or len(df) < 1000:
                continue
            df = enrich(df, or_minutes=15)
            if train_end is not None:
                df = df[df.index <= pd.Timestamp(train_end)]
            cache[sym] = df
        except Exception as e:
            log.warning(f'  enrich fail {sym}: {e}')
        if (i + 1) % 50 == 0:
            log.info(f'  enriched {i+1}/{len(symbols)}')
    return cache


# ---------------------------------------------------------------------------
# Pass 1 - per-stock screen with default params
# ---------------------------------------------------------------------------

def pass1_screen(cache: dict, nifty: pd.DataFrame, default_params: dict,
                 default_exit: tuple) -> pd.DataFrame:
    """Run default-params signal on each stock, compute per-stock WR.

    Returns df with columns: symbol, n_trades, win_rate, profit_factor, avg_ret
    """
    rows = []
    for sym, df in cache.items():
        try:
            sig = trend_pullback_signal(df, **default_params)
            tp, sl, hold = default_exit
            rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold, direction='long')
            trades = simulate_signals(df, sig, rules)
            if len(trades) < 5:
                continue
            wr = (trades['ret_pct'] > 0).mean()
            pf_gp = trades.loc[trades['ret_pct'] > 0, 'ret_pct'].sum()
            pf_gl = -trades.loc[trades['ret_pct'] <= 0, 'ret_pct'].sum()
            pf = pf_gp / pf_gl if pf_gl > 0 else (np.inf if pf_gp > 0 else 0.0)
            rows.append(dict(
                symbol=sym,
                n_trades=int(len(trades)),
                win_rate=round(float(wr), 4),
                profit_factor=round(float(pf if np.isfinite(pf) else 999), 3),
                avg_ret=round(float(trades['ret_pct'].mean()), 4),
                total_ret=round(float(trades['ret_pct'].sum()), 3),
            ))
        except Exception as e:
            log.warning(f'  pass1 {sym} fail: {e}')
    df_out = pd.DataFrame(rows)
    return df_out.sort_values('win_rate', ascending=False) if len(df_out) else df_out


# ---------------------------------------------------------------------------
# Pass 2 - full sweep on diamond cohort
# ---------------------------------------------------------------------------

def pass2_sweep(cache: dict, nifty: pd.DataFrame, diamonds: list,
                csv_path: str, min_trades: int = 30,
                tag: str = 'full') -> int:
    """Run the full grid on `diamonds` only. Append to csv_path.

    Returns number of cells written.
    """
    PARAM_GRID = list(itertools.product(
        # bar_max
        [15, 24],
        # pullback_mode
        ['ema9_touch', 'vwap_within_0p3', 'ema9_or_vwap'],
        # rsi_floor
        [45, 50],
        # gap_min_pct
        [0.3, 0.5, 1.0],
        # nifty_filter
        ['none', 'nifty_gap_up', 'nifty_strong_both', 'nifty_b3_change_pos'],
    ))
    EXIT_GRID = [
        (0.5, 1.5, 60),
        (1.0, 1.5, 60),
        (0.7, 1.0, 48),
    ]

    fields = [
        'tag', 'family', 'params', 'tp_pct', 'sl_pct', 'max_hold_bars',
        'n_stocks', 'n_trades', 'win_rate', 'avg_win', 'avg_loss',
        'profit_factor', 'sharpe', 'max_dd_pct', 'total_return_pct',
        'trades_per_stock_year', 'aws', 'passes_gates',
    ]

    write_header = not os.path.exists(csv_path)
    cells_written = 0
    cell = 0
    n_total = len(PARAM_GRID) * len(EXIT_GRID)
    log.info(f'Pass 2 [tag={tag}]: {n_total} systems on {len(diamonds)} stocks')
    t_run = time.time()

    # filter cache to diamonds
    cohort = {s: cache[s] for s in diamonds if s in cache}
    log.info(f'  cohort cache size: {len(cohort)}')

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()

        for bar_max, pullback_mode, rsi_floor, gap_min, nifty_filter in PARAM_GRID:
            params = dict(
                bar_max=bar_max,
                pullback_mode=pullback_mode,
                rsi_floor=rsi_floor,
                gap_min_pct=gap_min,
                nifty_filter=nifty_filter,
            )
            for tp, sl, hold in EXIT_GRID:
                cell += 1
                all_trades = []
                stocks_with_trades = 0
                stock_sessions = 0
                for sym, df in cohort.items():
                    try:
                        sig = trend_pullback_signal(
                            df,
                            gap_min_pct=gap_min,
                            first_hour_strength_pct=0.5,
                            pullback_mode=pullback_mode,
                            rsi_floor=rsi_floor,
                            bar_min=7,
                            bar_max=bar_max,
                        )
                        sig = filter_by_nifty_long(sig, df, nifty, nifty_filter)
                        rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold, direction='long')
                        trades = simulate_signals(df, sig, rules)
                        if len(trades) == 0:
                            continue
                        trades['symbol'] = sym
                        all_trades.append(trades)
                        stocks_with_trades += 1
                        stock_sessions += df['session'].nunique()
                    except Exception as e:
                        log.warning(f'  cell {cell} {sym} fail: {e}')

                if not all_trades:
                    continue

                t = pd.concat(all_trades, ignore_index=True)
                avg_sessions = stock_sessions / max(stocks_with_trades, 1)
                stats = trade_stats(t, sessions=int(avg_sessions))
                trades_per_stock_year = stats['trades_per_year'] / max(stocks_with_trades, 1)
                passes = (
                    stats['n_trades'] >= min_trades
                    and stats['win_rate'] >= 0.75
                    and stats['profit_factor'] >= 2.0
                    and trades_per_stock_year >= 8
                )
                row = dict(
                    tag=tag,
                    family='trend_pullback_long',
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
                cells_written += 1

                if (stats['n_trades'] >= min_trades and stats['win_rate'] >= 0.70):
                    log.info(
                        f'  [{cell}/{n_total}] {tag} {pullback_mode} rsi={rsi_floor} '
                        f'gap={gap_min} nf={nifty_filter} bmax={bar_max} '
                        f'tp={tp} sl={sl} -> WR={stats["win_rate"]:.1%} '
                        f'n={stats["n_trades"]} PF={stats["profit_factor"]:.2f} '
                        f'DD={stats["max_dd_pct"]:.1f}% tpsy={trades_per_stock_year:.1f}'
                    )

                if cell % 50 == 0:
                    el = time.time() - t_run
                    log.info(f'  progress {cell}/{n_total} ({el:.0f}s elapsed)')

    log.info(f'Pass 2 [tag={tag}] done in {time.time()-t_run:.0f}s, {cells_written} cells written')
    return cells_written


# ---------------------------------------------------------------------------
# walk-forward validation
# ---------------------------------------------------------------------------

def evaluate_config(cache_train: dict, cache_test: dict, nifty: pd.DataFrame,
                    config: dict, exit_rule: tuple, diamonds: list,
                    min_trades: int = 20) -> dict:
    """Run a single config through both train + test caches; return train/test stats."""
    out = {}
    for label, cache in (('train', cache_train), ('test', cache_test)):
        all_trades = []
        stocks_with_trades = 0
        stock_sessions = 0
        for sym in diamonds:
            df = cache.get(sym)
            if df is None or df.empty:
                continue
            try:
                sig = trend_pullback_signal(
                    df,
                    gap_min_pct=config['gap_min_pct'],
                    first_hour_strength_pct=0.5,
                    pullback_mode=config['pullback_mode'],
                    rsi_floor=config['rsi_floor'],
                    bar_min=7,
                    bar_max=config['bar_max'],
                )
                sig = filter_by_nifty_long(sig, df, nifty, config['nifty_filter'])
                tp, sl, hold = exit_rule
                rules = TradeRules(tp_pct=tp, sl_pct=sl, max_hold_bars=hold, direction='long')
                trades = simulate_signals(df, sig, rules)
                if len(trades) == 0:
                    continue
                trades['symbol'] = sym
                all_trades.append(trades)
                stocks_with_trades += 1
                stock_sessions += df['session'].nunique()
            except Exception:
                pass
        if not all_trades:
            out[label] = dict(n_trades=0, win_rate=0.0, profit_factor=0.0,
                              max_dd_pct=0.0, sharpe=0.0,
                              total_return_pct=0.0,
                              trades_per_stock_year=0.0)
            continue
        t = pd.concat(all_trades, ignore_index=True)
        avg_sessions = stock_sessions / max(stocks_with_trades, 1)
        stats = trade_stats(t, sessions=int(avg_sessions))
        tpsy = stats['trades_per_year'] / max(stocks_with_trades, 1)
        out[label] = dict(
            n_trades=stats['n_trades'],
            win_rate=stats['win_rate'],
            profit_factor=stats['profit_factor'],
            max_dd_pct=stats['max_dd_pct'],
            sharpe=stats['sharpe'],
            total_return_pct=stats['total_return_pct'],
            trades_per_stock_year=tpsy,
        )
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0,
                    help='cap universe size (0=all)')
    ap.add_argument('--min-trades', type=int, default=30)
    ap.add_argument('--screen-only', action='store_true',
                    help='Run only pass 1 (per-stock screen)')
    ap.add_argument('--screen-wr', type=float, default=0.55,
                    help='WR threshold for diamond selection')
    ap.add_argument('--screen-min-trades', type=int, default=10,
                    help='min trades for stock to qualify in screen')
    ap.add_argument('--top-k', type=int, default=30,
                    help='max diamonds to keep')
    ap.add_argument('--use-existing-diamonds', action='store_true',
                    help='Skip pass 1, read diamonds from file')
    ap.add_argument('--walk-forward', action='store_true',
                    help='Run walk-forward on top systems')
    args = ap.parse_args()

    log.info('=' * 60)
    log.info('Stage 11b - LONG Trend-Pullback in confirmed-uptrend day')
    log.info('=' * 60)

    # NIFTY regime
    log.info('Building NIFTY regime ...')
    t0 = time.time()
    nifty = build_nifty_regime()
    log.info(f'  NIFTY: {len(nifty)} sessions ({time.time()-t0:.0f}s)')

    # If diamonds file exists and --use-existing-diamonds, skip universe load
    if args.use_existing_diamonds and os.path.exists(DIAMONDS_OUT):
        with open(DIAMONDS_OUT) as f:
            diamonds_preload = [s.strip() for s in f if s.strip()]
        log.info(f'Loaded {len(diamonds_preload)} existing diamonds (skipping universe screen)')
        log.info('Loading + enriching diamonds-only ...')
        t0 = time.time()
        cache = build_cache(diamonds_preload)
        log.info(f'  cached {len(cache)} diamond stocks ({time.time()-t0:.0f}s)')
    else:
        # universe
        log.info('Loading universe ...')
        universe = list_5min_universe(min_rows=20_000)
        if args.limit > 0:
            universe = universe[:args.limit]
        log.info(f'Universe size: {len(universe)}')

        # Cache full data
        log.info('Loading + enriching universe ...')
        t0 = time.time()
        cache = build_cache(universe)
        log.info(f'  cached {len(cache)} stocks ({time.time()-t0:.0f}s)')

    # ---- Pass 1: per-stock screen ----
    if args.use_existing_diamonds and os.path.exists(DIAMONDS_OUT):
        diamonds = diamonds_preload  # already loaded above
    else:
        log.info('Pass 1: per-stock screen (default params)')
        DEFAULT_PARAMS = dict(
            gap_min_pct=0.3,
            first_hour_strength_pct=0.5,
            pullback_mode='ema9_or_vwap',
            rsi_floor=45.0,
            bar_min=7,
            bar_max=30,
        )
        DEFAULT_EXIT = (0.5, 1.5, 60)  # tp, sl, hold

        t0 = time.time()
        screen_df = pass1_screen(cache, nifty, DEFAULT_PARAMS, DEFAULT_EXIT)
        log.info(f'  pass 1 done in {time.time()-t0:.0f}s, {len(screen_df)} stocks ranked')
        screen_df.to_csv(SCREEN_CSV, index=False)
        log.info(f'  screen csv -> {SCREEN_CSV}')

        # Select diamonds
        if len(screen_df):
            qual = screen_df[
                (screen_df['win_rate'] >= args.screen_wr)
                & (screen_df['n_trades'] >= args.screen_min_trades)
            ].copy()
            qual = qual.sort_values('win_rate', ascending=False).head(args.top_k)
            diamonds = qual['symbol'].tolist()
        else:
            diamonds = []

        with open(DIAMONDS_OUT, 'w') as f:
            f.write('\n'.join(diamonds))
        log.info(f'  selected {len(diamonds)} diamonds (WR >= {args.screen_wr}, n >= {args.screen_min_trades})')
        log.info(f'  diamonds: {diamonds}')
        log.info(f'  diamonds file -> {DIAMONDS_OUT}')

    if args.screen_only:
        log.info('Screen-only mode; exiting.')
        return 0

    if not diamonds:
        log.error('No diamonds qualified; cannot run pass 2.')
        return 1

    # ---- Pass 2: full sweep on diamonds ----
    # Wipe csv
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
    log.info(f'Pass 2: full sweep -> {OUT_CSV}')
    pass2_sweep(cache, nifty, diamonds, OUT_CSV, min_trades=args.min_trades, tag='full')

    # ---- Top-results summary ----
    if os.path.exists(OUT_CSV):
        try:
            df = pd.read_csv(OUT_CSV)
            df_filt = df[(df['n_trades'] >= args.min_trades)].copy()
            df_filt = df_filt.sort_values(['win_rate', 'profit_factor'], ascending=False)
            log.info('--- TOP 10 by WR (n>=%d) ---' % args.min_trades)
            for i, row in df_filt.head(10).iterrows():
                log.info(
                    f'  WR={row["win_rate"]:.3f} n={int(row["n_trades"])} '
                    f'PF={row["profit_factor"]:.2f} DD={row["max_dd_pct"]:.1f}% '
                    f'Sh={row["sharpe"]:.2f} tpsy={row["trades_per_stock_year"]:.1f} '
                    f'tp={row["tp_pct"]} sl={row["sl_pct"]} hold={int(row["max_hold_bars"])} '
                    f'{row["params"]}'
                )

            # candidates that pass all gates
            passers = df_filt[df_filt['passes_gates'] == 1]
            log.info(f'Systems passing all gates (WR>=75%, PF>=2, tpsy>=8, n>={args.min_trades}): {len(passers)}')
            for i, row in passers.head(10).iterrows():
                log.info(
                    f'  PASS: WR={row["win_rate"]:.3f} n={int(row["n_trades"])} '
                    f'PF={row["profit_factor"]:.2f} tpsy={row["trades_per_stock_year"]:.1f} '
                    f'tp={row["tp_pct"]} sl={row["sl_pct"]} {row["params"]}'
                )
        except Exception as e:
            log.error(f'summary fail: {e}\n{traceback.format_exc()}')

    # ---- Walk-forward on top candidates ----
    if args.walk_forward and os.path.exists(OUT_CSV):
        try:
            df = pd.read_csv(OUT_CSV)
            df_filt = df[df['n_trades'] >= args.min_trades].copy()
            df_filt = df_filt.sort_values(['win_rate', 'profit_factor'], ascending=False)
            top = df_filt.head(5)
            if not len(top):
                log.info('No candidates for walk-forward.')
                return 0

            log.info('--- WALK-FORWARD on top 5 ---')
            log.info('  Loading train window 2024-03-18 to 2025-09-30 ...')
            train_cache = {}
            for sym in diamonds:
                df_s = load_5min(sym, start='2024-03-18', end='2025-09-30')
                if df_s.empty:
                    continue
                train_cache[sym] = enrich(df_s, or_minutes=15)
            log.info(f'    train cache: {len(train_cache)}')

            log.info('  Loading test window 2025-10-01 to 2026-03-25 ...')
            test_cache = {}
            for sym in diamonds:
                df_s = load_5min(sym, start='2025-10-01', end='2026-03-25')
                if df_s.empty:
                    continue
                test_cache[sym] = enrich(df_s, or_minutes=15)
            log.info(f'    test cache: {len(test_cache)}')

            wf_path = os.path.join(RESULTS_DIR, '11b_walk_forward.csv')
            wf_fields = [
                'rank', 'params', 'tp_pct', 'sl_pct', 'max_hold_bars',
                'train_n', 'train_wr', 'train_pf', 'train_dd', 'train_sharpe',
                'test_n', 'test_wr', 'test_pf', 'test_dd', 'test_sharpe',
                'drift_wr', 'test_total_ret', 'test_tpsy',
            ]
            with open(wf_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=wf_fields)
                w.writeheader()

                for rank, (_, row) in enumerate(top.iterrows(), 1):
                    params_dict = eval(row['params'])
                    exit_rule = (row['tp_pct'], row['sl_pct'], row['max_hold_bars'])
                    res = evaluate_config(train_cache, test_cache, nifty,
                                          params_dict, exit_rule, diamonds,
                                          min_trades=20)
                    drift = res['test']['win_rate'] - res['train']['win_rate']
                    out_row = dict(
                        rank=rank,
                        params=str(params_dict),
                        tp_pct=row['tp_pct'], sl_pct=row['sl_pct'],
                        max_hold_bars=int(row['max_hold_bars']),
                        train_n=res['train']['n_trades'],
                        train_wr=round(res['train']['win_rate'], 4),
                        train_pf=round(res['train']['profit_factor'], 3),
                        train_dd=round(res['train']['max_dd_pct'], 3),
                        train_sharpe=round(res['train']['sharpe'], 3),
                        test_n=res['test']['n_trades'],
                        test_wr=round(res['test']['win_rate'], 4),
                        test_pf=round(res['test']['profit_factor'], 3),
                        test_dd=round(res['test']['max_dd_pct'], 3),
                        test_sharpe=round(res['test']['sharpe'], 3),
                        drift_wr=round(drift, 4),
                        test_total_ret=round(res['test']['total_return_pct'], 3),
                        test_tpsy=round(res['test']['trades_per_stock_year'], 2),
                    )
                    w.writerow(out_row)
                    f.flush()
                    log.info(
                        f'  WF rank={rank}: train WR={res["train"]["win_rate"]:.3f} '
                        f'(n={res["train"]["n_trades"]}) -> test WR={res["test"]["win_rate"]:.3f} '
                        f'(n={res["test"]["n_trades"]}) drift={drift:+.3f} '
                        f'PF train={res["train"]["profit_factor"]:.2f} test={res["test"]["profit_factor"]:.2f}'
                    )
            log.info(f'WF csv -> {wf_path}')
        except Exception as e:
            log.error(f'walk-forward fail: {e}\n{traceback.format_exc()}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
