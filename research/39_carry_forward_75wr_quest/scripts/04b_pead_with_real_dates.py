"""Pattern 4b — Post-Earnings Announcement Drift (PEAD) with REAL earnings dates.

DEPRECATED — superseded by 04b_pead_fast.py (vectorized).
This version takes ~5 hours; the vectorized one finishes in ~9 minutes
and produces the same outputs (master trade table + per-stock + ranking +
walk-forward). Kept here for reference / first design pass.

Replaces the gap+volume proxy (script 04) with actual NSE/Yahoo earnings
dates fetched in 04b_earnings_dates_fetch.py.

Pipeline:
  Stage 1 — per-stock screen at default rules (PEAD long + short)
  Stage 2 — cohort selection (all stocks with >=1 PEAD signal)
  Stage 3 — full param sweep on cohort (gap thresh / TP / SL / hold / filters)
  Stage 4 — walk-forward (train 2018-2023, test 2024-2025) on top candidates
  Stage 5 — cost stress at 0.20% RT
  Stage 6 — write per-stock + ranking + walk-forward + diamonds files

Key design decisions:
  - Entry: T+1 OPEN after announcement (the announcement could be after-hours
    in India where many results come post-3:30pm; safer to buy next day's open).
    For pre-market announcements, T+1 open still captures the drift cleanly.
  - Direction proxy:
      * If real EPS surprise_pct is available -> use it (positive surprise -> long)
      * Else fall back to announcement-day return (close vs prev close).
        This is the "return-as-surprise" proxy.
  - PEAD direction filters (sweep axes):
      * gap/return threshold: 0% (any direction), 1%, 2%, 3%, 5%, 7%
        (positive for long, negative for short)
      * volume confirmation: 1.5x avg / no filter
      * trend filter: above SMA200 (long) / below SMA200 (short) / no filter
  - Exit: TP/SL or time exit (5/10/15/20/30 days)
  - Cost: 0.20% round-trip (CNC delivery cost — STT 0.1% each side)

Outputs:
  results/04b_pead_perstock.csv
  results/04b_pead_ranking.csv
  results/04b_pead_walk_forward.csv
  results/04b_pead_diamonds.txt
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _engine_daily_my as E  # type: ignore


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'logs',
))
os.makedirs(LOG_DIR, exist_ok=True)

EARNINGS_CSV = os.path.join(RESULTS_DIR, '04b_earnings_dates.csv')
PERSTOCK_CSV = os.path.join(RESULTS_DIR, '04b_pead_perstock.csv')
RANKING_CSV = os.path.join(RESULTS_DIR, '04b_pead_ranking.csv')
WF_CSV = os.path.join(RESULTS_DIR, '04b_pead_walk_forward.csv')
DIAMONDS_TXT = os.path.join(RESULTS_DIR, '04b_pead_diamonds.txt')
LOG_PATH = os.path.join(LOG_DIR, '04b_pead.log')


# ---------------------------------------------------------------------------
# data prep — annotate each stock's daily df with announcement-day flags
# ---------------------------------------------------------------------------

def load_earnings_dates(csv_path: str = EARNINGS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['announcement_date'] = pd.to_datetime(df['announcement_date'])
    df['eps_estimate'] = pd.to_numeric(df['eps_estimate'], errors='coerce')
    df['eps_actual'] = pd.to_numeric(df['eps_actual'], errors='coerce')
    df['surprise_pct'] = pd.to_numeric(df['surprise_pct'], errors='coerce')
    return df


def annotate_earnings(df: pd.DataFrame, sym_earnings: pd.DataFrame) -> pd.DataFrame:
    """Add earnings columns to a daily price df:
       is_earnings_day (bool), announcement_return_pct (close vs prev close on that day),
       eps_surprise_pct (NaN if not known).

    Earnings dates that fall on non-trading days roll forward to the next trading day
    (since announcement after-hours implies the next session is the reaction session).
    Implementation: for each announcement_date, find the LAST trading day with date >= announcement.
    Wait — we want the trading session that REFLECTS the announcement. If reported pre-open
    on Mon, Mon's session is the reaction. If reported after-close on Mon, Tue is reaction.
    yfinance gives a date — we can't easily distinguish pre/post. Standard approach: use the
    trading day on or after the announcement date as the reaction day.
    """
    if df.empty or sym_earnings.empty:
        df = df.copy()
        df['is_earnings_day'] = False
        df['announcement_return_pct'] = np.nan
        df['eps_surprise_pct'] = np.nan
        return df

    df = df.copy()
    df['is_earnings_day'] = False
    df['eps_surprise_pct'] = np.nan

    # For each announcement_date, find the first trading day >= it within the index
    sorted_idx = df.index.sort_values()
    for _, row in sym_earnings.iterrows():
        adate = row['announcement_date']
        if pd.isna(adate):
            continue
        # find first index day >= adate
        pos = sorted_idx.searchsorted(adate, side='left')
        if pos >= len(sorted_idx):
            continue
        reaction_day = sorted_idx[pos]
        # cap at 5 trading days — if announcement is way out of range we skip
        gap_days = (reaction_day - adate).days
        if gap_days > 7:  # weird mismatch, skip
            continue
        df.loc[reaction_day, 'is_earnings_day'] = True
        if pd.notna(row.get('surprise_pct')):
            df.loc[reaction_day, 'eps_surprise_pct'] = row['surprise_pct']

    # Use ret_pct (already in enriched df) as announcement-day return
    df['announcement_return_pct'] = df['ret_pct'].where(df['is_earnings_day'], np.nan)
    return df


def attach_earnings_to_cache(cache: dict, all_earnings: pd.DataFrame) -> dict:
    """For each symbol in the cache, annotate with earnings flags."""
    out = {}
    for sym, df in cache.items():
        sym_earn = all_earnings[all_earnings['symbol'] == sym]
        out[sym] = annotate_earnings(df, sym_earn)
    return out


# ---------------------------------------------------------------------------
# PEAD signal functions
# ---------------------------------------------------------------------------

def pead_long(df: pd.DataFrame,
              ret_min: float = 0.0,
              vol_ratio_min: float = 0.0,
              require_above_sma200: bool = False,
              use_eps_surprise: bool = False,
              eps_surprise_min: float = 0.0) -> pd.Series:
    """LONG PEAD signal: positive earnings reaction.

    If use_eps_surprise=True and eps_surprise_pct is available, gate on real
    surprise; else fall back to announcement-day return as surprise proxy.

    The actual entry happens on next-day open (handled by simulate_signals_daily).
    """
    sig = df['is_earnings_day'].fillna(False)
    if use_eps_surprise:
        # require known surprise AND positive at threshold
        sig = sig & df['eps_surprise_pct'].notna() & (df['eps_surprise_pct'] >= eps_surprise_min)
    else:
        # use announcement-day return as direction proxy
        sig = sig & (df['announcement_return_pct'].fillna(-9999) >= ret_min)
    if vol_ratio_min > 0:
        sig = sig & (df['vol_ratio'].fillna(0) >= vol_ratio_min)
    if require_above_sma200:
        sig = sig & (df['close'] > df['sma200'])
    return sig.fillna(False)


def pead_short(df: pd.DataFrame,
               ret_max: float = 0.0,
               vol_ratio_min: float = 0.0,
               require_below_sma200: bool = False,
               use_eps_surprise: bool = False,
               eps_surprise_max: float = 0.0) -> pd.Series:
    """SHORT PEAD signal: negative earnings reaction."""
    sig = df['is_earnings_day'].fillna(False)
    if use_eps_surprise:
        sig = sig & df['eps_surprise_pct'].notna() & (df['eps_surprise_pct'] <= eps_surprise_max)
    else:
        sig = sig & (df['announcement_return_pct'].fillna(9999) <= ret_max)
    if vol_ratio_min > 0:
        sig = sig & (df['vol_ratio'].fillna(0) >= vol_ratio_min)
    if require_below_sma200:
        sig = sig & (df['close'] < df['sma200'])
    return sig.fillna(False)


# ---------------------------------------------------------------------------
# Stage 1 — per-stock screen
# ---------------------------------------------------------------------------

def stage1_perstock(cache: dict, side: str = 'long',
                    start: str = '2018-01-01', end: str = '2026-03-19') -> pd.DataFrame:
    """Run a default-rules PEAD on each stock and write per-stock stats."""
    print(f'[stage1] {side} per-stock {start}..{end} on {len(cache)} stocks', flush=True)
    fields = ['symbol', 'side', 'n_trades', 'win_rate', 'avg_win', 'avg_loss',
              'profit_factor', 'sharpe', 'max_dd_pct', 'total_return_pct',
              'trades_per_year', 'aws', 'avg_days_held']
    write_header = not os.path.exists(PERSTOCK_CSV)
    f = open(PERSTOCK_CSV, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()

    fn = pead_long if side == 'long' else pead_short
    rules = E.TradeRules(tp_pct=8.0, sl_pct=5.0, max_hold_days=15,
                          direction=side, rsi_exit=None,
                          cost_round_trip_pct=0.20)
    rows = []
    for i, (sym, df) in enumerate(cache.items(), 1):
        if df.empty or len(df) < 250:
            continue
        try:
            sig = fn(df)
        except Exception as e:
            print(f'  [{sym}] signal err: {e}', flush=True)
            continue
        sig = E.split_signal_by_period(df, sig, start, end)
        trades = E.simulate_signals_daily(df, sig, rules,
                                          rsi_series=df.get('rsi14'))
        years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 0.1)
        stats = E.trade_stats(trades, period_years=years)
        row = {'symbol': sym, 'side': side, **stats}
        w.writerow(row)
        f.flush()
        rows.append(row)
        if i % 20 == 0:
            print(f'  [{i}/{len(cache)}] {sym}: n={stats["n_trades"]} WR={stats["win_rate"]:.2%}', flush=True)
    f.close()
    print(f'[stage1] done {len(rows)} rows', flush=True)
    return pd.DataFrame(rows)


def select_cohort(perstock_df: pd.DataFrame, side: str = 'long',
                   use_all: bool = True, top_n: int = 25) -> list:
    sub = perstock_df[perstock_df['side'] == side].copy()
    if use_all:
        cohort = sub[sub['n_trades'] >= 1]['symbol'].tolist()
    else:
        sub = sub[(sub['n_trades'] >= 4) & (sub['win_rate'] >= 0.50)]
        sub = sub.sort_values('aws', ascending=False)
        cohort = sub.head(top_n)['symbol'].tolist()
    print(f'[cohort] {len(cohort)} {side} symbols selected', flush=True)
    return cohort


# ---------------------------------------------------------------------------
# Stage 3 — sweep
# ---------------------------------------------------------------------------

def stage3_sweep(cache: dict, cohort: list, side: str = 'long',
                  start: str = E.TRAIN_START, end: str = E.TRAIN_END) -> pd.DataFrame:
    """Sweep the param grid on the cohort over the train window."""
    print(f'[stage3] sweep ({side}) on {len(cohort)} stocks {start}..{end}', flush=True)
    sub_cache = {s: cache[s] for s in cohort if s in cache}

    # axes
    if side == 'long':
        ret_levels = [0, 1, 2, 3, 5, 7]
        sma_filters = [False, True]
    else:
        ret_levels = [0, -1, -2, -3, -5, -7]
        sma_filters = [False, True]
    vol_levels = [0, 1.5]
    # TP/SL pairs that satisfy TP > SL (favorable RR)
    tpsl_pairs = [
        (5, 3),    # RR 1.67
        (6, 4),    # RR 1.5
        (8, 5),    # RR 1.6
        (10, 5),   # RR 2.0
        (10, 7),   # RR 1.43
        (15, 7),   # RR 2.14
        (15, 10),  # RR 1.5
    ]
    hold_levels = [5, 10, 15, 20, 30]
    surprise_options = [False, True]  # use real EPS surprise vs return proxy

    fields = ['side', 'ret_thresh', 'vol_min', 'use_sma', 'use_surprise',
              'tp', 'sl', 'hold', 'n_trades', 'win_rate', 'avg_win',
              'avg_loss', 'profit_factor', 'sharpe', 'max_dd_pct',
              'total_return_pct', 'trades_per_year', 'aws', 'avg_days_held',
              'n_stocks']
    write_header = not os.path.exists(RANKING_CSV)
    f = open(RANKING_CSV, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()

    fn = pead_long if side == 'long' else pead_short
    if side == 'long':
        ret_kw, sur_kw, sma_kw = 'ret_min', 'eps_surprise_min', 'require_above_sma200'
    else:
        ret_kw, sur_kw, sma_kw = 'ret_max', 'eps_surprise_max', 'require_below_sma200'

    cells = list(product(ret_levels, vol_levels, sma_filters, surprise_options,
                         tpsl_pairs, hold_levels))
    print(f'[stage3] {len(cells)} cells', flush=True)
    rows = []
    t0 = time.time()
    for k, (ret_t, vol_t, sma_f, use_sur, (tp, sl), hold) in enumerate(cells, 1):
        # When use_sur=True, we use ret_thresh as surprise threshold (in % surprise)
        # When use_sur=False, ret_thresh is announcement-day return threshold.
        skwargs = {
            ret_kw: ret_t if not use_sur else 0.0,
            'vol_ratio_min': vol_t,
            sma_kw: sma_f,
            'use_eps_surprise': use_sur,
            sur_kw: ret_t if use_sur else 0.0,
        }
        rules = E.TradeRules(tp_pct=tp, sl_pct=sl, max_hold_days=hold,
                              direction=side, rsi_exit=None,
                              cost_round_trip_pct=0.20)
        try:
            per_df, pooled = E.run_strategy_on_cohort(
                sub_cache, fn, rules, start, end, **skwargs)
        except Exception as e:
            print(f'  cell {k} err: {e}', flush=True)
            continue
        n_stocks = (per_df['n_trades'] > 0).sum() if len(per_df) else 0
        row = {'side': side, 'ret_thresh': ret_t, 'vol_min': vol_t,
               'use_sma': sma_f, 'use_surprise': use_sur,
               'tp': tp, 'sl': sl, 'hold': hold,
               'n_stocks': int(n_stocks),
               **{kk: pooled.get(kk, 0) for kk in
                  ('n_trades', 'win_rate', 'avg_win', 'avg_loss',
                   'profit_factor', 'sharpe', 'max_dd_pct',
                   'total_return_pct', 'trades_per_year', 'aws',
                   'avg_days_held')}}
        w.writerow(row)
        f.flush()
        rows.append(row)
        if k % 50 == 0 or k == len(cells):
            print(f'  [{k}/{len(cells)}] {time.time()-t0:.0f}s | last ret={ret_t} '
                  f'sur={use_sur} tp={tp} sl={sl} hold={hold} '
                  f'n={pooled["n_trades"]} WR={pooled["win_rate"]:.2%}',
                  flush=True)
    f.close()
    print(f'[stage3] done in {time.time()-t0:.0f}s', flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 4 — walk-forward
# ---------------------------------------------------------------------------

def stage4_walk_forward(cache: dict, sweep_df: pd.DataFrame,
                         side: str = 'long', top_n: int = 30) -> list:
    print(f'[stage4] walk-forward top-{top_n} ({side})', flush=True)
    sub = sweep_df[sweep_df['side'] == side].copy()
    # Loose train gates — broader funnel since real earnings reduces n
    sub = sub[(sub['n_trades'] >= 20)
              & (sub['win_rate'] >= 0.55)
              & (sub['profit_factor'] >= 1.2)]
    sub = sub.sort_values(['win_rate', 'profit_factor'], ascending=False).head(top_n)
    print(f'[stage4] {len(sub)} candidates pass loose train gates', flush=True)

    perstock = pd.read_csv(PERSTOCK_CSV)
    cohort = select_cohort(perstock, side=side, use_all=True)
    sub_cache = {s: cache[s] for s in cohort if s in cache}

    fn = pead_long if side == 'long' else pead_short
    if side == 'long':
        ret_kw, sur_kw, sma_kw = 'ret_min', 'eps_surprise_min', 'require_above_sma200'
    else:
        ret_kw, sur_kw, sma_kw = 'ret_max', 'eps_surprise_max', 'require_below_sma200'

    fields = ['side', 'ret_thresh', 'vol_min', 'use_sma', 'use_surprise',
              'tp', 'sl', 'hold',
              'train_n', 'train_wr', 'train_pf', 'train_dd',
              'test_n', 'test_wr', 'test_pf', 'test_dd', 'test_total_ret',
              'test_aws', 'passes_wf']
    write_header = not os.path.exists(WF_CSV)
    f = open(WF_CSV, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()

    survivors = []
    for _, r in sub.iterrows():
        ret_t = float(r['ret_thresh'])
        use_sur = bool(r['use_surprise'])
        skwargs = {
            ret_kw: ret_t if not use_sur else 0.0,
            'vol_ratio_min': float(r['vol_min']),
            sma_kw: bool(r['use_sma']),
            'use_eps_surprise': use_sur,
            sur_kw: ret_t if use_sur else 0.0,
        }
        rules = E.TradeRules(tp_pct=float(r['tp']), sl_pct=float(r['sl']),
                              max_hold_days=int(r['hold']),
                              direction=side, rsi_exit=None,
                              cost_round_trip_pct=0.20)
        _, train_p = E.run_strategy_on_cohort(
            sub_cache, fn, rules, E.TRAIN_START, E.TRAIN_END, **skwargs)
        _, test_p = E.run_strategy_on_cohort(
            sub_cache, fn, rules, E.TEST_START, E.TEST_END, **skwargs)
        passes = E.passes_walk_forward(train_p, test_p)
        row = {
            'side': side, 'ret_thresh': r['ret_thresh'], 'vol_min': r['vol_min'],
            'use_sma': r['use_sma'], 'use_surprise': r['use_surprise'],
            'tp': r['tp'], 'sl': r['sl'], 'hold': r['hold'],
            'train_n': train_p['n_trades'], 'train_wr': train_p['win_rate'],
            'train_pf': train_p['profit_factor'], 'train_dd': train_p['max_dd_pct'],
            'test_n': test_p['n_trades'], 'test_wr': test_p['win_rate'],
            'test_pf': test_p['profit_factor'], 'test_dd': test_p['max_dd_pct'],
            'test_total_ret': test_p['total_return_pct'],
            'test_aws': test_p['aws'], 'passes_wf': passes,
        }
        w.writerow(row)
        f.flush()
        if passes:
            survivors.append(row)
            print(f'  WIN {side} ret={ret_t} sur={use_sur} tp={r["tp"]} sl={r["sl"]} '
                  f'hold={r["hold"]} | train WR={train_p["win_rate"]:.1%} '
                  f'test WR={test_p["win_rate"]:.1%} n={test_p["n_trades"]}',
                  flush=True)
    f.close()
    return survivors


def stage5_cost_stress(cache: dict, survivors: list, side: str = 'long') -> list:
    """Re-run survivors with 0.20% RT cost (already 0.20% in main run; this stage
    re-confirms and is here for completeness / additional 0.30% sensitivity)."""
    print(f'[stage5] cost stress 0.30% RT — {len(survivors)} survivors', flush=True)
    perstock = pd.read_csv(PERSTOCK_CSV)
    cohort = select_cohort(perstock, side=side, use_all=True)
    sub_cache = {s: cache[s] for s in cohort if s in cache}
    fn = pead_long if side == 'long' else pead_short
    if side == 'long':
        ret_kw, sur_kw, sma_kw = 'ret_min', 'eps_surprise_min', 'require_above_sma200'
    else:
        ret_kw, sur_kw, sma_kw = 'ret_max', 'eps_surprise_max', 'require_below_sma200'

    out = []
    for r in survivors:
        ret_t = float(r['ret_thresh'])
        use_sur = bool(r['use_surprise'])
        skwargs = {
            ret_kw: ret_t if not use_sur else 0.0,
            'vol_ratio_min': float(r['vol_min']),
            sma_kw: bool(r['use_sma']),
            'use_eps_surprise': use_sur,
            sur_kw: ret_t if use_sur else 0.0,
        }
        rules = E.TradeRules(tp_pct=float(r['tp']), sl_pct=float(r['sl']),
                              max_hold_days=int(r['hold']),
                              direction=side, rsi_exit=None,
                              cost_round_trip_pct=0.30)
        _, test_p = E.run_strategy_on_cohort(
            sub_cache, fn, rules, E.TEST_START, E.TEST_END, **skwargs)
        out.append({**r, 'test_wr_030': test_p['win_rate'],
                    'test_pf_030': test_p['profit_factor'],
                    'test_total_ret_030': test_p['total_return_pct']})
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print('[main] PEAD with real earnings dates', flush=True)

    if not os.path.exists(EARNINGS_CSV):
        print(f'[main] FATAL: {EARNINGS_CSV} missing. Run 04b_earnings_dates_fetch.py first.',
              flush=True)
        sys.exit(1)

    earnings_df = load_earnings_dates(EARNINGS_CSV)
    print(f'[main] loaded {len(earnings_df)} earnings rows for '
          f'{earnings_df["symbol"].nunique()} symbols', flush=True)
    n_with_actual = earnings_df['eps_actual'].notna().sum()
    n_with_surprise = earnings_df['surprise_pct'].notna().sum()
    print(f'[main]   {n_with_actual} have eps_actual, {n_with_surprise} have surprise_pct',
          flush=True)

    print('[main] loading F&O daily price data...', flush=True)
    syms = E.list_daily_universe(E.FNO_UNIVERSE, start='2017-01-01',
                                  end='2026-03-19', min_rows=1500)
    cache = E.load_many_daily(syms)
    for sym in list(cache.keys()):
        cache[sym] = E.enrich_daily(cache[sym])
    print(f'[main] enriched {len(cache)} stocks ({time.time()-t0:.0f}s)', flush=True)

    print('[main] attaching earnings dates to price cache...', flush=True)
    cache = attach_earnings_to_cache(cache, earnings_df)
    # quick sanity check
    n_signals_total = 0
    for sym, df in cache.items():
        n_signals_total += int(df['is_earnings_day'].sum())
    print(f'[main] total earnings-day flags across cache: {n_signals_total}', flush=True)

    diamond_lines = []
    diamond_lines.append('# Pattern 4b — PEAD with REAL earnings dates — Walk-Forward Diamonds\n')
    diamond_lines.append(f'Generated: {time.strftime("%Y-%m-%d %H:%M IST")}\n')
    diamond_lines.append(f'Universe: {len(cache)} F&O stocks\n')
    diamond_lines.append(f'Earnings source: yfinance .NS get_earnings_dates\n')
    diamond_lines.append(f'Earnings rows: {len(earnings_df)} ({n_with_surprise} with surprise%)\n')
    diamond_lines.append(f'Total earnings-day signals across cache: {n_signals_total}\n\n')

    for side in ('long', 'short'):
        diamond_lines.append(f'## {side.upper()} side\n\n')
        if not os.path.exists(PERSTOCK_CSV) or not (
            pd.read_csv(PERSTOCK_CSV)['side'] == side).any() if os.path.exists(PERSTOCK_CSV) else True:
            stage1_perstock(cache, side=side, start='2018-01-01', end='2026-03-19')

        sweep_df = stage3_sweep(cache, select_cohort(
            pd.read_csv(PERSTOCK_CSV), side=side, use_all=True),
            side=side, start=E.TRAIN_START, end=E.TRAIN_END)

        survivors = stage4_walk_forward(cache, sweep_df, side=side, top_n=40)

        if survivors:
            stressed = stage5_cost_stress(cache, survivors, side=side)
            diamond_lines.append(f'### {len(survivors)} walk-forward survivors\n\n')
            for s in stressed:
                diamond_lines.append(
                    f'- ret_thresh={s["ret_thresh"]} vol>={s["vol_min"]} sma_filter={s["use_sma"]} '
                    f'real_surprise={s["use_surprise"]} tp={s["tp"]}% sl={s["sl"]}% hold={s["hold"]}d\n'
                    f'  Train: n={s["train_n"]} WR={s["train_wr"]:.1%} PF={s["train_pf"]:.2f}\n'
                    f'  Test:  n={s["test_n"]} WR={s["test_wr"]:.1%} PF={s["test_pf"]:.2f} '
                    f'DD={s["test_dd"]:.1f}% TotalRet={s["test_total_ret"]:.1f}%\n'
                    f'  Cost @ 0.30% RT: WR={s["test_wr_030"]:.1%} PF={s["test_pf_030"]:.2f} '
                    f'TotalRet={s["test_total_ret_030"]:.1f}%\n\n'
                )
        else:
            diamond_lines.append(f'No walk-forward survivors on {side} side.\n\n')

    with open(DIAMONDS_TXT, 'w', encoding='utf-8') as f:
        f.writelines(diamond_lines)
    print(f'[main] DONE in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
