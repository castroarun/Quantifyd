"""Pattern 4 — Earnings post-drift continuation (5-20 day hold).

Hypothesis: Post-Earnings Announcement Drift (PEAD) — stocks with positive
earnings surprises drift up over the following weeks; negative surprises
drift down.

Indian-data complication: market_data_unified has NO earnings table. We
proxy earnings days as: gap_pct >= 4% (or <= -4%) on a day with vol_ratio
>= 2.0, falling within Indian quarterly results months
(Jan, Feb, Apr, May, Jul, Aug, Oct, Nov).

Once we have a candidate earnings day:
  - LONG signal: positive gap+ (gap >= 4% AND ret_pct >= 0) on a results
    month with above-average volume.
  - SHORT signal: negative gap+ (gap <= -4% AND ret_pct <= 0).

Entry: signal-day NEXT bar's open.
Exit: TP/SL or time-stop 10/15/20 days.

Pipeline mirrors pattern 3 — per-stock screen, cohort, sweep, walk-forward,
cost stress.

Outputs:
    results/04_earnings_perstock.csv
    results/04_earnings_ranking.csv
    results/04_earnings_walk_forward.csv
    results/04_earnings_diamonds.txt
"""

from __future__ import annotations

import csv
import os
import sys
import time
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

PERSTOCK_CSV = os.path.join(RESULTS_DIR, '04_earnings_perstock.csv')
RANKING_CSV = os.path.join(RESULTS_DIR, '04_earnings_ranking.csv')
WF_CSV = os.path.join(RESULTS_DIR, '04_earnings_walk_forward.csv')
DIAMONDS_TXT = os.path.join(RESULTS_DIR, '04_earnings_diamonds.txt')

# Indian quarterly results months — when companies typically report
RESULTS_MONTHS = [1, 2, 4, 5, 7, 8, 10, 11]


# ---------------------------------------------------------------------------
# strategy
# ---------------------------------------------------------------------------

def earnings_long(df: pd.DataFrame,
                  gap_pct_min: float = 4.0,
                  vol_ratio_min: float = 2.0,
                  same_day_ret_min: float = 0.0,
                  results_months_only: bool = True) -> pd.Series:
    """LONG: positive earnings-day gap + price holds (close >= open)."""
    sig = (df['gap_pct'] >= gap_pct_min)
    sig &= (df['vol_ratio'] >= vol_ratio_min)
    sig &= (df['ret_pct'] >= same_day_ret_min)
    if results_months_only:
        sig &= df.index.month.isin(RESULTS_MONTHS)
    sig &= df['vol_avg20'].notna()
    return sig.fillna(False)


def earnings_short(df: pd.DataFrame,
                   gap_pct_max: float = -4.0,
                   vol_ratio_min: float = 2.0,
                   same_day_ret_max: float = 0.0,
                   results_months_only: bool = True) -> pd.Series:
    """SHORT: negative earnings-day gap + price holds (close <= open)."""
    sig = (df['gap_pct'] <= gap_pct_max)
    sig &= (df['vol_ratio'] >= vol_ratio_min)
    sig &= (df['ret_pct'] <= same_day_ret_max)
    if results_months_only:
        sig &= df.index.month.isin(RESULTS_MONTHS)
    sig &= df['vol_avg20'].notna()
    return sig.fillna(False)


# ---------------------------------------------------------------------------
# Stage 1 — per-stock
# ---------------------------------------------------------------------------

def stage1_perstock(cache, default_rules, side='long', start='2018-01-01', end='2026-03-19'):
    print(f'[stage1] earnings drift per-stock ({side}) {start}..{end} on {len(cache)} stocks',
          flush=True)
    fn = earnings_long if side == 'long' else earnings_short
    fields = ['symbol','side','n_trades','win_rate','avg_win','avg_loss',
              'profit_factor','sharpe','max_dd_pct','total_return_pct',
              'trades_per_year','aws','avg_days_held']
    write_header = not os.path.exists(PERSTOCK_CSV)
    f = open(PERSTOCK_CSV, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()
    rows = []
    for i, (sym, df) in enumerate(cache.items(), 1):
        if df.empty or len(df) < 250:
            continue
        sig = fn(df)
        sig = E.split_signal_by_period(df, sig, start, end)
        trades = E.simulate_signals_daily(df, sig, default_rules,
                                           rsi_series=df['rsi14'])
        years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 0.1)
        stats = E.trade_stats(trades, period_years=years)
        row = {'symbol': sym, 'side': side, **stats}
        w.writerow(row)
        f.flush()
        rows.append(row)
        if i % 25 == 0:
            print(f'  [{i}/{len(cache)}] {sym}: n={stats["n_trades"]} '
                  f'WR={stats["win_rate"]:.2%}', flush=True)
    f.close()
    print(f'[stage1] done {len(rows)} rows', flush=True)
    return pd.DataFrame(rows)


def select_cohort(perstock_df, side='long', top_n=20,
                  use_all_with_signal: bool = True):
    sub = perstock_df[perstock_df['side'] == side].copy()
    if use_all_with_signal:
        cohort = sub[sub['n_trades'] >= 1]['symbol'].tolist()
    else:
        sub = sub[(sub['n_trades'] >= 4) & (sub['win_rate'] >= 0.50)]
        sub = sub.sort_values('aws', ascending=False)
        cohort = sub.head(top_n)['symbol'].tolist()
    print(f'[cohort] selected {len(cohort)} {side} symbols', flush=True)
    return cohort


# ---------------------------------------------------------------------------
# Stage 3 — sweep
# ---------------------------------------------------------------------------

def stage3_sweep(cache, cohort, side='long', start=E.TRAIN_START, end=E.TRAIN_END):
    print(f'[stage3] sweep ({side}) on {len(cohort)} stocks', flush=True)
    sub_cache = {s: cache[s] for s in cohort if s in cache}

    gap_levels = [3, 4, 5, 7] if side == 'long' else [-3, -4, -5, -7]
    vol_levels = [1.5, 2.0, 3.0]
    tp_levels = [4, 6, 8, 12]
    sl_levels = [3, 5, 7]
    hold_levels = [10, 15, 20]
    months_options = [True, False]

    fields = ['side','gap','vol','months_only','tp','sl','hold',
              'n_trades','win_rate','avg_win','avg_loss','profit_factor',
              'sharpe','max_dd_pct','total_return_pct','trades_per_year',
              'aws','avg_days_held','n_stocks']
    write_header = not os.path.exists(RANKING_CSV)
    f = open(RANKING_CSV, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()

    fn = earnings_long if side == 'long' else earnings_short
    gap_kw = 'gap_pct_min' if side == 'long' else 'gap_pct_max'
    ret_kw = 'same_day_ret_min' if side == 'long' else 'same_day_ret_max'

    cells = list(product(gap_levels, vol_levels, tp_levels, sl_levels,
                         hold_levels, months_options))
    print(f'[stage3] {len(cells)} cells', flush=True)
    rows = []
    t0 = time.time()
    for k, (gap, vol, tp, sl, hold, months) in enumerate(cells, 1):
        if tp < sl + 1:
            continue
        skwargs = {gap_kw: gap, 'vol_ratio_min': vol, ret_kw: 0.0,
                   'results_months_only': months}
        rules = E.TradeRules(tp_pct=tp, sl_pct=sl, max_hold_days=hold,
                             direction=side, rsi_exit=None,
                             cost_round_trip_pct=0.06)
        per_df, pooled = E.run_strategy_on_cohort(
            sub_cache, fn, rules, start, end, **skwargs)
        n_stocks = (per_df['n_trades'] > 0).sum() if len(per_df) else 0
        row = {'side': side, 'gap': gap, 'vol': vol, 'months_only': months,
               'tp': tp, 'sl': sl, 'hold': hold, 'n_stocks': int(n_stocks),
               **pooled}
        w.writerow(row)
        f.flush()
        rows.append(row)
        if k % 50 == 0 or k == len(cells):
            print(f'  [{k}/{len(cells)}] {time.time()-t0:.0f}s | '
                  f'last gap={gap} vol={vol} tp={tp} sl={sl} hold={hold} '
                  f'n={pooled["n_trades"]} WR={pooled["win_rate"]:.2%}',
                  flush=True)
    f.close()
    print(f'[stage3] done in {time.time()-t0:.0f}s', flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 4 — walk-forward
# ---------------------------------------------------------------------------

def stage4_walk_forward(cache, sweep_df, side='long', top_n=25):
    print(f'[stage4] walk-forward top-{top_n} ({side})', flush=True)
    sub = sweep_df[sweep_df['side'] == side].copy()
    sub = sub[(sub['n_trades'] >= 30)
             & (sub['win_rate'] >= 0.60)
             & (sub['profit_factor'] >= 1.3)]
    sub = sub.sort_values('aws', ascending=False).head(top_n)
    print(f'[stage4] {len(sub)} candidates pass loose train gates', flush=True)

    perstock = pd.read_csv(PERSTOCK_CSV)
    cohort = select_cohort(perstock, side=side, top_n=15)
    sub_cache = {s: cache[s] for s in cohort if s in cache}

    fn = earnings_long if side == 'long' else earnings_short
    gap_kw = 'gap_pct_min' if side == 'long' else 'gap_pct_max'

    fields = ['side','gap','vol','months_only','tp','sl','hold',
              'train_n','train_wr','train_pf','train_dd',
              'test_n','test_wr','test_pf','test_dd','test_total_ret',
              'test_aws','passes_wf']
    write_header = not os.path.exists(WF_CSV)
    f = open(WF_CSV, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()

    survivors = []
    for _, r in sub.iterrows():
        skwargs = {gap_kw: float(r['gap']),
                   'vol_ratio_min': float(r['vol']),
                   'results_months_only': bool(r['months_only'])}
        rules = E.TradeRules(tp_pct=float(r['tp']), sl_pct=float(r['sl']),
                             max_hold_days=int(r['hold']),
                             direction=side, rsi_exit=None,
                             cost_round_trip_pct=0.06)
        _, train_p = E.run_strategy_on_cohort(
            sub_cache, fn, rules, E.TRAIN_START, E.TRAIN_END, **skwargs)
        _, test_p = E.run_strategy_on_cohort(
            sub_cache, fn, rules, E.TEST_START, E.TEST_END, **skwargs)
        passes = E.passes_walk_forward(train_p, test_p)
        row = {
            'side': side, 'gap': r['gap'], 'vol': r['vol'],
            'months_only': r['months_only'],
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
            print(f'  WIN {side} gap={r["gap"]} tp={r["tp"]} sl={r["sl"]} hold={r["hold"]} '
                  f'| train WR={train_p["win_rate"]:.1%} test WR={test_p["win_rate"]:.1%}',
                  flush=True)
    f.close()
    return survivors


def stage5_cost_stress(cache, survivors, side='long'):
    print(f'[stage5] cost stress 0.20% RT — {len(survivors)} survivors', flush=True)
    perstock = pd.read_csv(PERSTOCK_CSV)
    cohort = select_cohort(perstock, side=side, top_n=15)
    sub_cache = {s: cache[s] for s in cohort if s in cache}
    fn = earnings_long if side == 'long' else earnings_short
    gap_kw = 'gap_pct_min' if side == 'long' else 'gap_pct_max'
    out = []
    for r in survivors:
        skwargs = {gap_kw: float(r['gap']),
                   'vol_ratio_min': float(r['vol']),
                   'results_months_only': bool(r['months_only'])}
        rules = E.TradeRules(tp_pct=float(r['tp']), sl_pct=float(r['sl']),
                             max_hold_days=int(r['hold']),
                             direction=side, rsi_exit=None,
                             cost_round_trip_pct=0.20)
        _, test_p = E.run_strategy_on_cohort(
            sub_cache, fn, rules, E.TEST_START, E.TEST_END, **skwargs)
        out.append({**r, 'test_wr_020': test_p['win_rate'],
                    'test_pf_020': test_p['profit_factor'],
                    'test_total_ret_020': test_p['total_return_pct']})
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print('[main] loading F&O daily data...', flush=True)
    symbols = E.list_daily_universe(E.FNO_UNIVERSE, start='2017-01-01',
                                    end='2026-03-19', min_rows=1500)
    print(f'[main] {len(symbols)} F&O symbols', flush=True)
    cache = E.load_many_daily(symbols)
    for sym in list(cache.keys()):
        cache[sym] = E.enrich_daily(cache[sym])
    print(f'[main] enriched {len(cache)} stocks ({time.time()-t0:.0f}s)', flush=True)

    diamond_lines = []
    diamond_lines.append('# Pattern 4 — Earnings Post-Drift — Walk-Forward Diamonds\n')
    diamond_lines.append(f'Generated: {time.strftime("%Y-%m-%d %H:%M IST")}\n')
    diamond_lines.append(f'Universe: {len(cache)} F&O stocks\n')
    diamond_lines.append('Earnings proxy: gap_pct >= 4% AND vol_ratio >= 2.0 in results months.\n\n')

    for side in ('long', 'short'):
        diamond_lines.append(f'## {side.upper()} side\n\n')
        default_rules = E.TradeRules(
            tp_pct=8.0, sl_pct=5.0, max_hold_days=15, direction=side,
            rsi_exit=None, cost_round_trip_pct=0.06,
        )
        if not os.path.exists(PERSTOCK_CSV) or \
           not (pd.read_csv(PERSTOCK_CSV)['side'] == side).any():
            stage1_perstock(cache, default_rules, side=side,
                            start='2018-01-01', end='2026-03-19')

        sweep_df = stage3_sweep(cache, select_cohort(
            pd.read_csv(PERSTOCK_CSV), side=side, top_n=15),
            side=side, start=E.TRAIN_START, end=E.TRAIN_END)

        survivors = stage4_walk_forward(cache, sweep_df, side=side, top_n=25)

        if survivors:
            stressed = stage5_cost_stress(cache, survivors, side=side)
            diamond_lines.append(f'### {len(survivors)} walk-forward survivors\n\n')
            for s in stressed:
                diamond_lines.append(
                    f'- gap={s["gap"]}% vol_ratio>={s["vol"]} months_only={s["months_only"]} '
                    f'tp={s["tp"]}% sl={s["sl"]}% hold={s["hold"]}d\n'
                    f'  Train: n={s["train_n"]} WR={s["train_wr"]:.1%} PF={s["train_pf"]:.2f}\n'
                    f'  Test:  n={s["test_n"]} WR={s["test_wr"]:.1%} PF={s["test_pf"]:.2f} '
                    f'DD={s["test_dd"]:.1f}% TotalRet={s["test_total_ret"]:.1f}%\n'
                    f'  Cost @ 0.20% RT: WR={s["test_wr_020"]:.1%} PF={s["test_pf_020"]:.2f} '
                    f'TotalRet={s["test_total_ret_020"]:.1f}%\n\n'
                )
        else:
            diamond_lines.append(f'No walk-forward survivors on {side} side.\n\n')

    with open(DIAMONDS_TXT, 'w') as f:
        f.writelines(diamond_lines)
    print(f'[main] DONE in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
