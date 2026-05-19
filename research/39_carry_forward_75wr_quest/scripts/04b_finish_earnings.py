"""Pattern 4 finisher — runs ONLY the missing pieces."""

from __future__ import annotations

import csv
import os
import sys
import time
from itertools import product

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _engine_daily_my as E  # type: ignore
mod = __import__('04_earnings_drift', fromlist=['earnings_long', 'earnings_short'])

RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
PERSTOCK_CSV = os.path.join(RESULTS_DIR, '04_earnings_perstock.csv')
RANKING_CSV = os.path.join(RESULTS_DIR, '04_earnings_ranking.csv')
WF_CSV = os.path.join(RESULTS_DIR, '04_earnings_walk_forward.csv')
DIAMONDS_TXT = os.path.join(RESULTS_DIR, '04_earnings_diamonds.txt')


def stage1(cache, side):
    perstock = pd.read_csv(PERSTOCK_CSV) if os.path.exists(PERSTOCK_CSV) else pd.DataFrame()
    if len(perstock) and (perstock['side'] == side).any():
        print(f'[04b] {side} stage1 already done', flush=True)
        return
    print(f'[04b] {side} per-stock screen', flush=True)
    fn = mod.earnings_long if side == 'long' else mod.earnings_short
    rules = E.TradeRules(tp_pct=8.0, sl_pct=5.0, max_hold_days=15, direction=side,
                         rsi_exit=None, cost_round_trip_pct=0.06)
    fields = ['symbol','side','n_trades','win_rate','avg_win','avg_loss',
              'profit_factor','sharpe','max_dd_pct','total_return_pct',
              'trades_per_year','aws','avg_days_held']
    write_header = not os.path.exists(PERSTOCK_CSV)
    f = open(PERSTOCK_CSV, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()
    for i, (sym, df) in enumerate(cache.items(), 1):
        if df.empty or len(df) < 250:
            continue
        sig = fn(df)
        sig = E.split_signal_by_period(df, sig, '2018-01-01', '2026-03-19')
        trades = E.simulate_signals_daily(df, sig, rules, rsi_series=df['rsi14'])
        stats = E.trade_stats(trades, period_years=8.2)
        w.writerow({'symbol': sym, 'side': side, **stats})
        f.flush()
        if i % 20 == 0:
            print(f'  [{i}/{len(cache)}] {side} {sym} n={stats["n_trades"]} WR={stats["win_rate"]:.2%}',
                  flush=True)
    f.close()


def sweep(cache, side):
    print(f'[04b] {side} sweep — skip-done', flush=True)
    done = set()
    if os.path.exists(RANKING_CSV):
        df = pd.read_csv(RANKING_CSV)
        for _, r in df[df['side'] == side].iterrows():
            done.add((r['gap'], r['vol'], r['months_only'], r['tp'], r['sl'], r['hold']))
    print(f'[04b] {len(done)} {side} cells already done', flush=True)

    if side == 'long':
        gap_levels = [3, 4, 5, 7]
    else:
        gap_levels = [-3, -4, -5, -7]
    vol_levels = [1.5, 2.0, 3.0]
    tp_levels = [4, 6, 8, 12]
    sl_levels = [3, 5, 7]
    hold_levels = [10, 15, 20]
    months_options = [True, False]

    cells = list(product(gap_levels, vol_levels, tp_levels, sl_levels,
                         hold_levels, months_options))

    fields = ['side','gap','vol','months_only','tp','sl','hold',
              'n_trades','win_rate','avg_win','avg_loss','profit_factor',
              'sharpe','max_dd_pct','total_return_pct','trades_per_year',
              'aws','avg_days_held','n_stocks']
    write_header = not os.path.exists(RANKING_CSV)
    f = open(RANKING_CSV, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()

    fn = mod.earnings_long if side == 'long' else mod.earnings_short
    gap_kw = 'gap_pct_min' if side == 'long' else 'gap_pct_max'
    ret_kw = 'same_day_ret_min' if side == 'long' else 'same_day_ret_max'

    t0 = time.time()
    skipped = 0
    for k, (gap, vol, tp, sl, hold, months) in enumerate(cells, 1):
        if tp < sl + 1:
            skipped += 1; continue
        if (gap, vol, months, tp, sl, hold) in done:
            skipped += 1; continue
        skwargs = {gap_kw: gap, 'vol_ratio_min': vol, ret_kw: 0.0,
                   'results_months_only': months}
        rules = E.TradeRules(tp_pct=tp, sl_pct=sl, max_hold_days=hold,
                             direction=side, rsi_exit=None,
                             cost_round_trip_pct=0.06)
        per_df, pooled = E.run_strategy_on_cohort(
            cache, fn, rules, E.TRAIN_START, E.TRAIN_END, **skwargs)
        n_stocks = (per_df['n_trades'] > 0).sum() if len(per_df) else 0
        row = {'side': side, 'gap': gap, 'vol': vol, 'months_only': months,
               'tp': tp, 'sl': sl, 'hold': hold, 'n_stocks': int(n_stocks),
               **pooled}
        w.writerow(row); f.flush()
        if k % 50 == 0:
            print(f'  [{k}/{len(cells)}] {time.time()-t0:.0f}s | skipped={skipped} '
                  f'last gap={gap} tp={tp} n={pooled["n_trades"]} '
                  f'WR={pooled["win_rate"]:.2%}', flush=True)
    f.close()
    print(f'[04b] {side} sweep done {time.time()-t0:.0f}s', flush=True)


def walk_forward(cache, side, top_n=25):
    df = pd.read_csv(RANKING_CSV)
    sub = df[df['side'] == side].copy()
    sub = sub[(sub['n_trades'] >= 30) & (sub['win_rate'] >= 0.55)
              & (sub['profit_factor'] >= 1.2)]
    sub = sub.sort_values('aws', ascending=False).head(top_n)
    print(f'[wf] {side}: {len(sub)} candidates', flush=True)

    fn = mod.earnings_long if side == 'long' else mod.earnings_short
    gap_kw = 'gap_pct_min' if side == 'long' else 'gap_pct_max'
    ret_kw = 'same_day_ret_min' if side == 'long' else 'same_day_ret_max'

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
                   'vol_ratio_min': float(r['vol']), ret_kw: 0.0,
                   'results_months_only': bool(r['months_only'])}
        rules = E.TradeRules(tp_pct=float(r['tp']), sl_pct=float(r['sl']),
                             max_hold_days=int(r['hold']),
                             direction=side, rsi_exit=None,
                             cost_round_trip_pct=0.06)
        _, train_p = E.run_strategy_on_cohort(
            cache, fn, rules, E.TRAIN_START, E.TRAIN_END, **skwargs)
        _, test_p = E.run_strategy_on_cohort(
            cache, fn, rules, E.TEST_START, E.TEST_END, **skwargs)
        passes = E.passes_walk_forward(train_p, test_p)
        row = {'side': side, 'gap': r['gap'], 'vol': r['vol'],
               'months_only': r['months_only'],
               'tp': r['tp'], 'sl': r['sl'], 'hold': r['hold'],
               'train_n': train_p['n_trades'], 'train_wr': train_p['win_rate'],
               'train_pf': train_p['profit_factor'], 'train_dd': train_p['max_dd_pct'],
               'test_n': test_p['n_trades'], 'test_wr': test_p['win_rate'],
               'test_pf': test_p['profit_factor'], 'test_dd': test_p['max_dd_pct'],
               'test_total_ret': test_p['total_return_pct'],
               'test_aws': test_p['aws'], 'passes_wf': passes}
        w.writerow(row); f.flush()
        if passes:
            survivors.append(row)
    f.close()
    return survivors


def cost_stress(cache, survivors, side):
    fn = mod.earnings_long if side == 'long' else mod.earnings_short
    gap_kw = 'gap_pct_min' if side == 'long' else 'gap_pct_max'
    ret_kw = 'same_day_ret_min' if side == 'long' else 'same_day_ret_max'
    out = []
    for r in survivors:
        skwargs = {gap_kw: float(r['gap']),
                   'vol_ratio_min': float(r['vol']), ret_kw: 0.0,
                   'results_months_only': bool(r['months_only'])}
        rules = E.TradeRules(tp_pct=float(r['tp']), sl_pct=float(r['sl']),
                             max_hold_days=int(r['hold']),
                             direction=side, rsi_exit=None,
                             cost_round_trip_pct=0.20)
        _, p = E.run_strategy_on_cohort(cache, fn, rules,
                                          E.TEST_START, E.TEST_END, **skwargs)
        out.append({**r, 'test_wr_020': p['win_rate'],
                    'test_pf_020': p['profit_factor'],
                    'test_total_ret_020': p['total_return_pct']})
    return out


def main():
    t0 = time.time()
    print('[04b] loading cache', flush=True)
    symbols = E.list_daily_universe(E.FNO_UNIVERSE, start='2017-01-01',
                                    end='2026-03-19', min_rows=1500)
    cache = E.load_many_daily(symbols)
    for s in list(cache):
        cache[s] = E.enrich_daily(cache[s])
    print(f'[04b] {len(cache)} stocks loaded ({time.time()-t0:.0f}s)', flush=True)

    for side in ('long', 'short'):
        stage1(cache, side)
        sweep(cache, side)

    sur_l = walk_forward(cache, 'long')
    sur_s = walk_forward(cache, 'short')

    lines = ['# Pattern 4 - Earnings Post-Drift - Walk-Forward Diamonds\n']
    lines.append(f'Generated: {time.strftime("%Y-%m-%d %H:%M IST")}\n')
    lines.append(f'Universe: {len(cache)} F&O stocks\n')
    lines.append('Earnings proxy: gap_pct >= 4% AND vol_ratio >= 2.0 in results months.\n\n')
    for side, survivors in (('long', sur_l), ('short', sur_s)):
        lines.append(f'## {side.upper()} side\n\n')
        if survivors:
            stressed = cost_stress(cache, survivors, side)
            lines.append(f'### {len(survivors)} walk-forward survivors\n\n')
            for s in stressed:
                lines.append(
                    f'- gap={s["gap"]}% vol_ratio>={s["vol"]} months_only={s["months_only"]} '
                    f'tp={s["tp"]}% sl={s["sl"]}% hold={s["hold"]}d\n'
                    f'  Train: n={s["train_n"]} WR={s["train_wr"]:.1%} PF={s["train_pf"]:.2f}\n'
                    f'  Test:  n={s["test_n"]} WR={s["test_wr"]:.1%} PF={s["test_pf"]:.2f} '
                    f'DD={s["test_dd"]:.1f}% TotalRet={s["test_total_ret"]:.1f}%\n'
                    f'  Cost @ 0.20% RT: WR={s["test_wr_020"]:.1%} PF={s["test_pf_020"]:.2f} '
                    f'TotalRet={s["test_total_ret_020"]:.1f}%\n\n'
                )
        else:
            lines.append('No walk-forward survivors.\n\n')
            df = pd.read_csv(RANKING_CSV)
            sub = df[df['side'] == side].sort_values('win_rate', ascending=False).head(5)
            if len(sub):
                lines.append('Best WR (train) candidates for reference:\n\n')
                for _, r in sub.iterrows():
                    lines.append(
                        f'- gap={r["gap"]}% vol={r["vol"]} tp={r["tp"]}% sl={r["sl"]}% '
                        f'hold={r["hold"]}d n={r["n_trades"]} WR={r["win_rate"]:.1%} '
                        f'PF={r["profit_factor"]:.2f} TotalRet={r["total_return_pct"]:.1f}%\n'
                    )
                lines.append('\n')

    with open(DIAMONDS_TXT, 'w') as f:
        f.writelines(lines)
    print(f'[04b] DONE in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
