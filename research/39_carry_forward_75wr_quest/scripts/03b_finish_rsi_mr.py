"""Pattern 3 finisher — runs ONLY the missing pieces:
  - Short-side stage 1 (per-stock screen) if not done
  - Short-side stage 3 sweep (skips already-done cells)
  - Walk-forward both sides on top-25 candidates by AWS
  - Cost stress at 0.20% RT
  - Diamonds writeup
"""

from __future__ import annotations

import csv
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _engine_daily_my as E  # type: ignore
mod = __import__('03_rsi_mean_reversion', fromlist=['rsi_mr_long', 'rsi_mr_short'])

RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
PERSTOCK_CSV = os.path.join(RESULTS_DIR, '03_rsi_mr_perstock.csv')
RANKING_CSV = os.path.join(RESULTS_DIR, '03_rsi_mr_ranking.csv')
WF_CSV = os.path.join(RESULTS_DIR, '03_rsi_mr_walk_forward.csv')
DIAMONDS_TXT = os.path.join(RESULTS_DIR, '03_rsi_mr_diamonds.txt')


def stage1_short(cache):
    print('[03b] short-side per-stock screen', flush=True)
    perstock = pd.read_csv(PERSTOCK_CSV) if os.path.exists(PERSTOCK_CSV) else pd.DataFrame()
    if len(perstock) and (perstock['side'] == 'short').any():
        print('[03b] short stage1 already done', flush=True)
        return
    rules = E.TradeRules(tp_pct=5.0, sl_pct=3.0, max_hold_days=10,
                         direction='short', rsi_exit=50.0, cost_round_trip_pct=0.06)
    fields = ['symbol','side','n_trades','win_rate','avg_win','avg_loss',
              'profit_factor','sharpe','max_dd_pct','total_return_pct',
              'trades_per_year','aws','avg_days_held']
    write_header = not os.path.exists(PERSTOCK_CSV)
    with open(PERSTOCK_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for i, (sym, df) in enumerate(cache.items(), 1):
            if df.empty or len(df) < 250:
                continue
            sig = mod.rsi_mr_short(df, rsi_hi=75.0)
            sig = E.split_signal_by_period(df, sig, '2018-01-01', '2026-03-19')
            trades = E.simulate_signals_daily(df, sig, rules, rsi_series=df['rsi14'])
            stats = E.trade_stats(trades, period_years=8.2)
            w.writerow({'symbol': sym, 'side': 'short', **stats})
            if i % 20 == 0:
                print(f'  [{i}/{len(cache)}] short {sym} n={stats["n_trades"]}', flush=True)


def sweep_short(cache):
    print('[03b] short sweep — skip-done', flush=True)
    # already-done cells
    done = set()
    if os.path.exists(RANKING_CSV):
        df = pd.read_csv(RANKING_CSV)
        for _, r in df[df['side'] == 'short'].iterrows():
            key = (r['rsi_lvl'], r['bb'], r['low_or_high'], r['tp'], r['sl'], r['hold'],
                   r['rsi_exit'] if not pd.isna(r['rsi_exit']) else None)
            done.add(key)
    print(f'[03b] {len(done)} short cells already done', flush=True)

    rsi_levels = [70, 75, 80]
    bb_options = [False, True]
    high_options = [False, True]
    tp_levels = [3, 5, 7, 10]
    sl_levels = [2, 3, 5]
    hold_levels = [5, 10, 15]
    rsi_exit_levels = [None, 50]
    from itertools import product
    cells = list(product(rsi_levels, bb_options, high_options, tp_levels,
                         sl_levels, hold_levels, rsi_exit_levels))

    fields = ['side','rsi_lvl','bb','low_or_high','tp','sl','hold','rsi_exit',
              'n_trades','win_rate','avg_win','avg_loss','profit_factor',
              'sharpe','max_dd_pct','total_return_pct','trades_per_year',
              'aws','avg_days_held','n_stocks']
    write_header = not os.path.exists(RANKING_CSV)
    f = open(RANKING_CSV, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()

    t0 = time.time()
    skipped = 0
    for k, (rsi, bb, hi, tp, sl, hold, rxit) in enumerate(cells, 1):
        if tp < sl + 1:
            skipped += 1
            continue
        key = (rsi, bb, hi, tp, sl, hold, rxit if rxit is not None else None)
        # ranking CSV stores rsi_exit as float NaN -> None mapping
        rxit_check = rxit if rxit is not None else None
        if rxit is None:
            already = (rsi, bb, hi, tp, sl, hold, None) in done
        else:
            already = (rsi, bb, hi, tp, sl, hold, float(rxit)) in done
        if already:
            skipped += 1
            continue
        skwargs = {'rsi_hi': rsi, 'require_below_sma200': True,
                   'require_60d_high': hi, 'require_above_bb_hi': bb}
        rules = E.TradeRules(tp_pct=tp, sl_pct=sl, max_hold_days=hold,
                             direction='short', rsi_exit=rxit,
                             cost_round_trip_pct=0.06)
        per_df, pooled = E.run_strategy_on_cohort(
            cache, mod.rsi_mr_short, rules,
            E.TRAIN_START, E.TRAIN_END, **skwargs)
        n_stocks = (per_df['n_trades'] > 0).sum() if len(per_df) else 0
        row = {'side': 'short', 'rsi_lvl': rsi, 'bb': bb, 'low_or_high': hi,
               'tp': tp, 'sl': sl, 'hold': hold, 'rsi_exit': rxit,
               'n_stocks': int(n_stocks), **pooled}
        w.writerow(row)
        f.flush()
        if k % 50 == 0:
            print(f'  [{k}/{len(cells)}] {time.time()-t0:.0f}s | skipped={skipped} '
                  f'last n={pooled["n_trades"]} WR={pooled["win_rate"]:.2%}',
                  flush=True)
    f.close()
    print(f'[03b] short sweep done ({time.time()-t0:.0f}s, {skipped} skipped)', flush=True)


def walk_forward(cache, side: str, top_n: int = 25):
    df = pd.read_csv(RANKING_CSV)
    sub = df[df['side'] == side].copy()
    sub = sub[(sub['n_trades'] >= 30) & (sub['win_rate'] >= 0.55)
              & (sub['profit_factor'] >= 1.2)]
    sub = sub.sort_values('aws', ascending=False).head(top_n)
    print(f'[wf] {side}: {len(sub)} candidates from sweep', flush=True)

    fn = mod.rsi_mr_long if side == 'long' else mod.rsi_mr_short
    rsi_kw = 'rsi_lo' if side == 'long' else 'rsi_hi'
    sma_kw = 'require_above_sma200' if side == 'long' else 'require_below_sma200'
    low_kw = 'require_60d_low' if side == 'long' else 'require_60d_high'
    bb_kw = 'require_below_bb_lo' if side == 'long' else 'require_above_bb_hi'

    fields = ['side','rsi_lvl','bb','low_or_high','tp','sl','hold','rsi_exit',
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
        rxit = None if pd.isna(r['rsi_exit']) else float(r['rsi_exit'])
        skwargs = {rsi_kw: float(r['rsi_lvl']), sma_kw: True,
                   low_kw: bool(r['low_or_high']), bb_kw: bool(r['bb'])}
        rules = E.TradeRules(tp_pct=float(r['tp']), sl_pct=float(r['sl']),
                             max_hold_days=int(r['hold']),
                             direction=side, rsi_exit=rxit,
                             cost_round_trip_pct=0.06)
        _, train_p = E.run_strategy_on_cohort(
            cache, fn, rules, E.TRAIN_START, E.TRAIN_END, **skwargs)
        _, test_p = E.run_strategy_on_cohort(
            cache, fn, rules, E.TEST_START, E.TEST_END, **skwargs)
        passes = E.passes_walk_forward(train_p, test_p)
        row = {
            'side': side,
            'rsi_lvl': r['rsi_lvl'], 'bb': r['bb'], 'low_or_high': r['low_or_high'],
            'tp': r['tp'], 'sl': r['sl'], 'hold': r['hold'], 'rsi_exit': rxit,
            'train_n': train_p['n_trades'], 'train_wr': train_p['win_rate'],
            'train_pf': train_p['profit_factor'], 'train_dd': train_p['max_dd_pct'],
            'test_n': test_p['n_trades'], 'test_wr': test_p['win_rate'],
            'test_pf': test_p['profit_factor'], 'test_dd': test_p['max_dd_pct'],
            'test_total_ret': test_p['total_return_pct'],
            'test_aws': test_p['aws'], 'passes_wf': passes,
        }
        w.writerow(row); f.flush()
        if passes:
            survivors.append(row)
    f.close()
    print(f'[wf] {side}: {len(survivors)} survivors', flush=True)
    return survivors


def cost_stress(cache, survivors, side: str):
    fn = mod.rsi_mr_long if side == 'long' else mod.rsi_mr_short
    rsi_kw = 'rsi_lo' if side == 'long' else 'rsi_hi'
    sma_kw = 'require_above_sma200' if side == 'long' else 'require_below_sma200'
    low_kw = 'require_60d_low' if side == 'long' else 'require_60d_high'
    bb_kw = 'require_below_bb_lo' if side == 'long' else 'require_above_bb_hi'
    out = []
    for r in survivors:
        skwargs = {rsi_kw: float(r['rsi_lvl']), sma_kw: True,
                   low_kw: bool(r['low_or_high']), bb_kw: bool(r['bb'])}
        rules = E.TradeRules(tp_pct=float(r['tp']), sl_pct=float(r['sl']),
                             max_hold_days=int(r['hold']),
                             direction=side, rsi_exit=r['rsi_exit'],
                             cost_round_trip_pct=0.20)
        _, p = E.run_strategy_on_cohort(cache, fn, rules,
                                          E.TEST_START, E.TEST_END, **skwargs)
        out.append({**r, 'test_wr_020': p['win_rate'],
                    'test_pf_020': p['profit_factor'],
                    'test_total_ret_020': p['total_return_pct']})
    return out


def main():
    t0 = time.time()
    print('[03b] loading cache', flush=True)
    symbols = E.list_daily_universe(E.FNO_UNIVERSE, start='2017-01-01',
                                    end='2026-03-19', min_rows=1500)
    cache = E.load_many_daily(symbols)
    for s in list(cache):
        cache[s] = E.enrich_daily(cache[s])
    print(f'[03b] {len(cache)} stocks loaded ({time.time()-t0:.0f}s)', flush=True)

    stage1_short(cache)
    sweep_short(cache)

    survivors_long = walk_forward(cache, 'long')
    survivors_short = walk_forward(cache, 'short')

    diamond_lines = ['# Pattern 3 - RSI Mean-Reversion - Walk-Forward Diamonds\n']
    diamond_lines.append(f'Generated: {time.strftime("%Y-%m-%d %H:%M IST")}\n')
    diamond_lines.append(f'Universe: {len(cache)} F&O stocks\n')
    diamond_lines.append(f'Train: {E.TRAIN_START}..{E.TRAIN_END}\n')
    diamond_lines.append(f'Test:  {E.TEST_START}..{E.TEST_END}\n\n')

    for side, survivors in (('long', survivors_long), ('short', survivors_short)):
        diamond_lines.append(f'## {side.upper()} side\n\n')
        if survivors:
            stressed = cost_stress(cache, survivors, side)
            diamond_lines.append(f'### {len(survivors)} walk-forward survivors\n\n')
            for s in stressed:
                diamond_lines.append(
                    f'- rsi_lvl={s["rsi_lvl"]} bb={s["bb"]} extreme={s["low_or_high"]} '
                    f'tp={s["tp"]}% sl={s["sl"]}% hold={s["hold"]}d rsi_exit={s["rsi_exit"]}\n'
                    f'  Train: n={s["train_n"]} WR={s["train_wr"]:.1%} PF={s["train_pf"]:.2f}\n'
                    f'  Test:  n={s["test_n"]} WR={s["test_wr"]:.1%} PF={s["test_pf"]:.2f} '
                    f'DD={s["test_dd"]:.1f}% TotalRet={s["test_total_ret"]:.1f}%\n'
                    f'  Cost @ 0.20% RT: WR={s["test_wr_020"]:.1%} PF={s["test_pf_020"]:.2f} '
                    f'TotalRet={s["test_total_ret_020"]:.1f}%\n\n'
                )
        else:
            diamond_lines.append('No walk-forward survivors.\n\n')
            # Surface top-3 best candidates anyway for narrative
            df = pd.read_csv(RANKING_CSV)
            sub = df[df['side'] == side].sort_values('win_rate', ascending=False).head(5)
            if len(sub):
                diamond_lines.append(f'Best WR (train) candidates for reference:\n\n')
                for _, r in sub.iterrows():
                    diamond_lines.append(
                        f'- rsi_lvl={r["rsi_lvl"]} tp={r["tp"]}% sl={r["sl"]}% hold={r["hold"]}d '
                        f'n={r["n_trades"]} WR={r["win_rate"]:.1%} PF={r["profit_factor"]:.2f} '
                        f'TotalRet={r["total_return_pct"]:.1f}%\n'
                    )
                diamond_lines.append('\n')

    with open(DIAMONDS_TXT, 'w') as f:
        f.writelines(diamond_lines)
    print(f'[03b] DONE in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
