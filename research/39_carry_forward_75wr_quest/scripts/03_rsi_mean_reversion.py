"""Pattern 3 — Daily RSI mean-reversion (5-15 day hold).

Hypothesis: in a long-term uptrend, sharp pullbacks to oversold daily RSI
tend to bounce over 5-15 days.

LONG signal:
  Close > 200-SMA (long-term uptrend filter)
  Daily RSI(14) < threshold (oversold)
  Optional gates:
    - 60-day low touch (close at multi-month low)
    - Below lower Bollinger band

SHORT mirror (long-term downtrend + RSI > threshold).

Pipeline:
  1. Per-stock screen with default params -> filter to high-frequency cohort
  2. Param sweep on cohort: rsi_lo, sma_filter on/off, bb_filter on/off,
     tp_pct, sl_pct, max_hold_days, rsi_exit
  3. Rank by AWS, gated by hard floors
  4. Walk-forward winners on 2024-01-01..2026-03-19 test
  5. Cost-stress test all winners at 0.20% RT

Outputs:
    results/03_rsi_mr_perstock.csv     -- per-stock screening (default rules)
    results/03_rsi_mr_ranking.csv      -- full param sweep ranking
    results/03_rsi_mr_walk_forward.csv -- walk-forward survivors
    results/03_rsi_mr_diamonds.txt     -- final winners narrative
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

PERSTOCK_CSV = os.path.join(RESULTS_DIR, '03_rsi_mr_perstock.csv')
RANKING_CSV = os.path.join(RESULTS_DIR, '03_rsi_mr_ranking.csv')
WF_CSV = os.path.join(RESULTS_DIR, '03_rsi_mr_walk_forward.csv')
DIAMONDS_TXT = os.path.join(RESULTS_DIR, '03_rsi_mr_diamonds.txt')


# ---------------------------------------------------------------------------
# strategy
# ---------------------------------------------------------------------------

def rsi_mr_long(df: pd.DataFrame,
                rsi_lo: float = 25.0,
                require_above_sma200: bool = True,
                require_60d_low: bool = False,
                require_below_bb_lo: bool = False) -> pd.Series:
    sig = (df['rsi14'] < rsi_lo)
    if require_above_sma200:
        sig &= (df['close'] > df['sma200'])
    if require_60d_low:
        sig &= df['is_60d_low']
    if require_below_bb_lo:
        sig &= (df['close'] < df['bb_lo'])
    # require at least 250 bars history
    sig &= df['sma200'].notna()
    return sig.fillna(False)


def rsi_mr_short(df: pd.DataFrame,
                 rsi_hi: float = 75.0,
                 require_below_sma200: bool = True,
                 require_60d_high: bool = False,
                 require_above_bb_hi: bool = False) -> pd.Series:
    sig = (df['rsi14'] > rsi_hi)
    if require_below_sma200:
        sig &= (df['close'] < df['sma200'])
    if require_60d_high:
        sig &= df['is_60d_high']
    if require_above_bb_hi:
        sig &= (df['close'] > df['bb_hi'])
    sig &= df['sma200'].notna()
    return sig.fillna(False)


# ---------------------------------------------------------------------------
# Stage 1 — per-stock screen
# ---------------------------------------------------------------------------

def stage1_perstock(cache, default_rules, side='long', start='2018-01-01', end='2026-03-19'):
    print(f'[stage1] per-stock screen ({side}) over {start}..{end} on {len(cache)} stocks',
          flush=True)
    rows = []
    fn = rsi_mr_long if side == 'long' else rsi_mr_short
    kwargs = {'rsi_lo': 25.0} if side == 'long' else {'rsi_hi': 75.0}

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
            try:
                sig = fn(df, **kwargs)
            except Exception:
                continue
            sig = E.split_signal_by_period(df, sig, start, end)
            trades = E.simulate_signals_daily(
                df, sig, default_rules,
                rsi_series=df['rsi14'],
            )
            years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 0.1)
            stats = E.trade_stats(trades, period_years=years)
            row = {'symbol': sym, 'side': side, **stats}
            w.writerow(row)
            rows.append(row)
            if i % 25 == 0:
                print(f'  [{i}/{len(cache)}] {sym}: n={stats["n_trades"]} '
                      f'WR={stats["win_rate"]:.2%} PF={stats["profit_factor"]:.2f}',
                      flush=True)
    print(f'[stage1] done: {len(rows)} rows', flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 2 — cohort selection
# ---------------------------------------------------------------------------

def select_cohort(perstock_df: pd.DataFrame, side='long', top_n=20,
                  use_all_with_signal: bool = True) -> list:
    """Pick cohort. Default: ALL stocks that have >=1 trade in the screen
    (since RSI<25 + above SMA200 fires only ~3-8 times per stock per 8yr).
    Cohort then expands the population for the sweep."""
    sub = perstock_df[perstock_df['side'] == side].copy()
    if use_all_with_signal:
        cohort = sub[sub['n_trades'] >= 1]['symbol'].tolist()
    else:
        sub = sub[(sub['n_trades'] >= 5) & (sub['win_rate'] >= 0.50)]
        sub = sub.sort_values('aws', ascending=False)
        cohort = sub.head(top_n)['symbol'].tolist()
    print(f'[cohort] selected {len(cohort)} {side} symbols', flush=True)
    return cohort


# ---------------------------------------------------------------------------
# Stage 3 — param sweep
# ---------------------------------------------------------------------------

def stage3_sweep(cache, cohort, side='long', start=E.TRAIN_START, end=E.TRAIN_END):
    print(f'[stage3] param sweep ({side}) on {len(cohort)} stocks, {start}..{end}',
          flush=True)
    sub_cache = {s: cache[s] for s in cohort if s in cache}

    if side == 'long':
        rsi_levels = [20, 25, 30]
        bb_options = [False, True]
        low_options = [False, True]
    else:
        rsi_levels = [70, 75, 80]
        bb_options = [False, True]
        low_options = [False, True]
    tp_levels = [3, 5, 7, 10]
    sl_levels = [2, 3, 5]
    hold_levels = [5, 10, 15]
    rsi_exit_levels = [None, 50]

    fields = ['side','rsi_lvl','bb','low_or_high','tp','sl','hold','rsi_exit',
              'n_trades','win_rate','avg_win','avg_loss','profit_factor',
              'sharpe','max_dd_pct','total_return_pct','trades_per_year',
              'aws','avg_days_held','n_stocks']

    # write header & truncate per side; we only ever call once per side
    write_header = not os.path.exists(RANKING_CSV)
    f = open(RANKING_CSV, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()

    fn = rsi_mr_long if side == 'long' else rsi_mr_short
    rsi_kw = 'rsi_lo' if side == 'long' else 'rsi_hi'
    sma_kw = 'require_above_sma200' if side == 'long' else 'require_below_sma200'
    low_kw = 'require_60d_low' if side == 'long' else 'require_60d_high'
    bb_kw = 'require_below_bb_lo' if side == 'long' else 'require_above_bb_hi'

    cells = list(product(rsi_levels, bb_options, low_options, tp_levels,
                         sl_levels, hold_levels, rsi_exit_levels))
    print(f'[stage3] {len(cells)} cells', flush=True)
    rows = []
    t0 = time.time()
    for k, (rsi, bb, low, tp, sl, hold, rxit) in enumerate(cells, 1):
        # filter SL < TP strictly (favorable RR)
        if tp < sl + 1:
            continue
        # build strategy kwargs
        skwargs = {rsi_kw: rsi, sma_kw: True, low_kw: low, bb_kw: bb}
        rules = E.TradeRules(tp_pct=tp, sl_pct=sl, max_hold_days=hold,
                             direction=side,
                             rsi_exit=rxit, cost_round_trip_pct=0.06)
        per_df, pooled = E.run_strategy_on_cohort(
            sub_cache, fn, rules, start, end, **skwargs)
        n_stocks = (per_df['n_trades'] > 0).sum() if len(per_df) else 0
        row = {
            'side': side, 'rsi_lvl': rsi, 'bb': bb, 'low_or_high': low,
            'tp': tp, 'sl': sl, 'hold': hold, 'rsi_exit': rxit,
            'n_stocks': int(n_stocks), **pooled,
        }
        w.writerow(row)
        f.flush()
        rows.append(row)
        if k % 50 == 0 or k == len(cells):
            elapsed = time.time() - t0
            print(f'  [{k}/{len(cells)}] {elapsed:.0f}s | '
                  f'last: rsi={rsi} tp={tp} sl={sl} hold={hold} '
                  f'n={pooled["n_trades"]} WR={pooled["win_rate"]:.2%}',
                  flush=True)
    f.close()
    print(f'[stage3] done in {time.time()-t0:.0f}s', flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 4 — walk-forward
# ---------------------------------------------------------------------------

def stage4_walk_forward(cache, sweep_df: pd.DataFrame, side='long', top_n=20):
    print(f'[stage4] walk-forward top-{top_n} ({side})', flush=True)
    # pick top-N candidates by AWS that ALSO meet train hard gates
    sub = sweep_df[sweep_df['side'] == side].copy()
    sub = sub[
        (sub['n_trades'] >= 30)
        & (sub['win_rate'] >= 0.60)  # train >=60% (allow loosening — high-WR comes via tight params)
        & (sub['profit_factor'] >= 1.3)
    ]
    sub = sub.sort_values('aws', ascending=False).head(top_n)
    print(f'[stage4] {len(sub)} candidates pass loose train gates', flush=True)

    # Walk-forward on FULL F&O cache (sweep was full universe too)
    sub_cache = cache

    fn = rsi_mr_long if side == 'long' else rsi_mr_short
    rsi_kw = 'rsi_lo' if side == 'long' else 'rsi_hi'
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
        rxit = r['rsi_exit']
        if pd.isna(rxit):
            rxit = None
        else:
            rxit = float(rxit)
        skwargs = {rsi_kw: float(r['rsi_lvl']),
                   low_kw: bool(r['low_or_high']),
                   bb_kw: bool(r['bb'])}
        rules = E.TradeRules(tp_pct=float(r['tp']), sl_pct=float(r['sl']),
                             max_hold_days=int(r['hold']),
                             direction=side, rsi_exit=rxit,
                             cost_round_trip_pct=0.06)
        _, train_p = E.run_strategy_on_cohort(
            sub_cache, fn, rules, E.TRAIN_START, E.TRAIN_END, **skwargs)
        _, test_p = E.run_strategy_on_cohort(
            sub_cache, fn, rules, E.TEST_START, E.TEST_END, **skwargs)
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
        w.writerow(row)
        f.flush()
        if passes:
            survivors.append(row)
            print(f'  WIN side={side} rsi={r["rsi_lvl"]} tp={r["tp"]} sl={r["sl"]} '
                  f'hold={r["hold"]} | train WR={train_p["win_rate"]:.1%} test WR={test_p["win_rate"]:.1%}',
                  flush=True)
    f.close()
    return survivors


# ---------------------------------------------------------------------------
# Stage 5 — cost stress
# ---------------------------------------------------------------------------

def stage5_cost_stress(cache, survivors, side='long'):
    print(f'[stage5] cost stress at 0.20% RT - {len(survivors)} survivors',
          flush=True)
    sub_cache = cache
    fn = rsi_mr_long if side == 'long' else rsi_mr_short
    rsi_kw = 'rsi_lo' if side == 'long' else 'rsi_hi'
    low_kw = 'require_60d_low' if side == 'long' else 'require_60d_high'
    bb_kw = 'require_below_bb_lo' if side == 'long' else 'require_above_bb_hi'
    out = []
    for r in survivors:
        rxit = r['rsi_exit']
        skwargs = {rsi_kw: float(r['rsi_lvl']),
                   low_kw: bool(r['low_or_high']),
                   bb_kw: bool(r['bb'])}
        rules = E.TradeRules(tp_pct=float(r['tp']), sl_pct=float(r['sl']),
                             max_hold_days=int(r['hold']),
                             direction=side, rsi_exit=rxit,
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
    # 1. load data
    print('[main] loading F&O daily data...', flush=True)
    symbols = E.list_daily_universe(E.FNO_UNIVERSE, start='2017-01-01',
                                    end='2026-03-19', min_rows=1500)
    print(f'[main] {len(symbols)} F&O symbols with sufficient daily history',
          flush=True)
    cache = E.load_many_daily(symbols)
    for sym in list(cache.keys()):
        cache[sym] = E.enrich_daily(cache[sym])
    print(f'[main] enriched cache: {len(cache)} stocks ({time.time()-t0:.0f}s)',
          flush=True)

    diamond_lines = []
    diamond_lines.append('# Pattern 3 — RSI Mean-Reversion — Walk-Forward Diamonds\n')
    diamond_lines.append(f'Generated: {time.strftime("%Y-%m-%d %H:%M IST")}\n')
    diamond_lines.append(f'Universe: {len(cache)} F&O stocks\n')
    diamond_lines.append(f'Train: {E.TRAIN_START}..{E.TRAIN_END}\n')
    diamond_lines.append(f'Test:  {E.TEST_START}..{E.TEST_END}\n\n')

    for side in ('long', 'short'):
        diamond_lines.append(f'## {side.upper()} side\n\n')
        # Stage 1
        default_rules = E.TradeRules(
            tp_pct=5.0, sl_pct=3.0, max_hold_days=10,
            direction=side, rsi_exit=50.0, cost_round_trip_pct=0.06,
        )
        if side == 'long' or not os.path.exists(PERSTOCK_CSV) or \
           not (pd.read_csv(PERSTOCK_CSV)['side'] == side).any():
            stage1_perstock(cache, default_rules, side=side,
                            start='2018-01-01', end='2026-03-19')

        # Stage 3 sweep — use ALL F&O stocks (not the screen cohort) because
        # the per-stock screen used default rules; the sweep varies RSI etc.
        all_symbols = list(cache.keys())
        sweep_df = stage3_sweep(cache, all_symbols,
            side=side, start=E.TRAIN_START, end=E.TRAIN_END)

        # Stage 4 walk-forward (also full universe)
        survivors = stage4_walk_forward(cache, sweep_df, side=side, top_n=25)

        # Stage 5 cost stress
        if survivors:
            stressed = stage5_cost_stress(cache, survivors, side=side)
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
            diamond_lines.append(f'No walk-forward survivors on {side} side.\n\n')

    with open(DIAMONDS_TXT, 'w') as f:
        f.writelines(diamond_lines)
    print(f'[main] DONE in {time.time()-t0:.0f}s — see {DIAMONDS_TXT}', flush=True)


if __name__ == '__main__':
    main()
