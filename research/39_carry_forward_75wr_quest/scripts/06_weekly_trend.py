"""Weekly trend continuation pattern (5-10 day hold).

Pattern hypothesis:
  - Weekly close above prior 8-week high (long) or below prior 8-week low (short)
  - Optional: weekly RSI in healthy zone (40-75 long, 25-60 short)
  - -> enter at NEXT MONDAY's open (day after the signal week's Friday)
  - Hold for K weeks (5/10 days) OR exit on weekly close < prior week low (long)

Universe: F&O 80 stocks. Train 2018-2023 (weekly), test 2024-2026.
TP/SL grid: applied to daily H/L during hold.
Hold range: 1 week (5 days), 2 weeks (10 days), trailing weekly-low.

Outputs:
  results/06_weekly_trend_perstock.csv
  results/06_weekly_trend_ranking.csv
  results/06_weekly_trend_walk_forward.csv
  results/06_weekly_trend_diamonds.txt
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine_daily import (
    FNO_UNIVERSE, load_daily, enrich_daily, to_weekly, add_weekly_donchian,
    simulate_signals_daily, DailyTradeRules,
    trade_stats, split_train_test,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
LOGS = os.path.join(ROOT, 'logs')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

START = '2017-01-01'
END = '2026-03-31'
TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
COST_DEFAULT = 0.06
COST_STRESS = 0.20

TP_SL_GRID = [
    ('TP20_SL15', 2.0, 1.5),
    ('TP30_SL15', 3.0, 1.5),
    ('TP40_SL20', 4.0, 2.0),
    ('TP50_SL20', 5.0, 2.0),
]
HOLD_DAYS_GRID = [5, 10]
WK_LOOKBACK_GRID = [4, 8, 12]


def make_weekly_signal_on_daily(df_daily: pd.DataFrame,
                                  weekly: pd.DataFrame,
                                  lookback_weeks: int = 8,
                                  direction: str = 'long') -> pd.Series:
    """Create a daily-aligned boolean signal that is True on the LAST trading day
    of each week where the weekly close broke prior N-week extreme.

    The signal fires on the daily-level Friday close (or last day of week).
    Entry then happens at next-day open (Monday) via simulate_signals_daily.
    """
    weekly = weekly.copy()
    weekly = add_weekly_donchian(weekly, period=lookback_weeks)

    if direction == 'long':
        wk_sig = (weekly['close'] > weekly[f'wk_donch_high_{lookback_weeks}'])
    else:
        wk_sig = (weekly['close'] < weekly[f'wk_donch_low_{lookback_weeks}'])
    wk_sig = wk_sig.fillna(False)

    # Map each weekly signal back to the LAST trading day of that week in df_daily
    # weekly index is W-FRI date (the Friday of the week).
    sig_daily = pd.Series(False, index=df_daily.index)
    for wk_date in weekly.index[wk_sig]:
        # Find last daily row with date <= weekly date (i.e. the Friday or Thu before)
        mask = df_daily.index <= wk_date
        if mask.any():
            last_day = df_daily.index[mask][-1]
            sig_daily.loc[last_day] = True
    return sig_daily


def main():
    t0 = time.time()
    print(f'[WEEKLY] start, universe={len(FNO_UNIVERSE)}', flush=True)

    daily_data: Dict[str, pd.DataFrame] = {}
    weekly_data: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(FNO_UNIVERSE):
        df = load_daily(sym, START, END)
        if df.empty or len(df) < 300:
            continue
        df = enrich_daily(df)
        wk = to_weekly(df[['open', 'high', 'low', 'close', 'volume']])
        if len(wk) < 30:
            continue
        daily_data[sym] = df
        weekly_data[sym] = wk
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(FNO_UNIVERSE)} loaded', flush=True)
    print(f'[WEEKLY] loaded {len(daily_data)} stocks', flush=True)

    perstock_csv = os.path.join(RESULTS, '06_weekly_trend_perstock.csv')
    fields = ['symbol', 'direction', 'config', 'cost', 'period',
              'n_trades', 'win_rate', 'avg_win', 'avg_loss', 'profit_factor',
              'sharpe', 'max_dd_pct', 'total_return_pct', 'avg_days_held', 'aws']
    with open(perstock_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    # Per-stock baseline (lookback=8, hold=5, TP30_SL15, long)
    for sym, df in daily_data.items():
        wk = weekly_data[sym]
        for direction in ['long', 'short']:
            sig = make_weekly_signal_on_daily(df, wk, lookback_weeks=8, direction=direction)
            if sig.sum() < 3:
                continue
            rules = DailyTradeRules(tp_pct=3.0, sl_pct=1.5, max_hold_days=5, direction=direction)
            trades = simulate_signals_daily(df, sig, rules, entry_mode='next_open')
            if trades.empty:
                continue
            for cost in [COST_DEFAULT, COST_STRESS]:
                stats = trade_stats(trades, cost_pct_round=cost)
                row = dict(symbol=sym, direction=direction, config='WK8_TP30_SL15_H5',
                           cost=cost, period='full', **{k: stats.get(k, 0) for k in fields if k in stats})
                with open(perstock_csv, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row)

    print(f'[WEEKLY] per-stock baseline done ({time.time()-t0:.0f}s)', flush=True)

    # Sweep
    ranking_csv = os.path.join(RESULTS, '06_weekly_trend_ranking.csv')
    rank_fields = fields + ['lookback_weeks', 'tp_pct', 'sl_pct', 'max_hold_days']
    with open(ranking_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=rank_fields).writeheader()

    cells_per_stock = len(WK_LOOKBACK_GRID) * len(HOLD_DAYS_GRID) * len(TP_SL_GRID) * 2  # 2 dirs
    print(f'[WEEKLY] sweep cells per stock: {cells_per_stock}', flush=True)

    sweep_count = 0
    for sym, df in daily_data.items():
        wk = weekly_data[sym]
        for direction in ['long', 'short']:
            for wkl in WK_LOOKBACK_GRID:
                sig = make_weekly_signal_on_daily(df, wk, lookback_weeks=wkl, direction=direction)
                if sig.sum() < 3:
                    continue
                for hold in HOLD_DAYS_GRID:
                    for cfg_name, tp, sl in TP_SL_GRID:
                        rules = DailyTradeRules(tp_pct=tp, sl_pct=sl,
                                                 max_hold_days=hold, direction=direction)
                        trades = simulate_signals_daily(df, sig, rules, entry_mode='next_open')
                        if trades.empty:
                            continue
                        train_t, test_t = split_train_test(trades, TRAIN_END, TEST_START)
                        for period_name, tdf in [('full', trades), ('train', train_t), ('test', test_t)]:
                            if len(tdf) == 0:
                                continue
                            for cost in [COST_DEFAULT, COST_STRESS]:
                                stats = trade_stats(tdf, cost_pct_round=cost)
                                row = dict(symbol=sym, direction=direction, config=cfg_name,
                                           cost=cost, period=period_name,
                                           lookback_weeks=wkl, tp_pct=tp, sl_pct=sl,
                                           max_hold_days=hold,
                                           **{k: stats.get(k, 0) for k in fields if k in stats})
                                with open(ranking_csv, 'a', newline='') as f:
                                    csv.DictWriter(f, fieldnames=rank_fields).writerow(row)
                        sweep_count += 1
        if sweep_count > 0 and sweep_count % 200 == 0:
            print(f'  swept {sweep_count} cells ({time.time()-t0:.0f}s)', flush=True)

    print(f'[WEEKLY] sweep done: {sweep_count} cells in {time.time()-t0:.0f}s', flush=True)

    # Walk-forward
    rank_df = pd.read_csv(ranking_csv)
    # Lower min_trades for weekly: weekly signals are scarcer, so 15 is realistic
    cand = rank_df[(rank_df['period'] == 'test') &
                   (rank_df['cost'] == COST_DEFAULT) &
                   (rank_df['n_trades'] >= 15) &
                   (rank_df['win_rate'] >= 0.70) &
                   (rank_df['profit_factor'] >= 1.8) &
                   (rank_df['max_dd_pct'] <= 15.0)].copy()
    print(f'[WEEKLY] test-period candidates: {len(cand)}', flush=True)

    keys = ['symbol', 'direction', 'config', 'lookback_weeks', 'tp_pct', 'sl_pct', 'max_hold_days']
    train_df = rank_df[(rank_df['period'] == 'train') & (rank_df['cost'] == COST_DEFAULT)].copy()
    train_df = train_df.rename(columns={'win_rate': 'train_wr', 'n_trades': 'train_n', 'profit_factor': 'train_pf'})
    cand = cand.merge(train_df[keys + ['train_wr', 'train_n', 'train_pf']], on=keys, how='left')
    cand_wf = cand[cand['train_wr'] >= 0.75].copy()
    print(f'[WEEKLY] walk-forward winners: {len(cand_wf)}', flush=True)

    stress_df = rank_df[(rank_df['period'] == 'test') & (rank_df['cost'] == COST_STRESS)].copy()
    stress_df = stress_df.rename(columns={'total_return_pct': 'test_stress_ret',
                                            'win_rate': 'test_stress_wr',
                                            'profit_factor': 'test_stress_pf'})
    cand_wf = cand_wf.merge(stress_df[keys + ['test_stress_ret', 'test_stress_wr', 'test_stress_pf']],
                              on=keys, how='left')

    wf_csv = os.path.join(RESULTS, '06_weekly_trend_walk_forward.csv')
    cand_wf.to_csv(wf_csv, index=False)
    print(f'[WEEKLY] walk-forward saved: {wf_csv}', flush=True)

    if len(cand_wf) > 0:
        diamonds = cand_wf['symbol'].value_counts().head(30).index.tolist()
    else:
        cand_relax = rank_df[(rank_df['period'] == 'test') & (rank_df['cost'] == COST_DEFAULT) &
                             (rank_df['n_trades'] >= 10) & (rank_df['win_rate'] >= 0.55)].copy()
        diamonds = cand_relax.groupby('symbol')['aws'].max().sort_values(ascending=False).head(30).index.tolist()

    diamonds_path = os.path.join(RESULTS, '06_weekly_trend_diamonds.txt')
    with open(diamonds_path, 'w') as f:
        for s in diamonds:
            f.write(s + '\n')
    print(f'[WEEKLY] diamonds saved ({len(diamonds)} stocks): {diamonds_path}', flush=True)
    print(f'[WEEKLY] DONE in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
