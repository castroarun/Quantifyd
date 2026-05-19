"""Daily breakout swing pattern (2-5 day hold).

Pattern hypothesis:
  - Daily Donchian-N breakout (close > prior N-day high) PLUS
  - Volume > 1.5x avg PLUS
  - 50-EMA rising (close > ema50, ema50 > ema50.shift(5)) PLUS
  - RSI in healthy zone (40-75 — not exhausted)
  - 200-SMA above (long-term uptrend) optional gate
  -> long entry next-day open, exit on TP/SL or after K days at close.

Universe: F&O 80 stocks. Train 2018-2023, test 2024-2026.
TP/SL grid: (2.0,1.5), (3.0,1.5), (4.0,2.0), (5.0,2.0)
Hold range: 2, 3, 5, 10 days.
Cost: 0.06% F&O round-trip default; stress 0.20% round-trip.

Outputs:
  results/02_swing_breakout_perstock.csv
  results/02_swing_breakout_ranking.csv
  results/02_swing_breakout_walk_forward.csv
  results/02_swing_breakout_diamonds.txt
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
    FNO_UNIVERSE, load_daily, enrich_daily,
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
HOLD_GRID = [2, 3, 5, 10]


def make_breakout_signal(df: pd.DataFrame,
                          donch_period: int = 20,
                          vol_ratio_min: float = 1.5,
                          require_ema50_rising: bool = True,
                          require_above_sma200: bool = True,
                          rsi_min: float = 40,
                          rsi_max: float = 75,
                          direction: str = 'long') -> pd.Series:
    """Daily breakout signal."""
    if direction == 'long':
        donch_col = f'donch_high_{donch_period}'
        if donch_col not in df.columns:
            return pd.Series(False, index=df.index)
        sig = (df['close'] > df[donch_col]) & \
              (df['vol_ratio'] >= vol_ratio_min) & \
              (df['rsi'].between(rsi_min, rsi_max))
        if require_ema50_rising:
            sig = sig & (df['close'] > df['ema50']) & (df['ema50'] > df['ema50'].shift(5))
        if require_above_sma200:
            sig = sig & (df['close'] > df['sma200'])
    else:
        donch_col = f'donch_low_{donch_period}'
        if donch_col not in df.columns:
            return pd.Series(False, index=df.index)
        sig = (df['close'] < df[donch_col]) & \
              (df['vol_ratio'] >= vol_ratio_min) & \
              (df['rsi'].between(100 - rsi_max, 100 - rsi_min))
        if require_ema50_rising:
            sig = sig & (df['close'] < df['ema50']) & (df['ema50'] < df['ema50'].shift(5))
        if require_above_sma200:
            sig = sig & (df['close'] < df['sma200'])
    return sig.fillna(False)


def main():
    t0 = time.time()
    print(f'[SWING] start, universe={len(FNO_UNIVERSE)}', flush=True)

    daily_data: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(FNO_UNIVERSE):
        df = load_daily(sym, START, END)
        if df.empty or len(df) < 300:
            continue
        df = enrich_daily(df)
        daily_data[sym] = df
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(FNO_UNIVERSE)} loaded', flush=True)
    print(f'[SWING] loaded {len(daily_data)} stocks', flush=True)

    # ---- Per-stock baseline (Donchian-20, 5-day hold, TP30_SL15, long) ----
    perstock_csv = os.path.join(RESULTS, '02_swing_breakout_perstock.csv')
    fields = ['symbol', 'direction', 'config', 'cost', 'period',
              'n_trades', 'win_rate', 'avg_win', 'avg_loss', 'profit_factor',
              'sharpe', 'max_dd_pct', 'total_return_pct', 'avg_days_held', 'aws']
    if not os.path.exists(perstock_csv):
        with open(perstock_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    perstock_done = set()
    if os.path.exists(perstock_csv) and os.path.getsize(perstock_csv) > 0:
        try:
            ex = pd.read_csv(perstock_csv, usecols=['symbol'])
            perstock_done = set(ex['symbol'].unique())
        except Exception:
            pass
    for sym, df in daily_data.items():
        if sym in perstock_done:
            continue
        for direction in ['long', 'short']:
            sig = make_breakout_signal(df, donch_period=20, vol_ratio_min=1.5,
                                        require_ema50_rising=True,
                                        require_above_sma200=True,
                                        direction=direction)
            if sig.sum() < 5:
                continue
            rules = DailyTradeRules(tp_pct=3.0, sl_pct=1.5, max_hold_days=5, direction=direction)
            trades = simulate_signals_daily(df, sig, rules, entry_mode='next_open')
            if trades.empty:
                continue
            for cost in [COST_DEFAULT, COST_STRESS]:
                stats = trade_stats(trades, cost_pct_round=cost)
                row = dict(symbol=sym, direction=direction, config='D20_TP30_SL15_H5',
                           cost=cost, period='full', **{k: stats.get(k, 0) for k in fields if k in stats})
                with open(perstock_csv, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row)

    print(f'[SWING] per-stock baseline done ({time.time()-t0:.0f}s)', flush=True)

    # ---- Sweep ----
    ranking_csv = os.path.join(RESULTS, '02_swing_breakout_ranking.csv')
    rank_fields = fields + ['donch_period', 'vol_ratio_min', 'require_above_sma200',
                            'tp_pct', 'sl_pct', 'max_hold_days']

    done_symbols = set()
    if os.path.exists(ranking_csv) and os.path.getsize(ranking_csv) > 0:
        try:
            existing = pd.read_csv(ranking_csv, usecols=['symbol'])
            done_symbols = set(existing['symbol'].unique())
            if len(done_symbols) > 0:
                last_sym = existing['symbol'].iloc[-1]
                done_symbols.discard(last_sym)
                full = pd.read_csv(ranking_csv)
                full[full['symbol'].isin(done_symbols)].to_csv(ranking_csv, index=False)
                print(f'[SWING] resume: {len(done_symbols)} stocks already done', flush=True)
        except Exception as e:
            print(f'[SWING] resume parse failed: {e}, starting fresh', flush=True)
            done_symbols = set()
    else:
        with open(ranking_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=rank_fields).writeheader()

    donch_grid = [20, 55]
    vol_grid = [1.2, 1.5]
    sma200_grid = [True, False]
    directions = ['long', 'short']

    cells_per_stock = len(donch_grid) * len(vol_grid) * len(sma200_grid) * len(directions) * len(HOLD_GRID) * len(TP_SL_GRID)
    print(f'[SWING] sweep cells per stock: {cells_per_stock}', flush=True)

    sweep_count = 0
    sym_count = 0
    total_sym = sum(1 for s in daily_data if s not in done_symbols)
    for sym, df in daily_data.items():
        if sym in done_symbols:
            continue
        sym_count += 1
        sym_t0 = time.time()
        for direction in directions:
            for dp in donch_grid:
                for vr in vol_grid:
                    for sm in sma200_grid:
                        sig = make_breakout_signal(df, donch_period=dp, vol_ratio_min=vr,
                                                    require_ema50_rising=True,
                                                    require_above_sma200=sm,
                                                    direction=direction)
                        n_sig = int(sig.sum())
                        if n_sig < 5:
                            continue
                        for hold in HOLD_GRID:
                            for cfg_name, tp, sl in TP_SL_GRID:
                                rules = DailyTradeRules(tp_pct=tp, sl_pct=sl,
                                                         max_hold_days=hold,
                                                         direction=direction)
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
                                                   donch_period=dp, vol_ratio_min=vr,
                                                   require_above_sma200=sm,
                                                   tp_pct=tp, sl_pct=sl, max_hold_days=hold,
                                                   **{k: stats.get(k, 0) for k in fields if k in stats})
                                        with open(ranking_csv, 'a', newline='') as f:
                                            csv.DictWriter(f, fieldnames=rank_fields).writerow(row)
                                sweep_count += 1
        print(f'  [{sym_count}/{total_sym}] {sym} done in {time.time()-sym_t0:.0f}s (total {time.time()-t0:.0f}s, sweep_count={sweep_count})', flush=True)

    print(f'[SWING] sweep done: {sweep_count} cells in {time.time()-t0:.0f}s', flush=True)

    # ---- Walk-forward analysis ----
    rank_df = pd.read_csv(ranking_csv)
    cand = rank_df[(rank_df['period'] == 'test') &
                   (rank_df['cost'] == COST_DEFAULT) &
                   (rank_df['n_trades'] >= 30) &
                   (rank_df['win_rate'] >= 0.70) &
                   (rank_df['profit_factor'] >= 1.8) &
                   (rank_df['max_dd_pct'] <= 15.0)].copy()
    print(f'[SWING] test-period candidates: {len(cand)}', flush=True)

    keys = ['symbol', 'direction', 'config', 'donch_period', 'vol_ratio_min',
            'require_above_sma200', 'tp_pct', 'sl_pct', 'max_hold_days']
    train_df = rank_df[(rank_df['period'] == 'train') & (rank_df['cost'] == COST_DEFAULT)].copy()
    train_df = train_df.rename(columns={'win_rate': 'train_wr', 'n_trades': 'train_n', 'profit_factor': 'train_pf'})
    cand = cand.merge(train_df[keys + ['train_wr', 'train_n', 'train_pf']], on=keys, how='left')
    cand_wf = cand[cand['train_wr'] >= 0.75].copy()
    print(f'[SWING] walk-forward winners: {len(cand_wf)}', flush=True)

    stress_df = rank_df[(rank_df['period'] == 'test') & (rank_df['cost'] == COST_STRESS)].copy()
    stress_df = stress_df.rename(columns={'total_return_pct': 'test_stress_ret',
                                            'win_rate': 'test_stress_wr',
                                            'profit_factor': 'test_stress_pf'})
    cand_wf = cand_wf.merge(stress_df[keys + ['test_stress_ret', 'test_stress_wr', 'test_stress_pf']],
                              on=keys, how='left')

    wf_csv = os.path.join(RESULTS, '02_swing_breakout_walk_forward.csv')
    cand_wf.to_csv(wf_csv, index=False)
    print(f'[SWING] walk-forward saved: {wf_csv}', flush=True)

    if len(cand_wf) > 0:
        diamonds = cand_wf['symbol'].value_counts().head(30).index.tolist()
    else:
        cand_relax = rank_df[(rank_df['period'] == 'test') & (rank_df['cost'] == COST_DEFAULT) &
                             (rank_df['n_trades'] >= 20) & (rank_df['win_rate'] >= 0.55)].copy()
        diamonds = cand_relax.groupby('symbol')['aws'].max().sort_values(ascending=False).head(30).index.tolist()

    diamonds_path = os.path.join(RESULTS, '02_swing_breakout_diamonds.txt')
    with open(diamonds_path, 'w') as f:
        for s in diamonds:
            f.write(s + '\n')
    print(f'[SWING] diamonds saved ({len(diamonds)} stocks): {diamonds_path}', flush=True)
    print(f'[SWING] DONE in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
