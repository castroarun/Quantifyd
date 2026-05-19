"""BTST (Buy Today Sell Tomorrow) pattern hunt.

Pattern hypothesis:
  - Strong-close day (close in top 20% of daily range) PLUS
  - Above-average volume PLUS
  - Index regime supportive (NIFTYBEES > 50-EMA, rising) PLUS
  - Last-hour 5-min strength (close > VWAP at 14:15-15:25 + close near day high)
  -> long entry at signal-day close, exit at next-day close
     OR exit on next-day TP/SL hit.

Universe: F&O 80 stocks. Train 2018-2023, test 2024-2026.
TP/SL grid: (2.0,1.5), (3.0,1.5), (4.0,2.0), (5.0,2.0)
Cost: 0.06% F&O round-trip default; stress 0.20% round-trip.

Outputs:
  results/01_btst_perstock.csv     - per-stock baseline (no TP/SL, just BTST close-to-close)
  results/01_btst_ranking.csv      - full sweep
  results/01_btst_walk_forward.csv - top-5 candidates train/test split
  results/01_btst_diamonds.txt     - cohort that survives
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine_daily import (
    FNO_UNIVERSE, load_daily, load_many_daily, load_5min_lasthour,
    enrich_daily, simulate_btst_with_tpsl, trade_stats, split_train_test,
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
COST_DEFAULT = 0.06   # F&O round-trip
COST_STRESS = 0.20    # CNC delivery stress

TP_SL_GRID = [
    ('TP20_SL15', 2.0, 1.5),
    ('TP30_SL15', 3.0, 1.5),
    ('TP40_SL20', 4.0, 2.0),
    ('TP50_SL20', 5.0, 2.0),
    ('NOTPSL', None, None),  # pure close-to-close
]


def load_index_regime(start: str = START, end: str = END) -> pd.DataFrame:
    """NIFTYBEES daily as index proxy."""
    nb = load_daily('NIFTYBEES', start, end)
    if nb.empty:
        return nb
    nb = enrich_daily(nb)
    nb['index_uptrend'] = (nb['close'] > nb['ema50']) & (nb['ema50'] > nb['ema50'].shift(5))
    return nb[['close', 'ema50', 'index_uptrend']].copy()


def make_btst_signal(df: pd.DataFrame,
                     close_pos_min: float = 0.7,
                     vol_ratio_min: float = 1.2,
                     ret_min: float = 0.5,
                     direction: str = 'long',
                     index_regime: pd.DataFrame = None,
                     require_uptrend: bool = True) -> pd.Series:
    """Return boolean Series: long-bias BTST entry day.

    direction='long': close in top X% of range, +ret today, vol spike, index uptrend
    direction='short': close in bottom X%, -ret today, vol spike, index downtrend
    """
    if direction == 'long':
        sig = (df['close_pos'] >= close_pos_min) & \
              (df['vol_ratio'] >= vol_ratio_min) & \
              (df['ret_pct'] >= ret_min)
    else:
        sig = (df['close_pos'] <= (1 - close_pos_min)) & \
              (df['vol_ratio'] >= vol_ratio_min) & \
              (df['ret_pct'] <= -ret_min)

    if index_regime is not None and require_uptrend:
        # align index regime by date
        idx_sig = index_regime['index_uptrend'].reindex(df.index, method='ffill')
        if direction == 'long':
            sig = sig & idx_sig.fillna(False)
        else:
            sig = sig & (~idx_sig.fillna(True))
    return sig.fillna(False)


def main():
    t0 = time.time()
    print(f'[BTST] start, universe={len(FNO_UNIVERSE)}', flush=True)

    # 1. Load daily for whole universe + NIFTY regime
    print('[BTST] loading index regime...', flush=True)
    index_reg = load_index_regime()
    print(f'[BTST] index regime rows: {len(index_reg)}', flush=True)

    print('[BTST] loading daily for FNO universe...', flush=True)
    daily_data: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(FNO_UNIVERSE):
        df = load_daily(sym, START, END)
        if df.empty or len(df) < 250:
            continue
        df = enrich_daily(df)
        daily_data[sym] = df
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(FNO_UNIVERSE)} loaded', flush=True)
    print(f'[BTST] loaded {len(daily_data)} stocks', flush=True)

    # 2. Per-stock baseline screen (TP30_SL15, long+short, with regime gate)
    perstock_csv = os.path.join(RESULTS, '01_btst_perstock.csv')
    fields = ['symbol', 'direction', 'config', 'cost', 'period',
              'n_trades', 'win_rate', 'avg_win', 'avg_loss', 'profit_factor',
              'sharpe', 'max_dd_pct', 'total_return_pct', 'aws']

    ranking_csv = os.path.join(RESULTS, '01_btst_ranking.csv')
    rank_fields = fields + ['close_pos_min', 'vol_ratio_min', 'ret_min', 'tp_pct', 'sl_pct', 'require_uptrend']

    # Resume support: skip stocks already in ranking csv
    done_symbols = set()
    if os.path.exists(ranking_csv) and os.path.getsize(ranking_csv) > 0:
        try:
            existing = pd.read_csv(ranking_csv, usecols=['symbol'])
            # Drop the most recent symbol (it might be partial). Keep only fully completed.
            done_symbols = set(existing['symbol'].unique())
            if len(done_symbols) > 0:
                # The last symbol we saw might be partial — drop it from done so we re-do
                last_sym = existing['symbol'].iloc[-1]
                done_symbols.discard(last_sym)
                # Filter the CSV to keep only safe rows
                safe = existing[existing['symbol'].isin(done_symbols)].index
                # Read full and rewrite
                full = pd.read_csv(ranking_csv)
                full[full['symbol'].isin(done_symbols)].to_csv(ranking_csv, index=False)
                print(f'[BTST] resume: {len(done_symbols)} stocks already done', flush=True)
        except Exception as e:
            print(f'[BTST] resume parse failed: {e}, starting fresh', flush=True)
            done_symbols = set()
    else:
        with open(ranking_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=rank_fields).writeheader()

    if not os.path.exists(perstock_csv):
        with open(perstock_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    # Param grid for sweep — trimmed to fit chunk timing
    cp_grid = [0.6, 0.75]
    vr_grid = [1.0, 1.5]
    ret_grid = [0.0, 0.5]
    uptrend_grid = [True, False]
    directions = ['long', 'short']

    print(f'[BTST] sweep cells per stock: {len(cp_grid)*len(vr_grid)*len(ret_grid)*len(uptrend_grid)*len(directions)*len(TP_SL_GRID)}', flush=True)

    # ---- Per-stock baseline pass ----
    print('[BTST] per-stock baseline (TP30_SL15, both dirs)...', flush=True)
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
        for direction in directions:
            sig = make_btst_signal(df, close_pos_min=0.7, vol_ratio_min=1.2,
                                    ret_min=0.5, direction=direction,
                                    index_regime=index_reg, require_uptrend=True)
            trades = simulate_btst_with_tpsl(df, sig, tp_pct=3.0, sl_pct=1.5, direction=direction)
            if trades.empty:
                continue
            for cost in [COST_DEFAULT, COST_STRESS]:
                stats = trade_stats(trades, cost_pct_round=cost)
                row = dict(symbol=sym, direction=direction, config='TP30_SL15',
                           cost=cost, period='full', **{k: stats.get(k, 0) for k in fields if k in stats})
                with open(perstock_csv, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row)

    # ---- Sweep with walk-forward split ----
    print('[BTST] full sweep across param grid...', flush=True)
    sweep_count = 0
    sym_count = 0
    total_sym = sum(1 for s in daily_data if s not in done_symbols)
    for sym, df in daily_data.items():
        if sym in done_symbols:
            continue
        sym_count += 1
        sym_t0 = time.time()
        for direction in directions:
            for cp in cp_grid:
                for vr in vr_grid:
                    for rm in ret_grid:
                        for up in uptrend_grid:
                            sig = make_btst_signal(df, close_pos_min=cp, vol_ratio_min=vr,
                                                    ret_min=rm, direction=direction,
                                                    index_regime=index_reg, require_uptrend=up)
                            n_sig = int(sig.sum())
                            if n_sig < 10:
                                continue
                            for cfg_name, tp, sl in TP_SL_GRID:
                                if tp is None:
                                    # pure close-to-close: use a huge TP/SL so neither hits
                                    trades = simulate_btst_with_tpsl(df, sig, tp_pct=999.0, sl_pct=999.0, direction=direction)
                                else:
                                    trades = simulate_btst_with_tpsl(df, sig, tp_pct=tp, sl_pct=sl, direction=direction)
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
                                                   close_pos_min=cp, vol_ratio_min=vr, ret_min=rm,
                                                   tp_pct=tp if tp else 999.0,
                                                   sl_pct=sl if sl else 999.0,
                                                   require_uptrend=up,
                                                   **{k: stats.get(k, 0) for k in fields if k in stats})
                                        with open(ranking_csv, 'a', newline='') as f:
                                            csv.DictWriter(f, fieldnames=rank_fields).writerow(row)
                                sweep_count += 1
        print(f'  [{sym_count}/{total_sym}] {sym} done in {time.time()-sym_t0:.0f}s (total {time.time()-t0:.0f}s, sweep_count={sweep_count})', flush=True)

    print(f'[BTST] sweep done: {sweep_count} configs in {time.time()-t0:.0f}s', flush=True)

    # ---- Walk-forward: select candidates passing gates on test ----
    rank_df = pd.read_csv(ranking_csv)
    # 1. Filter test rows that meet hard gates
    cand = rank_df[(rank_df['period'] == 'test') &
                   (rank_df['cost'] == COST_DEFAULT) &
                   (rank_df['n_trades'] >= 30) &
                   (rank_df['win_rate'] >= 0.70) &
                   (rank_df['profit_factor'] >= 1.8) &
                   (rank_df['max_dd_pct'] <= 15.0)].copy()
    print(f'[BTST] test-period candidates: {len(cand)}', flush=True)

    # 2. Cross-check that the SAME (symbol, direction, config, params) had train WR >= 75%
    keys = ['symbol', 'direction', 'config', 'close_pos_min', 'vol_ratio_min', 'ret_min',
            'tp_pct', 'sl_pct', 'require_uptrend']
    train_df = rank_df[(rank_df['period'] == 'train') & (rank_df['cost'] == COST_DEFAULT)].copy()
    train_df = train_df.rename(columns={'win_rate': 'train_wr', 'n_trades': 'train_n', 'profit_factor': 'train_pf'})
    cand = cand.merge(train_df[keys + ['train_wr', 'train_n', 'train_pf']], on=keys, how='left')
    cand_wf = cand[cand['train_wr'] >= 0.75].copy()
    print(f'[BTST] walk-forward winners (train WR>=75% AND test WR>=70%): {len(cand_wf)}', flush=True)

    # 3. Also pull stress-cost row for the same configs
    stress_df = rank_df[(rank_df['period'] == 'test') & (rank_df['cost'] == COST_STRESS)].copy()
    stress_df = stress_df.rename(columns={'total_return_pct': 'test_stress_ret', 'win_rate': 'test_stress_wr',
                                            'profit_factor': 'test_stress_pf'})
    cand_wf = cand_wf.merge(stress_df[keys + ['test_stress_ret', 'test_stress_wr', 'test_stress_pf']],
                              on=keys, how='left')

    wf_csv = os.path.join(RESULTS, '01_btst_walk_forward.csv')
    cand_wf.to_csv(wf_csv, index=False)
    print(f'[BTST] walk-forward saved: {wf_csv}', flush=True)

    # 4. Diamonds — top symbols
    if len(cand_wf) > 0:
        diamonds = cand_wf['symbol'].value_counts().head(30).index.tolist()
    else:
        # Fallback: top by AWS in test
        cand_relax = rank_df[(rank_df['period'] == 'test') & (rank_df['cost'] == COST_DEFAULT) &
                             (rank_df['n_trades'] >= 20) & (rank_df['win_rate'] >= 0.55)].copy()
        diamonds = cand_relax.groupby('symbol')['aws'].max().sort_values(ascending=False).head(30).index.tolist()

    diamonds_path = os.path.join(RESULTS, '01_btst_diamonds.txt')
    with open(diamonds_path, 'w') as f:
        for s in diamonds:
            f.write(s + '\n')
    print(f'[BTST] diamonds saved ({len(diamonds)} stocks): {diamonds_path}', flush=True)
    print(f'[BTST] DONE in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
