"""Pattern 4b — PEAD with REAL earnings dates — FAST sweep (vectorized).

Approach: precompute one EVENT TABLE (one row per real earnings event), with
columns: symbol, ann_date, entry_idx (in price df), entry_price, ann_return,
vol_ratio, above_sma200, surprise_pct + numpy arrays of high/low/close per
symbol. Then for each (tp, sl, hold) variant, vectorize the TP/SL hit
detection across all events (not a Python loop).

Final master table: (events x tp/sl/hold) trade outcomes — saved to CSV.

Then sweep aggregations are pure pandas groupby filtering — fast.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _engine_daily_my as E  # type: ignore


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))
os.makedirs(RESULTS_DIR, exist_ok=True)

EARNINGS_CSV = os.path.join(RESULTS_DIR, '04b_earnings_dates.csv')
EVENTS_CSV = os.path.join(RESULTS_DIR, '04b_pead_events.csv')
MASTER_CSV = os.path.join(RESULTS_DIR, '04b_pead_trades_master.csv')
PERSTOCK_CSV = os.path.join(RESULTS_DIR, '04b_pead_perstock.csv')
RANKING_CSV = os.path.join(RESULTS_DIR, '04b_pead_ranking.csv')
WF_CSV = os.path.join(RESULTS_DIR, '04b_pead_walk_forward.csv')
DIAMONDS_TXT = os.path.join(RESULTS_DIR, '04b_pead_diamonds.txt')

TRAIN_START = '2018-01-01'
TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2026-03-19'

COST_PCT = 0.20  # round-trip CNC delivery cost


# ---------------------------------------------------------------------------
# Pass 1a — build EVENT table (one row per earnings event, per stock)
# ---------------------------------------------------------------------------

def build_events(cache: Dict[str, pd.DataFrame],
                  earnings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """For every (symbol, announcement_date) pair, find the next-trading-day
    reaction and record:
      - entry_idx in df (we enter on entry_idx open)
      - entry_price, ann_return_pct, vol_ratio, above/below SMA200, surprise%

    Also returns a per-symbol dict of pre-extracted numpy arrays of OHLC + idx.
    """
    print('[events] building event table...', flush=True)
    rows = []
    sym_arrays = {}
    for sym, df in cache.items():
        sym_earn = earnings_df[earnings_df['symbol'] == sym]
        if df.empty or sym_earn.empty:
            continue
        idx = df.index
        # Pre-cache numpy arrays
        arrs = {
            'open': df['open'].to_numpy(dtype=np.float64),
            'high': df['high'].to_numpy(dtype=np.float64),
            'low': df['low'].to_numpy(dtype=np.float64),
            'close': df['close'].to_numpy(dtype=np.float64),
            'ret_pct': df['ret_pct'].to_numpy(dtype=np.float64),
            'vol_ratio': df['vol_ratio'].to_numpy(dtype=np.float64),
            'sma200': df['sma200'].to_numpy(dtype=np.float64),
            'idx': idx.to_numpy(),
        }
        sym_arrays[sym] = arrs
        N = len(arrs['close'])
        for _, erow in sym_earn.iterrows():
            adate = erow['announcement_date']
            if pd.isna(adate):
                continue
            adate = pd.Timestamp(adate)
            pos = idx.searchsorted(adate, side='left')
            if pos >= N:
                continue
            reaction_day = idx[pos]
            gap_days = (reaction_day - adate).days
            if gap_days > 7:
                continue
            i = pos
            if i + 1 >= N:
                continue
            entry_i = i + 1
            entry_price = arrs['open'][entry_i]
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            close_i = arrs['close'][i]
            sma200 = arrs['sma200'][i]
            above = bool(np.isfinite(sma200)) and (close_i > sma200)
            below = bool(np.isfinite(sma200)) and (close_i < sma200)
            ann_ret = float(arrs['ret_pct'][i]) if np.isfinite(arrs['ret_pct'][i]) else 0.0
            vol_ratio = float(arrs['vol_ratio'][i]) if np.isfinite(arrs['vol_ratio'][i]) else 0.0
            surprise = float(erow['surprise_pct']) if pd.notna(erow.get('surprise_pct')) else np.nan
            rows.append({
                'symbol': sym,
                'ann_date': adate.strftime('%Y-%m-%d'),
                'reaction_day': pd.Timestamp(reaction_day).strftime('%Y-%m-%d'),
                'entry_date': pd.Timestamp(idx[entry_i]).strftime('%Y-%m-%d'),
                'entry_idx': int(entry_i),
                'entry_price': float(entry_price),
                'ann_return_pct': ann_ret,
                'vol_ratio': vol_ratio,
                'above_sma200': above,
                'below_sma200': below,
                'surprise_pct': surprise,
            })
    events = pd.DataFrame(rows)
    print(f'[events] {len(events)} events across {events["symbol"].nunique()} symbols', flush=True)
    return events, sym_arrays


# ---------------------------------------------------------------------------
# Pass 1b — for each (tp, sl, hold), vectorized TP/SL/TIME hit detection
# ---------------------------------------------------------------------------

def simulate_event_outcomes(events: pd.DataFrame,
                              sym_arrays: Dict[str, dict],
                              side: str, tp_pct: float, sl_pct: float,
                              hold: int) -> pd.DataFrame:
    """Vectorized within each symbol: for all events of that symbol,
    compute TP/SL hit + exit price/return for given (side, tp, sl, hold)."""
    is_long = side == 'long'
    tp_mult = (1 + tp_pct / 100) if is_long else (1 - tp_pct / 100)
    sl_mult = (1 - sl_pct / 100) if is_long else (1 + sl_pct / 100)

    out_rows = []
    for sym, sym_df in events.groupby('symbol', sort=False):
        if sym not in sym_arrays:
            continue
        arrs = sym_arrays[sym]
        N = len(arrs['close'])
        entry_idxs = sym_df['entry_idx'].to_numpy()
        entry_prices = sym_df['entry_price'].to_numpy()
        # For each event, scan k=1..hold day-by-day; this loop is per-event
        # but is in tight numpy with no pandas. Cython-like speed.
        for i_event, (e_i, ep) in enumerate(zip(entry_idxs, entry_prices)):
            tp_p = ep * tp_mult
            sl_p = ep * sl_mult
            last_i = min(e_i + hold, N - 1)
            offset = -1
            exit_price = 0.0
            exit_reason = 'TIME'
            for k in range(1, last_i - e_i + 1):
                j = e_i + k
                hi = arrs['high'][j]
                lo = arrs['low'][j]
                if is_long:
                    tp_hit = hi >= tp_p
                    sl_hit = lo <= sl_p
                else:
                    tp_hit = lo <= tp_p
                    sl_hit = hi >= sl_p
                if sl_hit:
                    offset = k
                    exit_price = sl_p
                    exit_reason = 'SL'
                    break
                if tp_hit:
                    offset = k
                    exit_price = tp_p
                    exit_reason = 'TP'
                    break
            if offset == -1:
                offset = last_i - e_i
                exit_price = arrs['close'][last_i]
                exit_reason = 'TIME'
            if not np.isfinite(exit_price) or exit_price <= 0:
                continue
            if is_long:
                ret = (exit_price - ep) / ep * 100.0
            else:
                ret = (ep - exit_price) / ep * 100.0
            out_rows.append((i_event, sym, side, tp_pct, sl_pct, hold,
                             float(exit_price), exit_reason, int(offset),
                             float(ret)))
    cols = ['_event_pos', 'symbol', 'side', 'tp', 'sl', 'hold',
            'exit_price', 'exit_reason', 'days_held', 'ret_pct']
    return pd.DataFrame(out_rows, columns=cols)


def build_master(events: pd.DataFrame, sym_arrays: Dict[str, dict],
                  tpsl_pairs: List[Tuple[float, float]],
                  hold_levels: List[int]) -> pd.DataFrame:
    """Build the master trade table by stacking all (side, tp, sl, hold) variants."""
    print(f'[master] building {2 * len(tpsl_pairs) * len(hold_levels)} variants '
          f'over {len(events)} events...', flush=True)
    t0 = time.time()
    parts = []
    n_variants = 2 * len(tpsl_pairs) * len(hold_levels)
    k = 0
    for side in ('long', 'short'):
        for (tp, sl) in tpsl_pairs:
            for hold in hold_levels:
                k += 1
                tx = simulate_event_outcomes(events, sym_arrays, side, tp, sl, hold)
                if len(tx) == 0:
                    continue
                # Attach event-context columns by joining on (symbol, _event_pos)
                # Faster: re-use events index.
                # But events_pos was per-symbol. Easier: just merge on (symbol, _event_pos)
                # where _event_pos is the within-symbol order. We'll add it to events too.
                parts.append(tx)
                print(f'  variant {k}/{n_variants} {side} tp={tp} sl={sl} hold={hold} '
                      f'-> {len(tx)} trades ({time.time()-t0:.1f}s)', flush=True)
    master = pd.concat(parts, ignore_index=True)

    # Attach event context columns (ann_return_pct, vol_ratio, above/below_sma200, surprise%)
    # by mapping (symbol, _event_pos) back to events positions.
    events_indexed = events.copy()
    events_indexed['_event_pos'] = events_indexed.groupby('symbol').cumcount()
    keep = ['symbol', '_event_pos', 'ann_date', 'entry_date',
            'ann_return_pct', 'vol_ratio', 'above_sma200', 'below_sma200',
            'surprise_pct']
    master = master.merge(events_indexed[keep], on=['symbol', '_event_pos'], how='left')
    master = master.drop(columns=['_event_pos'])
    print(f'[master] {len(master)} rows in {time.time()-t0:.1f}s', flush=True)
    return master


# ---------------------------------------------------------------------------
# Pass 2 — fast pandas filter aggregations
# ---------------------------------------------------------------------------

def trades_stats(trades: pd.DataFrame, cost_pct: float = COST_PCT) -> dict:
    if trades is None or len(trades) == 0:
        return dict(n_trades=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                    profit_factor=0.0, sharpe=0.0, max_dd_pct=0.0,
                    total_return_pct=0.0, avg_days_held=0.0, aws=0.0,
                    n_stocks=0)
    net = trades['ret_pct'].to_numpy() - cost_pct
    wins = net > 0
    losses = ~wins
    wr = float(wins.mean())
    avg_win = float(net[wins].mean()) if wins.any() else 0.0
    avg_loss = float(net[losses].mean()) if losses.any() else 0.0
    gp = float(net[wins].sum())
    gl = float(-net[losses].sum())
    pf = gp / gl if gl > 0 else (np.inf if gp > 0 else 0.0)
    eq = net.cumsum()
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0.0
    if len(net) > 1 and net.std() > 0:
        avg_days = max(float(trades['days_held'].mean()), 1.0)
        sharpe = float(net.mean() / net.std() * np.sqrt(252.0 / avg_days))
    else:
        sharpe = 0.0
    total_ret = float(net.sum())
    avg_days = float(trades['days_held'].mean()) if len(trades) else 0.0
    aws = (
        wr
        * float(np.log1p(len(trades)))
        * float(min((pf if np.isfinite(pf) else 999) / 2, 1.5))
        * float(min(max(sharpe, 0) / 1.5, 1.5))
    )
    return dict(
        n_trades=int(len(trades)), win_rate=wr,
        avg_win=avg_win, avg_loss=avg_loss,
        profit_factor=float(pf if np.isfinite(pf) else 999.0),
        sharpe=sharpe, max_dd_pct=max_dd, total_return_pct=total_ret,
        avg_days_held=avg_days, aws=aws,
        n_stocks=int(trades['symbol'].nunique()),
    )


def filter_master(master: pd.DataFrame, side: str, tp: float, sl: float,
                   hold: int, ret_thresh: float, vol_min: float,
                   use_sma: bool, use_surprise: bool,
                   surprise_thresh: float = 0.0,
                   period_start: str = None, period_end: str = None) -> pd.DataFrame:
    df = master[
        (master['side'] == side)
        & (master['tp'] == tp)
        & (master['sl'] == sl)
        & (master['hold'] == hold)
    ]
    if side == 'long':
        df = df[df['ann_return_pct'] >= ret_thresh]
    else:
        df = df[df['ann_return_pct'] <= ret_thresh]
    if vol_min > 0:
        df = df[df['vol_ratio'] >= vol_min]
    if use_sma:
        if side == 'long':
            df = df[df['above_sma200']]
        else:
            df = df[df['below_sma200']]
    if use_surprise:
        if side == 'long':
            df = df[df['surprise_pct'].notna() & (df['surprise_pct'] >= surprise_thresh)]
        else:
            df = df[df['surprise_pct'].notna() & (df['surprise_pct'] <= surprise_thresh)]
    if period_start is not None:
        df = df[df['entry_date'] >= period_start]
    if period_end is not None:
        df = df[df['entry_date'] <= period_end]
    return df


# ---------------------------------------------------------------------------
# Stage 1 — per-stock at defaults
# ---------------------------------------------------------------------------

def stage1_perstock(master: pd.DataFrame) -> None:
    print('[stage1] per-stock at defaults', flush=True)
    fields = ['symbol', 'side', 'n_trades', 'win_rate', 'avg_win', 'avg_loss',
              'profit_factor', 'sharpe', 'max_dd_pct', 'total_return_pct',
              'aws', 'avg_days_held', 'n_stocks']
    if os.path.exists(PERSTOCK_CSV):
        os.remove(PERSTOCK_CSV)
    with open(PERSTOCK_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sym in sorted(master['symbol'].unique()):
            for side in ('long', 'short'):
                sub = filter_master(master, side=side, tp=8.0, sl=5.0, hold=15,
                                     ret_thresh=0, vol_min=0,
                                     use_sma=False, use_surprise=False)
                sub = sub[sub['symbol'] == sym]
                stats = trades_stats(sub)
                w.writerow({'symbol': sym, 'side': side, **stats})
    print('[stage1] done', flush=True)


# ---------------------------------------------------------------------------
# Stage 3 — full sweep
# ---------------------------------------------------------------------------

def run_sweep(master: pd.DataFrame, period_start: str, period_end: str,
                tag: str = 'train') -> pd.DataFrame:
    print(f'[stage3] sweep {tag} {period_start}..{period_end}', flush=True)
    long_ret = [0, 1, 2, 3, 5, 7]
    short_ret = [0, -1, -2, -3, -5, -7]
    vol_levels = [0, 1.5]
    sma_filters = [False, True]
    surprise_options = [False, True]
    surprise_threshes = [0, 5, 10, 20]
    tpsl_pairs = sorted(set(zip(master['tp'], master['sl'])))
    hold_levels = sorted(master['hold'].unique().tolist())

    fields = ['side', 'ret_thresh', 'vol_min', 'use_sma', 'use_surprise',
              'surprise_thresh', 'tp', 'sl', 'hold',
              'n_trades', 'win_rate', 'avg_win', 'avg_loss',
              'profit_factor', 'sharpe', 'max_dd_pct',
              'total_return_pct', 'avg_days_held', 'aws',
              'n_stocks']
    out_csv = RANKING_CSV if tag == 'train' else RANKING_CSV.replace('.csv', f'_{tag}.csv')
    if os.path.exists(out_csv):
        os.remove(out_csv)
    fout = open(out_csv, 'w', newline='')
    w = csv.DictWriter(fout, fieldnames=fields)
    w.writeheader()
    rows = []
    t0 = time.time()
    n_cells = 0
    for side in ('long', 'short'):
        ret_levels = long_ret if side == 'long' else short_ret
        for ret_t, vol_t, sma_f, use_sur, sur_t, (tp, sl), hold in product(
                ret_levels, vol_levels, sma_filters, surprise_options,
                surprise_threshes, tpsl_pairs, hold_levels):
            if not use_sur and sur_t != 0:
                continue
            sur_signed = sur_t if side == 'long' else -sur_t
            sub = filter_master(master, side=side, tp=tp, sl=sl, hold=hold,
                                 ret_thresh=ret_t, vol_min=vol_t,
                                 use_sma=sma_f, use_surprise=use_sur,
                                 surprise_thresh=sur_signed,
                                 period_start=period_start, period_end=period_end)
            stats = trades_stats(sub)
            row = {'side': side, 'ret_thresh': ret_t, 'vol_min': vol_t,
                   'use_sma': sma_f, 'use_surprise': use_sur,
                   'surprise_thresh': sur_t,
                   'tp': tp, 'sl': sl, 'hold': hold, **stats}
            w.writerow(row)
            rows.append(row)
            n_cells += 1
        fout.flush()
        print(f'  {side}: {n_cells} cells done ({time.time()-t0:.1f}s)', flush=True)
    fout.close()
    print(f'[stage3] {tag} done {n_cells} rows in {time.time()-t0:.1f}s', flush=True)
    return pd.DataFrame(rows)


def stage4_walk_forward(sweep_train: pd.DataFrame,
                          sweep_test: pd.DataFrame) -> pd.DataFrame:
    print('[stage4] walk-forward join', flush=True)
    keys = ['side', 'ret_thresh', 'vol_min', 'use_sma', 'use_surprise',
            'surprise_thresh', 'tp', 'sl', 'hold']
    train_renamed = sweep_train.rename(columns={
        'n_trades': 'train_n', 'win_rate': 'train_wr',
        'profit_factor': 'train_pf', 'max_dd_pct': 'train_dd',
        'total_return_pct': 'train_ret', 'aws': 'train_aws',
    })[keys + ['train_n', 'train_wr', 'train_pf', 'train_dd', 'train_ret', 'train_aws']]
    test_renamed = sweep_test.rename(columns={
        'n_trades': 'test_n', 'win_rate': 'test_wr',
        'profit_factor': 'test_pf', 'max_dd_pct': 'test_dd',
        'total_return_pct': 'test_ret', 'aws': 'test_aws',
    })[keys + ['test_n', 'test_wr', 'test_pf', 'test_dd', 'test_ret', 'test_aws']]
    merged = train_renamed.merge(test_renamed, on=keys, how='inner')
    merged['passes_wf'] = (
        (merged['train_n'] >= 30)
        & (merged['train_wr'] >= 0.75)
        & (merged['test_n'] >= 30)
        & (merged['test_wr'] >= 0.70)
        & (merged['test_pf'] >= 1.8)
        & (merged['test_dd'] <= 15.0)
    )
    merged.to_csv(WF_CSV, index=False)
    survivors = merged[merged['passes_wf']]
    print(f'[stage4] {len(merged)} cells, {len(survivors)} pass walk-forward', flush=True)
    return merged


def main():
    t0 = time.time()
    print('[main] PEAD FAST pipeline (vectorized)', flush=True)

    if not os.path.exists(EARNINGS_CSV):
        print(f'[main] FATAL: {EARNINGS_CSV} missing.', flush=True)
        sys.exit(1)
    earnings_df = pd.read_csv(EARNINGS_CSV)
    earnings_df['announcement_date'] = pd.to_datetime(earnings_df['announcement_date'])
    earnings_df['surprise_pct'] = pd.to_numeric(earnings_df['surprise_pct'], errors='coerce')
    print(f'[main] {len(earnings_df)} earnings rows / '
          f'{earnings_df["symbol"].nunique()} symbols', flush=True)

    print('[main] loading + enriching F&O daily data...', flush=True)
    syms = E.list_daily_universe(E.FNO_UNIVERSE, start='2017-01-01',
                                  end='2026-03-19', min_rows=1500)
    cache = E.load_many_daily(syms)
    for sym in list(cache.keys()):
        cache[sym] = E.enrich_daily(cache[sym])
    print(f'[main] {len(cache)} stocks ({time.time()-t0:.0f}s)', flush=True)

    # Build master: ~2371 events x 2 sides x 8 tpsl x 5 holds = 190K trades
    tpsl_pairs = [(5, 3), (6, 4), (8, 5), (10, 5), (10, 7), (15, 7), (15, 10), (20, 10)]
    hold_levels = [5, 10, 15, 20, 30]

    if os.path.exists(MASTER_CSV):
        master = pd.read_csv(MASTER_CSV)
        print(f'[main] loaded master from CSV: {master.shape}', flush=True)
    else:
        events, sym_arrays = build_events(cache, earnings_df)
        events.to_csv(EVENTS_CSV, index=False)
        master = build_master(events, sym_arrays, tpsl_pairs, hold_levels)
        master.to_csv(MASTER_CSV, index=False)
    print(f'[main] master shape: {master.shape}', flush=True)

    stage1_perstock(master)

    sweep_train = run_sweep(master, period_start=TRAIN_START, period_end=TRAIN_END, tag='train')
    sweep_test = run_sweep(master, period_start=TEST_START, period_end=TEST_END, tag='test')

    wf = stage4_walk_forward(sweep_train, sweep_test)

    # Cost stress on top 50 by test_wr
    top50 = wf[wf['test_n'] >= 30].sort_values(['test_wr', 'test_pf'], ascending=False).head(50)
    stress_rows = []
    for _, r in top50.iterrows():
        sub = filter_master(master, side=r['side'], tp=r['tp'], sl=r['sl'],
                             hold=r['hold'], ret_thresh=r['ret_thresh'],
                             vol_min=r['vol_min'], use_sma=bool(r['use_sma']),
                             use_surprise=bool(r['use_surprise']),
                             surprise_thresh=r['surprise_thresh'] if r['side'] == 'long' else -r['surprise_thresh'],
                             period_start=TEST_START, period_end=TEST_END)
        st = trades_stats(sub, cost_pct=0.30)
        stress_rows.append({**r.to_dict(),
                             'stress_n': st['n_trades'],
                             'stress_wr': st['win_rate'],
                             'stress_pf': st['profit_factor'],
                             'stress_ret': st['total_return_pct']})
    pd.DataFrame(stress_rows).to_csv(WF_CSV.replace('.csv', '_top50_stressed.csv'), index=False)

    # Diamonds
    survivors = wf[wf['passes_wf']].sort_values('test_aws', ascending=False)
    lines = []
    lines.append('# Pattern 4b — PEAD with REAL earnings dates — Walk-Forward Diamonds\n')
    lines.append(f'Generated: {time.strftime("%Y-%m-%d %H:%M IST")}\n')
    lines.append(f'Universe: {len(cache)} F&O stocks; train 2018-2023, test 2024-2025\n')
    lines.append(f'Earnings source: yfinance .NS get_earnings_dates ({len(earnings_df)} rows)\n')
    lines.append(f'Master trade table: {len(master)} rows\n')
    lines.append(f'Walk-forward gates: train_n>=30, train_wr>=75%, test_n>=30, '
                 f'test_wr>=70%, test_pf>=1.8, test_dd<=15%\n\n')

    if len(survivors) == 0:
        lines.append('## NO walk-forward survivors at gates.\n\n')
    else:
        lines.append(f'## {len(survivors)} walk-forward survivors\n\n')
        for _, s in survivors.head(30).iterrows():
            lines.append(
                f'- {s["side"]} ret={s["ret_thresh"]} vol>={s["vol_min"]} '
                f'sma_filter={s["use_sma"]} use_surprise={s["use_surprise"]} '
                f'surprise_thresh={s["surprise_thresh"]} '
                f'tp={s["tp"]}% sl={s["sl"]}% hold={s["hold"]}d\n'
                f'  Train: n={s["train_n"]} WR={s["train_wr"]:.1%} PF={s["train_pf"]:.2f} '
                f'DD={s["train_dd"]:.1f}% Ret={s["train_ret"]:.1f}%\n'
                f'  Test:  n={s["test_n"]} WR={s["test_wr"]:.1%} PF={s["test_pf"]:.2f} '
                f'DD={s["test_dd"]:.1f}% Ret={s["test_ret"]:.1f}%\n\n'
            )

    lines.append('## Top 20 by test WR (n>=20, regardless of pass)\n\n')
    top_wr = wf[wf['test_n'] >= 20].sort_values('test_wr', ascending=False).head(20)
    for _, s in top_wr.iterrows():
        lines.append(
            f'- {s["side"]} ret={s["ret_thresh"]} vol>={s["vol_min"]} sma={s["use_sma"]} '
            f'sur={s["use_surprise"]}/{s["surprise_thresh"]} tp={s["tp"]} sl={s["sl"]} '
            f'hold={s["hold"]} | train n={s["train_n"]} WR={s["train_wr"]:.1%} | '
            f'test n={s["test_n"]} WR={s["test_wr"]:.1%} PF={s["test_pf"]:.2f} '
            f'DD={s["test_dd"]:.1f}% Ret={s["test_ret"]:.1f}%\n'
        )

    lines.append('\n## Top 20 by test_aws (n>=30)\n\n')
    top_aws = wf[wf['test_n'] >= 30].sort_values('test_aws', ascending=False).head(20)
    for _, s in top_aws.iterrows():
        lines.append(
            f'- {s["side"]} ret={s["ret_thresh"]} vol>={s["vol_min"]} sma={s["use_sma"]} '
            f'sur={s["use_surprise"]}/{s["surprise_thresh"]} tp={s["tp"]} sl={s["sl"]} '
            f'hold={s["hold"]} | train n={s["train_n"]} WR={s["train_wr"]:.1%} PF={s["train_pf"]:.2f} | '
            f'test n={s["test_n"]} WR={s["test_wr"]:.1%} PF={s["test_pf"]:.2f} '
            f'DD={s["test_dd"]:.1f}% Ret={s["test_ret"]:.1f}%\n'
        )

    with open(DIAMONDS_TXT, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'[main] DONE in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
