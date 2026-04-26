"""Signal 3 — Relative-strength leadership.

Stock's N-day return MINUS NIFTYBEES N-day return must exceed Z standard
deviations of its own 252-day distribution.

Variants (5):
  rs_30d_z1.5         30-day RS lookback, 1.5 sigma
  rs_60d_z1.5         60-day RS lookback, 1.5 sigma   (default)
  rs_60d_z2.0         60-day RS lookback, 2.0 sigma   (stricter)
  rs_60d_z1.5_200sma  + 200 SMA regime filter
  rs_60d_z1.5_vol_2x  + volume confirmation (>= 2x avg)

Direction: LONG.
"""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SHARED = HERE.parents[1] / '22_fno_bearish_breakdown' / 'scripts'
sys.path.insert(0, str(SHARED))

from _shared_engine import (   # type: ignore
    load_universe, load_all_bars, run_engine, compute_metrics,
    write_trades, write_summary, heartbeat, START_DATE, END_DATE,
)

RES = HERE.parent / 'results'
LOGS = HERE.parent / 'logs'
RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
HB = LOGS / 'sweep_heartbeat.txt'
UNI_CSV = RES / 'universe.csv'


def make_cfg(name, lookback=60, z_thr=1.5, use_regime=False, use_vol=False,
             vol_mult=2.0, target_pct=0.25, hard_stop_pct=0.08, cost_pct=0.0020):
    return dict(name=name, lookback=lookback, z_thr=z_thr,
                use_regime=use_regime, use_vol=use_vol, vol_mult=vol_mult,
                target_pct=target_pct, hard_stop_pct=hard_stop_pct, cost_pct=cost_pct)


VARIANTS = [
    make_cfg('rs_30d_z1.5',         lookback=30, z_thr=1.5),
    make_cfg('rs_60d_z1.5',         lookback=60, z_thr=1.5),
    make_cfg('rs_60d_z2.0',         lookback=60, z_thr=2.0),
    make_cfg('rs_60d_z1.5_200sma',  lookback=60, z_thr=1.5, use_regime=True),
    make_cfg('rs_60d_z1.5_vol_2x',  lookback=60, z_thr=1.5, use_vol=True, vol_mult=2.0),
]


def signal_fn(row: pd.Series, cfg: dict) -> bool:
    if cfg['lookback'] == 30:
        z = row.get('rs_30_z')
    else:
        z = row.get('rs_60_z')
    if pd.isna(z):
        return False
    if not (z >= cfg['z_thr']):
        return False
    if cfg.get('use_regime'):
        sma200 = row.get('sma_200')
        if pd.isna(sma200) or row['close'] <= sma200:
            return False
    if cfg.get('use_vol'):
        vavg = row.get('vol_avg')
        if pd.isna(vavg) or vavg <= 0:
            return False
        if not (row['volume'] >= cfg['vol_mult'] * vavg):
            return False
    return True


def rank_fn(row: pd.Series, cfg: dict) -> float:
    """Rank candidates by RS z-score (highest first)."""
    if cfg['lookback'] == 30:
        z = row.get('rs_30_z', 0)
    else:
        z = row.get('rs_60_z', 0)
    return float(z) if pd.notna(z) else 0.0


def main():
    t_start = time.time()
    heartbeat(HB, 'start')
    uni = load_universe(UNI_CSV); bars = load_all_bars(uni)
    print(f'Loaded {len(bars)} stocks; period {START_DATE} -> {END_DATE}', flush=True)

    summary_rows = []
    print(f'\n{"Variant":>22} {"Trades":>7} {"WR%":>6} {"PF":>6} {"AvgD":>6} '
          f'{"CAGR%":>7} {"Sharpe":>7} {"MaxDD%":>7} {"Calmar":>7}', flush=True)
    print('-' * 90, flush=True)

    best_pf, best_var, best_trades = -1.0, None, None
    for cfg in VARIANTS:
        heartbeat(HB, f'running {cfg["name"]}')
        t0 = time.time()
        trades, eq = run_engine(bars, cfg, signal_fn, 'LONG',
                                candidate_rank_fn=rank_fn)
        m = compute_metrics(trades, eq)
        summary_rows.append({'variant': cfg['name'], 'direction':'LONG', 'metrics': m})
        print(f'{cfg["name"]:>22} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} '
              f'{m["profit_factor"]:>6.2f} {m["avg_days_held"]:>6.1f} '
              f'{m["cagr_pct"]:>+7.2f} {m["sharpe"]:>7.2f} '
              f'{m["max_dd_pct"]:>7.2f} {m["calmar"]:>7.2f}  ({time.time()-t0:.1f}s)', flush=True)
        eq.to_csv(RES / f'equity_{cfg["name"]}.csv', header=['equity'])
        if m['profit_factor'] > best_pf and m['trades'] >= 5:
            best_pf, best_var, best_trades = m['profit_factor'], cfg['name'], trades

    write_summary(summary_rows, RES / 'summary.csv')
    if best_trades is not None:
        write_trades(best_trades, RES / f'trades_{best_var}.csv')
    print(f'\nBest by PF: {best_var} (PF={best_pf:.2f})')
    print(f'Total runtime: {time.time()-t_start:.1f}s', flush=True)
    heartbeat(HB, f'DONE | best={best_var} PF={best_pf:.2f}')


if __name__ == '__main__':
    sys.exit(main())
