"""Signal 5 — ADX-based trend strength entry.

ADX(14) just crossed above THR (default 25) from below within last 5 bars,
AND +DI > -DI for LONG, -DI > +DI for SHORT.

Variants (5):
  adx_25_pure                   no extras
  adx_25_above_50sma            + regime filter (long: close>50SMA, short: <)
  adx_25_above_200sma           + 200-SMA regime
  adx_30                        ADX threshold 30 (stricter)
  adx_25_confluence             50-SMA + ADX>25 + DI margin > 5
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


def make_cfg(name, adx_thr=25, regime=None, di_margin=0.0,
             target_pct=0.25, hard_stop_pct=0.08, cost_pct=0.0020):
    return dict(name=name, adx_thr=adx_thr, regime=regime, di_margin=di_margin,
                target_pct=target_pct, hard_stop_pct=hard_stop_pct, cost_pct=cost_pct)


VARIANTS_LONG = [
    make_cfg('adx_25_pure',         adx_thr=25),
    make_cfg('adx_25_above_50sma',  adx_thr=25, regime='sma_50'),
    make_cfg('adx_25_above_200sma', adx_thr=25, regime='sma_200'),
    make_cfg('adx_30',              adx_thr=30),
    make_cfg('adx_25_confluence',   adx_thr=25, regime='sma_50', di_margin=5.0),
]
VARIANTS_SHORT = [{**v, 'name': v['name']} for v in VARIANTS_LONG]


def _annotate_adx_cross(bars):
    """Add adx_cross_up_<thr> flags within last 5 bars for THR in {25, 30}."""
    for sym, df in bars.items():
        adx = df['adx_14']
        for thr in (25, 30):
            cross_up = (adx.shift(1) <= thr) & (adx > thr)
            df[f'adx_cross_{thr}'] = cross_up.rolling(5).sum() > 0


def signal_fn_long(row: pd.Series, cfg: dict) -> bool:
    flag = row.get(f'adx_cross_{cfg["adx_thr"]}', False)
    if not bool(flag):
        return False
    plus_di = row.get('plus_di'); minus_di = row.get('minus_di')
    if pd.isna(plus_di) or pd.isna(minus_di):
        return False
    if not (plus_di - minus_di > cfg.get('di_margin', 0)):
        return False
    if cfg.get('regime'):
        ma = row.get(cfg['regime'])
        if pd.isna(ma) or row['close'] <= ma:
            return False
    return True


def signal_fn_short(row: pd.Series, cfg: dict) -> bool:
    flag = row.get(f'adx_cross_{cfg["adx_thr"]}', False)
    if not bool(flag):
        return False
    plus_di = row.get('plus_di'); minus_di = row.get('minus_di')
    if pd.isna(plus_di) or pd.isna(minus_di):
        return False
    if not (minus_di - plus_di > cfg.get('di_margin', 0)):
        return False
    if cfg.get('regime'):
        ma = row.get(cfg['regime'])
        if pd.isna(ma) or row['close'] >= ma:
            return False
    return True


def main():
    t_start = time.time()
    heartbeat(HB, 'start')
    uni = load_universe(UNI_CSV); bars = load_all_bars(uni)
    print(f'Loaded {len(bars)} stocks; period {START_DATE} -> {END_DATE}', flush=True)
    _annotate_adx_cross(bars)

    summary_rows = []
    print(f'\n{"Variant":>22} {"Dir":>5} {"Trades":>7} {"WR%":>6} {"PF":>6} {"AvgD":>6} '
          f'{"CAGR%":>7} {"Sharpe":>7} {"MaxDD%":>7} {"Calmar":>7}', flush=True)
    print('-' * 95, flush=True)

    best_pf, best_var, best_trades, best_dir = -1.0, None, None, None
    for direction, vlist, sfn in [('LONG', VARIANTS_LONG, signal_fn_long),
                                  ('SHORT', VARIANTS_SHORT, signal_fn_short)]:
        for cfg in vlist:
            heartbeat(HB, f'running {direction} {cfg["name"]}')
            t0 = time.time()
            trades, eq = run_engine(bars, cfg, sfn, direction)
            m = compute_metrics(trades, eq)
            summary_rows.append({'variant': cfg['name'], 'direction': direction, 'metrics': m})
            print(f'{cfg["name"]:>22} {direction:>5} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} '
                  f'{m["profit_factor"]:>6.2f} {m["avg_days_held"]:>6.1f} '
                  f'{m["cagr_pct"]:>+7.2f} {m["sharpe"]:>7.2f} '
                  f'{m["max_dd_pct"]:>7.2f} {m["calmar"]:>7.2f}  ({time.time()-t0:.1f}s)', flush=True)
            eq.to_csv(RES / f'equity_{cfg["name"]}_{direction}.csv', header=['equity'])
            if m['profit_factor'] > best_pf and m['trades'] >= 5:
                best_pf, best_var, best_trades, best_dir = m['profit_factor'], cfg['name'], trades, direction

    write_summary(summary_rows, RES / 'summary.csv')
    if best_trades is not None:
        write_trades(best_trades, RES / f'trades_{best_var}_{best_dir}.csv')
    print(f'\nBest by PF: {best_var} ({best_dir}) PF={best_pf:.2f}')
    print(f'Total runtime: {time.time()-t_start:.1f}s', flush=True)
    heartbeat(HB, f'DONE | best={best_var}/{best_dir} PF={best_pf:.2f}')


if __name__ == '__main__':
    sys.exit(main())
