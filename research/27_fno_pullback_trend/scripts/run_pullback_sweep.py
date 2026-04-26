"""Signal 6 — Pullback in trend.

LONG: stock in uptrend (close > 200-SMA AND 50-SMA > 200-SMA)
      AND today's close < 20-EMA
      AND today's RSI(14) < threshold (default 40).
SHORT: mirror — downtrend + close > 20-EMA + RSI > 60 (or 70 for stricter).

Variants (5):
  pb_rsi40                       default
  pb_rsi30                       deeper pullback
  pb_rsi40_macd                  + MACD histogram positive (long) / negative (short)
  pb_rsi40_volume_climax         + today's volume >= 1.5x avg (capitulation)
  pb_rsi40_confluence            RSI<40 + MACD>0 + vol>=1.5x

Both directions tested. Stops/targets identical to research/21.
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


def make_cfg(name, rsi_thr=40, use_macd=False, use_vol=False, vol_mult=1.5,
             target_pct=0.25, hard_stop_pct=0.08, cost_pct=0.0020):
    return dict(name=name, rsi_thr=rsi_thr, use_macd=use_macd,
                use_vol=use_vol, vol_mult=vol_mult,
                target_pct=target_pct, hard_stop_pct=hard_stop_pct, cost_pct=cost_pct)


VARIANTS_LONG = [
    make_cfg('pb_rsi40',                rsi_thr=40),
    make_cfg('pb_rsi30',                rsi_thr=30),
    make_cfg('pb_rsi40_macd',           rsi_thr=40, use_macd=True),
    make_cfg('pb_rsi40_volume_climax',  rsi_thr=40, use_vol=True, vol_mult=1.5),
    make_cfg('pb_rsi40_confluence',     rsi_thr=40, use_macd=True,
                                          use_vol=True, vol_mult=1.5),
]
VARIANTS_SHORT = [{**v, 'name': v['name']} for v in VARIANTS_LONG]


def signal_fn_long(row: pd.Series, cfg: dict) -> bool:
    sma200 = row.get('sma_200'); sma50 = row.get('sma_50')
    if pd.isna(sma200) or pd.isna(sma50):
        return False
    if not (row['close'] > sma200 and sma50 > sma200):
        return False
    ema20 = row.get('ema_20')
    if pd.isna(ema20) or row['close'] >= ema20:
        return False
    rsi = row.get('rsi_14')
    if pd.isna(rsi) or rsi >= cfg['rsi_thr']:
        return False
    if cfg.get('use_macd'):
        mh = row.get('macd_hist')
        if pd.isna(mh) or mh <= 0:
            return False
    if cfg.get('use_vol'):
        vavg = row.get('vol_avg')
        if pd.isna(vavg) or vavg <= 0:
            return False
        if not (row['volume'] >= cfg['vol_mult'] * vavg):
            return False
    return True


def signal_fn_short(row: pd.Series, cfg: dict) -> bool:
    sma200 = row.get('sma_200'); sma50 = row.get('sma_50')
    if pd.isna(sma200) or pd.isna(sma50):
        return False
    if not (row['close'] < sma200 and sma50 < sma200):
        return False
    ema20 = row.get('ema_20')
    if pd.isna(ema20) or row['close'] <= ema20:
        return False
    rsi = row.get('rsi_14')
    rsi_short_thr = 100 - cfg['rsi_thr']   # mirror: 40 -> 60, 30 -> 70
    if pd.isna(rsi) or rsi <= rsi_short_thr:
        return False
    if cfg.get('use_macd'):
        mh = row.get('macd_hist')
        if pd.isna(mh) or mh >= 0:
            return False
    if cfg.get('use_vol'):
        vavg = row.get('vol_avg')
        if pd.isna(vavg) or vavg <= 0:
            return False
        if not (row['volume'] >= cfg['vol_mult'] * vavg):
            return False
    return True


def main():
    t_start = time.time()
    heartbeat(HB, 'start')
    uni = load_universe(UNI_CSV); bars = load_all_bars(uni)
    print(f'Loaded {len(bars)} stocks; period {START_DATE} -> {END_DATE}', flush=True)

    summary_rows = []
    print(f'\n{"Variant":>26} {"Dir":>5} {"Trades":>7} {"WR%":>6} {"PF":>6} {"AvgD":>6} '
          f'{"CAGR%":>7} {"Sharpe":>7} {"MaxDD%":>7} {"Calmar":>7}', flush=True)
    print('-' * 100, flush=True)

    best_pf, best_var, best_trades, best_dir = -1.0, None, None, None
    for direction, vlist, sfn in [('LONG', VARIANTS_LONG, signal_fn_long),
                                  ('SHORT', VARIANTS_SHORT, signal_fn_short)]:
        for cfg in vlist:
            heartbeat(HB, f'running {direction} {cfg["name"]}')
            t0 = time.time()
            trades, eq = run_engine(bars, cfg, sfn, direction)
            m = compute_metrics(trades, eq)
            summary_rows.append({'variant': cfg['name'], 'direction': direction, 'metrics': m})
            print(f'{cfg["name"]:>26} {direction:>5} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} '
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
