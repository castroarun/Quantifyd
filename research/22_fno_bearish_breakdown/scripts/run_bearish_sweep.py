"""Signal 1 — Bearish 252-day breakdown sweep (mirror of research/21, SHORT side).

Variants (5):
  baseline         vol >= 2.5x, no extras
  vol_3x           vol >= 3.0x
  vol_3x_atr_floor vol >= 3.0x + ATR%/close >= 1%
  vol_3x_adx       vol >= 3.0x + ADX(14) > 25
  vol_3x_confluence vol >= 3.0x + ATR floor + ADX > 25

Entry filter: close < 252-day low AND close < 200-SMA AND vol >= N x avg.
Direction:    SHORT.
"""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _shared_engine import (   # type: ignore
    load_universe, load_all_bars, run_engine, compute_metrics,
    write_trades, write_summary, heartbeat,
    START_DATE, END_DATE,
)

RES = HERE.parent / 'results'
LOGS = HERE.parent / 'logs'
RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
HB = LOGS / 'sweep_heartbeat.txt'
UNI_CSV = RES / 'universe.csv'


def make_cfg(name, vol_mult=2.5, use_atr_floor=False, use_adx=False,
             target_pct=0.25, hard_stop_pct=0.08, cost_pct=0.0020):
    return dict(name=name, vol_mult=vol_mult, use_atr_floor=use_atr_floor,
                use_adx=use_adx, target_pct=target_pct,
                hard_stop_pct=hard_stop_pct, cost_pct=cost_pct)


VARIANTS = [
    make_cfg('baseline',           vol_mult=2.5),
    make_cfg('vol_3x',             vol_mult=3.0),
    make_cfg('vol_3x_atr_floor',   vol_mult=3.0, use_atr_floor=True),
    make_cfg('vol_3x_adx',         vol_mult=3.0, use_adx=True),
    make_cfg('vol_3x_confluence',  vol_mult=3.0, use_atr_floor=True, use_adx=True),
]


def signal_fn(row: pd.Series, cfg: dict) -> bool:
    low_n = row.get('low_252d')
    if pd.isna(low_n):
        return False
    if not (row['close'] < low_n):
        return False
    # Volume filter
    vavg = row.get('vol_avg')
    if pd.isna(vavg) or vavg <= 0:
        return False
    if not (row['volume'] >= cfg['vol_mult'] * vavg):
        return False
    # Regime: below 200 SMA (bearish)
    sma200 = row.get('sma_200')
    if pd.isna(sma200):
        return False
    if not (row['close'] < sma200):
        return False
    # ATR floor (volatility minimum)
    if cfg.get('use_atr_floor'):
        atr_p = row.get('atr_pct')
        if pd.isna(atr_p) or atr_p < 0.01:
            return False
    # ADX confirmation
    if cfg.get('use_adx'):
        adx = row.get('adx_14')
        if pd.isna(adx) or adx <= 25:
            return False
    return True


def main():
    t_start = time.time()
    heartbeat(HB, 'start')
    print('Loading F&O universe + bars...', flush=True)
    uni = load_universe(UNI_CSV)
    bars = load_all_bars(uni)
    print(f'Loaded {len(bars)} stocks; period {START_DATE} -> {END_DATE}', flush=True)

    summary_rows = []
    print()
    print(f'{"Variant":>22} {"Trades":>7} {"WR%":>6} {"PF":>6} {"AvgD":>6} '
          f'{"CAGR%":>7} {"Sharpe":>7} {"MaxDD%":>7} {"Calmar":>7}', flush=True)
    print('-' * 90, flush=True)

    all_trades_path = RES / 'fno_bearish_trades.csv'
    with all_trades_path.open('w', newline='') as f:
        csv.writer(f).writerow(['variant','symbol','direction','entry_date','exit_date',
                                'entry_price','exit_price','qty','days_held','exit_reason',
                                'gross_pnl','net_pnl'])

    best_pf, best_var, best_trades = -1.0, None, None
    for cfg in VARIANTS:
        heartbeat(HB, f'running variant {cfg["name"]}')
        t0 = time.time()
        trades, eq = run_engine(bars, cfg, signal_fn, direction='SHORT')
        m = compute_metrics(trades, eq)
        summary_rows.append({'variant': cfg['name'], 'direction': 'SHORT', 'metrics': m})
        print(f'{cfg["name"]:>22} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} '
              f'{m["profit_factor"]:>6.2f} {m["avg_days_held"]:>6.1f} '
              f'{m["cagr_pct"]:>+7.2f} {m["sharpe"]:>7.2f} '
              f'{m["max_dd_pct"]:>7.2f} {m["calmar"]:>7.2f}  ({time.time()-t0:.1f}s)', flush=True)
        # Append per-variant trades to aggregate log
        with all_trades_path.open('a', newline='') as f:
            w = csv.writer(f)
            for t in trades:
                w.writerow([t.variant, t.symbol, t.direction, t.entry_date, t.exit_date,
                            f'{t.entry_price:.2f}', f'{t.exit_price:.2f}',
                            t.qty, t.days_held, t.exit_reason,
                            f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])
        eq.to_csv(RES / f'equity_{cfg["name"]}.csv', header=['equity'])
        if m['profit_factor'] > best_pf and m['trades'] >= 5:
            best_pf, best_var, best_trades = m['profit_factor'], cfg['name'], trades

    write_summary(summary_rows, RES / 'summary.csv')
    if best_trades is not None:
        write_trades(best_trades, RES / f'trades_{best_var}.csv')
    print(f'\nBest variant by PF: {best_var} (PF={best_pf:.2f})', flush=True)
    print(f'Total runtime: {time.time()-t_start:.1f}s', flush=True)
    heartbeat(HB, f'DONE | best={best_var} PF={best_pf:.2f}')


if __name__ == '__main__':
    sys.exit(main())
