"""Walk-forward validation for bearish breakdown — top 3 variants."""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _shared_engine import (   # type: ignore
    load_universe, load_all_bars, run_engine, compute_metrics, heartbeat,
)
from run_bearish_sweep import VARIANTS, signal_fn   # type: ignore

RES  = HERE.parent / 'results'
LOGS = HERE.parent / 'logs'
HB   = LOGS / 'walk_forward_heartbeat.txt'
UNI_CSV = RES / 'universe.csv'

TRAIN_START, TRAIN_END = '2018-01-01', '2022-12-31'
TEST_START,  TEST_END  = '2023-01-01', '2025-12-31'


def main():
    # Pick top 3 variants by full-period PF from summary.csv if present;
    # otherwise default to a sensible top-3 list.
    top_names = []
    summary = RES / 'summary.csv'
    if summary.exists():
        rows = list(csv.DictReader(summary.open()))
        rows.sort(key=lambda r: float(r.get('profit_factor', 0) or 0), reverse=True)
        top_names = [r['variant'] for r in rows[:3]]
    if not top_names:
        top_names = ['vol_3x', 'baseline', 'vol_3x_confluence']
    cfgs = [v for v in VARIANTS if v['name'] in top_names]

    heartbeat(HB, 'start')
    uni = load_universe(UNI_CSV)
    bars = load_all_bars(uni)
    keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
            'avg_days_held','profit_factor','net_pnl','cagr_pct','sharpe',
            'max_dd_pct','calmar','final_equity']

    print(f'{"Variant":>22} {"Phase":>5} {"Trades":>7} {"WR%":>6} {"PF":>6} '
          f'{"CAGR%":>7} {"Sharpe":>7} {"MaxDD%":>7} {"Calmar":>7}', flush=True)
    print('-' * 95, flush=True)

    out_rows = []
    for cfg in cfgs:
        heartbeat(HB, f'running {cfg["name"]} IS')
        is_t, is_eq = run_engine(bars, cfg, signal_fn, 'SHORT', TRAIN_START, TRAIN_END)
        is_m = compute_metrics(is_t, is_eq)
        out_rows.append({'variant': cfg['name'], 'phase': 'IS', **is_m})
        print(f'{cfg["name"]:>22} {"IS":>5} {is_m["trades"]:>7} {is_m["win_rate_pct"]:>6.1f} '
              f'{is_m["profit_factor"]:>6.2f} {is_m["cagr_pct"]:>+6.2f} {is_m["sharpe"]:>7.2f} '
              f'{is_m["max_dd_pct"]:>7.2f} {is_m["calmar"]:>7.2f}', flush=True)

        heartbeat(HB, f'running {cfg["name"]} OOS')
        oos_t, oos_eq = run_engine(bars, cfg, signal_fn, 'SHORT', TEST_START, TEST_END)
        oos_m = compute_metrics(oos_t, oos_eq)
        out_rows.append({'variant': cfg['name'], 'phase': 'OOS', **oos_m})
        print(f'{cfg["name"]:>22} {"OOS":>5} {oos_m["trades"]:>7} {oos_m["win_rate_pct"]:>6.1f} '
              f'{oos_m["profit_factor"]:>6.2f} {oos_m["cagr_pct"]:>+6.2f} {oos_m["sharpe"]:>7.2f} '
              f'{oos_m["max_dd_pct"]:>7.2f} {oos_m["calmar"]:>7.2f}', flush=True)
        pf_ok  = oos_m['profit_factor'] >= 1.20
        sh_ok  = oos_m['sharpe'] >= 0.8
        dd_ok  = oos_m['max_dd_pct'] <= 30
        verdict = 'PASS' if (pf_ok and sh_ok and dd_ok) else 'FAIL'
        print(f'  -> verdict: {verdict}  PF={oos_m["profit_factor"]:.2f} '
              f'Sharpe={oos_m["sharpe"]:.2f} MaxDD={oos_m["max_dd_pct"]:.1f}%')

    out = RES / 'walk_forward.csv'
    with out.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['variant','phase'] + keys)
        for r in out_rows:
            w.writerow([r['variant'], r['phase']] + [r.get(k, '') for k in keys])
    print(f'Saved {out}', flush=True)
    heartbeat(HB, 'DONE')


if __name__ == '__main__':
    sys.exit(main())
