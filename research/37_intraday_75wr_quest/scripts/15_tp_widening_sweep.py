"""Stage 15 — TP-widening sweep at realistic cost.

Phase 1c showed the system fails at 0.10%/side cost with TP 0.5%/SL 1.5%.
This sweep tests several TP/SL pairs to find the smallest TP that:
  - Survives 0.10%/side cost (PF > 1.5, return positive)
  - Holds WR >= 70% (allow 5% drift from 78% baseline)
  - MaxDD <= 10%

Each combo runs the full Stage-13 portfolio backtest with Rs.3K fixed
risk cap, max-5 concurrent, 0.10%/side cost.

Output: results/15_tp_sweep_summary.csv
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(ROOT, '..', 'results'))
SCRIPT_13 = os.path.join(ROOT, '13_combined_portfolio.py')


# (tp_pct, sl_pct, label, hold_bars)
COMBOS = [
    # baseline (already known to fail at 0.10%/side)
    (0.5, 1.5, 'tp05_sl15', 60),
    # widening TP, keeping SL
    (0.7, 1.5, 'tp07_sl15', 60),
    (0.8, 1.5, 'tp08_sl15', 60),
    (1.0, 1.5, 'tp10_sl15', 60),
    # asymmetric tighter TP/SL ratios
    (1.0, 1.0, 'tp10_sl10', 60),
    (1.5, 1.0, 'tp15_sl10', 60),
    # favorable RR
    (1.5, 1.5, 'tp15_sl15', 60),
    (2.0, 1.5, 'tp20_sl15', 60),
    # widening hold for slower targets
    (1.5, 2.0, 'tp15_sl20', 60),
]


def run_one(tp: float, sl: float, label: str, hold: int, cost: float = 0.10) -> dict:
    suffix = f'sweep_{label}_c{int(cost*100):03d}'
    cmd = [
        sys.executable, SCRIPT_13,
        '--risk-per-trade-rs', '3000',
        '--max-concurrent', '5',
        '--cost-per-side-pct', f'{cost}',
        '--tp-pct', f'{tp}',
        '--sl-pct', f'{sl}',
        '--hold-bars', f'{hold}',
        '--label', suffix,
    ]
    print(f'\n>>> RUN tp={tp} sl={sl} hold={hold} cost={cost} ...', flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0

    summary_path = os.path.join(RESULTS_DIR, f'13_portfolio_summary_{suffix}.txt')
    if not os.path.exists(summary_path):
        print(f'    ERROR no summary file at {summary_path}')
        if proc.returncode != 0:
            print(f'    stderr: {proc.stderr[:500]}')
        return None

    text = open(summary_path, encoding='utf-8').read()

    def grab(key, parser=float, default=None):
        for line in text.splitlines():
            if line.strip().startswith(key):
                try:
                    val = line.split(':', 1)[1].strip()
                    val = val.replace('Rs.', '').replace(',', '').replace('%', '')
                    return parser(val)
                except Exception:
                    return default
        return default

    row = dict(
        tp_pct=tp, sl_pct=sl, hold_bars=hold, cost_per_side_pct=cost, label=label,
        elapsed_s=int(dt),
        final_nav=grab('Final NAV'),
        total_return_pct=grab('Total return'),
        cagr_pct=grab('CAGR'),
        sharpe=grab('Sharpe (daily)'),
        max_dd_pct=grab('Max drawdown'),
        calmar=grab('Calmar'),
        n_trades=grab('Total trades', int, 0),
        win_rate_pct=grab('Win rate'),
        profit_factor=grab('Profit factor'),
        trading_days=grab('Trading days', int, 0),
    )
    print(f'    -> ret={row["total_return_pct"]:+.2f}% WR={row["win_rate_pct"]:.1f}% '
          f'PF={row["profit_factor"]:.2f} DD={row["max_dd_pct"]:.2f}% n={row["n_trades"]} ({dt:.0f}s)',
          flush=True)
    return row


def main() -> int:
    out_csv = os.path.join(RESULTS_DIR, '15_tp_sweep_summary.csv')
    fields = ['tp_pct', 'sl_pct', 'hold_bars', 'cost_per_side_pct', 'label',
              'elapsed_s', 'final_nav', 'total_return_pct', 'cagr_pct',
              'sharpe', 'max_dd_pct', 'calmar',
              'n_trades', 'win_rate_pct', 'profit_factor', 'trading_days']

    rows = []
    for tp, sl, label, hold in COMBOS:
        r = run_one(tp, sl, label, hold, cost=0.10)
        if r is None:
            continue
        rows.append(r)

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f'\n=== TP-widening sweep complete — {len(rows)} configs ===')
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f'\nWrote {out_csv}')

    # winners: positive return, WR >= 70, PF >= 1.5, DD <= 10
    if not df.empty:
        winners = df[
            (df['total_return_pct'] > 0)
            & (df['win_rate_pct'] >= 70)
            & (df['profit_factor'] >= 1.5)
            & (df['max_dd_pct'] <= 10)
        ].copy()
        if len(winners) == 0:
            print('\nNO COMBO clears all four gates at 0.10%/side cost.')
            print('Best by total_return:')
            print(df.sort_values('total_return_pct', ascending=False).head(3).to_string(index=False))
        else:
            winners = winners.sort_values('total_return_pct', ascending=False)
            print(f'\nWINNERS clearing all gates: {len(winners)}')
            print(winners.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
