"""Stage 10 — favorable-RR sweep on the 3-system signals from research/37.

The TP-widening sweep already mapped TP/SL where TP <= SL. This sweep
zeroes in on TP > SL ratios specifically (the user's hard constraint
of RR >= 1.1 on the SL:TP ratio with TP > SL). Goal: find the highest-WR
combo on the existing strong signal stack where RR is favorable.

If even the strongest signal stack we have can't hit WR >= 75% AND
RR >= 1.1, the structural-no-go finding is locked.
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
SCRIPT_13 = os.path.normpath(os.path.join(
    ROOT, '..', '..', '37_intraday_75wr_quest', 'scripts', '13_combined_portfolio.py'
))


# Favorable RR only (TP > SL). Spans tight to wide stops.
COMBOS = [
    # (tp, sl, label, hold)
    (0.5, 0.3, 'tp05_sl03', 60),  # RR 1.67
    (0.7, 0.4, 'tp07_sl04', 60),  # RR 1.75
    (0.7, 0.5, 'tp07_sl05', 60),  # RR 1.4
    (0.8, 0.5, 'tp08_sl05', 60),  # RR 1.6
    (1.0, 0.5, 'tp10_sl05', 60),  # RR 2.0
    (1.0, 0.7, 'tp10_sl07', 60),  # RR 1.43
    (1.0, 0.8, 'tp10_sl08', 60),  # RR 1.25
    (1.0, 0.9, 'tp10_sl09', 60),  # RR 1.11 (just above the 1.1 floor)
    (1.5, 1.0, 'tp15_sl10', 60),  # RR 1.5 (already tested but include for ranking)
    (1.5, 1.3, 'tp15_sl13', 60),  # RR 1.15
    (2.0, 1.0, 'tp20_sl10', 60),  # RR 2.0
    (2.0, 1.3, 'tp20_sl13', 60),  # RR 1.54
]


def run_one(tp: float, sl: float, label: str, hold: int, cost: float = 0.10) -> dict:
    suffix = f'rr_{label}_c{int(cost*100):03d}'
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
    print(f'\n>>> RUN tp={tp} sl={sl} RR={tp/sl:.2f} cost={cost} ...', flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0

    summary_path = os.path.normpath(os.path.join(
        RESULTS_DIR, '..', '..', '37_intraday_75wr_quest', 'results',
        f'13_portfolio_summary_{suffix}.txt',
    ))
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
        tp_pct=tp, sl_pct=sl, rr=round(tp/sl, 2), hold_bars=hold,
        cost_per_side_pct=cost, label=label,
        total_return_pct=grab('Total return'),
        cagr_pct=grab('CAGR'),
        sharpe=grab('Sharpe (daily)'),
        max_dd_pct=grab('Max drawdown'),
        n_trades=grab('Total trades', int, 0),
        win_rate_pct=grab('Win rate'),
        profit_factor=grab('Profit factor'),
        elapsed_s=int(dt),
    )
    print(f'    -> WR={row["win_rate_pct"]:.1f}% RR={row["rr"]} ret={row["total_return_pct"]:+.2f}% '
          f'PF={row["profit_factor"]:.2f} DD={row["max_dd_pct"]:.2f}% n={row["n_trades"]} ({dt:.0f}s)',
          flush=True)
    return row


def main() -> int:
    out_csv = os.path.join(RESULTS_DIR, '10_favorable_rr_sweep.csv')
    fields = ['tp_pct', 'sl_pct', 'rr', 'hold_bars', 'cost_per_side_pct', 'label',
              'total_return_pct', 'cagr_pct', 'sharpe', 'max_dd_pct',
              'n_trades', 'win_rate_pct', 'profit_factor', 'elapsed_s']

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

    df = pd.DataFrame(rows)
    print('\n=== Favorable-RR sweep complete ===')
    print(df.sort_values('win_rate_pct', ascending=False).to_string(index=False))

    if not df.empty:
        target = df[(df['win_rate_pct'] >= 75) & (df['rr'] >= 1.1)]
        print(f'\nCells clearing WR>=75% AND RR>=1.1 with TP>SL: {len(target)}')
        if len(target):
            print(target.to_string(index=False))
        else:
            best_wr_with_rr = df[df['rr'] >= 1.1].sort_values('win_rate_pct', ascending=False).head(3)
            print('Top 3 favorable-RR by WR (none hit 75%):')
            print(best_wr_with_rr.to_string(index=False))
    print(f'\nWrote {out_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
