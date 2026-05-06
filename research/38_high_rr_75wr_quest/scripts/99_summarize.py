"""Summarise the per-stock screen — show top candidates per pattern/direction.

Reads results/01_<pattern>_perstock.csv files and prints/writes a concise
ranking of which (stock, pattern, direction) combinations have the best
WR, profit factor, and trade count under the tight-SL/wide-TP exit.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(_HERE, '..', 'results'))
sys.path.insert(0, _HERE)
from pattern_lib import PATTERN_FUNCS  # type: ignore


def main() -> int:
    rows = []
    for p in PATTERN_FUNCS:
        path = os.path.join(RESULTS_DIR, f'01_{p}_perstock.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['pattern'] = p
        rows.append(df)
    if not rows:
        print('No screen data yet')
        return 1
    df = pd.concat(rows, ignore_index=True)

    print(f'Total rows: {len(df)}')
    print(f'Patterns: {sorted(df["pattern"].unique())}')

    # filter: n_trades >= 20
    sub = df[df['n_trades'] >= 20].copy()
    print(f'\nWith n>=20: {len(sub)}')
    print(f'  WR>=60%: {(sub["win_rate"] >= 0.60).sum()}')
    print(f'  WR>=65%: {(sub["win_rate"] >= 0.65).sum()}')
    print(f'  WR>=70%: {(sub["win_rate"] >= 0.70).sum()}')
    print(f'  WR>=75%: {(sub["win_rate"] >= 0.75).sum()}')

    # break out per (pattern, direction)
    for p in sorted(df['pattern'].unique()):
        for d in ('long', 'short'):
            s = sub[(sub['pattern'] == p) & (sub['direction'] == d)]
            if len(s) == 0:
                continue
            top = s.nlargest(15, 'win_rate')[['symbol', 'n_trades', 'win_rate',
                                              'profit_factor', 'max_dd_pct',
                                              'total_return_pct']]
            print(f'\n=== {p} / {d} (n={len(s)}, top 15 by WR) ===')
            print(top.to_string(index=False))

    # also write combined summary CSV
    out = os.path.join(RESULTS_DIR, '99_combined_screen_summary.csv')
    sub.sort_values(['pattern', 'direction', 'win_rate'], ascending=[True, True, False]).to_csv(out, index=False)
    print(f'\nWrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
