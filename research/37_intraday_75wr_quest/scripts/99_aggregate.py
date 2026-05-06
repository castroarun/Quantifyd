"""Aggregator — read all ranking CSVs, surface top systems by WR / AWS.

Run this any time to see the current best candidates. Updates STATUS.md
findings section if --update-status is passed.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))


def load_all() -> pd.DataFrame:
    paths = (
        glob.glob(os.path.join(RESULTS_DIR, '02_*_ranking.csv'))
        + glob.glob(os.path.join(RESULTS_DIR, '04_*_ranking.csv'))
    )
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        # tag the source file
        df['source'] = os.path.basename(p)
        if 'family' not in df.columns and 'side' in df.columns:
            df['family'] = 'confluence_' + df['side']
        if 'side' not in df.columns:
            df['side'] = 'long'
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-wr', type=int, default=20)
    ap.add_argument('--top-aws', type=int, default=20)
    ap.add_argument('--min-wr', type=float, default=0.65)
    ap.add_argument('--min-trades', type=int, default=30)
    args = ap.parse_args()

    df = load_all()
    if df.empty:
        print('No ranking files found.')
        return 0
    print(f'Loaded {len(df)} systems across {df["source"].nunique()} files')
    print()

    qualify = (df['win_rate'] >= args.min_wr) & (df['n_trades'] >= args.min_trades)
    pool = df[qualify].copy()
    print(f'{len(pool)} systems with WR >= {args.min_wr:.0%} and trades >= {args.min_trades}')
    print()

    if pool.empty:
        # show best WR period anyway
        best_wr = df.nlargest(20, 'win_rate')
        print('Top 20 by WR overall (regardless of gates):')
        print(best_wr[['family', 'win_rate', 'n_trades', 'profit_factor',
                        'max_dd_pct', 'tp_pct', 'sl_pct']].to_string(index=False))
        return 0

    cols = ['family', 'win_rate', 'n_trades', 'profit_factor', 'max_dd_pct',
            'sharpe', 'aws', 'tp_pct', 'sl_pct', 'max_hold_bars',
            'trades_per_stock_year', 'params']

    print('=== Top by WIN RATE ===')
    print(pool.nlargest(args.top_wr, 'win_rate')[cols].to_string(index=False))
    print()
    print('=== Top by AWS (adjusted score) ===')
    print(pool.nlargest(args.top_aws, 'aws')[cols].to_string(index=False))
    print()
    # systems clearing all hard gates
    gates = (
        (pool['win_rate'] >= 0.75)
        & (pool['profit_factor'] >= 2.0)
        & (pool['max_dd_pct'] <= 15.0)
        & (pool['sharpe'] >= 1.5)
    )
    final = pool[gates].sort_values('aws', ascending=False)
    print(f'=== Systems clearing ALL HARD GATES ({len(final)}) ===')
    if len(final):
        print(final[cols].head(20).to_string(index=False))
    else:
        print('(none yet)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
