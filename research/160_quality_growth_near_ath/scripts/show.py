# -*- coding: utf-8 -*-
"""research/160 STUDY leg - read a cells CSV and print it as a ranked table.

    show.py results/cells_g1.csv [--sort calmar] [--grep A_] [--top 40] [--yearly LABEL]

Columns are the pre-registered ranking metric (after-tax Calmar at 25 bps) plus the whole
tradeability gate, because a book is only interesting if you can actually sit in it.
`inv%` (avg_pct_invested) is printed next to every CAGR on purpose: a thinly invested
book's CAGR is the idle-cash yield wearing a strategy's name (ENGINE leg self-test 6).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

COLS = ['label', 'cagr_net_tax', 'cagr_net_tax_worstpath', 'maxdd', 'maxdd_worst',
        'calmar', 'sharpe', 'avg_pct_invested', 'trades_per_yr', 'win_rate',
        'avg_win_pct', 'avg_loss_pct', 'expectancy_net_pct', 'max_losing_streak',
        'turnover_x_nav_yr', 'capacity_ratio', 'n_paths']
HDR = ['label', 'CAGR', 'worst', 'DD', 'DDwst', 'Calm', 'Shrp', 'inv%', 'tr/yr',
       'win%', 'avgW', 'avgL', 'exp%', 'strk', 'turn', 'cap', 'n']


def load(path):
    df = pd.read_csv(path)
    for c in COLS[1:]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def table(df, cols=COLS, hdr=HDR):
    cols = [c for c in cols if c in df.columns]
    hdr = [h for c, h in zip(COLS, HDR) if c in cols]
    w = max(len(str(x)) for x in df['label']) if len(df) else 5
    w = max(w, 5)
    out = ['%-*s %s' % (w, hdr[0], ' '.join('%6s' % h for h in hdr[1:]))]
    for _, r in df.iterrows():
        cells = []
        for c in cols[1:]:
            v = r[c]
            cells.append('%6.2f' % v if pd.notna(v) else '%6s' % '-')
        out.append('%-*s %s' % (w, r['label'], ' '.join(cells)))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--sort', default='calmar')
    ap.add_argument('--grep', default=None)
    ap.add_argument('--top', type=int, default=0)
    ap.add_argument('--yearly', default=None, help='print the per-year cells of one label')
    a = ap.parse_args()

    df = load(a.csv)
    if a.yearly:
        row = df[df.label == a.yearly]
        if not len(row):
            sys.exit('no such label')
        y = json.loads(row.iloc[0]['yearly'])
        print('%s  (return %% / intra-year DD %%, medians across paths)' % a.yearly)
        for k in sorted(y):
            print('  %s  %8.2f   (%7.2f)' % (k, y[k][0], y[k][1]))
        return
    if a.grep:
        df = df[df.label.str.contains(a.grep)]
    df = df.sort_values(a.sort, ascending=False)
    if a.top:
        df = df.head(a.top)
    print(table(df))
    print('\n%d cells' % len(df))


if __name__ == '__main__':
    sys.exit(main())
