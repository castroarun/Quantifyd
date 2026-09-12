# -*- coding: utf-8 -*-
"""research/163 — rebuild the two curve files the Momentum Portfolio report reads, with
True North and Open Alpha · Base Age re-run at 5% idle cash.

IDENTICAL columns, IDENTICAL index, IDENTICAL rebasing. Only two columns change.

Both source files are built the same way upstream (research/159 `full_period.py` and
`compare_all.py`): each book's own NAV series is aligned onto a common index, forward
filled, the leading NaN rows dropped, and then the whole frame divided by its first row.
So replacing a column means exactly

    new_column(t) = new_nav(t) / new_nav(t0)          t0 = the frame's first date

with new_nav reindexed onto the frame's index and forward filled — which is what the
upstream `ffill()` does. Nothing else is recomputed, so every other column is byte-identical
to the file it came from, by construction.

Writes:
    results/full_period_after_tax_cash05.csv     (20.4-year headline window)
    results/all_systems_after_tax_cash05.csv     (2016+ roster window)
    results/curve_swap_report.json               what moved, and by how much
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
R159 = ROOT / 'research/159_oa_honest_reoptimization/results'
OUT = ROOT / 'research/163_mpf_cash_yield_harmonisation/results'

TN_NEW = OUT / 'tn_nav_INC_cash_n8_d15_tax1_cash05.csv'
BA_NEW = OUT / 'ba_nav_winner_cash05.csv'


def load(path):
    return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].dropna()


def stats(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    c = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    d = (s / s.cummax() - 1).min()
    return dict(cagr=round(c * 100, 2), maxdd=round(d * 100, 2),
                calmar=round(c / abs(d), 2), growth=round(float(s.iloc[-1] / s.iloc[0]), 2))


def swap(src, dst, mapping, label, report):
    old = pd.read_csv(src, index_col=0, parse_dates=True)
    new = old.copy()
    for col, series in mapping.items():
        r = series.reindex(old.index.union(series.index)).ffill().reindex(old.index)
        if r.isna().any():
            raise SystemExit('%s: %s has no value at or before %s'
                             % (label, col, old.index[0].date()))
        new[col] = (r / r.iloc[0]).values
    new.to_csv(dst)
    report[label] = {'file': str(dst), 'rows': int(len(new)),
                     'window': [str(new.index[0].date()), str(new.index[-1].date())],
                     'columns': list(new.columns), 'changed': list(mapping), 'rows_moved': {}}
    print('\n%s   %s -> %s   (%d rows)'
          % (label, new.index[0].date(), new.index[-1].date(), len(new)))
    print('  %-28s %19s %19s' % ('column', 'before (old yield)', 'after (5% yield)'))
    for c in old.columns:
        a, b = stats(old[c].dropna()), stats(new[c].dropna())
        moved = (a != b)
        report[label]['rows_moved'][c] = {'before': a, 'after': b, 'moved': moved}
        flag = '  <- CHANGED' if moved else ''
        print('  %-28s %7.2f%% %6.2f%% %.2f %7.2f%% %6.2f%% %.2f%s'
              % (c, a['cagr'], a['maxdd'], a['calmar'],
                 b['cagr'], b['maxdd'], b['calmar'], flag))
        if moved and c not in mapping:
            raise SystemExit('!! %s moved but was NOT swapped — aborting' % c)
        if not moved and c in mapping:
            print('     (note: swapped column is numerically identical — check the inputs)')
    return new


def main():
    tn, ba = load(TN_NEW), load(BA_NEW)
    print('True North 5%%   : %s -> %s (%d rows)'
          % (tn.index[0].date(), tn.index[-1].date(), len(tn)))
    print('Base Age 5%%     : %s -> %s (%d rows)'
          % (ba.index[0].date(), ba.index[-1].date(), len(ba)))

    report = {}
    swap(R159 / 'full_period_after_tax.csv', OUT / 'full_period_after_tax_cash05.csv',
         {'True North': tn, 'Open Alpha - Base Age': ba}, 'FULL PERIOD (20.4y)', report)
    swap(R159 / 'all_systems_after_tax.csv', OUT / 'all_systems_after_tax_cash05.csv',
         {'TN incumbent': tn, 'OA v2': ba}, 'ROSTER (2016+)', report)

    json.dump(report, open(OUT / 'curve_swap_report.json', 'w'), indent=1, default=str)
    print('\nwrote the two harmonised curve files + curve_swap_report.json')


if __name__ == '__main__':
    main()
