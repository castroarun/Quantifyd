# -*- coding: utf-8 -*-
"""research/163 — rebuild the two curve files the Momentum Portfolio report reads, with
EVERY book re-run at 5.2% idle cash (the arbitrage-fund rate after 20% short-term tax).

Same construction as this folder's `build_inputs.py`, which produced the 5.0% pair: start
from research/159's originals, replace one column per re-run book, leave everything else
byte-identical by construction. The difference is that last night only True North and Base
Age were replaced; tonight IPO Base and Open Alpha · ATH + VIX are replaced too, and only
NIFTYBEES is left alone — it holds no cash.

    new_column(t) = new_nav(t) / new_nav(t0)      t0 = the frame's first date

with new_nav reindexed onto the frame's index and forward filled, which is exactly what the
upstream `ffill()` does.

Writes into results/cash052/:
    full_period_after_tax_cash052.csv    (20.4-year headline window)
    all_systems_after_tax_cash052.csv    (2016+ roster window, feeds the 2018 section)
    curve_swap_report_052.json           what moved, and by how much
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path('/home/arun/quantifyd')
R159 = ROOT / 'research/159_oa_honest_reoptimization/results'
OUT = ROOT / 'research/163_mpf_cash_yield_harmonisation/results/cash052'

TN_NEW = OUT / 'tn_nav_INC_cash_n8_d15_tax1_cash052.csv'
BA_NEW = OUT / 'ba_nav_winner_cash052.csv'
IPO_NEW = OUT / 'ipo_honest_curve_cash052.csv'
OAV_NEW = OUT / 'oa_gated_cash052.csv'


def load(path):
    return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].dropna()


def stats(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    c = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    d = (s / s.cummax() - 1).min()
    return dict(cagr=round(c * 100, 2), maxdd=round(d * 100, 2),
                calmar=round(c / abs(d), 2), growth=round(float(s.iloc[-1] / s.iloc[0]), 2))


def swap(src, dst, mapping, label, report, frozen):
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
    print('  %-28s %21s %21s' % ('column', 'before (5.0%)', 'after (5.2%)'))
    for c in old.columns:
        a, b = stats(old[c].dropna()), stats(new[c].dropna())
        moved = (a != b)
        report[label]['rows_moved'][c] = {'before': a, 'after': b, 'moved': moved}
        flag = '  <- CHANGED' if moved else ''
        print('  %-28s %7.2f%% %6.2f%% %.2f %9.2f%% %6.2f%% %.2f%s'
              % (c, a['cagr'], a['maxdd'], a['calmar'],
                 b['cagr'], b['maxdd'], b['calmar'], flag))
        if moved and c not in mapping:
            raise SystemExit('!! %s moved but was NOT swapped — aborting' % c)
        if c in frozen and moved:
            raise SystemExit('!! %s is supposed to hold no cash but moved — aborting' % c)
        if not moved and c in mapping:
            print('     (note: swapped column is numerically identical — check the inputs)')
    return new


def main():
    tn, ba, ipo, oav = load(TN_NEW), load(BA_NEW), load(IPO_NEW), load(OAV_NEW)
    for nm, s in (('True North', tn), ('Base Age', ba), ('IPO honest', ipo),
                  ('OA ATH+VIX', oav)):
        print('%-12s 5.2%%: %s -> %s (%d rows)'
              % (nm, s.index[0].date(), s.index[-1].date(), len(s)))

    report = {}
    # NOTE the two files use DIFFERENT column names for the same books; that is research/159's
    # own convention and is preserved so the generator's rename maps keep working.
    swap(R159 / 'full_period_after_tax.csv', OUT / 'full_period_after_tax_cash052.csv',
         {'True North': tn, 'Open Alpha - Base Age': ba, 'IPO Base - First Base': ipo},
         'FULL PERIOD (20.4y)', report, frozen={'NIFTYBEES (index)'})
    swap(R159 / 'all_systems_after_tax.csv', OUT / 'all_systems_after_tax_cash052.csv',
         {'TN incumbent': tn, 'OA v2': ba, 'IPO (honest)': ipo, 'OA v3 (gated)': oav},
         'ROSTER (2016+)', report, frozen={'NIFTYBEES'})

    json.dump(report, open(OUT / 'curve_swap_report_052.json', 'w'), indent=1, default=str)
    print('\nwrote the two 5.2%-idle-cash curve files + curve_swap_report_052.json')


if __name__ == '__main__':
    main()
