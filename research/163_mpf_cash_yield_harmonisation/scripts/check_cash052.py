# -*- coding: utf-8 -*-
"""research/163 — the 5.2% consistency check.

THREE questions.

  1. NIFTYBEES must be BIT-IDENTICAL. It is a price series, it holds no cash; if it moved,
     something other than the idle-cash yield changed and nothing below can be trusted.

  2. Per book, the PAIRED yield effect must agree with the arithmetic the change implies:

         expected CAGR gain  ~  (1 - invested fraction) x 0.2 pp

     because the extra 20 bps a year is earned only on the share of the book in cash.
     "Paired" means seed-by-seed / offset-by-offset, the SAME path index at both yields —
     never the difference between two ensemble medians, which mixes different draws.

  3. The DRAWN-PATH move (what the page's tables and charts actually show) is reported
     beside it, because for the ensemble books the two are NOT the same number, and the
     reason is worth stating: a 20 bps change in the cash rate changes the cash balance,
     which changes INTEGER SHARE COUNTS, which changes whether a given buy is affordable,
     which re-draws every later selection in that path. That re-draw is worth up to +-2
     points on a single path — an order of magnitude more than the 0.02-0.16 pp the cash
     rate itself is worth. So the drawn path can move the "wrong" way while the ensemble
     is consistent. The check is (2); (3) is disclosure.

Run:
    venv/bin/python3 research/163_mpf_cash_yield_harmonisation/scripts/check_cash052.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
R159 = ROOT / 'research/159_oa_honest_reoptimization/results'
R160 = ROOT / 'research/160_quality_growth_near_ath/results'
R163 = ROOT / 'research/163_mpf_cash_yield_harmonisation/results'
C52 = R163 / 'cash052'

INVESTED = {'True North': 43.0, 'TN incumbent': 43.0,
            'Open Alpha - Base Age': 72.9, 'OA v2': 72.9,
            'IPO Base - First Base': 31.8, 'IPO (honest)': 31.8,
            'OA v3 (gated)': 79.0,
            'NIFTYBEES (index)': 100.0, 'NIFTYBEES': 100.0}
FROZEN = {'NIFTYBEES (index)', 'NIFTYBEES'}

PAIRS = [('FULL PERIOD (20.4y)  — the headline table',
          R163 / 'full_period_after_tax_cash05.csv',
          C52 / 'full_period_after_tax_cash052.csv'),
         ('ROSTER (2016+)  — feeds the 2018 section',
          R163 / 'all_systems_after_tax_cash05.csv',
          C52 / 'all_systems_after_tax_cash052.csv')]


def stats(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    c = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    d = (s / s.cummax() - 1).min()
    return c * 100, d * 100, c / abs(d)


def cagr_cols(df):
    return {c: stats(df[c].dropna())[0] for c in df.columns}


def paired_verdict(name, deltas, inv, n_label):
    """deltas: paired per-path CAGR differences in pp. Judged against the standard error of
    the median, because for these books the per-path spread is wide."""
    d = np.asarray(deltas, dtype=float)
    med = float(np.median(d))
    pred = (1 - inv / 100.0) * 0.2
    se = 1.2533 * float(np.std(d, ddof=1)) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    tol = max(2.0 * se, 0.05)
    ok = abs(med - pred) <= tol
    print('  %-26s inv %5.1f%%  predicted %+.3f pp   paired median %+.3f pp  '
          '[%+.2f .. %+.2f] over %s  SE %.3f  ->  %s'
          % (name, inv, pred, med, d.min(), d.max(), n_label, se,
             'CONSISTENT' if ok else 'CHECK'))
    return ok, med, pred


def main():
    fail = 0

    # ---------------- 1 + 3: the two curve files --------------------------------------
    for label, p50, p52 in PAIRS:
        a = pd.read_csv(p50, index_col=0, parse_dates=True)
        b = pd.read_csv(p52, index_col=0, parse_dates=True)
        print('\n=== %s ===' % label)
        if not a.index.equals(b.index) or list(a.columns) != list(b.columns):
            print('!! index or columns differ between the two files'); fail += 1; continue
        print('%s -> %s, %d rows, %d columns'
              % (a.index[0].date(), a.index[-1].date(), len(a), a.shape[1]))
        print('  %-26s %9s %9s %9s %10s %10s   %s'
              % ('column', 'CAGR 5.0', 'CAGR 5.2', 'd CAGR', 'DD 5.0', 'DD 5.2', 'state'))
        for c in a.columns:
            sa, sb = a[c].dropna(), b[c].dropna()
            ca, da, _ = stats(sa)
            cb, db, _ = stats(sb)
            dmax = float(np.abs(sa.values - sb.reindex(sa.index).values).max())
            if c in FROZEN:
                ok = (dmax == 0.0)
                if not ok:
                    fail += 1
                state = 'BIT-IDENTICAL' if ok else 'MOVED !! holds no cash'
            else:
                state = 'drawn path'
            print('  %-26s %8.2f%% %8.2f%% %+8.3f  %9.2f%% %9.2f%%   %s'
                  % (c, ca, cb, cb - ca, da, db, state))
        oa = sorted(a.columns, key=lambda c: -stats(a[c].dropna())[0])
        ob = sorted(b.columns, key=lambda c: -stats(b[c].dropna())[0])
        print('  CAGR ranking 5.0%%: %s' % ' > '.join(oa))
        print('  CAGR ranking 5.2%%: %s%s' % (' > '.join(ob),
                                              '' if oa == ob else '   <-- REORDERED'))

    # ---------------- 2: the paired consistency test, per book -------------------------
    print('\n=== PAIRED YIELD EFFECT vs (1 - invested) x 0.2 pp ===')

    # True North — one deterministic path, so the drawn move IS the paired move
    tn50 = pd.read_csv(R163 / 'tn_nav_INC_cash_n8_d15_tax1_cash05.csv',
                       index_col=0, parse_dates=True).iloc[:, 0]
    tn52 = pd.read_csv(C52 / 'tn_nav_INC_cash_n8_d15_tax1_cash052.csv',
                       index_col=0, parse_dates=True).iloc[:, 0]
    ok, _, _ = paired_verdict('True North', [stats(tn52)[0] - stats(tn50)[0]], 43.0,
                              '1 path')
    fail += (not ok)

    # Base Age — 30 seeds
    ss = pd.read_csv(C52 / 'ba_seed_stats_052.csv')
    piv = ss.pivot(index='seed', columns='idle_yield', values='cagr')
    ok, _, _ = paired_verdict('Open Alpha · Base Age', (piv[0.052] - piv[0.05]).values,
                              float(ss[ss.idle_yield == 0.052].invested_pct.median()),
                              '30 seeds')
    fail += (not ok)

    # IPO Base — 30 seeds
    i50 = pd.read_csv(C52 / 'ipo_seed_stats_050.csv')
    i52 = pd.read_csv(C52 / 'ipo_seed_stats_052.csv')
    ok, _, _ = paired_verdict('IPO Base', (i52['cagr'].values - i50['cagr'].values),
                              float(i50['invested_pct'].median()), '30 seeds')
    fail += (not ok)

    # Open Alpha · ATH + VIX — 30 seeds, measured on the aligned index
    av = json.load(open(C52 / 'athvix_summary_cash052.json'))
    ok, _, _ = paired_verdict('Open Alpha · ATH + VIX',
                              np.array(av['aligned_band_052']['cagr_all'])
                              - np.array(av['aligned_band_050']['cagr_all']),
                              av['invested_median_pct'], '30 seeds')
    fail += (not ok)

    # Quality Summit — 12 rebalance offsets, straight off the two equity files
    q50 = pd.read_csv(R160 / 'F_Bb7_equity.csv', index_col=0, parse_dates=True)
    q52 = pd.read_csv(C52 / 'F_Bb7_equity_cash052.csv', index_col=0, parse_dates=True)
    qs = json.load(open(C52 / 'qs_cash052_summary.json'))
    d = np.array([stats(q52[c].dropna())[0] - stats(q50[c].dropna())[0] for c in q50.columns])
    ok, _, _ = paired_verdict('Quality Summit', d, qs['invested_pct'], '12 offsets')
    fail += (not ok)

    print('\n%s' % ('CHECK FAILED — %d item(s)' % fail if fail else
                    'PASS — NIFTYBEES bit-identical in both files, and every cash-holding '
                    'book\'s PAIRED yield effect agrees with (1 - invested) x 0.2 pp within '
                    'its own path noise.'))
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
