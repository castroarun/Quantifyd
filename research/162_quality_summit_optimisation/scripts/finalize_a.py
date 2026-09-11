# -*- coding: utf-8 -*-
"""research/162 Part A G3 — the finalists, paired against the incumbent, in both windows.

Phase 1 and 2 chose on W1 only, as pre-registered. This script is the first and only place
W2 is read, and it produces everything the adoption decision needs in one process:

  * the finalists and the two controls on FULL / W1 / W2, all three arms (gross, net,
    after tax);
  * **paired** deltas against the Quality Summit incumbent on the SAME 12 rebalance-day
    offsets, in each window — median delta, how many of 12 offsets the candidate wins,
    and the same against the screenable-sub-universe control (does the SCREEN add value at
    this construction, which is the question r/160 answered "no" to at N=15 / k=0.90);
  * the cost ladder (25 / 40 / 60 bps), the idle-cash-off arm and the missing-data policy
    both ways;
  * outlier dependence: the trade-level compounding proxy with the ten best trades deleted;
  * daily equity curves for every finalist (one column per offset) for the YoY table, the
    tearsheet and the Part C blend.

Writes results/partA_final.{md,csv}, results/<name>_equity.csv and results/partA_paired.csv.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qg_engine2 as E                                                     # noqa: E402

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
R160 = ROOT / 'research/160_quality_growth_near_ath/results'
RES = Path(__file__).resolve().parents[1] / 'results'
AUX = str(RES / 'aux_162.npz')
B7 = str(R160 / 'masks_study' / 'b7_g10_qual_mc.npz')
HASDATA = str(RES / 'masks162' / 'has_data.npz')

FULL = ('2018-08-01', '2026-09-10')
W1 = ('2018-08-01', '2022-06-30')
W2 = ('2022-07-01', '2026-09-10')

# name -> the spec, one change at a time away from the incumbent
SPECS = {
    # the incumbent: research/160 Family B b7, exactly as published
    'QS_base':      dict(mask=B7, k=0.90, slots=15, weights='equal', exits='none'),
    # the candidate: same screen, wider near-ATH band, ten slots, inverse-vol sizing
    'QS_v2':        dict(mask=B7, k=0.85, slots=10, weights='invvol', exits='none'),
    # the candidate without the sizing change (is the sizing doing the work?)
    'QS_v2_eq':     dict(mask=B7, k=0.85, slots=10, weights='equal', exits='none'),
    # the plateau neighbour
    'QS_v2_nbr':    dict(mask=B7, k=0.825, slots=12, weights='equal', exits='none'),
    # controls at the CANDIDATE's construction
    'CTRL_hasdata': dict(mask=HASDATA, k=0.85, slots=10, weights='invvol', exits='none'),
    'CTRL_none':    dict(mask='', k=0.85, slots=10, weights='invvol', exits='none'),
    # control at the INCUMBENT's construction (r/160's own comparison)
    'CTRL_hd_base': dict(mask=HASDATA, k=0.90, slots=15, weights='equal', exits='none'),
}
FINALISTS = ['QS_v2', 'QS_v2_eq', 'QS_v2_nbr']


def cell(label, spec, start, end, **kw):
    c = dict(label=label, start=start, end=end, aux=AUX, offsets=12, arms='all',
             mask_missing='fail')
    c.update(spec)
    c.update(kw)
    return E.Cell(**c)


def outlier_proxy(trades):
    if not trades:
        return None
    r = pd.Series([t['ret_net_pct'] for t in trades]) / 100.0
    full = float((1 + r).prod())
    ex10 = float((1 + r.drop(r.nlargest(10).index)).prod())
    cap50 = float((1 + r.clip(upper=0.50)).prod())
    return dict(n=len(r), mean_pct=round(float(r.mean() * 100), 2), full=full,
                ex10=ex10, cap50=cap50,
                ratio=round(full / ex10, 1) if ex10 else float('nan'))


def main():
    t0 = time.time()
    panel = E.Panel.load(str(R160 / 'panel_2000.npz'))
    der = E.Derived(panel)
    print('panel %d dates, %d symbols' % (len(panel.dates), len(panel.syms)), flush=True)

    runs, rows = {}, []
    for nm, spec in SPECS.items():
        for wtag, (s, e) in (('full', FULL), ('W1', W1), ('W2', W2)):
            key = '%s|%s' % (nm, wtag)
            r = E.run_cell(panel, der, cell(key, spec, s, e), verbose=False)
            runs[key] = r
            row = dict(r['row'])
            row['name'], row['window'] = nm, wtag
            rows.append(row)
            print('%-22s CAGR gross %6.2f net %6.2f tax %6.2f  DD %7.2f (worst %7.2f)  '
                  'Calmar %5.2f  inv %5.1f  tr/yr %5.1f  streak %2d'
                  % (key, row['cagr_gross'], row['cagr_net'], row['cagr_net_tax'],
                     row['maxdd'], row['maxdd_worst'], row['calmar'],
                     row['avg_pct_invested'], row['trades_per_yr'],
                     row['max_losing_streak']), flush=True)

    # ---- robustness arms on the finalists and the incumbent -------------------------
    for nm in ['QS_base'] + FINALISTS:
        spec = SPECS[nm]
        for tag, kw in (('cost40', dict(cost_bps=40.0)), ('cost60', dict(cost_bps=60.0)),
                        ('cash0', dict(cash_yield=0.0)),
                        ('misspass', dict(mask_missing='pass'))):
            key = '%s|%s' % (nm, tag)
            r = E.run_cell(panel, der, cell(key, spec, *FULL, arms='tax', **kw),
                           verbose=False)
            row = dict(r['row'])
            row['name'], row['window'] = nm, tag
            rows.append(row)
            print('%-22s after tax %6.2f  DD %7.2f  Calmar %5.2f'
                  % (key, row['cagr_net_tax'], row['maxdd'], row['calmar']), flush=True)

    pd.DataFrame(rows).to_csv(RES / 'partA_final.csv', index=False)

    # ---- paired comparisons ----------------------------------------------------------
    pairs = []
    for wtag in ('full', 'W1', 'W2'):
        base = runs['QS_base|%s' % wtag]
        for nm in FINALISTS + ['CTRL_hasdata', 'CTRL_none']:
            a = runs['%s|%s' % (nm, wtag)]
            for metric in ('cagr_net_tax', 'calmar', 'maxdd'):
                d = E.paired_diff(a, base, metric)
                pairs.append(dict(a=nm, b='QS_base', window=wtag, metric=metric,
                                  n=d['n'], median_delta=round(d['median_delta'], 3),
                                  a_wins=d['a_wins']))
        # does the SCREEN add value at each construction?
        for cand, ctrl, tag in (('QS_v2', 'CTRL_hasdata', 'candidate construction'),
                                ('QS_base', 'CTRL_hd_base', 'incumbent construction')):
            a, b = runs['%s|%s' % (cand, wtag)], runs['%s|%s' % (ctrl, wtag)]
            for metric in ('cagr_net_tax', 'calmar', 'maxdd'):
                d = E.paired_diff(a, b, metric)
                pairs.append(dict(a=cand, b=ctrl, window=wtag, metric=metric, n=d['n'],
                                  median_delta=round(d['median_delta'], 3),
                                  a_wins=d['a_wins'], note=tag))
    pp = pd.DataFrame(pairs)
    pp.to_csv(RES / 'partA_paired.csv', index=False)
    print('\n=== paired vs QS_base (12 offsets, same offset on both sides) ===', flush=True)
    print(pp.to_string(index=False), flush=True)

    # ---- equity dumps + outliers -----------------------------------------------------
    outl = {}
    for nm in ['QS_base'] + FINALISTS + ['CTRL_hasdata']:
        r = runs['%s|full' % nm]
        pd.DataFrame(r['curves']).to_csv(RES / ('%s_equity.csv' % nm))
        outl[nm] = outlier_proxy(r['trades'])
        pd.DataFrame(r['trades']).to_csv(RES / ('%s_trades.csv' % nm), index=False)
    json.dump(outl, open(RES / 'partA_outliers.json', 'w'), indent=1)
    print('\n=== outlier dependence (one path, net of costs) ===', flush=True)
    for nm, o in outl.items():
        if o:
            print('%-14s trades %4d  mean %+.2f%%  full %.3g  ex-top-10 %.3g  '
                  'ratio %s  capped+50%% %.3g'
                  % (nm, o['n'], o['mean_pct'], o['full'], o['ex10'], o['ratio'],
                     o['cap50']), flush=True)
    print('\nPART A G3 DONE in %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
