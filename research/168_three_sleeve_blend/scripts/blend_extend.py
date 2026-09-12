# -*- coding: utf-8 -*-
"""research/168 step 4 - close the two holes the first grid left open.

HOLE 1: 33% was the EDGE of the IPO weight axis and arm A was still improving there, so the
        optimum was not shown to be interior. Fix: the FULL weight simplex in 5% steps
        (w_TN, w_BA, w_IPO), 231 combinations, so the weight curve is seen end to end
        including 100% IPO.

HOLE 2: the frequency response was NON-MONOTONIC - quarterly beat monthly by ~2.9pp of CAGR
        while annual and never fell in between. A frequency that beats its own neighbours on
        both sides is a phase-luck candidate: 'quarterly' in the first grid meant ONE phase
        (Jan/Apr/Jul/Oct). Fix: every frequency is run over ALL of its phases and reported as
        a phase ensemble - median [min..max] across phases - so no single lucky calendar
        alignment can be mistaken for a rebalancing premium.

Both cost bases. Drawdowns from the FULL curve's running peak throughout (r/154).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/168_three_sleeve_blend/scripts'))
import blend_grid as G                                        # noqa: E402

RES = G.RES
STEP = 5
FREQS = {'monthly': 1, '2-monthly': 2, 'quarterly': 3, 'half-yearly': 6, 'annual': 12}
THIRDS = ['INC', 'A', 'CASH']


def month_boundaries(idx, k, phase):
    mi = idx.year * 12 + idx.month
    first = np.r_[True, mi[1:] != mi[:-1]]
    sel = first & (((mi - phase) % k) == 0)
    b = list(np.flatnonzero(sel))
    return [0] + [x for x in b if x != 0]


def simplex(step):
    out = []
    for wi in range(0, 101, step):
        for wt in range(0, 101 - wi, step):
            out.append((wt, 100 - wi - wt, wi))
    return out


def main():
    t0 = time.time()
    idx, S, bench = G.load()
    combos = simplex(STEP)
    bnds = {('never', 0): [0]}
    for f, k in FREQS.items():
        for p in range(k):
            bnds[(f, p)] = month_boundaries(idx, k, p)
    print('%d weight combos x %d rebalance (freq,phase) x %d thirds x %d bases'
          % (len(combos), len(bnds), len(THIRDS), len(G.BASES)), flush=True)

    rows = []
    keep = {}
    for basis, bp in G.BASES.items():
        for third in THIRDS:
            navs = np.stack([S['TN_%d' % bp['tn']], S['BA_%d' % bp['ba']],
                             S['%s_%d' % (third, bp['ipo'])]])
            for (wt, wb, wi) in combos:
                w = np.array([wt, wb, wi], float) / 100.0
                for (f, p), b in bnds.items():
                    nv = G.blend(navs, w, b)
                    m = G.metrics(nv, idx)
                    r = dict(basis=basis, third=third, w_tn=wt, w_ba=wb, w_ipo=wi,
                             freq=f, phase=p,
                             cagr=round(float(np.median(m['cagr'])), 3),
                             maxdd=round(float(np.median(m['maxdd'])), 3),
                             calmar=round(float(np.median(m['calmar'])), 4),
                             cagr_worst=round(float(m['cagr'].min()), 3),
                             maxdd_worst=round(float(m['maxdd'].min()), 3),
                             calmar_worst=round(float(m['calmar'].min()), 4),
                             wa=round(float(np.median(m['WA 2006-2015_cagr'])), 2),
                             wb_=round(float(np.median(m['WB 2016-2026_cagr'])), 2),
                             wa_dd=round(float(np.median(m['WA 2006-2015_dd'])), 2),
                             wb_dd=round(float(np.median(m['WB 2016-2026_dd'])), 2))
                    for k2 in G.WINDOWS:
                        r[k2] = round(float(np.median(m[k2 + '_ret'])), 2)
                        r[k2 + '_dd'] = round(float(np.median(m[k2 + '_dd'])), 2)
                    rows.append(r)
                    keep[(basis, third, wt, wb, wi, f, p)] = np.vstack(
                        [m['cagr'], m['maxdd'], m['calmar']])
        print('  %s done (%.0fs, %d rows)' % (basis, time.time() - t0, len(rows)), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / 'extend_grid.csv', index=False)
    print('\n%d cells in %.0fs' % (len(df), time.time() - t0), flush=True)

    # ─────────────── HOLE 2: the frequency phase ensemble, at three reference weights
    print('\n\n=== REBALANCE FREQUENCY: phase ensemble, published basis ===')
    print('A frequency is only trustworthy if its PHASES AGREE. median [min..max] over phases'
          ' of the 30-path median.')
    for label, (wt, wb, wi), third in (('2-sleeve TN50:BA50', (50, 50, 0), 'INC'),
                                       ('2-sleeve TN75:BA25', (75, 25, 0), 'INC'),
                                       ('+IPO-A 30%, TN50:BA50 rest', (35, 35, 30), 'A'),
                                       ('+IPO-A 30%, TN75:BA25 rest', (52, 18, 30), 'A')):
        d = df[(df.basis == 'published') & (df.third == third) & (df.w_tn == wt)
               & (df.w_ba == wb) & (df.w_ipo == wi)]
        if d.empty:
            continue
        print('\n--- %s' % label)
        print('%-13s %-28s %-28s %s' % ('freq', 'CAGR med [min..max phases]',
                                        'MaxDD med [min..max]', 'Calmar med [min..max]'))
        for f in list(FREQS) + ['never']:
            s = d[d.freq == f]
            if s.empty:
                continue
            print('%-13s %6.2f [%6.2f .. %6.2f]        %7.2f [%7.2f .. %7.2f]      '
                  '%5.3f [%5.3f .. %5.3f]   (%d phases)'
                  % (f, s.cagr.median(), s.cagr.min(), s.cagr.max(), s.maxdd.median(),
                     s.maxdd.min(), s.maxdd.max(), s.calmar.median(), s.calmar.min(),
                     s.calmar.max(), len(s)))

    # ─────────────── phase-averaged frequency choice
    pub = df[df.basis == 'published']
    ph = (pub.groupby(['third', 'w_tn', 'w_ba', 'w_ipo', 'freq'])
             .agg(cagr=('cagr', 'median'), maxdd=('maxdd', 'median'),
                  calmar=('calmar', 'median'), calmar_lo=('calmar', 'min'),
                  calmar_hi=('calmar', 'max'), cagr_lo=('cagr', 'min'),
                  cagr_hi=('cagr', 'max'), nph=('phase', 'count')).reset_index())
    ph.to_csv(RES / 'freq_phase_ensemble.csv', index=False)

    # ─────────────── HOLE 1: the weight curve end to end
    print('\n\n=== THE IPO WEIGHT CURVE, end to end (published basis) ===')
    print('TN:BA held at the ratio that wins the two-sleeve book; every frequency shown as its'
          ' phase median.')
    best2 = ph[(ph.third == 'INC') & (ph.w_ipo == 0)].sort_values('calmar',
                                                                  ascending=False).iloc[0]
    print('two-sleeve best (phase-median Calmar): TN%d:BA%d %s -> CAGR %.2f  DD %.2f  Cal %.3f'
          % (best2.w_tn, best2.w_ba, best2.freq, best2.cagr, best2.maxdd, best2.calmar))
    for freq in ['monthly', 'quarterly', 'annual']:
        for third in ('INC', 'A', 'CASH'):
            print('\n--- %s, rebalance %s, TN:BA fixed at %d:%d of the remainder'
                  % (third, freq, best2.w_tn, best2.w_ba))
            print('%6s %8s %9s %8s %9s %10s %8s %8s %8s %8s %8s'
                  % ('w_ipo', 'CAGR', 'MaxDD', 'Calmar', 'CAGRwst', 'MaxDDwst', 'WA', 'WB',
                     '2008', '2018', '2022H1'))
            for wi in range(0, 101, 5):
                r = (100 - wi)
                wt = int(round(r * best2.w_tn / (best2.w_tn + best2.w_ba) / 5.0) * 5)
                wb = r - wt
                s = ph[(ph.third == third) & (ph.w_ipo == wi) & (ph.w_tn == wt)
                       & (ph.w_ba == wb) & (ph.freq == freq)]
                if s.empty:
                    continue
                s = s.iloc[0]
                det = pub[(pub.third == third) & (pub.w_ipo == wi) & (pub.w_tn == wt)
                          & (pub.w_ba == wb) & (pub.freq == freq)]
                print('%5d%% %8.2f %9.2f %8.3f %9.2f %10.2f %8.2f %8.2f %8.2f %8.2f %8.2f'
                      % (wi, s.cagr, s.maxdd, s.calmar, det.cagr_worst.median(),
                         det.maxdd_worst.median(), det.wa.median(), det.wb_.median(),
                         det['2008'].median(), det['2018'].median(),
                         det['2022H1'].median()))

    # ─────────────── top of the full simplex
    print('\n\n=== TOP 15 OF THE FULL SIMPLEX by phase-median Calmar (published basis) ===')
    for third in ('INC', 'A', 'CASH'):
        t = ph[ph.third == third].sort_values('calmar', ascending=False).head(15)
        print('\n--- third = %s' % third)
        print(t[['w_tn', 'w_ba', 'w_ipo', 'freq', 'cagr', 'maxdd', 'calmar', 'calmar_lo',
                 'calmar_hi', 'nph']].to_string(index=False))

    print('\n\n=== SAME, harmonised basis (True North also at 25 bps a side) ===')
    har = df[df.basis == 'harmonised']
    ph2 = (har.groupby(['third', 'w_tn', 'w_ba', 'w_ipo', 'freq'])
              .agg(cagr=('cagr', 'median'), maxdd=('maxdd', 'median'),
                   calmar=('calmar', 'median'), nph=('phase', 'count')).reset_index())
    for third in ('INC', 'A', 'CASH'):
        t = ph2[ph2.third == third].sort_values('calmar', ascending=False).head(8)
        print('\n--- third = %s' % third)
        print(t.to_string(index=False))

    json.dump(dict(best_two_sleeve=best2.to_dict()),
              open(RES / 'extend_summary.json', 'w'), indent=1, default=str)
    np.savez_compressed(RES / 'extend_paths.npz',
                        **{'%s__%s__%d_%d_%d__%s__%d' % k: v for k, v in keep.items()
                           if k[0] == 'published' and k[2] % 10 == 0 and k[3] % 10 == 0})
    print('\nblend_extend done in %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
