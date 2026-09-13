# -*- coding: utf-8 -*-
"""research/170 Part A -- the pre-registered PAIRED test, and the equity dumps.

A rank-leeway candidate counts as an improvement only if, paired against the incumbent
across the SAME 12 rebalance-day offsets, it wins by >= +2pp after-tax CAGR at no worse
drawdown OR >= +0.15 Calmar, on >= 8 of 12 offsets, in BOTH windows. Unpaired medians lie
at small n, which is why this file exists at all.

    paired170a.py            -> results/pairedA.md  (+ <label>_equity.csv for the figure)
"""
import json
import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qg_engine170 import Panel, Derived, Cell, run_cell                # noqa: E402

ROOT = STUDY.parent.parent
MASK = str(ROOT / 'research' / '160_quality_growth_near_ath' / 'results' / 'masks_study'
           / 'b7_g10_qual_mc.npz')
PANEL_NPZ = ROOT / 'research' / '160_quality_growth_near_ath' / 'results' / 'panel_2000.npz'

WINDOWS = [('full', '2018-08-01', '2026-09-10'),
           ('W1 fit', '2018-08-01', '2022-06-30'),
           ('W2 holdout', '2022-07-01', '2026-09-10')]

INCUMBENT = dict(buffer=1.5, slots=15, cadence='monthly', k=0.90)

# label -> (the one thing changed, human description)
CANDIDATES = [
    ('leeway rank 15 (no leeway)', dict(buffer=1.0)),
    ('leeway rank 20', dict(buffer=1.33)),
    ('leeway rank 26 (Arun: "top 25")', dict(buffer=1.67)),
    ('leeway rank 30', dict(buffer=2.0)),
    ('leeway rank 38', dict(buffer=2.5)),
    ('leeway rank 45', dict(buffer=3.0)),
    ('band k = 0.85', dict(k=0.85)),
    ('band k = 0.85 + leeway rank 26', dict(k=0.85, buffer=1.67)),
    ('N = 10, quarterly', dict(slots=10, cadence='quarterly')),
    ('retain = loose (state not required)', dict(retain='loose')),
]

DUMP = {'D_incumbent': {}, 'D_leeway26': dict(buffer=1.67), 'D_leeway38': dict(buffer=2.5),
        'D_noleeway': dict(buffer=1.0), 'D_loose': dict(retain='loose')}


def cell(label, start, end, **kw):
    d = dict(label=label, start=start, end=end, entry='rebalance', cadence='monthly',
             rank='rs', slots=15, buffer=1.5, retain='strict', state='near', k=0.90,
             tv_floor=2.0, mask=MASK, mask_missing='fail', exits='none',
             index_gate='none', gate_action='block_new', fill='next_open',
             cost_bps=25.0, tax=True, cash_yield=0.052, max_position_pct=0.30,
             offsets=12, seeds=0, arms='all')
    d.update(INCUMBENT)
    d.update(kw)
    return Cell(**d)


def per_path(res, key):
    return {p['path']: p[key] for p in res['paths']}


def diff(a, b, key):
    pa, pb = per_path(a, key), per_path(b, key)
    ks = sorted(set(pa) & set(pb))
    d = [pa[k] - pb[k] for k in ks]
    return float(np.median(d)), int(sum(1 for x in d if x > 0)), len(d)


def main():
    panel = Panel.load(str(PANEL_NPZ))
    der = Derived(panel)
    cache = {}

    def run(kw, start, end, label):
        key = json.dumps([kw, start, end], sort_keys=True, default=str)
        if key not in cache:
            cache[key] = run_cell(panel, der, cell(label, start, end, **kw), verbose=False)
            print('  ran %-40s %s..%s  CAGR %6.2f  DD %7.2f  Calmar %5.2f'
                  % (label, start, end, cache[key]['row']['cagr_net_tax'],
                     cache[key]['row']['maxdd'], cache[key]['row']['calmar']), flush=True)
        return cache[key]

    lines = ['# research/170 Part A -- paired test against the Quality Summit incumbent',
             '',
             'Incumbent = research/160 Family-B `b7`: k 0.90, N 15, monthly, leeway '
             '`buffer` 1.5 -> **a holding is kept while ranked 23rd or better**, no exit, '
             '25 bps, after tax, idle cash 5.2%, 12 rebalance-day offsets.',
             '',
             'Every row changes exactly ONE thing. Deltas are medians of the per-offset '
             'difference, A minus incumbent, on the same offset.',
             '']

    for wname, ws, we in WINDOWS:
        base = run({}, ws, we, 'BASE_%s' % wname.split()[0])
        lines += ['## %s (%s -> %s)' % (wname, ws, we), '',
                  '| candidate | CAGR | MaxDD | Calmar | dCAGR | CAGR wins | dCalmar | '
                  'Calmar wins | dMaxDD | DD wins |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
                  '| **incumbent (leeway rank 23)** | **%.2f** | **%.2f** | **%.3f** | '
                  '— | — | — | — | — | — |'
                  % (base['row']['cagr_net_tax'], base['row']['maxdd'],
                     base['row']['calmar'])]
        for desc, kw in CANDIDATES:
            a = run(kw, ws, we, desc.replace(' ', '_')[:30] + '_' + wname.split()[0])
            dc, wc, n = diff(a, base, 'cagr_net_tax')
            dk, wk, _ = diff(a, base, 'calmar')
            dd, wd, _ = diff(a, base, 'maxdd')
            lines.append('| %s | %.2f | %.2f | %.3f | %+.2f | %d/%d | %+.3f | %d/%d | '
                         '%+.2f | %d/%d |'
                         % (desc, a['row']['cagr_net_tax'], a['row']['maxdd'],
                            a['row']['calmar'], dc, wc, n, dk, wk, n, dd, wd, n))
        lines.append('')

    out = STUDY / 'results' / 'pairedA.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('wrote %s' % out)

    # ---- equity dumps for the figure -------------------------------------------------
    import pandas as pd
    for lab, kw in DUMP.items():
        r = run(kw, WINDOWS[0][1], WINDOWS[0][2], lab)
        pd.DataFrame(r['curves']).to_csv(STUDY / 'results' / ('%s_equity.csv' % lab))
        print('  dumped %s_equity.csv' % lab)


if __name__ == '__main__':
    main()
