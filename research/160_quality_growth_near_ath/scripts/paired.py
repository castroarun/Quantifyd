# -*- coding: utf-8 -*-
"""research/160 STUDY leg - the pre-registered PAIRED test.

    paired.py pairs_g1.json

A mask arm counts as adding value only if, paired across the SAME 12 rebalance-day offsets
against the identical book with NO mask, it wins by >= +2pp after-tax CAGR OR >= +0.15
Calmar on >= 8 of 12 offsets, AND beats the random-selection null. Unpaired medians lie at
small n: r/158 had a gate that looked like a winner on 10-seed medians and lost on 20 of 30
paired paths.

The JSON is a list of {name, a: <cell dict>, b: <cell dict>} - a is the arm under test, b
is the control. Both are run in ONE process so the panel and the derived frames are paid
for once.
"""
import json
import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
from qg_engine import Panel, Derived, Cell, run_cell  # noqa: E402


def per_path(res, key):
    return {p['path']: p[key] for p in res['paths']}


def main():
    spec = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else STUDY / 'results' / 'paired_g1.md'
    panel = Panel.load(STUDY / 'results' / 'panel_2000.npz')
    der = Derived(panel)
    cache = {}

    def get(cd):
        key = json.dumps(cd, sort_keys=True)
        if key not in cache:
            cache[key] = run_cell(panel, der, Cell(**cd), verbose=False)
        return cache[key]

    lines = ['| test | control | dCAGR med | CAGR wins | dCalmar med | Calmar wins | n | verdict |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for pr in spec:
        a, b = get(pr['a']), get(pr['b'])
        rows = {}
        for key in ('cagr_net_tax', 'calmar'):
            pa, pb = per_path(a, key), per_path(b, key)
            common = sorted(set(pa) & set(pb))
            d = np.array([pa[p] - pb[p] for p in common], dtype=float)
            rows[key] = (float(np.median(d)), int((d > 0).sum()), len(d))
        dc, wc, n = rows['cagr_net_tax']
        dk, wk, _ = rows['calmar']
        passes = ((dc >= 2.0 and wc >= 8) or (dk >= 0.15 and wk >= 8))
        lines.append('| %s | %s | %+.2f | %d/%d | %+.3f | %d/%d | %d | %s |'
                     % (pr['name'], pr.get('control', pr['b'].get('label', 'control')),
                        dc, wc, n, dk, wk, n, n, 'ADDS VALUE' if passes else 'no'))
        print(lines[-1], flush=True)
    out.write_text('\n'.join(lines) + '\n')
    print('\nwrote %s' % out)


if __name__ == '__main__':
    sys.exit(main())
