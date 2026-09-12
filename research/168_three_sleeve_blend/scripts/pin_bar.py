# -*- coding: utf-8 -*-
"""research/168 step 6 - pin the exact weight at which the PRE-REGISTERED bar clears, and
check both verdicts under the drift (never-rebalance) arm as well as monthly."""
import sys

import numpy as np

sys.path.insert(0, '/home/arun/quantifyd/research/168_three_sleeve_blend/scripts')
import blend_grid as G                                        # noqa: E402
from final_report import mk, PUB, HAR                         # noqa: E402

idx, S, bench = G.load()
bnd = G.boundaries(idx, 'monthly')
drift = G.boundaries(idx, 'never')


def pp(a, b, label):
    ma, mb = G.metrics(a, idx), G.metrics(b, idx)
    dc, dd, dk = ma['cagr'] - mb['cagr'], ma['maxdd'] - mb['maxdd'], ma['calmar'] - mb['calmar']
    print('%-40s dCAGR %+7.3f (%2d/30)  dDD %+7.3f (%2d/30 shallower)  dCalmar %+7.4f (%2d/30)'
          % (label, np.median(dc), (dc > 0).sum(), np.median(dd), (dd > 0).sum(),
             np.median(dk), (dk > 0).sum()), flush=True)


print('\n--- A vs INC at the same weight, monthly (where does bar (b) clear?)')
for w in (20, 25, 30, 33, 35, 40, 45, 50):
    pp(mk(S, 'A', w, 0.5, PUB, bnd), mk(S, 'INC', w, 0.5, PUB, bnd),
       'A %d%% minus INC %d%%' % (w, w))

b2 = mk(S, 'INC', 0, 0.5, PUB, bnd)
print('\n--- A vs the two-sleeve baseline, monthly (where does bar (a) clear?)')
for w in (10, 15, 20, 25, 30, 35, 40, 50):
    pp(mk(S, 'A', w, 0.5, PUB, bnd), b2, 'A %d%% minus 2-sleeve' % w)
print('\n--- INC vs the two-sleeve baseline, monthly')
for w in (10, 15, 20, 25, 30, 35, 40, 50):
    pp(mk(S, 'INC', w, 0.5, PUB, bnd), b2, 'INC %d%% minus 2-sleeve' % w)

print('\n--- the same two verdicts under DRIFT (never rebalanced) - does the conclusion depend'
      ' on the frictionless-rebalancing assumption?')
b2d = mk(S, 'INC', 0, 0.5, PUB, drift)
for w in (25, 35):
    pp(mk(S, 'A', w, 0.5, PUB, drift), mk(S, 'INC', w, 0.5, PUB, drift),
       'DRIFT A %d%% minus INC %d%%' % (w, w))
    pp(mk(S, 'A', w, 0.5, PUB, drift), b2d, 'DRIFT A %d%% minus 2-sleeve' % w)

print('\n--- harmonised cost basis, monthly')
b2h = mk(S, 'INC', 0, 0.5, HAR, bnd)
for w in (25, 35):
    pp(mk(S, 'A', w, 0.5, HAR, bnd), mk(S, 'INC', w, 0.5, HAR, bnd),
       'HARM A %d%% minus INC %d%%' % (w, w))
    pp(mk(S, 'A', w, 0.5, HAR, bnd), b2h, 'HARM A %d%% minus 2-sleeve' % w)
    pp(mk(S, 'INC', w, 0.5, HAR, bnd), b2h, 'HARM INC %d%% minus 2-sleeve' % w)

print('\n--- WA (2006-2015) and WB (2016-2026) separately, monthly: does the A advantage hold'
      ' in BOTH halves?')
for w in (25, 35):
    for arm in ('A', 'INC'):
        m = G.metrics(mk(S, arm, w, 0.5, PUB, bnd), idx)
        mb = G.metrics(b2, idx)
        for win in ('WA 2006-2015', 'WB 2016-2026'):
            d = m[win + '_cagr'] - mb[win + '_cagr']
            print('%-40s %s  dCAGR %+7.3f (%2d/30)   level %6.2f vs %6.2f'
                  % ('%s %d%% minus 2-sleeve' % (arm, w), win, np.median(d), (d > 0).sum(),
                     np.median(m[win + '_cagr']), np.median(mb[win + '_cagr'])))
