# -*- coding: utf-8 -*-
"""research/169 S1c - why does Spec A's edge over its null shrink on the clean panel?

On r/167's panel (W2 only) Spec A beats its null by +4.78pp on 30/30 seeds. On this study's
clean panel it is +2.25pp and only 14/30 in WB. r/167 never ran the null in WA/WB. This runs
Spec A real + null in W2/WA/WB on each panel variant, at 5.0% idle cash (r/167's rate) so the
E1 row is directly comparable to r/167's published +4.78.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/169_ipo_rules_universe_transplant/scripts'))
import xpanel as xp   # noqa: E402
import run169 as r    # noqa: E402

path = xp.RES / 's1c_null_attribution.csv'
done = r.done_labels(path)
P = xp.Panel()
steps = [('N1_r167like', dict(adjust=False, robust=False, drop_phantom=False)),
         ('N2_plus_robust', dict(adjust=False, robust=True, drop_phantom=False)),
         ('N3_plus_phantom_drop', dict(adjust=False, robust=True, drop_phantom=True)),
         ('N4_plus_split_adjust', dict(adjust=True, robust=True, drop_phantom=True))]
for label, flags in steps:
    if label in done:
        continue
    P.build(**flags)
    row = r.cell(P, label, 'all', 'le6', cash=0.05, min_bars=25)
    row.update(flags)
    r.append(path, row)
cols = ['label', 'n_signals_w2', 'w2_cagr', 'w2_null_cagr', 'w2_edge', 'w2_wins', 'wa_cagr',
        'wa_null_cagr', 'wa_edge', 'wa_wins', 'wb_cagr', 'wb_null_cagr', 'wb_edge', 'wb_wins']
print(pd.read_csv(path)[cols].to_string(index=False), flush=True)
