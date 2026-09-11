# -*- coding: utf-8 -*-
"""IPO Base's after-tax equity curve on the HONEST next-day entry.

research/153 publishes 31.0%, measured by entering on the SAME day the close clears the
pivot, priced at that day's open - which needs the closing price known at the open.
services/ipo_paper.py is correct and fills the NEXT morning. research/158 measured the
difference: 31.0% becomes 15.0%.

The study saves equity seeds only for its own (same-day) spec, so the honest curve does not
exist on disk. This produces it, on exactly the same spec, seeds, window and engine, shifting
only the entry day - the same transformation r/158 used:

    TRIG[i] <- TRIG[i-1]  AND  high[i] >= PIV[i-1]
    PIV[i]  <- PIV[i-1]

The median-CAGR seed is taken, never the mean of the thirty paths: averaging equity curves
manufactures a smoother line than any single book could have run.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/153_ipo_base/scripts'))
sys.path.insert(0, str(ROOT))
import ipo_replay as ir          # noqa: E402
import ipo_g3 as g3              # noqa: E402

OUT = ROOT / 'research/159_oa_honest_reoptimization/results/ipo_honest_curve.csv'
SPEC = json.load(open(ROOT / 'research/153_ipo_base/results/ipo_adopted_spec.json'))
SEEDS = list(range(1, 31))

print('building the IPO panel ...', flush=True)
ctx = ir.Ctx()
trig, piv, lo, sma = g3.build(ctx, SPEC)

pivn = np.full_like(piv, np.nan)
pivn[1:] = piv[:-1]
tn = np.zeros_like(trig)
tn[1:] = trig[:-1]
with np.errstate(invalid='ignore'):
    reached = ctx.H >= pivn
trign = tn & reached & np.isfinite(pivn)
lon = np.full_like(lo, np.nan)
lon[1:] = lo[:-1]
print('same-day signals %d -> next-day reachable %d (%.1f%%)'
      % (int(trig.sum()), int(trign.sum()), 100.0 * trign.sum() / max(trig.sum(), 1)),
      flush=True)

navs, _trades, stats = g3.run(ctx, SPEC, SEEDS, g3.W2, trign, pivn, lon, sma)
cagrs = stats['cagr'].tolist()
k = int(np.argsort(cagrs)[len(cagrs) // 2])
s = navs[k]
s = s / s.iloc[0]
s.to_csv(OUT, header=['nav'])
print()
print('IPO honest, after tax: CAGR median %.2f%%  [%.2f..%.2f]  DD median %.2f%%'
      % (stats.cagr.median(), stats.cagr.min(), stats.cagr.max(), stats.dd.median()))
print('median-CAGR seed curve: %s -> %s, final %.2fx'
      % (s.index[0].date(), s.index[-1].date(), s.iloc[-1]))
print('wrote %s' % OUT)
