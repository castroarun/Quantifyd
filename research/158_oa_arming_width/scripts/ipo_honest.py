# -*- coding: utf-8 -*-
"""The IPO Base number the LIVE engine can actually earn.

research/153's headline enters on the SAME day the close clears the pivot, priced at that
day's open:

    trig = ... & (ctx.C > piv)          # day i's CLOSE cleared the pivot
    cand = np.nonzero(TRIG[i])[0]       # entered on day i
    fill = max(pv, float(O[i, c]))      # at day i's OPEN

Deciding at the open with that day's closing price in hand is look-ahead, and it is not what
`services/ipo_paper.py` does. The live engine is correct: it triggers on tonight's close and
fills from yesterday's pending list the next morning. Its own docstring says so -
"trigger close[t] > pivot; fill next day, buy-stop AT the pivot, filled max(pivot, open)".

So the live book has been running an honest mechanic against a study that measured a
dishonest one. This produces the number the live mechanic earns, by shifting the decision
one day and requiring the resting stop to actually be reached:

    TRIG[i] <- TRIG[i-1]  AND  high[i] >= PIV[i-1]
    PIV[i]  <- PIV[i-1]

so the fill simulate_ipo already computes, max(PIV[i], O[i]), becomes max(pivot broken
yesterday, today's open) - exactly a buy-stop resting overnight at that pivot. A gap down
that never recovers is a MISS, which the same-day mode cannot represent.

Nothing else changes: same spec file, same seeds, same window, same exits, same costs, same
engine. Every difference in the result is the entry day.

Reported against two reference points from the study's own controls, so the comparison is
like for like: its headline, and its fill-at-the-signal-day-close arm.
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

SPEC = json.load(open(ROOT / 'research/153_ipo_base/results/ipo_adopted_spec.json'))
SEEDS = list(range(1, 31))
W = g3.W2


def report(label, df, extra=''):
    print('  %-42s CAGR %6.2f [%6.2f..%6.2f]  DD %7.2f%%  Calmar %5.2f  %s'
          % (label, df.cagr.median(), df.cagr.min(), df.cagr.max(),
             df.dd.median(), (df.cagr / df.dd.abs()).median(), extra))


print('building the panel ...', flush=True)
ctx = ir.Ctx()
trig, piv, lo, sma = g3.build(ctx, SPEC)
print('signals (same-day mode): %d' % int(trig.sum()), flush=True)

print()
print('=== IPO Base, adopted spec, %s -> %s, 30 seeds, 25bps ===' % W)

# 1. the published mechanic, reproduced so the baseline is this machine's own
_, _, d_pub = g3.run(ctx, SPEC, SEEDS, W, trig, piv, lo, sma)
report('PUBLISHED: same-day, fill max(pivot, open)', d_pub, '<- look-ahead')

# 2. the study's own close-fill control, for the far end of the range
_, _, d_cls = g3.run(ctx, SPEC, SEEDS, W, trig, piv, lo, sma, fill_close=True)
report('study control: fill at the signal-day CLOSE', d_cls)

# 3. what the LIVE engine does
pivn = np.full_like(piv, np.nan)
pivn[1:] = piv[:-1]
tn = np.zeros_like(trig)
tn[1:] = trig[:-1]
reached = np.zeros_like(trig)
with np.errstate(invalid='ignore'):
    reached = ctx.H >= pivn
trign = tn & reached & np.isfinite(pivn)
lon = np.full_like(lo, np.nan)
lon[1:] = lo[:-1]
print()
print('  next-day mode: %d of %d same-day signals survive as reachable fills (%.1f%%)'
      % (int(trign.sum()), int(trig.sum()), 100.0 * trign.sum() / max(trig.sum(), 1)))
_, _, d_live = g3.run(ctx, SPEC, SEEDS, W, trign, pivn, lon, sma)
report('LIVE ENGINE: next-day stop at the broken pivot', d_live, '<- the honest number')

# 4. and after tax, since the house rule decides net of tax
_, _, d_live_tax = g3.run(ctx, SPEC, SEEDS, W, trign, pivn, lon, sma,
                          stcg=0.20, ltcg=0.125)
report('LIVE ENGINE, after tax (20/12.5)', d_live_tax)
_, _, d_pub_tax = g3.run(ctx, SPEC, SEEDS, W, trig, piv, lo, sma, stcg=0.20, ltcg=0.125)
report('published, after tax, for scale', d_pub_tax)

print()
rows = [('published_same_day', d_pub), ('study_close_fill', d_cls),
        ('live_next_day', d_live), ('live_next_day_after_tax', d_live_tax),
        ('published_after_tax', d_pub_tax)]
out = pd.DataFrame([dict(arm=k, cagr_med=v.cagr.median(), cagr_min=v.cagr.min(),
                         cagr_max=v.cagr.max(), dd_med=v.dd.median())
                    for k, v in rows])
p = ROOT / 'research/158_oa_arming_width/results/ipo_honest_entry.csv'
out.to_csv(p, index=False)
print('wrote %s' % p)
