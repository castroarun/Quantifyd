# -*- coding: utf-8 -*-
"""research/169 S1d - the MECHANISM behind S1c.

S1c: making the rolling windows NaN-robust lifts Spec A's REAL arm by +0.44pp but its NULL by
+3.2pp (and +6.2pp in WB), halving the edge. Hypothesis: on the union-index panel a single missing
row (partial-coverage special sessions, phantom holiday rows) makes SMA-50 NaN for 50 bars, and
the engine never trails on a NaN SMA - so those holdings exit only by stop or target. If that
blackout hurts random names more than breakouts, the old null was handicapped.

Measures, per panel x arm (W2, 30 seeds, 5.0% cash): exit-reason mix, mean return by reason,
share of trades whose SMA-50 was NaN on the exit bar or on any bar of the hold, and the share of
young-eligible cells (BARS >= 50) with a NaN SMA-50, by year.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/169_ipo_rules_universe_transplant/scripts'))
import xpanel as xp   # noqa: E402

out = {}
P = xp.Panel()
for label, flags in (('r167like', dict(adjust=False, robust=False, drop_phantom=False)),
                     ('robust', dict(adjust=False, robust=True, drop_phantom=False))):
    P.build(**flags)
    trig, pvn, lon, base = xp.build_signals(P, 'all', 'le6', min_bars=25)
    null = xp.build_null(P, trig, pvn, base)
    S50 = P.SMA[50]
    nan50 = ~np.isfinite(S50)
    cum = np.vstack([np.zeros((1, nan50.shape[1]), dtype=np.int64),
                     np.cumsum(nan50, axis=0, dtype=np.int64)])
    o = {}
    for arm, tg in (('real', trig), ('null', null)):
        res = xp.run(P, tg, pvn, lon, windows=('w2',), cash_yield=0.05, keep=True)
        tr = pd.DataFrame([dict(x, seed=s) for s, t in zip(xp.SEEDS, res['trades']) for x in t])
        tr['wb'] = P.dates[tr.ei.values] >= pd.Timestamp('2016-01-01')
        ei, xi, c = tr.ei.values, tr.xi.values, tr.col.values
        tr['nan_exit'] = nan50[xi, c]
        tr['nan_any'] = (cum[xi + 1, c] - cum[ei + 1, c]) > 0
        d = {}
        for per, g in (('w2', tr), ('wa', tr[~tr.wb]), ('wb', tr[tr.wb])):
            d[per] = dict(
                n=int(len(g)),
                reason_share={k: round(100 * v, 1) for k, v in
                              g.reason.value_counts(normalize=True).items()},
                mean_ret_by_reason={k: round(100 * v, 2) for k, v in
                                    g.groupby('reason').ret.mean().items()},
                mean_ret=round(100 * float(g.ret.mean()), 3),
                pct_trades_sma50_nan_on_some_hold_bar=round(100 * float(g.nan_any.mean()), 1),
                mean_ret_when_nan_hold=round(100 * float(g[g.nan_any].ret.mean()), 2)
                if g.nan_any.any() else None,
                mean_ret_when_clean_hold=round(100 * float(g[~g.nan_any].ret.mean()), 2)
                if (~g.nan_any).any() else None)
        o[arm] = d
        o[arm + '_cagr_w2'] = round(float(res['w2'].cagr.median()), 2)
    elig = base & (P.BARS >= 50)
    yrs = P.dates.year.values
    o['pct_young_eligible_cells_with_nan_sma50_by_year'] = {
        int(y): round(100 * float(nan50[yrs == y][elig[yrs == y]].mean()), 1)
        for y in sorted(set(yrs)) if elig[yrs == y].any()}
    out[label] = o
    print(label, json.dumps(o, indent=1), flush=True)
json.dump(out, open(xp.RES / 's1d_mechanism.json', 'w'), indent=1)
