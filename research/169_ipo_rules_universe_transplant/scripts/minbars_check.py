# -*- coding: utf-8 -*-
"""research/169 S1b - independent check of the min_bars finding with research/167's OWN engine.

E5 in s1_equiv.csv says Spec A at min_bars=60 returns ~12% against 21.80% at min_bars=25.
services/ipo_paper.py runs MIN_BARS = 60. Confirm on the r/167 panel, unchanged, with the null.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/167_ipo_base_honest_reopt/scripts'))
import ipo_honest as ih  # noqa: E402

RES = ROOT / 'research/169_ipo_rules_universe_transplant/results'
ctx, irr = ih.load_ctx(clean=True)
nb = ctx.close.get('NIFTYBEES').dropna()
g150 = ((nb < nb.rolling(150).mean()).shift(1).reindex(ctx.dates).ffill().fillna(False)
        .to_numpy(bool))
out = {}
for mb in (25, 40, 60):
    cfg = {**ih.INCUMBENT, 'trail': 50, 'stop': 0.10, 'target': 0.25, 'min_bars': mb,
           'gate': 'custom', 'weak_series': g150}
    setup, piv, lo0 = ih.build_setup(ctx, cfg)
    trig, lvl, lo, fc = ih.apply_mechanic(ctx, setup, piv, lo0, 'nextday_pivot')
    o, kR = ih.run_cell(ctx, irr, cfg, trig=trig, level=lvl, lo=lo, keep=True,
                        windows=('w2', 'wa', 'wb'), fill_close=fc, cash_yield=0.05)
    pvn = ih._shift(piv)
    young = ((ctx.AGE > 0) & (ctx.AGE <= 6 * 30.44) & (ctx.BARS >= mb) & ctx.ELIG)
    with np.errstate(invalid='ignore'):
        reach = ctx.H >= pvn
    ysh = ih._shift(young)
    rng = np.random.default_rng(20260912)
    nper = trig.sum(axis=1)
    null = np.zeros_like(trig)
    for i in np.nonzero(nper)[0]:
        pool = np.nonzero(ysh[i] & reach[i] & np.isfinite(pvn[i]))[0]
        if len(pool):
            null[i, rng.choice(pool, size=min(int(nper[i]), len(pool)), replace=False)] = True
    oN, kN = ih.run_cell(ctx, irr, cfg, trig=null, level=pvn, lo=lo, keep=True,
                         windows=('w2',), fill_close=False, cash_yield=0.05)
    d = kR['stats'].cagr.values - kN['stats'].cagr.values
    out[mb] = dict(signals=int(trig.sum()), w2_cagr=o['w2_cagr'], w2_worst=o['w2_cagr_lo'],
                   w2_dd=o['w2_dd'], w2_calmar=o['w2_calmar'], wa=o['wa_cagr'], wb=o['wb_cagr'],
                   tpy=o['w2_tpy'], null=oN['w2_cagr'], edge=round(float(np.median(d)), 2),
                   wins=int((d > 0).sum()))
    print(mb, out[mb], flush=True)
json.dump(out, open(RES / 's1b_minbars_r167engine.json', 'w'), indent=1)
