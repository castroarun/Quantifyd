# -*- coding: utf-8 -*-
"""research/163 stage 7 - the null control on the ADOPTION CANDIDATE (gated refit).

Stage 3 ran the null on the ungated refit. The candidate carries the NIFTYBEES<SMA150 gate,
so the null has to carry it too - otherwise the real arm is being credited with the gate's
work. Same days, same counts, same fill convention, same gate, 30 paired seeds.

Also decomposes the refit's gain: how much of it is "a longer trail helps ANY young liquid
name" (visible in the null) versus "the base breakout finally pays" (the paired edge).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/167_ipo_base_honest_reopt/scripts'))
import ipo_honest as ih                                      # noqa: E402

RES = ih.RES
REFIT = {**ih.INCUMBENT, 'trail': 50, 'target': 0.25, 'stop': 0.15}
# the discounted-plateau spec: the centre of the trail 30-75 x stop 8-15 region, not the peak
PLATEAU = {**ih.INCUMBENT, 'trail': 30, 'target': 0.25, 'stop': 0.10}


def main():
    ctx, ir = ih.load_ctx(clean=True)
    nb = ctx.close.get('NIFTYBEES').dropna()
    g150 = ((nb < nb.rolling(150).mean()).shift(1).reindex(ctx.dates)
            .ffill().fillna(False).to_numpy(bool))
    rows = []
    for name, base, gate in (('incumbent_gated', dict(ih.INCUMBENT), g150),
                             ('refit_sma150', dict(REFIT), g150),
                             ('plateau_tr30_sl10_sma150', dict(PLATEAU), g150),
                             ('plateau_tr30_sl10_nogate', dict(PLATEAU), None)):
        cfg = dict(base)
        if gate is not None:
            cfg['gate'] = 'custom'
            cfg['weak_series'] = gate
        else:
            cfg['gate'] = False
        setup, piv, lo0 = ih.build_setup(ctx, cfg)
        trig, lvl, lo, fc = ih.apply_mechanic(ctx, setup, piv, lo0, 'nextday_pivot')
        pvn, lon = ih._shift(piv), ih._shift(lo)
        young = ((ctx.AGE > 0) & (ctx.AGE <= cfg['max_age_m'] * 30.44)
                 & (ctx.BARS >= cfg['min_bars']) & ctx.ELIG)
        with np.errstate(invalid='ignore'):
            reach = ctx.H >= pvn
        yshift = ih._shift(young)
        rng = np.random.default_rng(20260912)
        nper = trig.sum(axis=1)
        null = np.zeros_like(trig)
        for i in np.nonzero(nper)[0]:
            pool = np.nonzero(yshift[i] & reach[i] & np.isfinite(pvn[i]))[0]
            if not len(pool):
                continue
            k = min(int(nper[i]), len(pool))
            null[i, rng.choice(pool, size=k, replace=False)] = True
        oR, kR = ih.run_cell(ctx, ir, cfg, trig=trig, level=pvn, lo=lon, keep=True,
                             windows=('w2',), fill_close=False)
        oN, kN = ih.run_cell(ctx, ir, cfg, trig=null, level=pvn, lo=lon, keep=True,
                             windows=('w2',), fill_close=False)
        d = kR['stats'].cagr.values - kN['stats'].cagr.values
        dd = kR['stats'].dd.values - kN['stats'].dd.values
        rows.append(dict(spec=name, real_cagr=oR['w2_cagr'], real_worst=oR['w2_cagr_lo'],
                         real_dd=oR['w2_dd'], real_dd_worst=oR['w2_dd_worst'],
                         real_calmar=oR['w2_calmar'], real_inv=oR['w2_inv'],
                         real_mean_tr=oR['w2_mean'], real_tpy=oR['w2_tpy'],
                         null_cagr=oN['w2_cagr'], null_dd=oN['w2_dd'],
                         null_calmar=oN['w2_calmar'], null_mean_tr=oN['w2_mean'],
                         paired_cagr_delta=round(float(np.median(d)), 2),
                         paired_lo=round(float(d.min()), 2),
                         paired_hi=round(float(d.max()), 2),
                         real_wins=int((d > 0).sum()),
                         paired_dd_delta=round(float(np.median(dd)), 2)))
        print(f'{name:<28} real {oR["w2_cagr"]:6.2f} (worst {oR["w2_cagr_lo"]:5.2f}, '
              f'DD {oR["w2_dd"]:7.2f}, Cal {oR["w2_calmar"]:5.3f}, inv {oR["w2_inv"]:4.1f}%, '
              f'{oR["w2_mean"]:5.2f}%/tr)  vs NULL {oN["w2_cagr"]:6.2f} '
              f'({oN["w2_mean"]:5.2f}%/tr)  paired {np.median(d):+6.2f}pp '
              f'[{d.min():+6.2f}..{d.max():+6.2f}] real wins {int((d>0).sum())}/30',
              flush=True)
    pd.DataFrame(rows).to_csv(RES / 'stage7_null_gated.csv', index=False)
    print('\nstage7 written', flush=True)


if __name__ == '__main__':
    main()
