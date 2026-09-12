# -*- coding: utf-8 -*-
"""research/163 stage 8 - IS THE EDGE A PLATEAU OR A SPIKE?

Stage 7 left the decisive ambiguity: the gated refit beats the date-matched null by +4.91pp
at trail 50 (30/30 seeds) but by only +1.26pp at trail 30. Trail 50 is also the cell that
spikes ~4pp above its CAGR neighbours. Either the edge genuinely needs a slow exit - in
which case the edge should rise smoothly along the trail axis - or trail 50 is an overfit
cell and the real edge is near +1pp.

So: run the real arm AND its date-matched null at every trail value, gated, 30 paired seeds,
and look at the shape of the DIFFERENCE. A smooth rise is a finding. A spike at 50 is noise.
Stop and target held at the plateau values (10% / +25%).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/167_ipo_base_honest_reopt/scripts'))
import ipo_honest as ih                                      # noqa: E402

RES = ih.RES
TRAILS = (10, 15, 20, 30, 40, 50, 60, 75, 100)


def main():
    ctx, ir = ih.load_ctx(clean=True)
    nb = ctx.close.get('NIFTYBEES').dropna()
    g150 = ((nb < nb.rolling(150).mean()).shift(1).reindex(ctx.dates)
            .ffill().fillna(False).to_numpy(bool))
    rows = []
    for stop in (0.10, 0.15):
        for tr in TRAILS:
            cfg = {**ih.INCUMBENT, 'trail': tr, 'target': 0.25, 'stop': stop,
                   'gate': 'custom', 'weak_series': g150}
            setup, piv, lo0 = ih.build_setup(ctx, cfg)
            trig, lvl, lo, fc = ih.apply_mechanic(ctx, setup, piv, lo0, 'nextday_pivot')
            pvn, lon = ih._shift(piv), ih._shift(lo)
            young = ((ctx.AGE > 0) & (ctx.AGE <= cfg['max_age_m'] * 30.44)
                     & (ctx.BARS >= cfg['min_bars']) & ctx.ELIG)
            with np.errstate(invalid='ignore'):
                reach = ctx.H >= pvn
            ysh = ih._shift(young)
            rng = np.random.default_rng(20260912)      # SAME draw at every trail value
            nper = trig.sum(axis=1)
            null = np.zeros_like(trig)
            for i in np.nonzero(nper)[0]:
                pool = np.nonzero(ysh[i] & reach[i] & np.isfinite(pvn[i]))[0]
                if not len(pool):
                    continue
                k = min(int(nper[i]), len(pool))
                null[i, rng.choice(pool, size=k, replace=False)] = True
            oR, kR = ih.run_cell(ctx, ir, cfg, trig=trig, level=pvn, lo=lon, keep=True,
                                 windows=('w2',), fill_close=False)
            oN, kN = ih.run_cell(ctx, ir, cfg, trig=null, level=pvn, lo=lon, keep=True,
                                 windows=('w2',), fill_close=False)
            d = kR['stats'].cagr.values - kN['stats'].cagr.values
            rows.append(dict(stop=stop, trail=tr, real=oR['w2_cagr'], null=oN['w2_cagr'],
                             real_worst=oR['w2_cagr_lo'], real_dd=oR['w2_dd'],
                             real_calmar=oR['w2_calmar'], real_inv=oR['w2_inv'],
                             real_mean_tr=oR['w2_mean'], null_mean_tr=oN['w2_mean'],
                             edge_per_trade=round(oR['w2_mean'] - oN['w2_mean'], 3),
                             paired_edge=round(float(np.median(d)), 2),
                             edge_lo=round(float(d.min()), 2),
                             edge_hi=round(float(d.max()), 2),
                             real_wins=int((d > 0).sum())))
            r = rows[-1]
            print(f'stop {stop:.2f} trail {tr:>4}  real {r["real"]:6.2f}  null {r["null"]:6.2f}'
                  f'  paired edge {r["paired_edge"]:+6.2f}pp [{r["edge_lo"]:+6.2f}..'
                  f'{r["edge_hi"]:+6.2f}] wins {r["real_wins"]:2d}/30  '
                  f'per-trade {r["real_mean_tr"]:5.2f} vs {r["null_mean_tr"]:5.2f} '
                  f'= {r["edge_per_trade"]:+5.2f}pp  DD {r["real_dd"]:7.2f} '
                  f'Cal {r["real_calmar"]:5.3f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / 'stage8_null_axis.csv', index=False)
    print('\n--- paired edge (pp CAGR, real - null) by trail ---')
    print(df.pivot_table(index='stop', columns='trail', values='paired_edge').to_string())
    print('\n--- per-trade edge (pp) by trail ---')
    print(df.pivot_table(index='stop', columns='trail', values='edge_per_trade').to_string())
    print('\nstage8 written', flush=True)


if __name__ == '__main__':
    main()
