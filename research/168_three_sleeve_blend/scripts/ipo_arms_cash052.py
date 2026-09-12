# -*- coding: utf-8 -*-
"""research/168 step 1 - the two IPO arms, 30 seeds each, at 5.2% post-tax idle cash.

IPO-INC  = the r/153 adopted spec on the PLACEABLE next-day buy-stop entry
           (trail SMA-20, stop 8%, target +25%, no gate) -- what the live paper book runs.
IPO-A    = r/167's re-fit (trail SMA-50, stop 10%, target +25%, NIFTYBEES < SMA-150 gate).

BOTH arms come out of ONE engine (research/167's fork, which carries the name-based fund
exclusion) so the comparison is apples-to-apples. Three cost levels: 25 / 40 / 60 bps a side.

STEP 1 IS A REPRODUCTION GATE. At cash_yield = 0.05 both arms must reproduce research/167's
published stage-9 medians (IPO-INC 14.90 / -38.57, IPO-A 21.80 / -26.63) to within 0.15pp.
If they do not, market_data.db has moved under the study -> stop, do not write a curve.

Nothing outside research/168_three_sleeve_blend/results/ is written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/167_ipo_base_honest_reopt/scripts'))
import ipo_honest as ih                                      # noqa: E402

OUT = ROOT / 'research/168_three_sleeve_blend/results'
OUT.mkdir(parents=True, exist_ok=True)
R167 = ROOT / 'research/167_ipo_base_honest_reopt/results/stage9_adoption.csv'

OLD_Y, NEW_Y = 0.05, 0.052
COSTS = (0.0025, 0.0040, 0.0060)
TOL = 0.15          # pp of CAGR the reproduction gate allows

ARMS = [('INC', {**ih.INCUMBENT}, None),
        ('A', {**ih.INCUMBENT, 'trail': 50, 'stop': 0.10, 'target': 0.25}, 150)]


def make_cfg(ctx, base, gate_n, cost):
    cfg = dict(base)
    cfg['cost'] = cost
    if gate_n:
        nb = ctx.close.get('NIFTYBEES').dropna()
        w = (nb < nb.rolling(gate_n).mean()).shift(1)
        cfg['gate'] = 'custom'
        cfg['weak_series'] = w.reindex(ctx.dates).ffill().fillna(False).to_numpy(bool)
    else:
        cfg['gate'] = False
    return cfg


def main():
    t0 = time.time()
    ctx, ir = ih.load_ctx(clean=True)
    pub = pd.read_csv(R167).set_index('spec')
    want = {'INC': ('incumbent_r153_ungated',), 'A': ('A_trail50_sl10_tp25_sma150',)}

    sf = OUT / 'ipo_seed_stats.csv'
    if sf.exists():
        sf.unlink()
    store, rows = {}, []
    dates_ref = None
    for arm, base, gate_n in ARMS:
        for cost in COSTS:
            cfg = make_cfg(ctx, base, gate_n, cost)
            setup, piv, lo0 = ih.build_setup(ctx, cfg)
            trig, lvl, lo, fc = ih.apply_mechanic(ctx, setup, piv, lo0, 'nextday_pivot')
            yields = (OLD_Y, NEW_Y) if cost == 0.0025 else (NEW_Y,)
            for y in yields:
                o, kept = ih.run_cell(ctx, ir, cfg, trig=trig, level=lvl, lo=lo, keep=True,
                                      windows=('w2',), fill_close=fc, cash_yield=y)
                st = kept['stats']
                navs = kept['navs']
                idx = navs[0].index
                if dates_ref is None:
                    dates_ref = idx
                elif not idx.equals(dates_ref):
                    print('!! date index changed between cells - stopping'); sys.exit(2)
                M = np.vstack([(s / s.iloc[0]).to_numpy(float) for s in navs])
                tag = '%s_%dbps_y%g' % (arm, int(cost * 10000), y * 1000)
                store[tag] = M
                rows.append(dict(arm=arm, cost_bps=int(cost * 10000), cash_y=y,
                                 cagr_med=o['w2_cagr'], cagr_worst=o['w2_cagr_lo'],
                                 cagr_best=o['w2_cagr_hi'], dd_med=o['w2_dd'],
                                 dd_worst=o['w2_dd_worst'], calmar=o['w2_calmar'],
                                 invested=o['w2_inv'], tpy=o['w2_tpy'], win=o['w2_win']))
                print('%-4s cost %3d bps  cash %.1f%%  CAGR %6.2f [%.2f..%.2f]  '
                      'DD %7.2f (worst %7.2f)  Cal %5.3f  inv %4.1f%%'
                      % (arm, int(cost * 10000), y * 100, o['w2_cagr'], o['w2_cagr_lo'],
                         o['w2_cagr_hi'], o['w2_dd'], o['w2_dd_worst'], o['w2_calmar'],
                         o['w2_inv']), flush=True)
                st.insert(0, 'cash_y', y)
                st.insert(0, 'cost_bps', int(cost * 10000))
                st.insert(0, 'arm', arm)
                st.insert(0, 'seed', ih.SEEDS)
                st.to_csv(sf, mode='a', header=not sf.exists(), index=False)

    # ------------------------------------------------ reproduction gate at 5.0%
    print('\n--- REPRODUCTION GATE vs research/167 stage9 (cash 5.0%%, 25 bps) ---', flush=True)
    ok = True
    for arm in ('INC', 'A'):
        spec = want[arm][0]
        r = [x for x in rows if x['arm'] == arm and x['cost_bps'] == 25
             and x['cash_y'] == OLD_Y][0]
        dc = r['cagr_med'] - float(pub.loc[spec, 'cagr'])
        dd = r['dd_med'] - float(pub.loc[spec, 'dd'])
        flag = 'OK' if abs(dc) <= TOL else 'FAIL'
        if abs(dc) > TOL:
            ok = False
        print('%-4s published %6.2f / %7.2f   re-run %6.2f / %7.2f   delta %+.3f pp / '
              '%+.3f pp   %s' % (arm, pub.loc[spec, 'cagr'], pub.loc[spec, 'dd'],
                                 r['cagr_med'], r['dd_med'], dc, dd, flag), flush=True)
    if not ok:
        print('!! REPRODUCTION FAILED - the engine or the data has moved. Stopping; '
              'no npz written.')
        sys.exit(2)
    print('REPRODUCTION PASSED.', flush=True)

    np.savez_compressed(OUT / 'ipo_navs_cash052.npz',
                        dates=np.array([str(d.date()) for d in dates_ref]),
                        seeds=np.array(ih.SEEDS), **store)
    pd.DataFrame(rows).to_csv(OUT / 'ipo_arms_summary.csv', index=False)
    print('\nwrote %s  (%d arrays, %.0fs)'
          % (OUT / 'ipo_navs_cash052.npz', len(store), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
