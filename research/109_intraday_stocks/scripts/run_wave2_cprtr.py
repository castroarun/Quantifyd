#!/usr/bin/env python3
"""research/109 wave-2: CPRTR_S refinement (the sole stable wave-1 passer).
Narrow-CPR day + open below BC + first-15m red -> SHORT at 09:30.
Axes: CPR width {0.2,0.3,0.5}% x exit {EOD, TRAIL30 (close>30m high), STOP1(+1%)}.
Windows: IS 2015-02..2021-09, VAL 2021-10..2023-12. Costs 10bps RT.
Gate for book candidacy: VAL net >= +10bps, t >= 2."""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
sys.path.insert(0, str(STUDY.parents[1]))
from run_wave1 import universe, load5

COST = 0.0010
IS_END = pd.Timestamp('2021-09-30')

def run():
    res = {}
    uni = universe()
    for k, sym in enumerate(uni):
        df = load5(sym)
        if df is None: continue
        df['day'] = df['date'].dt.normalize()
        prev = None
        for day, dd in df.groupby('day'):
            o = dd['open'].to_numpy(); h = dd['high'].to_numpy()
            l = dd['low'].to_numpy(); c = dd['close'].to_numpy()
            n = len(dd)
            if n < 20: prev = None if n < 5 else {'H': h.max(),'L': l.min(),'C': c[-1]}; continue
            if prev:
                P = (prev['H']+prev['L']+prev['C'])/3; BC = (prev['H']+prev['L'])/2; TC = 2*P-BC
                width = abs(TC-BC)/prev['C']
                if o[0] < min(TC,BC) and c[2] < o[0]:
                    for wthr, wtag in ((0.002,'w20'),(0.003,'w30'),(0.005,'w50')):
                        if width >= wthr: continue
                        e = o[3]
                        if e <= 0: continue
                        eod = c[max(0,n-4)]
                        # E_EOD
                        exits = {'EOD': eod}
                        # E_TRAIL30: exit next open after close > rolling 6-bar high
                        x = None
                        for i in range(4, n-4):
                            hh = h[max(0,i-6):i].max()
                            if c[i] > hh and i+1 < n-3:
                                x = o[i+1]; break
                        exits['TRAIL30'] = x if x is not None else eod
                        # E_STOP1: +1% stop else EOD
                        x = None
                        stop = e*1.01
                        for i in range(3, n-4):
                            if o[i] >= stop: x = o[i]; break
                            if h[i] >= stop: x = stop; break
                        exits['STOP1'] = x if x is not None else eod
                        for etag, xp in exits.items():
                            net = (1 - xp/e) - COST
                            res.setdefault((wtag,etag), []).append((day, sym, net))
            prev = {'H': h.max(),'L': l.min(),'C': c[-1]}
        if (k+1) % 30 == 0: print(f'{k+1}/{len(uni)}', flush=True)
    print(f"{'cell':16s} {'win':>4s} {'n':>7s} {'net_bps':>8s} {'t':>6s} {'names+':>6s}")
    for (wtag,etag), rows in sorted(res.items()):
        d = pd.DataFrame(rows, columns=['day','sym','net'])
        for wname, sel in (('IS', d['day']<=IS_END), ('VAL', d['day']>IS_END)):
            g = d[sel]
            if len(g) < 200: continue
            v = g['net'].to_numpy()
            t = v.mean()/(v.std()/np.sqrt(len(v))+1e-12)
            np_ = (g.groupby('sym')['net'].mean() > 0).mean()
            print(f'{wtag}_{etag:10s} {wname:>4s} {len(v):7d} {v.mean()*1e4:8.1f} {t:6.2f} {np_:6.2f}')
    print('WAVE2 DONE')

run()
