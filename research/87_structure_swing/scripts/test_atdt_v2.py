#!/usr/bin/env python3
"""ATDT corrected detector (lows must be BELOW resistance) + entry variants.
Window from R87_WIN_START/R87_WIN_END env (also drives load_daily cap)."""
import sys, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_g1_daily_screen import confirmed_pivots, load_daily, load_universe

H = 15
COST_RT = 0.0010
WS = os.getenv('R87_WIN_START'); WE = os.getenv('R87_WIN_END')

def patterns(df):
    h, l = df['high'].to_numpy(), df['low'].to_numpy()
    c = df['close'].to_numpy()
    n = len(c)
    ph, pl, phi, pli = confirmed_pivots(h, l, 3)
    highs, lows = [], []
    armed = None
    out = []
    for t in range(n):
        if not np.isnan(ph[t]): highs.append((ph[t], phi[t]))
        if not np.isnan(pl[t]): lows.append((pl[t], pli[t]))
        if len(highs) >= 2 and len(lows) >= 2:
            (h1, hi1), (h2, hi2) = highs[-2:]
            (l1, li1), (l2, li2) = lows[-2:]
            span = max(hi2, li2) - min(hi1, li1)
            if (10 <= span <= 60 and t - max(hi2, li2) <= 25
                    and abs(h1 - h2) / h1 <= 0.01
                    and l2 > l1 * 1.005
                    and max(l1, l2) < min(h1, h2) * 0.995):   # CORRECTED: lows below resistance
                armed = (l2, t + 25)
        if armed:
            sup, exp = armed
            if t > exp:
                armed = None
            elif c[t] < sup:
                out.append(('A', t, sup)); armed = None
            elif l[t] < sup:
                out.append(('B', t, sup)); armed = None
    return out

uni = load_universe()
data = {}
for s in uni:
    d = load_daily(s)
    if d is not None: data[s] = d
frames = []
for s, df in data.items():
    o = df['open'].to_numpy()
    r = np.full(len(o), np.nan)
    if len(o) > H + 1:
        r[:-(H + 1)] = o[H + 1:] / o[1:len(o) - H] - 1
    frames.append(pd.Series(r, index=df.index, name=s))
wide = pd.concat(frames, axis=1, sort=True)
dmean = wide.mean(axis=1)
res = {'A': [], 'B': [], 'AB': []}   # AB = stop-at-line fills on everything
pn = {'A': {}, 'B': {}, 'AB': {}}
for s, df in data.items():
    o = df['open'].to_numpy(); c = df['close'].to_numpy()
    m = (df.index >= WS) & (df.index <= WE)
    dms = dmean.reindex(df.index).to_numpy()
    for kind, t, sup in patterns(df):
        if t + 1 + H >= len(df) or not m[t] or np.isnan(dms[t]): continue
        xo = o[t + 1 + H]; bench = dms[t]
        if kind == 'A':
            eA = o[t + 1]
            rA = -(xo / eA - 1) - COST_RT + bench
            res['A'].append(rA); pn['A'].setdefault(s, []).append(rA)
            eB = min(o[t], sup)
            rB = -(xo / eB - 1) - COST_RT + bench
            res['AB'].append(rB); pn['AB'].setdefault(s, []).append(rB)
        else:
            eB = min(o[t], sup)
            rB = -(xo / eB - 1) - COST_RT + bench
            res['B'].append(rB); pn['B'].setdefault(s, []).append(rB)
            res['AB'].append(rB); pn['AB'].setdefault(s, []).append(rB)
label = {'A': 'A close-confirm, next-open entry', 'B': 'B false-breaks only (touch, no close-below)',
         'AB': 'A+B trader version: stop order AT the line'}
for k in ('A', 'B', 'AB'):
    v = np.array(res[k])
    if not len(v): print(f'{k}: n=0'); continue
    t_ = v.mean() / (v.std() / np.sqrt(len(v)) + 1e-12)
    np_ = np.mean([np.mean(x) > 0 for x in pn[k].values()])
    print(f'{label[k]:46s} n={len(v):5d} rel_net={v.mean()*1e4:7.1f}bps t={t_:6.2f} names+={np_:.2f}')
print('V2 DONE')
