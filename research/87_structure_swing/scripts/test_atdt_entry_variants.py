#!/usr/bin/env python3
"""ATDT asc-failure short: close-confirmation entry vs stop-at-the-line
entry (user critique, SBIN 2014-07-11 example). Same patterns, same H=15
session horizon, date-matched relative scoring."""
import sys, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_g1_daily_screen import confirmed_pivots, load_daily, load_universe

H = 15
COST_RT = 0.0010

def patterns(df):
    h, l = df['high'].to_numpy(), df['low'].to_numpy()
    c = df['close'].to_numpy()
    n = len(c)
    ph, pl, phi, pli = confirmed_pivots(h, l, 3)
    highs, lows = [], []
    armed = None   # (support_level, arm_bar, expiry)
    out = []
    for t in range(n):
        if not np.isnan(ph[t]): highs.append((ph[t], phi[t]))
        if not np.isnan(pl[t]): lows.append((pl[t], pli[t]))
        if len(highs) >= 2 and len(lows) >= 2:
            (h1, hi1), (h2, hi2) = highs[-2:]
            (l1, li1), (l2, li2) = lows[-2:]
            span = max(hi2, li2) - min(hi1, li1)
            if (10 <= span <= 60 and t - max(hi2, li2) <= 25
                    and abs(h1 - h2) / h1 <= 0.01 and l2 > l1 * 1.005):
                armed = (l2, t, t + 25)
        if armed:
            sup, ab, exp = armed
            if t > exp:
                armed = None
            else:
                # variant A: close-confirmation (original)
                if c[t] < sup:
                    out.append(('A', t, sup))
                    armed = None
                # variant B: intraday touch of the line
                elif l[t] < sup:
                    out.append(('B', t, sup))
                    armed = None
    return out

def main(win_start, win_end, label):
    uni = load_universe()
    data = {}
    for s in uni:
        d = load_daily(s)
        if d is not None: data[s] = d
    # date-matched forward means (from next open, H sessions)
    frames = []
    for s, df in data.items():
        o = df['open'].to_numpy()
        r = np.full(len(o), np.nan)
        if len(o) > H + 1:
            r[:-(H + 1)] = o[H + 1:] / o[1:len(o) - H] - 1
        frames.append(pd.Series(r, index=df.index, name=s))
    wide = pd.concat(frames, axis=1, sort=True)
    dmean = wide.mean(axis=1)
    res = {'A': [], 'B': [], 'B_fill_gain': []}
    per_name = {'A': {}, 'B': {}}
    for s, df in data.items():
        o = df['open'].to_numpy(); l = df['low'].to_numpy()
        c = df['close'].to_numpy()
        m = (df.index >= win_start) & (df.index <= win_end)
        dms = dmean.reindex(df.index).to_numpy()
        for kind, t, sup in patterns(df):
            if t + 1 + H >= len(df) or not m[t] or np.isnan(dms[t]): continue
            xo = o[t + 1 + H]
            bench = dms[t]
            if kind == 'A':
                e = o[t + 1]                       # next open after close-confirm
                stock = xo / e - 1
                res['A'].append(-(stock) - COST_RT - (-bench))
                per_name['A'].setdefault(s, []).append(-(stock) + bench)
            else:
                e = min(o[t], sup)                 # stop-sell at line (gap->open)
                stock = xo / e - 1
                res['B'].append(-(stock) - COST_RT - (-bench))
                per_name['B'].setdefault(s, []).append(-(stock) + bench)
        # ALSO: apply variant-B pricing to A-type signals (collapse bars):
        for kind, t, sup in patterns(df):
            if kind != 'A' or t + 1 + H >= len(df) or not m[t]: continue
            eA = o[t + 1]; eB = min(o[t], sup)
            res['B_fill_gain'].append((eA / eB - 1))
    for k in ('A', 'B'):
        v = np.array(res[k])
        t_ = v.mean() / (v.std() / np.sqrt(len(v)) + 1e-12)
        np_ = np.mean([np.mean(x) > 0 for x in per_name[k].values()])
        print(f'{label} variant {k}: n={len(v)} rel_net={v.mean()*1e4:.1f}bps t={t_:.2f} names+={np_:.2f}')
    fg = np.array(res['B_fill_gain'])
    print(f'{label} fill-price gain if stop-at-line on A-signals: mean {fg.mean()*1e4:.0f}bps (median {np.median(fg)*1e4:.0f})')

main('2005-01-01', '2017-12-31', 'IS ')
main('2018-01-01', '2022-06-30', 'VAL')
print('ENTRY VARIANTS DONE')
