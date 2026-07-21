#!/usr/bin/env python3
"""User-specified structured trade on corrected asc-triangle failure:
SHORT at the support line break (red line), SL above flat resistance
(maroon = pattern invalidation), TARGET = measured move (green: entry minus
triangle height at break). Time backstop 30 sessions. Daily, F&O universe."""
import sys, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_g1_daily_screen import confirmed_pivots, load_daily, load_universe

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
                    and max(l1, l2) < min(h1, h2) * 0.995):
                armed = (l2, min(h1, h2), t + 25)
        if armed:
            sup, res, exp = armed
            if t > exp:
                armed = None
            elif l[t] < sup:            # line touched -> stop-entry fills
                out.append((t, sup, res))
                armed = None
    return out

def run(label):
    uni = load_universe()
    data = {}
    for s in uni:
        d = load_daily(s)
        if d is not None: data[s] = d
    trades = []
    per_name = {}
    for s, df in data.items():
        o = df['open'].to_numpy(); h = df['high'].to_numpy()
        l = df['low'].to_numpy()
        m = (df.index >= WS) & (df.index <= WE)
        n = len(o)
        for (t, sup, res) in patterns(df):
            if not m[t] or t + 2 >= n: continue
            e = min(o[t], sup)              # stop-sell at the line
            sl = res * 1.005                # invalidation above resistance
            height = res - sup
            if height <= 0 or (sl - e) / e > 0.25: continue
            tgt = e - height                # measured move
            if tgt <= 0: continue
            risk = sl - e
            outcome = None
            for j in range(t + 1, min(t + 31, n)):
                if o[j] >= sl:   outcome = (o[j], j); break
                if h[j] >= sl:   outcome = (sl, j); break
                if o[j] <= tgt:  outcome = (o[j], j); break
                if l[j] <= tgt:  outcome = (tgt, j); break
            if outcome is None:
                j = min(t + 30, n - 1)
                outcome = (o[j], j)
            x, j = outcome
            net = (1 - x / e) - COST_RT
            trades.append((net, (e - x) / risk, j - t))
            per_name.setdefault(s, []).append(net)
    v = np.array([tr[0] for tr in trades])
    R = np.array([tr[1] for tr in trades])
    hold = np.mean([tr[2] for tr in trades])
    t_ = v.mean() / (v.std() / np.sqrt(len(v)) + 1e-12)
    np_ = np.mean([np.mean(x) > 0 for x in per_name.values()])
    win = (v > 0).mean()
    print(f'{label}: n={len(v)} net/trade={v.mean()*1e4:.1f}bps t={t_:.2f} '
          f'win={win:.2f} avgR={R.mean():.2f} names+={np_:.2f} hold={hold:.1f}d')

run(os.getenv('LBL', 'window'))
print('STRUCTURED DONE')
