# -*- coding: utf-8 -*-
"""research/164 — freeze the event list of the ADOPTED research/161 Base Age spec, with the
four ranking attributes the contested-slot axis needs.

Spec (unchanged, this study varies ONLY slots / slot_pct / selection rule):
    new all-time-high CLOSE, prior ATH >= 60 trading bars old, stock fell >= 20% below it
    in between, 20-day median traded value >= Rs 2 cr, no volume filter, no saucer filter,
    60-bar per-symbol re-arm. Fill at the NEXT open.

Ranking attributes, all causal (known on the trigger close, before the fill):
    rs252    12-month price return to the trigger close  (IBD-style relative strength)
    ext_pct  how far the entry open sits ABOVE the prior ATH -- "least extended" wins
    tv20_cr  20-day median traded value in Rs cr at the trigger
    x_bars   base age: trading bars from the prior ATH to the trigger
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
sys.path.insert(0, str(HERE))

REARM = 60
X_MIN, DEPTH_MIN, LIQ_MIN = 60, 20.0, 2.0
RS_LOOKBACK = 252


def rearm(df):
    keep = []
    for _, g in df.sort_values(['symbol', 'hist_bars']).groupby('symbol', sort=False):
        last = -10 ** 9
        for idx, hb in zip(g.index, g['hist_bars'].to_numpy()):
            if hb - last >= REARM:
                keep.append(idx); last = hb
    return df.loc[keep]


def main():
    t0 = time.time()
    panel = pickle.load(open(RES / 'panel164.pkl', 'rb'))
    print('panel: %d symbols, %d days (%s .. %s)'
          % (len(panel.close), panel.n, panel.cal[0], panel.cal[-1]), flush=True)

    raw = pd.read_csv(RES / 'ath_events.csv')
    ev = raw[raw['symbol'].isin(panel.close) & raw['entry_date'].isin(panel.pos)].copy()
    ev['entry_i'] = ev['entry_date'].map(panel.pos)
    s = ev[(ev['tv20_cr'] >= LIQ_MIN) & (ev['x_bars'] >= X_MIN)
           & (ev['depth_pct'] >= DEPTH_MIN)]
    s = rearm(s).copy()
    print('adopted-spec events after re-arm: %d' % len(s), flush=True)

    # ---- ranking attributes ------------------------------------------------------
    s['ext_pct'] = 100.0 * (s['entry_open'] / s['prev_ath'] - 1.0)
    rs = np.full(len(s), np.nan)
    for k, (sym, ei) in enumerate(zip(s['symbol'].to_numpy(), s['entry_i'].to_numpy(int))):
        ti = ei - 1                                  # the trigger bar
        c = panel.close[sym]
        if ti - RS_LOOKBACK >= 0:
            a, b = c[ti - RS_LOOKBACK], c[ti]
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                rs[k] = 100.0 * (b / a - 1.0)
    s['rs252'] = rs
    # a name with < 12 months of usable history ranks LAST on relative strength rather
    # than being dropped -- the book still owns the event, only its priority changes.
    s['rs252'] = s['rs252'].fillna(-1e9)
    print('rs252 available for %d / %d events (%.1f%%)'
          % (int((rs > -1e8).sum() if np.isfinite(rs).any() else 0), len(s),
             100.0 * np.isfinite(rs).mean()), flush=True)

    cols = ['symbol', 'trigger_date', 'entry_date', 'entry_i', 'x_bars', 'depth_pct',
            'tv20_cr', 'ext_pct', 'rs252', 'entry_open', 'prev_ath']
    out = s[cols].sort_values(['entry_i', 'symbol']).reset_index(drop=True)
    out.to_csv(RES / 'events164.csv', index=False)

    per_day = out.groupby('entry_i').size()
    print('\nevents %d over %d distinct signal days' % (len(out), len(per_day)))
    print('signals per signal-day: mean %.2f  median %d  p90 %d  max %d'
          % (per_day.mean(), per_day.median(), per_day.quantile(0.90), per_day.max()))
    for k in (1, 2, 3, 5, 8, 16):
        print('   days with >= %2d signals: %5d (%.1f%%)'
              % (k, int((per_day >= k).sum()), 100.0 * (per_day >= k).mean()))
    print('\nwrote events164.csv in %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
