"""Stepwise signal-count probe: where does sweep eligibility diverge from the engine?"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
import bluesky_replay as br

START, END = '2006-01-01', '2026-08-31'
w = br.load_frames('2004-07-01')
close, high, athcp, tv20 = w['close'], w['high'], w['athcp'], w['tv20']
prev_close = close.shift(1)
tv_prev = tv20.shift(1)
dates = close.index
in_per = pd.Series([(START <= str(d.date()) <= END) for d in dates], index=dates)

def count(elig, rs_min=70.0, depth=0.20):
    score = 2*(close/close.shift(63)-1) + (close/close.shift(126)-1) \
        + (close/close.shift(189)-1) + (close/close.shift(252)-1)
    rs = (score.where(elig).rank(axis=1, pct=True) * 100).shift(1)
    setup = (prev_close < athcp) & (prev_close >= (1-depth)*athcp) & elig & (rs >= rs_min)
    trig = (setup & (close > athcp) & athcp.notna()).fillna(False)
    return int(trig.loc[in_per.values].sum().sum())

etf = [c for c in close.columns if br.ETF_RE.search(c)]
e1 = tv_prev >= br.TV_FLOOR
e1[etf] = False
print('tv+etf only          :', count(e1))

snap = json.load(open(STUDY / 'results' / 'mcap_snapshot.json'))
shares = pd.Series({s: v['mcap']/v['px'] for s, v in snap.items()
                    if v.get('mcap') and v.get('px')}).reindex(close.columns)
print('shares known:', int(shares.notna().sum()), 'of', len(close.columns))
mcap_prev = prev_close.mul(shares, axis=1)
print('mcap matrix dtype:', mcap_prev.dtypes.iloc[0], ' sample max:', np.nanmax(mcap_prev.values))
mc_ok = mcap_prev >= 500 * 1e7
print('tv+etf+mcap          :', count(e1 & mc_ok))
