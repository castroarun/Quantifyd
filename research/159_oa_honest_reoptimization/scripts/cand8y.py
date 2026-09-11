# -*- coding: utf-8 -*-
"""Piece 2, step 1: every OA breakout candidate over EIGHT years, not two.

The two-year fundamental test could not arbitrate: one entry said the screen was worth
twenty points, the other said nothing, and in that window merely changing the entry mechanic
moved the unfiltered result sixteen points on its own. With 120-200 trades in a single
regime that is unresolvable by construction.

It was two years only because Yahoo returns four fiscal years. Screener returned about
twelve, and 471 of the already-cached names have accounts back to FY2015. Needing four filed
years, decisions become testable from roughly mid-2018 - a window spanning the 2018-19 grind,
the pandemic crash, the 2020-21 melt-up, the 2022 correction and the recent smallcap cycle,
with five to ten times the trades.

This enumerates the candidates for that window so their accounts can be fetched. The scan
uses the CONFIRMED-breakout condition (close clears the prior all-time-high close) and the
CLEAN universe, so funds are already out - r/158 found 221 gold, silver and index funds in
the old ticker-based filter, and the two-year overlay's most eye-catching arm turned out to
be buying gold.

Writes the symbol list only. The fetch is a separate, resumable step.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/158_oa_arming_width/scripts'))
sys.path.insert(0, str(ROOT))
import oa_entry_mechanics as em          # noqa: E402

WIN = ('2018-07-01', '2026-09-04')
OUT = ROOT / 'research/159_oa_honest_reoptimization/results/oa_candidates_8y.json'

print('loading frames from 2005 so the pivot is a true all-time high ...', flush=True)
w = em.load_frames('2005-01-01', trail_sma=20)
close, tv20, athcp = w['close'], w['tv20'], w['athcp']

etf = [c for c in close.columns if em.is_etf(c)]
tv_prev, prev_close = tv20.shift(1), close.shift(1)
elig = tv_prev >= em.TV_FLOOR
elig[etf] = False
r = {n: close / close.shift(n) - 1 for n in (63, 126, 189, 252)}
rs = ((2 * r[63] + r[126] + r[189] + r[252]).where(elig).rank(axis=1, pct=True) * 100).shift(1)
setup = (prev_close < athcp) & (prev_close >= 0.8 * athcp) & elig & (rs >= 70.0)
trig = setup & (close > athcp) & athcp.notna()

m = (trig.index >= WIN[0]) & (trig.index <= WIN[1])
hit = trig.loc[m]
syms = sorted([c for c in hit.columns if bool(hit[c].any())])

old = json.load(open(ROOT / 'research/158_oa_arming_width/results/oa_candidates_2y.json'))
cached = {p.stem for p in
          (ROOT / 'research/158_oa_arming_width/results/screener_cache').glob('*.json')}
new = sorted(set(syms) - cached)

print()
print('window                 : %s -> %s' % WIN)
print('funds excluded         : %d' % len(etf))
print('distinct candidates    : %d   (two-year list was %d)' % (len(syms), len(old)))
print('already cached         : %d' % len(set(syms) & cached))
print('TO FETCH               : %d  (~%.1f hours at 5s each)'
      % (len(new), len(new) * 5 / 3600.0))
json.dump(syms, open(OUT, 'w'))
json.dump(new, open(OUT.with_name('to_fetch_8y.json'), 'w'))
print()
print('wrote %s and %s' % (OUT.name, 'to_fetch_8y.json'))
