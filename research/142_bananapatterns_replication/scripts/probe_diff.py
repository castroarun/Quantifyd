"""A/B probe: CLI-style vs sweep-style input construction — where do they diverge?"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
import bluesky_replay as br

START, END = '2006-01-01', '2026-08-31'

# --- common load (CLI uses start-550d = 2004-06-29; sweep used 2004-07-01) ---
wA = br.load_frames('2004-06-29')   # CLI style
wB = br.load_frames('2004-07-01')   # sweep style

def build(w, sweep_style):
    close, high, open_, athcp, sma50, tv20 = (w[k] for k in
        ('close', 'high', 'open', 'athcp', 'sma50', 'tv20'))
    etf = [c for c in close.columns if br.ETF_RE.search(c)]
    tv_prev = tv20.shift(1)
    prev_close = close.shift(1)
    elig = tv_prev >= br.TV_FLOOR
    elig[etf] = False
    r63 = close/close.shift(63)-1; r126 = close/close.shift(126)-1
    r189 = close/close.shift(189)-1; r252 = close/close.shift(252)-1
    score = 2*r63 + r126 + r189 + r252
    rs = (score.where(elig).rank(axis=1, pct=True)*100).shift(1)
    setup = (prev_close < athcp) & (prev_close >= 0.8*athcp) & elig & (rs >= 70.0)
    trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
    nb = close['NIFTYBEES']
    weak = (nb < nb.rolling(200).mean()).shift(1).fillna(False).values
    if sweep_style:
        S50 = close.rolling(50).mean().values      # sweep recomputes on trimmed frame
    else:
        S50 = sma50.values                          # CLI uses load_frames sma (full history)
    dates = close.index
    days_idx = np.array([i for i, d in enumerate(dates) if START <= str(d.date()) <= END])
    return dict(close=close, trig=trig, weak=weak, S50=S50, days=days_idx, dates=dates,
                C=close.values, H=high.values, O=open_.values, ATH=athcp.values,
                RS=rs.values, TV=tv_prev.values)

A = build(wA, sweep_style=False)
B = build(wB, sweep_style=True)

print('cols equal order:', list(A['close'].columns) == list(B['close'].columns))
print('rows A/B:', len(A['dates']), len(B['dates']))
ta = A['trig'][A['days']]; tb = B['trig'][B['days']]
print('trig sums:', ta.sum(), tb.sum(), '| equal:', np.array_equal(ta, tb) if ta.shape == tb.shape else 'shape diff', ta.shape, tb.shape)
wa = A['weak'][A['days']]; wb = B['weak'][B['days']]
print('weak days:', wa.sum(), wb.sum())
sa = A['S50'][A['days']]; sb = B['S50'][B['days']]
if sa.shape == sb.shape:
    d = np.nanmax(np.abs(sa - sb))
    print('S50 max abs diff:', d, ' nan mismatch:', int((np.isnan(sa) != np.isnan(sb)).sum()))

for tag, X in (('CLI-style', A), ('sweep-style', B)):
    eq, trades, _ = br.simulate(1, 'random', X['days'], X['dates'], X['C'], X['H'],
                                X['O'], X['ATH'], X['S50'], X['RS'], X['TV'],
                                X['trig'], X['weak'], True, 0.0025,
                                stop=0.08, slots=8, size_pct=0.1875)
    st, _ = br.stats_from(eq, X['dates'][X['days']], trades, br.CAPITAL)
    print(f"{tag}: seed1 x={st['x']:.1f} cagr={st['cagr']:.1f} dd={st['dd']:.1f} trades={st['n']}")
