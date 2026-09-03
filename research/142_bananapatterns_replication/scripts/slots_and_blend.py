"""(1) Slots/size sweep: does holding MORE stocks shrink the seed spread (path
dependence), and what does it cost in CAGR? (2) 50-50 monthly-rebalanced blends
with Momentum r/75 for each gate (DD10 / SMA200 / no gate): summary + YoY medians.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
import bluesky_replay as br

print('loading frames (2004-06-01 ->) ...', flush=True)
w = br.load_frames('2004-06-01', trail_sma=20)
close, high, open_, athcp, sma, tv20 = (w[k] for k in
    ('close', 'high', 'open', 'athcp', 'sma50', 'tv20'))
etf = [c for c in close.columns if br.ETF_RE.search(c)]
tv_prev = tv20.shift(1)
prev_close = close.shift(1)
elig = tv_prev >= br.TV_FLOOR
elig[etf] = False
score = 2*(close/close.shift(63)-1) + (close/close.shift(126)-1) \
    + (close/close.shift(189)-1) + (close/close.shift(252)-1)
rs = (score.where(elig).rank(axis=1, pct=True)*100).shift(1)
setup = (prev_close < athcp) & (prev_close >= 0.8*athcp) & elig & (rs >= 70.0)
trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
dates = close.index
C, H, O, ATH, S = close.values, high.values, open_.values, athcp.values, sma.values
RSv, TVv = rs.values, tv_prev.values
days = np.array([i for i, d in enumerate(dates) if str(d.date()) >= '2006-01-01'])
nb = close['NIFTYBEES'].dropna()


def align(raw):
    return raw.shift(1).reindex(dates).ffill().fillna(False).astype(bool).values


GATES = {'DD10': align(nb < 0.9 * nb.rolling(252).max()),
         'SMA200': align(nb < nb.rolling(200).mean()),
         'No gate': np.zeros(len(dates), dtype=bool)}


def ensemble(weak, slots, size_pct):
    eqs = []
    for seed in range(1, 11):
        eq, _, _ = br.simulate(seed, 'random', days, dates, C, H, O, ATH, S,
                               RSv, TVv, trig, weak, True, 0.0025, stop=0.08,
                               slots=slots, size_pct=size_pct)
        eqs.append(np.asarray(eq, dtype=float))
    return eqs


def stats(eqs):
    terms = [e[-1]/e[0] for e in eqs]
    dds = [float((e/np.maximum.accumulate(e)-1).min()*100) for e in eqs]
    n = len(days)/247.0
    cagrs = sorted(t**(1/n)-1 for t in terms)
    return (float(np.median(terms)), min(terms), max(terms),
            float(np.median(dds)), 100*np.median(cagrs),
            100*cagrs[0], 100*cagrs[-1])

print('\n== (1) SLOTS / SIZE SWEEP (gate DD10, 2006->now, pre-tax) ==', flush=True)
print(f"{'config':22s} {'med x':>8s} {'min..max x':>16s} {'med dd':>8s} "
      f"{'CAGR med':>9s} {'CAGR range':>16s} {'spread':>7s}")
for slots, size in [(8, 0.1875), (8, 0.125), (12, 0.0833), (16, 0.0625), (20, 0.05)]:
    t0 = time.time()
    med, mn, mx, dd, cm, c0, c1 = stats(ensemble(GATES['DD10'], slots, size))
    print(f"slots={slots:2d} size={size*100:5.2f}% {med:8.1f} {mn:7.0f}..{mx:<7.0f} "
          f"{dd:7.1f}% {cm:8.1f}% {c0:6.1f}..{c1:<6.1f}% {mx/mn:6.1f}x  ({time.time()-t0:.0f}s)",
          flush=True)

print('\n== (2) 50-50 BLENDS with Momentum r/75 ==', flush=True)
mom = pd.read_csv(ROOT / 'research' / '75_nifty250_momentum_top15' / 'results' / 'nav_armed_spec.csv',
                  index_col=0, parse_dates=True)['nav']
mm_full = mom.resample('ME').last().pct_change()

for gname, weak in GATES.items():
    eqs = ensemble(weak, 8, 0.1875)
    navs = [pd.Series(e, index=dates[days]) for e in eqs]
    idx = navs[0].index.intersection(mom.index)
    cagr_l, dd_l, yoy_l = [], [], []
    for nav in navs:
        b_m = nav.loc[idx].resample('ME').last().pct_change().fillna(0)
        m_m = mom.loc[idx].resample('ME').last().pct_change().fillna(0)
        blend = (1 + 0.5*b_m + 0.5*m_m).cumprod()
        yrs = (blend.index[-1] - blend.index[0]).days / 365.25
        cagr_l.append((blend.iloc[-1]) ** (1/yrs) - 1)
        dd_l.append(float((blend/blend.cummax()-1).min()*100))
        yoy_l.append(blend.resample('YE').last().pct_change().dropna()*100)
    yoy = pd.concat(yoy_l, axis=1).median(axis=1)
    print(f'\nBLEND 50-50 {gname} x Momentum ({idx[0].date()}->{idx[-1].date()}):')
    print(f'  CAGR median {np.median(cagr_l)*100:.1f}% '
          f'[{min(cagr_l)*100:.1f}..{max(cagr_l)*100:.1f}]  '
          f'maxDD median {np.median(dd_l):.1f}% [{min(dd_l):.1f}..{max(dd_l):.1f}]')
    print('  YoY median-across-seeds: ' +
          ' '.join(f'{d.year}:{v:+.0f}%' for d, v in yoy.items()), flush=True)
print('DONE', flush=True)
