"""(a) DD10 gate across all 5 index series on 2012->now (fair window - all series
exist); (b) comparison chart: DD10 vs SMA200 vs gate-OFF vs NIFTY 50 benchmark
(NIFTYBEES, the Nifty-50 ETF, is the benchmark series - full 2006+ history),
equity (log) + drawdown panels, median-seed paths.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def align(raw):
    return raw.shift(1).reindex(dates).ffill().fillna(False).astype(bool).values


def dd10(sym):
    s = close[sym].dropna()
    return align(s < 0.9 * s.rolling(252).max())


def run(days, weak, seeds=10):
    out = []
    for seed in range(1, seeds+1):
        eq, _, _ = br.simulate(seed, 'random', days, dates, C, H, O, ATH, S,
                               RSv, TVv, trig, weak, True, 0.0025, stop=0.08, slots=8)
        out.append(np.asarray(eq, dtype=float))
    return out

# ── (a) DD10 across series, 2012->now ──
days12 = np.array([i for i, d in enumerate(dates) if str(d.date()) >= '2012-01-01'])
print('\nDD10 gate by series (2012->now, all series live):', flush=True)
for sym in ['NIFTYBEES', 'NIFTY50', 'NIFTY500', 'NIFTYMIDCAP150', 'NIFTYSMLCAP250', None]:
    wk = dd10(sym) if sym else np.zeros(len(dates), dtype=bool)
    eqs = run(days12, wk)
    terms = [e[-1]/e[0] for e in eqs]
    dds = [float((e/np.maximum.accumulate(e)-1).min()*100) for e in eqs]
    name = f'DD10 on {sym}' if sym else 'gate OFF'
    print(f'  {name:24s} med x{np.median(terms):7.1f} [{min(terms):.0f}..{max(terms):.0f}] '
          f'dd {np.median(dds):.1f}% blocked {100*wk[days12].mean():.1f}%', flush=True)

# ── (b) comparison plot, 2006->now ──
days06 = np.array([i for i, d in enumerate(dates) if str(d.date()) >= '2006-01-01'])
nb = close['NIFTYBEES'].dropna()
GATES = {
    'DD10 gate (recommended)': dd10('NIFTYBEES'),
    '200-SMA gate (old spec)': align(nb < nb.rolling(200).mean()),
    'No gate': np.zeros(len(dates), dtype=bool),
}
curves = {}
for name, wk in GATES.items():
    eqs = run(days06, wk)
    terms = [e[-1] for e in eqs]
    med_i = int(np.argmin(np.abs(np.array(terms) - np.median(terms))))
    curves[name] = pd.Series(eqs[med_i] / eqs[med_i][0] * 100, index=dates[days06])
    print(f'plot: {name} median-seed terminal x{terms[med_i]/eqs[med_i][0]:.0f}', flush=True)
bench = nb.reindex(dates[days06]).ffill()
curves['NIFTY 50 (NIFTYBEES)'] = bench / bench.iloc[0] * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True,
                               gridspec_kw={'height_ratios': [2.4, 1]})
fig.patch.set_facecolor('#101318')
COLORS = {'DD10 gate (recommended)': '#38d996', '200-SMA gate (old spec)': '#e05555',
          'No gate': '#6ea8fe', 'NIFTY 50 (NIFTYBEES)': '#9aa0a6'}
for ax in (ax1, ax2):
    ax.set_facecolor('#101318')
    ax.grid(alpha=0.18, color='#666')
    for sp in ax.spines.values():
        sp.set_color('#444')
    ax.tick_params(colors='#c8ccd2')
for name, ser in curves.items():
    lw = 2.2 if 'DD10' in name else 1.5
    ax1.plot(ser.index, ser.values, label=f'{name}  (x{ser.iloc[-1]/100:,.0f})',
             color=COLORS[name], lw=lw)
    ddc = ser / ser.cummax() - 1
    ax2.plot(ddc.index, ddc.values * 100, color=COLORS[name], lw=1.2)
ax1.set_yscale('log')
ax1.set_title('Open Alpha gate comparison - Rs 100 grows to... (2006->2026, pre-tax, '
              'median seed of 10; adopted trail-20 spec)', color='#e8e8e8', fontsize=12)
ax1.legend(loc='upper left', facecolor='#181c22', edgecolor='#444', labelcolor='#e8e8e8')
ax2.set_ylabel('drawdown %', color='#c8ccd2')
ax2.set_title('Drawdown', color='#e8e8e8', fontsize=10)
plt.tight_layout()
out = STUDY / 'results' / 'gate_comparison.png'
plt.savefig(out, dpi=110, facecolor=fig.get_facecolor())
print('chart saved:', out, flush=True)

# ── YoY returns table (median-seed paths + benchmark) ──
print('\nYoY returns % (median-seed path per gate):')
yoy = pd.DataFrame({n: s.resample('YE').last().pct_change().dropna() * 100
                    for n, s in curves.items()})
first = pd.Series({n: (s.resample('YE').last().iloc[0] / 100 - 1) * 100
                   for n, s in curves.items()}, name='2006')
yoy.index = yoy.index.year
tbl = pd.concat([first.to_frame().T, yoy])
print(tbl.round(1).to_string())
tbl.round(2).to_csv(STUDY / 'results' / 'gate_yoy.csv')
print('\nsaved results/gate_yoy.csv', flush=True)
