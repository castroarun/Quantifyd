# -*- coding: utf-8 -*-
"""research/172 Phase 2 - the stop/trail surface figure + the Phase 2 curve overlay."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/arun/quantifyd'); RES = ROOT / 'research/172_52wk_channel_n100/results'
PUB = ROOT / 'frontend/public'
BG, FG, MUT = '#0f1419', '#e6edf3', '#8b949e'

def style(a):
    a.set_facecolor(BG)
    for s in a.spines.values(): s.set_color('#30363d')
    a.tick_params(colors=MUT, labelsize=8); a.grid(True, color='#21262d', lw=0.5)

d = pd.read_csv(RES / 'stops.csv'); b = d[d.offset.isna() & (d.cost_bps == 15.0)]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=BG)
for a in ax: style(a)

# panel 1 - hard stop alone, both entries
for e, c in (('E189', '#4fd1c5'), ('E252', '#f0883e')):
    s = b[(b.group == 'hard_alone') & (b.entry == e) & b['stack'].str.startswith('HARD') & ~b['stack'].str.contains('ATR')]
    s = s.assign(x=s['stack'].str[4:].astype(int)).sort_values('x')
    ax[0].plot(s.x, s.calmar, 'o-', color=c, label=e)
ns = b[(b['stack'] == 'NOSTOP_CC252')]
ax[0].axhline(float(ns[ns.entry == 'E252'].calmar.iloc[0]), color=MUT, ls='--', lw=1.2,
              label='no stop at all (E252)')
ax[0].set_title('Initial hard stop from entry, alone\n(otherwise the literal 52-week-low exit)',
                color=FG, fontsize=10)
ax[0].set_xlabel('initial stop %', color=MUT); ax[0].set_ylabel('after-tax Calmar', color=FG)
ax[0].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=FG, fontsize=8)

# panel 2 - trailing stop alone
for e, c in (('E189', '#4fd1c5'), ('E252', '#f0883e')):
    s = b[(b.group == 'trail_alone') & (b.entry == e) & b['stack'].str.startswith('TRAIL')]
    s = s.assign(x=s['stack'].str[5:].astype(int)).sort_values('x')
    ax[1].plot(s.x, s.calmar, 'o-', color=c, label=e)
r = b[(b['stack'] == 'REF_ST_14_4')]
ax[1].axhline(float(r[r.entry == 'E252'].calmar.iloc[0]), color='#d2a8ff', ls='--', lw=1.2,
              label='SuperTrend(14,4) = 52W OPT')
ax[1].set_title('Percentage trailing stop from the highest\nclose since entry, alone', color=FG, fontsize=10)
ax[1].set_xlabel('trail %', color=MUT); ax[1].set_ylabel('after-tax Calmar', color=FG)
ax[1].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=FG, fontsize=8)

# panel 3 - growth of Rs 100
z = np.load(RES / 'p2_curves.npz', allow_pickle=True)
idx = pd.DatetimeIndex(pd.to_datetime(z['dates']))
for k, lab, c, lw in (('TRAIL20_T63_g0_E189', '52W STOPPED (20% trail + 63d time stop)', '#7ee787', 2.4),
                      ('52W_OPT', '52W OPT (SuperTrend 14,4)', '#4fd1c5', 1.6),
                      ('52W_Spec_A', '52W Spec A (literal)', '#f0883e', 1.4),
                      ('TRAIL10_T63_g0_BK20_E189', 'auto-ranked "winner" (10% trail + book kill) - a trap', '#ff7b72', 1.2),
                      ('NIFTYBEES', 'NIFTYBEES', MUT, 1.2)):
    if k not in z.files: continue
    s = pd.Series(z[k], index=idx)
    ax[2].plot(idx, 100 * s / s.iloc[0], color=c, lw=lw, label=lab)
ax[2].set_yscale('log'); ax[2].set_ylabel('Growth of Rs 100 (log)', color=FG)
ax[2].set_title('Phase 2 winner against Phase 1 and the trap', color=FG, fontsize=10)
ax[2].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=FG, fontsize=7.5, loc='upper left')

fig.suptitle('52W Phase 2 - stop-loss and trailing-stop stacks, after tax, 15 bps a side, '
             'Rs 1 crore, 20 slots. Both entries shown so every cell is paired.',
             color=FG, fontsize=11, y=1.0)
fig.tight_layout()
out = PUB / '52wk-channel-n100-research172-stops.png'
fig.savefig(out, dpi=115, facecolor=BG, bbox_inches='tight'); print('wrote', out)
