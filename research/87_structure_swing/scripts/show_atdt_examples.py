#!/usr/bin/env python3
"""Extract + chart sample failed-ascending-triangle signals (research/87 1d)."""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from run_g1_daily_screen import confirmed_pivots, load_daily, load_universe

H = 15  # sessions held
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), '..', 'static', 'r87_atdt')
OUTDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../static/r87_atdt'))

def find_signals(df):
    """Re-run the asc-triangle-failure detector, returning rich context."""
    h, l, c = df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy()
    n = len(c)
    ph, pl, phi, pli = confirmed_pivots(h, l, 3)
    highs, lows = [], []
    out = []
    for t in range(n):
        if not np.isnan(ph[t]): highs.append((ph[t], phi[t]))
        if not np.isnan(pl[t]): lows.append((pl[t], pli[t]))
        if len(highs) < 2 or len(lows) < 2: continue
        (h1, hi1), (h2, hi2) = highs[-2:]
        (l1, li1), (l2, li2) = lows[-2:]
        span = max(hi2, li2) - min(hi1, li1)
        if span < 10 or span > 60 or t - max(hi2, li2) > 25: continue
        if abs(h1 - h2) / h1 <= 0.01 and l2 > l1 * 1.005:   # ascending triangle
            sup = l2
            if c[t] < sup:                                   # FAILURE break-down
                out.append(dict(t=t, r1=(hi1, h1), r2=(hi2, h2),
                                s1=(li1, l1), s2=(li2, l2), brk=c[t]))
    return out

uni = load_universe()
shown = 0
for sym in ['RELIANCE','TATASTEEL','SBIN','MARUTI','ICICIBANK','LT','INFY','HDFCBANK','AXISBANK','TATAMOTORS']:
    if shown >= 3: break
    df = load_daily(sym)
    if df is None: continue
    sigs = find_signals(df)
    # prefer examples 2010-2021 with clean spacing
    sigs = [s for s in sigs if 2010 <= df.index[s['t']].year <= 2021]
    if not sigs: continue
    s = sigs[len(sigs)//2]
    t = s['t']; dates = df.index
    o = df['open'].to_numpy()
    if t + 1 + H >= len(df): continue
    entry_d, entry = dates[t+1], o[t+1]
    exit_d, exitp = dates[t+1+H], o[t+1+H]
    stock_ret = (exitp/entry - 1) * 100
    print(f'=== {sym} ===')
    print(f'  resistance touch 1: {dates[s["r1"][0]].date()}  high {s["r1"][1]:.1f}')
    print(f'  resistance touch 2: {dates[s["r2"][0]].date()}  high {s["r2"][1]:.1f}  (flat within 1%)')
    print(f'  rising support lo1: {dates[s["s1"][0]].date()}  low  {s["s1"][1]:.1f}')
    print(f'  rising support lo2: {dates[s["s2"][0]].date()}  low  {s["s2"][1]:.1f}  (higher low)')
    print(f'  FAILURE bar:        {dates[t].date()}  close {s["brk"]:.1f} < support {s["s2"][1]:.1f}')
    print(f'  SHORT entry:        {entry_d.date()} at open {entry:.1f}')
    print(f'  exit (+{H} sessions): {exit_d.date()} at open {exitp:.1f}   stock move {stock_ret:+.1f}% (short P&L {-stock_ret:+.1f}% before hedge)')
    # chart
    w0, w1 = max(0, min(s['r1'][0], s['s1'][0]) - 15), min(len(df)-1, t + H + 5)
    seg = df.iloc[w0:w1+1]
    fig, ax = plt.subplots(figsize=(13, 6), facecolor='#111')
    ax.set_facecolor('#111')
    x = np.arange(len(seg))
    for i, (_, r) in enumerate(seg.iterrows()):
        col = '#2ecc71' if r['close'] >= r['open'] else '#e74c3c'
        ax.plot([i, i], [r['low'], r['high']], color=col, lw=0.7)
        ax.add_patch(plt.Rectangle((i-0.3, min(r['open'], r['close'])), 0.6,
                     abs(r['close']-r['open']) + 1e-9, color=col))
    def xi(gi): return gi - w0
    rl = (s['r1'][1] + s['r2'][1]) / 2
    ax.hlines(rl, xi(s['r1'][0]), xi(t)+2, color='#f1c40f', lw=1.5, linestyle='--', label='flat resistance')
    ax.plot([xi(s['s1'][0]), xi(t)], [s['s1'][1], s['s1'][1] + (s['s2'][1]-s['s1'][1])/(s['s2'][0]-s['s1'][0])*(t-s['s1'][0])],
            color='#3498db', lw=1.5, linestyle='--', label='rising support')
    for (pi, pv), m in [(s['r1'], 'v'), (s['r2'], 'v'), (s['s1'], '^'), (s['s2'], '^')]:
        ax.scatter([xi(pi)], [pv], marker=m, s=90, color='#f39c12', zorder=5)
    ax.scatter([xi(t)], [s['brk']], marker='x', s=140, color='#e74c3c', zorder=6, label='failure close')
    ax.scatter([xi(t+1)], [entry], marker='o', s=90, color='#e67e22', zorder=6, label=f'SHORT entry {entry:.0f}')
    ax.scatter([xi(t+1+H)], [exitp], marker='s', s=90, color='#9b59b6', zorder=6, label=f'exit {exitp:.0f}')
    step = max(1, len(seg)//10)
    ax.set_xticks(x[::step]); ax.set_xticklabels([d.strftime('%d-%b-%y') for d in seg.index[::step]], rotation=30, color='#aaa', fontsize=8)
    ax.tick_params(colors='#aaa'); [sp.set_color('#444') for sp in ax.spines.values()]
    ax.set_title(f'{sym} — FAILED ascending triangle (short) · signal {dates[t].date()} · stock {stock_ret:+.1f}% over {H} sessions', color='#eee')
    ax.legend(facecolor='#222', labelcolor='#eee', fontsize=9)
    fig.tight_layout()
    fp = os.path.join(OUTDIR, f'{sym}_{dates[t].date()}.png')
    fig.savefig(fp, dpi=110); plt.close(fig)
    print(f'  chart: /static/r87_atdt/{os.path.basename(fp)}')
    shown += 1
print('EXAMPLES DONE')
