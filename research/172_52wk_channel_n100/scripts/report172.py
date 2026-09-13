# -*- coding: utf-8 -*-
"""research/172 - report artifacts: factsheet PNG, equity/drawdown figure, YoY house table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/172_52wk_channel_n100/results'
PUB = ROOT / 'frontend/public'
sys.path.insert(0, str(ROOT / 'research/_utilities'))

BG = '#0f1419'
FG = '#e6edf3'
MUTED = '#8b949e'
COLS = {'52W Spec A': '#f0883e', '52W OPT': '#4fd1c5', '52W OPT (PIT-100)': '#d2a8ff',
        'EW B&H Nifty100': '#7ee787', 'NIFTYBEES': '#8b949e',
        'Random-entry null (median path)': '#ff7b72', 'Cash 5.2%': '#58a6ff'}
ORDER = ['52W Spec A', '52W OPT', '52W OPT (PIT-100)',
         'Random-entry null (median path)', 'EW B&H Nifty100', 'NIFTYBEES']


def load_curves():
    z = np.load(RES / 'curves.npz', allow_pickle=True)
    idx = pd.DatetimeIndex(pd.to_datetime(z['dates']))
    inv = {k.replace('_', ' ').replace('B&H', 'B&H'): k for k in z.files if k != 'dates'}
    out = {}
    for name in ORDER + ['Cash 5.2%']:
        key = name.replace(' ', '_').replace('%', 'p')
        if key in z.files:
            out[name] = pd.Series(z[key], index=idx)
    return idx, out


def fig_equity(idx, curves):
    fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1]), facecolor=BG)
    for a in ax:
        a.set_facecolor(BG)
        for sp in a.spines.values():
            sp.set_color('#30363d')
        a.tick_params(colors=MUTED, labelsize=9)
        a.grid(True, color='#21262d', lw=0.6)
    for name in ORDER:
        if name not in curves:
            continue
        s = curves[name]
        g = 100 * s / s.iloc[0]
        lw = 2.4 if name in ('52W OPT', '52W Spec A') else 1.3
        ax[0].plot(idx, g, color=COLS[name], lw=lw, label=name)
        dd = 100 * (s / s.cummax() - 1.0)
        ax[1].plot(idx, dd, color=COLS[name], lw=lw)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Growth of Rs 100 (log)', color=FG, fontsize=10)
    ax[0].set_title('52W - the 52-week channel book on Nifty 100 vs its own controls  '
                    '(after tax, 15 bps/side, Rs 1 cr, 20 slots, idle cash 5.2%)',
                    color=FG, fontsize=12, pad=12)
    ax[0].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=FG, fontsize=9,
                 loc='upper left')
    ax[1].set_ylabel('Drawdown %', color=FG, fontsize=10)
    ax[1].axhline(0, color=MUTED, lw=0.8)
    fig.tight_layout()
    out = PUB / '52wk-channel-n100-research172-curves.png'
    fig.savefig(out, dpi=115, facecolor=BG)
    plt.close(fig)
    print('wrote', out, flush=True)


def yoy_table(curves):
    names = [n for n in ORDER if n in curves]
    py = {}
    for n in names:
        s = curves[n]
        pk = s.cummax()
        dd = s / pk - 1.0
        row = {}
        for y in sorted(set(s.index.year)):
            j = np.flatnonzero(s.index.year == y)
            s0 = j[0] - 1 if j[0] > 0 else j[0]
            row[y] = (round(100 * (s.iloc[j[-1]] / s.iloc[s0] - 1), 1),
                      round(100 * float(dd.iloc[j].min()), 1))
        py[n] = row
    years = sorted(py[names[0]].keys())
    picks = [n for n in names if n not in ('NIFTYBEES', 'EW B&H Nifty100')]
    lines = []
    hdr = ['Year'] + names + ['BEST CAGR', 'LEAST DD', 'BEST OVERALL']
    lines.append('| ' + ' | '.join(hdr) + ' |')
    lines.append('|' + '---|' * len(hdr))
    for y in years:
        cells = []
        for n in names:
            r, d = py[n][y]
            cells.append('%+.1f<br><sub>(%.1f)</sub>' % (r, d))
        bc = max(picks, key=lambda n: py[n][y][0])
        ld = max(picks, key=lambda n: py[n][y][1])
        bo = max(picks, key=lambda n: py[n][y][0] + py[n][y][1])
        lines.append('| %d | %s | %s | %s | %s |' % (y, ' | '.join(cells), bc, ld, bo))
    summ = []
    for n in names:
        s = curves[n]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        cagr = 100 * ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1)
        mdd = 100 * float((s / s.cummax() - 1.0).min())
        summ.append('**%.2f%% / %.1f%%<br><sub>Calmar %.2f</sub>**'
                    % (cagr, mdd, cagr / abs(mdd)))
    lines.append('| **FULL 2006-2026** | ' + ' | '.join(summ) + ' |  |  |  |')
    txt = '\n'.join(lines)
    (RES / 'yoy_table.md').write_text(txt, encoding='utf-8')
    json.dump({n: {str(k): v for k, v in py[n].items()} for n in names},
              open(RES / 'yoy_data.json', 'w'), indent=1)
    print(txt, flush=True)
    return py, names


def factsheet(idx, curves):
    try:
        from tearsheet import generate_tearsheet
    except Exception as e:                                   # pragma: no cover
        print('tearsheet import failed: %s' % e, flush=True)
        return
    generate_tearsheet(curves['52W OPT'], curves['NIFTYBEES'],
                       '52W - 52-week channel, Nifty 100 (research/172)',
                       meta={'Spec': 'entry: close > 252d channel high; exit: SuperTrend(14,4); '
                                     'next-open fills',
                             'Book': 'Rs 1 cr, 20 slots @ 5%, NSE cash CNC',
                             'Costs': '15 bps/side, after tax (20/12.5%), idle cash 5.2%',
                             'Window': '2006-01-02 to 2026-09-11',
                             'Verdict': 'SIGNAL, not a STRATEGY - loses to a momentum-matched null'},
                       out_dir=str(PUB), rf=0.052,
                       extra_nav=curves['52W Spec A'], extra_label='52W Spec A (literal)')
    for p in sorted(PUB.glob('*.png')):
        print('  png:', p.name, p.stat().st_size // 1024, 'KB', flush=True)


def main():
    idx, curves = load_curves()
    fig_equity(idx, curves)
    yoy_table(curves)
    factsheet(idx, curves)


if __name__ == '__main__':
    main()
