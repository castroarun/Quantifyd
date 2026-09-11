# -*- coding: utf-8 -*-
"""research/160 — tearsheet + curves wrapper.

Two artefacts from one equity CSV (the --dump-equity output: index = date, one column per
seed/offset path):

  1. the house client factsheet via research/_utilities/tearsheet.py — KPI strip, equity vs
     benchmark (log), underwater drawdown, yearly bars, monthly heatmap;
  2. `curves_<name>.png` — log growth of 100 against NIFTY 50 / MIDCAP 150 / SMALLCAP 250
     with a drawdown panel underneath, which is the figure the roster report format wants.

Which path is drawn: by default the MEDIAN path by terminal value (labelled as such, never
presented as the expectation). `--median-curve` instead draws the cross-sectional median
NAV across paths; `--column` picks one explicitly. The ensemble band (min..max) is shaded
behind it so a single path is never mistaken for the system.

Usage:
  tearsheet_wrap.py --curve results/A_equity.csv --name "QG k90 N15" --out-dir results
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
sys.path.insert(0, str(ROOT / 'research' / '_utilities'))

BG, PANEL, INK, MUT = '#0e1116', '#161b22', '#e6edf3', '#8b949e'
GOLD, GREEN, RED, BLUE, TEAL = '#e3b341', '#3fb950', '#f85149', '#58a6ff', '#4fd1c5'
INDEXES = [('NIFTY50', 'NIFTY 50', MUT), ('NIFTYMIDCAP150', 'Midcap 150', BLUE),
           ('NIFTYSMLCAP250', 'Smallcap 250', TEAL)]


def load_curves(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime([str(x)[:10] for x in df.index])
    return df.sort_index().astype(float)


def bench(sym, idx):
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    q = con.execute("select date, close from market_data_unified where symbol=? and "
                    "timeframe='day' order by date", (sym,)).fetchall()
    con.close()
    if not q:
        return None
    s = pd.Series({pd.Timestamp(str(d)[:10]): float(c) for d, c in q if c}).sort_index()
    s = s.reindex(s.index.union(idx)).ffill().reindex(idx)
    return s.dropna()


def curves_png(df, nav, name, out):
    idx = df.index
    fig = plt.figure(figsize=(12, 7.2), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1], hspace=0.14)
    ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax)
    for a in (ax, ax2):
        a.set_facecolor(PANEL)
        for sp in a.spines.values():
            sp.set_color('#30363d')
        a.tick_params(colors=MUT, labelsize=9)
        a.grid(True, color='#21262d', lw=0.6)

    lo = (df.min(axis=1) / df.iloc[0].min()) * 100
    hi = (df.max(axis=1) / df.iloc[0].max()) * 100
    if df.shape[1] > 1:
        ax.fill_between(idx, lo, hi, color=GOLD, alpha=0.15, lw=0,
                        label='ensemble min..max (%d paths)' % df.shape[1])
    g = nav / nav.iloc[0] * 100
    ax.plot(idx, g, color=GOLD, lw=1.8, label=name)
    dds = {name: (nav / nav.cummax() - 1) * 100}
    for sym, lbl, col in INDEXES:
        b = bench(sym, idx)
        if b is None or len(b) < 200:
            continue
        ax.plot(b.index, b / b.iloc[0] * 100, color=col, lw=1.1, alpha=0.85, label=lbl)
        dds[lbl] = (b / b.cummax() - 1) * 100
    ax.set_yscale('log')
    ax.set_ylabel('growth of 100 (log)', color=INK, fontsize=10)
    ax.set_title('%s — log growth vs the indices, with drawdown' % name, color=INK,
                 fontsize=13, loc='left', pad=10)
    ax.legend(facecolor=PANEL, edgecolor='#30363d', labelcolor=INK, fontsize=9, loc='upper left')

    for lbl, d in dds.items():
        col = GOLD if lbl == name else dict((l, c) for _, l, c in INDEXES).get(lbl, MUT)
        ax2.plot(d.index, d, color=col, lw=1.4 if lbl == name else 0.9,
                 alpha=1.0 if lbl == name else 0.7)
    ax2.fill_between(dds[name].index, dds[name], 0, color=RED, alpha=0.18, lw=0)
    ax2.set_ylabel('drawdown %', color=INK, fontsize=10)
    ax2.text(0.005, 0.06, 'drawdown measured from the running peak of the full curve',
             transform=ax2.transAxes, color=MUT, fontsize=8)
    fig.savefig(out, dpi=140, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--curve', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out-dir', default=str(Path(__file__).resolve().parents[1] / 'results'))
    ap.add_argument('--column', default=None)
    ap.add_argument('--median-curve', action='store_true')
    ap.add_argument('--benchmark', default='NIFTY50')
    ap.add_argument('--rf', type=float, default=0.065)
    ap.add_argument('--no-tearsheet', action='store_true')
    a = ap.parse_args()

    df = load_curves(a.curve)
    if a.column:
        nav, how = df[a.column], 'path %s' % a.column
    elif a.median_curve or df.shape[1] == 1:
        nav = df.median(axis=1) if df.shape[1] > 1 else df.iloc[:, 0]
        how = 'cross-sectional median NAV' if df.shape[1] > 1 else 'single path'
    else:
        fin = df.iloc[-1]
        col = fin.sort_values().index[len(fin) // 2]
        nav, how = df[col], 'median path by terminal value (%s of %d)' % (col, df.shape[1])
    print('%s: drawing %s' % (a.name, how))

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    slug = a.name.lower().replace(' ', '_').replace('/', '_')
    curves_png(df, nav, a.name, out_dir / ('curves_%s.png' % slug))

    if not a.no_tearsheet:
        try:
            from tearsheet import generate_tearsheet
        except Exception as e:                                    # pragma: no cover
            print('tearsheet unavailable (%s) — the curves PNG is still written' % e)
            return
        b = bench(a.benchmark, nav.index)
        extra = bench('NIFTYMIDCAP150', nav.index)
        generate_tearsheet(nav, b, a.name,
                           meta={'window': '%s to %s' % (nav.index[0].date(), nav.index[-1].date()),
                                 'paths': df.shape[1], 'curve shown': how,
                                 'basis': 'after tax, net of costs, idle cash 5% p.a.'},
                           out_dir=str(out_dir), rf=a.rf,
                           extra_nav=extra, extra_label='Midcap 150')
        # the shared utility writes fixed filenames; give each system its own so a second
        # call cannot silently overwrite the first system's factsheet
        for ext in ('png', 'html'):
            src = out_dir / ('tearsheet.%s' % ext)
            if src.exists():
                dst = out_dir / ('tearsheet_%s.%s' % (slug, ext))
                src.replace(dst)
                print('wrote', dst)


if __name__ == '__main__':
    main()
