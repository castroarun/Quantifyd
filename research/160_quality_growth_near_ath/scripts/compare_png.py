# -*- coding: utf-8 -*-
"""research/160 - the one figure that carries the study: every book on one log axis, with
the drawdown panel underneath and the same window for all of them.

    compare_png.py --out results/qg_compare.png NAME=path.csv [NAME=path.csv ...]

Each curve is the cross-sectional MEDIAN NAV across that book's paths (12 offsets, 30 seeds
or 360 pairs), re-based to 100 at the common first date. The three indices are drawn behind
in muted colours. Drawdowns are from the running peak of the full curve, never a window
slice (the r/154 retraction).
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'

BG, PANEL, INK, MUT = '#0e1116', '#161b22', '#e6edf3', '#8b949e'
PALETTE = ['#e3b341', '#f85149', '#3fb950', '#58a6ff', '#bc8cff', '#4fd1c5']
INDEXES = [('NIFTY50', 'NIFTY 50', '#6e7681'), ('NIFTYMIDCAP150', 'Midcap 150', '#484f58')]


def med(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime([str(x)[:10] for x in df.index])
    return df.sort_index().astype(float).median(axis=1)


def bench(sym, idx):
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    q = con.execute("select date, close from market_data_unified where symbol=? and "
                    "timeframe='day' order by date", (sym,)).fetchall()
    con.close()
    s = pd.Series({pd.Timestamp(str(d)[:10]): float(c) for d, c in q if c}).sort_index()
    return s.reindex(s.index.union(idx)).ffill().reindex(idx).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('curves', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='Quality-growth near the all-time high — research/160')
    a = ap.parse_args()

    series = {}
    for spec in a.curves:
        name, _, path = spec.partition('=')
        series[name] = med(path)
    idx = None
    for s in series.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    series = {k: (v.reindex(idx).ffill()) for k, v in series.items()}

    fig = plt.figure(figsize=(12.5, 8.0), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    for x in (ax, ax2):
        x.set_facecolor(PANEL)
        for sp in x.spines.values():
            sp.set_color('#30363d')
        x.tick_params(colors=MUT, labelsize=9)
        x.grid(True, color='#21262d', lw=0.6)

    for sym, lbl, col in INDEXES:
        b = bench(sym, idx)
        if len(b) < 200:
            continue
        ax.plot(b.index, b / b.iloc[0] * 100, color=col, lw=1.0, alpha=0.9, label=lbl)
        ax2.plot(b.index, (b / b.cummax() - 1) * 100, color=col, lw=0.8, alpha=0.7)
    for i, (name, s) in enumerate(series.items()):
        col = PALETTE[i % len(PALETTE)]
        g = s / s.iloc[0] * 100
        cagr = (s.iloc[-1] / s.iloc[0]) ** (365.25 / (idx[-1] - idx[0]).days) - 1
        dd = (s / s.cummax() - 1) * 100
        ax.plot(idx, g, color=col, lw=2.0,
                label='%s  %.1f%% / %.0f%%' % (name, cagr * 100, dd.min()))
        ax2.plot(idx, dd, color=col, lw=1.3)

    ax.set_yscale('log')
    ax.set_ylabel('growth of 100 (log)', color=INK, fontsize=10)
    ax.set_title(a.title, color=INK, fontsize=14, loc='left', pad=12)
    ax.text(0.0, 1.005, '', transform=ax.transAxes)
    ax.legend(facecolor=PANEL, edgecolor='#30363d', labelcolor=INK, fontsize=9,
              loc='upper left', ncol=2)
    ax2.set_ylabel('drawdown %', color=INK, fontsize=10)
    ax2.text(0.005, 0.06, 'after tax, 25 bps/side, median across paths; drawdown from the '
             'running peak of the full curve', transform=ax2.transAxes, color=MUT, fontsize=8)
    fig.savefig(a.out, dpi=140, facecolor=BG, bbox_inches='tight')
    print('wrote', a.out)


if __name__ == '__main__':
    sys.exit(main())
