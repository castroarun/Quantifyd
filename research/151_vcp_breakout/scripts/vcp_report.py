"""P9 — report artifacts: the house YoY comparison table and the tearsheet chart.

House format (project CLAUDE.md, 2026-09-04): one column per system AND per blend, plus
benchmarks; every year cell = annual return with the intra-year max drawdown beneath it;
three best-of columns on the right (BEST CAGR / LEAST DD / BEST OVERALL) with benchmarks
excluded from the picks; a summary row with full-period CAGR / MaxDD / Calmar.
All figures after tax, net of 25 bps per side, medians across seeds/offsets.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt        # noqa: E402

STUDY = Path(__file__).resolve().parents[1]
RES = STUDY / 'results'
R146 = Path('/home/arun/quantifyd/research/146_complementary_third_sleeve/results')
DB = Path('/home/arun/quantifyd/backtest_data/market_data.db')
BENCH = {'NIFTY 50': 'NIFTY50', 'Midcap 150': 'NIFTYMIDCAP150',
         'Smallcap 250': 'NIFTYSMLCAP250'}


def series(sym):
    con = sqlite3.connect(str(DB))
    df = pd.read_sql("SELECT date, close FROM market_data_unified WHERE symbol=? AND "
                     "timeframe='day' AND close IS NOT NULL ORDER BY date", con,
                     params=(sym,), parse_dates=['date'])
    con.close()
    return df.set_index('date')['close'] if len(df) else None


def yearly_cells(nav):
    """(annual return %, intra-year max drawdown %) per calendar year."""
    out = {}
    ye = nav.resample('YE').last()
    prev = nav.iloc[0]
    for ts, v in ye.items():
        y = ts.year
        seg = nav[nav.index.year == y]
        dd = float((seg / seg.cummax() - 1).min()) * 100
        out[y] = (float(v / prev - 1) * 100, dd)
        prev = v
    return out


def full(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    d = float((nav / nav.cummax() - 1).min())
    return c * 100, d * 100, (c * 100) / abs(d * 100)


def median_nav(navs):
    """Median across paths of the normalised curve (a robustness-basis composite)."""
    df = pd.concat([n / n.iloc[0] for n in navs], axis=1).dropna()
    return df.median(axis=1)


def main():
    oa = pd.read_csv(R146 / 'oa_navs.csv', index_col=0, parse_dates=True)
    tn = {o: pd.read_csv(R146 / f'tn_nav_off{o}.csv', index_col=0,
                         parse_dates=True).iloc[:, 0].dropna() for o in (0, 4, 8)}
    vc = pd.read_csv(RES / 'vcp_equity_seeds.csv', index_col=0, parse_dates=True)
    idx = oa.index
    for s in list(tn.values()) + [vc]:
        idx = idx.intersection(s.index)
    idx = idx.sort_values()

    oa_n = [oa[c].loc[idx] for c in oa.columns]
    tn_n = [tn[o].loc[idx] for o in (0, 4, 8)]
    vc_n = [vc[c].loc[idx] for c in vc.columns]

    def blend(w_vcp):
        outs = []
        for i, o in enumerate(oa_n):
            for t in tn_n:
                leg = (1 - w_vcp) / 2
                mo = o.resample('ME').last().pct_change().fillna(0)
                mt = t.resample('ME').last().pct_change().fillna(0)
                r = leg * mo + leg * mt
                if w_vcp:
                    mv = vc_n[i % len(vc_n)].resample('ME').last().pct_change().fillna(0)
                    r = r + w_vcp * mv
                outs.append((1 + r).cumprod())
        return median_nav(outs)

    cols = {}
    cols['VCP (r/151)'] = median_nav(vc_n)
    cols['Open Alpha'] = median_nav(oa_n)
    cols['True North'] = median_nav(tn_n)
    cols['TN+OA 50-50'] = blend(0.0)
    cols['TN+OA+VCP 40/40/20'] = blend(0.20)
    for nm, sym in BENCH.items():
        s = series(sym)
        if s is not None:
            s = s.reindex(idx).ffill().dropna()
            if len(s) > 100:
                cols[nm] = s / s.iloc[0]
    systems = [k for k in cols if k not in BENCH]

    cells = {k: yearly_cells(v) for k, v in cols.items()}
    years = sorted({y for c in cells.values() for y in c})
    lines = []
    hdr = f"{'Year':6s}" + ''.join(f'{k[:18]:>20s}' for k in cols) + \
          f"{'BEST CAGR':>22s}{'LEAST DD':>22s}{'BEST OVERALL':>22s}"
    lines.append(hdr)
    for y in years:
        row = f'{y:<6d}'
        for k in cols:
            r, d = cells[k].get(y, (np.nan, np.nan))
            row += f'{r:>13.1f} ({d:>5.1f})' if r == r else f"{'-':>20s}"
        cand = {k: cells[k][y] for k in systems if y in cells[k]}
        if cand:
            bc = max(cand, key=lambda k: cand[k][0])
            ld = max(cand, key=lambda k: cand[k][1])
            bo = max(cand, key=lambda k: cand[k][0] + cand[k][1])
            row += f'{bc[:20]:>22s}{ld[:20]:>22s}{bo[:20]:>22s}'
        lines.append(row)
    srow = f"{'ALL':6s}"
    for k in cols:
        c, d, kk = full(cols[k])
        srow += f'{c:>8.1f}/{d:>6.1f}/{kk:>4.2f}'
    lines.append('-' * len(hdr))
    lines.append(srow + '   (CAGR / MaxDD / Calmar)')
    txt = '\n'.join(lines)
    (RES / 'vcp_yoy_table.txt').write_text(txt)
    print(txt)
    print(f'\nwindow {idx[0].date()}..{idx[-1].date()}; after tax, 25bps/side, '
          f'medians across {len(vc_n)} VCP seeds / {len(oa_n)} OA seeds / {len(tn_n)} TN offsets')

    # ---- chart: growth of Rs 100 (log) + drawdown panel
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[3, 1], sharex=True)
    style = {'VCP (r/151)': ('#f5a623', 2.4), 'Open Alpha': ('#4aa3ff', 1.8),
             'True North': ('#7ed321', 1.8), 'TN+OA 50-50': ('#ffffff', 1.8),
             'TN+OA+VCP 40/40/20': ('#ff6b6b', 1.8)}
    for k, v in cols.items():
        c, lw = style.get(k, ('#888888', 1.0))
        ax1.plot(v.index, v * 100, color=c, lw=lw, label=k,
                 ls='-' if k in style else '--', alpha=1.0 if k in style else 0.65)
    ax1.set_yscale('log')
    ax1.set_ylabel('Growth of Rs 100 (log)')
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.grid(alpha=0.25)
    ax1.set_title('research/151 VCP breakout vs the deployed book and Indian indices — '
                  'after tax, 25 bps/side, seed/offset medians')
    for k in ('VCP (r/151)', 'TN+OA 50-50', 'TN+OA+VCP 40/40/20'):
        v = cols[k]
        ax2.plot(v.index, (v / v.cummax() - 1) * 100, color=style[k][0], lw=1.4, label=k)
    ax2.set_ylabel('Drawdown %')
    ax2.grid(alpha=0.25)
    ax2.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / 'vcp_tearsheet.png', dpi=130, facecolor='#111318')
    print('wrote vcp_tearsheet.png')


if __name__ == '__main__':
    main()
