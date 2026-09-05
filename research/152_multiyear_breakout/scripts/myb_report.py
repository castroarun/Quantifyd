"""research/152 — report artifacts: the house-format YoY comparison table (HTML + CSV)
and the growth-of-100 chart vs NIFTY 50 / Midcap 150 / Smallcap 250 with a drawdown panel.

All figures after tax, net of costs, medians across seeds (MYB, OA) / offsets (TN).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt            # noqa: E402

RES = Path(__file__).resolve().parents[1] / 'results'
R146 = Path('/home/arun/quantifyd/research/146_complementary_third_sleeve/results')
DB = '/home/arun/quantifyd/backtest_data/market_data.db'
IDX = {'NIFTY 50': 'NIFTY50', 'Midcap 150': 'NIFTYMIDCAP150', 'Smallcap 250': 'NIFTYSMLCAP250'}


def idx_series(sym):
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT substr(date,1,10) d, close FROM market_data_unified WHERE "
                     "symbol=? AND timeframe='day' AND close IS NOT NULL ORDER BY d",
                     con, params=(sym,))
    con.close()
    df['d'] = pd.to_datetime(df['d'])
    return df.set_index('d')['close']


def yearly(nav, capital=None):
    """(annual return %, intra-year max drawdown %) per calendar year."""
    out = {}
    for y, seg in nav.groupby(nav.index.year):
        prev = nav[nav.index.year < y]
        start = prev.iloc[-1] if len(prev) else (capital if capital else seg.iloc[0])
        run = pd.concat([pd.Series([start], index=[seg.index[0]]), seg])
        out[int(y)] = (float(seg.iloc[-1] / start - 1) * 100,
                       float((run / run.cummax() - 1).min() * 100))
    return out


def summary(nav):
    y = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / y) - 1
    d = float((nav / nav.cummax() - 1).min())
    return c * 100, d * 100, (c / abs(d) if d < 0 else np.nan)


def median_of(navs):
    """Median column-wise across an ensemble of NAV series on their common index."""
    df = pd.concat(navs, axis=1).dropna()
    return df.median(axis=1)


def blend_monthly(legs, weights):
    idx = legs[0].index
    for l_ in legs[1:]:
        idx = idx.intersection(l_.index)
    m = [l_.loc[idx].resample('ME').last().pct_change().fillna(0) for l_ in legs]
    r = sum(w * x for w, x in zip(weights, m))
    return (1 + r).cumprod()


def main():
    myb = pd.read_csv(RES / 'myb_equity_seeds.csv', index_col=0, parse_dates=True)
    oa = pd.read_csv(R146 / 'oa_navs.csv', index_col=0, parse_dates=True)
    tns = [pd.read_csv(R146 / f'tn_nav_off{o}.csv', index_col=0, parse_dates=True).iloc[:, 0]
           for o in (0, 4, 8) if (R146 / f'tn_nav_off{o}.csv').exists()]
    w3 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20

    myb_m = myb.median(axis=1)
    oa_m = oa.median(axis=1)
    tn_m = median_of(tns)
    idxs = {label: idx_series(sym) for label, sym in IDX.items()}
    # EVERY column on the SAME window: the ragged first pass compared a 2006-start
    # TN+OA baseline against 2010/2011-start candidates (2008 in one column only).
    start = max([myb_m.index[0], oa_m.index[0], tn_m.index[0]]
                + [v.index[0] for v in idxs.values()])
    print('common window start:', start.date())
    myb_m, oa_m, tn_m = (x[x.index >= start] for x in (myb_m, oa_m, tn_m))
    cols = {}
    cols['MYB (this study)'] = myb_m
    cols['Open Alpha'] = oa_m
    cols['True North'] = tn_m
    cols['TN+OA 50-50'] = blend_monthly([oa_m, tn_m], [0.5, 0.5])
    cols[f'TN+OA+MYB {int((1-w3)*50)}/{int((1-w3)*50)}/{int(w3*100)}'] = blend_monthly(
        [oa_m, tn_m, myb_m], [(1 - w3) / 2, (1 - w3) / 2, w3])
    for label, s in idxs.items():
        cols[label] = s[s.index >= start]

    strategies = [c for c in cols if c not in IDX]
    yr_tables = {k: yearly(v) for k, v in cols.items()}
    years = sorted(set().union(*[set(v) for v in yr_tables.values()]))

    rows = []
    for y in years:
        row = {'year': y}
        for k in cols:
            if y in yr_tables[k]:
                r_, d_ = yr_tables[k][y]
                row[k] = f'{r_:+.1f}'
                row[k + '__dd'] = f'{d_:.1f}'
        cand = {k: yr_tables[k][y] for k in strategies if y in yr_tables[k]}
        if cand:
            row['BEST CAGR'] = max(cand, key=lambda k: cand[k][0])
            row['LEAST DD'] = max(cand, key=lambda k: cand[k][1])
            row['BEST OVERALL'] = max(cand, key=lambda k: cand[k][0] + cand[k][1])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RES / 'yoy_table.csv', index=False)

    # ── HTML (house format: return with intra-year DD in small muted type beneath) ──
    head = ''.join(f'<th>{k}</th>' for k in cols) + \
        '<th>BEST CAGR</th><th>LEAST DD</th><th>BEST OVERALL</th>'
    body = ''
    for r in rows:
        cells = ''
        for k in cols:
            if k in r:
                cells += (f'<td><div class="ret">{r[k]}%</div>'
                          f'<div class="dd">({r[k+"__dd"]}%)</div></td>')
            else:
                cells += '<td class="na">—</td>'
        cells += (f'<td class="pick">{r.get("BEST CAGR","")}</td>'
                  f'<td class="pick">{r.get("LEAST DD","")}</td>'
                  f'<td class="pick">{r.get("BEST OVERALL","")}</td>')
        body += f'<tr><th>{r["year"]}</th>{cells}</tr>'
    srow = ''
    for k in cols:
        c_, d_, cal = summary(cols[k])
        srow += (f'<td><div class="ret">{c_:.1f}%</div>'
                 f'<div class="dd">({d_:.1f}%) Cal {cal:.2f}</div></td>')
    srow += '<td colspan="3" class="pick">full period, after tax, medians</td>'
    html = f"""<!doctype html><meta charset="utf-8"><style>
body{{background:#0e1116;color:#e6edf3;font:13px/1.4 -apple-system,Segoe UI,sans-serif;padding:18px}}
table{{border-collapse:collapse}} th,td{{border:1px solid #263041;padding:5px 8px;text-align:center}}
th{{background:#161b22;font-weight:600}} .ret{{font-weight:600}}
.dd{{font-size:10px;color:#8b949e}} .pick{{font-size:11px;color:#7ee787}} .na{{color:#484f58}}
caption{{text-align:left;padding-bottom:8px;color:#8b949e}}</style>
<table><caption>research/152 Multi-Year Breakout — YoY returns with intra-year max drawdown
(after tax, net 25 bps/side, idle cash 5%; MYB/OA = seed medians, TN = offset median;
benchmarks excluded from the best-of picks). Window from {start.date()}.</caption>
<tr><th>Year</th>{head}</tr>{body}
<tr><th>CAGR / MaxDD</th>{srow}</tr></table>"""
    (RES / 'yoy_table.html').write_text(html, encoding='utf-8')

    # ── growth of 100 (log) + drawdown panel ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                   gridspec_kw=dict(height_ratios=[3, 1]))
    for k, v in cols.items():
        v = v[v.index >= start]
        g = 100 * v / v.iloc[0]
        st = '--' if k in IDX else '-'
        lw = 1.3 if k in IDX else 2.0
        ax1.plot(g.index, g.values, st, lw=lw, label=f'{k} ({summary(v)[0]:.1f}%)')
        if k not in IDX:
            ax2.plot(v.index, (v / v.cummax() - 1) * 100, lw=1.2, label=k)
    ax1.set_yscale('log'); ax1.set_ylabel('Growth of ₹100 (log)')
    ax1.legend(fontsize=8, loc='upper left'); ax1.grid(alpha=.25)
    ax1.set_title('research/152 Multi-Year Breakout vs deployed books and indices '
                  '(after tax, 25 bps/side)')
    ax2.set_ylabel('Drawdown %'); ax2.grid(alpha=.25); ax2.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RES / 'myb_curve_vs_indices.png', dpi=110)
    print('report written:', RES / 'yoy_table.html', RES / 'myb_curve_vs_indices.png')


if __name__ == '__main__':
    main()
