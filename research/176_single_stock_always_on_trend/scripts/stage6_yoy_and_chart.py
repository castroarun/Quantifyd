"""research/176 stage 6 — the house YoY comparison table + the study factsheet PNG."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import (connect, load_bars, all_directions, apply_policy,
                    supertrend_dir, IDLE_YIELD)

RES = os.path.join(HERE, '..', 'results')
ARUN3 = ['MARUTI', 'RELIANCE', 'HDFCBANK']
START, END = '2006-01-01', '2026-09-15'


def exposure_series(df, kind):
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    if kind == 'BH':
        return np.ones(len(df))
    if kind == 'EMA_9_21_LF':
        return apply_policy(all_directions(df)['EMA_9_21'], 'long_flat').astype(float)
    if kind == 'ST_7_3_LF':
        return apply_policy(all_directions(df)['ST_7_3.0'], 'long_flat').astype(float)
    if kind == 'ST_7_3_LS':
        return all_directions(df)['ST_7_3.0'].astype(float)
    if kind == 'MST_STACK_LF':
        m = supertrend_dir(h, l, c, 7, 5.0)
        ch = supertrend_dir(h, l, c, 7, 2.0)
        n = len(m)
        e = np.zeros(n)
        u = 0
        pm = 0
        pc = 0
        for i in range(n):
            if m[i] != pm:
                u = 0
                pm = m[i]
            if m[i] != 0 and ch[i] == m[i] and pc != m[i] and u < 5:
                u += 1
            pc = ch[i]
            e[i] = (m[i] if m[i] > 0 else 0) * u / 5
        return e
    raise ValueError(kind)


def book(symbols, kind, cost=20.0):
    con = connect()
    legs = {}
    for s in symbols:
        df = load_bars(con, s, 'day')
        df = df[(df.index >= START) & (df.index <= END)]
        o = df['open'].values.astype(float)
        ret = np.zeros(len(o))
        ret[:-1] = o[1:] / o[:-1] - 1.0
        e = exposure_series(df, kind)
        held = np.zeros(len(e))
        held[1:] = e[:-1]
        by = 1 / 252.0
        idle = (1 - np.abs(held)).clip(0) * ((1 + IDLE_YIELD) ** by - 1)
        chg = np.abs(np.diff(np.concatenate([[0.0], held])))
        legs[s] = pd.Series(held * ret + idle - chg * (cost / 2 / 10000.0), index=df.index)
    con.close()
    M = pd.DataFrame(legs).fillna(0.0)
    return M.mean(axis=1)


def index_series(sym='NIFTY50'):
    con = connect()
    df = load_bars(con, sym, 'day')
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df = df[(df.index >= START) & (df.index <= END)]
    r = df['close'].pct_change().fillna(0.0)
    return r


def yoy(r):
    out = {}
    for y, g in r.groupby(r.index.year):
        eq = (1 + g).cumprod()
        out[y] = (float(eq.iloc[-1] - 1), float((eq / eq.cummax() - 1).min()))
    return out


def summary(r):
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cg = eq.iloc[-1] ** (1 / yrs) - 1
    dd = float((eq / eq.cummax() - 1).min())
    return cg, dd, cg / abs(dd)


def main():
    cols = {
        'ARUN3 EMA(9,21) long/flat': book(ARUN3, 'EMA_9_21_LF'),
        'ARUN3 ST(7,3) long/flat': book(ARUN3, 'ST_7_3_LF'),
        'ARUN3 ST(7,3) long/short': book(ARUN3, 'ST_7_3_LS'),
        'ARUN3 MST stacked long/flat': book(ARUN3, 'MST_STACK_LF'),
        'ARUN3 buy & hold': book(ARUN3, 'BH'),
    }
    nif = index_series('NIFTY50')
    if not nif.empty:
        cols['NIFTY 50 (benchmark)'] = nif

    years = sorted(set().union(*[set(v.index.year) for v in cols.values()]))
    tabs = {k: yoy(v) for k, v in cols.items()}
    picks = [k for k in cols if 'benchmark' not in k]

    lines = []
    hdr = f"| Year | " + " | ".join(cols.keys()) + " | BEST CAGR | LEAST DD | BEST OVERALL |"
    sep = "|" + "---|" * (len(cols) + 4)
    lines += [hdr, sep]
    for y in years:
        cells = []
        for k in cols:
            v = tabs[k].get(y)
            cells.append(f"{v[0]:+.1%}<br><sub>({v[1]:.1%})</sub>" if v else "—")
        av = {k: tabs[k][y] for k in picks if y in tabs[k]}
        if av:
            bc = max(av, key=lambda k: av[k][0])
            ld = max(av, key=lambda k: av[k][1])
            bo = max(av, key=lambda k: av[k][0] + av[k][1])
        else:
            bc = ld = bo = '—'
        lines.append(f"| {y} | " + " | ".join(cells) + f" | {bc} | {ld} | {bo} |")
    sm = []
    for k, v in cols.items():
        cg, dd, cal = summary(v)
        sm.append(f"**{cg:+.2%}**<br><sub>DD {dd:.1%} · Calmar {cal:.2f}</sub>")
    lines.append(f"| **2006-2026** | " + " | ".join(sm) + " | | | |")
    txt = "\n".join(lines)
    open(os.path.join(RES, 'yoy_table.md'), 'w').write(txt)
    print(txt)

    print('\n\nSUMMARY')
    for k, v in cols.items():
        cg, dd, cal = summary(v)
        print(f'  {k:32s} CAGR {cg:+.2%}  MaxDD {dd:6.1%}  Calmar {cal:.3f}')

    # ---- factsheet
    fig = plt.figure(figsize=(15, 9), facecolor='#0d1117')
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1.2], hspace=0.45, wspace=0.22)
    for ax in []:
        pass
    ax1 = fig.add_subplot(gs[0, :])
    palette = ['#4c9aff', '#f2994a', '#eb5757', '#27ae60', '#c792ea', '#8a8f98']
    for i, (k, v) in enumerate(cols.items()):
        eq = (1 + v).cumprod()
        ax1.plot(eq.index, eq.values, lw=1.6, color=palette[i % len(palette)], label=k)
    ax1.set_yscale('log')
    ax1.set_title('research/176 — single-stock always-on trend on MARUTI / RELIANCE / HDFCBANK\n'
                  'equal-weight book, daily bars, next-open fills, 20 bps round trip, cash at 5.2%',
                  color='#e6edf3', fontsize=13, loc='left')
    ax1.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', ncol=3)
    ax2 = fig.add_subplot(gs[1, :])
    for i, (k, v) in enumerate(cols.items()):
        eq = (1 + v).cumprod()
        ax2.plot(eq.index, (eq / eq.cummax() - 1).values * 100, lw=1.2,
                 color=palette[i % len(palette)])
    ax2.set_ylabel('drawdown %', color='#e6edf3', fontsize=9)

    ax3 = fig.add_subplot(gs[2, 0])
    agg = pd.read_csv(os.path.join(RES, 'agg_day.csv'))
    a = agg[(agg.window == 'full') & (agg.fill == 'next_open') &
            (agg.policy == 'long_flat') & (agg.signal.str.startswith('ST_'))].copy()
    a['p'] = a.signal.str.split('_').str[1].astype(int)
    a['m'] = a.signal.str.split('_').str[2].astype(float)
    pv = a.pivot_table(index='p', columns='m', values='beat_rate20')
    im = ax3.imshow(pv.values, cmap='RdYlGn', vmin=0.2, vmax=0.6, aspect='auto')
    ax3.set_xticks(range(len(pv.columns)))
    ax3.set_xticklabels(pv.columns, color='#e6edf3', fontsize=8)
    ax3.set_yticks(range(len(pv.index)))
    ax3.set_yticklabels(pv.index, color='#e6edf3', fontsize=8)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax3.text(j, i, f'{pv.values[i,j]:.2f}', ha='center', va='center', fontsize=7)
    ax3.set_title('SuperTrend plateau map — share of 146 names where the rule\n'
                  'BEATS buy-and-hold (daily, long/flat). Gate was 0.55.',
                  color='#e6edf3', fontsize=9, loc='left')
    ax3.set_xlabel('ATR multiplier', color='#e6edf3', fontsize=8)
    ax3.set_ylabel('period', color='#e6edf3', fontsize=8)

    ax4 = fig.add_subplot(gs[2, 1])
    tfs = ['daily', '60-min', '30-min']
    lf = [0.432, 0.336, 0.295]
    ls = [0.062, 0.144, 0.137]
    x = np.arange(3)
    ax4.bar(x - 0.18, lf, 0.36, color='#4c9aff', label='long/flat')
    ax4.bar(x + 0.18, ls, 0.36, color='#eb5757', label='long/short')
    ax4.axhline(0.55, color='#f2c94c', ls='--', lw=1.4)
    ax4.text(2.35, 0.565, 'pre-registered gate 0.55', color='#f2c94c', fontsize=8, ha='right')
    ax4.set_xticks(x)
    ax4.set_xticklabels(tfs, color='#e6edf3', fontsize=9)
    ax4.set_ylim(0, 0.7)
    ax4.set_title('Best cell on each timeframe — share of names beating buy-and-hold',
                  color='#e6edf3', fontsize=9, loc='left')
    ax4.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        ax.grid(alpha=0.15, color='#30363d')
    out = '/home/arun/quantifyd/frontend/public/research176-alwayson-trend.png'
    fig.savefig(out, dpi=110, facecolor='#0d1117', bbox_inches='tight')
    print('\nwrote', out)


if __name__ == '__main__':
    main()
