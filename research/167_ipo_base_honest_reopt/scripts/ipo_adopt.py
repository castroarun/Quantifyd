# -*- coding: utf-8 -*-
"""research/163 stage 9 - the full adoption battery on the two finalists + the incumbent.

Finalists, both on the LIVE entry mechanic (next-day buy-stop at the pivot, filled
max(pivot, open)) and the NIFTYBEES<SMA150 gate:
    A  trail SMA-50, stop 10%, target +25%     <- best Calmar, the recommendation
    B  trail SMA-50, stop 15%, target +25%     <- best CAGR
Comparator: the r/153 adopted spec, ungated, as it stands today.

Produces: cash-yield arms, invested fraction, median AND worst-seed drawdown, cost ladder
25/40/60 bps, outlier dependence, capacity, tradeability gate, per-window rows (2008 / 2020
crash, 2018 / 2022H1 grind) measured from the FULL curve's running peak (r/154), per-year
table, weekly correlations, and the curves for the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/167_ipo_base_honest_reopt/scripts'))
import ipo_honest as ih                                      # noqa: E402

RES = ih.RES
R159 = ROOT / 'research/159_oa_honest_reoptimization/results/full_period_after_tax.csv'

SPECS = [('incumbent_r153_ungated', {**ih.INCUMBENT}, None),
         ('A_trail50_sl10_tp25_sma150',
          {**ih.INCUMBENT, 'trail': 50, 'stop': 0.10, 'target': 0.25}, 150),
         ('B_trail50_sl15_tp25_sma150',
          {**ih.INCUMBENT, 'trail': 50, 'stop': 0.15, 'target': 0.25}, 150)]

WINDOWS = {'2008 crash': ('2008-01-01', '2008-12-31'),
           '2020 crash': ('2020-01-01', '2020-06-30'),
           '2018 grind': ('2018-01-01', '2018-12-31'),
           '2022H1 grind': ('2022-01-01', '2022-06-30'),
           '2025 drawdown': ('2025-01-01', '2025-12-31')}


def win_stats(nav, a, b):
    """Return over the window, and the drawdown measured from the FULL curve's running peak."""
    peak = nav.cummax()
    dd = nav / peak - 1.0
    m = (nav.index >= a) & (nav.index <= b)
    if not m.any():
        return None, None
    seg = nav[m]
    prev = nav[nav.index < a]
    start = float(prev.iloc[-1]) if len(prev) else float(seg.iloc[0])
    return (round(100 * (float(seg.iloc[-1]) / start - 1), 2),
            round(100 * float(dd[m].min()), 2))


def main():
    ctx, ir = ih.load_ctx(clean=True)
    nb = ctx.close.get('NIFTYBEES').dropna()
    rows, curves, peryear, perwin = [], {}, {}, {}
    for name, base, gn in SPECS:
        cfg = dict(base)
        if gn:
            w = (nb < nb.rolling(gn).mean()).shift(1)
            cfg['gate'] = 'custom'
            cfg['weak_series'] = (w.reindex(ctx.dates).ffill().fillna(False).to_numpy(bool))
        else:
            cfg['gate'] = False
        setup, piv, lo0 = ih.build_setup(ctx, cfg)
        trig, lvl, lo, fc = ih.apply_mechanic(ctx, setup, piv, lo0, 'nextday_pivot')
        o, kept = ih.run_cell(ctx, ir, cfg, trig=trig, level=lvl, lo=lo, keep=True,
                              windows=('w2', 'wa', 'wb'), fill_close=fc)
        o0, _ = ih.run_cell(ctx, ir, cfg, trig=trig, level=lvl, lo=lo, windows=('w2',),
                            fill_close=fc, cash_yield=0.0)
        st = kept['stats']
        r = dict(spec=name, gate=gn or 'none', trail=cfg['trail'], stop=cfg['stop'],
                 target=cfg['target'],
                 cagr=o['w2_cagr'], cagr_worst=o['w2_cagr_lo'], cagr_best=o['w2_cagr_hi'],
                 cagr_no_sweep=o0['w2_cagr'],
                 sweep_pp=round(o['w2_cagr'] - o0['w2_cagr'], 2),
                 sweep_share=round(100 * (o['w2_cagr'] - o0['w2_cagr']) / o['w2_cagr'], 1),
                 invested=o['w2_inv'],
                 dd=o['w2_dd'], dd_worst=o['w2_dd_worst'], calmar=o['w2_calmar'],
                 calmar_worst=round(o['w2_cagr_lo'] / abs(o['w2_dd_worst']), 3),
                 tpy=o['w2_tpy'], hold=o['w2_hold'], win=o['w2_win'],
                 avg_win=o['w2_avg_win'], avg_loss=o['w2_avg_loss'],
                 mean_tr=o['w2_mean'], netexp=o['w2_netexp'], streak=o['w2_streak'],
                 wa_cagr=o['wa_cagr'], wa_dd=o['wa_dd'], wa_netexp=o['wa_netexp'],
                 wb_cagr=o['wb_cagr'], wb_dd=o['wb_dd'], wb_netexp=o['wb_netexp'])
        for c in (0.0040, 0.0060):
            oc, _ = ih.run_cell(ctx, ir, {**cfg, 'cost': c}, trig=trig, level=lvl, lo=lo,
                                seeds=ih.SEEDS, windows=('w2',), fill_close=fc)
            r[f'cagr_{int(c*10000)}bps'] = oc['w2_cagr']
        tr = pd.DataFrame([x for t in kept['trades'] for x in t])
        tr['seed'] = np.repeat(ih.SEEDS, [len(t) for t in kept['trades']])
        ex = pd.Series({s: g.drop(g['ret'].nlargest(10).index)['ret'].mean()
                        for s, g in tr.groupby('seed')})
        r['mean_tr_ex_top10'] = round(100 * float(ex.median()), 2)
        for cp in (0.50, 1.00):
            cc = tr.copy(); cc['ret'] = cc['ret'].clip(upper=cp)
            r[f'mean_tr_cap{int(cp*100)}'] = round(
                100 * float(cc.groupby('seed')['ret'].mean().median()), 2)
        tops = tr.groupby('seed')['ret'].apply(lambda s: s.nlargest(10).sum())
        tot = tr.groupby('seed')['ret'].sum()
        r['top10_share_pct'] = round(100 * float(tops.median() / tot.median()), 1)
        tr['frac_tv'] = 100 * tr['notional'] / tr['tv']
        r['cap_med_pct_tv_10L'] = round(float(tr.frac_tv.median()), 3)
        r['cap_p90_pct_tv_10L'] = round(float(tr.frac_tv.quantile(.9)), 3)
        r['cap_p90_pct_tv_1cr'] = round(float(tr.frac_tv.quantile(.9)) * 10, 2)
        r['cap_p90_pct_tv_10cr'] = round(float(tr.frac_tv.quantile(.9)) * 100, 1)
        r['n_distinct_names'] = int(tr.col.nunique())
        rows.append(r)
        k = int(np.argsort(st.cagr.values)[len(st) // 2])
        nav = kept['navs'][k] / kept['navs'][k].iloc[0]
        curves[name] = nav
        yr = nav.groupby(nav.index.year).last()
        ret = yr.pct_change(); ret.iloc[0] = yr.iloc[0] - 1
        peryear[name] = dict(ret={int(a): round(100 * b, 2) for a, b in ret.items()},
                             intra_dd=ih.full_curve_dd_by_year(nav))
        perwin[name] = {}
        for wn, (a, b) in WINDOWS.items():
            rr, dd = win_stats(nav, a, b)
            perwin[name][wn] = dict(ret=rr, dd_from_full_peak=dd)
        print(f'{name:<30} CAGR {r["cagr"]:6.2f} [{r["cagr_worst"]:.2f}..{r["cagr_best"]:.2f}]'
              f'  noSweep {r["cagr_no_sweep"]:6.2f} ({r["sweep_share"]:4.1f}% is the sweep)'
              f'  inv {r["invested"]:4.1f}%  DD {r["dd"]:7.2f} (worst seed {r["dd_worst"]:7.2f})'
              f'  Cal {r["calmar"]:5.3f} (worst {r["calmar_worst"]:5.3f})', flush=True)
        print(f'{"":<30} cost 25/40/60 bps: {r["cagr"]:.2f} / {r["cagr_40bps"]:.2f} / '
              f'{r["cagr_60bps"]:.2f}   win {r["win"]:.1f}%  avgW {r["avg_win"]:.1f}% '
              f'avgL {r["avg_loss"]:.1f}%  netexp {r["netexp"]:.2f}%/tr  streak {r["streak"]}'
              f'  tpy {r["tpy"]:.1f}  hold {r["hold"]:.0f}d  top10 share {r["top10_share_pct"]:.0f}%'
              f'  mean/tr ex-top10 {r["mean_tr_ex_top10"]:.2f}%', flush=True)

    pd.DataFrame(rows).to_csv(RES / 'stage9_adoption.csv', index=False)
    json.dump(dict(peryear=peryear, perwindow=perwin),
              open(RES / 'stage9_peryear.json', 'w'), indent=1)
    pd.DataFrame(curves).to_csv(RES / 'stage9_curves.csv')

    print('\n=== PER YEAR: return (intra-year DD from the FULL curve running peak) ===')
    names = list(peryear)
    yrs = sorted(set().union(*[set(v['ret']) for v in peryear.values()]))
    print('year  ' + '  '.join(f'{n[:26]:>26}' for n in names))
    for y in yrs:
        print(f'{y}  ' + '  '.join(
            f'{peryear[n]["ret"].get(y, float("nan")):+8.1f} '
            f'({peryear[n]["intra_dd"].get(y, float("nan")):+6.1f})'.rjust(26)
            for n in names))

    print('\n=== PER WINDOW (return over the window; DD from the FULL curve peak) ===')
    for wn in WINDOWS:
        print(f'  {wn:<16} ' + '  '.join(
            f'{n[:10]}: {perwin[n][wn]["ret"]}% (dd {perwin[n][wn]["dd_from_full_peak"]}%)'
            for n in names))

    if R159.exists():
        ot = pd.read_csv(R159, index_col=0, parse_dates=True)
        cc = {}
        for n, nav in curves.items():
            j = pd.concat([nav.rename('x'), ot], axis=1).dropna()
            w = j.resample('W').last().pct_change().dropna()
            cc[n] = {kk: round(float(w['x'].corr(w[kk])), 3) for kk in ot.columns}
            cc[n]['n_weeks'] = int(len(w))
        json.dump(cc, open(RES / 'stage9_correlations.json', 'w'), indent=1)
        print('\n=== weekly correlations (r/159 after-tax curves) ===')
        print(pd.DataFrame(cc).T.to_string())
    print('\nstage9 done', flush=True)


if __name__ == '__main__':
    main()
