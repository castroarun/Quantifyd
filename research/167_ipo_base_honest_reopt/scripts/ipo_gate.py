# -*- coding: utf-8 -*-
"""research/163 stage 6 - the gate, PAIRED, and the final adoption rows.

Stage 4 printed unpaired medians. Unpaired medians lie at n=30 (r/146's DD10 gate looked
like a winner on 10-seed medians and lost on 20 of 30 paired paths). So: same seed, gate on
vs gate off, and the distribution of differences.

Also produces the per-year table for the three headline specs, with the intra-year drawdown
measured from the running peak of the FULL curve (r/154), and the cost ladder / cash-yield /
outlier columns for the gated arm.
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

REFIT = dict(ih.INCUMBENT)
REFIT.update(trail=50, target=0.25, stop=0.15)               # stage-1 best CAGR cell

SPECS = [('incumbent', dict(ih.INCUMBENT), None),
         ('refit_nogate', dict(REFIT), None),
         ('refit_sma150', dict(REFIT), 150),
         ('refit_sma200', dict(REFIT), 200),
         ('refit_sma100', dict(REFIT), 100),
         ('refit_calmar_tp100_sl10_nogate',
          {**ih.INCUMBENT, 'trail': 50, 'target': 1.00, 'stop': 0.10}, None),
         ('refit_calmar_tp100_sl10_sma150',
          {**ih.INCUMBENT, 'trail': 50, 'target': 1.00, 'stop': 0.10}, 150)]


def gate_series(ctx, n):
    if n is None:
        return np.zeros(len(ctx.dates), dtype=bool)
    nb = ctx.close.get('NIFTYBEES').dropna()
    w = (nb < nb.rolling(n).mean()).shift(1)
    return w.reindex(ctx.dates).ffill().fillna(False).to_numpy(bool)


def main():
    ctx, ir = ih.load_ctx(clean=True)
    out, curves, peryear = {}, {}, {}
    for name, cfg, gn in SPECS:
        c = dict(cfg)
        if gn is not None:
            c['gate'] = 'custom'
            c['weak_series'] = gate_series(ctx, gn)
        else:
            c['gate'] = False
        setup, piv, lo0 = ih.build_setup(ctx, c)
        trig, lvl, lo, fc = ih.apply_mechanic(ctx, setup, piv, lo0, 'nextday_pivot')
        o, kept = ih.run_cell(ctx, ir, c, trig=trig, level=lvl, lo=lo, keep=True,
                              windows=('w2', 'wa', 'wb'), fill_close=fc)
        o0, _ = ih.run_cell(ctx, ir, c, trig=trig, level=lvl, lo=lo, windows=('w2',),
                            fill_close=fc, cash_yield=0.0)
        st = kept['stats']
        row = dict(spec=name, gate_sma=gn, trail=c['trail'], target=c['target'],
                   stop=c['stop'],
                   cagr=o['w2_cagr'], cagr_worst=o['w2_cagr_lo'], cagr_best=o['w2_cagr_hi'],
                   cagr_no_sweep=o0['w2_cagr'],
                   sweep_pp=round(o['w2_cagr'] - o0['w2_cagr'], 2),
                   sweep_share=round(100 * (o['w2_cagr'] - o0['w2_cagr']) / o['w2_cagr'], 1),
                   invested=o['w2_inv'], dd=o['w2_dd'], dd_worst=o['w2_dd_worst'],
                   calmar=o['w2_calmar'],
                   calmar_worstcase=round(o['w2_cagr_lo'] / abs(o['w2_dd_worst']), 3),
                   tpy=o['w2_tpy'], hold=o['w2_hold'], win=o['w2_win'],
                   avg_win=o['w2_avg_win'], avg_loss=o['w2_avg_loss'],
                   netexp=o['w2_netexp'], streak=o['w2_streak'],
                   wa_cagr=o['wa_cagr'], wa_dd=o['wa_dd'], wa_netexp=o['wa_netexp'],
                   wb_cagr=o['wb_cagr'], wb_dd=o['wb_dd'], wb_netexp=o['wb_netexp'])
        for cc in (0.0040, 0.0060):
            oc, _ = ih.run_cell(ctx, ir, {**c, 'cost': cc}, trig=trig, level=lvl, lo=lo,
                                seeds=ih.SEEDS[:10], windows=('w2',), fill_close=fc)
            row[f'cagr_{int(cc*10000)}bps'] = oc['w2_cagr']
        tr = pd.DataFrame([x for t in kept['trades'] for x in t])
        tr['seed'] = np.repeat(ih.SEEDS, [len(t) for t in kept['trades']])
        row['mean_tr'] = round(100 * float(tr.groupby('seed')['ret'].mean().median()), 2)
        ex = pd.Series({s: g.drop(g['ret'].nlargest(10).index)['ret'].mean()
                        for s, g in tr.groupby('seed')})
        row['mean_tr_ex_top10'] = round(100 * float(ex.median()), 2)
        cap = tr.copy(); cap['ret'] = cap['ret'].clip(upper=0.50)
        row['mean_tr_cap50'] = round(100 * float(cap.groupby('seed')['ret'].mean().median()), 2)
        tr['frac_tv'] = 100 * tr['notional'] / tr['tv']
        row['cap_p90_pct_of_tv_at_10L'] = round(float(tr.frac_tv.quantile(.9)), 3)
        row['cap_p90_pct_of_tv_at_1cr'] = round(float(tr.frac_tv.quantile(.9)) * 10, 2)
        out[name] = dict(row=row, cagrs=st.cagr.values.tolist(),
                         dds=st.dd.values.tolist())
        k = int(np.argsort(st.cagr.values)[len(st) // 2])
        nav = kept['navs'][k] / kept['navs'][k].iloc[0]
        curves[name] = nav
        yr = nav.groupby(nav.index.year).last()
        ret = yr.pct_change(); ret.iloc[0] = yr.iloc[0] - 1
        peryear[name] = dict(ret={int(a): round(100 * b, 2) for a, b in ret.items()},
                             intra_dd=ih.full_curve_dd_by_year(nav))
        print(f'{name:<34} CAGR {row["cagr"]:6.2f} [{row["cagr_worst"]:.2f}..'
              f'{row["cagr_best"]:.2f}]  noSweep {row["cagr_no_sweep"]:6.2f}  '
              f'inv {row["invested"]:4.1f}%  DD {row["dd"]:7.2f} (worst {row["dd_worst"]:7.2f})'
              f'  Cal {row["calmar"]:5.3f}  WA {row["wa_cagr"]:6.2f} WB {row["wb_cagr"]:6.2f}',
              flush=True)

    print('\n=== PAIRED deltas, same seed, 30 seeds ===')
    pairs = [('refit_nogate', 'incumbent'), ('refit_sma150', 'refit_nogate'),
             ('refit_sma200', 'refit_nogate'), ('refit_sma100', 'refit_nogate'),
             ('refit_calmar_tp100_sl10_nogate', 'refit_nogate'),
             ('refit_calmar_tp100_sl10_sma150', 'refit_sma150')]
    prows = []
    for a, b in pairs:
        da = np.array(out[a]['cagrs']) - np.array(out[b]['cagrs'])
        dd = np.array(out[a]['dds']) - np.array(out[b]['dds'])
        prows.append(dict(a=a, b=b, cagr_delta_med=round(float(np.median(da)), 2),
                          cagr_delta_min=round(float(da.min()), 2),
                          cagr_delta_max=round(float(da.max()), 2),
                          a_wins_cagr=int((da > 0).sum()),
                          dd_delta_med=round(float(np.median(dd)), 2),
                          a_shallower_dd=int((dd > 0).sum())))
        print(f'  {a:<34} vs {b:<16} CAGR {np.median(da):+6.2f}pp '
              f'[{da.min():+6.2f}..{da.max():+6.2f}]  wins {int((da>0).sum())}/30  | '
              f'DD {np.median(dd):+6.2f}pp shallower on {int((dd>0).sum())}/30', flush=True)

    pd.DataFrame([v['row'] for v in out.values()]).to_csv(RES / 'stage6_final.csv', index=False)
    pd.DataFrame(prows).to_csv(RES / 'stage6_paired.csv', index=False)
    json.dump(peryear, open(RES / 'stage6_peryear.json', 'w'), indent=1)
    pd.DataFrame(curves).to_csv(RES / 'stage6_curves.csv')

    print('\n=== PER YEAR: return (intra-year DD from the FULL curve peak) ===')
    yrs = sorted(set().union(*[set(v['ret']) for v in peryear.values()]))
    names = list(peryear)
    print('year  ' + '  '.join(f'{n[:18]:>20}' for n in names))
    for y in yrs:
        cells = []
        for n in names:
            r = peryear[n]['ret'].get(y)
            d = peryear[n]['intra_dd'].get(y)
            cells.append('n/a'.rjust(20) if r is None
                         else f'{r:+7.1f} ({d:+6.1f})'.rjust(20))
        print(f'{y}  ' + '  '.join(cells))

    if R159.exists():
        ot = pd.read_csv(R159, index_col=0, parse_dates=True)
        cc = {}
        for n, nav in curves.items():
            j = pd.concat([nav.rename('x'), ot], axis=1).dropna()
            w = j.resample('W').last().pct_change().dropna()
            cc[n] = {k: round(float(w['x'].corr(w[k])), 3) for k in ot.columns}
            cc[n]['n_weeks'] = int(len(w))
        json.dump(cc, open(RES / 'stage6_correlations.json', 'w'), indent=1)
        print('\n=== weekly correlations (r/159 after-tax curves) ===')
        print(pd.DataFrame(cc).T.to_string())
    print('\nstage6 done', flush=True)


if __name__ == '__main__':
    main()
