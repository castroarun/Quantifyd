# -*- coding: utf-8 -*-
"""research/163 stage 5 - the adoption arithmetic for the shortlist.

For the incumbent spec and every shortlisted refit, side by side:
  * after-tax CAGR WITH the 5% idle-cash sweep and with the sweep set to ZERO - the gap
    IS the sweep's contribution, and IPO Base is only ~32-47% invested, so it is large
  * the invested fraction (mean of marked positions / equity over trading days)
  * median AND worst-seed drawdown (the median alone misleads about the unlucky path)
  * cost ladder 25 / 40 / 60 bps per side
  * outlier dependence - per-trade mean with each seed's 10 best trades deleted
  * per-year return with the intra-year drawdown measured from the running peak of the
    FULL curve, never from the window's first bar (r/154)
  * weekly correlation to True North and Open Alpha - Base Age, from r/159's curve file
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/167_ipo_base_honest_reopt/scripts'))
import ipo_honest as ih                                   # noqa: E402

RES = ih.RES
R159 = ROOT / 'research/159_oa_honest_reoptimization/results/full_period_after_tax.csv'


def spec_from_row(r):
    cfg = {}
    for k in ih.CFGKEYS:
        v = r[k]
        if isinstance(v, float) and pd.isna(v):
            v = None
        cfg[k] = v
    for k in ('trail', 'L', 'min_bars', 'max_age_m', 'slots'):
        cfg[k] = int(cfg[k])
    cfg['gate'] = False
    if isinstance(cfg['rs_policy'], str) is False:
        cfg['rs_policy'] = 'off'
    return cfg


def shortlist():
    out = [('INCUMBENT r/153 (trail20 tp25 sl8)', dict(ih.INCUMBENT))]
    s1 = pd.read_csv(RES / 'stage1_exits.csv')
    ok = s1[(s1.wa_netexp > 0) & (s1.wb_netexp > 0) & (s1.w2_cagr_lo > 0)]
    for _, r in ok.sort_values('w2_cagr', ascending=False).head(2).iterrows():
        out.append((f'S1 bestCAGR {r.label}', spec_from_row(r)))
    for _, r in ok.sort_values('w2_calmar', ascending=False).head(4).iterrows():
        lab = f'S1 bestCalmar {r.label}'
        if not any(l.endswith(r.label) for l, _ in out):
            out.append((lab, spec_from_row(r)))
    for nm, f in (('S2a geometry', 'stage2a_geometry.csv'), ('S2b book', 'stage2b_book.csv')):
        p = RES / f
        if not p.exists():
            continue
        g = pd.read_csv(p)
        g = g[(g.wa_netexp > 0) & (g.wb_netexp > 0) & (g.w2_cagr_lo > 0)]
        if not len(g):
            continue
        for key in ('w2_cagr', 'w2_calmar'):
            r = g.sort_values(key, ascending=False).iloc[0]
            lab = f'{nm} best{key[3:]} {r.label}'
            if not any(l.endswith(r.label) for l, _ in out):
                out.append((lab, spec_from_row(r)))
    return out


def main():
    ctx, ir = ih.load_ctx(clean=True)
    rows, curves, peryear = [], {}, {}
    for name, cfg in shortlist():
        setup, piv, lo0 = ih.build_setup(ctx, cfg)
        trig, lvl, lo, fc = ih.apply_mechanic(ctx, setup, piv, lo0, 'nextday_pivot')
        o, kept = ih.run_cell(ctx, ir, cfg, trig=trig, level=lvl, lo=lo, keep=True,
                              windows=('w2', 'wa', 'wb'), fill_close=fc)
        o0, _ = ih.run_cell(ctx, ir, cfg, trig=trig, level=lvl, lo=lo,
                            windows=('w2',), fill_close=fc, cash_yield=0.0)
        row = dict(spec=name, trail=cfg['trail'], target=cfg['target'], stop=cfg['stop'],
                   age_m=cfg['max_age_m'], L=cfg['L'], depth=cfg['max_depth'],
                   rs=cfg['rs_policy'], slots=cfg['slots'], size_pct=cfg['size_pct'],
                   cagr=o['w2_cagr'], cagr_worst_seed=o['w2_cagr_lo'],
                   cagr_no_cash_yield=o0['w2_cagr'],
                   cash_sweep_pp=round(o['w2_cagr'] - o0['w2_cagr'], 2),
                   sweep_share_pct=round(100 * (o['w2_cagr'] - o0['w2_cagr'])
                                         / o['w2_cagr'], 1) if o['w2_cagr'] else np.nan,
                   invested_pct=o['w2_inv'], dd=o['w2_dd'], dd_worst_seed=o['w2_dd_worst'],
                   calmar=o['w2_calmar'],
                   calmar_worst_seed=round(o['w2_cagr_lo'] / abs(o['w2_dd_worst']), 3),
                   tpy=o['w2_tpy'], hold_d=o['w2_hold'], win=o['w2_win'],
                   avg_win=o['w2_avg_win'], avg_loss=o['w2_avg_loss'],
                   netexp=o['w2_netexp'], streak=o['w2_streak'],
                   wa_cagr=o['wa_cagr'], wa_netexp=o['wa_netexp'],
                   wb_cagr=o['wb_cagr'], wb_netexp=o['wb_netexp'])
        # cost ladder (10 seeds, enough to rank)
        for c in (0.0040, 0.0060):
            oc, _ = ih.run_cell(ctx, ir, {**cfg, 'cost': c}, trig=trig, level=lvl, lo=lo,
                                seeds=ih.SEEDS[:10], windows=('w2',), fill_close=fc)
            row[f'cagr_{int(c*10000)}bps'] = oc['w2_cagr']
        # outlier dependence
        tr = pd.DataFrame([x for t in kept['trades'] for x in t])
        tr['seed'] = np.repeat(ih.SEEDS, [len(t) for t in kept['trades']])
        row['mean_tr'] = round(100 * float(tr.groupby('seed')['ret'].mean().median()), 2)
        ex10 = pd.Series({sd: g.drop(g['ret'].nlargest(10).index)['ret'].mean()
                          for sd, g in tr.groupby('seed')})
        row['mean_tr_ex_top10'] = round(100 * float(ex10.median()), 2)
        cap = tr.copy(); cap['ret'] = cap['ret'].clip(upper=0.50)
        row['mean_tr_cap50'] = round(100 * float(cap.groupby('seed')['ret'].mean().median()), 2)
        tops = tr.groupby('seed')['ret'].apply(lambda s: s.nlargest(10).sum())
        tot = tr.groupby('seed')['ret'].sum()
        row['top10_share_of_sum_pct'] = round(100 * float(tops.median() / tot.median()), 1)
        rows.append(row)
        # median-CAGR seed curve (never the mean of the paths)
        st = kept['stats']
        k = int(np.argsort(st.cagr.values)[len(st) // 2])
        nav = kept['navs'][k] / kept['navs'][k].iloc[0]
        curves[name] = nav
        yr = nav.groupby(nav.index.year).last()
        ret = yr.pct_change()
        ret.iloc[0] = yr.iloc[0] - 1
        peryear[name] = {'ret': {int(a): round(100 * b, 2) for a, b in ret.items()},
                         'intra_dd_full_curve_peak': ih.full_curve_dd_by_year(nav)}
        print(f'{name:<52} CAGR {row["cagr"]:6.2f} (no sweep {row["cagr_no_cash_yield"]:6.2f}, '
              f'sweep {row["cash_sweep_pp"]:+5.2f}pp = {row["sweep_share_pct"]:4.1f}%)  '
              f'inv {row["invested_pct"]:4.1f}%  DD {row["dd"]:7.2f} (worst seed '
              f'{row["dd_worst_seed"]:7.2f})  Cal {row["calmar"]:5.3f}', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / 'stage5_adoption.csv', index=False)
    json.dump(peryear, open(RES / 'stage5_peryear.json', 'w'), indent=1)
    pd.DataFrame(curves).to_csv(RES / 'stage5_curves.csv')

    # correlations against the other books (r/159's after-tax curves, read-only)
    if R159.exists():
        ot = pd.read_csv(R159, index_col=0, parse_dates=True)
        cc = {}
        for name, nav in curves.items():
            j = pd.concat([nav.rename('ipo'), ot], axis=1).dropna()
            w = j.resample('W').last().pct_change().dropna()
            cc[name] = {c: round(float(w['ipo'].corr(w[c])), 3)
                        for c in ot.columns}
            cc[name]['n_weeks'] = int(len(w))
        json.dump(cc, open(RES / 'stage5_correlations.json', 'w'), indent=1)
        print('\nweekly correlations vs the other books:')
        print(json.dumps(cc, indent=1))
    print('\nstage5 written ->', RES / 'stage5_adoption.csv', flush=True)


if __name__ == '__main__':
    main()
