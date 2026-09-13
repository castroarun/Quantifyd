# -*- coding: utf-8 -*-
"""research/172 - runner. Phases: specA / grid / axes / robust / blend / all.

Every phase appends ONE CSV ROW PER COMPLETED CELL and skips labels already present,
so re-launching the same command resumes.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path('/home/arun/quantifyd/research/172_52wk_channel_n100')
sys.path.insert(0, str(HERE / 'scripts'))
RES = HERE / 'results'
RES.mkdir(parents=True, exist_ok=True)

import p172                     # noqa: E402
import bt172 as B               # noqa: E402

FIELDS = ['label', 'phase', 'uni', 'entry_L', 'entry_ref', 'buffer', 'ath_within', 'new_ath',
          'exit', 'slots', 'gate', 'cost_bps', 'seed', 'window', 'tax',
          'cagr', 'maxdd', 'calmar', 'sharpe', 'final', 'years', 'trades', 'win_rate',
          'avg_win', 'avg_loss', 'expectancy', 'med_hold_d', 'worst_mae', 'med_mae',
          'max_loss_streak', 'trades_per_yr', 'tax_paid', 'cost_paid', 'avg_invested']

EXITS = (['CC%d' % w for w in p172.EXIT_LOOKBACKS]
         + ['CL%d' % w for w in p172.EXIT_LOOKBACKS]
         + ['ST_7_3', 'ST_10_3', 'ST_14_4',
            'ATR25', 'ATR40', 'ATR60',
            'SMA15', 'SMA15_S8', 'EMA50'])

_EXCACHE = {}


def exarr(P, key):
    if key not in _EXCACHE:
        base = key.replace('_S8', '')
        _EXCACHE[key] = B.exit_array(P, base) if not base.startswith('ATR') else None
    return _EXCACHE[key]


def exit_cfg(P, key):
    """-> dict(exit_arr=..., atr_mult=..., hard_stop=...)"""
    if key.startswith('ATR'):
        return dict(exit_arr=None, atr_mult=float(key[3:]) / 10.0, hard_stop=0.0)
    if key.endswith('_S8'):
        return dict(exit_arr=exarr(P, key), atr_mult=0.0, hard_stop=0.08)
    return dict(exit_arr=exarr(P, key), atr_mult=0.0, hard_stop=0.0)


def done_labels(path):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        return set()
    with open(path) as f:
        return {r['label'] for r in csv.DictReader(f)}


def append(path, row):
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore').writerow(row)


def run_cell(P, label, phase, uni='n100', L=252, ref='close', buffer=0.0,
             ath_within=None, new_ath=False, exit='CC252', slots=20, gate='none',
             cost_bps=15.0, seed=None, window='full', tax=True, same_close=False,
             keep=False, idle_yield=B.IDLE_YIELD):
    trig = B.entry_signal(P, uni, L, ref=ref, buffer=buffer, ath_within=ath_within,
                          new_ath=new_ath)
    if same_close:                       # look-ahead reference arm only
        trig = np.zeros_like(trig)
        s = B.entry_signal(P, uni, L, ref=ref, buffer=buffer, ath_within=ath_within,
                           new_ath=new_ath)
        trig[:-1] = s[1:]
    g = {'none': None, 'nb200': P.GATE200, 'nb100': P.GATE100}[gate]
    cfg = dict(days=P.days[window], slots=slots, cost_bps=cost_bps, gate=g, seed=seed,
               tax=tax, same_close_fill=same_close, idle_yield=idle_yield)
    cfg.update(exit_cfg(P, exit))
    r = B.simulate(P, trig, cfg)
    m = B.metrics(r['nav'], r['dates'], r['trades'])
    row = dict(label=label, phase=phase, uni=uni, entry_L=L, entry_ref=ref,
               buffer=buffer, ath_within=ath_within, new_ath=new_ath,
               exit=exit, slots=slots, gate=gate,
               cost_bps=cost_bps, seed=seed, window=window, tax=tax,
               tax_paid=round(r['tax_paid']), cost_paid=round(r['cost_paid']),
               avg_invested=round(float(np.nanmean(r['invested'])), 3))
    row.update(m)
    if keep:
        return row, r
    return row, None


# ------------------------------------------------------------------ phase: specA
def phase_specA(P):
    path = RES / 'specA.csv'
    done = done_labels(path)
    out = {}
    cells = [
        ('specA_literal', dict()),
        ('specA_gross_nocost_notax', dict(cost_bps=0.0, tax=False)),
        ('specA_net_notax', dict(tax=False)),
        ('specA_sameclose_LOOKAHEAD', dict(same_close=True)),
        ('specA_cost0', dict(cost_bps=0.0)),
        ('specA_cost30', dict(cost_bps=30.0)),
        ('specA_cost45', dict(cost_bps=45.0)),
        ('specA_noidle', dict(idle_yield=0.0)),
        ('specA_w1', dict(window='w1')),
        ('specA_w2', dict(window='w2')),
        ('specA_n50', dict(uni='n50')),
        ('specA_nn50', dict(uni='nn50')),
        ('specA_n500', dict(uni='n500')),
        ('specA_pit100', dict(uni='pit100')),
        ('specA_highref', dict(ref='high')),
    ]
    for lab, kw in cells:
        if lab in done:
            continue
        t0 = time.time()
        row, r = run_cell(P, lab, 'specA', keep=(lab == 'specA_literal'), **kw)
        append(path, row)
        if r is not None:
            np.savez_compressed(RES / 'specA_nav.npz', nav=r['nav'], dates=r['dates'])
            pd.DataFrame(r['trades']).to_csv(RES / 'specA_trades.csv', index=False)
            out['nav'] = r['nav']
        print('  [specA] %-28s cagr=%6.2f dd=%7.2f calmar=%5.2f tr=%s (%.0fs)'
              % (lab, row.get('cagr', float('nan')), row.get('maxdd', float('nan')),
                 row.get('calmar', float('nan')), row.get('trades'), time.time() - t0),
              flush=True)

    # ---- controls: random-entry null (trade-count matched), drift, cash
    npath = RES / 'controls.csv'
    ndone = done_labels(npath)
    days = P.days['full']
    if True:
        for uni in ('n100', 'n50', 'nn50', 'n500'):
            lab = 'drift_ew_' + uni
            if lab in ndone:
                continue
            nav = B.bh_equal_weight(P, uni, days)
            m = B.metrics(nav, P.dstr[days[0]:days[-1] + 1])
            row = dict(label=lab, phase='control', uni=uni, window='full')
            row.update(m)
            append(npath, row)
            print('  [control] %-20s cagr=%6.2f dd=%7.2f calmar=%5.2f'
                  % (lab, m['cagr'], m['maxdd'], m['calmar']), flush=True)
    if 'bench_niftybees' not in ndone:
        nb = pd.Series(P.bench, index=pd.to_datetime(P.dstr))
        nb = nb.iloc[days[0]:days[-1] + 1]
        m = B.metrics(nb.to_numpy(), P.dstr[days[0]:days[-1] + 1])
        row = dict(label='bench_niftybees', phase='control', window='full')
        row.update(m)
        append(npath, row)
        print('  [control] bench_niftybees cagr=%6.2f dd=%7.2f' % (m['cagr'], m['maxdd']),
              flush=True)
    if 'cash_only' not in ndone:
        n = len(days)
        nav = B.START_CAPITAL * (1.0 + (1.052 ** (1 / 252.0) - 1)) ** np.arange(n)
        m = B.metrics(nav, P.dstr[days[0]:days[-1] + 1])
        row = dict(label='cash_only', phase='control', window='full')
        row.update(m)
        append(npath, row)
    # random-entry null: same number of fills per day, drawn from the eligible pool
    for r_ in range(1, 31):
        lab = 'null_random_%02d' % r_
        if lab in ndone:
            continue
        trig = B.entry_signal(P, 'n100', 252)
        elig = P.eligible('n100', 252)
        rng = np.random.default_rng(20260913 + r_)
        null = np.zeros_like(trig)
        nper = trig.sum(axis=1)
        for i in np.nonzero(nper)[0]:
            pool = np.nonzero(elig[i - 1] if i else elig[i])[0]
            if not len(pool):
                continue
            k = min(int(nper[i]), len(pool))
            null[i, rng.choice(pool, size=k, replace=False)] = True
        cfg = dict(days=days, slots=20, cost_bps=15.0, gate=None, seed=None, tax=True)
        cfg.update(exit_cfg(P, 'CC252'))
        res = B.simulate(P, null, cfg)
        m = B.metrics(res['nav'], res['dates'], res['trades'])
        row = dict(label=lab, phase='control', uni='n100', exit='CC252', window='full')
        row.update(m)
        append(npath, row)
        print('  [control] %s cagr=%6.2f dd=%7.2f exp=%.3f'
              % (lab, m['cagr'], m['maxdd'], m['expectancy']), flush=True)
    return out


# ------------------------------------------------------------------ phase: grid
def phase_grid(P):
    path = RES / 'grid.csv'
    done = done_labels(path)
    total = len(p172.ENTRY_LOOKBACKS) * 2 * len(EXITS)
    k = 0
    t0 = time.time()
    for L in p172.ENTRY_LOOKBACKS:
        for ref in ('close', 'high'):
            for ex in EXITS:
                k += 1
                lab = 'G_L%d_%s_%s' % (L, ref, ex)
                if lab in done:
                    continue
                row, _ = run_cell(P, lab, 'grid', L=L, ref=ref, exit=ex)
                append(path, row)
                if k % 20 == 0 or k == total:
                    print('  [grid] %d/%d  %-26s cagr=%6.2f dd=%7.2f calmar=%5.2f (%.1f min)'
                          % (k, total, lab, row.get('cagr', np.nan), row.get('maxdd', np.nan),
                             row.get('calmar', np.nan), (time.time() - t0) / 60), flush=True)
    print('  [grid] complete %d cells in %.1f min' % (total, (time.time() - t0) / 60),
          flush=True)


def top_from_grid(n=3):
    df = pd.read_csv(RES / 'grid.csv')
    df = df[df.trades.fillna(0) >= 50]
    df = df.sort_values('calmar', ascending=False)
    return df.head(n).to_dict('records')


# ------------------------------------------------------------------ phase: axes
def phase_axes(P):
    path = RES / 'axes.csv'
    done = done_labels(path)
    tops = top_from_grid(3)
    base = [dict(L=int(t['entry_L']), ref=t['entry_ref'], exit=t['exit']) for t in tops]
    base.append(dict(L=252, ref='close', exit='CC252'))          # the literal spec
    t0 = time.time()
    jobs = []
    for b in base:
        tag = 'L%d_%s_%s' % (b['L'], b['ref'], b['exit'])
        for s in (10, 15, 20, 30, 100):
            jobs.append(('AX_slots%d_%s' % (s, tag), dict(slots=s, **b)))
        for u in ('n50', 'nn50', 'n100', 'n500', 'pit100', 'pit50'):
            jobs.append(('AX_uni%s_%s' % (u, tag), dict(uni=u, **b)))
        for g in ('none', 'nb200', 'nb100'):
            jobs.append(('AX_gate%s_%s' % (g, tag), dict(gate=g, **b)))
        for bu, aw, na, nm in ((0.0, None, False, 'buf0'), (0.01, None, False, 'buf1'),
                               (0.03, None, False, 'buf3'), (0.0, 0.05, False, 'ath5'),
                               (0.0, None, True, 'athnew')):
            jobs.append(('AX_%s_%s' % (nm, tag),
                         dict(buffer=bu, ath_within=aw, new_ath=na, **b)))
        for c in (0.0, 15.0, 30.0, 45.0):
            jobs.append(('AX_cost%d_%s' % (int(c), tag), dict(cost_bps=c, **b)))
        for w in ('w1', 'w2'):
            jobs.append(('AX_%s_%s' % (w, tag), dict(window=w, **b)))
        for sd in range(1, 31):
            jobs.append(('AX_seed%02d_%s' % (sd, tag), dict(seed=sd, **b)))
    seen = set()
    jobs = [(a, kw) for a, kw in jobs if not (a in seen or seen.add(a))]
    for k, (lab, kw) in enumerate(jobs, 1):
        if lab in done:
            continue
        row, _ = run_cell(P, lab, 'axes', **kw)
        append(path, row)
        if k % 25 == 0:
            print('  [axes] %d/%d %-38s cagr=%6.2f calmar=%5.2f (%.1f min)'
                  % (k, len(jobs), lab, row.get('cagr', np.nan), row.get('calmar', np.nan),
                     (time.time() - t0) / 60), flush=True)
    print('  [axes] complete %d cells in %.1f min' % (len(jobs), (time.time() - t0) / 60),
          flush=True)


# ------------------------------------------------------------------ phase: robust
def phase_robust(P):
    """Shortlist -> per-year, outlier deletion, MAE, trade export, NAV archive."""
    ax = pd.read_csv(RES / 'axes.csv')
    gr = pd.read_csv(RES / 'grid.csv')
    cand = pd.concat([gr, ax], ignore_index=True)
    cand = cand[(cand.window == 'full') & (cand.seed.isna()) & (cand.cost_bps == 15.0)
                & (cand.trades.fillna(0) >= 50)]
    cand = cand.sort_values('calmar', ascending=False).head(6)
    shortlist = []
    for _, c in cand.iterrows():
        shortlist.append(dict(label=str(c.label), uni=str(c.uni), L=int(c.entry_L),
                              ref=str(c.entry_ref),
                              buffer=float(c.buffer) if pd.notna(c.buffer) else 0.0,
                              ath_within=(float(c.ath_within)
                                          if pd.notna(c.ath_within) else None),
                              new_ath=bool(c.new_ath) if pd.notna(c.new_ath) else False,
                              exit=str(c['exit']), slots=int(c.slots), gate=str(c.gate)))
    shortlist.append(dict(label='SPEC_A', uni='n100', L=252, ref='close', buffer=0.0,
                          ath_within=None, new_ath=False, exit='CC252', slots=20,
                          gate='none'))
    json.dump(shortlist, open(RES / 'shortlist.json', 'w'), indent=1)

    navs, peryear, outl = {}, [], []
    for s in shortlist:
        kw = {k: v for k, v in s.items() if k != 'label'}
        row, r = run_cell(P, s['label'], 'robust', keep=True, **kw)
        navs[s['label']] = r['nav']
        nav = pd.Series(r['nav'], index=pd.to_datetime(r['dates']))
        peak = nav.cummax()
        for y, g in nav.groupby(nav.index.year):
            pk = peak.reindex(g.index)
            prev = nav.reindex(nav.index[nav.index.year == y - 1])
            start = prev.iloc[-1] if len(prev) else g.iloc[0]
            peryear.append(dict(label=s['label'], year=int(y),
                                ret=round(100 * (g.iloc[-1] / start - 1), 2),
                                dd=round(100 * float((g / pk - 1).min()), 2)))
        t = pd.DataFrame(r['trades'])
        t.to_csv(RES / ('trades_%s.csv' % s['label']), index=False)
        # outlier dependence: delete the top-10 trades, cap winners
        base_pnl = t.pnl.sum()
        top10 = t.nlargest(10, 'pnl').pnl.sum()
        outl.append(dict(label=s['label'], n=len(t), total_pnl=round(base_pnl),
                         top10_pnl=round(top10),
                         top10_share=round(100 * top10 / base_pnl, 1) if base_pnl else None,
                         pnl_ex_top10=round(base_pnl - top10),
                         mean_ret=round(float(t.ret_pct.mean()), 3),
                         mean_ret_cap50=round(float(t.ret_pct.clip(upper=50).mean()), 3),
                         mean_ret_cap100=round(float(t.ret_pct.clip(upper=100).mean()), 3),
                         worst_mae=round(float(t.mae_pct.min()), 1),
                         med_mae=round(float(t.mae_pct.median()), 1),
                         p95_hold_d=int(t.days.quantile(0.95))))
        print('  [robust] %-34s cagr=%6.2f dd=%7.2f calmar=%5.2f'
              % (s['label'], row.get('cagr', np.nan), row.get('maxdd', np.nan),
                 row.get('calmar', np.nan)), flush=True)
    np.savez_compressed(RES / 'navs.npz', dates=P.dstr[P.days['full'][0]:P.days['full'][-1] + 1],
                        **navs)
    pd.DataFrame(peryear).to_csv(RES / 'peryear.csv', index=False)
    pd.DataFrame(outl).to_csv(RES / 'outliers.csv', index=False)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    P = p172.Panel()
    print('[run172] universes: n50=%d nn50=%d n100=%d n500=%d | cols=%d'
          % (P.UNI['n50'].sum(), P.UNI['nn50'].sum(), P.UNI['n100'].sum(),
             P.UNI['n500'].sum(), P.N), flush=True)
    if len(P.events):
        P.events.to_csv(RES / 'split_events.csv', index=False)
    steps = {'specA': [phase_specA], 'grid': [phase_grid], 'axes': [phase_axes],
             'robust': [phase_robust],
             'all': [phase_specA, phase_grid, phase_axes, phase_robust]}[what]
    for fn in steps:
        print('=== %s ===' % fn.__name__, flush=True)
        fn(P)
    print('DONE %s in %.1f min' % (what, (time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
