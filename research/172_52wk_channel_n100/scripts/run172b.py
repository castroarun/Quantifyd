# -*- coding: utf-8 -*-
"""research/172 part B - the controls that decide the verdict, run at the OPTIMUM cell
(not just at the literal spec), plus the start-date phase ensemble and the per-year table.

Phases: nullopt / phase / peryear / all
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path('/home/arun/quantifyd/research/172_52wk_channel_n100')
sys.path.insert(0, str(HERE / 'scripts'))
RES = HERE / 'results'

import p172                     # noqa: E402
import bt172 as B               # noqa: E402
import run172 as R              # noqa: E402

OPT = dict(uni='n100', L=252, ref='close', exit='ST_14_4', slots=20, gate='none')
LIT = dict(uni='n100', L=252, ref='close', exit='CC252', slots=20, gate='none')


def matched_null(P, base, seed, days):
    """Random entry drawn from the SAME eligible pool, matched fill-for-fill on count."""
    trig = B.entry_signal(P, base['uni'], base['L'], ref=base['ref'])
    elig = P.eligible(base['uni'], base['L'])
    rng = np.random.default_rng(seed)
    null = np.zeros_like(trig)
    nper = trig.sum(axis=1)
    for i in np.nonzero(nper)[0]:
        pool = np.nonzero(elig[i - 1] if i else elig[i])[0]
        if not len(pool):
            continue
        k = min(int(nper[i]), len(pool))
        null[i, rng.choice(pool, size=k, replace=False)] = True
    cfg = dict(days=days, slots=base['slots'], cost_bps=15.0, gate=None, seed=None, tax=True)
    cfg.update(R.exit_cfg(P, base['exit']))
    return B.simulate(P, null, cfg)


def phase_nullopt(P):
    path = RES / 'controls2.csv'
    done = R.done_labels(path)
    days = P.days['full']
    for nm, base in (('OPT', OPT), ('LIT', LIT)):
        for s in range(1, 31):
            lab = 'null_%s_%02d' % (nm, s)
            if lab in done:
                continue
            r = matched_null(P, base, 30000 + s, days)
            m = B.metrics(r['nav'], r['dates'], r['trades'])
            row = dict(label=lab, phase='null', uni=base['uni'], entry_L=base['L'],
                       entry_ref=base['ref'], exit=base['exit'], slots=base['slots'],
                       gate='none', cost_bps=15.0, seed=s, window='full', tax=True,
                       avg_invested=round(float(np.nanmean(r['invested'])), 3))
            row.update(m)
            R.append(path, row)
        print('  [null_%s] 30 draws done' % nm, flush=True)

    # risk-matched cash null: the SAME buy-and-hold universe, de-levered to the
    # book's own average invested fraction, remainder at 5.2% post-tax.
    bh = B.bh_equal_weight(P, 'n100', days)
    rbh = np.r_[0.0, np.diff(bh) / bh[:-1]]
    cashd = (1.052 ** (1 / 252.0)) - 1.0
    for w in (0.60, 0.65, 0.712, 0.75, 0.80, 0.90, 1.00):
        lab = 'riskmatch_bh_%02d' % int(round(w * 100))
        if lab in done:
            continue
        nav = B.START_CAPITAL * np.cumprod(1.0 + w * rbh + (1 - w) * cashd)
        m = B.metrics(nav, P.dstr[days[0]:days[-1] + 1])
        row = dict(label=lab, phase='control', uni='n100', window='full')
        row.update(m)
        R.append(path, row)
        print('  [riskmatch] w=%.3f cagr=%6.2f dd=%7.2f calmar=%5.3f'
              % (w, m['cagr'], m['maxdd'], m['calmar']), flush=True)


def phase_phase(P):
    """Start-date phase ensemble: the deterministic-book analogue of a seed ensemble."""
    path = RES / 'startphase.csv'
    done = R.done_labels(path)
    for nm, base in (('OPT', OPT), ('LIT', LIT)):
        for m_ in range(12):
            lab = 'phase_%s_m%02d' % (nm, m_)
            if lab in done:
                continue
            start = '2006-%02d-01' % (m_ + 1)
            days = np.nonzero((P.dstr >= start) & (P.dstr <= p172.TRADE_END))[0]
            cfg = dict(days=days, slots=base['slots'], cost_bps=15.0, gate=None,
                       seed=None, tax=True)
            cfg.update(R.exit_cfg(P, base['exit']))
            trig = B.entry_signal(P, base['uni'], base['L'], ref=base['ref'])
            r = B.simulate(P, trig, cfg)
            mm = B.metrics(r['nav'], r['dates'], r['trades'])
            row = dict(label=lab, phase='startphase', uni=base['uni'], entry_L=base['L'],
                       entry_ref=base['ref'], exit=base['exit'], slots=base['slots'],
                       gate='none', cost_bps=15.0, window='full', tax=True)
            row.update(mm)
            R.append(path, row)
        print('  [startphase %s] 12 offsets done' % nm, flush=True)


def phase_peryear(P):
    """Per-year returns + intra-year drawdown FROM THE FULL CURVE'S PEAK (r/154 rule)."""
    days = P.days['full']
    idx = pd.to_datetime(P.dstr[days[0]:days[-1] + 1])
    series = {}

    def add(name, nav):
        series[name] = pd.Series(np.asarray(nav, float), index=idx)

    for nm, base in (('52W OPT', OPT), ('52W Spec A', LIT)):
        cfg = dict(days=days, slots=base['slots'], cost_bps=15.0, gate=None, seed=None,
                   tax=True)
        cfg.update(R.exit_cfg(P, base['exit']))
        r = B.simulate(P, B.entry_signal(P, base['uni'], base['L'], ref=base['ref']), cfg)
        add(nm, r['nav'])
        pd.DataFrame(r['trades']).to_csv(RES / ('trades_%s.csv' % nm.replace(' ', '_')),
                                         index=False)
    # PIT (survivorship-free) version of the optimum
    cfg = dict(days=days, slots=20, cost_bps=15.0, gate=None, seed=None, tax=True)
    cfg.update(R.exit_cfg(P, 'ST_14_4'))
    r = B.simulate(P, B.entry_signal(P, 'pit100', 252), cfg)
    add('52W OPT (PIT-100)', r['nav'])

    add('EW B&H Nifty100', B.bh_equal_weight(P, 'n100', days))
    nb = pd.Series(P.bench, index=pd.to_datetime(P.dstr)).iloc[days[0]:days[-1] + 1]
    add('NIFTYBEES', nb.to_numpy(float))
    n = len(idx)
    add('Cash 5.2%', B.START_CAPITAL * (1.052 ** (1 / 252.0)) ** np.arange(n))
    # median null path at the optimum
    c2 = pd.read_csv(RES / 'controls2.csv')
    nl = c2[c2.label.str.startswith('null_OPT')]
    med_seed = int(nl.sort_values('cagr').iloc[len(nl) // 2]['seed'])
    rn = matched_null(P, OPT, 30000 + med_seed, days)
    add('Random-entry null (median path)', rn['nav'])

    rows = []
    summary = {}
    for name, s in series.items():
        peak = s.cummax()
        dd = s / peak - 1.0
        for y in sorted(set(s.index.year)):
            m = s.index.year == y
            j = np.flatnonzero(m)
            s0 = j[0] - 1 if j[0] > 0 else j[0]
            rows.append(dict(series=name, year=int(y),
                             ret=round(100 * (s.iloc[j[-1]] / s.iloc[s0] - 1), 2),
                             dd=round(100 * float(dd[m].min()), 2)))
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        cagr = 100 * ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1)
        mdd = 100 * float(dd.min())
        summary[name] = dict(cagr=round(cagr, 2), maxdd=round(mdd, 2),
                             calmar=round(cagr / abs(mdd), 3) if mdd < 0 else None,
                             final=round(float(s.iloc[-1])))
    pd.DataFrame(rows).to_csv(RES / 'peryear_all.csv', index=False)
    json.dump(summary, open(RES / 'peryear_summary.json', 'w'), indent=1)
    np.savez_compressed(RES / 'curves.npz',
                        dates=np.array([str(d.date()) for d in idx]),
                        **{k.replace(' ', '_').replace('%', 'p'): v.to_numpy(float)
                           for k, v in series.items()})
    print(json.dumps(summary, indent=1), flush=True)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    P = p172.Panel()
    steps = {'nullopt': [phase_nullopt], 'phase': [phase_phase], 'peryear': [phase_peryear],
             'all': [phase_nullopt, phase_phase, phase_peryear]}[what]
    for fn in steps:
        print('=== %s ===' % fn.__name__, flush=True)
        fn(P)
    print('DONE %s in %.1f min' % (what, (time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
