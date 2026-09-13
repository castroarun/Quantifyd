# -*- coding: utf-8 -*-
"""research/172 part C - the adversarial nulls that decide whether the 52-week HIGH is
doing any work at all, plus the shortlist robustness on the cells that actually pass the
pre-registered CAGR gate.

Three nulls, each matched fill-for-fill on the number of entries per day:
  N1 plain      random name from the eligible universe
  N2 TREND      random name from the eligible universe that is ALSO in a SuperTrend(14,4)
                UPTREND that day. This is the fair null for a trend exit: without it the
                random book is thrown out on the next bar and the comparison measures the
                exit, not the entry.
  N3 MOM        random name from the top half of the universe by 252-day relative strength
                (momentum-matched) - does "52-week high" beat "simply strong"?
Plus two decompositions:
  E0 no-exit    the 52-week-high entry with NO exit rule at all (hold to the end)
  X0 exit-only  every eligible name entered on the first day it is in a ST(14,4) uptrend
                (no channel condition), same exit - the "trend-following without the
                breakout" book.
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

OPT = dict(uni='n100', L=252, ref='close', exit='ST_14_4', slots=20)
SHORT = [
    ('52W_OPT', dict(uni='n100', L=252, ref='close', exit='ST_14_4', slots=20)),
    ('52W_OPT_buf1', dict(uni='n100', L=252, ref='close', buffer=0.01, exit='ST_14_4',
                          slots=20)),
    ('52W_OPT_s15', dict(uni='n100', L=252, ref='close', exit='ST_14_4', slots=15)),
    ('52W_OPT_L189', dict(uni='n100', L=189, ref='close', exit='ST_14_4', slots=20)),
    ('52W_OPT_n500', dict(uni='n500', L=252, ref='close', exit='ST_14_4', slots=20)),
    ('52W_OPT_pit100', dict(uni='pit100', L=252, ref='close', exit='ST_14_4', slots=20)),
    ('52W_SpecA', dict(uni='n100', L=252, ref='close', exit='CC252', slots=20)),
    ('52W_noexit', dict(uni='n100', L=252, ref='close', exit='NONE', slots=20)),
]


def sim(P, trig, exitkey, slots, days, cost=15.0):
    cfg = dict(days=days, slots=slots, cost_bps=cost, gate=None, seed=None, tax=True)
    cfg.update(R.exit_cfg(P, exitkey))
    return B.simulate(P, trig, cfg)


def draw(P, trig, pool_mask, seed):
    rng = np.random.default_rng(seed)
    null = np.zeros_like(trig)
    nper = trig.sum(axis=1)
    for i in np.nonzero(nper)[0]:
        pool = np.nonzero(pool_mask[i])[0]
        if not len(pool):
            continue
        k = min(int(nper[i]), len(pool))
        null[i, rng.choice(pool, size=k, replace=False)] = True
    return null


def phase_nulls(P):
    path = RES / 'nulls3.csv'
    done = R.done_labels(path)
    days = P.days['full']
    trig = B.entry_signal(P, OPT['uni'], OPT['L'], ref=OPT['ref'])
    elig = P.eligible(OPT['uni'], OPT['L'])
    up = (P.ST['ST_14_4'] == 1)
    eshift = np.zeros_like(elig)
    eshift[1:] = elig[:-1]
    ushift = np.zeros_like(up)
    ushift[1:] = up[:-1]
    rs = P.RS252.copy()
    rs[~eshift] = np.nan
    med = np.nanmedian(np.where(np.isfinite(rs), rs, np.nan), axis=1)
    strong = np.zeros_like(elig)
    with np.errstate(invalid='ignore'):
        strong[1:] = (elig & (P.RS252 >= med[:, None]))[:-1]
    pools = {'N1_plain': eshift, 'N2_trend': eshift & ushift, 'N3_mom': strong,
             'N4_trendmom': eshift & ushift & strong}
    for nm, pool in pools.items():
        for s in range(1, 31):
            lab = '%s_%02d' % (nm, s)
            if lab in done:
                continue
            r = sim(P, draw(P, trig, pool, 70000 + s), OPT['exit'], OPT['slots'], days)
            m = B.metrics(r['nav'], r['dates'], r['trades'])
            row = dict(label=lab, phase='null3', uni='n100', entry_L=252,
                       entry_ref='close', exit=OPT['exit'], slots=20, gate='none',
                       cost_bps=15.0, seed=s, window='full', tax=True,
                       avg_invested=round(float(np.nanmean(r['invested'])), 3))
            row.update(m)
            R.append(path, row)
        sub = pd.read_csv(path)
        sub = sub[sub.label.str.startswith(nm)]
        print('  [%s] n=%d cagr med %.2f [%.2f..%.2f] calmar med %.3f [%.3f..%.3f] '
              'trades med %.0f' % (nm, len(sub), sub.cagr.median(), sub.cagr.min(),
                                   sub.cagr.max(), sub.calmar.median(), sub.calmar.min(),
                                   sub.calmar.max(), sub.trades.median()), flush=True)

    # X0: trend-entry-only book (no channel condition at all)
    if 'X0_trend_entry_only' not in done:
        trig0 = np.zeros_like(elig)
        with np.errstate(invalid='ignore'):
            fire = elig & up & ~np.r_[np.zeros((1, P.N), bool), up[:-1]]
        trig0[1:] = fire[:-1]
        r = sim(P, trig0, OPT['exit'], OPT['slots'], days)
        m = B.metrics(r['nav'], r['dates'], r['trades'])
        row = dict(label='X0_trend_entry_only', phase='null3', uni='n100', exit='ST_14_4',
                   slots=20, window='full', tax=True, cost_bps=15.0,
                   avg_invested=round(float(np.nanmean(r['invested'])), 3))
        row.update(m)
        R.append(path, row)
        print('  [X0_trend_entry_only] cagr=%.2f dd=%.2f calmar=%.3f trades=%d'
              % (m['cagr'], m['maxdd'], m['calmar'], m['trades']), flush=True)


def phase_short(P):
    path = RES / 'shortlist.csv'
    done = R.done_labels(path)
    days = P.days['full']
    navs, outl, peryear = {}, [], []
    idx = pd.to_datetime(P.dstr[days[0]:days[-1] + 1])
    for lab, kw in SHORT:
        trig = B.entry_signal(P, kw['uni'], kw['L'], ref=kw['ref'],
                              buffer=kw.get('buffer', 0.0))
        for cost in (0.0, 15.0, 30.0, 45.0):
            cl = '%s_c%d' % (lab, int(cost))
            if cl in done:
                continue
            r = sim(P, trig, kw['exit'], kw['slots'], days, cost=cost)
            m = B.metrics(r['nav'], r['dates'], r['trades'])
            row = dict(label=cl, phase='short', uni=kw['uni'], entry_L=kw['L'],
                       entry_ref=kw['ref'], buffer=kw.get('buffer', 0.0),
                       exit=kw['exit'], slots=kw['slots'], gate='none', cost_bps=cost,
                       window='full', tax=True, tax_paid=round(r['tax_paid']),
                       cost_paid=round(r['cost_paid']),
                       avg_invested=round(float(np.nanmean(r['invested'])), 3))
            row.update(m)
            R.append(path, row)
            if cost == 15.0:
                navs[lab] = r['nav']
                t = pd.DataFrame(r['trades'])
                t.to_csv(RES / ('trades_%s.csv' % lab), index=False)
                tot = t.pnl.sum()
                t10 = t.nlargest(10, 'pnl').pnl.sum()
                outl.append(dict(label=lab, n=len(t), total_pnl=round(tot),
                                 top10_share=round(100 * t10 / tot, 1) if tot else None,
                                 mean_ret=round(float(t.ret_pct.mean()), 3),
                                 mean_ret_cap50=round(float(t.ret_pct.clip(upper=50).mean()), 3),
                                 mean_ret_cap100=round(float(t.ret_pct.clip(upper=100).mean()), 3),
                                 win_rate=round(100 * float((t.pnl > 0).mean()), 1),
                                 avg_win=round(float(t[t.pnl > 0].ret_pct.mean()), 2),
                                 avg_loss=round(float(t[t.pnl <= 0].ret_pct.mean()), 2),
                                 worst_mae=round(float(t.mae_pct.min()), 1),
                                 med_mae=round(float(t.mae_pct.median()), 1),
                                 p05_mae=round(float(t.mae_pct.quantile(0.05)), 1),
                                 med_hold_d=int(t.days.median()),
                                 p95_hold_d=int(t.days.quantile(0.95))))
                s = pd.Series(r['nav'], index=idx)
                pk = s.cummax()
                dd = s / pk - 1.0
                for y in sorted(set(idx.year)):
                    j = np.flatnonzero(idx.year == y)
                    s0 = j[0] - 1 if j[0] > 0 else j[0]
                    peryear.append(dict(label=lab, year=int(y),
                                        ret=round(100 * (s.iloc[j[-1]] / s.iloc[s0] - 1), 2),
                                        dd=round(100 * float(dd.iloc[j].min()), 2)))
        print('  [short] %s done' % lab, flush=True)
    if navs:
        np.savez_compressed(RES / 'shortlist_navs.npz',
                            dates=np.array([str(d.date()) for d in idx]), **navs)
        pd.DataFrame(outl).to_csv(RES / 'shortlist_outliers.csv', index=False)
        pd.DataFrame(peryear).to_csv(RES / 'shortlist_peryear.csv', index=False)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    P = p172.Panel()
    steps = {'nulls': [phase_nulls], 'short': [phase_short],
             'all': [phase_nulls, phase_short]}[what]
    for fn in steps:
        print('=== %s ===' % fn.__name__, flush=True)
        fn(P)
    print('DONE %s in %.1f min' % (what, (time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
