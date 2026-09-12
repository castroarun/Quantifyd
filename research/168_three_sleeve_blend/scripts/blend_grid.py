# -*- coding: utf-8 -*-
"""research/168 step 3 - the three-sleeve weight grid, paired, with the cash null.

Sleeves, all at 5.2% post-tax idle cash, all after tax, all net of cost:
    TN      True North            12 rebalance-day offsets
    BA      Open Alpha . Base Age 30 seeds
    IPO     IPO-INC or IPO-A      30 seeds   (or CASH, the null, in the same weight)

Blend path p (p = 0..29) = IPO seed p+1, BA seed p+1, TN offset p % 12. Every comparison in
this study is PAIRED on p.

Grid: IPO weight {0,5,10,15,20,25,33}% x TN:BA split {25:75,33:67,50:50,67:33,75:25}
      x rebalance {monthly, quarterly, annual, never} x third sleeve {INC, A, CASH}
      = 420 cells, each on 30 paired paths. Run on TWO cost bases (see COST NOTE below).

COST NOTE. research/144's True North uses rt = 0.003 (15 bps a side); Open Alpha and IPO Base
both use 25 bps a side. The 'published' basis keeps each sleeve on its own published cost so the
figures tie to the live mpf report; the 'harmonised' basis puts True North on 25 bps a side too.
The verdict must hold on both or it is reported as basis-dependent.

Every drawdown - full period, per year, per window - is measured from the running peak of the
FULL curve, never from the window's own first bar (the r/154 convention error).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/168_three_sleeve_blend/results'
R163 = ROOT / 'research/163_mpf_cash_yield_harmonisation/results/cash052'

CASH_Y = 0.052
IPO_WS = [0, 5, 10, 15, 20, 25, 33]
RATIOS = [(25, 75), (33, 67), (50, 50), (67, 33), (75, 25)]
REBALS = ['monthly', 'quarterly', 'annual', 'never']
THIRDS = ['INC', 'A', 'CASH']
BASES = {'published': dict(tn=15, ba=25, ipo=25),
         'harmonised': dict(tn=25, ba=25, ipo=25)}
LADDER = {25: dict(tn=15, ba=25, ipo=25), 40: dict(tn=40, ba=40, ipo=40),
          60: dict(tn=60, ba=60, ipo=60)}
WINDOWS = {'2008': ('2008-01-01', '2008-12-31'),
           '2020H1': ('2020-01-01', '2020-06-30'),
           '2018': ('2018-01-01', '2018-12-31'),
           '2022H1': ('2022-01-01', '2022-06-30')}
SUBS = {'WA 2006-2015': ('2006-01-01', '2015-12-31'),
        'WB 2016-2026': ('2016-01-01', '2026-12-31')}


# ───────────────────────────────────────────────────────────── load + align
def load():
    tn = np.load(RES / 'tn_navs_cash052.npz', allow_pickle=True)
    ba = np.load(RES / 'ba_navs_cash052.npz', allow_pickle=True)
    ip = np.load(RES / 'ipo_navs_cash052.npz', allow_pickle=True)
    d_tn = pd.DatetimeIndex(pd.to_datetime(tn['dates']))
    d_ba = pd.DatetimeIndex(pd.to_datetime(ba['dates']))
    d_ip = pd.DatetimeIndex(pd.to_datetime(ip['dates']))
    idx = d_tn.intersection(d_ba).intersection(d_ip)
    print('sleeve spans : TN %s..%s (%d)  BA %s..%s (%d)  IPO %s..%s (%d)'
          % (d_tn[0].date(), d_tn[-1].date(), len(d_tn), d_ba[0].date(), d_ba[-1].date(),
             len(d_ba), d_ip[0].date(), d_ip[-1].date(), len(d_ip)))
    print('COMMON WINDOW: %s .. %s  (%d trading days, %.2f y)'
          % (idx[0].date(), idx[-1].date(), len(idx),
             (idx[-1] - idx[0]).days / 365.25), flush=True)

    def align(arr, src, n):
        df = pd.DataFrame(arr.T, index=src)
        out = df.reindex(idx).to_numpy(float).T
        if not np.isfinite(out).all():
            print('!! non-finite after align'); sys.exit(2)
        return out / out[:, :1]

    S = {}
    for bps in (15, 25, 40, 60):
        M = np.vstack([tn['off%d_%dbps' % (p % 12, bps)] for p in range(30)])
        S['TN_%d' % bps] = align(M, d_tn, 30)
    for bps in (25, 40, 60):
        S['BA_%d' % bps] = align(ba['ba_%dbps' % bps], d_ba, 30)
        for arm in ('INC', 'A'):
            S['%s_%d' % (arm, bps)] = align(ip['%s_%dbps_y52' % (arm, bps)], d_ip, 30)
    n = len(idx)
    step = (1.0 + CASH_Y) ** (1.0 / 252.0)
    cash = step ** np.arange(n)
    S['CASH_25'] = np.tile(cash, (30, 1))
    S['CASH_40'] = S['CASH_25']
    S['CASH_60'] = S['CASH_25']
    bench = pd.read_csv(R163 / 'full_period_after_tax_cash052.csv', index_col=0,
                        parse_dates=True)['NIFTYBEES (index)'].reindex(idx).ffill()
    bench = (bench / bench.iloc[0]).to_numpy(float)
    return idx, S, bench


def boundaries(idx, rebal):
    if rebal == 'never':
        return [0]
    code = {'monthly': 'M', 'quarterly': 'Q', 'annual': 'Y'}[rebal]
    per = idx.to_period(code)
    chg = np.r_[True, per[1:] != per[:-1]]
    return list(np.flatnonzero(chg))


def blend(navs, w, bnds):
    """navs: (K, P, T). w: (K,). Returns (P, T) blended NAVs, start 1.0.

    A rebalance at day b means: the target weights are restored at the CLOSE of day b-1, so
    day b's return is earned on the restored weights. Returns inside a period are measured
    relative to the sleeve NAVs at the period's base close (b-1), NEVER relative to day b
    itself -- doing the latter silently throws away the return of every rebalance day and
    manufactures a fake 'rebalance frequency effect' (caught here on 13-Sep-2026: it made
    monthly rebalancing look 1.8pp of CAGR worse than drift).
    """
    K, P, T = navs.shape
    assert abs(float(w.sum()) - 1.0) < 1e-12, 'weights must sum to 1'
    out = np.empty((P, T))
    out[:, 0] = 1.0
    bs = sorted(set([0] + [int(b) for b in bnds if b >= 1]))
    ends = bs[1:] + [T]
    for s, e in zip(bs, ends):
        base = max(s - 1, 0)
        d0 = max(s, 1)
        if d0 >= e:
            continue
        rel = navs[:, :, d0:e] / navs[:, :, base][:, :, None]
        mult = np.tensordot(w, rel, axes=(0, 0))          # (P, e-d0)
        out[:, d0:e] = out[:, base][:, None] * mult
    return out


def selftest(S, idx, bnd):
    """A pure sleeve must reproduce itself EXACTLY under every rebalance frequency."""
    for key in ('TN_15', 'BA_25', 'A_25'):
        for rb, b in bnd.items():
            got = blend(np.stack([S[key], S['BA_25'], S['CASH_25']]),
                        np.array([1.0, 0.0, 0.0]), b)
            mx = float(np.abs(got - S[key]).max())
            if mx > 1e-12:
                print('!! SELFTEST FAILED %s %s max abs %.3e' % (key, rb, mx))
                sys.exit(2)
    print('selftest OK: a 100%% sleeve reproduces itself under every rebalance frequency',
          flush=True)


# ───────────────────────────────────────────────────────────── metrics
def metrics(nav, idx):
    """nav (P,T) -> dict of per-path arrays."""
    P, T = nav.shape
    yrs = (idx[-1] - idx[0]).days / 365.25
    cagr = 100 * ((nav[:, -1] / nav[:, 0]) ** (1 / yrs) - 1)
    peak = np.maximum.accumulate(nav, axis=1)
    dd = nav / peak - 1.0
    mdd = 100 * dd.min(axis=1)
    out = dict(cagr=cagr, maxdd=mdd, calmar=cagr / np.abs(mdd))
    for nm, (a, b) in {**SUBS, **WINDOWS}.items():
        m = (idx >= a) & (idx <= b)
        if not m.any():
            continue
        j = np.flatnonzero(m)
        s0 = j[0] - 1 if j[0] > 0 else j[0]
        r = 100 * (nav[:, j[-1]] / nav[:, s0] - 1)
        wy = (idx[j[-1]] - idx[s0]).days / 365.25
        out[nm + '_ret'] = r
        out[nm + '_cagr'] = 100 * ((nav[:, j[-1]] / nav[:, s0]) ** (1 / wy) - 1)
        out[nm + '_dd'] = 100 * dd[:, m].min(axis=1)
    return out


def peryear(nav, idx):
    """single path (T,) -> {year: (ret%, intra-year dd% from the FULL curve peak)}"""
    s = pd.Series(nav, index=idx)
    peak = s.cummax()
    dd = s / peak - 1.0
    out = {}
    yrs = sorted(set(idx.year))
    for i, y in enumerate(yrs):
        m = idx.year == y
        j = np.flatnonzero(m)
        s0 = j[0] - 1 if j[0] > 0 else j[0]
        out[int(y)] = (round(100 * (s.iloc[j[-1]] / s.iloc[s0] - 1), 2),
                       round(100 * float(dd[m].min()), 2))
    return out


def med_path(m):
    return int(np.argsort(m['cagr'])[len(m['cagr']) // 2])


# ───────────────────────────────────────────────────────────── the grid
def main():
    t0 = time.time()
    idx, S, bench = load()
    bnd = {rb: boundaries(idx, rb) for rb in REBALS}
    for rb in REBALS:
        print('  %-10s %d rebalance boundaries' % (rb, len(bnd[rb])))
    selftest(S, idx, bnd)

    rows, percell = [], {}
    out_csv = RES / 'blend_grid.csv'
    if out_csv.exists():
        out_csv.unlink()
    for basis, bp in BASES.items():
        for third in THIRDS:
            for w_ipo in IPO_WS:
                for tnr, bar in RATIOS:
                    rem = (100 - w_ipo) / 100.0
                    w = np.array([rem * tnr / 100.0, rem * bar / 100.0, w_ipo / 100.0])
                    navs = np.stack([S['TN_%d' % bp['tn']], S['BA_%d' % bp['ba']],
                                     S['%s_%d' % (third, bp['ipo'])]])
                    for rb in REBALS:
                        b = blend(navs, w, bnd[rb])
                        m = metrics(b, idx)
                        key = '%s|%s|%d|%d:%d|%s' % (basis, third, w_ipo, tnr, bar, rb)
                        percell[key] = np.vstack([m['cagr'], m['maxdd'], m['calmar']])
                        r = dict(basis=basis, third=third, w_ipo=w_ipo, tn=tnr, ba=bar,
                                 rebal=rb)
                        for k in ('cagr', 'maxdd', 'calmar'):
                            r[k + '_med'] = round(float(np.median(m[k])), 3)
                        r['cagr_worst'] = round(float(m['cagr'].min()), 3)
                        r['maxdd_worst'] = round(float(m['maxdd'].min()), 3)
                        r['calmar_worst'] = round(float(m['calmar'].min()), 3)
                        for nm in list(SUBS) + list(WINDOWS):
                            r[nm + '_cagr'] = round(float(np.median(m[nm + '_cagr'])), 2)
                            r[nm + '_ret'] = round(float(np.median(m[nm + '_ret'])), 2)
                            r[nm + '_dd'] = round(float(np.median(m[nm + '_dd'])), 2)
                        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    np.savez_compressed(RES / 'blend_paths.npz',
                        keys=np.array(list(percell)), **{k.replace('|', '__'): v
                                                         for k, v in percell.items()})
    print('\n%d cells scored in %.0fs' % (len(df), time.time() - t0), flush=True)

    # ───────────────────── sleeve standalone table (median path of each ensemble)
    print('\n=== SLEEVE STANDALONE, 5.2%% idle cash, common window ===')
    sl = {}
    for nm, key in (('TN (15bps)', 'TN_15'), ('TN (25bps)', 'TN_25'),
                    ('OA BaseAge', 'BA_25'), ('IPO-INC', 'INC_25'), ('IPO-A', 'A_25'),
                    ('CASH 5.2%', 'CASH_25')):
        m = metrics(S[key], idx)
        k = med_path(m)
        sl[nm] = dict(cagr=round(float(np.median(m['cagr'])), 2),
                      cagr_worst=round(float(m['cagr'].min()), 2),
                      dd=round(float(np.median(m['maxdd'])), 2),
                      dd_worst=round(float(m['maxdd'].min()), 2),
                      calmar=round(float(np.median(m['calmar'])), 3),
                      med_path=k,
                      wa=round(float(np.median(m['WA 2006-2015_cagr'])), 2),
                      wb=round(float(np.median(m['WB 2016-2026_cagr'])), 2),
                      y2008=round(float(np.median(m['2008_ret'])), 2),
                      y2020=round(float(np.median(m['2020H1_ret'])), 2),
                      y2018=round(float(np.median(m['2018_ret'])), 2),
                      y2022=round(float(np.median(m['2022H1_ret'])), 2))
        print('%-12s CAGR %6.2f [worst %6.2f]  DD %7.2f [worst %7.2f]  Cal %5.3f   '
              'WA %6.2f WB %6.2f   2008 %+7.2f 2020H1 %+7.2f 2018 %+7.2f 2022H1 %+7.2f'
              % (nm, sl[nm]['cagr'], sl[nm]['cagr_worst'], sl[nm]['dd'], sl[nm]['dd_worst'],
                 sl[nm]['calmar'], sl[nm]['wa'], sl[nm]['wb'], sl[nm]['y2008'],
                 sl[nm]['y2020'], sl[nm]['y2018'], sl[nm]['y2022']), flush=True)
    bm = metrics(bench[None, :], idx)
    sl['NIFTYBEES'] = dict(cagr=round(float(bm['cagr'][0]), 2),
                           dd=round(float(bm['maxdd'][0]), 2),
                           calmar=round(float(bm['calmar'][0]), 3))
    print('%-12s CAGR %6.2f  DD %7.2f  Cal %5.3f'
          % ('NIFTYBEES', sl['NIFTYBEES']['cagr'], sl['NIFTYBEES']['dd'],
             sl['NIFTYBEES']['calmar']))

    # ───────────────────── correlations (median paths)
    print('\n=== CORRELATION of the sleeves (median path each), common window ===')
    cols = {'TN': S['TN_15'][med_path(metrics(S['TN_15'], idx))],
            'OA BaseAge': S['BA_25'][med_path(metrics(S['BA_25'], idx))],
            'IPO-INC': S['INC_25'][med_path(metrics(S['INC_25'], idx))],
            'IPO-A': S['A_25'][med_path(metrics(S['A_25'], idx))],
            'NIFTYBEES': bench}
    cdf = pd.DataFrame(cols, index=idx)
    for lbl, fr in (('daily', cdf.pct_change()),
                    ('weekly', cdf.resample('W').last().pct_change()),
                    ('monthly', cdf.resample('ME').last().pct_change())):
        print('--- %s' % lbl)
        print(fr.dropna().corr().round(3).to_string())
    corr_json = {lbl: fr.dropna().corr().round(4).to_dict()
                 for lbl, fr in (('daily', cdf.pct_change()),
                                 ('weekly', cdf.resample('W').last().pct_change()),
                                 ('monthly', cdf.resample('ME').last().pct_change()))}
    json.dump(corr_json, open(RES / 'correlations.json', 'w'), indent=1)

    # ───────────────────── baseline + best cells per basis
    report = {}
    for basis in BASES:
        d = df[df.basis == basis]
        base2 = d[(d.w_ipo == 0) & (d.third == 'INC')]
        print('\n\n################ BASIS = %s ################' % basis.upper())
        print('\n=== TWO-SLEEVE BASELINE (IPO weight 0) ===')
        print(base2[['tn', 'ba', 'rebal', 'cagr_med', 'maxdd_med', 'calmar_med',
                     'cagr_worst', 'maxdd_worst']].to_string(index=False))
        bb = base2.sort_values('calmar_med', ascending=False).iloc[0]
        print('best two-sleeve by median Calmar: TN%d:BA%d %s -> CAGR %.2f  DD %.2f  Cal %.3f'
              % (bb.tn, bb.ba, bb.rebal, bb.cagr_med, bb.maxdd_med, bb.calmar_med))

        best = {}
        for third in THIRDS:
            t = d[(d.third == third) & (d.w_ipo > 0)].sort_values('calmar_med',
                                                                  ascending=False)
            best[third] = t.iloc[0]
            print('\n=== TOP 10 CELLS, third = %s ===' % third)
            print(t.head(10)[['w_ipo', 'tn', 'ba', 'rebal', 'cagr_med', 'maxdd_med',
                              'calmar_med', 'cagr_worst', 'maxdd_worst',
                              'WA 2006-2015_cagr', 'WB 2016-2026_cagr', '2008_ret',
                              '2018_ret', '2022H1_ret']].to_string(index=False))

        def paths(basis, third, w, tnr, bar, rb):
            return percell['%s|%s|%d|%d:%d|%s' % (basis, third, w, tnr, bar, rb)]

        def paired(a, b, label):
            dc = a[0] - b[0]; dd = a[1] - b[1]; dk = a[2] - b[2]
            o = dict(label=label,
                     d_cagr_med=round(float(np.median(dc)), 3),
                     cagr_wins=int((dc > 0).sum()),
                     d_dd_med=round(float(np.median(dd)), 3),
                     dd_wins=int((dd > 0).sum()),
                     d_calmar_med=round(float(np.median(dk)), 4),
                     calmar_wins=int((dk > 0).sum()),
                     n=len(dc))
            print('%-58s dCAGR %+7.3f pp (wins %2d/%d)  dDD %+7.3f pp (shallower %2d/%d)  '
                  'dCalmar %+7.4f (wins %2d/%d)'
                  % (label, o['d_cagr_med'], o['cagr_wins'], o['n'], o['d_dd_med'],
                     o['dd_wins'], o['n'], o['d_calmar_med'], o['calmar_wins'], o['n']),
                  flush=True)
            return o

        pr = []
        print('\n=== PAIRED: A vs INC at the SAME weight / ratio / rebalance ===')
        for rb in REBALS:
            for w in IPO_WS[1:]:
                for tnr, bar in RATIOS:
                    if (tnr, bar) != (50, 50):
                        continue
                    pr.append(paired(paths(basis, 'A', w, tnr, bar, rb),
                                     paths(basis, 'INC', w, tnr, bar, rb),
                                     'A minus INC   w=%d%%  TN%d:BA%d  %s'
                                     % (w, tnr, bar, rb)))
        print('\n=== PAIRED: A vs INC, FULL weight x ratio grid, monthly rebalance ===')
        for w in IPO_WS[1:]:
            for tnr, bar in RATIOS:
                pr.append(paired(paths(basis, 'A', w, tnr, bar, 'monthly'),
                                 paths(basis, 'INC', w, tnr, bar, 'monthly'),
                                 'A minus INC   w=%d%%  TN%d:BA%d  monthly' % (w, tnr, bar)))

        print('\n=== PAIRED: each IPO arm vs the TWO-SLEEVE baseline (same ratio+rebal) ===')
        for third in ('INC', 'A'):
            for rb in REBALS:
                for w in IPO_WS[1:]:
                    for tnr, bar in RATIOS:
                        if (tnr, bar) != (50, 50):
                            continue
                        pr.append(paired(paths(basis, third, w, tnr, bar, rb),
                                         paths(basis, 'INC', 0, tnr, bar, rb),
                                         '%s minus 2-sleeve  w=%d%%  TN%d:BA%d  %s'
                                         % (third, w, tnr, bar, rb)))

        print('\n=== PAIRED: THE CASH NULL - each IPO arm vs CASH in the same weight ===')
        for third in ('INC', 'A'):
            for rb in REBALS:
                for w in IPO_WS[1:]:
                    for tnr, bar in RATIOS:
                        if (tnr, bar) != (50, 50):
                            continue
                        pr.append(paired(paths(basis, third, w, tnr, bar, rb),
                                         paths(basis, 'CASH', w, tnr, bar, rb),
                                         '%s minus CASH     w=%d%%  TN%d:BA%d  %s'
                                         % (third, w, tnr, bar, rb)))

        print('\n=== PAIRED: best A cell vs best INC cell (each at its OWN optimum) ===')
        ba_, bi_ = best['A'], best['INC']
        pr.append(paired(paths(basis, 'A', int(ba_.w_ipo), int(ba_.tn), int(ba_.ba), ba_.rebal),
                         paths(basis, 'INC', int(bi_.w_ipo), int(bi_.tn), int(bi_.ba),
                               bi_.rebal),
                         'bestA(w%d %d:%d %s) minus bestINC(w%d %d:%d %s)'
                         % (ba_.w_ipo, ba_.tn, ba_.ba, ba_.rebal, bi_.w_ipo, bi_.tn, bi_.ba,
                            bi_.rebal)))
        pd.DataFrame(pr).to_csv(RES / ('paired_%s.csv' % basis), index=False)
        report[basis] = dict(best_two_sleeve=bb.to_dict(),
                             best={k: v.to_dict() for k, v in best.items()})

    # ───────────────────── per-year house table at the published basis
    print('\n\n=== PER-YEAR HOUSE TABLE (published basis, median path of each column) ===')
    d = df[df.basis == 'published']
    b2 = d[(d.w_ipo == 0) & (d.third == 'INC')].sort_values('calmar_med',
                                                            ascending=False).iloc[0]
    RB, TNR, BAR = b2.rebal, int(b2.tn), int(b2.ba)
    picks = {}
    navs2 = np.stack([S['TN_15'], S['BA_25'], S['INC_25']])
    rem = 1.0
    w2 = np.array([rem * TNR / 100, rem * BAR / 100, 0.0])
    picks['TN+OA 2-sleeve'] = blend(navs2, w2, bnd[RB])
    for third in ('INC', 'A', 'CASH'):
        t = d[(d.third == third) & (d.w_ipo > 0)].sort_values('calmar_med',
                                                              ascending=False).iloc[0]
        w = int(t.w_ipo); tnr, bar = int(t.tn), int(t.ba)
        r = (100 - w) / 100.0
        ww = np.array([r * tnr / 100, r * bar / 100, w / 100.0])
        nv = np.stack([S['TN_15'], S['BA_25'], S['%s_25' % third]])
        picks['+%s %d%% (%d:%d %s)' % (third, w, tnr, bar, t.rebal)] = blend(nv, ww,
                                                                            bnd[t.rebal])
    py = {}
    for nm, key in (('TN', 'TN_15'), ('OA BaseAge', 'BA_25'), ('IPO-INC', 'INC_25'),
                    ('IPO-A', 'A_25')):
        m = metrics(S[key], idx)
        k = med_path(m)
        py[nm] = dict(years=peryear(S[key][k], idx),
                      cagr=round(float(np.median(m['cagr'])), 2),
                      dd=round(float(np.median(m['maxdd'])), 2),
                      calmar=round(float(np.median(m['calmar'])), 3))
    for nm, nv in picks.items():
        m = metrics(nv, idx)
        k = med_path(m)
        py[nm] = dict(years=peryear(nv[k], idx),
                      cagr=round(float(np.median(m['cagr'])), 2),
                      dd=round(float(np.median(m['maxdd'])), 2),
                      calmar=round(float(np.median(m['calmar'])), 3))
    py['NIFTYBEES'] = dict(years=peryear(bench, idx), cagr=sl['NIFTYBEES']['cagr'],
                           dd=sl['NIFTYBEES']['dd'], calmar=sl['NIFTYBEES']['calmar'])
    json.dump(dict(peryear=py, sleeves=sl, report=report,
                   common_window=[str(idx[0].date()), str(idx[-1].date())]),
              open(RES / 'peryear.json', 'w'), indent=1)
    names = [n for n in py]
    yrs = sorted(py['TN']['years'])
    hdr = 'year  ' + ''.join('%26s' % n[:25] for n in names)
    print(hdr)
    for y in yrs:
        line = '%4d  ' % y
        for n in names:
            v = py[n]['years'].get(y)
            line += ('%+9.1f (%+6.1f)     ' % v) if v else ' ' * 26
        print(line)
    print('CAGR  ' + ''.join('%26s' % ('%.2f / %.1f / %.3f' % (py[n]['cagr'], py[n]['dd'],
                                                               py[n]['calmar']))
                             for n in names))

    # ───────────────────── cost ladder on the chosen blends
    print('\n=== COST LADDER 25 / 40 / 60 bps a side, on the blends chosen above ===')
    lad = []
    for label, spec in (('2-sleeve TN%d:BA%d %s' % (TNR, BAR, RB),
                         (0, TNR, BAR, RB, 'INC')),):
        pass
    chosen = [('TN+OA 2-sleeve', 0, TNR, BAR, RB, 'INC')]
    for third in ('INC', 'A', 'CASH'):
        t = d[(d.third == third) & (d.w_ipo > 0)].sort_values('calmar_med',
                                                              ascending=False).iloc[0]
        chosen.append(('+%s %d%%' % (third, int(t.w_ipo)), int(t.w_ipo), int(t.tn),
                       int(t.ba), t.rebal, third))
    for label, w, tnr, bar, rb, third in chosen:
        line = '%-22s' % label
        for bps, bp in LADDER.items():
            r = (100 - w) / 100.0
            ww = np.array([r * tnr / 100, r * bar / 100, w / 100.0])
            nv = np.stack([S['TN_%d' % bp['tn']], S['BA_%d' % bp['ba']],
                           S['%s_%d' % (third, bp['ipo'])]])
            m = metrics(blend(nv, ww, bnd[rb]), idx)
            line += '  %3dbps: %6.2f%% / %7.2f%% / %5.3f' % (
                bps, np.median(m['cagr']), np.median(m['maxdd']), np.median(m['calmar']))
            lad.append(dict(blend=label, cost_bps=bps,
                            cagr=round(float(np.median(m['cagr'])), 2),
                            maxdd=round(float(np.median(m['maxdd'])), 2),
                            calmar=round(float(np.median(m['calmar'])), 3)))
        print(line, flush=True)
    pd.DataFrame(lad).to_csv(RES / 'cost_ladder.csv', index=False)
    print('\nblend_grid done in %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
