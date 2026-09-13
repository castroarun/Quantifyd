# -*- coding: utf-8 -*-
"""research/169 - stages. Usage: run169.py <repro|equiv|transplant|age|refit|costs|blend|proxy>

All figures after tax (20% STCG / 12.5% LTCG, FY loss netting), 8 slots @ 18.75%, Rs 10L,
next-day buy-stop at the pivot filled max(pivot, open) only if the high reached it, stop 10% /
target 25% / trail SMA-50 on the close, NIFTYBEES < SMA-150 entry gate, 30 seeds.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/169_ipo_rules_universe_transplant/scripts'))
import xpanel as xp                                                   # noqa: E402

RES = xp.RES
R167 = ROOT / 'research/167_ipo_base_honest_reopt'
R168 = ROOT / 'research/168_three_sleeve_blend'
R163 = ROOT / 'research/163_mpf_cash_yield_harmonisation/results/cash052'
TOL = 0.15


def append(path, row):
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def done_labels(path):
    return set(pd.read_csv(path).label) if path.exists() else set()


def passes(r):
    return bool(r['w2_edge'] >= 1.0 and r['w2_wins'] >= 25 and r['wa_wins'] >= 25
                and r['wb_wins'] >= 25)


def cell(P, label, uni, age, L=25, trail=50, cost=0.0025, with_null=True,
         windows=('w2', 'wa', 'wb'), cash=0.052, min_bars=25):
    t = time.time()
    trig, pvn, lon, base = xp.build_signals(P, uni, age, L=L, min_bars=min_bars)
    real = xp.run(P, trig, pvn, lon, trail=trail, windows=windows, cost=cost,
                  cash_yield=cash, keep=True)
    null = None
    if with_null:
        nt = xp.build_null(P, trig, pvn, base)
        null = xp.run(P, nt, pvn, lon, trail=trail, windows=windows, cost=cost,
                      cash_yield=cash)
    row = dict(label=label, universe=uni, age=age, L=L, trail=trail,
               cost_bps=int(round(cost * 1e4)), cash_y=cash, min_bars=min_bars,
               n_signals_w2=int(trig[xp.np.asarray(P.days['w2'])].sum()))
    row.update(xp.summarize(real, cost, null))
    row.update(xp.trade_diag(P, real))
    row['secs'] = round(time.time() - t, 0)
    xp.save_navs(label, real)
    e = (' | null %6.2f edge %+5.2f [%+.2f..%+.2f] wins w2 %d wa %d wb %d %s'
         % (row['w2_null_cagr'], row['w2_edge'], row['w2_edge_lo'], row['w2_edge_hi'],
            row['w2_wins'], row.get('wa_wins', -1), row.get('wb_wins', -1),
            'PASS' if 'wa_wins' in row and passes(row) else '')) if with_null else ''
    print('%-34s sig %6d | CAGR %6.2f [%5.2f..%5.2f] DD %6.2f (w %6.2f) Cal %5.3f inv %4.1f '
          'tpy %5.1f win %4.1f net %5.2f | wa %6.2f wb %6.2f%s | young %s%% %ds'
          % (label, row['n_signals_w2'], row['w2_cagr'], row['w2_cagr_lo'], row['w2_cagr_hi'],
             row['w2_dd'], row['w2_dd_worst'], row['w2_calmar'], row['w2_inv'], row['w2_tpy'],
             row['w2_win'], row['w2_netexp'], row.get('wa_cagr', np.nan),
             row.get('wb_cagr', np.nan), e, row.get('young6m_share_pct'), row['secs']),
          flush=True)
    return row


# ───────────────────────────────────────────── S0 reproduction with r/167's own engine
def repro():
    sys.path.insert(0, str(R167 / 'scripts'))
    import ipo_honest as ih
    ctx, irr = ih.load_ctx(clean=True)
    nb = ctx.close.get('NIFTYBEES').dropna()
    g150 = ((nb < nb.rolling(150).mean()).shift(1).reindex(ctx.dates)
            .ffill().fillna(False).to_numpy(bool))
    cfg = {**ih.INCUMBENT, 'trail': 50, 'stop': 0.10, 'target': 0.25,
           'gate': 'custom', 'weak_series': g150}
    setup, piv, lo0 = ih.build_setup(ctx, cfg)
    trig, lvl, lo, fc = ih.apply_mechanic(ctx, setup, piv, lo0, 'nextday_pivot')
    out, ok = {}, True
    pub = pd.read_csv(R167 / 'results/stage9_adoption.csv').set_index('spec') \
        .loc['A_trail50_sl10_tp25_sma150']
    npz = np.load(R168 / 'results/ipo_navs_cash052.npz', allow_pickle=True)
    for y in (0.05, 0.052):
        o, kept = ih.run_cell(ctx, irr, cfg, trig=trig, level=lvl, lo=lo, keep=True,
                              windows=('w2',), fill_close=fc, cash_yield=y)
        out[f'real_y{y}'] = {k: o[k] for k in ('w2_cagr', 'w2_cagr_lo', 'w2_dd', 'w2_dd_worst',
                                               'w2_calmar', 'w2_inv', 'w2_tpy', 'w2_win')}
        if y == 0.05:
            dc = o['w2_cagr'] - float(pub.cagr)
            ddd = o['w2_dd'] - float(pub.dd)
            out['gate_vs_r167_stage9'] = dict(published=[float(pub.cagr), float(pub.dd)],
                                              rerun=[o['w2_cagr'], o['w2_dd']],
                                              delta_pp=[round(dc, 3), round(ddd, 3)])
            ok &= abs(dc) <= TOL and abs(ddd) <= TOL
            kR = kept
        else:
            M = np.vstack([(s / s.iloc[0]).to_numpy(float) for s in kept['navs']])
            ref = npz['A_25bps_y52']
            mx = float(np.abs(M - ref).max()) if M.shape == ref.shape else None
            out['per_seed_vs_r168_A_25bps_y52'] = dict(shape=list(M.shape),
                                                       ref_shape=list(ref.shape),
                                                       max_abs_diff=mx)
            ok &= mx is not None and mx < 1e-9
    # the null, r/167 stage 7/8 construction, at 5.0%
    pvn = ih._shift(piv)
    young = ((ctx.AGE > 0) & (ctx.AGE <= cfg['max_age_m'] * 30.44)
             & (ctx.BARS >= cfg['min_bars']) & ctx.ELIG)
    with np.errstate(invalid='ignore'):
        reach = ctx.H >= pvn
    ysh = ih._shift(young)
    rng = np.random.default_rng(20260912)
    nper = trig.sum(axis=1)
    null = np.zeros_like(trig)
    for i in np.nonzero(nper)[0]:
        pool = np.nonzero(ysh[i] & reach[i] & np.isfinite(pvn[i]))[0]
        if len(pool):
            null[i, rng.choice(pool, size=min(int(nper[i]), len(pool)), replace=False)] = True
    oN, kN = ih.run_cell(ctx, irr, cfg, trig=null, level=pvn, lo=lo, keep=True,
                         windows=('w2',), fill_close=False, cash_yield=0.05)
    d = kR['stats'].cagr.values - kN['stats'].cagr.values
    s8 = pd.read_csv(R167 / 'results/stage8_null_axis.csv')
    s8 = s8[(s8.stop == 0.1) & (s8.trail == 50)].iloc[0]
    out['null_gate'] = dict(published=dict(null=float(s8.null), edge=float(s8.paired_edge),
                                           wins=int(s8.real_wins)),
                            rerun=dict(null=oN['w2_cagr'], edge=round(float(np.median(d)), 2),
                                       wins=int((d > 0).sum())))
    ok &= abs(float(np.median(d)) - float(s8.paired_edge)) <= TOL
    out['PASSED'] = bool(ok)
    json.dump(out, open(RES / 's0_repro.json', 'w'), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    if not ok:
        print('!! REPRODUCTION FAILED - stop.', flush=True)
        sys.exit(2)
    print('REPRODUCTION PASSED', flush=True)


# ───────────────────────────────────────────── S1 panel equivalence
def equiv():
    path = RES / 's1_equiv.csv'
    P = xp.Panel()
    steps = [('E1_r167like', dict(adjust=False, robust=False, drop_phantom=False), 25),
             ('E2_plus_robust', dict(adjust=False, robust=True, drop_phantom=False), 25),
             ('E3_plus_phantom_drop', dict(adjust=False, robust=True, drop_phantom=True), 25),
             ('E4_plus_split_adjust', dict(adjust=True, robust=True, drop_phantom=True), 25),
             ('E5_default_minbars60', dict(adjust=True, robust=True, drop_phantom=True), 60)]
    done = done_labels(path)
    built = None
    for label, flags, mb in steps:
        if label in done:
            continue
        if built != flags:
            P.build(**flags)
            built = flags
        r = cell(P, label, 'all', 'le6', with_null=False, windows=('w2',), cash=0.05,
                 min_bars=mb)
        r.update({k: v for k, v in flags.items()})
        append(path, r)
        if label == 'E1_r167like' and abs(r['w2_cagr'] - 21.80) > TOL:
            print('!! E1 does not reproduce r/167 21.80 (got %.2f) - panel bug, stop'
                  % r['w2_cagr'], flush=True)
            sys.exit(2)
    if hasattr(P, 'events') and len(P.events):
        P.events.to_csv(RES / 's1_split_events.csv', index=False)
    print(pd.read_csv(path)[['label', 'w2_cagr', 'w2_cagr_lo', 'w2_dd', 'w2_calmar', 'w2_n',
                             'w2_tpy', 'w2_inv']].to_string(index=False), flush=True)


def default_panel():
    return xp.Panel().build(adjust=True, robust=True, drop_phantom=True)


# ───────────────────────────────────────────── S2 / S3
def transplant(P=None):
    """min_bars = 25 is the VALIDATED Spec A (r/167 INCUMBENT). The live book runs 60; that
    variant is carried as its own labelled cell (S1 E5 found it worth ~10pp less)."""
    path = RES / 's2_transplant.csv'
    cells = [('all__le6', 'all', 'le6', 25), ('all__le6__mb60_livebook', 'all', 'le6', 60)]
    for u in xp.UNIVERSES:
        for a in ('none', 'gt6'):
            cells.append((f'{u}__{a}', u, a, 25))
    done = done_labels(path)
    todo = [c for c in cells if c[0] not in done]
    print('S2: %d of %d cells to run' % (len(todo), len(cells)), flush=True)
    if not todo:
        return P
    P = P or default_panel()
    for i, (label, u, a, mb) in enumerate(todo, 1):
        append(path, cell(P, label, u, a, min_bars=mb))
        print('  [S2 %d/%d done]' % (i, len(todo)), flush=True)
    return P


def age(P=None):
    path = RES / 's3_age.csv'
    cells = [(f'all__{a}', 'all', a) for a in ('le12', 'le24', 'gt24', 'vet_any')]
    done = done_labels(path)
    todo = [c for c in cells if c[0] not in done]
    print('S3: %d of %d cells to run' % (len(todo), len(cells)), flush=True)
    if not todo:
        return P
    P = P or default_panel()
    for i, (label, u, a) in enumerate(todo, 1):
        append(path, cell(P, label, u, a))
        print('  [S3 %d/%d done]' % (i, len(todo)), flush=True)
    return P


def grid():
    P = transplant()
    P = age(P)
    proxy(P)


def survivors():
    rows = []
    for f in ('s2_transplant.csv', 's3_age.csv'):
        if (RES / f).exists():
            rows.append(pd.read_csv(RES / f))
    df = pd.concat(rows, ignore_index=True)
    df = df[~df.label.str.startswith('all__le6')]
    ok = df[df.apply(passes, axis=1)].sort_values('w2_edge', ascending=False)
    return df, ok


def refit():
    df, ok = survivors()
    path = RES / 's4_refit.csv'
    carried = list(ok.head(4).itertuples(index=False))
    print('S4: %d cells pass the pre-registered edge test; carrying %d: %s'
          % (len(ok), len(carried), [c.label for c in carried]), flush=True)
    if not carried:
        json.dump(dict(survivors=[], note='nothing passed the null - refit skipped'),
                  open(RES / 's4_refit_skipped.json', 'w'), indent=1)
        return
    done = done_labels(path)
    P = default_panel()
    for c in carried:
        for tr in xp.TRAILS:
            for L in xp.LS:
                if (tr, L) == (50, 25):
                    continue
                label = f'{c.label}__tr{tr}_L{L}'
                if label in done:
                    continue
                append(path, cell(P, label, c.universe, c.age, L=L, trail=tr))


def carried_specs():
    df, ok = survivors()
    specs = []
    if len(ok):
        for c in ok.head(4).itertuples(index=False):
            specs.append(dict(label=c.label, universe=c.universe, age=c.age, L=25, trail=50,
                              passed=True))
            p4 = RES / 's4_refit.csv'
            if p4.exists():
                r4 = pd.read_csv(p4)
                r4 = r4[r4.label.str.startswith(c.label + '__')]
                r4 = r4[r4.apply(passes, axis=1)].sort_values('w2_edge', ascending=False)
                if len(r4):
                    b = r4.iloc[0]
                    specs.append(dict(label=b.label, universe=b.universe, age=b.age,
                                      L=int(b.L), trail=int(b.trail), passed=True))
    else:
        for c in df.sort_values('w2_edge', ascending=False).head(2).itertuples(index=False):
            specs.append(dict(label=c.label, universe=c.universe, age=c.age, L=25, trail=50,
                              passed=False))
    # always carry the best-CAGR seasoned transplant for the blend, labelled, if not present
    big = df[df.age.isin(['none', 'gt6'])].sort_values('w2_calmar', ascending=False).head(1)
    for c in big.itertuples(index=False):
        if c.label not in [s['label'] for s in specs]:
            specs.append(dict(label=c.label, universe=c.universe, age=c.age, L=25, trail=50,
                              passed=bool(passes(c._asdict()))))
    specs.append(dict(label='all__le6', universe='all', age='le6', L=25, trail=50,
                      passed=None))
    specs.append(dict(label='all__le6__mb60_livebook', universe='all', age='le6', L=25,
                      trail=50, passed=None, min_bars=60))
    json.dump(specs, open(RES / 'carried.json', 'w'), indent=1)
    return specs


def costs():
    specs = carried_specs()
    print('carried:', json.dumps(specs), flush=True)
    path = RES / 's5_costs.csv'
    done = done_labels(path)
    P = None
    for s in specs:
        for c in (0.0040, 0.0060):
            label = '%s__c%d' % (s['label'], int(c * 1e4))
            if label in done:
                continue
            if P is None:
                P = default_panel()
            append(path, cell(P, label, s['universe'], s['age'], L=s['L'], trail=s['trail'],
                              cost=c, with_null=False, windows=('w2',),
                              min_bars=s.get('min_bars', 25)))


# ───────────────────────────────────────────── S6 portfolio fit
def blend():
    sys.path.insert(0, str(R168 / 'scripts'))
    import blend_grid as bg
    specs = json.load(open(RES / 'carried.json'))
    tn = np.load(R168 / 'results/tn_navs_cash052.npz', allow_pickle=True)
    ba = np.load(R168 / 'results/ba_navs_cash052.npz', allow_pickle=True)
    ip = np.load(R168 / 'results/ipo_navs_cash052.npz', allow_pickle=True)
    src = {}
    for bps in (15, 40, 60):
        src['TN_%d' % (25 if bps == 15 else bps)] = (
            tn['dates'], np.vstack([tn['off%d_%dbps' % (p % 12, bps)] for p in range(30)]))
    for bps in (25, 40, 60):
        src['BA_%d' % bps] = (ba['dates'], ba['ba_%dbps' % bps])
        src['A_%d' % bps] = (ip['dates'], ip['A_%dbps_y52' % bps])
    for s in specs:
        for bps, suf in ((25, ''), (40, '__c40'), (60, '__c60')):
            f = RES / 'navs' / (s['label'] + suf + '.npz')
            if f.exists():
                z = np.load(f)
                src['X:%s_%d' % (s['label'], bps)] = (z['dates'], z['navs'].astype(float))
    idx = None
    for k, (d, _) in src.items():
        di = pd.DatetimeIndex(pd.to_datetime(d))
        idx = di if idx is None else idx.intersection(di)
    print('common window %s .. %s (%d days)' % (idx[0].date(), idx[-1].date(), len(idx)),
          flush=True)
    S = {}
    for k, (d, M) in src.items():
        df = pd.DataFrame(np.asarray(M, float).T, index=pd.DatetimeIndex(pd.to_datetime(d)))
        out = df.reindex(idx).to_numpy(float).T
        if not np.isfinite(out).all():
            print('!! non-finite after align', k)
            sys.exit(2)
        S[k] = out / out[:, :1]
    T = len(idx)
    S['CASH'] = np.tile((1.052 ** (1 / 252.0)) ** np.arange(T), (30, 1))
    bench = pd.read_csv(R163 / 'full_period_after_tax_cash052.csv', index_col=0,
                        parse_dates=True)['NIFTYBEES (index)'].reindex(idx).ffill()
    bench = (bench / bench.iloc[0]).to_numpy(float)
    bnd = bg.boundaries(idx, 'monthly')

    def mk(keys, w, bps=25):
        return bg.blend(np.stack([S[k.format(b=bps)] for k in keys]), np.array(w), bnd)

    def med(m, k):
        return round(float(np.median(m[k])), 3)

    def paired(a, b):
        ma, mb = bg.metrics(a, idx), bg.metrics(b, idx)
        dc, dd, dk = ma['cagr'] - mb['cagr'], ma['maxdd'] - mb['maxdd'], ma['calmar'] - mb['calmar']
        bar = (dk >= 0.10) | ((dc >= 2.0) & (dd >= 0))
        return dict(cagr=med(ma, 'cagr'), dd=med(ma, 'maxdd'), calmar=med(ma, 'calmar'),
                    base_cagr=med(mb, 'cagr'), base_dd=med(mb, 'maxdd'),
                    base_calmar=med(mb, 'calmar'),
                    d_cagr=round(float(np.median(dc)), 3), cagr_wins=int((dc > 0).sum()),
                    d_dd=round(float(np.median(dd)), 3), dd_wins=int((dd > 0).sum()),
                    d_calmar=round(float(np.median(dk)), 4), calmar_wins=int((dk > 0).sum()),
                    bar_paths=int(bar.sum()),
                    wa_d_cagr=round(float(np.median(ma['WA 2006-2015_cagr']
                                                    - mb['WA 2006-2015_cagr'])), 2),
                    wb_d_cagr=round(float(np.median(ma['WB 2016-2026_cagr']
                                                    - mb['WB 2016-2026_cagr'])), 2),
                    wa_wins=int(((ma['WA 2006-2015_cagr'] - mb['WA 2006-2015_cagr']) > 0).sum()),
                    wb_wins=int(((ma['WB 2016-2026_cagr'] - mb['WB 2016-2026_cagr']) > 0).sum()))

    def cash_match(blend_nav, make_cash):
        target = float(np.median(bg.metrics(blend_nav, idx)['maxdd']))
        best = None
        for c in range(0, 61):
            cb = make_cash(c / 100.0)
            md = float(np.median(bg.metrics(cb, idx)['maxdd']))
            if best is None or abs(md - target) < best[0]:
                best = (abs(md - target), c, cb, md)
        ma, mc = bg.metrics(blend_nav, idx), bg.metrics(best[2], idx)
        d = ma['cagr'] - mc['cagr']
        return dict(cash_w=best[1], cash_cagr=med(mc, 'cagr'), cash_dd=round(best[3], 2),
                    d_cagr_vs_cash=round(float(np.median(d)), 3), wins_vs_cash=int((d > 0).sum()))

    B3 = mk(['TN_{b}', 'BA_{b}', 'A_{b}'], [0.375, 0.375, 0.25])
    B2 = mk(['TN_{b}', 'BA_{b}'], [0.5, 0.5])
    report = dict(window=[str(idx[0].date()), str(idx[-1].date()), T],
                  B3=paired(B3, B2), tests={})
    # correlations on median paths
    def mpath(M):
        m = bg.metrics(M, idx)
        return M[int(np.argsort(m['cagr'])[len(m['cagr']) // 2])]
    cols = {'TN': mpath(S['TN_25']), 'OA_BaseAge': mpath(S['BA_25']), 'IPO_A': mpath(S['A_25'])}
    for s in specs:
        k = 'X:%s_25' % s['label']
        if k in S:
            cols[s['label']] = mpath(S[k])
    cdf = pd.DataFrame(cols, index=idx)
    corr = {lbl: fr.dropna().corr().round(3).to_dict() for lbl, fr in
            (('weekly', cdf.resample('W').last().pct_change()),
             ('monthly', cdf.resample('ME').last().pct_change()))}
    report['correlations'] = corr
    print('monthly corr:\n', pd.DataFrame(corr['monthly']).to_string(), flush=True)
    for s in specs:
        lab = s['label']
        k = 'X:%s_{b}' % lab
        if 'X:%s_25' % lab not in S:
            continue
        t = dict(spec=s)
        RX = mk(['TN_{b}', 'BA_{b}', k], [0.375, 0.375, 0.25])
        t['replace_ipo_at_25'] = paired(RX, B3)
        t['replace_ipo_at_25'].update(cash_match(
            RX, lambda c: mk(['TN_{b}', 'BA_{b}', 'CASH'], [0.5 * (1 - c), 0.5 * (1 - c), c])))
        t['fourth'] = {}
        for w in (10, 15, 20, 25):
            ww = w / 100.0
            F = mk(['TN_{b}', 'BA_{b}', 'A_{b}', k],
                   [0.375 * (1 - ww), 0.375 * (1 - ww), 0.25 * (1 - ww), ww])
            r = paired(F, B3)
            r.update(cash_match(F, lambda c: mk(['TN_{b}', 'BA_{b}', 'A_{b}', 'CASH'],
                                                [0.375 * (1 - c), 0.375 * (1 - c),
                                                 0.25 * (1 - c), c])))
            t['fourth'][w] = r
        mcor = corr['monthly'][lab]['OA_BaseAge']
        tests = [('replace', t['replace_ipo_at_25'])] + [('fourth_%d' % w, v)
                                                         for w, v in t['fourth'].items()]
        t['sleeve_bar'] = {nm: bool(bool(s['passed']) and v['bar_paths'] >= 20
                                    and v['wins_vs_cash'] >= 25 and mcor < 0.60)
                           for nm, v in tests}
        # cost ladder on replace and fourth-15
        lad = {}
        for bps in (25, 40, 60):
            if 'X:%s_%d' % (lab, bps) not in S:
                continue
            b3 = mk(['TN_{b}', 'BA_{b}', 'A_{b}'], [0.375, 0.375, 0.25], bps)
            rx = mk(['TN_{b}', 'BA_{b}', k], [0.375, 0.375, 0.25], bps)
            m3, mr = bg.metrics(b3, idx), bg.metrics(rx, idx)
            lad[bps] = dict(B3=[med(m3, 'cagr'), med(m3, 'maxdd'), med(m3, 'calmar')],
                            replace=[med(mr, 'cagr'), med(mr, 'maxdd'), med(mr, 'calmar')])
        t['cost_ladder'] = lad
        report['tests'][lab] = t
        rp = t['replace_ipo_at_25']
        print('%-28s corrOA %.2f | REPLACE IPO@25: CAGR %.2f DD %.2f Cal %.3f  dCAGR %+.2f (%d/30) '
              'dCal %+.3f (%d/30) bar %d/30 vsCash %+.2f (%d/30)'
              % (lab, mcor, rp['cagr'], rp['dd'], rp['calmar'], rp['d_cagr'], rp['cagr_wins'],
                 rp['d_calmar'], rp['calmar_wins'], rp['bar_paths'], rp['d_cagr_vs_cash'],
                 rp['wins_vs_cash']), flush=True)
        for w, v in t['fourth'].items():
            print('%-28s   FOURTH @%d%%: CAGR %.2f DD %.2f Cal %.3f  dCAGR %+.2f (%d/30) dCal %+.3f '
                  '(%d/30) bar %d/30 vsCash %+.2f (%d/30)'
                  % ('', w, v['cagr'], v['dd'], v['calmar'], v['d_cagr'], v['cagr_wins'],
                     v['d_calmar'], v['calmar_wins'], v['bar_paths'], v['d_cagr_vs_cash'],
                     v['wins_vs_cash']), flush=True)
    json.dump(report, open(RES / 's6_blend.json', 'w'), indent=1, default=str)

    # per-year house table
    colsets = [('TN', S['TN_25']), ('OA BaseAge', S['BA_25']), ('IPO-A', S['A_25'])]
    for s in specs:
        k = 'X:%s_25' % s['label']
        if k in S and s['label'] != 'all__le6':
            colsets.append((s['label'], S[k]))
    colsets.append(('TN/OA/IPO-A 37.5/37.5/25', B3))
    for s in specs:
        k = 'X:%s_{b}' % s['label']
        if 'X:%s_25' % s['label'] in S and s['label'] != 'all__le6':
            colsets.append(('TN/OA/%s 37.5/37.5/25' % s['label'],
                            mk(['TN_{b}', 'BA_{b}', k], [0.375, 0.375, 0.25])))
    py = {}
    for nm, M in colsets:
        m = bg.metrics(M, idx)
        kk = int(np.argsort(m['cagr'])[len(m['cagr']) // 2])
        py[nm] = dict(years=bg.peryear(M[kk], idx), cagr=med(m, 'cagr'), dd=med(m, 'maxdd'),
                      calmar=med(m, 'calmar'))
    bm = bg.metrics(bench[None, :], idx)
    py['NIFTYBEES'] = dict(years=bg.peryear(bench, idx), cagr=round(float(bm['cagr'][0]), 2),
                           dd=round(float(bm['maxdd'][0]), 2),
                           calmar=round(float(bm['calmar'][0]), 3))
    json.dump(py, open(RES / 's6_peryear.json', 'w'), indent=1)
    names = list(py)
    pick = [n for n in names if n != 'NIFTYBEES']
    lines = ['| year | ' + ' | '.join(names) + ' | BEST CAGR | LEAST DD | BEST OVERALL |',
             '|' + '---|' * (len(names) + 4)]
    for y in sorted(py['TN']['years']):
        cells = []
        for n in names:
            v = py[n]['years'].get(y)
            cells.append('%+.1f<br><sub>(%.1f)</sub>' % v if v else '')
        vals = {n: py[n]['years'][y] for n in pick if y in py[n]['years']}
        bc = max(vals, key=lambda n: vals[n][0])
        ld = max(vals, key=lambda n: vals[n][1])
        bo = max(vals, key=lambda n: vals[n][0] + vals[n][1])
        lines.append('| %d | ' % y + ' | '.join(cells) + ' | %s | %s | %s |' % (bc, ld, bo))
    lines.append('| **full** | ' + ' | '.join('**%.2f**<br><sub>%.1f / %.2f</sub>'
                                              % (py[n]['cagr'], py[n]['dd'], py[n]['calmar'])
                                              for n in names) + ' | | | |')
    (RES / 's6_peryear_table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


# ───────────────────────────────────────────── S7 proxy validation
def proxy(P=None):
    P = P or default_panel()
    con = sqlite3.connect(str(ROOT / 'backtest_data/fundamentals.db'))
    f = pd.read_sql_query('select date, symbol, mcap_pit from features_pit_monthly '
                          'where mcap_pit is not null', con)
    con.close()
    f['m'] = pd.to_datetime(f.date).dt.to_period('M')
    colidx = {c: j for j, c in enumerate(P.cols)}
    mon = P.dates.to_period('M')
    starts = np.nonzero(np.r_[True, mon[1:] != mon[:-1]])[0]
    rows = []
    for a in starts:
        m = mon[a]
        if str(m) < '2018-08':
            continue
        g = f[f.m == m]
        if not len(g):
            continue
        g = g[g.symbol.isin(colidx)]
        g = g.assign(tvr=[float(P.RANK[a, colidx[s]]) for s in g.symbol])
        g = g[np.isfinite(g.tvr)]
        if len(g) < 600:
            continue
        g = g.assign(mr=g.mcap_pit.rank(ascending=False), tr=g.tvr.rank())
        r = dict(month=str(m), n=len(g))
        for N in (50, 100, 200, 500):
            r['top%d' % N] = round(len(set(g[g.tr <= N].symbol) & set(g[g.mr <= N].symbol)) / N, 3)
        r['mid101_250'] = round(len(set(g[(g.tr > 100) & (g.tr <= 250)].symbol)
                                    & set(g[(g.mr > 100) & (g.mr <= 250)].symbol)) / 150, 3)
        r['small251_500'] = round(len(set(g[(g.tr > 250) & (g.tr <= 500)].symbol)
                                      & set(g[(g.mr > 250) & (g.mr <= 500)].symbol)) / 250, 3)
        top = g[g.mr <= 500]
        r['spearman_top500mcap'] = round(float(top.mr.corr(top.tr, method='spearman')), 3)
        rows.append(r)
    df = pd.DataFrame(rows)
    out = dict(months=len(df), median=df.drop(columns=['month', 'n']).median().round(3).to_dict(),
               by_year=df.assign(y=df.month.str[:4]).groupby('y')
               .median(numeric_only=True).round(3).to_dict(orient='index'))
    a = starts[-1]
    rk = {c: float(P.RANK[a, j]) for j, c in enumerate(P.cols)}

    def band(lo, hi):
        return {c for c, v in rk.items() if lo <= v <= hi}

    def off(fn):
        return set(pd.read_csv(ROOT / 'backtest_data' / fn).Symbol)

    n100 = off('nifty50_official.csv') | off('niftynext50_official.csv')
    latest = dict(month=str(mon[a]),
                  top50_vs_nifty50=round(len(band(1, 50) & off('nifty50_official.csv')) / 50, 3),
                  top100_vs_nifty100=round(len(band(1, 100) & n100) / 100, 3),
                  top200_vs_nifty200=round(len(band(1, 200) & off('nifty200_official.csv')) / 200, 3),
                  mid_vs_midcap150=round(len(band(101, 250) & off('niftymidcap150_official.csv'))
                                         / 150, 3),
                  small_vs_smallcap250=round(len(band(251, 500)
                                                 & off('niftysmallcap250_official.csv')) / 250, 3),
                  top500_vs_nifty500=round(len(band(1, 500) & off('nifty500_proxy.csv')) / 500, 3))
    out['latest_vs_official'] = latest
    json.dump(out, open(RES / 's7_proxy.json', 'w'), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    what = sys.argv[1]
    t0 = time.time()
    {'repro': repro, 'equiv': equiv, 'transplant': transplant, 'age': age, 'refit': refit,
     'costs': costs, 'blend': blend, 'proxy': proxy, 'grid': grid}[what]()
    print('\n%s DONE in %.1f min' % (what, (time.time() - t0) / 60), flush=True)
