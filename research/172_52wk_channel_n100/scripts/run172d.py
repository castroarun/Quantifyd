# -*- coding: utf-8 -*-
"""research/172 Phase 2 - stop-loss and trailing-stop combinations.

Arun, mid-turn: "u can add some stop loss variations/trailing SL etc, try different
combinations as well."

PAIRED BY CONSTRUCTION: every exit stack is run on BOTH entries - the plateau centre
(189-day close channel) and the literal Spec A entry (252-day close channel) - so the
comparison is stack-vs-stack on identical signals.

Book held fixed at the Phase 1 settings: Nifty 100 current list, 20 slots at 5% of a
Rs 1 crore book, next-open fills on both legs, 15 bps a side, 5.2% post-tax idle cash,
after tax (20/12.5, FY-netted, Rs 1.25 L LTCG exemption). Ranking metric: after-tax
Calmar, with the same pre-registered eligibility clause (CAGR above NIFTYBEES 11.37%).

Phases: A (stacks) / B (time stop, book kill, offsets, per-year, cost) / C (null + blend)
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

import p172                     # noqa: E402
import bt172 as B               # noqa: E402
import bt172b as B2             # noqa: E402
import run172 as R              # noqa: E402

BENCH_CAGR = 11.37              # NIFTYBEES over the same window - the eligibility floor
OPT_CALMAR = 0.575              # Phase 1 winner: 52W OPT, L252 close + ST(14,4)
UNI = 'n100'
ENTRIES = [('E189', 189), ('E252', 252)]

FIELDS = ['label', 'stack', 'entry', 'entry_L', 'group', 'hard_pct', 'hard_atr',
          'be_after', 'lock_after', 'lock_trail', 'trail_pct', 'trail_chand', 'trail_atr',
          'rule', 'time_bars', 'time_min_gain', 'block_bars', 'book_dd_kill',
          'cost_bps', 'offset', 'window',
          'cagr', 'maxdd', 'calmar', 'sharpe', 'trades', 'win_rate', 'avg_win',
          'avg_loss', 'expectancy', 'med_hold_d', 'worst_mae', 'med_mae',
          'max_loss_streak', 'trades_per_yr', 'avg_invested', 'eligible',
          'pct_STOP', 'pct_RULE', 'pct_TIME', 'pct_BOOKKILL', 'book_kills',
          'tax_paid', 'cost_paid']

_TRIG = {}
_EX = {}


def trig_for(P, L):
    if L not in _TRIG:
        _TRIG[L] = B.entry_signal(P, UNI, L, ref='close')
    return _TRIG[L]


def rule_arr(P, key):
    if not key:
        return None
    if key not in _EX:
        _EX[key] = B.exit_array(P, key)
    return _EX[key]


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


def run_stack(P, name, group, stack, entry_tag, L, cost=15.0, days=None, offset=None,
              keep=False):
    cfg = dict(days=days if days is not None else P.days['full'], slots=20,
               cost_bps=cost, gate=None, seed=None, tax=True)
    cfg.update({k: v for k, v in stack.items() if k != 'rule'})
    cfg['exit_arr'] = rule_arr(P, stack.get('rule'))
    r = B2.simulate_stack(P, trig_for(P, L), cfg)
    m = B.metrics(r['nav'], r['dates'], r['trades'])
    tot = max(sum(r['reasons'].values()), 1)
    row = dict(label='%s__%s%s%s' % (name, entry_tag,
                                     '' if cost == 15.0 else '_c%d' % int(cost),
                                     '' if offset is None else '_o%02d' % offset),
               stack=name, entry=entry_tag, entry_L=L, group=group, cost_bps=cost,
               offset=offset, window='full',
               rule=stack.get('rule') or '', book_kills=r['book_kills'],
               tax_paid=round(r['tax_paid']), cost_paid=round(r['cost_paid']),
               avg_invested=round(float(np.nanmean(r['invested'])), 3))
    for k in ('hard_pct', 'hard_atr', 'be_after', 'lock_after', 'lock_trail',
              'trail_pct', 'trail_chand', 'trail_atr', 'time_bars', 'time_min_gain',
              'block_bars', 'book_dd_kill'):
        row[k] = stack.get(k, 0)
    for rs in ('STOP', 'RULE', 'TIME', 'BOOKKILL'):
        row['pct_' + rs] = round(100.0 * r['reasons'].get(rs, 0) / tot, 1)
    row.update(m)
    row['eligible'] = bool(m.get('cagr', 0) > BENCH_CAGR)
    return (row, r) if keep else (row, None)


# ─────────────────────────────────────────────────────────── the stacks
def build_stacks():
    S = []

    def add(name, group, **kw):
        S.append((name, group, kw))

    # 1. INITIAL HARD STOP ALONE - otherwise the literal 52-week-low exit
    for x in (5, 8, 10, 12, 15, 20, 25, 30):
        add('HARD%d' % x, 'hard_alone', hard_pct=x / 100.0, rule='CC252')
    for k in (1.5, 2.0, 3.0, 4.0):
        add('HARDATR%s' % str(k).replace('.', ''), 'hard_alone', hard_atr=k, rule='CC252')
    add('NOSTOP_CC252', 'reference', rule='CC252')

    # 2. TRAILING ALONE
    for x in (8, 10, 12, 15, 20, 25, 30):
        add('TRAIL%d' % x, 'trail_alone', trail_pct=x / 100.0)
    for k in (2.0, 3.0, 4.0):
        add('CHAND%s' % str(k).replace('.', ''), 'trail_alone', trail_chand=k)
    for rl in ('ST_14_4', 'ST_10_3', 'EMA50', 'CC42', 'CC63', 'CC126'):
        add('REF_%s' % rl, 'reference', rule=rl)

    # 3. COMBINATIONS - initial hard stop x trailing leg
    TRAILS = {'TRAIL15': dict(trail_pct=0.15), 'TRAIL20': dict(trail_pct=0.20),
              'CHAND30': dict(trail_chand=3.0), 'ST_14_4': dict(rule='ST_14_4'),
              'CC63': dict(rule='CC63'), 'CC252': dict(rule='CC252')}
    for hp in (0.10, 0.15, 0.20, 0.0):
        for tn, tk in TRAILS.items():
            if hp == 0.0:
                continue                        # the no-stop row already exists above
            add('H%d+%s' % (int(hp * 100), tn), 'combo', hard_pct=hp, **tk)
    # breakeven-move variants
    for be in (0.10, 0.20):
        for tn in ('TRAIL15', 'ST_14_4', 'CC63'):
            add('BE%d+%s' % (int(be * 100), tn), 'breakeven', be_after=be, **TRAILS[tn])
    # profit-lock: after +30 / +50 the trail tightens to -10%
    for la in (0.30, 0.50):
        for tn in ('TRAIL20', 'ST_14_4', 'CC63'):
            add('LOCK%d+%s' % (int(la * 100), tn), 'profit_lock', lock_after=la,
                lock_trail=0.10, **TRAILS[tn])
    # re-entry toggle after a stop-out
    for bb, tag in ((63, 'BLK63'), (10 ** 9, 'BLKALL')):
        for base, kw in (('H15+ST_14_4', dict(hard_pct=0.15, rule='ST_14_4')),
                         ('TRAIL20', dict(trail_pct=0.20)),
                         ('H15+TRAIL20', dict(hard_pct=0.15, trail_pct=0.20))):
            add('%s_%s' % (base, tag), 'reentry', block_bars=bb, **kw)
    return S


def phase_a(P):
    path = RES / 'stops.csv'
    done = done_labels(path)
    S = build_stacks()
    total = len(S) * len(ENTRIES)
    t0 = time.time()
    k = 0
    for name, group, stack in S:
        for tag, L in ENTRIES:
            k += 1
            lab = '%s__%s' % (name, tag)
            if lab in done:
                continue
            row, _ = run_stack(P, name, group, stack, tag, L)
            append(path, row)
            if k % 30 == 0:
                print('  [A] %d/%d %-22s %s cagr=%6.2f dd=%7.2f calmar=%5.3f (%.1f min)'
                      % (k, total, name, tag, row.get('cagr', np.nan),
                         row.get('maxdd', np.nan), row.get('calmar', np.nan),
                         (time.time() - t0) / 60), flush=True)
    print('  [A] %d stacks x %d entries = %d rows, %.1f min'
          % (len(S), len(ENTRIES), total, (time.time() - t0) / 60), flush=True)


def top_stacks(n=5, group_in=None, exclude_groups=()):
    df = pd.read_csv(RES / 'stops.csv')
    df = df[(df.entry == 'E189') & df.offset.isna() & (df.cost_bps == 15.0)
            & (df.eligible.astype(str) == 'True') & (df.trades.fillna(0) >= 50)]
    if group_in:
        df = df[df.group.isin(group_in)]
    if exclude_groups:
        df = df[~df.group.isin(exclude_groups)]
    return df.sort_values('calmar', ascending=False).head(n)


def stack_of(name):
    """Resolve ANY stack label back to its spec, including the phase-B derivatives
    (`<base>_T<bars>_g<gain>` time stop, `<base>_BK<pct>` book kill), so a resume in a
    fresh process can reconstruct a cell it did not itself build."""
    base = {nm: (grp, kw) for nm, grp, kw in build_stacks()}
    if name in base:
        return base[name]
    if '_BK' in name:
        root, k = name.rsplit('_BK', 1)
        grp, kw = stack_of(root)
        return 'book_kill', dict(kw, book_dd_kill=int(k) / 100.0)
    if '_T' in name and '_g' in name:
        root, rest = name.rsplit('_T', 1)
        tb, mg = rest.split('_g')
        grp, kw = stack_of(root)
        return 'time_stop', dict(kw, time_bars=int(tb), time_min_gain=int(mg) / 100.0)
    raise KeyError(name)


def phase_b(P):
    path = RES / 'stops.csv'
    done = done_labels(path)
    t0 = time.time()

    # ---- 4. TIME STOP on the best two trailing legs from phase A
    best = top_stacks(2, exclude_groups=('reference',))
    tops = list(best['stack'])
    print('  [B] time stop applied to: %s' % tops, flush=True)
    for nm in tops:
        grp, base = stack_of(nm)
        for tb in (63, 126):
            for mg in (0.0, 0.05):
                name = '%s_T%d_g%d' % (nm, tb, int(mg * 100))
                st = dict(base, time_bars=tb, time_min_gain=mg)
                for tag, L in ENTRIES:
                    if '%s__%s' % (name, tag) in done:
                        continue
                    row, _ = run_stack(P, name, 'time_stop', st, tag, L)
                    append(path, row)

    # ---- 5. BOOK-LEVEL trailing drawdown kill on the best three stacks
    best3 = top_stacks(3, exclude_groups=('reference',))
    for nm in list(best3['stack']):
        grp, base = stack_of(nm)
        for kill in (0.15, 0.20):
            name = '%s_BK%d' % (nm, int(kill * 100))
            st = dict(base, book_dd_kill=kill)
            for tag, L in ENTRIES:
                if '%s__%s' % (name, tag) in done:
                    continue
                row, _ = run_stack(P, name, 'book_kill', st, tag, L)
                append(path, row)
    print('  [B] time stop + book kill done (%.1f min)' % ((time.time() - t0) / 60),
          flush=True)

    # ---- 12 start-date offsets on the top five stacks (both entries)
    top5 = list(top_stacks(5)['stack'])
    json.dump(top5, open(RES / 'p2_top5.json', 'w'), indent=1)
    for nm in top5:
        grp, base = stack_of(nm)
        for m_ in range(12):
            start = '2006-%02d-01' % (m_ + 1)
            days = np.nonzero((P.dstr >= start) & (P.dstr <= p172.TRADE_END))[0]
            for tag, L in ENTRIES:
                if '%s__%s_o%02d' % (nm, tag, m_) in done:
                    continue
                row, _ = run_stack(P, nm, grp, base, tag, L, days=days, offset=m_)
                append(path, row)
    print('  [B] 12 offsets x %d stacks x 2 entries done (%.1f min)'
          % (len(top5), (time.time() - t0) / 60), flush=True)

    # ---- cost ladder on the single winner
    win = top_stacks(1).iloc[0]
    grp, base = stack_of(win['stack'])
    for cost in (0.0, 30.0, 45.0):
        for tag, L in ENTRIES:
            if '%s__%s_c%d' % (win['stack'], tag, int(cost)) in done:
                continue
            row, _ = run_stack(P, win['stack'], grp, base, tag, L, cost=cost)
            append(path, row)

    # ---- per-year + NAV archive for the top three, plus the Phase 1 references
    curves, outl = {}, []
    idx = pd.to_datetime(P.dstr[P.days['full'][0]:P.days['full'][-1] + 1])
    for nm in list(top_stacks(3)['stack']):
        grp, base = stack_of(nm)
        for tag, L in ENTRIES:
            _, r = run_stack(P, nm, grp, base, tag, L, keep=True)
            curves['%s %s' % (nm, tag)] = r['nav']
            t = pd.DataFrame(r['trades'])
            if tag == 'E189':
                t.to_csv(RES / ('p2_trades_%s.csv' % nm.replace('+', '_')), index=False)
                tot = t.pnl.sum()
                outl.append(dict(stack=nm, n=len(t),
                                 top10_share=round(100 * t.nlargest(10, 'pnl').pnl.sum()
                                                   / tot, 1) if tot else None,
                                 mean_ret=round(float(t.ret_pct.mean()), 3),
                                 mean_ret_cap50=round(float(t.ret_pct.clip(upper=50).mean()), 3),
                                 worst_mae=round(float(t.mae_pct.min()), 1),
                                 med_mae=round(float(t.mae_pct.median()), 1),
                                 p05_mae=round(float(t.mae_pct.quantile(0.05)), 1),
                                 med_hold_d=int(t.days.median()),
                                 p95_hold_d=int(t.days.quantile(0.95))))
    z = np.load(RES / 'curves.npz', allow_pickle=True)
    for k_ in ('52W_OPT', '52W_Spec_A', 'NIFTYBEES'):
        curves[k_] = z[k_]
    rows, summ = [], {}
    for nm, nv in curves.items():
        s = pd.Series(np.asarray(nv, float), index=idx)
        pk = s.cummax()
        dd = s / pk - 1.0
        for y in sorted(set(idx.year)):
            j = np.flatnonzero(idx.year == y)
            s0 = j[0] - 1 if j[0] > 0 else j[0]
            rows.append(dict(series=nm, year=int(y),
                             ret=round(100 * (s.iloc[j[-1]] / s.iloc[s0] - 1), 2),
                             dd=round(100 * float(dd.iloc[j].min()), 2)))
        yrs = (idx[-1] - idx[0]).days / 365.25
        cg = 100 * ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1)
        md = 100 * float(dd.min())
        summ[nm] = dict(cagr=round(cg, 2), maxdd=round(md, 2),
                        calmar=round(cg / abs(md), 3))
    pd.DataFrame(rows).to_csv(RES / 'p2_peryear.csv', index=False)
    pd.DataFrame(outl).to_csv(RES / 'p2_outliers.csv', index=False)
    json.dump(summ, open(RES / 'p2_summary.json', 'w'), indent=1)
    np.savez_compressed(RES / 'p2_curves.npz',
                        dates=np.array([str(d.date()) for d in idx]),
                        **{k_.replace(' ', '_').replace('+', 'p'): np.asarray(v, float)
                           for k_, v in curves.items()})
    print(json.dumps(summ, indent=1), flush=True)


def phase_c(P):
    """Momentum-matched nulls + blend, ONLY if the winner clears 52W OPT by >= 0.05."""
    win = top_stacks(1).iloc[0]
    gate = float(win['calmar']) >= OPT_CALMAR + 0.05
    out = dict(winner=str(win['stack']), entry='E189', calmar=float(win['calmar']),
               cagr=float(win['cagr']), maxdd=float(win['maxdd']),
               opt_calmar=OPT_CALMAR, threshold=OPT_CALMAR + 0.05, ran_null_and_blend=gate)
    if not gate:
        json.dump(out, open(RES / 'p2_gate.json', 'w'), indent=1)
        print('  [C] SKIPPED by the pre-registered gate: best Phase 2 Calmar %.3f < %.3f'
              % (win['calmar'], OPT_CALMAR + 0.05), flush=True)
        return
    grp, base = stack_of(win['stack'])
    path = RES / 'p2_nulls.csv'
    done = done_labels(path)
    days = P.days['full']
    L = 189
    trig = trig_for(P, L)
    elig = P.eligible(UNI, L)
    up = (P.ST['ST_14_4'] == 1)
    esh = np.zeros_like(elig)
    esh[1:] = elig[:-1]
    ush = np.zeros_like(up)
    ush[1:] = up[:-1]
    rs = np.where(esh, P.RS252, np.nan)
    med = np.nanmedian(rs, axis=1)
    strong = np.zeros_like(elig)
    with np.errstate(invalid='ignore'):
        strong[1:] = (elig & (P.RS252 >= med[:, None]))[:-1]
    pools = {'P2N3_mom': strong, 'P2N4_trendmom': esh & ush & strong}
    for nm, pool in pools.items():
        for s in range(1, 31):
            lab = '%s_%02d' % (nm, s)
            if lab in done:
                continue
            rng = np.random.default_rng(90000 + s)
            null = np.zeros_like(trig)
            nper = trig.sum(axis=1)
            for i in np.nonzero(nper)[0]:
                pl = np.nonzero(pool[i])[0]
                if not len(pl):
                    continue
                null[i, rng.choice(pl, size=min(int(nper[i]), len(pl)),
                                   replace=False)] = True
            cfg = dict(days=days, slots=20, cost_bps=15.0, gate=None, seed=None, tax=True)
            cfg.update({k: v for k, v in base.items() if k != 'rule'})
            cfg['exit_arr'] = rule_arr(P, base.get('rule'))
            r = B2.simulate_stack(P, null, cfg)
            m = B.metrics(r['nav'], r['dates'], r['trades'])
            row = dict(label=lab, stack=nm, entry='E189', entry_L=L, group='null',
                       cost_bps=15.0, window='full')
            row.update(m)
            append(path, row)
        sub = pd.read_csv(path)
        sub = sub[sub.label.str.startswith(nm)]
        print('  [C] %s n=%d cagr med %.2f [%.2f..%.2f] calmar med %.3f'
              % (nm, len(sub), sub.cagr.median(), sub.cagr.min(), sub.cagr.max(),
                 sub.calmar.median()), flush=True)
    json.dump(out, open(RES / 'p2_gate.json', 'w'), indent=1)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    P = p172.Panel()
    steps = {'a': [phase_a], 'b': [phase_b], 'c': [phase_c],
             'all': [phase_a, phase_b, phase_c]}[what]
    for fn in steps:
        print('=== %s ===' % fn.__name__, flush=True)
        fn(P)
    print('DONE phase2 %s in %.1f min' % (what, (time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
