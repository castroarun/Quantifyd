#!/usr/bin/env python3
"""
research/115_pcr_oi_adjustments -- G1c: SECOND round of controls.

G1b killed the PCR-level / max-pain / OI-wall family by showing that a zero-information
placebo (-intraday price deviation) scores a LARGER raw IC (+0.51) than any real signal.
Three families survived that first cut and need a second, harder control:

  A. atm_iv  -> fdprem   (partial IC -0.39).  Suspected artifact: ATM IV is a monotone
     proxy for the CURRENT premium level, and the outcome is the premium's own forward
     INCREMENT measured in rupees.  Regressing a level-proxy on its own forward increment
     inside one finite path is the same finite-sample mean-reversion bias, and a big
     rupee premium mechanically produces a big rupee decay.  CONTROL: project off
     rank(prem_t), and additionally score a premium-NORMALISED outcome.

  B. doi_ce_* / doi_pe_* / doi_net_15 / d_pcr_oi_*  -> fret  (partial IC ~0.08-0.10).
     Suspected artifact: OI accumulates where price has just been, so a 15-min DELTA-OI
     is contaminated by the trailing 15-min RETURN, and IC(dOI, forward return) is then
     partly plain intraday price reversal -- an effect research/109 already found to be
     un-tradeable after costs.  CONTROL: project off rank(trailing 15- and 30-min return),
     and score placebo_pastret15 (the trailing return itself) side by side.

  C. skew -> fret.  Same treatment.

CONTROL SET:  [1, rank(spot deviation), rank(minute), rank(prem_t),
               rank(trailing 15m return), rank(trailing 30m return)]
PLACEBOS   :  negspot, time, noise, pastret15 (pure price momentum), prem (pure level)
OUTCOMES   :  fret, fabs, fdprem (rupees), fdprem_n (normalised by entry premium)

A signal is ALIVE only if |partial IC| >= 0.05, sign-stable across both venues and every
month, at least 2x the shuffled-day IC, AND strictly larger in magnitude than every
placebo's partial IC on the same outcome/horizon.
"""
import csv, os, math, json, random
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', 'results'))
MASTER = os.path.join(OUT, 'features_master.csv')

HORIZONS = (5, 15, 30, 60)
OUTCOMES = ('fret', 'fabs', 'fdprem', 'fdprem_n')
SIGNALS = ['pcr_oi_all', 'pcr_oi_atm5', 'pcr_oi_atm10', 'pcr_vol_all', 'pcr_vol_atm5',
           'd_pcr_oi_all_15', 'd_pcr_oi_atm5_15',
           'doi_ce_5', 'doi_ce_15', 'doi_ce_30', 'doi_pe_5', 'doi_pe_15', 'doi_pe_30',
           'doi_net_15', 'doi_atm_15', 'doi_wing_15',
           'dist_wall_ce', 'dist_wall_pe', 'wall_squeeze',
           'mp_dev', 'mp_drift_15', 'mp_drift_30',
           'atm_iv', 'd_atm_iv_15', 'skew', 'd_prem_15']
PLACEBOS = ['placebo_negspot', 'placebo_time', 'placebo_noise', 'placebo_pastret15', 'placebo_prem']
MIN_N = 60


def rankv(a):
    order = np.argsort(a, kind='mergesort')
    r = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    dx, dy = math.sqrt(float(x @ x)), math.sqrt(float(y @ y))
    return None if dx <= 0 or dy <= 0 else float(x @ y) / (dx * dy)


def resid(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def tstat(a):
    n = len(a)
    if n < 5:
        return 0.0, 0.0
    m = float(np.mean(a)); v = float(np.var(a, ddof=1))
    return m, (m / math.sqrt(v / n) if v > 0 else 0.0)


def main():
    days = defaultdict(list)
    with open(MASTER) as f:
        for r in csv.DictReader(f):
            days[(r['date'], r['venue'])].append(r)
    units = sorted(days)
    print('units %d' % len(units), flush=True)

    byven = defaultdict(list)
    for u in units:
        byven[u[1]].append(u)
    pair = {}
    for v, lst in byven.items():
        rot = lst[7:] + lst[:7]
        for a, b in zip(lst, rot):
            pair[a] = b

    def arrs(u):
        rows = days[u]
        n = len(rows)
        o = {}
        need = SIGNALS + ['spot', 'prem', 'prem_entry'] + \
               ['%s_%d' % (oc, h) for oc in ('fret', 'fabs', 'fdprem') for h in HORIZONS]
        for c in need:
            a = np.full(n, np.nan)
            for i, r in enumerate(rows):
                s = r.get(c, '')
                if s not in ('', None):
                    try:
                        a[i] = float(s)
                    except ValueError:
                        pass
            o[c] = a
        sp, pr = o['spot'], o['prem']
        pe = o['prem_entry'][0] if n and not np.isnan(o['prem_entry'][0]) else np.nan
        base = sp[~np.isnan(sp)][0] if np.any(~np.isnan(sp)) else np.nan
        o['placebo_negspot'] = -(sp / base - 1.0)
        o['placebo_time'] = np.arange(n, dtype=float)
        rng = random.Random(hash(u[0]) & 0xffff)
        o['placebo_noise'] = np.cumsum(np.array([rng.gauss(0, 1) for _ in range(n)]))
        o['placebo_prem'] = pr.copy()
        for lagn, key in ((15, 'placebo_pastret15'), (30, '_pastret30')):
            a = np.full(n, np.nan)
            a[lagn:] = sp[lagn:] / sp[:-lagn] - 1.0
            o[key] = a
        for h in HORIZONS:
            a = np.full(n, np.nan)
            if not np.isnan(pe) and pe > 0:
                a = o['fdprem_%d' % h] / pe
            o['fdprem_n_%d' % h] = a
        return o

    cache = {u: arrs(u) for u in units}
    print('arrays built', flush=True)

    ic = defaultdict(lambda: defaultdict(list))
    for ui, u in enumerate(units):
        d, v = u
        A, Ap = cache[u], cache[pair[u]]
        ctl_src = [A['placebo_negspot'], A['placebo_time'], A['placebo_prem'],
                   A['placebo_pastret15'], A['_pastret30']]
        for oc in OUTCOMES:
            for h in HORIZONS:
                y0 = A['%s_%d' % (oc, h)]
                for sig in SIGNALS + PLACEBOS:
                    x0 = A[sig]
                    m = ~(np.isnan(x0) | np.isnan(y0))
                    for cs in ctl_src:
                        m &= ~np.isnan(cs)
                    if m.sum() < MIN_N:
                        continue
                    x, y = x0[m], y0[m]
                    if len(np.unique(x)) < 5 or len(np.unique(y)) < 5:
                        continue
                    rx, ry = rankv(x), rankv(y)
                    c = corr(rx, ry)
                    if c is not None:
                        ic[(sig, oc, h)]['raw'].append((v, d, c))
                    X = np.column_stack([np.ones(m.sum())] + [rankv(cs[m]) for cs in ctl_src])
                    c2 = corr(resid(rx, X), resid(ry, X))
                    if c2 is not None:
                        ic[(sig, oc, h)]['partial'].append((v, d, c2))
                    xs0 = Ap[sig]
                    L = min(len(xs0), len(x0))
                    xf = np.full(len(x0), np.nan); xf[:L] = xs0[:L]
                    ms = m & ~np.isnan(xf)
                    if ms.sum() >= MIN_N:
                        xs, ys = xf[ms], y0[ms]
                        if len(np.unique(xs)) >= 5 and len(np.unique(ys)) >= 5:
                            c3 = corr(rankv(xs), rankv(ys))
                            if c3 is not None:
                                ic[(sig, oc, h)]['shuffled'].append((v, d, c3))
        if (ui + 1) % 30 == 0:
            print('  %d/%d' % (ui + 1, len(units)), flush=True)

    out = []
    for (sig, oc, h), kinds in ic.items():
        rec = dict(signal=sig, outcome=oc, horizon=h, is_placebo=int(sig in PLACEBOS))
        for kind in ('raw', 'partial', 'shuffled'):
            lst = kinds.get(kind, [])
            if not lst:
                continue
            m, t = tstat(np.array([c for _, _, c in lst]))
            rec['ic_' + kind] = round(m, 4); rec['t_' + kind] = round(t, 2)
            rec['n_' + kind] = len(lst)
            signs = []
            for v in ('NIFTY', 'SENSEX'):
                sub = [(dd, c) for vv, dd, c in lst if vv == v]
                if sub:
                    mv = sum(c for _, c in sub) / len(sub)
                    rec['ic_%s_%s' % (kind, v)] = round(mv, 4)
                    signs.append(1 if mv > 0 else -1)
                    bym = defaultdict(list)
                    for dd, c in sub:
                        bym[dd[:7]].append(c)
                    for mo, cc in bym.items():
                        signs.append(1 if sum(cc) / len(cc) > 0 else -1)
            rec['stable_' + kind] = int(len(set(signs)) == 1)
        rec['abs_partial'] = abs(rec.get('ic_partial', 0.0))
        raw = abs(rec.get('ic_raw', 0.0))
        rec['pct_raw_surviving'] = round(100.0 * rec['abs_partial'] / raw, 1) if raw > 1e-9 else ''
        out.append(rec)

    # placebo ceiling per (outcome, horizon)
    ceil = defaultdict(float)
    for r in out:
        if r['is_placebo']:
            ceil[(r['outcome'], r['horizon'])] = max(ceil[(r['outcome'], r['horizon'])], r['abs_partial'])
    for r in out:
        c = ceil[(r['outcome'], r['horizon'])]
        r['placebo_ceiling'] = round(c, 4)
        r['alive'] = int((not r['is_placebo']) and r['abs_partial'] >= 0.05
                         and r.get('stable_partial', 0) == 1
                         and r['abs_partial'] > 2 * abs(r.get('ic_shuffled', 0.0))
                         and r['abs_partial'] > c)

    out.sort(key=lambda r: -r['abs_partial'])
    head = ['signal', 'outcome', 'horizon', 'is_placebo', 'ic_raw', 't_raw', 'ic_partial',
            't_partial', 'stable_partial', 'ic_shuffled', 'abs_partial', 'placebo_ceiling',
            'pct_raw_surviving', 'alive', 'n_raw', 'ic_partial_NIFTY', 'ic_partial_SENSEX']
    extra = sorted({k for r in out for k in r} - set(head))
    with open(os.path.join(OUT, 'g1c_controls2.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=head + extra, extrasaction='ignore')
        w.writeheader()
        for r in out:
            w.writerow(r)

    print('\n%-19s %-9s %3s %8s %8s %8s %8s %6s' %
          ('signal', 'outcome', 'h', 'IC_raw', 'IC_part', 'IC_shuf', 'plc_ceil', 'alive'))
    for r in out[:45]:
        print('%s%-18s %-9s %3d %8.4f %8.4f %8.4f %8.4f %6d' % (
            '*' if r['is_placebo'] else ' ', r['signal'], r['outcome'], r['horizon'],
            r.get('ic_raw', 0), r.get('ic_partial', 0), r.get('ic_shuffled', 0),
            r['placebo_ceiling'], r['alive']))
    alive = [r for r in out if r['alive']]
    print('\n(* = placebo, zero option information)')
    print('ALIVE AFTER FULL CONTROLS: %d' % len(alive))
    for r in alive:
        print('   %-18s %-9s h=%-3d partial IC %+.4f  t %+.2f  NIFTY %+.4f SENSEX %+.4f' % (
            r['signal'], r['outcome'], r['horizon'], r['ic_partial'], r['t_partial'],
            r.get('ic_partial_NIFTY', 0), r.get('ic_partial_SENSEX', 0)))
    json.dump(dict(tests=len(out), alive=len(alive),
                   alive_list=[[r['signal'], r['outcome'], r['horizon'], r['ic_partial']] for r in alive]),
              open(os.path.join(OUT, 'g1c_summary.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
