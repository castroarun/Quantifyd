#!/usr/bin/env python3
"""
research/115_pcr_oi_adjustments -- G1b: ADVERSARIAL CONTROLS on the G1 information
coefficients.

WHY THIS SCRIPT EXISTS.  The raw G1 pass produced enormous ICs (|IC| up to 0.41,
t up to 22).  Edges that large do not exist in a liquid index.  Two mechanical
explanations must be eliminated before any of it is believed:

 (1) PRICE-ANCHOR ARTIFACT.  Signals like dist_wall_ce, mp_dev, wall_squeeze and
     (empirically) the PCR levels are all monotone functions of "where is spot relative
     to a sticky intraday anchor".  Regressing a day's forward returns on a persistent,
     contemporaneously price-linked variable inside ONE finite path yields a
     systematically negative sample correlation even for a pure random walk
     (Stambaugh / finite-sample mean-reversion bias).  The sign of that bias is the same
     every day -- so "sign stable across venues and months" is exactly what the ARTIFACT
     predicts, and is not evidence of anything.

 (2) TIME-OF-DAY ARTIFACT.  The straddle premium decays monotonically through the
     session and ATM IV drifts with it.  Any signal with an intraday time trend will
     correlate strongly with fdprem_h for that reason alone.

CONTROLS RUN HERE
  * placebo_negspot  : -(spot_t/spot_first - 1).  Pure price-anchor deviation.
                       Contains ZERO option information.  If it reproduces the ICs, the
                       "PCR edge" is the artifact.
  * placebo_time     : minute index.  Pure clock.  Zero market information.
  * placebo_noise    : a per-day random walk with matched persistence, seeded
                       deterministically (reproducible) and independent of the market.
  * PARTIAL IC       : Spearman between signal and outcome AFTER both are projected off
                       [rank(spot deviation), rank(minute)] within the same day.  This is
                       the incremental information the OI signal carries beyond
                       "where is price" and "what time is it".
  * SHUFFLED-DAY IC  : the signal path from a DIFFERENT day (same venue, deterministic
                       pairing) against this day's outcomes.  Preserves the signal's
                       shape and persistence, destroys any genuine link.  Whatever IC
                       survives here is pure shape artifact.

PASS criterion for a signal to still be alive after G1b:
  |partial IC| >= 0.05, sign-stable, AND materially larger than both the shuffled-day IC
  and the placebo ICs.
"""
import csv, os, math, json, random
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', 'results'))
MASTER = os.path.join(OUT, 'features_master.csv')

HORIZONS = (5, 15, 30, 60)
OUTCOMES = ('fret', 'fabs', 'fdprem')
SIGNALS = ['pcr_oi_all', 'pcr_oi_atm5', 'pcr_oi_atm10', 'pcr_vol_all', 'pcr_vol_atm5',
           'd_pcr_oi_all_15', 'd_pcr_oi_atm5_15',
           'doi_ce_5', 'doi_ce_15', 'doi_ce_30', 'doi_pe_5', 'doi_pe_15', 'doi_pe_30',
           'doi_net_15', 'doi_atm_15', 'doi_wing_15',
           'dist_wall_ce', 'dist_wall_pe', 'wall_squeeze',
           'mp_dev', 'mp_drift_15', 'mp_drift_30',
           'atm_iv', 'd_atm_iv_15', 'skew', 'd_prem_15']
PLACEBOS = ['placebo_negspot', 'placebo_time', 'placebo_noise']
MIN_N = 60


def rankv(a):
    """average-rank transform of a 1-D float array (NaNs must be pre-removed)."""
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
    if dx <= 0 or dy <= 0:
        return None
    return float(x @ y) / (dx * dy)


def resid(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


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
    print('day-venue units: %d' % len(units))

    # deterministic shuffle pairing: each day gets another day of the SAME venue
    byven = defaultdict(list)
    for d, v in units:
        byven[v].append((d, v))
    pair = {}
    for v, lst in byven.items():
        rot = lst[7:] + lst[:7]          # fixed rotation -> reproducible, no seed drift
        for a, b in zip(lst, rot):
            pair[a] = b

    # ---- materialise per-unit numeric arrays -------------------------------------
    def arrs(u):
        rows = days[u]
        n = len(rows)
        out = {}
        for c in SIGNALS + ['spot', 'prem'] + \
                 ['%s_%d' % (o, h) for o in OUTCOMES for h in HORIZONS]:
            a = np.full(n, np.nan)
            for i, r in enumerate(rows):
                s = r.get(c, '')
                if s not in ('', None):
                    try:
                        a[i] = float(s)
                    except ValueError:
                        pass
            out[c] = a
        sp = out['spot']
        base = sp[~np.isnan(sp)][0] if np.any(~np.isnan(sp)) else np.nan
        out['placebo_negspot'] = -(sp / base - 1.0)
        out['placebo_time'] = np.arange(n, dtype=float)
        rng = random.Random(hash(u[0]) & 0xffff)
        w = np.cumsum(np.array([rng.gauss(0, 1) for _ in range(n)]))
        out['placebo_noise'] = w
        return out

    cache = {u: arrs(u) for u in units}
    print('arrays built')

    ic = defaultdict(lambda: defaultdict(list))       # (sig,oc,h) -> kind -> [(venue,date,ic)]
    for u in units:
        d, v = u
        A = cache[u]
        Ap = cache[pair[u]]
        sd = A['placebo_negspot']; tm = A['placebo_time']
        for oc in OUTCOMES:
            for h in HORIZONS:
                y0 = A['%s_%d' % (oc, h)]
                for sig in SIGNALS + PLACEBOS:
                    x0 = A[sig]
                    m = ~(np.isnan(x0) | np.isnan(y0) | np.isnan(sd) | np.isnan(tm))
                    if m.sum() < MIN_N:
                        continue
                    x, y = x0[m], y0[m]
                    if len(np.unique(x)) < 5 or len(np.unique(y)) < 5:
                        continue
                    rx, ry = rankv(x), rankv(y)
                    c_raw = corr(rx, ry)
                    if c_raw is not None:
                        ic[(sig, oc, h)]['raw'].append((v, d, c_raw))
                    # partial: project off rank(spot deviation) and rank(minute)
                    X = np.column_stack([np.ones(m.sum()), rankv(sd[m]), rankv(tm[m])])
                    c_par = corr(resid(rx, X), resid(ry, X))
                    if c_par is not None:
                        ic[(sig, oc, h)]['partial'].append((v, d, c_par))
                    # shuffled-day signal vs this day's outcome
                    xs0 = Ap[sig]
                    L = min(len(xs0), len(x0))
                    ms = np.zeros(len(x0), bool); ms[:L] = True
                    ms &= ~np.isnan(y0) & ~np.isnan(np.concatenate([xs0[:L], np.full(len(x0) - L, np.nan)]))
                    if ms.sum() >= MIN_N:
                        xs = np.concatenate([xs0[:L], np.full(len(x0) - L, np.nan)])[ms]
                        ys = y0[ms]
                        if len(np.unique(xs)) >= 5 and len(np.unique(ys)) >= 5:
                            c_sh = corr(rankv(xs), rankv(ys))
                            if c_sh is not None:
                                ic[(sig, oc, h)]['shuffled'].append((v, d, c_sh))
        if (units.index(u) + 1) % 20 == 0:
            print('  %d/%d units' % (units.index(u) + 1, len(units)), flush=True)

    rows_out = []
    for (sig, oc, h), kinds in ic.items():
        rec = dict(signal=sig, outcome=oc, horizon=h,
                   is_placebo=int(sig in PLACEBOS))
        for kind in ('raw', 'partial', 'shuffled'):
            lst = kinds.get(kind, [])
            if not lst:
                continue
            m, t = tstat(np.array([c for _, _, c in lst]))
            rec['ic_' + kind] = round(m, 4)
            rec['t_' + kind] = round(t, 2)
            rec['n_' + kind] = len(lst)
            signs = []
            for v in ('NIFTY', 'SENSEX'):
                sub = [(d, c) for vv, d, c in lst if vv == v]
                if sub:
                    mv = sum(c for _, c in sub) / len(sub)
                    rec['ic_%s_%s' % (kind, v)] = round(mv, 4)
                    signs.append(1 if mv > 0 else -1)
                    bym = defaultdict(list)
                    for d, c in sub:
                        bym[d[:7]].append(c)
                    for mo, cc in bym.items():
                        signs.append(1 if sum(cc) / len(cc) > 0 else -1)
            rec['stable_' + kind] = int(len(set(signs)) == 1)
        rec['abs_partial'] = abs(rec.get('ic_partial', 0.0))
        rows_out.append(rec)

    # how much of the raw IC survives the controls?
    for r in rows_out:
        raw = abs(r.get('ic_raw', 0.0))
        r['pct_raw_surviving'] = round(100.0 * r['abs_partial'] / raw, 1) if raw > 1e-9 else ''
        r['alive'] = int(r['abs_partial'] >= 0.05 and r.get('stable_partial', 0) == 1
                         and r['abs_partial'] > 2 * abs(r.get('ic_shuffled', 0.0)))

    rows_out.sort(key=lambda r: -r['abs_partial'])
    head = ['signal', 'outcome', 'horizon', 'is_placebo', 'ic_raw', 't_raw', 'stable_raw',
            'ic_partial', 't_partial', 'stable_partial', 'ic_shuffled', 't_shuffled',
            'abs_partial', 'pct_raw_surviving', 'alive', 'n_raw',
            'ic_raw_NIFTY', 'ic_raw_SENSEX', 'ic_partial_NIFTY', 'ic_partial_SENSEX']
    extra = sorted({k for r in rows_out for k in r} - set(head))
    with open(os.path.join(OUT, 'g1b_controls.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=head + extra, extrasaction='ignore')
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print('\n%-18s %-7s %3s %8s %8s %9s %8s %6s' %
          ('signal', 'outcome', 'h', 'IC_raw', 'IC_part', 'IC_shuff', '%surv', 'alive'))
    for r in rows_out[:40]:
        tag = '*' if r['is_placebo'] else ' '
        print('%s%-17s %-7s %3d %8.4f %8.4f %9.4f %8s %6d' % (
            tag, r['signal'], r['outcome'], r['horizon'], r.get('ic_raw', 0),
            r.get('ic_partial', 0), r.get('ic_shuffled', 0), r['pct_raw_surviving'], r['alive']))
    print('\n(* = PLACEBO: contains no option information at all)')
    alive = [r for r in rows_out if r['alive'] and not r['is_placebo']]
    print('SIGNALS ALIVE AFTER CONTROLS: %d' % len(alive))
    for r in alive[:20]:
        print('   %s %s h=%d  partial IC %.4f (t %.2f)' %
              (r['signal'], r['outcome'], r['horizon'], r['ic_partial'], r['t_partial']))
    json.dump(dict(tests=len(rows_out), alive=len(alive)),
              open(os.path.join(OUT, 'g1b_summary.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
