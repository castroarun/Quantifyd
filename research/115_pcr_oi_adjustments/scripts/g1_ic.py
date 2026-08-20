#!/usr/bin/env python3
"""
research/115_pcr_oi_adjustments -- G1: does ANY signal carry information?

Method (deliberately conservative about overlapping observations):
  * Within each (day, venue) we compute the SPEARMAN rank correlation between the signal
    at minute t and the forward outcome over the next h minutes.  Minutes within a day are
    massively overlapping/autocorrelated, so a pooled correlation's t-stat would be a lie.
  * The DAY is the independent unit.  We then t-test the ~85 daily ICs per venue.
    This is the standard "daily IC -> t-stat" construction and it is honest about n.

Outcomes:
  fret_h    signed forward underlying return   (direction signals)
  fabs_h    |forward return|                   (magnitude / "is a move coming")
  fdprem_h  forward change in the premium of the straddle WE ARE SHORT (fixed 09:16 ATM
            strike).  This is the one that actually maps to P&L: premium up = we lose.

PASS gate (pre-registered in the STATUS-MD):
  |mean IC| >= 0.05 AND the sign is stable across BOTH venues AND all months.

Multiple testing: 26 signals x 4 horizons x 3 outcomes = 312 tests.  Bonferroni 5%
=> per-test alpha 1.6e-4 => |t| >= ~3.9.  Reported alongside every raw t.
"""
import csv, os, math, json
from collections import defaultdict

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
MIN_N = 60          # minutes needed inside a day for the daily IC to mean anything


def ranks(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    n = len(x)
    if n < MIN_N:
        return None
    rx, ry = ranks(x), ranks(y)
    if len(set(rx)) < 5 or len(set(ry)) < 5:
        return None          # degenerate (e.g. a wall strike that never moved)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sxx = syy = 0.0
    for a, b in zip(rx, ry):
        da, db = a - mx, b - my
        sxy += da * db; sxx += da * da; syy += db * db
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def tstat(a):
    n = len(a)
    if n < 5:
        return 0.0, 0.0
    m = sum(a) / n
    var = sum((x - m) ** 2 for x in a) / (n - 1)
    if var <= 0:
        return m, 0.0
    return m, m / math.sqrt(var / n)


def main():
    # day -> venue -> {col: [values]}
    days = defaultdict(lambda: defaultdict(list))
    cols = None
    with open(MASTER) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        for r in rd:
            days[(r['date'], r['venue'])]['_rows'].append(r)
    print('day-venue units: %d' % len(days))

    # daily IC:  (signal, outcome, h) -> venue -> list of (date, ic)
    ic = defaultdict(lambda: defaultdict(list))
    for (d, v), bag in sorted(days.items()):
        rows = bag['_rows']
        for sig in SIGNALS:
            for oc in OUTCOMES:
                for h in HORIZONS:
                    ocol = '%s_%d' % (oc, h)
                    xs, ys = [], []
                    for r in rows:
                        a, b = r.get(sig, ''), r.get(ocol, '')
                        if a not in ('', None) and b not in ('', None):
                            try:
                                xs.append(float(a)); ys.append(float(b))
                            except ValueError:
                                pass
                    s = spearman(xs, ys)
                    if s is not None:
                        ic[(sig, oc, h)][v].append((d, s))

    rows_out = []
    for key, byv in ic.items():
        sig, oc, h = key
        allic = [s for v in byv for (_, s) in byv[v]]
        if len(allic) < 40:
            continue
        mean_all, t_all = tstat(allic)
        rec = dict(signal=sig, outcome=oc, horizon=h, n_days=len(allic),
                   mean_ic=round(mean_all, 4), t_stat=round(t_all, 2),
                   abs_ic=round(abs(mean_all), 4))
        signs = []
        for v in ('NIFTY', 'SENSEX'):
            lst = byv.get(v, [])
            m, t = tstat([s for _, s in lst])
            rec['ic_' + v] = round(m, 4)
            rec['t_' + v] = round(t, 2)
            rec['n_' + v] = len(lst)
            if lst:
                signs.append(1 if m > 0 else -1)
            # per month
            bym = defaultdict(list)
            for d, s in lst:
                bym[d[:7]].append(s)
            for mo, ss in bym.items():
                mm = sum(ss) / len(ss)
                signs.append(1 if mm > 0 else -1)
                rec['ic_%s_%s' % (v, mo)] = round(mm, 4)
        rec['sign_stable'] = int(len(set(signs)) == 1)
        rec['pass_g1'] = int(rec['abs_ic'] >= 0.05 and rec['sign_stable'] == 1)
        rec['pass_bonf'] = int(abs(t_all) >= 3.9)
        rows_out.append(rec)

    rows_out.sort(key=lambda r: -r['abs_ic'])
    keys = sorted({k for r in rows_out for k in r})
    head = ['signal', 'outcome', 'horizon', 'n_days', 'mean_ic', 'abs_ic', 't_stat',
            'sign_stable', 'pass_g1', 'pass_bonf', 'ic_NIFTY', 't_NIFTY', 'n_NIFTY',
            'ic_SENSEX', 't_SENSEX', 'n_SENSEX']
    head += [k for k in keys if k not in head]
    with open(os.path.join(OUT, 'g1_ic_table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=head, extrasaction='ignore')
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print('tests run: %d' % len(rows_out))
    print('%-18s %-7s %3s %8s %7s %6s %6s %6s' % ('signal', 'outcome', 'h', 'meanIC', 't', 'stable', 'G1', 'Bonf'))
    for r in rows_out[:35]:
        print('%-18s %-7s %3d %8.4f %7.2f %6d %6d %6d' % (
            r['signal'], r['outcome'], r['horizon'], r['mean_ic'], r['t_stat'],
            r['sign_stable'], r['pass_g1'], r['pass_bonf']))
    npass = sum(r['pass_g1'] for r in rows_out)
    print('\nG1 PASS count (|IC|>=0.05 and sign-stable): %d / %d' % (npass, len(rows_out)))
    json.dump(dict(tests=len(rows_out), g1_pass=npass,
                   bonf_pass=sum(r['pass_bonf'] for r in rows_out)),
              open(os.path.join(OUT, 'g1_summary.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
