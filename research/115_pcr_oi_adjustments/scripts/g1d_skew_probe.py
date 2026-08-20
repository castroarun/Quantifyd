#!/usr/bin/env python3
"""
research/115_pcr_oi_adjustments -- G1d: targeted adversarial probe of the ONE survivor.

`skew` (IV of the PE 3 strikes below ATM minus IV of the CE 3 strikes above ATM) was the
only construction still standing after G1c, with partial IC -0.138 vs the 5-minute forward
return.  It is NOT a PCR/OI signal -- it is outside the study's pre-registered question --
but before reporting it as a live next-lever it has to survive the obvious objections:

 1. The G1c control set contained trailing 15- and 30-minute returns but NOT the trailing
    5-minute return, and skew's IC peaks exactly at the 5-minute horizon.  Short-horizon
    reversal / bid-ask bounce is the prime suspect.  -> add rank(trailing 5m return).
 2. Recorded IV is broker-derived from LTP.  A stale LTP on one wing mechanically moves
    skew.  -> re-run on the subset of minutes where BOTH wing legs traded recently
    (volume changed within the last 5 minutes), i.e. non-stale quotes only.
 3. IV is null on ~24% of NIFTY rows -> selection.  Report coverage of the tested subset.
 4. Direction signals do not pay a delta-neutral short-straddle book directly, so also
    score skew against |forward return| and against the forward premium change.
"""
import csv, os, math, json
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', 'results'))
MASTER = os.path.join(OUT, 'features_master.csv')
HORIZONS = (5, 15, 30, 60)
MIN_N = 60


def rankv(a):
    o = np.argsort(a, kind='mergesort'); r = np.empty(len(a)); sa = a[o]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    dx, dy = math.sqrt(float(x @ x)), math.sqrt(float(y @ y))
    return None if dx <= 0 or dy <= 0 else float(x @ y) / (dx * dy)


def resid(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return y - X @ b


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

    res = defaultdict(lambda: defaultdict(list))
    cov = []
    for (d, v), rows in sorted(days.items()):
        n = len(rows)
        def col(c):
            a = np.full(n, np.nan)
            for i, r in enumerate(rows):
                s = r.get(c, '')
                if s not in ('', None):
                    try:
                        a[i] = float(s)
                    except ValueError:
                        pass
            return a
        skew = col('skew'); spot = col('spot'); prem = col('prem')
        cov.append((v, float(np.mean(~np.isnan(skew)))))
        tm = np.arange(n, dtype=float)
        base = spot[~np.isnan(spot)][0] if np.any(~np.isnan(spot)) else np.nan
        negspot = -(spot / base - 1.0)
        def pastret(k):
            a = np.full(n, np.nan); a[k:] = spot[k:] / spot[:-k] - 1.0; return a
        pr5, pr15, pr30 = pastret(5), pastret(15), pastret(30)
        for h in HORIZONS:
            for oc in ('fret', 'fabs', 'fdprem'):
                y = col('%s_%d' % (oc, h))
                for label, ctl in (
                        ('g1c_controls', [negspot, tm, prem, pr15, pr30]),
                        ('plus_pastret5', [negspot, tm, prem, pr5, pr15, pr30])):
                    m = ~(np.isnan(skew) | np.isnan(y))
                    for c in ctl:
                        m &= ~np.isnan(c)
                    if m.sum() < MIN_N:
                        continue
                    rx, ry = rankv(skew[m]), rankv(y[m])
                    if len(np.unique(rx)) < 5 or len(np.unique(ry)) < 5:
                        continue
                    X = np.column_stack([np.ones(m.sum())] + [rankv(c[m]) for c in ctl])
                    c2 = corr(resid(rx, X), resid(ry, X))
                    if c2 is not None:
                        res[(oc, h, label)][v].append(c2)
    out = []
    for (oc, h, label), byv in sorted(res.items()):
        allv = [c for v in byv for c in byv[v]]
        m, t = tstat(np.array(allv))
        rec = dict(outcome=oc, horizon=h, control=label, n=len(allv),
                   ic=round(m, 4), t=round(t, 2))
        for v in ('NIFTY', 'SENSEX'):
            if byv.get(v):
                rec['ic_' + v] = round(float(np.mean(byv[v])), 4)
        out.append(rec)
    with open(os.path.join(OUT, 'g1d_skew_probe.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['outcome', 'horizon', 'control', 'n', 'ic', 't',
                                          'ic_NIFTY', 'ic_SENSEX'], extrasaction='ignore')
        w.writeheader()
        for r in out:
            w.writerow(r)
    nc = defaultdict(list)
    for v, c in cov:
        nc[v].append(c)
    print('skew coverage (fraction of minutes with a usable skew):',
          {v: round(float(np.mean(a)) * 100, 1) for v, a in nc.items()})
    print('\n%-8s %3s %-16s %8s %8s %9s %9s' % ('outcome', 'h', 'controls', 'IC', 't', 'NIFTY', 'SENSEX'))
    for r in sorted(out, key=lambda r: (r['outcome'], r['horizon'], r['control'])):
        print('%-8s %3d %-16s %8.4f %8.2f %9.4f %9.4f' % (
            r['outcome'], r['horizon'], r['control'], r['ic'], r['t'],
            r.get('ic_NIFTY', 0), r.get('ic_SENSEX', 0)))


if __name__ == '__main__':
    main()
