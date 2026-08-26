#!/usr/bin/env python3
"""research/128 stage 4 - plateau map, family-wise honesty, OOS split, cost sensitivity.

Reads results/arm_episode.csv + results/arm_summary.csv and writes
results/plateau.csv, results/analysis.txt (the numbers that go into RESULTS.md).
"""
import csv, json, os, math, re
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
OUT = []


def P(s=''):
    print(s, flush=True)
    OUT.append(s)


def tstat(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def main():
    rows = list(csv.DictReader(open(os.path.join(RES, 'arm_episode.csv'))))
    net = defaultdict(dict)          # arm -> eid -> net
    meta = {}
    for r in rows:
        net[r['arm']][r['eid']] = float(r['net_lot'])
        meta[r['eid']] = dict(day=r['day'], weekday=r['weekday'], dte=int(r['dte']),
                              system=r['system'])
    eids = sorted(meta)
    days = sorted({meta[e]['day'] for e in eids})
    split = days[int(len(days) * 0.6)]
    IS = [e for e in eids if meta[e]['day'] < split]
    OOS = [e for e in eids if meta[e]['day'] >= split]
    be = net['BE_ONLY']
    inc = net['INCUMBENT']

    P('=' * 92)
    P('RESEARCH 128 - SENSEX naked-survivor trail | analysis')
    P('=' * 92)
    P('episodes %d | days %d (%s .. %s) | IS<%s n=%d | OOS>=%s n=%d'
      % (len(eids), len(days), days[0], days[-1], split, len(IS), split, len(OOS)))
    P()

    # ---------------- plateau map --------------------------------------------
    pat = re.compile(r'CEIL_p(\d+)_m([\d.]+)_N(\d+)_(SEED|COLD)')
    cells = []
    for arm in net:
        m = pat.fullmatch(arm)
        if not m:
            continue
        p, mu, N, sd = int(m.group(1)), float(m.group(2)), int(m.group(3)), m.group(4)
        v = np.array([net[arm][e] - be[e] for e in eids])
        cells.append(dict(arm=arm, period=p, mult=mu, N=N, seed=sd,
                          mean=float(np.mean([net[arm][e] for e in eids])),
                          vs_be=float(v.mean()), t=tstat(v),
                          worst=float(min(net[arm][e] for e in eids)),
                          is_mean=float(np.mean([net[arm][e] for e in IS])),
                          oos_mean=float(np.mean([net[arm][e] for e in OOS]))))
    with open(os.path.join(RES, 'plateau.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)

    P('PLATEAU MAP - mean net Rs/lot by ST period x multiplier (averaged over N and seed)')
    periods = sorted({c['period'] for c in cells})
    mults = sorted({c['mult'] for c in cells})
    P('  period |' + ''.join('  m%.1f ' % m for m in mults) + '   row-avg')
    for p in periods:
        line = '   %4d  |' % p
        vals = []
        for mu in mults:
            v = [c['mean'] for c in cells if c['period'] == p and c['mult'] == mu]
            line += ' %6.0f' % np.mean(v)
            vals += v
        P(line + '   %6.0f' % np.mean(vals))
    line = '  col-avg |'
    for mu in mults:
        line += ' %6.0f' % np.mean([c['mean'] for c in cells if c['mult'] == mu])
    P(line)
    P('  (BE_ONLY = %.0f   INCUMBENT = %.0f   HOLD_EOD = %.0f)'
      % (np.mean(list(be.values())), np.mean(list(inc.values())),
         np.mean(list(net['HOLD_EOD'].values()))))
    P()
    P('CONFIRM-POLL sensitivity (N minute-polls, averaged over period/mult/seed)')
    for N in sorted({c['N'] for c in cells}):
        v = [c for c in cells if c['N'] == N]
        P('   N=%d  mean %6.0f  vsBE %+6.0f  beat-BE %3d/%3d  worst %8.0f'
          % (N, np.mean([x['mean'] for x in v]), np.mean([x['vs_be'] for x in v]),
             sum(1 for x in v if x['vs_be'] > 0), len(v), min(x['worst'] for x in v)))
    P()
    P('SEEDED vs COLD warm-up (averaged over the whole ST grid)')
    for sd in ('SEED', 'COLD'):
        v = [c for c in cells if c['seed'] == sd]
        P('   %-4s mean %6.0f  vsBE %+6.0f  beat-BE %3d/%3d  worst %8.0f  median-worst %8.0f'
          % (sd, np.mean([x['mean'] for x in v]), np.mean([x['vs_be'] for x in v]),
             sum(1 for x in v if x['vs_be'] > 0), len(v),
             min(x['worst'] for x in v), np.median([x['worst'] for x in v])))
    P()

    # ---------------- family-wise honesty -------------------------------------
    P('FAMILY-WISE HONESTY')
    nc = len(cells)
    beats = sum(1 for c in cells if c['vs_be'] > 0)
    P('   ST-ceiling cells tried: %d | beating BE_ONLY: %d (%.0f%%) | beating INCUMBENT: %d'
      % (nc, beats, 100.0 * beats / nc,
         sum(1 for c in cells if c['mean'] > np.mean(list(inc.values())))))
    # sign test on the cell population (is the family, not a cell, better than BE?)
    from math import comb
    pbin = sum(comb(nc, k) for k in range(beats, nc + 1)) / 2.0 ** nc
    P('   sign test on cells vs BE_ONLY: p = %.3g  (null: a cell is better than BE with p=0.5)' % pbin)
    fam = np.array([np.mean([net[c['arm']][e] for c in cells]) - be[e] for e in eids])
    P('   FAMILY-AVERAGE arm (equal-weight over all %d cells) vs BE_ONLY: %+.0f Rs/lot  t=%.2f'
      % (nc, fam.mean(), tstat(fam)))
    famI = np.array([np.mean([net[c['arm']][e] for c in cells]) - inc[e] for e in eids])
    P('   FAMILY-AVERAGE arm vs INCUMBENT: %+.0f Rs/lot  t=%.2f' % (famI.mean(), tstat(famI)))
    best = max(cells, key=lambda c: c['vs_be'])
    n_arms = len(net)
    P('   best single cell %s: vsBE %+.0f t=%.2f' % (best['arm'], best['vs_be'], best['t']))
    P('   Bonferroni over %d arms needs |t| >= ~%.2f (alpha 5%%, df %d) -> best cell does %s'
      % (n_arms, 3.55, len(eids) - 1, 'NOT survive' if best['t'] < 3.55 else 'survive'))
    P()

    # ---------------- IS-selection -> OOS -------------------------------------
    P('OUT-OF-SAMPLE (params chosen on IS days only, scored on OOS days)')
    is_best = max(cells, key=lambda c: c['is_mean'])
    P('   IS-best cell            : %-24s IS %6.0f -> OOS %6.0f' %
      (is_best['arm'], is_best['is_mean'], is_best['oos_mean']))
    # plateau centroid: the cell nearest the centre of the top quartile by vs_be
    top = sorted(cells, key=lambda c: -c['is_mean'])[:max(4, len(cells) // 4)]
    cp = np.mean([c['period'] for c in top]); cm = np.mean([c['mult'] for c in top])
    cn = np.mean([c['N'] for c in top])
    P('   IS top-quartile centroid: period %.1f  mult %.2f  N %.1f' % (cp, cm, cn))
    cen = min(cells, key=lambda c: ((c['period'] - cp) / 4) ** 2 + ((c['mult'] - cm) / 0.7) ** 2
              + ((c['N'] - cn) / 1.4) ** 2 + (0.0 if c['seed'] == 'SEED' else 0.05))
    P('   plateau-centroid cell   : %-24s IS %6.0f -> OOS %6.0f' %
      (cen['arm'], cen['is_mean'], cen['oos_mean']))
    for lbl in ('BE_ONLY', 'INCUMBENT', 'HOLD_EOD'):
        P('   %-24s: IS %6.0f -> OOS %6.0f'
          % (lbl, np.mean([net[lbl][e] for e in IS]), np.mean([net[lbl][e] for e in OOS])))
    P('   median OOS of ALL %d ST cells: %.0f  (share beating BE_ONLY OOS: %.0f%%)'
      % (nc, np.median([c['oos_mean'] for c in cells]),
         100.0 * sum(1 for c in cells if c['oos_mean'] > np.mean([be[e] for e in OOS])) / nc))
    P()

    # ---------------- recommended arm detail ----------------------------------
    rec = os.environ.get('R128_ARM', cen['arm'])
    P('RECOMMENDED ARM: %s' % rec)
    v = np.array([net[rec][e] for e in eids])
    dbe = np.array([net[rec][e] - be[e] for e in eids])
    dinc = np.array([net[rec][e] - inc[e] for e in eids])
    P('   mean %+.0f  median %+.0f  worst %+.0f  win%% %.0f  t %.2f'
      % (v.mean(), np.median(v), v.min(), 100 * (v > 0).mean(), tstat(v)))
    P('   vs BE_ONLY %+.0f (t %.2f)   vs INCUMBENT %+.0f (t %.2f)   vs HOLD %+.0f'
      % (dbe.mean(), tstat(dbe), dinc.mean(), tstat(dinc),
         dbe.mean() + np.mean([be[e] - net['HOLD_EOD'][e] for e in eids])))
    P('   per DTE:')
    for d in sorted({meta[e]['dte'] for e in eids}):
        s = [e for e in eids if meta[e]['dte'] == d]
        P('      DTE%d n=%2d  rec %6.0f  BE %6.0f  INC %6.0f  HOLD %7.0f'
          % (d, len(s), np.mean([net[rec][e] for e in s]), np.mean([be[e] for e in s]),
             np.mean([inc[e] for e in s]), np.mean([net['HOLD_EOD'][e] for e in s])))
    P('   per weekday:')
    for d in ('Mon', 'Tue', 'Wed', 'Thu', 'Fri'):
        s = [e for e in eids if meta[e]['weekday'] == d]
        if not s:
            continue
        P('      %s n=%2d  rec %6.0f  BE %6.0f  INC %6.0f  HOLD %7.0f'
          % (d, len(s), np.mean([net[rec][e] for e in s]), np.mean([be[e] for e in s]),
             np.mean([inc[e] for e in s]), np.mean([net['HOLD_EOD'][e] for e in s])))
    P('   per system:')
    for d in ('ATM', 'ATM4'):
        s = [e for e in eids if meta[e]['system'] == d]
        P('      %-4s n=%2d  rec %6.0f  BE %6.0f  INC %6.0f  HOLD %7.0f'
          % (d, len(s), np.mean([net[rec][e] for e in s]), np.mean([be[e] for e in s]),
             np.mean([inc[e] for e in s]), np.mean([net['HOLD_EOD'][e] for e in s])))
    P()

    # ---------------- cost sensitivity ----------------------------------------
    P('COST SENSITIVITY (measured stop-slip 6.548 pt/leg-side scaled; Rs/lot mean)')
    ex = defaultdict(dict)
    for r in rows:
        ex[r['arm']][r['eid']] = (float(r['exit_px']), r['reason'], float(r['entry']))
    LOT = 20
    for mult in (1.0, 1.5, 2.0, 3.0):
        line = '   slip x%.1f :' % mult
        for lbl in (rec, 'BE_ONLY', 'INCUMBENT', 'HOLD_EOD'):
            tot = []
            for e in eids:
                px, reason, E = ex[lbl][e]
                extra = 0.0 if reason == 'EOD' else (6.548 * (mult - 1.0) * LOT)
                tot.append(net[lbl][e] - extra)
            line += '  %-10s %6.0f |' % (lbl[:10], np.mean(tot))
        P(line)
    P()
    P('BREAK-EVEN: the recommended arm still beats BE_ONLY until the stop-slip reaches ~%.1f pt/side'
      % (6.548 + dbe.mean() / LOT / max(1e-9, (np.mean([1.0 if ex[rec][e][1] != 'EOD' else 0.0 for e in eids])
                                                - np.mean([1.0 if ex['BE_ONLY'][e][1] != 'EOD' else 0.0 for e in eids])))))
    with open(os.path.join(RES, 'analysis.txt'), 'w') as f:
        f.write('\n'.join(OUT) + '\n')
    print('\nwrote results/analysis.txt and results/plateau.csv')


if __name__ == '__main__':
    main()
