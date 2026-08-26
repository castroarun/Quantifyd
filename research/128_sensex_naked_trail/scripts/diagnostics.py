#!/usr/bin/env python3
"""research/128 stage 5 - mechanism diagnostics.

(a) proves the self-trigger: where does the INCUMBENT's SuperTrend value sit relative to the
    live premium at the moment it is first written into sl_price?
(b) prices the COLD warm-up: how long is the survivor trail-less, and how much of the episode
    is that?
(c) round-trip timing: when does the survivor bottom, and when does it come back?
"""
import json, os, sys
import numpy as np
sys.path.insert(0, '/home/arun/quantifyd')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_arms import tv_supertrend, short_trail, bars_from, BAR, LOT

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')


def m2hm(m):
    return '%02d:%02d' % (m // 60, m % 60)


def main():
    eps = [json.loads(l) for l in open(os.path.join(RES, 'paths.jsonl'))]
    out = []

    def P(s=''):
        print(s, flush=True)
        out.append(s)

    P('=' * 88)
    P('(a) INCUMBENT self-trigger diagnostic - TV SuperTrend(7,3) vs the live premium')
    P('=' * 88)
    below = warm = 0
    firstwarm = []
    for e in eps:
        bars, cur, bkt = [], None, None
        for m, p in e['path']:
            b = (m // BAR) * BAR
            if b != bkt:
                if cur is not None:
                    bars.append(cur)
                cur = dict(open=p, high=p, low=p, close=p)
                bkt = b
            else:
                cur['high'] = max(cur['high'], p); cur['low'] = min(cur['low'], p); cur['close'] = p
            seq = bars + [cur]
            st = tv_supertrend(seq, 7, 3.0) if len(seq) >= 9 else None
            if st is None:
                continue
            warm += 1
            firstwarm.append(m - e['naked_m'])
            if st < p:
                below += 1
            break
    P('episodes where the ST value is first produced : %d / %d' % (warm, len(eps)))
    P('  ... and it lands BELOW the live premium      : %d  (%.0f%%)  -> instant self-trigger'
      % (below, 100.0 * below / max(1, warm)))
    P('  minutes from naked to first ST value        : median %.0f  (min %d, max %d)'
      % (np.median(firstwarm), min(firstwarm), max(firstwarm)))
    P()

    P('=' * 88)
    P('(b) CEILING design - is the stop correctly ABOVE the premium when it arms?')
    P('=' * 88)
    for seed in (1, 0):
        ok = tot = 0
        lag = []
        for e in eps:
            if seed:
                bars, cur, bkt = bars_from(e.get('pre') or [], BAR)
                nb = (e['naked_m'] // BAR) * BAR
                if cur is not None and bkt != nb:
                    bars = bars + [cur]; cur, bkt = None, None
            else:
                bars, cur, bkt = [], None, None
            st = short_trail(bars, 7, 3.0) if len(bars) >= 8 else None
            armed = None
            for m, p in e['path']:
                if st is not None and armed is None:
                    armed = (m, p, st)
                    break
                b = (m // BAR) * BAR
                if b != bkt:
                    if cur is not None:
                        bars.append(cur)
                        st = short_trail(bars, 7, 3.0) if len(bars) >= 8 else None
                    cur = dict(open=p, high=p, low=p, close=p); bkt = b
                else:
                    cur['high'] = max(cur['high'], p); cur['low'] = min(cur['low'], p); cur['close'] = p
            if armed:
                tot += 1
                lag.append(armed[0] - e['naked_m'])
                if armed[2] > armed[1]:
                    ok += 1
        P('  %-5s : armed on %d/%d episodes | stop ABOVE premium at arming %d (%.0f%%) | '
          'minutes trail-less: median %.0f max %d'
          % ('SEED' if seed else 'COLD', tot, len(eps), ok, 100.0 * ok / max(1, tot),
             np.median(lag), max(lag)))
    P()

    P('=' * 88)
    P('(c) ROUND-TRIP timing')
    P('=' * 88)
    tmin, trec, dur = [], [], []
    nrt = 0
    for e in eps:
        E = e['entry']
        pr = [p for _, p in e['path']]
        mm = [m for m, _ in e['path']]
        i = int(np.argmin(pr))
        tmin.append(mm[i] - e['naked_m'])
        dur.append(mm[-1] - e['naked_m'])
        back = [mm[j] for j in range(i, len(pr)) if pr[j] >= E]
        if back:
            nrt += 1
            trec.append(back[0] - mm[i])
    P('  episode length (naked -> 15:15)      : median %.0f min (min %d, max %d)'
      % (np.median(dur), min(dur), max(dur)))
    P('  time from naked to the LOW           : median %.0f min' % np.median(tmin))
    P('  full round-trips back to entry       : %d / %d = %.1f%%' % (nrt, len(eps), 100.0 * nrt / len(eps)))
    P('  ... time from the LOW back to entry  : median %.0f min (min %d, max %d)'
      % (np.median(trec), min(trec), max(trec)))
    P()
    P('  giveback distribution (recovery_frac = (max after low - low) / (entry - low))')
    rec = []
    for e in eps:
        E = e['entry']; pr = [p for _, p in e['path']]
        i = int(np.argmin(pr)); mn = pr[i]
        rec.append((max(pr[i:]) - mn) / (E - mn) if E > mn else 0.0)
    for q in (10, 25, 50, 75, 90, 95):
        P('    p%-3d %.2f' % (q, np.percentile(rec, q)))
    P('    share >= 1.00 (full round trip)   : %.1f%%' % (100.0 * np.mean(np.array(rec) >= 1.0)))
    P('    share >= 0.50 (half given back)   : %.1f%%' % (100.0 * np.mean(np.array(rec) >= 0.5)))
    P('    share <= 0.10 (clean one-way decay): %.1f%%' % (100.0 * np.mean(np.array(rec) <= 0.1)))
    with open(os.path.join(RES, 'diagnostics.txt'), 'w') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
