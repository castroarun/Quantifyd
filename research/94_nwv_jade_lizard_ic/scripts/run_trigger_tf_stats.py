#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 — trigger timeframe (30m/60m/120m/daily) + full distribution stats
===========================================================================
On the locked paper-book template (user JL, all non-ignore weeks, 10 lots):
pivot exit-side triggers evaluated on N-minute closes beyond weekly S1/R2
(spot from the 30-min series; option fills at that day's EOD — real prices).
Also full stats (win%, median, quartiles, worst/best, max drawdown of the
chronological equity curve) for: hold, pt50_stop1x, and each trigger TF.
"""
import sys, statistics, math
from datetime import date, datetime, timedelta
from collections import defaultdict

sys.path.insert(0, '/home/arun/quantifyd')
sys.path.insert(0, '/home/arun/quantifyd/research')
sys.path.insert(0, '/home/arun/quantifyd/research/94_nwv_jade_lizard_ic/scripts')
from nwv_phase1_regime import (build_view, mondays, week_sessions, expiries_after,
                               opt_eod, seg_hlc, m30_by_day)
from services.nwv_engine import compute_pivots
from run_jl_ic_sweep import traded, r50d, r50u

LOT, LOTS = 65, 10
COST = 0.5
START, END = date(2020, 2, 3), date(2026, 7, 13)

def day_closes(d, tf):
    """Bar-close series for day d at tf minutes (from 30-min bars)."""
    bars = sorted(m30_by_day.get(d, []), key=lambda b: b['date'])
    if not bars:
        return []
    if tf == 30:
        return [(b['date'][11:16], b['close']) for b in bars]
    step = tf // 30
    out = [( bars[i]['date'][11:16], bars[i]['close'])
           for i in range(step - 1, len(bars), step)]
    return out

def main():
    series = defaultdict(list)          # policy -> [(week, pnl)]
    for mon in mondays(START, END):
        v = build_view(mon)
        if not v or v['view'] == 'ignore':
            continue
        ms, spot = v['ms'], v['spot']
        sess = week_sessions(ms)
        if len(sess) < 2:
            continue
        pw = seg_hlc((mon - timedelta(days=7)).isoformat(),
                     (mon - timedelta(days=3)).isoformat())
        if not pw:
            continue
        piv = dict(compute_pivots(*pw))
        s1 = piv['s1']; r2 = piv.get('r2') or piv['pp'] + (pw[0] - pw[1])
        exps = [e for e in expiries_after(ms)
                if (datetime.strptime(e, '%Y-%m-%d').date() - mon).days >= 6]
        if not exps:
            continue
        expiry = exps[0]
        sp = min(r50d(s1), r50d(spot - 100)); lp = sp - 550
        sc = max(r50d(r2), r50u(spot + 200)); lc = sc + 200
        legs0 = [(-1, sp, 'PE'), (1, lp, 'PE'), (-1, sc, 'CE'), (1, lc, 'CE')]
        if not all(traded(ms, expiry, k, o) for _, k, o in legs0):
            continue

        def lv(d, legs):
            t = 0.0
            for q, k, ot in legs:
                p = opt_eod(d, expiry, k, ot)
                if p is None:
                    return None
                t += q * p
            return t

        e0 = lv(ms, legs0)
        if e0 is None or -e0 <= 0:
            continue
        credit = -e0

        # ---- hold ----
        val = lv(sess[-1], legs0)
        if val is None:
            continue
        pnl_hold = (-e0 + val - 8 * COST) * LOT * LOTS
        series['hold'].append((ms, pnl_hold))

        # ---- pt50_stop1x (EOD combined premium) ----
        pnl = None
        for d in sess[1:]:
            x = lv(d, legs0)
            if x is None:
                continue
            mtm = (x - e0)
            if mtm >= 0.5 * credit or mtm <= -1.0 * credit:
                pnl = (-e0 + x - 8 * COST) * LOT * LOTS
                break
        series['pt50_stop1x'].append((ms, pnl if pnl is not None else pnl_hold))

        # ---- pivot exit-side at each trigger TF ----
        for tf, name in ((30, 'x_side_30m'), (60, 'x_side_60m'),
                         (120, 'x_side_120m'), (1440, 'x_side_daily')):
            flows = -e0 - 4 * COST
            legs = list(legs0)
            done = False
            for d in sess[1:]:
                if done:
                    break
                closes = ([('EOD', None)] if tf == 1440 else day_closes(d, tf))
                if tf == 1440:
                    bars = sorted(m30_by_day.get(d, []), key=lambda b: b['date'])
                    closes = [('EOD', bars[-1]['close'])] if bars else []
                for _, c in closes:
                    if c is None:
                        continue
                    side = 'PE' if c < s1 else ('CE' if c > r2 else None)
                    if not side:
                        continue
                    thr = [(q, k, o) for q, k, o in legs if o == side]
                    if not thr:                    # that side already gone
                        continue
                    valx = lv(d, thr)              # fill at trigger-day EOD
                    if valx is None:
                        continue
                    flows += valx - 2 * COST
                    legs = [(q, k, o) for q, k, o in legs if o != side]
                    if not legs:
                        done = True
                    break                          # one action per day max
            if legs:
                valf = lv(sess[-1], legs)
                if valf is None:
                    continue
                flows += valf - len(legs) * COST
            series[name].append((ms, flows * LOT * LOTS))

    print(f"{'policy':14s} {'n':>4s} {'win%':>5s} {'avg':>8s} {'med':>8s} {'p25':>9s} "
          f"{'p75':>7s} {'worst':>9s} {'best':>8s} {'tot':>11s} {'maxDD':>9s} {'t':>5s}")
    for name in ('hold', 'pt50_stop1x', 'x_side_30m', 'x_side_60m',
                 'x_side_120m', 'x_side_daily'):
        rows = sorted(series[name])
        p = [x for _, x in rows]
        if not p:
            continue
        n = len(p)
        cum = peak = 0.0; dd = 0.0
        for x in p:
            cum += x
            peak = max(peak, cum)
            dd = min(dd, cum - peak)
        q = statistics.quantiles(p, n=4)
        tstat = statistics.mean(p) / (statistics.stdev(p) / math.sqrt(n))
        print(f"{name:14s} {n:4d} {100*len([x for x in p if x>0])/n:5.1f} "
              f"{statistics.mean(p):8,.0f} {statistics.median(p):8,.0f} {q[0]:9,.0f} "
              f"{q[2]:7,.0f} {min(p):9,.0f} {max(p):8,.0f} {sum(p):11,.0f} "
              f"{dd:9,.0f} {tstat:5.2f}")

if __name__ == '__main__':
    main()
