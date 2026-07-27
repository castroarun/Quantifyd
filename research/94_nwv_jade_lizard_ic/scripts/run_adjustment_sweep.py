#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 — WHEN/HOW to adjust the pivot-anchored jade lizard (research/94)
==========================================================================
On the locked paper-book template (user JL: short put @S1 +550 wing, short call
@R2 +200 wing, next-week expiry, all non-ignore weeks), test price-action
adjustment triggers at EOD granularity, real option prices 2020-02 -> 2026-07:

TRIGGERS (first day it fires, one adjustment per week):
  T1 breach_strike : daily close < short_put strike  OR > short_call strike
  T2 breach_pivot  : daily close < weekly S1         OR > weekly R2

ACTIONS on trigger:
  hold        : do nothing (baseline)
  exit_all    : flatten everything at that EOD
  exit_side   : close only the threatened side, ride the rest to Friday
  roll_away   : close threatened side; re-sell it further OUT (put: floor50(close-200)
                wing -550; call: ceil50(close+200) wing +200)  [defensive]
  roll_chase  : close the UNTHREATENED side; re-sell it CLOSER to spot for credit
                (Arun's manual habit per mentor W30 — testing it honestly)
All fills at that day's EOD close prices +/- nothing extra; costs 0.5pt/leg/side
on every leg traded. Friday EOD final exit for all arms. 10 lots.
"""
import sys, os, csv, statistics, math
from datetime import date, datetime, timedelta
from collections import defaultdict

sys.path.insert(0, '/home/arun/quantifyd')
sys.path.insert(0, '/home/arun/quantifyd/research')
sys.path.insert(0, '/home/arun/quantifyd/research/94_nwv_jade_lizard_ic/scripts')
from nwv_phase1_regime import (build_view, mondays, week_sessions, expiries_after,
                               opt_eod, seg_hlc, daily_close)
from services.nwv_engine import compute_pivots
from run_jl_ic_sweep import traded, r50d, r50u

LOT, LOTS = 65, 10
COST = 0.5
START, END = date(2020, 2, 3), date(2026, 7, 13)
OUT = '/home/arun/quantifyd/research/94_nwv_jade_lizard_ic/results'

def main():
    P = defaultdict(list); Y = defaultdict(lambda: defaultdict(float))
    trig_count = defaultdict(int)
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
        s1 = piv['s1']
        r2 = piv.get('r2') or piv['pp'] + (pw[0] - pw[1])
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

        def px(d, k, ot):
            return opt_eod(d, expiry, k, ot)

        def lv(d, legs):
            t = 0.0
            for q, k, ot in legs:
                p = px(d, k, ot)
                if p is None:
                    return None
                t += q * p
            return t

        e0 = lv(ms, legs0)
        if e0 is None or -e0 <= 0:
            continue
        year = ms[:4]

        # cash-flow engine: flows in POINTS (short sells +, buys -), legs mutable
        def run_policy(trigger, action):
            flows = -e0 - 4 * COST                    # entry credit net of entry costs
            legs = list(legs0)
            adjusted = False
            for i, d in enumerate(sess[1:], 1):       # check from Tuesday
                c = daily_close[d]
                if adjusted or action == 'hold':
                    continue
                if trigger == 'strike':
                    put_hit, call_hit = c < sp, c > sc
                else:
                    put_hit, call_hit = c < s1, c > r2
                if not (put_hit or call_hit):
                    continue
                adjusted = True
                trig_count[(trigger, 'put' if put_hit else 'call')] += 1
                side = 'PE' if put_hit else 'CE'
                threatened = [(q, k, o) for q, k, o in legs if o == side]
                other      = [(q, k, o) for q, k, o in legs if o != side]
                if action == 'exit_all':
                    val = lv(d, legs)
                    if val is None: return None
                    return (flows + val - 4 * COST) * LOT * LOTS
                if action == 'exit_side':
                    val = lv(d, threatened)
                    if val is None: return None
                    flows += val - 2 * COST
                    legs = other
                elif action == 'roll_away':
                    val = lv(d, threatened)
                    if val is None: return None
                    flows += val - 2 * COST
                    if side == 'PE':
                        nsp = r50d(c - 200); nlegs = [(-1, nsp, 'PE'), (1, nsp - 550, 'PE')]
                    else:
                        nsc = r50u(c + 200); nlegs = [(-1, nsc, 'CE'), (1, nsc + 200, 'CE')]
                    nval = lv(d, nlegs)
                    if nval is None: return None
                    flows += -nval - 2 * COST
                    legs = other + nlegs
                elif action == 'roll_chase':
                    val = lv(d, other)
                    if val is None: return None
                    flows += val - 2 * COST
                    if side == 'PE':      # market fell -> roll CALL spread closer
                        nsc = r50u(c + 150); nlegs = [(-1, nsc, 'CE'), (1, nsc + 200, 'CE')]
                    else:                 # market rallied -> roll PUT spread closer
                        nsp = r50d(c - 150); nlegs = [(-1, nsp, 'PE'), (1, nsp - 550, 'PE')]
                    nval = lv(d, nlegs)
                    if nval is None: return None
                    flows += -nval - 2 * COST
                    legs = threatened + nlegs
            if not legs:
                return flows * LOT * LOTS
            val = lv(sess[-1], legs)
            if val is None: return None
            return (flows + val - len(legs) * COST) * LOT * LOTS

        for trig in ('strike', 'pivot'):
            for act in ('hold', 'exit_all', 'exit_side', 'roll_away', 'roll_chase'):
                key = f'{trig}:{act}' if act != 'hold' else 'hold'
                pnl = run_policy(trig, act)
                if pnl is not None:
                    P[key].append(pnl)
                    Y[key][year] += pnl

    print(f"trigger fire counts: {dict(trig_count)}")
    print(f"\n{'policy':22s} {'n':>4s} {'win%':>6s} {'avg':>9s} {'tot':>11s} {'PF':>5s} {'worst':>10s} {'t':>5s}")
    order = ['hold'] + [f'{t}:{a}' for t in ('strike','pivot')
                        for a in ('exit_all','exit_side','roll_away','roll_chase')]
    rows = []
    for k in order:
        p = P.get(k)
        if not p:
            continue
        n = len(p); w = len([x for x in p if x > 0])
        gw = sum(x for x in p if x > 0); gl = -sum(x for x in p if x < 0)
        tstat = statistics.mean(p) / (statistics.stdev(p) / math.sqrt(n))
        print(f"{k:22s} {n:4d} {100*w/n:6.1f} {statistics.mean(p):9,.0f} {sum(p):11,.0f} "
              f"{(gw/gl if gl else 999):5.2f} {min(p):10,.0f} {tstat:5.2f}")
        rows.append([k, n, round(100*w/n,1), round(statistics.mean(p)), round(sum(p)),
                     round(gw/gl if gl else 999, 2), round(min(p)), round(tstat, 2)])
    print(f"\nper-year totals (Rs k):")
    years = sorted({y for k in Y for y in Y[k]})
    print(f"{'policy':22s} " + " ".join(y.rjust(7) for y in years))
    for k in order:
        if k not in Y: continue
        print(f"{k:22s} " + " ".join(f"{Y[k].get(y,0)/1000:7.0f}" for y in years))
    with open(f'{OUT}/adjustment_ranking.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['policy','n','winpct','avg','tot','pf','worst','t'])
        w.writerows(rows)
    print(f"\nwrote {OUT}/adjustment_ranking.csv")

if __name__ == '__main__':
    main()
