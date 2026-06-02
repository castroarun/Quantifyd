#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IC SKEW SWEEP for the BEARISH view — find the most bearish tilt that still pays.
================================================================================
User wants the IC aligned with the view (neutral -> slightly bearish), not
bull-skewed. Sweep the IC centre offset from spot (>0 bull, <0 bear), short
strikes at centre +/- HW, 200 wings. Exits: TP 50% credit, stop -1x credit,
else Friday. Real EOD 2024-26 + modeled 2020-26.
"""
import sys
from datetime import date, datetime, timedelta
from collections import defaultdict
sys.path.insert(0, '/home/arun/quantifyd'); sys.path.insert(0, '/home/arun/quantifyd/research')
from nwv_phase1_regime import (
    build_view, daily_close, mondays, week_sessions, expiries_after, opt_eod,
    realized_vol, price as bs_price, MULT, round_to_strike,
    VIEW_BEARISH, VIEW_NTB, VIEW_BULLISH, VIEW_NTBULL,
)
LOT, LOTS, WIDTH, HW = 65, 5, 200, 250

def rk(x): return round_to_strike(x, 100)

# offset (pts from spot) -> label. negative = bearish skew (profit zone below spot)
SKEWS = [(0, 'neutral'), (-25, 's.bear-25'), (-50, 's.bear-50'),
         (-75, 's.bear-75'), (-150, 'bear-150')]

def ic_legs(spot, off):
    c = spot + off
    sp = rk(c - HW); sc = rk(c + HW)
    return [(-1, sp, 'PE'), (1, sp - WIDTH, 'PE'), (-1, sc, 'CE'), (1, sc + WIDTH, 'CE')]

def sim(legs, sess, leg_px):
    def lv(d):
        t = 0.0
        for q, k, ot in legs:
            px = leg_px(d, k, ot)
            if px is None:
                return None
            t += q * px
        return t
    e = lv(sess[0])
    if e is None or -e <= 0:
        return None
    credit = -e; cap = credit * LOT * LOTS
    exit_pnl = None; last = None
    for d in sess:
        v = lv(d)
        if v is None:
            continue
        last = v; mtm = (v - e) * LOT * LOTS
        if exit_pnl is None and mtm >= 0.50 * cap:
            exit_pnl = mtm
        elif exit_pnl is None and mtm <= -1.0 * cap:
            exit_pnl = mtm
    return exit_pnl if exit_pnl is not None else (last - e) * LOT * LOTS

def run(real_only, want_side):
    start = date(2024, 3, 4) if real_only else date(2020, 2, 3)
    out = defaultdict(list)
    for mon in mondays(start, date(2026, 3, 23)):
        v = build_view(mon)
        if not v or v['view'] not in (VIEW_BEARISH, VIEW_NTB, VIEW_BULLISH, VIEW_NTBULL):
            continue
        side = 'BEAR' if v['view'] in (VIEW_BEARISH, VIEW_NTB) else 'BULL'
        if side != want_side:
            continue
        sess = week_sessions(v['ms'])
        if len(sess) < 2:
            continue
        if real_only:
            exps = expiries_after(v['ms'])
            if len(exps) < 2:
                continue
            exp = exps[1]; ed = datetime.strptime(exp, '%Y-%m-%d').date()
            leg_px = lambda d, k, ot: opt_eod(d, exp, k, ot)
        else:
            rv = realized_vol(v['ms'])
            if not rv:
                continue
            iv = MULT * rv; ed = mon + timedelta(days=8)
            leg_px = lambda d, k, ot: bs_price(daily_close[d], k,
                        (ed - datetime.strptime(d, '%Y-%m-%d').date()).days, iv, ot)
        for off, lbl in SKEWS:
            p = sim(ic_legs(v['spot'], off), sess, leg_px)
            if p is not None:
                out[lbl].append(p)
    tag = "REAL 2024-26" if real_only else "MODELED 2020-26"
    print(f"\n=== {want_side} view · IC skew sweep · {tag} ===")
    for _, lbl in SKEWS:
        p = out[lbl]
        if not p:
            print(f"  {lbl:12s}: (none)"); continue
        n = len(p); w = sum(1 for x in p if x > 0)
        gw = sum(x for x in p if x > 0); gl = -sum(x for x in p if x < 0)
        print(f"  {lbl:12s}: n={n:3d} win%={100*w/n:5.1f} avg=Rs{sum(p)/n:7.0f} "
              f"PF={gw/gl if gl else 999:5.2f} worst={min(p):8.0f}")

if __name__ == '__main__':
    run(True, 'BEAR'); run(False, 'BEAR')
    run(True, 'BULL'); run(False, 'BULL')
