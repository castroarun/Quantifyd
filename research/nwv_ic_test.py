#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skewed Iron Condor vs Debit Spread for directional NWV views — with exit points
================================================================================
Diagnostic showed the BEARISH view is directionally inverted (price drifts UP),
so a credit/neutral structure should beat a bear debit spread. Test:

Structures (entry Mon EOD, next-week expiry, 200-wide wings, 5 lots):
  debit       — the current ATM/200 debit spread (reference)
  debit_PT    — same, but take profit at 60% of max gain (EOD check)
  ic_neutral  — short put @ spot-250 / short call @ spot+250 (+/-200 wings)
  ic_bear     — bear-skew: short call spot+100, short put spot-400
  ic_bull     — bull-skew: short put spot-100, short call spot+400

Exit points (checked at each daily EOD Mon..Fri):
  IC: take profit at 50% of credit; stop at -1.0x credit; else Friday EOD.

Pricing: REAL option EOD for 2024+ (accurate); BS+realized-vol (MULT*RV) for the
full 2020+ regime read (≈22pt model error — relative ranking only).
"""
import sys
from datetime import date, datetime, timedelta
from collections import defaultdict
sys.path.insert(0, '/home/arun/quantifyd')
sys.path.insert(0, '/home/arun/quantifyd/research')
from nwv_phase1_regime import (
    build_view, daily_close, mondays, week_sessions, expiries_after, opt_eod,
    realized_vol, price as bs_price, MULT, round_to_strike,
    VIEW_BEARISH, VIEW_NTB, VIEW_BULLISH, VIEW_NTBULL,
)
LOT, LOTS, WIDTH = 65, 5, 200

def rk(x): return round_to_strike(x, 100)

def structures(spot):
    """Return dict name -> list of (qty, strike, otype). qty +long/-short."""
    L = rk(spot)
    return {
        'ic_neutral': [(-1, rk(spot-250), 'PE'), (1, rk(spot-250)-WIDTH, 'PE'),
                       (-1, rk(spot+250), 'CE'), (1, rk(spot+250)+WIDTH, 'CE')],
        'ic_bear':    [(-1, rk(spot+100), 'CE'), (1, rk(spot+100)+WIDTH, 'CE'),
                       (-1, rk(spot-400), 'PE'), (1, rk(spot-400)-WIDTH, 'PE')],
        'ic_bull':    [(-1, rk(spot-100), 'PE'), (1, rk(spot-100)-WIDTH, 'PE'),
                       (-1, rk(spot+400), 'CE'), (1, rk(spot+400)+WIDTH, 'CE')],
    }

def run(real_only):
    start = date(2024, 3, 4) if real_only else date(2020, 2, 3)
    res = defaultdict(lambda: defaultdict(list))   # side -> struct -> [pnl]
    for mon in mondays(start, date(2026, 3, 23)):
        v = build_view(mon)
        if not v:
            continue
        if v['view'] not in (VIEW_BEARISH, VIEW_NTB, VIEW_BULLISH, VIEW_NTBULL):
            continue
        side = 'BEAR' if v['view'] in (VIEW_BEARISH, VIEW_NTB) else 'BULL'
        ms, spot = v['ms'], v['spot']
        sess = week_sessions(ms)
        if len(sess) < 2:
            continue

        if real_only:
            exps = expiries_after(ms)
            if len(exps) < 2:
                continue
            expiry = exps[1]
            edate = datetime.strptime(expiry, '%Y-%m-%d').date()
            def leg_px(d, k, ot):
                return opt_eod(d, expiry, k, ot)
            def dte_on(d):
                return (edate - datetime.strptime(d, '%Y-%m-%d').date()).days
        else:
            rv = realized_vol(ms)
            if not rv:
                continue
            iv = MULT * rv
            edate = mon + timedelta(days=8)
            def leg_px(d, k, ot):
                return bs_price(daily_close[d], k, (edate - datetime.strptime(d, '%Y-%m-%d').date()).days, iv, ot)
            def dte_on(d):
                return (edate - datetime.strptime(d, '%Y-%m-%d').date()).days

        def legs_val(d, legs):
            tot = 0.0
            for q, k, ot in legs:
                px = leg_px(d, k, ot)
                if px is None:
                    return None
                tot += q * px
            return tot

        # ---- debit spread (+PT) ----
        ot = 'PE' if side == 'BEAR' else 'CE'
        L = rk(spot); M = L - WIDTH if side == 'BEAR' else L + WIDTH
        dl = [(1, L, ot), (-1, M, ot)]
        entry = legs_val(ms, dl)
        if entry is None or entry <= 0:
            continue
        debit = entry
        max_gain = WIDTH - debit
        pnl_hold = pt_done = None
        for d in sess:
            val = legs_val(d, dl)
            if val is None:
                continue
            if pt_done is None and (val - debit) >= 0.60 * max_gain:
                pt_done = (val - debit) * LOT * LOTS
            last = val
        pnl_hold = (last - debit) * LOT * LOTS
        res[side]['debit'].append(pnl_hold)
        res[side]['debit_PT'].append(pt_done if pt_done is not None else pnl_hold)

        # ---- IC variants ----
        # legs_val = (longs - shorts). At entry it's negative; credit = -entry_lv.
        # P&L per unit = legs_val_now - entry_lv  (theta decays shorts -> lv rises to ~0).
        for name, legs in structures(spot).items():
            entry_lv = legs_val(ms, legs)
            if entry_lv is None:
                res[side][name].append(None); continue
            credit = -entry_lv
            if credit <= 0:
                res[side][name].append(None); continue
            cap = credit * LOT * LOTS           # max profit (= credit kept)
            exit_pnl = None; last_lv = None
            for d in sess:
                lv = legs_val(d, legs)
                if lv is None:
                    continue
                last_lv = lv
                mtm = (lv - entry_lv) * LOT * LOTS
                if exit_pnl is None and mtm >= 0.50 * cap:
                    exit_pnl = mtm              # take profit at 50% of credit
                elif exit_pnl is None and mtm <= -1.0 * cap:
                    exit_pnl = mtm              # stop at -1x credit
            if exit_pnl is None and last_lv is not None:
                exit_pnl = (last_lv - entry_lv) * LOT * LOTS
            res[side][name].append(exit_pnl)

    tag = "REAL EOD 2024-03..2026-03" if real_only else "MODELED 2020..2026 (22pt err)"
    print(f"\n############ {tag} ############")
    for side in ('BEAR', 'BULL'):
        print(f"\n=== {side}-view weeks ===")
        for name in ('debit', 'debit_PT', 'ic_neutral', 'ic_bear', 'ic_bull'):
            summ(res[side][name], name)

def summ(p, label):
    p = [x for x in p if x is not None]
    if not p:
        print(f"  {label:12s}: (none)"); return
    n = len(p); w = [x for x in p if x > 0]
    gw = sum(x for x in p if x > 0); gl = -sum(x for x in p if x < 0)
    print(f"  {label:12s}: n={n:3d}  win%={100*len(w)/n:5.1f}  avg=Rs{sum(p)/n:8.0f}  "
          f"tot=Rs{sum(p):10.0f}  PF={gw/gl if gl else 999:5.2f}  worst={min(p):8.0f}")

if __name__ == '__main__':
    run(real_only=True)
    run(real_only=False)
