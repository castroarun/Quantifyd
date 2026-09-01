#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/141 replay engine — ATM2 with an optional re-center after the stop.

Forward-snapped strike selection (research/132), measured outcome-aware cost model
(research/122 cost_per_lot), 1-minute recorded chain, both venues.

A "cycle" = one straddle: sell at the snapped ATM, cover on the arm's stop or at
15:15. If the arm allows it and the stop fired, a new cycle opens (the re-center).
Every extra cycle pays a FULL extra round trip, and the exit that triggered it is a
FORCED stop (+6.548 pt per leg-side of measured slippage) — that is the churn cost.

READ-ONLY on every DB.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/arun/quantifyd/research/132_strike_misselection_cost/scripts")

from common132 import (VENUE, read_forward, cost_per_lot, m2hm)  # noqa: E402

ENTRY_M = 9 * 60 + 16          # ATM2 916 entry
EXIT_M = 15 * 60 + 15          # live squareoff
ENTRY_WIN_END = 14 * 60 + 30   # live re-entry window close
MAX_CYCLES = 6                 # 1 + up to 5 re-centers


def snap_strike(ks, spot, step):
    """Forward snap: K = round(F/step)*step where F = K_ref + (CE - PE)."""
    rf = read_forward(ks, spot, step)
    if rf is None:
        return None
    F = rf[0]
    return int(round(F / step) * step)


def _get(ks, K):
    v = ks.get(K)
    if v is None:
        v = ks.get(float(K))
    return v


def replay_day(spot, chain, venue, arm):
    """Replay one day for one arm. Returns a dict of results, or None if unplayable.

    arm keys: stop ('rupee'|'move'|None), rupee_per_lot, move_pct, max_recenter,
              cooldown_min, require_strike_change
    """
    lot = VENUE[venue]["lot"]
    step = VENUE[venue]["step"]
    mins = sorted(m for m in chain if ENTRY_M <= m <= EXIT_M)
    if len(mins) < 100:
        return None
    minset = set(mins)

    stop = arm.get("stop")
    rup = arm.get("rupee_per_lot", 0)
    mvp = arm.get("move_pct", 0.0)
    maxrc = arm.get("max_recenter", 0)
    cd = arm.get("cooldown_min", 0)
    req_chg = arm.get("require_strike_change", True)

    # rupee stop expressed in premium points (lots cancel: threshold is per lot)
    rup_pts = (rup / float(lot)) if (stop == "rupee" and rup) else None

    cycles = []
    entry_m = mins[0]
    last_entry_m = None
    prev_K = None
    while True:
        if entry_m not in minset:
            nxt = [m for m in mins if m >= entry_m]
            if not nxt:
                break
            entry_m = nxt[0]
        ks = chain.get(entry_m) or {}
        s0 = spot.get(entry_m)
        if s0 is None:
            break
        K = snap_strike(ks, s0, step)
        if K is None:
            break
        v = _get(ks, K)
        if v is None or not v[0] or not v[1] or v[0] <= 0 or v[1] <= 0:
            break
        credit = v[0] + v[1]
        if credit <= 0:
            break

        exit_m, exit_comb, reason = None, None, None
        last_m, last_comb = entry_m, credit
        for mi in range(entry_m + 1, EXIT_M + 1):
            d = chain.get(mi)
            if not d:
                continue
            vv = _get(d, K)
            if vv is None or vv[0] is None or vv[1] is None:
                continue
            comb = vv[0] + vv[1]
            if comb <= 0:
                continue
            last_m, last_comb = mi, comb
            if stop == "rupee" and (comb - credit) >= rup_pts:
                exit_m, exit_comb, reason = mi, comb, "SL"
                break
            if stop == "move":
                sp = spot.get(mi)
                if sp and abs(sp - s0) / s0 >= mvp:
                    exit_m, exit_comb, reason = mi, comb, "SL"
                    break
        if exit_m is None:
            if last_m == entry_m:
                break
            exit_m, exit_comb, reason = last_m, last_comb, "TIME"

        gross = (credit - exit_comb) * lot
        cst = cost_per_lot(credit, exit_comb, lot, reason)
        cycles.append(dict(entry_m=entry_m, exit_m=exit_m, K=K, credit=credit,
                           exit_comb=exit_comb, reason=reason,
                           gross=gross, cost=cst, net=gross - cst, spot0=s0))
        last_entry_m = entry_m
        prev_K = K

        # ---- re-center decision ----
        if reason != "SL":
            break
        if len(cycles) - 1 >= maxrc or len(cycles) >= MAX_CYCLES:
            break
        nxt_m = exit_m
        if cd:
            nxt_m = max(nxt_m, last_entry_m + cd)
        nxt_m = next((m for m in mins if m >= nxt_m), None)
        if nxt_m is None or nxt_m > ENTRY_WIN_END:
            break
        if req_chg:
            sp = spot.get(nxt_m)
            nk = snap_strike(chain.get(nxt_m) or {}, sp, step) if sp else None
            if nk is None or nk == prev_K:
                break
        entry_m = nxt_m

    if not cycles:
        return None
    n = len(cycles)
    return dict(
        n_cycles=n,
        n_recenters=n - 1,
        stop_fired=1 if cycles[0]["reason"] == "SL" else 0,
        gross=sum(c["gross"] for c in cycles),
        cost=sum(c["cost"] for c in cycles),
        net=sum(c["net"] for c in cycles),
        net_c1=cycles[0]["net"],
        gross_c1=cycles[0]["gross"],
        cost_c1=cycles[0]["cost"],
        net_after=sum(c["net"] for c in cycles[1:]),
        gross_after=sum(c["gross"] for c in cycles[1:]),
        cost_after=sum(c["cost"] for c in cycles[1:]),
        first_exit=m2hm(cycles[0]["exit_m"]),
        last_exit=m2hm(cycles[-1]["exit_m"]),
        credit0=cycles[0]["credit"],
        strikes="|".join(str(c["K"]) for c in cycles),
        reasons="|".join(c["reason"] for c in cycles),
    )


# ---- the pre-registered arm grid --------------------------------------------------
RUP = 2500


def _r(maxrc, cd=0, chg=True):
    return dict(stop="rupee", rupee_per_lot=RUP, max_recenter=maxrc,
                cooldown_min=cd, require_strike_change=chg)


ARMS = [
    ("ONE_AND_DONE",       _r(0)),
    ("RECENTER_1",         _r(1)),
    ("RECENTER_2",         _r(2)),
    ("RECENTER_3",         _r(3)),
    ("RECENTER_5",         _r(5)),
    ("RECENTER_2_CD15",    _r(2, 15)),
    ("RECENTER_3_CD15",    _r(3, 15)),
    ("RECENTER_5_CD15",    _r(5, 15)),
    ("RECENTER_5_NOGUARD", _r(5, 0, False)),
    ("MOVESTOP_ONE",       dict(stop="move", move_pct=0.004, max_recenter=0,
                                cooldown_min=0, require_strike_change=True)),
    ("MOVESTOP_RECENTER",  dict(stop="move", move_pct=0.004, max_recenter=5,
                                cooldown_min=0, require_strike_change=True)),
    ("MOVESTOP_RC1",       dict(stop="move", move_pct=0.004, max_recenter=1,
                                cooldown_min=0, require_strike_change=True)),
    ("MOVESTOP_RC_CD15",   dict(stop="move", move_pct=0.004, max_recenter=5,
                                cooldown_min=15, require_strike_change=True)),
    ("NOSTOP_HOLD",        dict(stop=None, max_recenter=0, cooldown_min=0,
                                require_strike_change=True)),
]
