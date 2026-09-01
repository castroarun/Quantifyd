#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/141 — locate the day(s) where our r/96 replica differs from r/96's own
engine (engine_mtm), so the reconciliation residual is explained, not hand-waved.

Runs research/96's ACTUAL code path (engine_mtm.load_day + the calib simulate logic)
day by day and diffs it against our reimplementation. READ-ONLY.
"""
import os
import sys
import sqlite3
from datetime import date, time as dtime

sys.path.insert(0, "/home/arun/quantifyd")
sys.path.insert(0, "/home/arun/quantifyd/research/90_nas_portfolio_bracket/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from engine_mtm import load_day, days, LOT, LOTS, BROK_PER_LEG  # noqa: E402
from recon96 import load_day_r96, simulate as sim_ours  # noqa: E402

QTY = LOT * LOTS
COST = 2 * BROK_PER_LEG
ENTRY, EXIT = dtime(9, 16), dtime(15, 15)
CUTOFF = "2026-07-28"
CHAIN = "/home/arun/quantifyd/backtest_data/options_data.db"


def sim_r96(b):
    chain, spot_s, times = b["chain"], b["spot_s"], b["times"]

    def prem(ts, t):
        ta, la, _, _ = chain[ts]
        i = np.searchsorted(ta, np.datetime64(t), side="right") - 1
        return float(la[i]) if i >= 0 and la[i] and la[i] > 0 else None

    def tsym(strike, typ):
        for ts, (_, _, st, ty) in chain.items():
            if int(st) == int(strike) and ty == typ:
                return ts
        return None

    t0 = next((t for t in times if t.time() >= ENTRY), None)
    if t0 is None:
        return None
    spot0 = float(spot_s.loc[t0])
    atm = round(spot0 / 50) * 50
    ce, pe = tsym(atm, "CE"), tsym(atm, "PE")
    if not ce or not pe:
        return None
    ce_e, pe_e = prem(ce, t0), prem(pe, t0)
    if ce_e is None or pe_e is None:
        return None
    strad_e = ce_e + pe_e
    walk = [t for t in times if t0 <= t and t.time() <= EXIT]
    path = []
    for t in walk:
        cn, pn = prem(ce, t), prem(pe, t)
        sp = float(spot_s.loc[t]) if t in spot_s.index else spot0
        if cn is None or pn is None:
            continue
        path.append((sp, cn, pn))
    if not path:
        return None
    last = path[-1]

    def pnl_at(cn, pn):
        return round((strad_e - (cn + pn)) * QTY - COST)

    out = {"K": atm, "entry_t": str(t0), "spot0": spot0, "credit": strad_e}
    ex = None
    for sp, cn, pn in path:
        if abs(sp - spot0) / spot0 >= 0.004:
            ex = (cn, pn)
            break
    out["MOVE0.4"] = pnl_at(*(ex or (last[1], last[2])))
    for L in (2000, 2500, 3000):
        thr = L * LOTS
        ex = None
        for sp, cn, pn in path:
            if (strad_e - (cn + pn)) * QTY <= -thr:
                ex = (cn, pn)
                break
        out["RUPEE%d" % L] = pnl_at(*(ex or (last[1], last[2])))
    return out


def main():
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    dd = [d for d in days if d <= CUTOFF]
    print("comparing %d days" % len(dd))
    print("%-12s %4s %10s %10s %9s | %10s %10s %9s | %8s" % (
        "day", "dte", "K_r96", "cred_r96", "MOVE_r96", "K_ours", "cred_ours",
        "MOVE_ours", "d_MOVE"))
    tot = {"MOVE0.4": 0, "RUPEE2500": 0}
    nd = 0
    for day in dd:
        b = load_day(day)
        if b is None:
            continue
        a = sim_r96(b)
        d2 = load_day_r96(c, day)
        o = sim_ours(d2[2], d2[3]) if d2 else None
        if not a or not o:
            print("%-12s  MISSING (r96=%s ours=%s)" % (day, bool(a), bool(o)))
            continue
        dm = o["MOVE0.4"] - a["MOVE0.4"]
        dr = o["RUPEE2500"] - a["RUPEE2500"]
        tot["MOVE0.4"] += dm
        tot["RUPEE2500"] += dr
        if dm or dr:
            nd += 1
            # recompute our K/credit for display
            print("%-12s %4d %10d %10.2f %+9d | %10d %10.2f %+9d | %+8d  dRUP2500=%+d"
                  % (day, b["dte_day"], a["K"], a["credit"], a["MOVE0.4"],
                     o["K"], o["credit"], o["MOVE0.4"], dm, dr))
    print("\ndays differing: %d" % nd)
    print("total delta MOVE0.4 = %+d   RUPEE2500 = %+d" % (tot["MOVE0.4"], tot["RUPEE2500"]))


if __name__ == "__main__":
    main()
