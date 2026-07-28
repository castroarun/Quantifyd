# -*- coding: utf-8 -*-
"""Calibrate ATM2's exit for the expiry-gamma problem. Entry = 09:16 ATM straddle (the 916/ATM2 book),
2 lots, net-of-cost, one-and-done (no re-center, so we isolate the STOP rule). Walk the chain minute by
minute and compare exit rules:
  CURRENT   : 0.4% spot-move stop (what ATM2 uses today)
  PREMIUM-K : exit both when combined straddle >= K x entry-straddle   (DTE-agnostic, rec 1)
  RUPEE-L   : exit both when MTM loss >= L rupees/lot                   (DTE-agnostic, rec 1)
  PERLEG-30 : exit both when EITHER leg >= 1.3 x its entry (rec 3 backstop, currently disabled)
Report total/avg/worst/p5/win%, split by DTE<=1 (near-expiry, the danger zone) vs DTE>=2. The goal:
a rule whose WORST near-expiry day is far shallower than CURRENT, without gutting the average."""
import sys
sys.path.insert(0, "/home/arun/quantifyd")
sys.path.insert(0, "/home/arun/quantifyd/research/90_nas_portfolio_bracket/scripts")
import numpy as np
from datetime import time as dtime
from engine_mtm import load_day, days, LOT, LOTS, BROK_PER_LEG  # noqa

QTY = LOT * LOTS
ENTRY, EXIT = dtime(9, 16), dtime(15, 15)
COST = 2 * BROK_PER_LEG  # 2 legs round-trip brokerage


def simulate(b):
    """Return dict rule->pnl for one day, plus dte. None if unpriceable."""
    chain, spot_s, times = b["chain"], b["spot_s"], b["times"]
    dte = b.get("dte_day")

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
    # precompute path of (spot, ce, pe)
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

    out = {"dte": dte}

    # CURRENT: 0.4% spot-move stop
    exited = None
    for sp, cn, pn in path:
        if abs(sp - spot0) / spot0 >= 0.004:
            exited = (cn, pn); break
    out["CURRENT_move0.4"] = pnl_at(*(exited or (last[1], last[2])))

    # PREMIUM-K
    for K in (1.4, 1.6, 1.8, 2.0):
        exited = None
        for sp, cn, pn in path:
            if (cn + pn) >= K * strad_e:
                exited = (cn, pn); break
        out["PREMx%.1f" % K] = pnl_at(*(exited or (last[1], last[2])))

    # RUPEE-L per lot
    for L in (1500, 2000, 2500, 3000):
        thr = L * LOTS
        exited = None
        for sp, cn, pn in path:
            if (strad_e - (cn + pn)) * QTY <= -thr:
                exited = (cn, pn); break
        out["RUPEE%d/lot" % L] = pnl_at(*(exited or (last[1], last[2])))

    # PERLEG-30 backstop
    exited = None
    for sp, cn, pn in path:
        if cn >= 1.3 * ce_e or pn >= 1.3 * pe_e:
            exited = (cn, pn); break
    out["PERLEG1.3"] = pnl_at(*(exited or (last[1], last[2])))

    return out


rows = []
for i, day in enumerate(days):
    b = load_day(day)
    if b is None:
        continue
    r = simulate(b)
    if r:
        r["day"] = day
        rows.append(r)
    if i % 15 == 0:
        print("  %d/%d %s" % (i, len(days), day), flush=True)

RULES = ["CURRENT_move0.4", "PREMx1.4", "PREMx1.6", "PREMx1.8", "PREMx2.0",
         "RUPEE1500/lot", "RUPEE2000/lot", "RUPEE2500/lot", "RUPEE3000/lot", "PERLEG1.3"]


def blk(title, subset):
    print("\n=== %s  (n=%d days) ===" % (title, len(subset)))
    print("%-16s %8s %8s %8s %8s %6s" % ("rule", "total", "avg/tr", "worst", "p5", "win%"))
    for k in RULES:
        a = np.array([r[k] for r in subset], float)
        print("%-16s %+8d %+8d %+8d %+8d %6.0f" %
              (k, a.sum(), a.mean(), a.min(), np.percentile(a, 5), 100 * (a > 0).mean()))


print("\nATM2 exit calibration — 09:16 ATM straddle, 2 lots, net, one-and-done (stop isolated)")
blk("ALL DAYS", rows)
near = [r for r in rows if (r["dte"] is not None and r["dte"] <= 1)]
far = [r for r in rows if (r["dte"] is not None and r["dte"] >= 2)]
if near:
    blk("NEAR-EXPIRY (DTE<=1) — the danger zone", near)
if far:
    blk("FAR (DTE>=2)", far)
