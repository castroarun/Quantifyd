#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/141 RECONCILIATION GATE — reproduce research/96's headline numbers.

r/96 (run_atm2_exit_calib.py) used:
  * engine_mtm.load_day  -> NIFTY front expiry from options_data.db, 1-min
  * days = every DISTINCT snapshot day for NIFTY (no holiday/partial guard!)
  * entry 09:16, exit 15:15, ATM by round(spot/50)*50 (SPOT rounding, pre-r/132)
  * QTY = 65 * 2 = 130 (2 lots), cost = 2 * 80 = Rs160 flat brokerage, no slippage
  * one-and-done (stop isolated, no re-center)
  * DTE bucket = CALENDAR (expiry - day).days,  near = DTE<=1

Targets to reproduce on the 68 NIFTY days <= 2026-07-28, near-expiry bucket:
  RUPEE 2500/lot : avg +2,153 | worst -6,972 | win 69%
  0.4% move-stop : avg +1,386 | worst -6,887 | win 62%

READ-ONLY on every DB.
"""
import os
import sys
import sqlite3
from datetime import date

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

CUTOFF = "2026-07-28"          # r/96 ran on 2026-07-28
LOT, LOTS = 65, 2
QTY = LOT * LOTS
COST = 160.0                   # r/96: 2 x BROK_PER_LEG, BROK_PER_LEG = 40*2
ENTRY_M = 9 * 60 + 16
EXIT_M = 15 * 60 + 15

OUT = []


def p(m):
    OUT.append(m)
    print(m, flush=True)


def hm2m(h):
    return int(h[:2]) * 60 + int(h[3:5])


def load_day_r96(c, day):
    """Faithful to engine_mtm.load_day: NO holiday guard, NO partial guard."""
    rows = c.execute(
        "SELECT snapshot_time, expiry_date, strike, instrument_type, ltp, underlying_spot "
        "FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? "
        "AND ltp IS NOT NULL", (day,)).fetchall()
    if not rows:
        return None
    exps = sorted({e for (_, e, _, _, _, _) in rows if e})
    fut = [e for e in exps if e >= day]
    fexp = fut[0] if fut else (exps[-1] if exps else None)
    if fexp is None:
        return None
    spot, chain = {}, {}
    for st, e, k, it, ltp, sp in rows:
        mi = hm2m(st[11:16])
        if sp is not None and mi not in spot:
            spot[mi] = sp
        if e != fexp:
            continue
        chain.setdefault(mi, {}).setdefault(k, {})[it] = ltp
    ch2 = {}
    for mi, ks in chain.items():
        ch2[mi] = {k: (v["CE"], v["PE"]) for k, v in ks.items()
                   if v.get("CE") and v.get("PE") and v["CE"] > 0 and v["PE"] > 0}
    dte_cal = (date.fromisoformat(fexp) - date.fromisoformat(day)).days
    return fexp, dte_cal, spot, ch2


def simulate(spot, chain):
    """One day, one-and-done, spot-rounded ATM. Returns dict rule -> pnl (2 lots)."""
    mins = sorted(m for m in chain if ENTRY_M <= m <= EXIT_M)
    if not mins:
        return None
    m0 = mins[0]
    s0 = spot.get(m0)
    if not s0:
        return None
    K = round(s0 / 50) * 50
    if K not in chain[m0]:
        return None
    ce0, pe0 = chain[m0][K]
    strad_e = ce0 + pe0
    path = []
    for mi in mins:
        d = chain.get(mi)
        if not d or K not in d:
            continue
        ce, pe = d[K]
        sp = spot.get(mi, s0)
        path.append((mi, sp, ce, pe))
    if not path:
        return None
    last = path[-1]

    def pnl_at(ce, pe):
        return round((strad_e - (ce + pe)) * QTY - COST)

    out = {"K": K, "credit": strad_e, "entry_m": m0, "spot0": s0}
    # 0.4% spot-move stop
    ex = None
    for mi, sp, ce, pe in path:
        if abs(sp - s0) / s0 >= 0.004:
            ex = (ce, pe)
            break
    out["MOVE0.4"] = pnl_at(*(ex or (last[2], last[3])))
    # rupee stops
    for L in (2000, 2500, 3000):
        thr = L * LOTS
        ex = None
        for mi, sp, ce, pe in path:
            if (strad_e - (ce + pe)) * QTY <= -thr:
                ex = (ce, pe)
                break
        out["RUPEE%d" % L] = pnl_at(*(ex or (last[2], last[3])))
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    days = [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM option_chain "
        "WHERE symbol='NIFTY' ORDER BY d")]
    days = [d for d in days if d <= CUTOFF]
    p("research/141 reconciliation gate vs research/96")
    p("NIFTY chain days <= %s : %d" % (CUTOFF, len(days)))

    rows = []
    for day in days:
        d = load_day_r96(c, day)
        if not d:
            continue
        fexp, dte_cal, spot, chain = d
        r = simulate(spot, chain)
        if r:
            r["day"] = day
            r["dte"] = dte_cal
            rows.append(r)
    p("days simulated: %d" % len(rows))

    RULES = ["MOVE0.4", "RUPEE2000", "RUPEE2500", "RUPEE3000"]

    def blk(title, sub):
        p("")
        p("=== %s (n=%d) ===" % (title, len(sub)))
        p("%-12s %9s %9s %9s %6s" % ("rule", "total", "avg/tr", "worst", "win%"))
        for k in RULES:
            a = [x[k] for x in sub]
            if not a:
                continue
            p("%-12s %+9d %+9d %+9d %6.0f" % (
                k, sum(a), sum(a) / len(a), min(a),
                100.0 * sum(1 for v in a if v > 0) / len(a)))

    blk("ALL DAYS", rows)
    near = [x for x in rows if x["dte"] <= 1]
    far = [x for x in rows if x["dte"] >= 2]
    blk("NEAR-EXPIRY calendar DTE<=1 (r/96 headline bucket)", near)
    blk("FAR calendar DTE>=2", far)

    p("")
    p("--- GATE ---")
    tgt = {"RUPEE2500": (2153, -6972, 69), "MOVE0.4": (1386, -6887, 62)}
    ok = True
    for k, (ta, tw, twin) in tgt.items():
        a = [x[k] for x in near]
        avg = sum(a) / len(a)
        wrst = min(a)
        win = 100.0 * sum(1 for v in a if v > 0) / len(a)
        da = abs(avg - ta) / abs(ta) * 100
        p("%-10s r/96 avg %+6d worst %+6d win %2d%%  |  ours avg %+7.0f worst %+7.0f win %4.0f%%"
          " | avg delta %.1f%%" % (k, ta, tw, twin, avg, wrst, win, da))
        if da > 5:
            ok = False
    p("GATE: %s" % ("PASS (all headline figures within 5%)" if ok else "MISMATCH"))

    with open(os.path.join(RES, "recon96.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
