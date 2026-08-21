#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/120 Part 2 - is a SECOND Friday slot worth adding, ON TOP of what already runs?

Not "is slot B profitable alone" but "does slot B add anything to a book that already holds
slot A (NIFTY TimeB Fri DTE2 10:00-12:00 SL20) and COMB (NIFTY 09:16-15:20 SL20)?"

Charges slot B a FULL extra round trip, measures the correlation of the two slots' bad days,
the joint worst Friday, and the peak concurrent margin.

Also runs the BLOCK test: instead of asking which of 110 cells is best (hopeless on 16 days),
it tests the 3 pre-specified time blocks the surface suggests. Far fewer tests, far more power.
"""
import sqlite3, csv, os, math, statistics as stt
import numpy as np
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stage_a_windows import (load_day, replay, fridays, hm2m, m2hm, VENUE, CHG,
                             LEG_SIDES, CHAIN, SESS_END)

RES = os.path.join(HERE, "..", "results")
lines = []
def P(s=""):
    lines.append(s); print(s, flush=True)

MARGIN = {"NIFTY": 165000.0, "SENSEX": 204000.0}   # Rs per lot, short straddle MIS
CAPITAL = 4470000.0

# name, venue, entry, exit, sl
BOOKS = [
    ("COMB_NIFTY_full",      "NIFTY",  "09:16", "15:20", 0.20),
    ("A_LIVE_TimeB_N_10_12", "NIFTY",  "10:00", "12:00", 0.20),
    # candidate second slots for NIFTY - must not overlap 10:00-12:00
    ("B_N_1200_1300",        "NIFTY",  "12:00", "13:00", 0.20),
    ("B_N_1230_1330",        "NIFTY",  "12:30", "13:30", 0.20),
    ("B_N_1300_1400_SL25",   "NIFTY",  "13:00", "14:00", 0.25),   # the TIMEB2 Mon/Tue shape
    ("B_N_1300_1520",        "NIFTY",  "13:00", "15:20", 0.20),
    ("B_N_1400_1520",        "NIFTY",  "14:00", "15:20", 0.20),
    ("B_N_1405_1535cap",     "NIFTY",  "14:05", "15:20", 0.20),
    ("B_N_1200_1520",        "NIFTY",  "12:00", "15:20", 0.20),
    ("B_N_0920_1000",        "NIFTY",  "09:20", "10:00", 0.20),   # pre-A slot
    ("B_N_0935_1000",        "NIFTY",  "09:35", "10:00", 0.20),
    # alternatives for MOVING slot A rather than adding one
    ("ALT_N_0935_1135",      "NIFTY",  "09:35", "11:35", 0.20),
    ("ALT_N_0950_1150",      "NIFTY",  "09:50", "11:50", 0.20),
    ("ALT_N_0935_1520",      "NIFTY",  "09:35", "15:20", 0.20),
    # SENSEX Friday - no live cell today; is one worth adding at all?
    ("SX_0935_1135",         "SENSEX", "09:35", "11:35", 0.20),
    ("SX_0950_1120",         "SENSEX", "09:50", "11:20", 0.20),
    ("SX_1000_1200",         "SENSEX", "10:00", "12:00", 0.20),
    ("SX_0916_1520",         "SENSEX", "09:16", "15:20", 0.20),
    ("SX_1300_1400_SL25",    "SENSEX", "13:00", "14:00", 0.25),
    ("SX_1400_1520",         "SENSEX", "14:00", "15:20", 0.20),
]


def run():
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    cache = {}
    series = {}
    detail = []
    for name, ven, e0, e1, sl in BOOKS:
        V = VENUE[ven]
        lot, step, slip = V["lot"], V["step"], V["slip"]
        cost = LEG_SIDES * slip * lot + LEG_SIDES * CHG
        days = fridays(c, ven)
        out = {}
        for day in days:
            if (ven, day) not in cache:
                cache[(ven, day)] = load_day(c, ven, day)
            d = cache[(ven, day)]
            if not d:
                continue
            _fexp, spot, chain = d
            mins = sorted(chain)
            m0n = hm2m(e0)
            cand = [m for m in mins if m0n <= m <= m0n + 10]
            if not cand:
                continue
            m0 = cand[0]
            m1 = hm2m(e1)
            sp0 = spot.get(m0)
            if not sp0:
                continue
            K = round(sp0 / step) * step
            r = replay(chain, spot, K, m0, m1, sl)
            if not r:
                continue
            gross = (r["credit"] - r["exit_comb"]) * lot
            out[day] = gross - cost
            detail.append(dict(book=name, venue=ven, day=day, entry=e0, exit=e1,
                               sl=sl, strike=K, credit=round(r["credit"], 2),
                               exit_hm=m2hm(r["exit_m"]), reason=r["reason"],
                               net=round(gross - cost), mae_rs=round(r["mae_full"] * lot),
                               und_exc_bp=round(1e4 * r["und_exc"] / sp0, 1)))
        series[name] = out
    with open(os.path.join(RES, "marginal_slot_trades.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys())); w.writeheader()
        for r in detail:
            w.writerow(r)
    return series


def stats(v):
    a = np.array(v, dtype=float)
    n = len(a)
    t = a.mean() / (a.std(ddof=1) / math.sqrt(n)) if n > 1 and a.std(ddof=1) > 0 else 0
    return dict(n=n, mean=a.mean(), med=float(np.median(a)), total=a.sum(),
                win=100 * (a > 0).mean(), worst=a.min(), best=a.max(), t=t,
                sd=a.std(ddof=1))


def main():
    series = run()
    days = sorted(set(series["A_LIVE_TimeB_N_10_12"]))
    P("=" * 108)
    P("PART 2 - STANDALONE BOOK STATS (net Rs per LOT per Friday, %d Fridays 2026-05-01..08-14)" % len(days))
    P("=" * 108)
    P("  %-24s %-7s %5s %7s %7s %6s %8s %8s %6s" %
      ("book", "venue", "n", "mean", "median", "win%", "worst", "best", "t"))
    rows = {}
    for name, ven, e0, e1, sl in BOOKS:
        d = series[name]
        common = [d[x] for x in days if x in d]
        if len(common) < 12:
            P("  %-24s %-7s   too few days (%d)" % (name, ven, len(common)))
            continue
        s = stats(common)
        rows[name] = (ven, np.array([d.get(x, 0.0) for x in days]))
        P("  %-24s %-7s %5d %7.0f %7.0f %6.0f %8.0f %8.0f %6.2f" %
          (name, ven, s["n"], s["mean"], s["med"], s["win"], s["worst"], s["best"], s["t"]))

    A = rows["A_LIVE_TimeB_N_10_12"][1]
    C = rows["COMB_NIFTY_full"][1]
    P()
    P("=" * 108)
    P("MARGINAL VALUE OF A SECOND SLOT  (A = live NIFTY TimeB Fri 10:00-12:00 SL20; COMB = 09:16-15:20 SL20)")
    P("Slot B already pays a FULL extra round trip (Rs250/lot NIFTY, Rs200/lot SENSEX).")
    P("=" * 108)
    P("  %-24s %7s %8s %8s | %8s %8s %8s | %6s %6s %7s" %
      ("candidate slot B", "B mean", "B worst", "B win%",
       "A+B mean", "A+B worst", "A+B win%", "r(A,B)", "r(C,B)", "badOvlp"))
    for name, ven, e0, e1, sl in BOOKS:
        if not name.startswith("B_"):
            continue
        if name not in rows:
            continue
        Bv = rows[name][1]
        AB = A + Bv
        rab = float(np.corrcoef(A, Bv)[0, 1])
        rcb = float(np.corrcoef(C, Bv)[0, 1])
        badA = A < 0
        badB = Bv < 0
        ovl = 100.0 * float((badA & badB).sum()) / max(1, int(badA.sum()))
        P("  %-24s %7.0f %8.0f %8.0f | %8.0f %8.0f %8.0f | %6.2f %6.2f %6.0f%%" %
          (name, Bv.mean(), Bv.min(), 100 * (Bv > 0).mean(),
           AB.mean(), AB.min(), 100 * (AB > 0).mean(), rab, rcb, ovl))
    P()
    P("  For reference: A alone  mean %.0f  worst %.0f  win %.0f%%   |  COMB alone mean %.0f worst %.0f"
      % (A.mean(), A.min(), 100 * (A > 0).mean(), C.mean(), C.min()))
    P("  A+COMB (what already runs on a Friday): mean %.0f  worst %.0f   r(A,COMB) = %.2f"
      % ((A + C).mean(), (A + C).min(), float(np.corrcoef(A, C)[0, 1])))

    P()
    P("=" * 108)
    P("ALTERNATIVE: MOVE slot A instead of adding one")
    P("=" * 108)
    P("  %-24s %7s %8s %6s %6s   %s" % ("window", "mean", "worst", "win%", "t", "vs live A"))
    for name in ("A_LIVE_TimeB_N_10_12", "ALT_N_0935_1135", "ALT_N_0950_1150", "ALT_N_0935_1520"):
        if name not in rows:
            continue
        v = rows[name][1]
        s = stats(list(v))
        dlt = v - A
        td = dlt.mean() / (dlt.std(ddof=1) / math.sqrt(len(dlt))) if dlt.std(ddof=1) > 0 else 0
        P("  %-24s %7.0f %8.0f %6.0f %6.2f   %+7.0f/Fri (paired t = %.2f)"
          % (name, s["mean"], s["worst"], s["win"], s["t"], dlt.mean(), td))

    P()
    P("=" * 108)
    P("SENSEX - is a Friday (DTE4) cell worth opening at all?")
    P("=" * 108)
    P("  %-24s %7s %8s %6s %6s %9s" % ("window", "mean", "worst", "win%", "t", "r vs N-A"))
    for name, ven, e0, e1, sl in BOOKS:
        if ven != "SENSEX" or name not in rows:
            continue
        v = rows[name][1]
        s = stats(list(v))
        P("  %-24s %7.0f %8.0f %6.0f %6.2f %9.2f"
          % (name, s["mean"], s["worst"], s["win"], s["t"], float(np.corrcoef(A, v)[0, 1])))

    P()
    P("=" * 108)
    P("MARGIN REALITY (capital Rs%.1fL; NIFTY Rs%.2fL/lot, SENSEX Rs%.2fL/lot straddle MIS)"
      % (CAPITAL / 1e5, MARGIN["NIFTY"] / 1e5, MARGIN["SENSEX"] / 1e5))
    P("=" * 108)
    P("  Friday book as deployed today: COMB 2L + TimeB-N 8L (10:00-12:00) + TimeB-SX 0L (no Fri cell)")
    P("    peak concurrent 10:00-12:00 : COMB 2L + TB-N 8L = 10 NIFTY lots = Rs%.2fL (%.0f%% of capital)"
      % (10 * MARGIN["NIFTY"] / 1e5, 100 * 10 * MARGIN["NIFTY"] / CAPITAL))
    for lots in (2, 3):
        P("    + a NIFTY second slot at %dL after 12:00 : COMB 2L + %dL = %d lots = Rs%.2fL (%.0f%%)"
          % (lots, lots, 2 + lots, (2 + lots) * MARGIN["NIFTY"] / 1e5,
             100 * (2 + lots) * MARGIN["NIFTY"] / CAPITAL))
        P("    + a SENSEX Friday slot at %dL, 09:35-11:35 (overlaps A) : 10 NIFTY + %d SENSEX = Rs%.2fL (%.0f%%)"
          % (lots, lots, (10 * MARGIN["NIFTY"] + lots * MARGIN["SENSEX"]) / 1e5,
             100 * (10 * MARGIN["NIFTY"] + lots * MARGIN["SENSEX"]) / CAPITAL))

    open(os.path.join(RES, "marginal_slot_report.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
