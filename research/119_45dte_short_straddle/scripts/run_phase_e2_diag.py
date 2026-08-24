#!/usr/bin/env python3
"""
Phase E2 — two follow-ups that decide whether the Phase E verdict is safe.

E2a  DIRECTION. Are the move-cuts symmetric, or is one side doing the damage?
     A short straddle usually bleeds worse on the way DOWN, because the fall
     comes with a vol spike that lifts both legs.

E2b  TRIGGER TIMING. Phase E measured the move on the daily CLOSE. If the rule is
     only losing because it reacts a day late, an intraday trigger should rescue it.
     Tested on the REAL 5-minute NIFTY spot (2015 -> 2026): the day the intraday
     range first breaches x% from the anchor, exit at that day's REAL bhav close.
     Trigger and fill are both real - no modelled option prices anywhere.

Writes results/phase_e2.csv
"""
import csv
import math
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine45 import (connect, trading_days, monthly_expiries, nifty_daily_close,
                      nifty_5min, chain_for_expiry, pick_atm, prev_session,
                      costs_points, dparse, dstr, QTY)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
WIN_LO, WIN_HI = "2019-01-01", "2026-06-30"
CAPITAL = 3_600_000.0
TARGET, STOP = 0.50, 2.00
SLIP = 0.0025
MOVES = [0.015, 0.020, 0.025, 0.030]


def leg_prices(day_chain, K):
    legs = day_chain.get(K)
    if not legs or "CE" not in legs or "PE" not in legs:
        return None
    ce, pe = legs["CE"], legs["PE"]
    if ce["close"] <= 0 or pe["close"] <= 0:
        return None
    return ce["close"] + pe["close"]


def campaign(chain, dates, spot, band, ed, xd, move, arm, intraday):
    """arm: 'recentre' | 'exit_only'. intraday: use real 5-min range for the trigger."""
    spot0 = spot.get(ed)
    K = pick_atm(chain[ed], spot0)
    if K is None:
        return None
    credit = leg_prices(chain[ed], K)
    if not credit:
        return None
    cyc = dict(k=K, credit=credit, entry=ed, anchor=spot0)
    out, dirs = [], []

    def close(c, d, prem, why, sgn=0):
        g = c["credit"] - prem
        cost = costs_points(c["credit"], prem, SLIP)
        out.append(dict(entry=c["entry"], exit=d, gross=g, cost=cost, net=g - cost, why=why))
        if why == "MOVE":
            dirs.append((sgn, g - cost))

    for d in dates:
        if d <= ed or d > xd or cyc is None:
            continue
        prem = leg_prices(chain.get(d, {}), cyc["k"])
        if prem is None:
            continue
        sp = spot.get(d)
        if prem <= TARGET * cyc["credit"]:
            close(cyc, d, prem, "TARGET"); cyc = None; break
        if prem >= STOP * cyc["credit"]:
            close(cyc, d, prem, "STOP"); cyc = None; break
        # --- the move trigger
        hit, sgn = False, 0
        if intraday and d in band:
            hiP, loP = band[d]
            up = hiP / cyc["anchor"] - 1.0
            dn = loP / cyc["anchor"] - 1.0
            if up >= move:
                hit, sgn = True, +1
            elif -dn >= move:
                hit, sgn = True, -1
        elif sp:
            r = sp / cyc["anchor"] - 1.0
            if abs(r) >= move:
                hit, sgn = True, (1 if r > 0 else -1)
        if hit:
            close(cyc, d, prem, "MOVE", sgn)
            cyc = None
            if arm == "recentre" and d < xd:
                K2 = pick_atm(chain.get(d, {}), sp)
                c2 = leg_prices(chain.get(d, {}), K2) if K2 is not None else None
                if K2 is not None and c2:
                    cyc = dict(k=K2, credit=c2, entry=d, anchor=sp)
            if cyc is None:
                break

    if cyc is not None:
        for d in reversed([x for x in dates if ed < x <= xd]):
            prem = leg_prices(chain.get(d, {}), cyc["k"])
            if prem is not None:
                close(cyc, d, prem, "TIME_21DTE")
                break
    if not out:
        return None
    return sum(c["net"] for c in out), len(out), dirs


def stats(nets, label):
    n = len(nets)
    mean = sum(nets) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in nets) / (n - 1)) if n > 1 else 0
    t = mean / (sd / math.sqrt(n)) if sd else 0
    eq, pk, mdd = CAPITAL, CAPITAL, 0
    for x in nets:
        eq += x * QTY; pk = max(pk, eq); mdd = min(mdd, eq - pk)
    cagr = ((CAPITAL + sum(nets) * QTY) / CAPITAL) ** (1 / 7.48) - 1
    ddp = abs(100 * mdd / CAPITAL)
    return dict(label=label, n=n, per=mean, total=sum(nets), t=t,
                cagr=100 * cagr, mdd=ddp, calmar=(100 * cagr) / ddp if ddp else 0,
                win=100 * sum(1 for x in nets if x > 0) / n)


def main():
    con = connect()
    days = trading_days(con, "2018-06-01")
    spot = nifty_daily_close(con)
    exps = monthly_expiries(con, days, "2018-06-01", "2026-08-31")

    book = []
    for ym, exp in exps.items():
        e = dparse(exp)
        ed = prev_session(days, dstr(e - timedelta(days=45)))
        xd = prev_session(days, dstr(e - timedelta(days=21)))
        if not ed or not xd or ed >= xd or not (WIN_LO <= ed <= WIN_HI):
            continue
        ch = chain_for_expiry(con, exp, ed, xd)
        if ed not in ch:
            continue
        book.append((exp, ed, xd, ch, sorted(ch)))
    print("campaigns: %d" % len(book))

    bars = nifty_5min(con, min(b[1] for b in book), max(b[2] for b in book))
    band = {d: (max(v for _, v in rows), min(v for _, v in rows)) for d, rows in bars.items()}
    cov = sum(1 for b in book if all(d in band for d in b[4] if b[1] < d <= b[2]))
    print("5-min spot days: %d | campaigns with FULL intraday coverage: %d/%d"
          % (len(band), cov, len(book)))

    grid = []
    print("\n%-6s %-9s %-9s %5s %8s %6s %7s %7s %7s" %
          ("move", "arm", "trigger", "n", "per", "t", "CAGR%", "MaxDD%", "Calmar"))
    for move in MOVES:
        for arm in ("recentre", "exit_only"):
            for intr in (False, True):
                nets, alldirs = [], []
                for exp, ed, xd, ch, ds in book:
                    r = campaign(ch, ds, spot, band, ed, xd, move, arm, intr)
                    if r:
                        nets.append(r[0]); alldirs += r[2]
                if not nets:
                    continue
                s = stats(nets, "%s/%s/%s" % (move, arm, "intraday" if intr else "close"))
                s.update(move="%.1f%%" % (100 * move), arm=arm,
                         trigger="intraday" if intr else "close")
                up = [p for g, p in alldirs if g > 0]
                dn = [p for g, p in alldirs if g < 0]
                s["up_n"], s["dn_n"] = len(up), len(dn)
                s["up_avg"] = sum(up) / len(up) if up else 0
                s["dn_avg"] = sum(dn) / len(dn) if dn else 0
                grid.append(s)
                print("%-6s %-9s %-9s %5d %8.1f %6.2f %7.2f %7.1f %7.2f" %
                      (s["move"], arm, s["trigger"], s["n"], s["per"], s["t"],
                       s["cagr"], s["mdd"], s["calmar"]))

    print("\nE2a — which direction does the damage? (cycles cut by the move rule)")
    print("%-6s %-9s %-9s %6s %10s %6s %10s" %
          ("move", "arm", "trigger", "up n", "up avg", "dn n", "dn avg"))
    for g in grid:
        if g["up_n"] or g["dn_n"]:
            print("%-6s %-9s %-9s %6d %10.1f %6d %10.1f" %
                  (g["move"], g["arm"], g["trigger"], g["up_n"], g["up_avg"],
                   g["dn_n"], g["dn_avg"]))

    out = os.path.join(RES, "phase_e2.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
        w.writeheader()
        for g in grid:
            w.writerow(g)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
