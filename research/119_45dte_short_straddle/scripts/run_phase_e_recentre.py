#!/usr/bin/env python3
"""
Phase E — delta management: hold the 45-DTE straddle until the UNDERLYING moves x%,
then close and (optionally) redeploy a fresh ATM straddle on the same expiry.

The question: a short straddle is hurt by movement, not by time. The baseline just
sits through the move and exits at 21 DTE. Does cutting at a spot-move threshold and
re-centring on the new ATM beat that — or does it just crystallise losses and pay
another round trip?

CAMPAIGN = one calendar month's expiry, entered 45 DTE, finished at 21 DTE.
A campaign contains 1..N straddle CYCLES. The unit of account stays the campaign, so
n is still ~89 and the t-stats are directly comparable to Phase A/B.

Everything here is REAL data:
  * the move trigger is measured on the real NIFTY daily close
  * every option leg — old and new — is priced off real NSE bhavcopy closes
  * a re-centred strike must have BOTH legs actually traded that day

Arms
  hold          baseline: no move rule (Phase A/B rule set)
  recentre      on trigger, close and immediately sell the new ATM, same expiry
  exit_only     on trigger, close and stay flat until 21 DTE (isolates the exit
                from the re-entry, so we can tell which half does the work)

Writes results/phase_e_grid.csv and results/trades_recentre_<x>.csv
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine45 import (connect, trading_days, monthly_expiries, nifty_daily_close,
                      chain_for_expiry, pick_atm, prev_session, next_session,
                      costs_points, dparse, dstr, QTY, LOT, LOTS)
from datetime import timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

WIN_LO, WIN_HI = "2019-01-01", "2026-06-30"
CAPITAL = 3_600_000.0          # Rs 3L/lot x 10 lots + 20% buffer
TARGET, STOP = 0.50, 2.00
SLIP = 0.0025

MOVES = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
ARMS = ["recentre", "exit_only"]
CAPS = [1, 2, 99]              # max re-centres allowed in a campaign


def leg_prices(day_chain, K):
    legs = day_chain.get(K)
    if not legs or "CE" not in legs or "PE" not in legs:
        return None
    ce, pe = legs["CE"], legs["PE"]
    if ce["close"] <= 0 or pe["close"] <= 0:
        return None
    return ce["close"] + pe["close"]


def run_campaign(chain, dates, spot_daily, ed, xd, move_pct, arm, cap):
    """One month. Returns (net_pts, gross_pts, n_cycles, n_triggers, end_reason, cycles)."""
    spot0 = spot_daily.get(ed)
    K = pick_atm(chain[ed], spot0)
    if K is None:
        return None
    credit = leg_prices(chain[ed], K)
    if not credit:
        return None

    cycles = []
    open_cyc = dict(k=K, credit=credit, entry=ed, anchor=spot0)
    n_trig = 0
    end_reason = "TIME_21DTE"

    def close_cycle(cyc, d, prem, why):
        g = cyc["credit"] - prem
        c = costs_points(cyc["credit"], prem, SLIP)
        cycles.append(dict(entry=cyc["entry"], exit=d, strike=cyc["k"],
                           credit=cyc["credit"], exit_prem=prem,
                           gross=g, cost=c, net=g - c, why=why))

    for d in dates:
        if d <= ed or d > xd:
            continue
        if open_cyc is None:
            continue
        prem = leg_prices(chain.get(d, {}), open_cyc["k"])
        if prem is None:
            continue
        sp = spot_daily.get(d)

        # 1) hard exits on the premium, same as the baseline rule set
        if prem <= TARGET * open_cyc["credit"]:
            close_cycle(open_cyc, d, prem, "TARGET"); open_cyc = None
            end_reason = "TARGET"; break
        if prem >= STOP * open_cyc["credit"]:
            close_cycle(open_cyc, d, prem, "STOP"); open_cyc = None
            end_reason = "STOP"; break

        # 2) the move rule, measured from THIS cycle's entry spot
        if move_pct and sp and abs(sp / open_cyc["anchor"] - 1.0) >= move_pct:
            n_trig += 1
            close_cycle(open_cyc, d, prem, "MOVE")
            open_cyc = None
            end_reason = "MOVE"
            if arm == "recentre" and n_trig <= cap and d < xd:
                K2 = pick_atm(chain.get(d, {}), sp)
                c2 = leg_prices(chain.get(d, {}), K2) if K2 is not None else None
                if K2 is not None and c2:
                    open_cyc = dict(k=K2, credit=c2, entry=d, anchor=sp)
                    end_reason = "TIME_21DTE"
            if open_cyc is None:
                break

    if open_cyc is not None:
        prem = None
        for d in reversed([x for x in dates if x <= xd and x > open_cyc["entry"]]):
            prem = leg_prices(chain.get(d, {}), open_cyc["k"])
            if prem is not None:
                close_cycle(open_cyc, d, prem, "TIME_21DTE")
                break
        if prem is None:
            return None

    if not cycles:
        return None
    return (sum(c["net"] for c in cycles), sum(c["gross"] for c in cycles),
            len(cycles), n_trig, end_reason, cycles)


def summarise(camps, label):
    n = len(camps)
    nets = [c["net"] for c in camps]
    wins = [x for x in nets if x > 0]
    mean = sum(nets) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in nets) / (n - 1)) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
    eq, pk, mdd = CAPITAL, CAPITAL, 0.0
    for x in nets:
        eq += x * QTY
        pk = max(pk, eq)
        mdd = min(mdd, eq - pk)
    yrs = 7.48
    cagr = ((CAPITAL + sum(nets) * QTY) / CAPITAL) ** (1 / yrs) - 1
    ddp = abs(100 * mdd / CAPITAL)
    return dict(label=label, n=n, win=100 * len(wins) / n,
                total=sum(nets), per=mean, t=t,
                cagr=100 * cagr, mdd_pct=ddp,
                calmar=(100 * cagr) / ddp if ddp else 0.0,
                worst=min(nets), best=max(nets),
                cycles=sum(c["cycles"] for c in camps) / n,
                trig=100 * sum(1 for c in camps if c["trig"]) / n,
                cost=sum(c["cost"] for c in camps) / n)


def main():
    con = connect()
    days = trading_days(con, "2018-06-01")
    spot = nifty_daily_close(con)
    exps = monthly_expiries(con, days, "2018-06-01", "2026-08-31")

    # preload every chain once — the sweep re-reads them many times
    book = []
    for ym, exp in exps.items():
        exp_dt = dparse(exp)
        ed = prev_session(days, dstr(exp_dt - timedelta(days=45)))
        xd = prev_session(days, dstr(exp_dt - timedelta(days=21)))
        if not ed or not xd or ed >= xd:
            continue
        if not (WIN_LO <= ed <= WIN_HI):
            continue
        chain = chain_for_expiry(con, exp, ed, xd)
        if ed not in chain:
            continue
        book.append((exp, ed, xd, chain, sorted(chain.keys())))
    print("campaigns loaded: %d (%s .. %s)" % (len(book), book[0][1], book[-1][1]))

    grid = []

    def evaluate(move, arm, cap, keep=None):
        camps = []
        for exp, ed, xd, chain, dates in book:
            r = run_campaign(chain, dates, spot, ed, xd, move, arm, cap)
            if not r:
                continue
            net, gross, ncyc, ntrig, why, cycles = r
            camps.append(dict(exp=exp, entry=ed, net=net, gross=gross, cycles=ncyc,
                              trig=ntrig, why=why,
                              cost=sum(c["cost"] for c in cycles)))
            if keep is not None:
                for c in cycles:
                    keep.append(dict(exp=exp, **c))
        return camps

    base = evaluate(None, "hold", 0)
    s0 = summarise(base, "HOLD (baseline)")
    grid.append(dict(move="none", arm="hold", cap="-", **s0))
    print("\nbaseline: n=%d net/campaign %.1f pts  t %.2f  CAGR %.2f%%  MaxDD %.1f%%" %
          (s0["n"], s0["per"], s0["t"], s0["cagr"], s0["mdd_pct"]))

    print("\n%-6s %-9s %-4s %4s %6s %9s %8s %6s %7s %7s %7s %8s" %
          ("move", "arm", "cap", "n", "win%", "net pts", "per", "t", "CAGR%", "MaxDD%", "Calmar", "cyc/camp"))
    print("%-6s %-9s %-4s %4d %6.1f %9.1f %8.1f %6.2f %7.2f %7.1f %7.2f %8.2f" %
          ("none", "hold", "-", s0["n"], s0["win"], s0["total"], s0["per"], s0["t"],
           s0["cagr"], s0["mdd_pct"], s0["calmar"], s0["cycles"]))
    for move in MOVES:
        for arm in ARMS:
            caps = CAPS if arm == "recentre" else [0]
            for cap in caps:
                keep = [] if (arm == "recentre" and cap == 99 and abs(move - 0.02) < 1e-9) else None
                camps = evaluate(move, arm, cap, keep)
                if not camps:
                    continue
                s = summarise(camps, "%s %s cap%s" % (move, arm, cap))
                grid.append(dict(move="%.1f%%" % (100 * move), arm=arm, cap=cap, **s))
                print("%-6s %-9s %-4s %4d %6.1f %9.1f %8.1f %6.2f %7.2f %7.1f %7.2f %8.2f" %
                      ("%.1f%%" % (100 * move), arm, cap, s["n"], s["win"], s["total"],
                       s["per"], s["t"], s["cagr"], s["mdd_pct"], s["calmar"], s["cycles"]))
                if keep:
                    out = os.path.join(RES, "trades_recentre_2pct.csv")
                    with open(out, "w", newline="") as f:
                        wtr = csv.DictWriter(f, fieldnames=list(keep[0].keys()))
                        wtr.writeheader()
                        for r in keep:
                            wtr.writerow(r)
                    print("      wrote %s (%d cycles)" % (out, len(keep)))

    out = os.path.join(RES, "phase_e_grid.csv")
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
        wtr.writeheader()
        for g in grid:
            wtr.writerow(g)
    print("\nwrote %s (%d cells)" % (out, len(grid)))

    # how often does the move rule even fire?
    print("\ntrigger frequency (share of campaigns where the move rule fired at least once):")
    for g in grid:
        if g["arm"] == "recentre" and g["cap"] == 99:
            print("   move %-5s  %5.1f%% of campaigns  avg %.2f cycles" %
                  (g["move"], g["trig"], g["cycles"]))


if __name__ == "__main__":
    main()
