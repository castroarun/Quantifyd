#!/usr/bin/env python3
"""
Phase D — the monitoring-frequency question answered on REAL 1-minute option prices.

Our own chain recorder (options_data.db -> option_chain, 28.3M rows) holds real 1-minute
NIFTY option quotes from 2026-04-20. It picks each contract up only ~27 calendar days
before expiry, so it cannot host a 45-DTE ENTRY. What it does cover is the back half of
every holding window, at 1-minute resolution, on real traded prices.

Two things are measured here, both on real ticks, no modelling:

  D1  INTRADAY TRAVEL. For every recorded NIFTY expiry and day, take the ATM straddle and
      measure how far the combined premium ranged intraday versus where it closed. This is
      exactly the quantity a daily-close backtest is blind to. If travel is small relative
      to the strategy's +100% / -50% trigger bands, no check frequency can matter.

  D2  RECONSTRUCTION CHECK. For the monthly expiries where the recorder overlaps the
      strategy's holding window, compare the real 1-minute combined premium against the
      reconstructed mark used in Phase B (Black-76 on real 5-min spot, causal IV).
      Reports mean abs error, max error, and whether the two disagree about any trigger.
"""
import os
import sqlite3
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine45 import (connect, trading_days, monthly_expiries, nifty_daily_close,
                      build_trade, dparse)

OPT_DB = "/home/arun/quantifyd/backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

TARGET, STOP = 0.50, 2.00
CAL = []          # trading calendar, filled in main()


def oc():
    return sqlite3.connect("file:%s?mode=ro" % OPT_DB, uri=True)


def recorded_expiries(con):
    return [r[0] for r in con.execute(
        "SELECT DISTINCT expiry_date FROM option_chain WHERE symbol='NIFTY' "
        "ORDER BY expiry_date")]


def days_for_expiry(con, exp):
    """Range predicates only — date(snapshot_time) would defeat every index on a 28M-row
    table. Get the contract's first/last tick, then intersect with the trading calendar."""
    r = con.execute("SELECT MIN(snapshot_time), MAX(snapshot_time) FROM option_chain "
                    "WHERE expiry_date=? AND symbol='NIFTY'", (exp,)).fetchone()
    if not r or not r[0]:
        return []
    a, b = r[0][:10], r[1][:10]
    return [d for d in CAL if a <= d <= b]


def atm_strike(con, exp, day):
    """ATM = strike nearest the day's first recorded underlying spot."""
    lo, hi = day + "T00:00", day + "T23:59:59"
    r = con.execute(
        "SELECT underlying_spot FROM option_chain WHERE symbol='NIFTY' AND expiry_date=? "
        "AND snapshot_time>=? AND snapshot_time<=? AND underlying_spot IS NOT NULL "
        "ORDER BY snapshot_time LIMIT 1", (exp, lo, hi)).fetchone()
    if not r or not r[0]:
        return None, None
    spot = float(r[0])
    r2 = con.execute(
        "SELECT strike FROM option_chain WHERE symbol='NIFTY' AND expiry_date=? "
        "AND snapshot_time>=? AND snapshot_time<=? ORDER BY ABS(strike-?) LIMIT 1",
        (exp, lo, hi, spot)).fetchone()
    return (float(r2[0]) if r2 and r2[0] is not None else None), spot


def combined_series(con, exp, day, strike):
    """Real 1-minute combined ATM straddle premium: CE.ltp + PE.ltp, aligned by minute."""
    rows = con.execute(
        "SELECT snapshot_time, instrument_type, ltp FROM option_chain "
        "WHERE expiry_date=? AND strike=? AND snapshot_time>=? AND snapshot_time<=? "
        "AND ltp IS NOT NULL AND ltp>0 ORDER BY snapshot_time",
        (exp, strike, day + "T00:00", day + "T23:59:59")).fetchall()
    ce, pe = {}, {}
    for ts, it, ltp in rows:
        m = ts[11:16]
        (ce if it == "CE" else pe)[m] = float(ltp)
    mins = sorted(set(ce) & set(pe))
    return [(m, ce[m] + pe[m]) for m in mins]


# --------------------------------------------------------------- D1 ----------
def d1_travel(con):
    print("=" * 96)
    print("D1 — REAL intraday travel of the ATM straddle (1-minute quotes, our recorder)")
    print("=" * 96)
    buckets = defaultdict(list)
    daily_rows = []
    for exp in recorded_expiries(con):
        for day in days_for_expiry(con, exp):
            k, spot = atm_strike(con, exp, day)
            if k is None:
                continue
            ser = combined_series(con, exp, day, k)
            if len(ser) < 60:
                continue
            vals = [v for _, v in ser]
            close, lo, hi = vals[-1], min(vals), max(vals)
            if close <= 0:
                continue
            dte = (dparse(exp) - dparse(day)).days
            up = hi / close - 1.0
            dn = 1.0 - lo / close
            buckets[dte].append((up, dn))
            daily_rows.append((day, exp, dte, k, close, lo, hi, up, dn, len(ser)))

    print("\n%-5s %6s %10s %10s %10s %10s" %
          ("DTE", "days", "up mean", "up p95", "down mean", "down p95"))
    print("      (how far above / below the CLOSE the combined premium travelled that day)")
    for dte in sorted(buckets):
        v = buckets[dte]
        if len(v) < 3:
            continue
        ups = sorted(x[0] for x in v)
        dns = sorted(x[1] for x in v)
        p95 = lambda a: a[min(len(a) - 1, int(.95 * len(a)))]
        print("%-5d %6d %9.1f%% %9.1f%% %9.1f%% %9.1f%%" %
              (dte, len(v), 100 * statistics.mean(ups), 100 * p95(ups),
               100 * statistics.mean(dns), 100 * p95(dns)))

    allv = [x for v in buckets.values() for x in v]
    ups = sorted(x[0] for x in allv); dns = sorted(x[1] for x in allv)
    print("\nAll %d recorded day-contracts:" % len(allv))
    print("  travel ABOVE the close : mean %.1f%%  p95 %.1f%%  max %.1f%%" %
          (100 * statistics.mean(ups), 100 * ups[int(.95 * len(ups))], 100 * max(ups)))
    print("  travel BELOW the close : mean %.1f%%  p95 %.1f%%  max %.1f%%" %
          (100 * statistics.mean(dns), 100 * dns[int(.95 * len(dns))], 100 * max(dns)))
    print("\n  The strategy's triggers sit +100%% (stop) and -50%% (target) away from the")
    print("  ENTRY CREDIT, not from the daily close. Days where a single session's travel")
    print("  is large enough to jump a trigger the close would have missed:")
    for band, label in ((1.00, "+100% in one day (stop side)"), (0.50, "-50% in one day (target side)")):
        n_up = sum(1 for x in allv if x[0] >= band)
        n_dn = sum(1 for x in allv if x[1] >= band)
        print("    %-32s up-side %d / %d    down-side %d / %d" %
              (label, n_up, len(allv), n_dn, len(allv)))
    return daily_rows


# --------------------------------------------------------------- D2 ----------
def d2_overlap(con, mcon):
    print("\n" + "=" * 96)
    print("D2 — where the recorder OVERLAPS a real 45-DTE trade: real ticks vs the model")
    print("=" * 96)
    days = trading_days(mcon, "2025-06-01")
    spot = nifty_daily_close(mcon)
    exps = monthly_expiries(mcon, days, "2026-01-01", "2026-12-31")
    rec = set(recorded_expiries(con))

    for ym, exp in exps.items():
        if exp not in rec:
            continue
        t = build_trade(mcon, exp, days, spot)
        if not t:
            continue
        hold = [r["date"] for r in t["path"]]
        rec_days = set(days_for_expiry(con, exp))
        overlap = [d for d in hold if d in rec_days]
        if not overlap:
            continue
        credit = t["credit"]
        print("\nExpiry %s | entry %s @ strike %.0f | credit %.1f pts | holding %d sessions | "
              "recorder covers %d of them (%s .. %s)" %
              (exp, t["entry_date"], t["strike"], credit, len(hold), len(overlap),
               overlap[0], overlap[-1]))
        print("  %-12s %8s %9s %9s %9s | %8s %8s" %
              ("date", "DTE", "real min", "real close", "real max", "min/cr", "max/cr"))
        worst_gap = 0.0
        for d in overlap:
            ser = combined_series(con, exp, d, t["strike"])
            if len(ser) < 30:
                continue
            vals = [v for _, v in ser]
            eod = next((r["comb"] for r in t["path"] if r["date"] == d), None)
            dte = (dparse(exp) - dparse(d)).days
            print("  %-12s %8d %9.1f %9.1f %9.1f | %8.2f %8.2f%s" %
                  (d, dte, min(vals), vals[-1], max(vals),
                   min(vals) / credit, max(vals) / credit,
                   ("   [bhav close %.1f, diff %.1f]" % (eod, vals[-1] - eod)) if eod else ""))
            if eod:
                worst_gap = max(worst_gap, abs(vals[-1] - eod))
            # would an intraday check have fired where the close did not?
            if min(vals) <= TARGET * credit and (eod or 1e9) > TARGET * credit:
                print("      !! intraday TARGET touch that the daily close missed")
            if max(vals) >= STOP * credit and (eod or 0) < STOP * credit:
                print("      !! intraday STOP touch that the daily close missed")
        print("  max |1-min last quote - bhav close| over the overlap: %.1f pts" % worst_gap)


if __name__ == "__main__":
    con, mcon = oc(), connect()
    CAL = trading_days(mcon, "2026-04-01")
    print("trading calendar loaded: %d sessions %s .. %s" % (len(CAL), CAL[0], CAL[-1]))
    rows = d1_travel(con)
    d2_overlap(con, mcon)
    import csv
    with open(os.path.join(RES, "intraday_travel.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "expiry", "dte", "strike", "close", "low", "high",
                    "up_vs_close", "down_vs_close", "minutes"])
        w.writerows(rows)
    print("\nwrote results/intraday_travel.csv (%d day-contracts)" % len(rows))
