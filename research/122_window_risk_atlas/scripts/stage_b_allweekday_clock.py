#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/122 Stage B - the ALL-WEEKDAY violent-move clock, DTE-labelled.

Extends research/120's stage_b_volclock.py from Fridays to every weekday, and labels
every day by TRADING-day DTE using the expiry-era history (r/118):

  SENSEX weekly expiry weekday:  2024-01..2024-12 Fri | 2025-01..2025-08 Tue | 2025-09.. Thu
                                 (before 2024: NOLABEL - no weekly-expiry evidence in our data)
  NIFTY  weekly expiry weekday:  2019-02-11..2025-08-31 Thu | 2025-09-01.. Tue
                                 (before weekly NIFTY options existed: NOLABEL)

DTE = (expiry_weekday - day_weekday) mod 5  (trading days, holiday shifts ignored ->
label noise of at most one day on holiday weeks; stated in RESULTS).

Data (market_data.db :: market_data_unified, READ-ONLY, ALWAYS filtered by symbol AND
timeframe):
  * SENSEX  'minute'   2021-01-01 -> now   (1-min truth)
  * NIFTY50 '5minute'  2015-02-02 -> now   (5-min under the r/121 licence: for the MAX
    EXCURSION inside a fixed window, 5-min == 1-min exactly - proved in r/121 s7;
    the no-5-min rule bites on the path only)

For every day x atlas cell (same grid + deployed windows as Stage A): the maximum
absolute excursion from the window-entry price, in bp, and the |terminal| move in bp.
Writes one row per day x cell -> results/stage_b_window_days.csv (analysis aggregates).
"""
import sqlite3, csv, os
from datetime import date

Q = "/home/arun/quantifyd/"
MD = Q + "backtest_data/market_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(RES, "stage_b.log")

SESS_END_M = 15 * 60 + 20
GRID_STARTS = []
_m = 9 * 60 + 20
while _m <= 14 * 60 + 20:
    GRID_STARTS.append(_m)
    _m += 30
GRID_STARTS.append(14 * 60 + 30)
DURATIONS = [60, 90, 120, "HOLD"]
DEPLOYED_WINDOWS = [
    ("DEP_0930_1100", 9 * 60 + 30, 11 * 60),
    ("DEP_1000_1200", 10 * 60, 12 * 60),
    ("DEP_1030_1200", 10 * 60 + 30, 12 * 60),
    ("DEP_1300_1400", 13 * 60, 14 * 60),
    ("DEP_1300_1520", 13 * 60, 15 * 60 + 20),
]

# expiry-era tables: (from_date, to_date, expiry_weekday 0=Mon..4=Fri)
ERAS = {
    "SENSEX": [("2024-01-01", "2024-12-31", 4),
               ("2025-01-01", "2025-08-31", 1),
               ("2025-09-01", "2099-01-01", 3)],
    "NIFTY":  [("2019-02-11", "2025-08-31", 3),
               ("2025-09-01", "2099-01-01", 1)],
}


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def m2hm(m):
    return "%02d:%02d" % (m // 60, m % 60)


def dte_label(venue, day, wd):
    for lo, hi, ew in ERAS[venue]:
        if lo <= day <= hi:
            return (ew - wd) % 5
    return None  # NOLABEL era


def load(sym, tf, dfrom):
    c = sqlite3.connect("file:%s?mode=ro" % MD, uri=True)
    days = {}
    for dt, cl, hi, lo in c.execute(
            "SELECT date, close, high, low FROM market_data_unified "
            "WHERE symbol=? AND timeframe=? AND date>=? ORDER BY date", (sym, tf, dfrom)):
        d = dt[:10]
        mi = int(dt[11:13]) * 60 + int(dt[14:16])
        days.setdefault(d, []).append((mi, cl, hi, lo))
    c.close()
    return days


def run_series(days, series, venue, w):
    WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    cells = []
    for s in GRID_STARTS:
        for dur in DURATIONS:
            m1 = SESS_END_M if dur == "HOLD" else min(s + dur, SESS_END_M)
            if m1 <= s:
                continue
            cells.append(("G_%s_%s" % (m2hm(s), "H" if dur == "HOLD" else str(dur)), s, m1))
    for lbl, s, m1 in DEPLOYED_WINDOWS:
        cells.append((lbl, s, m1))
    n = 0
    for d, bars in sorted(days.items()):
        wd = date.fromisoformat(d).weekday()
        if wd > 4:
            continue
        bars = sorted(bars)
        px = {mi: cl for mi, cl, _, _ in bars}
        hi = {mi: h for mi, _, h, _ in bars}
        lo = {mi: l for mi, _, _, l in bars}
        mins = sorted(m for m in px if 9 * 60 + 15 <= m <= SESS_END_M)
        if len(mins) < 40:
            continue
        n += 1
        dte = dte_label(venue, d, wd)
        for lbl, s, m1 in cells:
            m0 = next((m for m in mins if m >= s), None)
            if m0 is None or m0 > s + 10:
                continue
            p0 = px[m0]
            if not p0:
                continue
            exc = 0.0
            term = 0.0
            got = False
            for m in mins:
                if m <= m0 or m > m1:
                    continue
                got = True
                e = max(abs(hi[m] - p0), abs(lo[m] - p0))
                if e > exc:
                    exc = e
                term = px[m] - p0
            if not got:
                continue
            w.writerow(dict(series=series, venue=venue, day=d, weekday=WD[wd],
                            dte_trd=("" if dte is None else dte), cell=lbl,
                            start=m2hm(s), end=m2hm(m1),
                            exc_bp=round(1e4 * exc / p0, 1),
                            term_bp=round(1e4 * abs(term) / p0, 1)))
    return n


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    f = open(os.path.join(RES, "stage_b_window_days.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=["series", "venue", "day", "weekday", "dte_trd",
                                      "cell", "start", "end", "exc_bp", "term_bp"])
    w.writeheader()
    log("loading SENSEX 1-minute 2021-01-01 ->")
    sx = load("SENSEX", "minute", "2021-01-01")
    log("  SENSEX days=%d" % len(sx))
    n = run_series(sx, "SENSEX_1min", "SENSEX", w)
    log("  SENSEX windows built over %d days" % n)
    f.flush()
    del sx
    log("loading NIFTY50 5-minute 2015-01-01 ->")
    nf = load("NIFTY50", "5minute", "2015-01-01")
    log("  NIFTY50 days=%d" % len(nf))
    n = run_series(nf, "NIFTY50_5min", "NIFTY", w)
    log("  NIFTY windows built over %d days" % n)
    f.close()
    log("DONE")


if __name__ == "__main__":
    main()
