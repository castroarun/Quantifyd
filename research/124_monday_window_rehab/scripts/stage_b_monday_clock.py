#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/124 Stage B - the Monday calmness clock on the long sample, era-labelled.

Descendant of research/122's stage_b_allweekday_clock.py, run on the r/124 fine grid
(137 windows). For every day x window: max |excursion| from window-entry price (bp) and
|terminal| move (bp). All weekdays kept (Monday is the analysis cut; other weekdays are
the comparison + shuffle-null pool). DTE-era labels per r/118's expiry history.

Data (market_data.db :: market_data_unified, READ-ONLY, filtered symbol AND timeframe):
  * SENSEX  'minute'   2021-01-01 ->  (1-min truth)
  * NIFTY50 '5minute'  2015-01-01 ->  (r/121 licence: 5-min == 1-min EXACTLY for
    max-excursion-in-a-fixed-window; excursions only, never path fills)
Writes results/stage_b_window_days.csv + results/stage_b.log
"""
import sqlite3, csv, os
from datetime import date

Q = "/home/arun/quantifyd/"
MD = Q + "backtest_data/market_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(RES, "stage_b.log")

SESS_END_M = 15 * 60 + 20
STARTS = [9 * 60 + 16] + list(range(9 * 60 + 30, 14 * 60 + 1, 15))
DURATIONS = [30, 45, 60, 75, 90, 105, 120]

# expiry-era tables (r/118): (from_date, to_date, expiry_weekday 0=Mon..4=Fri)
ERAS = {
    "SENSEX": [("2024-01-01", "2024-12-31", 4),
               ("2025-01-01", "2025-08-31", 1),
               ("2025-09-01", "2099-01-01", 3)],
    "NIFTY":  [("2019-02-11", "2025-08-31", 3),
               ("2025-09-01", "2099-01-01", 1)],
}


def m2hm(m):
    return "%02d%02d" % (m // 60, m % 60)


def windows():
    seen, out = set(), []
    for s in STARTS:
        for dur in DURATIONS:
            m1 = min(s + dur, SESS_END_M)
            if m1 - s < 30 or (s, m1) in seen:
                continue
            seen.add((s, m1))
            out.append(("W_%s_%s" % (m2hm(s), m2hm(m1)), s, m1))
    return out


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def dte_label(venue, day, wd):
    for lo, hi, ew in ERAS[venue]:
        if lo <= day <= hi:
            return (ew - wd) % 5
    return None


def era_label(venue, day):
    for i, (lo, hi, ew) in enumerate(ERAS[venue]):
        if lo <= day <= hi:
            return "%s_exp%s" % (["era1", "era2", "era3"][i], "MTWTF"[ew])
    return "prelabel"


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
    cells = windows()
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
        era = era_label(venue, d)
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
                            dte_trd=("" if dte is None else dte), era=era, cell=lbl,
                            exc_bp=round(1e4 * exc / p0, 1),
                            term_bp=round(1e4 * abs(term) / p0, 1)))
        if n % 250 == 0:
            log("  %s: %d days done (%s)" % (venue, n, d))
    return n


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    f = open(os.path.join(RES, "stage_b_window_days.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=["series", "venue", "day", "weekday", "dte_trd",
                                      "era", "cell", "exc_bp", "term_bp"])
    w.writeheader()
    log("grid: %d windows" % len(windows()))
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
