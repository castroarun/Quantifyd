#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/133 Stage B - the multi-year 13:00->15:20 excursion clock, DTE-labelled from a
built EXPIRY CALENDAR (never from the weekday).

Reads backtest_data/market_data.db :: market_data_unified, READ-ONLY, always pinned on
symbol AND timeframe:
  * SENSEX  'minute'   2021-01-01 ->   (1-min truth, ~1,354 trading days)
  * NIFTY50 '5minute'  2015-01-01 ->   (cross-check; the ONLY series in our data that
                                        contains COVID March 2020)

r/121 licence: for the MAXIMUM EXCURSION inside a fixed window, 5-minute bars == 1-minute
bars exactly (same covering high/low set). Excursions are valid; PATHS are not, so the
5-minute series is never used for dwell / gap-through timing.

Expiry calendar: for each era we take the era's expiry weekday; if that calendar date is
not a trading day in the series, we walk BACK to the previous trading day - so holiday-
shifted expiries get DTE0 correctly. DTE = trading days from the day to its expiry.

Per day: the 13:00 reference, the ATM strike the live rule would pick, and inside
13:00->15:20 the max |S - ref| (excursion) and max |S - K| (the intrinsic-floor distance),
both intrabar, in points and bp; plus the terminal move.
Writes results/stage_b_days.csv
"""
import sqlite3, csv, os
from datetime import date, timedelta

Q = "/home/arun/quantifyd/"
MD = Q + "backtest_data/market_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
LOG = os.path.join(RES, "stage_b.log")

ENTRY_M = 13 * 60
EXIT_M = 15 * 60 + 20

SERIES = [
    # (label, symbol, timeframe, from_date, strike_step)
    ("SENSEX_1min", "SENSEX", "minute", "2021-01-01", 100),
    ("NIFTY50_5min", "NIFTY50", "5minute", "2015-01-01", 50),
]

# (from, to, expiry weekday 0=Mon..4=Fri).  Outside every era -> NOLABEL.
ERAS = {
    "SENSEX_1min": [("2024-01-01", "2024-12-31", 4),
                    ("2025-01-01", "2025-08-31", 1),
                    ("2025-09-01", "2099-01-01", 3)],
    "NIFTY50_5min": [("2019-02-11", "2025-08-31", 3),
                     ("2025-09-01", "2099-01-01", 1)],
}
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def era_wd(label, day):
    for lo, hi, ew in ERAS[label]:
        if lo <= day <= hi:
            return ew
    return None


def master_calendar():
    """The trading-day calendar, built from the UNION of several series so that a gap in
    one symbol cannot shift another symbol's expiry label. A single-symbol hole (e.g.
    SENSEX 1-min is missing 2026-05-14 and 2026-07-09, both real trading days) would
    otherwise walk the expiry back a day and mislabel the neighbours."""
    c = sqlite3.connect("file:%s?mode=ro" % MD, uri=True)
    days = set()
    for sym, tf in (("SENSEX", "day"), ("SENSEX", "minute"), ("NIFTY50", "5minute"),
                    ("NIFTY50", "day")):
        for (d,) in c.execute("SELECT DISTINCT substr(date,1,10) FROM market_data_unified "
                              "WHERE symbol=? AND timeframe=?", (sym, tf)):
            if date.fromisoformat(d).weekday() < 5:
                days.add(d)
    c.close()
    return sorted(days)


def build_expiry_calendar(label, tdays, tset):
    """tdays = sorted list of days to label. tset = the MASTER trading-day set."""
    exp_of = {}
    for d in tdays:
        ew = era_wd(label, d)
        if ew is None:
            exp_of[d] = None
            continue
        dd = date.fromisoformat(d)
        # this week's target expiry weekday; if the day is already past it, take next week
        delta = (ew - dd.weekday()) % 7
        cand = dd + timedelta(days=delta)
        # walk BACK from the target to the last actual trading day (holiday shift)
        for _ in range(6):
            cs = cand.isoformat()
            if cs in tset:
                break
            cand -= timedelta(days=1)
        else:
            exp_of[d] = None
            continue
        cs = cand.isoformat()
        if cs < d:
            # this week's expiry already passed (holiday pulled it before today) -> next week
            cand = dd + timedelta(days=delta + 7)
            for _ in range(6):
                cs = cand.isoformat()
                if cs in tset:
                    break
                cand -= timedelta(days=1)
            if cs < d:
                exp_of[d] = None
                continue
        exp_of[d] = cs
    return exp_of


def trading_dte(day, exp, tdays_idx):
    if exp is None:
        return None
    i, j = tdays_idx.get(day), tdays_idx.get(exp)
    if i is None or j is None:
        return None
    return j - i


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    global TSET
    TSET = set(master_calendar())
    log("master trading calendar: %d weekdays" % len(TSET))
    f = open(os.path.join(RES, "stage_b_days.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=[
        "series", "day", "weekday", "expiry", "dte_trd", "ref1300", "strike",
        "exc_pts", "exc_bp", "distK_pts", "distK_bp", "term_pts", "term_bp",
        "hi", "lo", "n_bars"])
    w.writeheader()

    for label, sym, tf, dfrom, step in SERIES:
        log("loading %s (%s / %s) from %s" % (label, sym, tf, dfrom))
        c = sqlite3.connect("file:%s?mode=ro" % MD, uri=True)
        days = {}
        for dt, op, hi, lo, cl in c.execute(
                "SELECT date, open, high, low, close FROM market_data_unified "
                "WHERE symbol=? AND timeframe=? AND date>=? ORDER BY date",
                (sym, tf, dfrom)):
            d = dt[:10]
            mi = int(dt[11:13]) * 60 + int(dt[14:16])
            days.setdefault(d, []).append((mi, op, hi, lo, cl))
        c.close()
        tdays = sorted(k for k in days if date.fromisoformat(k).weekday() < 5
                       and len(days[k]) >= 20)
        log("  %s trading days: %d (%s .. %s)" % (label, len(tdays), tdays[0], tdays[-1]))
        exp_of = build_expiry_calendar(label, tdays, TSET)
        idx = {d: i for i, d in enumerate(tdays)}
        n_lab = sum(1 for d in tdays if exp_of[d])
        log("  expiry-calendar labelled days: %d" % n_lab)
        kept = 0
        for d in tdays:
            bars = sorted(days[d])
            inwin = [b for b in bars if ENTRY_M <= b[0] <= EXIT_M]
            if len(inwin) < 5:
                continue
            # reference = the close of the last bar at/just before 13:00, else first in-window open
            pre = [b for b in bars if b[0] <= ENTRY_M]
            ref = pre[-1][4] if pre else inwin[0][1]
            if not ref:
                continue
            K = round(ref / float(step)) * step
            hi = max(b[2] for b in inwin)
            lo = min(b[3] for b in inwin)
            exc = max(abs(hi - ref), abs(ref - lo))
            distK = max(abs(hi - K), abs(K - lo))
            term = inwin[-1][4] - ref
            e = exp_of[d]
            dte = trading_dte(d, e, idx)
            kept += 1
            w.writerow(dict(series=label, day=d, weekday=WD[date.fromisoformat(d).weekday()],
                            expiry=e or "", dte_trd=("" if dte is None else dte),
                            ref1300=round(ref, 2), strike=K,
                            exc_pts=round(exc, 1), exc_bp=round(1e4 * exc / ref, 1),
                            distK_pts=round(distK, 1), distK_bp=round(1e4 * distK / ref, 1),
                            term_pts=round(term, 1), term_bp=round(1e4 * term / ref, 1),
                            hi=round(hi, 1), lo=round(lo, 1), n_bars=len(inwin)))
            f.flush() if kept % 500 == 0 else None
        log("  %s windows built: %d" % (label, kept))
        del days
    f.close()
    log("DONE stage B")


if __name__ == "__main__":
    main()
