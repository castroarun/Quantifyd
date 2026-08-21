#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/121 S2 - LONG-SAMPLE window outcomes, joined to the day-level features.

For each of the three live non-expiry TimeB window shapes:
    MON  13:00-14:00   (live: NIFTY DTE1)
    WED  10:30-12:00   (live: SENSEX DTE1)
    FRI  10:00-12:00   (live: NIFTY DTE2)
measure, on every day of the long sample, the intraday quantity that actually
kills a short straddle: the maximum absolute excursion of the underlying away
from the window's own start price, in basis points.

Series used
  SENSEX `minute`   2021-01-01 .. 2026-08-20  -> 1-MINUTE TRUTH
  NIFTY50 `5minute` 2015-02-02 .. 2026-07-17  -> 5-min, understates extremes
  SENSEX resampled to 5-min                   -> measures HOW MUCH 5-min understates

Two outcomes are recorded:
  exc_bp    raw max excursion, bp
  exc_norm  exc_bp divided by the day's VIX-implied 1-day sigma in bp,
            i.e. how big the move was RELATIVE TO WHAT THE OPTION MARKET CHARGED.
            This is the outcome that matters: research/120 showed that windows
            with more risk earn more, so a filter that only predicts raw move size
            may simply be predicting expensive days and skipping the premium too.

Also records the causal, at-window-start intraday features:
  pre_move_bp   |start price - day open| in bp     (how far today has already run)
  pre_range_bp  (high-low from 09:15 to window start) in bp

Writes results/window_outcomes_<SERIESTAG>.csv  (one row per day x window)
READ-ONLY on market_data.db.
"""
import sqlite3, csv, os, math
from datetime import date

MD = "/home/arun/quantifyd/backtest_data/market_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

WINDOWS = {
    "MON_1300_1400": ("13:00", "14:00", 0),
    "WED_1030_1200": ("10:30", "12:00", 2),
    "FRI_1000_1200": ("10:00", "12:00", 4),
}

SERIES = [
    ("SENSEX_1MIN", "SENSEX", "minute", 1),
    ("NIFTY50_5MIN", "NIFTY50", "5minute", 5),
    ("SENSEX_5MIN", "SENSEX", "minute", 5),   # resampled, for the understatement control
]


def hm2m(h):
    return int(h[:2]) * 60 + int(h[3:5])


def load_bars(c, sym, tf, resample_to):
    """-> {day: [(minute, o, h, l, close), ...]} sorted, session only."""
    out = {}
    q = ("SELECT date,open,high,low,close FROM market_data_unified "
         "WHERE symbol=? AND timeframe=? ORDER BY date")
    for dt, o, h, l, cl in c.execute(q, (sym, tf)):
        d, t = dt[:10], dt[11:16]
        if not t or t < "09:15" or t > "15:35":
            continue
        if o is None or h is None or l is None or cl is None or cl <= 0:
            continue
        m = hm2m(t)
        if resample_to > 1:
            m = (m // resample_to) * resample_to
        b = out.setdefault(d, {})
        a = b.get(m)
        if a is None:
            b[m] = [o, h, l, cl]
        else:
            a[1] = max(a[1], h); a[2] = min(a[2], l); a[3] = cl
    return {d: sorted((m, v[0], v[1], v[2], v[3]) for m, v in b.items())
            for d, b in out.items()}


def load_features(sym):
    p = os.path.join(RES, "daily_features_%s.csv" % sym)
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            out[r["day"]] = r
    return out


FEATCOLS = ["cpr_today", "cpr_prev", "wcpr_this", "wcpr_prev", "gap_pct", "gap_abs",
            "pdr_pct", "pdr_rel", "atr14_pct", "ret_prev", "vix_open", "vix_prevclose",
            "vix_chg_oc_pct", "vix_chg_oc_pts", "vix_chg_cc_pct", "vix_chg_cc_pts"]

OUT = ["series", "window", "day", "dow", "n_bars_window", "p0", "exc_bp", "exc_up_bp",
       "exc_dn_bp", "net_bp", "pre_move_bp", "pre_range_bp", "vix_sigma_bp",
       "exc_norm"] + FEATCOLS


def main():
    os.makedirs(RES, exist_ok=True)
    c = sqlite3.connect("file:%s?mode=ro" % MD, uri=True)
    for tag, sym, tf, res in SERIES:
        feats = load_features(sym)
        bars = load_bars(c, sym, tf, res)
        days = sorted(bars)
        print("%s: %d days %s..%s" % (tag, len(days), days[0], days[-1]), flush=True)
        p = os.path.join(RES, "window_outcomes_%s.csv" % tag)
        f = open(p, "w", newline="")
        w = csv.DictWriter(f, fieldnames=OUT); w.writeheader()
        nw = 0
        for d in days:
            bl = bars[d]
            if len(bl) < (60 if res == 1 else 12):
                continue
            fr = feats.get(d)
            if not fr:
                continue
            dayopen = bl[0][1]
            for wname, (s, e, dow) in WINDOWS.items():
                m0, m1 = hm2m(s), hm2m(e)
                inw = [b for b in bl if m0 <= b[0] < m1]
                if len(inw) < (30 if res == 1 else 6):
                    continue
                p0 = inw[0][1]
                if not p0 or p0 <= 0:
                    continue
                hi = max(b[2] for b in inw)
                lo = min(b[3] for b in inw)
                up = (hi - p0) / p0 * 1e4
                dn = (p0 - lo) / p0 * 1e4
                exc = max(up, dn, 0.0)
                net = (inw[-1][4] - p0) / p0 * 1e4
                pre = [b for b in bl if b[0] < m0]
                pre_move = abs(p0 - dayopen) / dayopen * 1e4 if dayopen else ""
                if pre:
                    ph = max(b[2] for b in pre); pl = min(b[3] for b in pre)
                    pre_range = (ph - pl) / p0 * 1e4
                else:
                    pre_range = ""
                vix = fr.get("vix_open") or ""
                if vix not in ("", None):
                    sig = float(vix) / 100.0 / math.sqrt(252.0) * 1e4
                    norm = exc / sig if sig > 0 else ""
                else:
                    sig, norm = "", ""
                row = dict(series=tag, window=wname, day=d, dow=fr["dow"],
                           n_bars_window=len(inw), p0=round(p0, 2),
                           exc_bp=round(exc, 2), exc_up_bp=round(up, 2),
                           exc_dn_bp=round(dn, 2), net_bp=round(net, 2),
                           pre_move_bp=round(pre_move, 2) if pre_move != "" else "",
                           pre_range_bp=round(pre_range, 2) if pre_range != "" else "",
                           vix_sigma_bp=round(sig, 2) if sig != "" else "",
                           exc_norm=round(norm, 4) if norm != "" else "")
                for k in FEATCOLS:
                    row[k] = fr.get(k, "")
                w.writerow(row); nw += 1
        f.close()
        print("  wrote %s (%d rows)" % (p, nw), flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
