#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/121 S1 - day-level regime features for both venues, LONG sample.

Every feature is CAUSAL: computable before 09:15 of the day it labels, except the
opening gap and the VIX open, which are known at 09:15 and therefore before every
window in this study (earliest window start = 10:00).

Sources (market_data.db, READ-ONLY):
  NIFTY50  day     2011-01-03 .. 2026-08-20   (3,875 rows)
  SENSEX   minute  2021-01-01 .. 2026-08-20   -> resampled to daily OHLC
  INDIAVIX day     2015-01-01 .. 2026-08-20   (3,396 rows)

CPR width convention follows research/67:
  BC=(H+L)/2, P=(H+L+C)/3, TC=2P-BC, width=|TC-BC|=|2C-H-L|/3, expressed as % of C.
  The CPR *in force on day d* is built from day d-1 OHLC. So `cpr_today` is known
  at the close of d-1 and `cpr_prev` is the band that was in force on d-1.
  Weekly CPR in force during week w is built from week w-1 OHLC.

Writes results/daily_features_<VENUE>.csv
"""
import sqlite3, csv, os, sys
from datetime import date

MD = "/home/arun/quantifyd/backtest_data/market_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def con():
    return sqlite3.connect("file:%s?mode=ro" % MD, uri=True)


def daily_from_daily(c, sym):
    out = {}
    q = "SELECT date,open,high,low,close FROM market_data_unified WHERE symbol=? AND timeframe=? ORDER BY date"
    for d, o, h, l, cl in c.execute(q, (sym, "day")):
        d = d[:10]
        if o is None or h is None or l is None or cl is None:
            continue
        if h <= 0 or cl <= 0:
            continue
        out[d] = (o, h, l, cl)
    return out


def daily_from_intraday(c, sym, tf):
    """Resample an intraday series to daily OHLC. Session bars only (09:15-15:35)."""
    agg = {}
    q = "SELECT date,open,high,low,close FROM market_data_unified WHERE symbol=? AND timeframe=? ORDER BY date"
    for dt, o, h, l, cl in c.execute(q, (sym, tf)):
        d, t = dt[:10], dt[11:16]
        if not t or t < "09:15" or t > "15:35":
            continue
        if o is None or h is None or l is None or cl is None or cl <= 0:
            continue
        a = agg.get(d)
        if a is None:
            agg[d] = [o, h, l, cl]
        else:
            a[1] = max(a[1], h); a[2] = min(a[2], l); a[3] = cl
    return {d: tuple(v) for d, v in agg.items() if v[1] > 0}


def cpr_width_pct(o, h, l, cl):
    return abs(2.0 * cl - h - l) / 3.0 / cl * 100.0


def isoweek(d):
    dt = date.fromisoformat(d)
    y, w, _ = dt.isocalendar()
    return (y, w)


def build(sym, bars, vix):
    days = sorted(bars)
    wk = {}
    for d in days:
        k = isoweek(d)
        o, h, l, cl = bars[d]
        a = wk.get(k)
        if a is None:
            wk[k] = [o, h, l, cl]
        else:
            a[1] = max(a[1], h); a[2] = min(a[2], l); a[3] = cl
    wkeys = sorted(wk)
    wpos = {k: i for i, k in enumerate(wkeys)}

    rows = []
    ranges = []
    trs = []
    for i, d in enumerate(days):
        o, h, l, cl = bars[d]
        rec = dict(day=d, dow=date.fromisoformat(d).weekday(),
                   open=round(o, 2), high=round(h, 2), low=round(l, 2), close=round(cl, 2))
        if i >= 1:
            o1, h1, l1, c1 = bars[days[i - 1]]
            rec["cpr_today"] = round(cpr_width_pct(o1, h1, l1, c1), 4)
            rec["pdr_pct"] = round((h1 - l1) / c1 * 100.0, 4)
            rec["gap_pct"] = round((o - c1) / c1 * 100.0, 4)
            rec["gap_abs"] = round(abs((o - c1) / c1 * 100.0), 4)
            rec["ret_prev"] = round((c1 - bars[days[i - 2]][3]) / bars[days[i - 2]][3] * 100.0, 4) if i >= 2 else ""
        else:
            for k in ("cpr_today", "pdr_pct", "gap_pct", "gap_abs", "ret_prev"):
                rec[k] = ""
        if i >= 2:
            o2, h2, l2, c2 = bars[days[i - 2]]
            rec["cpr_prev"] = round(cpr_width_pct(o2, h2, l2, c2), 4)
        else:
            rec["cpr_prev"] = ""
        if len(ranges) >= 20 and rec["pdr_pct"] != "":
            avg = sum(ranges[-20:]) / 20.0
            rec["pdr_rel"] = round(rec["pdr_pct"] / avg, 4) if avg > 0 else ""
            rec["avg_range20"] = round(avg, 4)
        else:
            rec["pdr_rel"] = ""; rec["avg_range20"] = ""
        if len(trs) >= 14:
            atr = sum(trs[-14:]) / 14.0
            rec["atr14_pct"] = round(atr / cl * 100.0, 4)
        else:
            rec["atr14_pct"] = ""
        k = isoweek(d)
        p = wpos[k]
        if p >= 1:
            wo, wh, wl, wc = wk[wkeys[p - 1]]
            rec["wcpr_this"] = round(cpr_width_pct(wo, wh, wl, wc), 4)
        else:
            rec["wcpr_this"] = ""
        if p >= 2:
            wo, wh, wl, wc = wk[wkeys[p - 2]]
            rec["wcpr_prev"] = round(cpr_width_pct(wo, wh, wl, wc), 4)
        else:
            rec["wcpr_prev"] = ""
        vd = vix.get(d)
        prev_d = days[i - 1] if i >= 1 else None
        vp = vix.get(prev_d) if prev_d else None
        vp2 = vix.get(days[i - 2]) if i >= 2 else None
        rec["vix_open"] = round(vd[0], 3) if vd else ""
        rec["vix_prevclose"] = round(vp[3], 3) if vp else ""
        if vd and vp and vp[3] > 0:
            rec["vix_chg_oc_pct"] = round((vd[0] - vp[3]) / vp[3] * 100.0, 3)
            rec["vix_chg_oc_pts"] = round(vd[0] - vp[3], 3)
        else:
            rec["vix_chg_oc_pct"] = ""; rec["vix_chg_oc_pts"] = ""
        if vp and vp2 and vp2[3] > 0:
            rec["vix_chg_cc_pct"] = round((vp[3] - vp2[3]) / vp2[3] * 100.0, 3)
            rec["vix_chg_cc_pts"] = round(vp[3] - vp2[3], 3)
        else:
            rec["vix_chg_cc_pct"] = ""; rec["vix_chg_cc_pts"] = ""
        rows.append(rec)
        if i >= 1:
            o1, h1, l1, c1 = bars[days[i - 1]]
            ranges.append((h1 - l1) / c1 * 100.0)
            pc = bars[days[i - 2]][3] if i >= 2 else c1
            trs.append(max(h1 - l1, abs(h1 - pc), abs(l1 - pc)))
    return rows


FIELDS = ["day", "dow", "open", "high", "low", "close", "cpr_today", "cpr_prev",
          "wcpr_this", "wcpr_prev", "gap_pct", "gap_abs", "pdr_pct", "pdr_rel",
          "avg_range20", "atr14_pct", "ret_prev", "vix_open", "vix_prevclose",
          "vix_chg_oc_pct", "vix_chg_oc_pts", "vix_chg_cc_pct", "vix_chg_cc_pts"]


def main():
    os.makedirs(RES, exist_ok=True)
    c = con()
    vix = daily_from_daily(c, "INDIAVIX")
    print("INDIAVIX daily rows:", len(vix), min(vix), max(vix), flush=True)
    for sym, how in (("NIFTY50", ("day", None)), ("SENSEX", ("intraday", "minute"))):
        if how[0] == "day":
            bars = daily_from_daily(c, sym)
        else:
            bars = daily_from_intraday(c, sym, how[1])
        print(sym, "daily bars:", len(bars), min(bars), max(bars), flush=True)
        rows = build(sym, bars, vix)
        p = os.path.join(RES, "daily_features_%s.csv" % sym)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
            for r in rows:
                w.writerow(r)
        nv = sum(1 for r in rows if r["vix_open"] != "")
        print("  wrote", p, len(rows), "rows;", nv, "with VIX", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
