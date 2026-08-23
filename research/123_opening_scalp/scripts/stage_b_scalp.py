#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/123 Stage B - the opening-scalp tail on the LONG price sample, plus the
pre-registered filter features (CPR widths, gap) for every day.

Windows: entry {09:16 both venues, 09:30 + 10:00 NIFTY, 10:30 SENSEX} x
T = 20..65 step 5: maximum absolute excursion from the entry price and the
|terminal| move, in bp of the entry price, inside (entry, entry+T].

Data (market_data.db :: market_data_unified, READ-ONLY):
  SENSEX 'minute' 2021-01-01 ->   (1-min truth; 09:16 entry exact to the minute)
  NIFTY50 '5minute' 2015-01-01 -> (r/121 licence: window MAX EXCURSION is
    resolution-invariant; entry bar = first bar labelled >= start, so the NIFTY
    09:16 entry is really the 09:20 bar - a ~1-bar imprecision, stated in RESULTS)

DTE labels via the r/122 expiry-era tables (holiday shifts ignored, +-1 label
noise on holiday weeks). Filter features per day, all knowable at entry:
  gap_bp        (open_d - close_{d-1}) / close_{d-1} * 1e4  (signed)
  cpr_t_bp      today's CPR width (from day d-1 OHLC), bp of close_{d-1}
  cpr_y_bp      yesterday's CPR width (from day d-2 OHLC)
  cpr_w_bp      this week's CPR width (from the previous ISO-week's aggregate OHLC)
CPR width = |TC - BC|, P=(H+L+C)/3, BC=(H+L)/2, TC=2P-BC.
Daily OHLC built from the intraday series itself (session bars 09:15-15:30 only)
- immune to the short/derived daily-series traps of r/121.

Writes results/stage_b_scalp_days.csv
"""
import sqlite3, csv, os
from datetime import date

Q = "/home/arun/quantifyd/"
MD = Q + "backtest_data/market_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(RES, "stage_b.log")

T_GRID = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
ENTRIES = {"NIFTY": [556, 570, 600], "SENSEX": [556, 630]}
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
    return None


def load(sym, tf, dfrom):
    c = sqlite3.connect("file:%s?mode=ro" % MD, uri=True)
    days = {}
    for dt, op, cl, hi, lo in c.execute(
            "SELECT date, open, close, high, low FROM market_data_unified "
            "WHERE symbol=? AND timeframe=? AND date>=? ORDER BY date", (sym, tf, dfrom)):
        d = dt[:10]
        mi = int(dt[11:13]) * 60 + int(dt[14:16])
        if mi < 9 * 60 + 15 or mi > 15 * 60 + 30:   # session bars only
            continue
        days.setdefault(d, []).append((mi, op, cl, hi, lo))
    c.close()
    return {d: sorted(v) for d, v in days.items()}


def cpr_width_bp(H, L, C, ref):
    P = (H + L + C) / 3.0
    BC = (H + L) / 2.0
    TC = 2.0 * P - BC
    return 1e4 * abs(TC - BC) / ref


def features(days_sorted, daily):
    """per-day dict of gap/cpr features; daily = d -> (O,H,L,C)."""
    feats = {}
    # weekly aggregates: iso (year, week) -> (H,L,C of the week)
    weeks = {}
    for d in days_sorted:
        O, H, L, C = daily[d]
        y, wnum, _ = date.fromisoformat(d).isocalendar()
        k = (y, wnum)
        if k not in weeks:
            weeks[k] = [H, L, C]
        else:
            weeks[k][0] = max(weeks[k][0], H)
            weeks[k][1] = min(weeks[k][1], L)
            weeks[k][2] = C
    wkeys = sorted(weeks)
    prev_week = {}
    for i, k in enumerate(wkeys):
        prev_week[k] = weeks[wkeys[i - 1]] if i > 0 else None
    for i, d in enumerate(days_sorted):
        if i < 2:
            continue
        d1, d2 = days_sorted[i - 1], days_sorted[i - 2]
        O, H, L, C = daily[d]
        O1, H1, L1, C1 = daily[d1]
        O2, H2, L2, C2 = daily[d2]
        y, wnum, _ = date.fromisoformat(d).isocalendar()
        pw = prev_week.get((y, wnum))
        feats[d] = dict(
            gap_bp=round(1e4 * (O - C1) / C1, 1),
            cpr_t_bp=round(cpr_width_bp(H1, L1, C1, C1), 1),
            cpr_y_bp=round(cpr_width_bp(H2, L2, C2, C2), 1),
            cpr_w_bp=round(cpr_width_bp(pw[0], pw[1], pw[2], pw[2]), 1) if pw else "")
    return feats


def run(sym, series, tf, dfrom, w):
    WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    days = load(sym if sym != "NIFTY" else "NIFTY50", tf, dfrom)
    ds = sorted(d for d in days if date.fromisoformat(d).weekday() < 5)
    daily = {}
    for d in ds:
        bars = days[d]
        daily[d] = (bars[0][1], max(b[3] for b in bars), min(b[4] for b in bars), bars[-1][2])
    feats = features(ds, daily)
    n = 0
    for d in ds:
        wd = date.fromisoformat(d).weekday()
        bars = days[d]
        mins = [b[0] for b in bars]
        if len(mins) < 40:
            continue
        px = {b[0]: b[2] for b in bars}
        hi = {b[0]: b[3] for b in bars}
        lo = {b[0]: b[4] for b in bars}
        dte = dte_label(sym, d, wd)
        ft = feats.get(d, {})
        n += 1
        for s in ENTRIES[sym]:
            m0 = next((m for m in mins if m >= s), None)
            if m0 is None or m0 > s + 10:
                continue
            p0 = px[m0]
            if not p0:
                continue
            for T in T_GRID:
                m1 = m0 + T
                exc = 0.0
                term = None
                for m in mins:
                    if m <= m0 or m > m1:
                        continue
                    e = max(abs(hi[m] - p0), abs(lo[m] - p0))
                    if e > exc:
                        exc = e
                    term = px[m] - p0
                if term is None:
                    continue
                w.writerow(dict(series=series, venue=sym, day=d, weekday=WD[wd],
                                dte_trd=("" if dte is None else dte),
                                entry=m2hm(s), T=T,
                                exc_bp=round(1e4 * exc / p0, 1),
                                term_bp=round(1e4 * abs(term) / p0, 1),
                                gap_bp=ft.get("gap_bp", ""),
                                cpr_t_bp=ft.get("cpr_t_bp", ""),
                                cpr_y_bp=ft.get("cpr_y_bp", ""),
                                cpr_w_bp=ft.get("cpr_w_bp", "")))
    return n


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    f = open(os.path.join(RES, "stage_b_scalp_days.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=["series", "venue", "day", "weekday", "dte_trd",
                                      "entry", "T", "exc_bp", "term_bp",
                                      "gap_bp", "cpr_t_bp", "cpr_y_bp", "cpr_w_bp"])
    w.writeheader()
    log("SENSEX 1-min ...")
    n = run("SENSEX", "SENSEX_1min", "minute", "2021-01-01", w)
    log("  days=%d" % n)
    f.flush()
    log("NIFTY50 5-min ...")
    n = run("NIFTY", "NIFTY50_5min", "5minute", "2015-01-01", w)
    log("  days=%d" % n)
    f.close()
    log("DONE")


if __name__ == "__main__":
    main()
