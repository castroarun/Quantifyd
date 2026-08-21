#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/120 Stage B - the Friday volatility clock on the UNDERLYING, long history.

Stage A has only 16 Fridays of option truth. This stage asks the risk half of the question
- "least possibility of volatile moves" - on a sample big enough to mean something, using
the index itself.

Data reality (probed 2026-08-21):
  * market_data_unified has SENSEX **1-minute** 2021-01-01..2026-08-20 (508,378 rows)  -> primary
  * it has NO NIFTY 1-minute at all; NIFTY intraday exists only as NIFTY50 **5-minute**
    2015-02-02..2026-07-17                                                             -> secondary
  * a SENSEX 5-minute resample is also computed so the 1-min-vs-5-min understatement of
    excursions can be quantified rather than assumed (project rule: never trust 5-min for
    intraday extremes).

Outputs (all in results/):
  volclock_buckets.csv   per 15-min bucket x weekday x series: mean |1-bar move| bp, bar vol
  volclock_windows.csv   per (start,duration) x series x Friday/other: excursion distribution
READ-ONLY.
"""
import sqlite3, csv, os, math, statistics as st
from datetime import date

Q = "/home/arun/quantifyd/"
MD = Q + "backtest_data/market_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(RES, "stage_b.log")

SESS_START = 9 * 60 + 20
SESS_END = 15 * 60 + 20
GRID_STARTS = []
_m = SESS_START
while _m <= 14 * 60 + 30:
    GRID_STARTS.append(_m)
    _m += 15
if 14 * 60 + 30 not in GRID_STARTS:
    GRID_STARTS.append(14 * 60 + 30)
DURATIONS = [45, 60, 90, 120, "HOLD"]


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def m2hm(m):
    return "%02d:%02d" % (m // 60, m % 60)


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


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


def resample5(bars):
    """1-min bars -> 5-min bars (bucket by floor to 5)."""
    out = {}
    for mi, cl, hi, lo in bars:
        b = (mi // 5) * 5
        e = out.get(b)
        if e is None:
            out[b] = [cl, hi, lo]
        else:
            e[0] = cl
            e[1] = max(e[1], hi)
            e[2] = min(e[2], lo)
    return [(b, v[0], v[1], v[2]) for b, v in sorted(out.items())]


def bucket_stats(days, label, wbuck, series_step):
    """mean |bar-to-bar move| bp and bar vol, per 15-min bucket per weekday."""
    acc = {}
    for d, bars in days.items():
        wd = date.fromisoformat(d).weekday()
        if wd > 4:
            continue
        bars = sorted(bars)
        for i in range(1, len(bars)):
            mi, cl = bars[i][0], bars[i][1]
            p0 = bars[i - 1][1]
            if mi < SESS_START or mi > SESS_END or p0 <= 0:
                continue
            if bars[i][0] - bars[i - 1][0] > series_step * 3:
                continue
            r = 1e4 * (cl - p0) / p0
            b = (mi // 15) * 15
            acc.setdefault((wd, b), []).append(r)
    for (wd, b), v in sorted(acc.items()):
        wbuck.writerow(dict(series=label, weekday=["Mon", "Tue", "Wed", "Thu", "Fri"][wd],
                            bucket=m2hm(b), n_bars=len(v),
                            mean_abs_move_bp=round(sum(abs(x) for x in v) / len(v), 2),
                            bar_vol_bp=round(st.pstdev(v), 2),
                            p95_abs_bp=round(pct([abs(x) for x in v], 0.95), 2)))


def window_stats(days, label, wwin, only_fri):
    """For every (start,duration): distribution of max |excursion from start| in bp."""
    acc = {}
    ndays = 0
    for d, bars in days.items():
        wd = date.fromisoformat(d).weekday()
        if wd > 4:
            continue
        isfri = (wd == 4)
        if only_fri != isfri:
            continue
        bars = sorted(bars)
        px = {mi: cl for mi, cl, _, _ in bars}
        hi = {mi: h for mi, _, h, _ in bars}
        lo = {mi: l for mi, _, _, l in bars}
        mins = [m for m in px if SESS_START <= m <= SESS_END]
        if len(mins) < 40:
            continue
        ndays += 1
        mins_sorted = sorted(mins)
        for s in GRID_STARTS:
            cand = [m for m in mins_sorted if m >= s]
            if not cand:
                continue
            m0 = cand[0]
            if m0 > s + 10:
                continue
            p0 = px[m0]
            for dur in DURATIONS:
                m1 = SESS_END if dur == "HOLD" else min(m0 + dur, SESS_END)
                if m1 <= m0:
                    continue
                exc = 0.0
                sgn = 0.0
                for m in mins_sorted:
                    if m <= m0 or m > m1:
                        continue
                    e = max(abs(hi[m] - p0), abs(lo[m] - p0))
                    if e > exc:
                        exc = e
                    sgn = px[m] - p0
                acc.setdefault((s, dur), []).append((1e4 * exc / p0, 1e4 * abs(sgn) / p0))
    for (s, dur), v in sorted(acc.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        ex = [a for a, _ in v]
        tm = [b for _, b in v]
        wwin.writerow(dict(series=label, scope="FRI" if only_fri else "MON-THU",
                           start=m2hm(s), dur=dur, n_days=len(ex),
                           mean_exc_bp=round(sum(ex) / len(ex), 1),
                           med_exc_bp=round(pct(ex, 0.5), 1),
                           p90_exc_bp=round(pct(ex, 0.90), 1),
                           p95_exc_bp=round(pct(ex, 0.95), 1),
                           max_exc_bp=round(max(ex), 1),
                           pct_gt_30bp=round(100.0 * sum(1 for x in ex if x > 30) / len(ex), 1),
                           pct_gt_50bp=round(100.0 * sum(1 for x in ex if x > 50) / len(ex), 1),
                           mean_terminal_bp=round(sum(tm) / len(tm), 1)))
    return ndays


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    fb = open(os.path.join(RES, "volclock_buckets.csv"), "w", newline="")
    wb = csv.DictWriter(fb, fieldnames=["series", "weekday", "bucket", "n_bars",
                                        "mean_abs_move_bp", "bar_vol_bp", "p95_abs_bp"])
    wb.writeheader()
    fw = open(os.path.join(RES, "volclock_windows.csv"), "w", newline="")
    ww = csv.DictWriter(fw, fieldnames=["series", "scope", "start", "dur", "n_days",
                                        "mean_exc_bp", "med_exc_bp", "p90_exc_bp", "p95_exc_bp",
                                        "max_exc_bp", "pct_gt_30bp", "pct_gt_50bp",
                                        "mean_terminal_bp"])
    ww.writeheader()

    log("loading SENSEX 1-minute 2021-01-01 ->")
    sx = load("SENSEX", "minute", "2021-01-01")
    log("  SENSEX days=%d  fridays=%d" % (len(sx), sum(1 for d in sx if date.fromisoformat(d).weekday() == 4)))
    bucket_stats(sx, "SENSEX_1min", wb, 1)
    n = window_stats(sx, "SENSEX_1min", ww, True)
    log("  SENSEX 1-min Friday windows built (%d Fridays)" % n)
    window_stats(sx, "SENSEX_1min", ww, False)
    fb.flush(); fw.flush()

    log("resampling SENSEX to 5-minute (resolution control)")
    sx5 = {d: resample5(b) for d, b in sx.items()}
    bucket_stats(sx5, "SENSEX_5min", wb, 5)
    window_stats(sx5, "SENSEX_5min", ww, True)
    fb.flush(); fw.flush()
    del sx, sx5

    log("loading NIFTY50 5-minute 2015-02-02 ->")
    nf = load("NIFTY50", "5minute", "2015-01-01")
    log("  NIFTY50 days=%d fridays=%d" % (len(nf), sum(1 for d in nf if date.fromisoformat(d).weekday() == 4)))
    bucket_stats(nf, "NIFTY50_5min", wb, 5)
    n = window_stats(nf, "NIFTY50_5min", ww, True)
    log("  NIFTY50 5-min Friday windows built (%d Fridays)" % n)
    window_stats(nf, "NIFTY50_5min", ww, False)

    # modern sub-period for NIFTY, matched to the SENSEX window
    nf2 = {d: b for d, b in nf.items() if d >= "2021-01-01"}
    window_stats(nf2, "NIFTY50_5min_2021+", ww, True)
    bucket_stats(nf2, "NIFTY50_5min_2021+", wb, 5)

    fb.close(); fw.close()
    log("DONE")


if __name__ == "__main__":
    main()
