#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/118 Stage B — 5.6 years of SENSEX 1-minute price action, one row per day.

Options do not exist in our data before 2026-04, so we measure the UNDERLYING behaviour
that decides a short straddle's fate. Everything here is causal and intrabar.

Per trading day (2021-01-01 .. 2026-08-20, symbol SENSEX, timeframe minute):
  * the 09:16 reference level (the live suite's entry minute) and the 15:15 close
  * terminal move from 09:16, signed and absolute
  * maximum adverse excursion from 09:16 (intrabar) and the minute it happened
  * the same pair measured from 13:00 (the afternoon-window construction)
  * intraday range, realised vol, overnight gap
  * a spike-vs-drift decomposition (largest 15-minute displacement vs the whole day)
  * a causal trailing-20-day realised-vol reading for regime bucketing

The 30 GB market_data.db is ALWAYS queried with symbol AND timeframe pinned, chunked
by year. READ-ONLY.
"""
import csv
import math
import os
import sqlite3
from datetime import date

Q = "/home/arun/quantifyd/"
MKT = "file:" + Q + "backtest_data/market_data.db?mode=ro"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "stage_b_daily_price_action.csv")
LOG = os.path.join(RES, "stage_b.log")

START, END = "2021-01-01", "2026-08-21"
ENTRY_HM, MID_HM, EOD_HM = "09:16", "13:00", "15:15"
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

FIELDS = [
    "day", "weekday", "year", "n_bars",
    "ref0916", "close_eod", "move_term", "abs_move_term",
    "mae_pts", "mae_hm", "mae_dir", "revert_ratio",
    "ref1300", "move_term13", "abs_move_term13", "mae13_pts", "mae13_hm",
    "day_open", "day_high", "day_low", "range_pts", "range_pct",
    "rvol_pct", "max15_pts", "spike_share", "gap_pct",
    "trail20_rvol_pct", "abs_move_pct",
]


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def load_days(m, y):
    """{day: [(hm, o,h,l,c), ...]} for one calendar year."""
    rows = m.execute(
        "SELECT date, open, high, low, close FROM market_data_unified "
        "WHERE symbol='SENSEX' AND timeframe='minute' AND date >= ? AND date < ? "
        "ORDER BY date", ("%d-01-01 00:00:00" % y, "%d-01-01 00:00:00" % (y + 1))).fetchall()
    d = {}
    for dt, o, h, l, c in rows:
        if None in (o, h, l, c):
            continue
        d.setdefault(dt[:10], []).append((dt[11:16], o, h, l, c))
    return d


def level_at(bars, hm):
    """Close of the first bar at or after hm (None if the session ended earlier)."""
    for b in bars:
        if b[0] >= hm:
            return b[4]
    return None


def excursion(bars, ref, from_hm, to_hm):
    mae, mae_hm, mae_dir = 0.0, "", ""
    for hm, o, h, l, c in bars:
        if hm < from_hm or hm > to_hm:
            continue
        up, dn = h - ref, ref - l
        if up > mae:
            mae, mae_hm, mae_dir = up, hm, "UP"
        if dn > mae:
            mae, mae_hm, mae_dir = dn, hm, "DOWN"
    return mae, mae_hm, mae_dir


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    m = sqlite3.connect(MKT, uri=True)
    rows_out = []
    prev_close = None
    rvol_hist = []
    for y in range(2021, 2027):
        by_day = load_days(m, y)
        log("%d: %d days, %d bars" % (y, len(by_day), sum(len(v) for v in by_day.values())))
        for day in sorted(by_day):
            if day < START or day > END:
                continue
            bars = [b for b in by_day[day] if "09:15" <= b[0] <= EOD_HM]
            if len(bars) < 200:
                log("  %s skip (only %d bars in session)" % (day, len(bars)))
                prev_close = by_day[day][-1][4] if by_day[day] else prev_close
                continue
            ref = level_at(bars, ENTRY_HM)
            eod = bars[-1][4]
            if ref is None or not ref:
                continue
            mae, mae_hm, mae_dir = excursion(bars, ref, ENTRY_HM, EOD_HM)
            move = eod - ref
            ref13 = level_at(bars, MID_HM)
            if ref13:
                mae13, mae13_hm, _ = excursion(bars, ref13, MID_HM, EOD_HM)
                move13 = eod - ref13
            else:
                mae13, mae13_hm, move13 = 0.0, "", 0.0
            hi = max(b[2] for b in bars)
            lo = min(b[3] for b in bars)
            op = bars[0][1]
            closes = [b[4] for b in bars]
            rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
                    if closes[i] > 0 and closes[i - 1] > 0]
            mu = sum(rets) / len(rets) if rets else 0.0
            var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1) if len(rets) > 2 else 0.0
            rvol = math.sqrt(var) * math.sqrt(len(rets)) * 100.0     # % move over the session
            # largest 15-minute displacement anywhere in the session
            max15 = 0.0
            for i in range(len(closes)):
                j = min(i + 15, len(closes) - 1)
                max15 = max(max15, abs(closes[j] - closes[i]))
            gap = ((op - prev_close) / prev_close * 100.0) if prev_close else 0.0
            trail = (sum(rvol_hist[-20:]) / len(rvol_hist[-20:])) if len(rvol_hist) >= 20 else ""
            rows_out.append({
                "day": day, "weekday": WD[date.fromisoformat(day).weekday()], "year": day[:4],
                "n_bars": len(bars),
                "ref0916": round(ref, 2), "close_eod": round(eod, 2),
                "move_term": round(move, 1), "abs_move_term": round(abs(move), 1),
                "mae_pts": round(mae, 1), "mae_hm": mae_hm, "mae_dir": mae_dir,
                "revert_ratio": round(abs(move) / mae, 3) if mae > 0 else 0.0,
                "ref1300": round(ref13, 2) if ref13 else "",
                "move_term13": round(move13, 1), "abs_move_term13": round(abs(move13), 1),
                "mae13_pts": round(mae13, 1), "mae13_hm": mae13_hm,
                "day_open": round(op, 2), "day_high": round(hi, 2), "day_low": round(lo, 2),
                "range_pts": round(hi - lo, 1), "range_pct": round((hi - lo) / ref * 100, 3),
                "rvol_pct": round(rvol, 3), "max15_pts": round(max15, 1),
                "spike_share": round(max15 / mae, 3) if mae > 0 else 0.0,
                "gap_pct": round(gap, 3),
                "trail20_rvol_pct": (round(trail, 3) if trail != "" else ""),
                "abs_move_pct": round(abs(move) / ref * 100, 3),
            })
            rvol_hist.append(rvol)
            prev_close = eod
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    log("wrote %s (%d days, %s .. %s)" % (OUT, len(rows_out), rows_out[0]["day"], rows_out[-1]["day"]))


if __name__ == "__main__":
    main()
