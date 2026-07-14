#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/78 — naked-leg trailing stop: exit on the FIRST breach, or confirm it?

Replays every naked-leg episode against its REAL premium path from the chain recorder and asks
what each exit rule would have made. Places no orders, touches no live state.

Rules: K=1 (instant), K=2/3/5/8 (breach must hold K consecutive prints, resets on any print back
below), close5 (first 5-min candle close above the stop), none (no trail -> ride to real exit).

Honest limit: the recorder's prints are ~4s median apart, not ticks (~1s). So K=3 here is ~12s of
confirmation vs ~3s live -- a SLOWER, more conservative test than what is deployed.
"""
from __future__ import annotations

import csv
import os
import sqlite3
import sys
from datetime import datetime

import numpy as np

ROOT = "/home/arun/quantifyd"
ODB = os.path.join(ROOT, "backtest_data", "options_data.db")
OUT = os.path.join(ROOT, "research", "78_naked_trail_exit_confirm", "results")

BOOKS = [
    ("nas_atm", "SQ-ATM"), ("nas_atm4", "SQ-ATM4"),
    ("nas_916_atm", "916-ATM"), ("nas_916_atm4", "916-ATM4"),
]
KS = [1, 2, 3, 5, 8]
RULES = [f"K{k}" for k in KS] + ["close5", "none"]
SLIPS = [0.0, 0.0015, 0.0030]


def tsec(s: str) -> int:
    return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def trailing_stop_series(candles, period=7, mult=3.0):
    """Stop for a SHORT premium: upper band, ratchets DOWN. Returns array aligned to candles
    (NaN until it arms). Same maths as services/nas_atm4_executor.compute_short_trailing_stop."""
    n = len(candles)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    high = np.array([c["high"] for c in candles], float)
    low = np.array([c["low"] for c in candles], float)
    close = np.array([c["close"] for c in candles], float)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    hl2 = (high + low) / 2.0
    upper = hl2 + mult * atr
    stop = np.zeros(n)
    stop[period - 1] = upper[period - 1]
    out[period - 1] = stop[period - 1]
    for i in range(period, n):
        # PURE ratchet -- no re-arm. The live function re-arms the stop above a breakout
        # (stop = upper[i]) which is harmless live because the breach closes the position, but in
        # a simulation it hides the breach: the stop leaps above the price, so "close > stop"
        # can never be true and a breach streak can never accumulate. That silently made K8 and
        # close5 identical to no-trail-at-all. Here the stop only ever ratchets DOWN.
        stop[i] = min(upper[i], stop[i - 1])
        out[i] = stop[i]
    return out


def load_prints(oc, tsym, day, t0, t1):
    """Real chain prints (snapshot_time, ltp) inside the leg's live window."""
    rows = oc.execute(
        "SELECT snapshot_time, ltp FROM option_chain WHERE tradingsymbol=? "
        "AND date(snapshot_time)=? AND ltp>0 ORDER BY snapshot_time", (tsym, day)).fetchall()
    out = [(tsec(r[0]), float(r[1])) for r in rows]
    return [(t, p) for t, p in out if t0 <= t <= t1]


def five_min_candles(prints, upto):
    """Build 5-min OHLC from the prints, from 09:15, for bars fully closed by `upto`."""
    bars = {}
    for t, p in prints:
        b = (t // 300) * 300
        if b + 300 > upto:      # bar not yet closed
            continue
        if b not in bars:
            bars[b] = {"open": p, "high": p, "low": p, "close": p}
        else:
            d = bars[b]
            d["high"] = max(d["high"], p)
            d["low"] = min(d["low"], p)
            d["close"] = p
    return [bars[k] for k in sorted(bars)], sorted(bars)


def simulate(day_prints, naked_from, real_exit_t, real_exit_px, entry, qty):
    """Walk the leg forward; return {rule: exit_price}."""
    res = {}
    # the stop is recomputed as bars close, exactly as live
    streak = 0
    fired = {r: None for r in RULES}
    for t, p in day_prints:
        if t < naked_from:
            continue
        bars, bar_ts = five_min_candles(day_prints, t)
        stops = trailing_stop_series(bars)
        stop = stops[-1] if len(stops) and not np.isnan(stops[-1]) else None
        if stop is None:
            continue
        # tick/print rules
        if p > stop:
            streak += 1
        else:
            streak = 0
        for k in KS:
            r = f"K{k}"
            if fired[r] is None and streak >= k:
                fired[r] = p
        # 5-min close rule: last CLOSED bar's close above its stop
        if fired["close5"] is None and len(bars) >= 8 and bars[-1]["close"] > stops[-1]:
            fired["close5"] = p
        if all(fired[r] is not None for r in RULES if r != "none"):
            break
    for r in RULES:
        if r == "none":
            res[r] = real_exit_px          # no trail -> whatever actually happened (EOD etc.)
        else:
            res[r] = fired[r] if fired[r] is not None else real_exit_px
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    oc = sqlite3.connect(f"file:{ODB}?mode=ro", uri=True)
    rows_out = []

    for db, lab in BOOKS:
        p = os.path.join(ROOT, "backtest_data", f"{db}_trading.db")
        if not os.path.exists(p):
            continue
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        legs = c.execute(
            "SELECT tradingsymbol, qty, entry_price, entry_time, exit_price, exit_time, exit_reason, mode "
            "FROM nas_atm_positions WHERE sl_price>900000 AND exit_time IS NOT NULL "
            "AND exit_price IS NOT NULL ORDER BY entry_time").fetchall()
        c.close()

        for L in legs:
            day = str(L["entry_time"])[:10]
            try:
                t_entry = tsec(str(L["entry_time"]))
                t_exit = tsec(str(L["exit_time"]))
            except Exception:
                continue
            if t_exit <= t_entry:
                continue
            prints = load_prints(oc, L["tradingsymbol"], day, 0, t_exit)
            if len(prints) < 20:
                continue
            # the leg is only NAKED from when its sibling died; we do not know that instant, so we
            # arm from the leg's own entry -- the stop needs 8 bars anyway (~09:55), which is after
            # essentially every observed naked-ing.
            sim = simulate(prints, t_entry, t_exit, float(L["exit_price"]),
                           float(L["entry_price"]), int(L["qty"]))
            row = {
                "book": lab, "day": day, "tsym": L["tradingsymbol"], "mode": L["mode"],
                "qty": L["qty"], "entry": L["entry_price"],
                "real_exit": L["exit_price"], "real_reason": L["exit_reason"],
            }
            for r in RULES:
                row[f"px_{r}"] = round(sim[r], 2)
                row[f"pnl_{r}"] = round((float(L["entry_price"]) - sim[r]) * int(L["qty"]))
            rows_out.append(row)
            print(f"  {lab:9s} {day} {L['tradingsymbol']:20s} " +
                  " ".join(f"{r}={row['pnl_'+r]:+6d}" for r in ("K1", "K3", "close5", "none")))
            sys.stdout.flush()

    if not rows_out:
        print("NO EPISODES")
        return

    fn = os.path.join(OUT, "episodes.csv")
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    n = len(rows_out)
    print(f"\n=== {n} naked-leg episodes ===\n")
    print(f"{'rule':8s} {'mean':>9s} {'median':>9s} {'total':>10s} {'win%':>6s} {'worst':>9s} {'p10':>9s}")
    summary = {}
    for r in RULES:
        v = np.array([x[f"pnl_{r}"] for x in rows_out], float)
        summary[r] = v
        print(f"{r:8s} {v.mean():+9.0f} {np.median(v):+9.0f} {v.sum():+10.0f} "
              f"{100*(v>0).mean():5.0f}% {v.min():+9.0f} {np.percentile(v,10):+9.0f}")

    print("\n--- vs K1 (instant) ---")
    base = summary["K1"]
    for r in RULES:
        if r == "K1":
            continue
        d = summary[r] - base
        better = int((d > 0).sum())
        worse = int((d < 0).sum())
        print(f"{r:8s} mean diff {d.mean():+8.0f} | better {better:3d} / worse {worse:3d} / same {n-better-worse:3d}")

    print("\n--- cost sensitivity (mean P&L per episode) ---")
    print(f"{'rule':8s}" + "".join(f"{int(s*10000):>10d}bp" for s in SLIPS))
    for r in RULES:
        line = f"{r:8s}"
        for s in SLIPS:
            px = np.array([x[f"px_{r}"] for x in rows_out], float)
            q = np.array([x["qty"] for x in rows_out], float)
            e = np.array([x["entry"] for x in rows_out], float)
            line += f"{((e - px*(1+s))*q).mean():>12.0f}"
        print(line)
    print("\nDONE")


if __name__ == "__main__":
    main()
