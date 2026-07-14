#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/78 P2 — naked-leg trail WIDTH: how wide should the ATR band be?

P1 found the trail is monotonic in "how rarely it fires": the less it triggers, the more you earn
(K1 +4,984 -> K8 +6,537 -> no-trail +9,507 per episode), while EVERY trail width shared the same
worst case (-9,750) vs no-trail's -20,280. So the stop's whole value is the tail cap, and its whole
cost is firing too often.

That points at WIDTH, not confirmation. This sweeps the ATR multiplier (currently 3) and asks:
is there a width that keeps the tail capped near -10k while capturing more of the +9,507 upside?

Fixed: K=3 confirmation (as deployed), ATR period 7, ratchet-down, seeded 09:15 today.
Varies:  multiplier in 2,3,4,5,6,8,10,  plus a no-trail baseline.

Reads only. No orders, no live state.
"""
from __future__ import annotations

import csv
import os
import sqlite3
import sys

import numpy as np

ROOT = "/home/arun/quantifyd"
ODB = os.path.join(ROOT, "backtest_data", "options_data.db")
OUT = os.path.join(ROOT, "research", "78_naked_trail_exit_confirm", "results")

BOOKS = [("nas_atm", "SQ-ATM"), ("nas_atm4", "SQ-ATM4"),
         ("nas_916_atm", "916-ATM"), ("nas_916_atm4", "916-ATM4")]
MULTS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
CONFIRM = 3          # K=3, as deployed
PERIOD = 7


def tsec(s):
    return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def stop_series(candles, mult, period=PERIOD):
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
        stop[i] = min(upper[i], stop[i - 1])     # pure ratchet, no re-arm
        out[i] = stop[i]
    return out


def bars_upto(prints, upto):
    bars = {}
    for t, p in prints:
        b = (t // 300) * 300
        if b + 300 > upto:
            continue
        if b not in bars:
            bars[b] = {"open": p, "high": p, "low": p, "close": p}
        else:
            d = bars[b]
            d["high"] = max(d["high"], p); d["low"] = min(d["low"], p); d["close"] = p
    return [bars[k] for k in sorted(bars)]


def simulate(prints, t0, real_exit_px):
    """Return {mult: (exit_px, fired)}."""
    res = {}
    for m in MULTS:
        streak = 0
        px = None
        for t, p in prints:
            if t < t0:
                continue
            b = bars_upto(prints, t)
            st = stop_series(b, m)
            if not len(st) or np.isnan(st[-1]):
                continue
            s = st[-1]
            streak = streak + 1 if p > s else 0
            if streak >= CONFIRM:
                px = p
                break
        res[m] = (px if px is not None else real_exit_px, px is not None)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    oc = sqlite3.connect(f"file:{ODB}?mode=ro", uri=True)
    rows = []

    for db, lab in BOOKS:
        p = os.path.join(ROOT, "backtest_data", f"{db}_trading.db")
        if not os.path.exists(p):
            continue
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        legs = c.execute(
            "SELECT tradingsymbol, qty, entry_price, entry_time, exit_price, exit_time, mode "
            "FROM nas_atm_positions WHERE sl_price>900000 AND exit_time IS NOT NULL "
            "AND exit_price IS NOT NULL ORDER BY entry_time").fetchall()
        c.close()

        for L in legs:
            day = str(L["entry_time"])[:10]
            try:
                t0, t1 = tsec(str(L["entry_time"])), tsec(str(L["exit_time"]))
            except Exception:
                continue
            if t1 <= t0:
                continue
            pr = oc.execute(
                "SELECT snapshot_time, ltp FROM option_chain WHERE tradingsymbol=? "
                "AND date(snapshot_time)=? AND ltp>0 ORDER BY snapshot_time",
                (L["tradingsymbol"], day)).fetchall()
            prints = [(tsec(r[0]), float(r[1])) for r in pr]
            prints = [(t, p) for t, p in prints if t0 <= t <= t1]
            if len(prints) < 20:
                continue
            sim = simulate(prints, t0, float(L["exit_price"]))
            row = {"book": lab, "day": day, "tsym": L["tradingsymbol"], "mode": L["mode"],
                   "qty": L["qty"], "entry": L["entry_price"], "real_exit": L["exit_price"]}
            e, q = float(L["entry_price"]), int(L["qty"])
            for m in MULTS:
                px, fired = sim[m]
                row[f"pnl_m{m:g}"] = round((e - px) * q)
                row[f"fired_m{m:g}"] = int(fired)
            row["pnl_none"] = round((e - float(L["exit_price"])) * q)
            rows.append(row)
            print(f"  {lab:9s} {day} {L['tradingsymbol']:20s} " +
                  " ".join(f"m{m:g}={row['pnl_m%g' % m]:+6d}" for m in (3.0, 5.0, 8.0)) +
                  f" none={row['pnl_none']:+6d}")
            sys.stdout.flush()

    if not rows:
        print("NO EPISODES"); return

    with open(os.path.join(OUT, "width_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    n = len(rows)
    print(f"\n=== {n} naked-leg episodes | K={CONFIRM} confirmation, ATR({PERIOD}) ===\n")
    print(f"{'mult':>6s} {'mean':>9s} {'median':>9s} {'total':>10s} {'win%':>6s} {'worst':>9s} "
          f"{'p10':>9s} {'fired%':>7s}")
    for m in MULTS:
        v = np.array([r[f"pnl_m{m:g}"] for r in rows], float)
        fr = np.array([r[f"fired_m{m:g}"] for r in rows], float)
        print(f"{m:>6.0f} {v.mean():+9.0f} {np.median(v):+9.0f} {v.sum():+10.0f} "
              f"{100*(v>0).mean():5.0f}% {v.min():+9.0f} {np.percentile(v,10):+9.0f} {100*fr.mean():6.0f}%")
    v = np.array([r["pnl_none"] for r in rows], float)
    print(f"{'none':>6s} {v.mean():+9.0f} {np.median(v):+9.0f} {v.sum():+10.0f} "
          f"{100*(v>0).mean():5.0f}% {v.min():+9.0f} {np.percentile(v,10):+9.0f} {0:6.0f}%")

    print("\n--- the trade-off: what each width buys vs no-trail ---")
    base = np.array([r["pnl_none"] for r in rows], float)
    print(f"{'mult':>6s} {'mean cost':>11s} {'tail saved':>12s} {'cost per 1 of tail saved':>26s}")
    for m in MULTS:
        v = np.array([r[f"pnl_m{m:g}"] for r in rows], float)
        cost = base.mean() - v.mean()            # what the stop gives up on the mean
        tail = v.min() - base.min()              # how much better the worst case is
        ratio = (cost / tail) if tail > 0 else float("nan")
        print(f"{m:>6.0f} {cost:>11.0f} {tail:>12.0f} {ratio:>26.2f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
