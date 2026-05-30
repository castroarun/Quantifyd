"""Prev-day range breakout — Phase 3: emit a per-trade log with CAUSAL beta.

Emits every LONG breakout that passes the HTF-trend filter and is an NR7 and/or
PCT25 compression day, with the stock's CAUSAL trailing-252d beta at entry (no
look-ahead). Results for 2R and 3R targets (net of cost) + entry/exit dates and
hold days. Beta is stored (not pre-filtered) so the analyzer can sweep the cutoff.

Output: results/trades_long.csv  (one row per qualifying long breakout)
"""
from __future__ import annotations

import csv
import sqlite3
import sys
import time
import warnings
import logging
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from nrb import simulate, sl_price  # noqa: E402
from run_nrb import (load_daily, load_5min_by_day, compression_flags,  # noqa: E402
                     cohort_of, all_symbols, COST_BPS, BREAK_MIN)
from run_nrb_p2 import _sma, cpr_width_series, CPR_PCTL  # noqa: E402


def build_market_returns(con):
    """Synthetic equal-weight market index = MEDIAN daily return across all stocks
    per date (NIFTY50 daily only starts 2023; daily stocks go back to 2016).
    Median is robust to penny-stock outliers."""
    import statistics as st
    rows = con.execute("SELECT symbol,date,close FROM market_data_unified "
                       "WHERE timeframe='day' ORDER BY symbol,date").fetchall()
    by_date = {}
    prev_sym = None
    prev_close = None
    for sym, dt, c in rows:
        d = dt[:10]
        if sym == prev_sym and prev_close and prev_close > 0 and c > 0:
            by_date.setdefault(d, []).append(c / prev_close - 1.0)
        prev_sym, prev_close = sym, c
    return {d: st.median(v) for d, v in by_date.items() if len(v) >= 20}

DB = ROOT / "backtest_data" / "market_data.db"
RESULTS = HERE.parent / "results"
TRADES_CSV = RESULTS / "trades_long.csv"
BETA_W = 252
BETA_MINP = 150
FIELDS = ["entry_date", "symbol", "year", "beta", "nr7", "pct25", "cprok",
          "entry", "sl", "R_price",
          "result_2R", "hold_2R", "exit_2R",
          "result_3R", "hold_3R", "exit_3R"]


def causal_beta(D, nifty_ret):
    """Trailing-252d beta usable on each daily index (uses returns up to i-1)."""
    c = D["c"]
    n = len(c)
    sret = np.full(n, np.nan)
    sret[1:] = c[1:] / c[:-1] - 1.0
    nret = np.array([nifty_ret.get(d, np.nan) for d in D["dates"]])
    beta = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - BETA_W)
        x = nret[lo:i]
        y = sret[lo:i]
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() >= BETA_MINP:
            x, y = x[m], y[m]
            v = np.var(x)
            if v > 0:
                beta[i] = np.cov(y, x)[0, 1] / v
    return beta


def run_symbol(con, symbol, nifty_ret, writer):
    D = load_daily(con, symbol)
    if D is None:
        return 0
    by_day = load_5min_by_day(con, symbol)
    if not by_day:
        return 0
    sma50 = _sma(D["c"], 50)
    cprw = cpr_width_series(D)
    beta = causal_beta(D, nifty_ret)
    dates = D["dates"]
    n = 0
    for d, bars in by_day.items():
        i = D["idx"].get(d)
        if i is None or i < 1:
            continue
        pdh, pdl = D["h"][i - 1], D["l"][i - 1]
        prevrange = pdh - pdl
        datr = D["atr"][i - 1]
        if prevrange <= 0 or np.isnan(datr):
            continue
        # long breakout only: first 5-min close above prev-day high
        entry = ebar_idx = None
        for k, (t, o, h, l, c) in enumerate(bars):
            if t < BREAK_MIN:
                continue
            if c > pdh:
                entry, ebar_idx = c, k
                break
            if c < pdl:           # broke down first -> not a long setup today
                break
        if entry is None:
            continue
        # HTF filter (causal: prev close vs prev SMA50)
        if np.isnan(sma50[i - 1]) or D["c"][i - 1] <= sma50[i - 1]:
            continue
        flags = compression_flags(D, i)
        if not (flags["NR7"] or flags["PCT25"]):
            continue
        b = beta[i]
        if np.isnan(b):
            continue
        cprok = False
        if not np.isnan(cprw[i - 1]):
            lo = max(0, i - 61)
            hist = cprw[lo:i - 1]
            hist = hist[~np.isnan(hist)]
            if len(hist) >= 20:
                cprok = bool(cprw[i - 1] <= np.percentile(hist, CPR_PCTL))

        SL = sl_price("BOX", 1, entry, pdl, pdh, prevrange, datr, 0, 0)
        R = abs(entry - SL)
        if R <= 0:
            continue
        cost_R = (COST_BPS / 1e4 * abs(entry)) / R
        post = bars[ebar_idx + 1:]
        d5h = np.array([x[2] for x in post], float)
        d5l = np.array([x[3] for x in post], float)
        d5c = np.array([x[4] for x in post], float)
        daily_tuples = list(zip(D["o"], D["h"], D["l"], D["c"]))
        daily_after = daily_tuples[i + 1:]

        out2 = simulate(1, entry, SL, "2R", prevrange, d5h, d5l, d5c, daily_after, "SWING")
        out3 = simulate(1, entry, SL, "3R", prevrange, d5h, d5l, d5c, daily_after, "SWING")
        if out2 is None or out3 is None:
            continue
        r2, h2 = out2
        r3, h3 = out3
        ex2 = dates[min(i + h2, len(dates) - 1)]
        ex3 = dates[min(i + h3, len(dates) - 1)]
        writer.writerow(dict(
            entry_date=d, symbol=symbol, year=int(d[:4]), beta=round(b, 3),
            nr7=int(flags["NR7"]), pct25=int(flags["PCT25"]), cprok=int(cprok),
            entry=round(entry, 2), sl=round(SL, 2), R_price=round(R, 2),
            result_2R=round(r2 - cost_R, 4), hold_2R=h2, exit_2R=ex2,
            result_3R=round(r3 - cost_R, 4), hold_3R=h3, exit_3R=ex3))
        n += 1
    return n


def main():
    con = sqlite3.connect(DB)
    print("building synthetic market index (median daily return, all stocks)...")
    nifty_ret = build_market_returns(con)
    print(f"  market index: {len(nifty_ret)} dates "
          f"({min(nifty_ret)} -> {max(nifty_ret)})")
    syms = [s for s in all_symbols(con) if cohort_of(s) == "stocks"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total = 0
    with open(TRADES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k, sym in enumerate(syms, 1):
            c = run_symbol(con, sym, nifty_ret, w)
            total += c
            if k % 50 == 0 or k == len(syms):
                print(f"[{k}/{len(syms)}] {sym:12s} +{c:4d}  total={total:6d} "
                      f"elapsed={time.time()-t0:5.0f}s", flush=True)
    print(f"done: {total} long trades -> {TRADES_CSV.name} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
