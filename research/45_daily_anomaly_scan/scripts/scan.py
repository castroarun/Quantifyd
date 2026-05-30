"""Daily anomaly scan — where does a real, tradeable edge live in OUR universe?

Cross-sectional probes on the liquid universe (the ~374 F&O/N500 names), daily
data 2000-2026. For each anomaly we rank stocks each day on a CAUSAL signal
(known at t), bucket into deciles, and measure the mean forward return per decile
+ the dollar-neutral long-short spread. Monotonic deciles + a clean LS spread =
a real, diversifiable, (often) market-neutral edge worth building.

Probes: short-term reversal, 12-1 momentum, overnight-vs-intraday drift,
turn-of-month seasonality, low-volatility, 52-week-high proximity.
"""
from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DB = ROOT / "backtest_data" / "market_data.db"
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
TRADING_DAYS = 252


def liquid_universe(con):
    rows = con.execute("SELECT DISTINCT symbol FROM market_data_unified "
                       "WHERE timeframe='5minute'").fetchall()
    return sorted({r[0] for r in rows} - {"NIFTY50", "BANKNIFTY"})


def load_panel(con, syms):
    q = ("SELECT date, symbol, open, close FROM market_data_unified "
         "WHERE timeframe='day' AND symbol IN (%s)" % ",".join("?" * len(syms)))
    df = pd.read_sql(q, con, params=syms)
    df["date"] = df["date"].str[:10]
    close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
    open_ = df.pivot_table(index="date", columns="symbol", values="open").sort_index()
    # liquidity/price sanity: drop ultra-cheap prints
    close = close.where(close > 20)
    open_ = open_.where(open_ > 20)
    return open_, close


def decile_forward(signal: pd.DataFrame, fwd: pd.DataFrame, ascending_good=False,
                   min_names=20):
    """Mean forward return per signal-decile (pooled). Returns (decile_means, ls)."""
    sig = signal.copy()
    enough = sig.notna().sum(axis=1) >= min_names
    sig = sig[enough]
    fwd = fwd.reindex(sig.index)
    rank = sig.rank(axis=1, pct=True)
    dec = np.clip((rank * 10).apply(np.floor), 0, 9)
    sv, dv, fv = sig.values, dec.values, fwd.values
    means = []
    for d in range(10):
        m = (dv == d) & ~np.isnan(fv)
        means.append(np.nanmean(fv[m]) if m.any() else np.nan)
    means = np.array(means)
    # long-short: top decile (9) minus bottom (0); flip if low-signal is "good"
    ls = means[9] - means[0]
    # monthly LS series for a hit-rate / t-stat
    monthly = []
    idx = pd.to_datetime(sig.index)
    for _, grp in pd.Series(range(len(sig)), index=idx).groupby([idx.year, idx.month]):
        rows = grp.values
        dd, ff = dv[rows], fv[rows]
        top = ff[(dd == 9) & ~np.isnan(ff)]
        bot = ff[(dd == 0) & ~np.isnan(ff)]
        if len(top) and len(bot):
            monthly.append(np.nanmean(top) - np.nanmean(bot))
    monthly = np.array(monthly)
    t = (monthly.mean() / monthly.std() * np.sqrt(len(monthly))) if len(monthly) > 2 and monthly.std() > 0 else np.nan
    hit = float((monthly > 0).mean()) if len(monthly) else np.nan
    return means, ls, t, hit, len(monthly)


def show(name, means, ls, t, hit, nm, note=""):
    print(f"\n## {name}{('  — ' + note) if note else ''}")
    print("   decile:  " + " ".join(f"D{i}" for i in range(10)))
    print("   fwd%:   " + " ".join(f"{m*100:+5.2f}" for m in means))
    print(f"   LS(D9-D0)={ls*100:+.2f}%   t~{t:.2f}   monthly-hit={hit*100:.0f}%   ({nm} months)")


def main():
    con = sqlite3.connect(DB)
    syms = liquid_universe(con)
    print(f"universe: {len(syms)} liquid names; loading daily panel...")
    open_, close = load_panel(con, syms)
    print(f"panel: {close.shape[0]} dates ({close.index[0]} -> {close.index[-1]}), "
          f"{close.shape[1]} symbols")
    ret = close.pct_change()

    # ---- forward returns ----
    fwd5 = close.shift(-5) / close - 1
    fwd21 = close.shift(-21) / close - 1

    # 1. Short-term reversal: signal = past 5d return (losers expected to bounce)
    sig = close / close.shift(5) - 1
    means, ls, t, hit, nm = decile_forward(sig, fwd5)
    show("1. Short-term reversal (past-5d rank -> fwd-5d)", means, ls, t, hit, nm,
         "want DECREASING (D0 losers beat D9 winners)")

    # 2. Momentum 12-1: signal = ret from t-252 to t-21
    sig = close.shift(21) / close.shift(252) - 1
    means, ls, t, hit, nm = decile_forward(sig, fwd21)
    show("2. Momentum 12-1 (skip 1m -> fwd-21d)", means, ls, t, hit, nm,
         "want INCREASING (D9 winners keep winning)")

    # 3. Overnight vs intraday drift (pooled means)
    ov = (open_ / close.shift(1) - 1)
    intra = (close / open_ - 1)
    print("\n## 3. Overnight vs intraday drift (pooled daily mean)")
    print(f"   overnight (close->open): {np.nanmean(ov.values)*100:+.4f}%/day  "
          f"x252 = {np.nanmean(ov.values)*252*100:+.1f}%/yr")
    print(f"   intraday  (open->close): {np.nanmean(intra.values)*100:+.4f}%/day  "
          f"x252 = {np.nanmean(intra.values)*252*100:+.1f}%/yr")

    # 4. Turn-of-month: mean daily ret on ToM days vs rest
    idx = pd.to_datetime(ret.index)
    dom = idx.day
    is_tom = (dom <= 3) | (dom >= 28)         # rough turn-of-month window
    rv = ret.values
    is_tom = np.asarray(is_tom)
    tom_mean = np.nanmean(rv[is_tom])
    rest_mean = np.nanmean(rv[~is_tom])
    print("\n## 4. Turn-of-month seasonality (mean daily ret)")
    print(f"   ToM days (<=3 or >=28): {tom_mean*100:+.4f}%/day   "
          f"rest: {rest_mean*100:+.4f}%/day   edge={ (tom_mean-rest_mean)*100:+.4f}%/day")

    # 5. Low-volatility: signal = past-60d vol; want LOW vol better risk-adj
    vol = ret.rolling(60).std()
    means, ls, t, hit, nm = decile_forward(vol, fwd21)
    show("5. Low-volatility (past-60d vol rank -> fwd-21d)", means, ls, t, hit, nm,
         "want DECREASING (D0 low-vol beats D9 high-vol)")

    # 6. 52-week-high proximity: signal = close / 252d-high
    sig = close / close.rolling(252).max()
    means, ls, t, hit, nm = decile_forward(sig, fwd21)
    show("6. 52wk-high proximity (close/252d-high -> fwd-21d)", means, ls, t, hit, nm,
         "want INCREASING (near-high keeps running)")

    print("\nDONE")


if __name__ == "__main__":
    main()
