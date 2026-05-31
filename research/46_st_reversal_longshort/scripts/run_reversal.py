"""Short-term reversal — market-neutral long-short portfolio backtest.

Long recent losers / short recent winners, dollar-neutral, daily rebalance with
H-day overlapping holding. Sweeps lookback x quantile x holding, applies a
realistic per-turnover cost, and reports net Sharpe/return/MaxDD on the full
sample and a 2018+ subperiod. Saves the best config's equity curve + per-year.
"""
from __future__ import annotations

import csv
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

COST_ONEWAY = 0.0010      # 10 bps per unit turnover (one way)
MIN_NAMES = 50
LOOKBACKS = (3, 5, 10)
QUANTILES = ("decile", "quintile", "rankwt")
HOLDS = (3, 5, 10, 21)


def load_close(con):
    syms = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='5minute'"
    ).fetchall()} - {"NIFTY50", "BANKNIFTY"})
    q = ("SELECT date,symbol,close FROM market_data_unified "
         "WHERE timeframe='day' AND symbol IN (%s)" % ",".join("?" * len(syms)))
    df = pd.read_sql(q, con, params=syms)
    df["date"] = df["date"].str[:10]
    close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
    return close.where(close > 20)


def target_weights(signal, qtype):
    """Per-day dollar-neutral weights: long low-signal (losers), short high (winners).
    Longs sum to +1, shorts to -1. Rows with < MIN_NAMES valid are zeroed."""
    valid = signal.notna()
    r = signal.rank(axis=1, pct=True)        # high rank = winner
    w = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    if qtype in ("decile", "quintile"):
        q = 0.10 if qtype == "decile" else 0.20
        lo = (r <= q) & valid                # losers -> long
        hi = (r >= 1 - q) & valid            # winners -> short
        nL = lo.sum(axis=1).replace(0, np.nan)
        nS = hi.sum(axis=1).replace(0, np.nan)
        w = lo.div(nL, axis=0) - hi.div(nS, axis=0)
    else:  # rank-weighted: linear in (0.5 - pct), normalized each side
        raw = (0.5 - r).where(valid)
        pos = raw.clip(lower=0)
        neg = raw.clip(upper=0)
        w = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0) \
            + neg.div(neg.sum(axis=1).replace(0, np.nan).abs(), axis=0)
    enough = valid.sum(axis=1) >= MIN_NAMES
    w = w.where(enough, 0.0).fillna(0.0)
    return w


def metrics(net, eq):
    if len(net) < 30 or net.std() == 0:
        return dict(ann=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    ann = net.mean() * 252
    vol = net.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    peak = eq.cummax()
    maxdd = (eq / peak - 1).min()
    return dict(ann=ann, vol=vol, sharpe=sharpe, maxdd=maxdd)


def backtest(close, ret, N, qtype, H, cost=COST_ONEWAY):
    signal = close / close.shift(N) - 1
    w = target_weights(signal, qtype)
    book = w.rolling(H).mean()                       # overlapping H-day holding
    gross = (book.shift(1) * ret).sum(axis=1)
    turnover = book.diff().abs().sum(axis=1)
    net = gross - turnover * cost
    net = net.dropna()
    res = {}
    for tag, mask in (("full", net.index >= "2000"),
                      ("2018+", net.index >= "2018")):
        s = net[mask]
        eq = (1 + s).cumprod()
        m = metrics(s, eq)
        res[tag] = m
        res[tag]["turn"] = float(turnover.reindex(s.index).mean())
        res[tag]["gross_ann"] = float(gross.reindex(s.index).mean() * 252)
    res["net_series"] = net
    return res


def main():
    con = sqlite3.connect(DB)
    print("loading daily panel...")
    close = load_close(con)
    ret = close.pct_change()
    print(f"panel {close.shape[0]} dates ({close.index[0]} -> {close.index[-1]}), "
          f"{close.shape[1]} names")

    rows = []
    best = None
    for N in LOOKBACKS:
        for q in QUANTILES:
            for H in HOLDS:
                r = backtest(close, ret, N, q, H)
                f, s = r["full"], r["2018+"]
                rows.append(dict(
                    N=N, quantile=q, H=H,
                    full_sharpe=round(f["sharpe"], 2), full_ann=round(f["ann"] * 100, 1),
                    full_maxdd=round(f["maxdd"] * 100, 1),
                    s18_sharpe=round(s["sharpe"], 2), s18_ann=round(s["ann"] * 100, 1),
                    s18_maxdd=round(s["maxdd"] * 100, 1),
                    s18_grossann=round(s["gross_ann"] * 100, 1),
                    turn=round(s["turn"], 3)))
                key = s["sharpe"] if not np.isnan(s["sharpe"]) else -9
                if best is None or key > best[0]:
                    best = (key, N, q, H, r)
    rows.sort(key=lambda x: (x["s18_sharpe"] if x["s18_sharpe"] == x["s18_sharpe"] else -9),
              reverse=True)
    with open(OUT / "grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Long-short reversal grid (sorted by 2018+ net Sharpe) ===")
    print(f"{'N':>2} {'quantile':9} {'H':>3} | {'2018+ Sharpe':>12} {'ann%':>6} {'MaxDD%':>7} "
          f"{'gross_ann%':>10} {'turn/d':>7} | {'full Sharpe':>11} {'ann%':>6}")
    for r in rows[:14]:
        print(f"{r['N']:>2} {r['quantile']:9} {r['H']:>3} | {r['s18_sharpe']:>12} "
              f"{r['s18_ann']:>6} {r['s18_maxdd']:>7} {r['s18_grossann']:>10} {r['turn']:>7} | "
              f"{r['full_sharpe']:>11} {r['full_ann']:>6}")

    # best config detail
    _, N, q, H, r = best
    print(f"\n=== BEST (2018+ Sharpe): N={N} {q} H={H} ===")
    net = r["net_series"]
    s18 = net[net.index >= "2018"]
    eq = (1 + s18).cumprod()
    eq.to_frame("equity").to_csv(OUT / "equity_best.csv")
    # per-year
    yr = net.copy()
    yr.index = pd.to_datetime(yr.index)
    print("   year   net%   sharpe   maxdd%")
    for y, g in yr.groupby(yr.index.year):
        if len(g) < 30:
            continue
        e = (1 + g).cumprod()
        m = metrics(g, e)
        print(f"   {y}  {(e.iloc[-1]-1)*100:+6.1f}  {m['sharpe']:6.2f}  {m['maxdd']*100:6.1f}")
    # cost sensitivity on best
    print("\n   cost sensitivity (2018+ net Sharpe / ann%):")
    for c in (0.0005, 0.0010, 0.0015, 0.0020):
        rc = backtest(close, ret, N, q, H, cost=c)["2018+"]
        print(f"     {c*1e4:.0f} bps/side -> Sharpe {rc['sharpe']:.2f}  ann {rc['ann']*100:+.1f}%")


if __name__ == "__main__":
    main()
