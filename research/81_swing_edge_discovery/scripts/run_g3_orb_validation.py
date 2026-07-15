"""EXP-G3: validation battery for gap-up + ORB long (locked configs, no re-opt).

Pre-registration: experiments/G3_orb_validation/GAPORB_G3_VALIDATION_5MIN_RUN_STATUS.md
Parts: 1 plateau heatmap (IS) - 2 walk-forward no-reopt (2015-2023) -
       3 Monte Carlo - 4 regime splits - 5 super-winner guard (stocks).
OOS 2024+ NOT touched.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
for p in (str(STUDY), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader, metrics                     # noqa: E402
from engine.backtester import BTConfig, run_symbol     # noqa: E402
from engine.costs import CostConfig                    # noqa: E402

RESULTS = STUDY / "experiments" / "G3_orb_validation" / "results"
IS_START, IS_END = "2015-02-01", "2021-09-30"
WF_END = "2023-12-31"           # walk-forward extends through Val (read-only)
STOCKS = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "TCS", "SBIN",
          "ITC", "HINDUNILVR", "BHARTIARTL"]
CFG1 = BTConfig(cost=CostConfig(product="FUTURES_PROXY", slippage=0.0001),
                time_stop_sessions=4)
CFG3 = BTConfig(cost=CostConfig(product="FUTURES_PROXY", slippage=0.0003),
                time_stop_sessions=4)


def orb_entries(df5, w, lo, hi, gap_thr):
    sess = df5.index.normalize()
    g = df5.groupby(sess)
    pos = g.cumcount()
    or_high = df5["high"].where(pos < w).groupby(sess).transform("max")
    or_low = df5["low"].where(pos < w).groupby(sess).transform("min")
    raw = (pos >= w) & (df5["close"] > or_high)
    first = raw & (raw.groupby(sess).cumsum() == 1)
    sig = df5.index[first]
    sig = sig[(sig >= pd.Timestamp(lo)) & (sig <= pd.Timestamp(hi + " 23:59:59"))]
    d = loader.to_daily(df5)
    gap = d["open"] / d["close"].shift(1) - 1
    ok_days = set(d.index[gap >= gap_thr])
    sig = pd.DatetimeIndex([t for t in sig if t.normalize() in ok_days])
    return pd.DataFrame({"direction": 1, "stop": or_low.loc[sig],
                         "target": np.nan}, index=sig)


def trades_for(df5, w, gap_thr, lo, hi, cfg, sym):
    return run_symbol(df5, orb_entries(df5, w, lo, hi, gap_thr), cfg, symbol=sym)


def stats(tr):
    r = tr["net_ret"].to_numpy(float)
    if len(r) < 2:
        return {"n": len(r), "net_bps": np.nan, "t": np.nan}
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    return {"n": len(r), "net_bps": round(r.mean() * 1e4, 2), "t": round(t, 2)}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nifty = loader.load_bars("NIFTY50", "5minute", start="2015-02-01", end=WF_END)
    stock_bars = {s: loader.load_bars(s, "5minute", start="2015-02-01", end=WF_END)
                  for s in STOCKS}

    # ---------- Part 1: plateau (IS, 1bp for index / 3bp stocks) ----------
    print("=== PART 1: parameter plateau (IS) ===", flush=True)
    rows = []
    for w in (4, 6, 9, 12, 15):
        for gt in (0.0010, 0.0020, 0.0025, 0.0035, 0.0050):
            tr = trades_for(nifty.loc[:IS_END + " 23:59"], w, gt,
                            IS_START, IS_END, CFG1, "NIFTY50")
            s = stats(tr)
            rows.append({"set": "index", "W": w, "gap_pct": gt * 100, **s})
            trs = pd.concat([trades_for(stock_bars[sym].loc[:IS_END + " 23:59"],
                                        w, gt, IS_START, IS_END, CFG3, sym)
                             for sym in STOCKS], ignore_index=True)
            s2 = stats(trs)
            rows.append({"set": "stocks", "W": w, "gap_pct": gt * 100, **s2})
    plat = pd.DataFrame(rows)
    plat.to_csv(RESULTS / "g3_plateau.csv", index=False)
    for st in ("index", "stocks"):
        piv = plat[plat["set"] == st].pivot(index="W", columns="gap_pct",
                                            values="net_bps")
        print(f"\n{st} net_bps heatmap (W x gap%):")
        print(piv.to_string(), flush=True)

    # ---------- Part 2: walk-forward, no re-opt ----------
    print("\n=== PART 2: rolling half-year performance (locked cfgs) ===", flush=True)
    tr_idx = trades_for(nifty, 12, 0.0025, IS_START, WF_END, CFG1, "NIFTY50")
    tr_stk = pd.concat([trades_for(stock_bars[s], 6, 0.0025, IS_START, WF_END,
                                   CFG3, s) for s in STOCKS], ignore_index=True)
    for name, tr in (("index W12", tr_idx), ("stocks W6", tr_stk)):
        et = pd.to_datetime(tr["entry_time"])
        half = et.dt.year.astype(str) + "H" + ((et.dt.month > 6) + 1).astype(str)
        tab = tr.groupby(half)["net_ret"].agg(["size", "mean"])
        tab["mean_bps"] = (tab["mean"] * 1e4).round(1)
        pos = (tab["mean"] > 0).mean()
        worst = tab["mean_bps"].min()
        print(f"\n{name}: {pos:.0%} of half-years positive, worst={worst}bps")
        print(tab[["size", "mean_bps"]].to_string(), flush=True)
        tab.to_csv(RESULTS / f"g3_wf_{name.split()[0]}.csv")

    # ---------- Part 3: Monte Carlo ----------
    print("\n=== PART 3: Monte Carlo (trade bootstrap, R-sized) ===", flush=True)
    for name, tr in (("index", tr_idx[pd.to_datetime(tr_idx['entry_time']) <= IS_END]),
                     ("stocks", tr_stk[pd.to_datetime(tr_stk['entry_time']) <= IS_END])):
        # size at 1% equity risk per trade: per-trade equity ret = R * 1%
        rr = tr["r_multiple"].dropna().to_numpy(float) * 0.01
        yrs = 6.66
        mc = metrics.bootstrap_mc(rr, trades_per_year=len(rr) / yrs)
        print(f"{name}: {mc}", flush=True)

    # ---------- Part 4: regime splits (locked cfgs, IS+Val window) ----------
    print("\n=== PART 4: regime splits ===", flush=True)
    vix5 = loader.load_bars("INDIAVIX", "5minute", start="2014-01-01", end=WF_END)
    vixd = loader.to_daily(vix5)["close"]
    lo_q = vixd.rolling(252).quantile(0.333).shift(1)
    hi_q = vixd.rolling(252).quantile(0.667).shift(1)
    vprev = vixd.shift(1)
    ndaily = loader.to_daily(nifty)["close"]
    dma200 = ndaily.rolling(200).mean().shift(1)
    risk_on = (ndaily.shift(1) > dma200)
    for name, tr in (("index", tr_idx), ("stocks", tr_stk)):
        d = pd.to_datetime(tr["entry_time"]).dt.normalize()
        vix_b = np.where(vprev.reindex(d).values <= lo_q.reindex(d).values, "low",
                 np.where(vprev.reindex(d).values >= hi_q.reindex(d).values,
                          "high", "mid"))
        ron = risk_on.reindex(d).values
        df = tr.assign(vix=vix_b, risk_on=ron)
        print(f"\n{name} by VIX tercile:")
        print((df.groupby("vix")["net_ret"].agg(["size", "mean"])["mean"] * 1e4)
              .round(1).to_string())
        print(f"{name} by NIFTY vs 200DMA (risk_on=True above):")
        print((df.groupby("risk_on")["net_ret"].agg(["size", "mean"])["mean"] * 1e4)
              .round(1).to_string(), flush=True)

    # ---------- Part 5: super-winner guard (stocks) ----------
    print("\n=== PART 5: super-winner guard ===", flush=True)
    contrib = tr_stk.groupby("symbol")["net_ret"].sum().sort_values(ascending=False)
    print("symbol contributions (net, sum):")
    print((contrib * 1e4).round(0).to_string())
    top2 = list(contrib.index[:2])
    guarded = tr_stk[~tr_stk["symbol"].isin(top2)]
    print(f"drop top-2 {top2}: {stats(guarded)}", flush=True)

    print(f"\nG3 battery done in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
