"""EXP-A2: stock ORB -> 2-4 session hold, 9 deep F&O names, 5-min, IS 2015-2021.

Pre-registration: experiments/A2_stock_orb_5min/STOCK_ORB_5MIN_SWEEP_STATUS.md
GRID LOCKED: W {6,12} x dir {L,S} x ts {2,4} = 8 cells x 9 symbols.
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

from engine import loader                              # noqa: E402
from engine.backtester import BTConfig, run_symbol     # noqa: E402
from engine.costs import CostConfig                    # noqa: E402

RESULTS = STUDY / "experiments" / "A2_stock_orb_5min" / "results"
IS_START, IS_END = "2015-02-01", "2021-09-30"
UNIVERSE = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "TCS", "SBIN",
            "ITC", "HINDUNILVR", "BHARTIARTL"]
GRID = [(w, d, ts) for w in (6, 12) for d in (1, -1) for ts in (2, 4)]


def build_entries(df, w, d):
    sess = df.index.normalize()
    g = df.groupby(sess)
    pos = g.cumcount()
    or_high = df["high"].where(pos < w).groupby(sess).transform("max")
    or_low = df["low"].where(pos < w).groupby(sess).transform("min")
    if d == 1:
        raw = (pos >= w) & (df["close"] > or_high)
        stop = or_low
    else:
        raw = (pos >= w) & (df["close"] < or_low)
        stop = or_high
    first = raw & (raw.groupby(sess).cumsum() == 1)
    sig = df.index[first]
    sig = sig[(sig >= pd.Timestamp(IS_START))
              & (sig <= pd.Timestamp(IS_END + " 23:59:59"))]
    return pd.DataFrame({"direction": d, "stop": stop.loc[sig],
                         "target": np.nan}, index=sig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = {ts: BTConfig(cost=CostConfig(product="FUTURES_PROXY"),
                        time_stop_sessions=ts) for ts in (2, 4)}
    all_tr = []
    for sym in UNIVERSE:
        df = loader.load_bars(sym, "5minute", start="2015-02-01", end=IS_END)
        if df.empty:
            print(f"{sym}: NO DATA, skipped")
            continue
        for (w, d, ts) in GRID:
            tr = run_symbol(df, build_entries(df, w, d), cfg[ts], symbol=sym)
            tr["cell"] = f"W{w}_{'L' if d == 1 else 'S'}_ts{ts}"
            all_tr.append(tr)
        print(f"{sym} done ({df.index.min().date()}->) "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)

    trades = pd.concat(all_tr, ignore_index=True)
    trades.to_csv(RESULTS / "a2_trades.csv", index=False)
    rows = []
    for lbl, g in trades.groupby("cell"):
        r = g["net_ret"].to_numpy(float)
        gr = g["gross_ret"].to_numpy(float)
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan
        sym_pnl = g.groupby("symbol")["net_ret"].sum()
        yr = pd.to_datetime(g["entry_time"]).dt.year
        yr_net = g.groupby(yr)["net_ret"].mean()
        rows.append({"cell": lbl, "n": len(r),
                     "gross_bps": round(gr.mean() * 1e4, 2),
                     "net_bps": round(r.mean() * 1e4, 2),
                     "t_net": round(t, 2), "win": round((r > 0).mean(), 3),
                     "pct_sym_pos": round((sym_pnl > 0).mean(), 3),
                     "pct_yr_pos": round((yr_net > 0).mean(), 3)})
    agg = pd.DataFrame(rows).sort_values("t_net", ascending=False)
    agg.to_csv(RESULTS / "a2_ranking.csv", index=False)
    print(f"\n=== A2 ranking (net, IS {IS_START}..{IS_END}) ===")
    print(agg.to_string(index=False))
    print(f"\ndone in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
