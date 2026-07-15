"""EXP-F1: NIFTY opening-range breakout -> 1-4 session hold, 5-min, IS 2015-2021.

Pre-registration: experiments/F1_nifty_orb_5min/NIFTY_ORB_5MIN_SWEEP_STATUS.md
GRID LOCKED: W {6,12} x dir {L,S} x ts {1,2,4} = 12 cells. Stop = opposite OR
bound. Entry next 5-min bar open. IS ONLY (2015-02-01 .. 2021-09-30).
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

RESULTS = STUDY / "experiments" / "F1_nifty_orb_5min" / "results"
IS_START, IS_END = "2015-02-01", "2021-09-30"
SYMBOL = "NIFTY50"
GRID = [(w, d, ts) for w in (6, 12) for d in (1, -1) for ts in (1, 2, 4)]


def build_entries(df: pd.DataFrame, w: int, d: int) -> pd.DataFrame:
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
    # first trigger per session only
    first = raw & (raw.groupby(sess).cumsum() == 1)
    sig = df.index[first]
    sig = sig[(sig >= pd.Timestamp(IS_START)) & (sig <= pd.Timestamp(IS_END + " 23:59:59"))]
    return pd.DataFrame({"direction": d, "stop": stop.loc[sig],
                         "target": np.nan}, index=sig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = loader.load_bars(SYMBOL, "5minute", start="2015-02-01", end=IS_END)
    print(f"{SYMBOL} 5m bars: {len(df):,} ({df.index.min()} -> {df.index.max()})")

    cfg = {ts: BTConfig(cost=CostConfig(product="FUTURES_PROXY"),
                        time_stop_sessions=ts) for ts in (1, 2, 4)}
    all_trades = []
    rows = []
    for (w, d, ts) in GRID:
        lbl = f"W{w}_{'L' if d == 1 else 'S'}_ts{ts}"
        tr = run_symbol(df, build_entries(df, w, d), cfg[ts], symbol=SYMBOL)
        tr["cell"] = lbl
        all_trades.append(tr)
        r = tr["net_ret"].to_numpy(float)
        gr = tr["gross_ret"].to_numpy(float)
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan
        yr = pd.to_datetime(tr["entry_time"]).dt.year
        yr_net = tr.groupby(yr)["net_ret"].mean()
        rows.append({"cell": lbl, "n": len(r),
                     "gross_bps": round(gr.mean() * 1e4, 2),
                     "net_bps": round(r.mean() * 1e4, 2),
                     "t_net": round(t, 2),
                     "win": round((r > 0).mean(), 3),
                     "pct_yr_pos": round((yr_net > 0).mean(), 3),
                     "avg_hold": round(tr["hold_sessions"].mean(), 2)})
        print(f"{lbl}: n={len(r)} net={rows[-1]['net_bps']}bps t={rows[-1]['t_net']}",
              flush=True)

    trades = pd.concat(all_trades, ignore_index=True)
    trades.to_csv(RESULTS / "f1_trades.csv", index=False)
    agg = pd.DataFrame(rows).sort_values("t_net", ascending=False)
    agg.to_csv(RESULTS / "f1_ranking.csv", index=False)
    print(f"\n=== F1 ranking (net, IS {IS_START}..{IS_END}) ===")
    print(agg.to_string(index=False))

    # per-year detail for best cell
    best = agg.iloc[0]["cell"]
    g = trades[trades["cell"] == best]
    yr = pd.to_datetime(g["entry_time"]).dt.year
    tab = g.groupby(yr).agg(n=("net_ret", "size"),
                            net_bps=("net_ret", lambda x: round(x.mean() * 1e4, 1)),
                            win=("net_ret", lambda x: round((x > 0).mean(), 3)))
    print(f"\n--- {best} per-year ---")
    print(tab.to_string())
    print(f"\ndone in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
