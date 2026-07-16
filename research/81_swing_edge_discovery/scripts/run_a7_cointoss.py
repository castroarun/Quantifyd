"""EXP-A7: first-candle coin-toss break, fixed-% stop, RR 1:1.5 (user idea).

Pre-registration: experiments/A7_coin_toss_orb/FIRST_CANDLE_COINTOSS_5MIN_SWEEP_STATUS.md
GRID LOCKED: s {0.5,0.75,1.0}% x ts {1,2} = 6 cells. NIFTY + 9 deep names, IS.
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

RESULTS = STUDY / "experiments" / "A7_coin_toss_orb" / "results"
IS_START, IS_END = "2015-02-01", "2021-09-30"
SYMS = ["NIFTY50", "HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "TCS",
        "SBIN", "ITC", "HINDUNILVR", "BHARTIARTL"]
RR = 1.5
GRID = [(s, ts) for s in (0.005, 0.0075, 0.010) for ts in (1, 2)]


def entries(df5: pd.DataFrame, s_pct: float) -> pd.DataFrame:
    sess = df5.index.normalize()
    g = df5.groupby(sess)
    pos = g.cumcount()
    b1_hi = df5["high"].where(pos == 0).groupby(sess).transform("max")
    b1_lo = df5["low"].where(pos == 0).groupby(sess).transform("min")
    up = (pos >= 1) & (df5["close"] > b1_hi)
    dn = (pos >= 1) & (df5["close"] < b1_lo)
    either = up | dn
    first = either & (either.groupby(sess).cumsum() == 1)
    sig = df5.index[first]
    sig = sig[(sig >= pd.Timestamp(IS_START))
              & (sig <= pd.Timestamp(IS_END + " 23:59:59"))]
    d = np.where(up.loc[sig], 1, -1)
    base = df5["close"].loc[sig]
    stop = base * (1 - s_pct * d)
    target = base * (1 + RR * s_pct * d)
    return pd.DataFrame({"direction": d, "stop": stop, "target": target},
                        index=sig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = {ts: BTConfig(cost=CostConfig(product="FUTURES_PROXY"),
                        time_stop_sessions=ts) for ts in (1, 2)}
    all_tr = []
    for sym in SYMS:
        df5 = loader.load_bars(sym, "5minute", start=IS_START, end=IS_END)
        if df5.empty:
            continue
        for (s, ts) in GRID:
            tr = run_symbol(df5, entries(df5, s), cfg[ts], symbol=sym)
            tr["cell"] = f"s{s*100:.2f}_ts{ts}"
            all_tr.append(tr)
        print(f"{sym} done ({(time.time()-t0)/60:.1f}m)", flush=True)

    trades = pd.concat(all_tr, ignore_index=True)
    trades.to_csv(RESULTS / "a7_trades.csv", index=False)
    rows = []
    for cell, g in trades.groupby("cell"):
        for leg, gg in (("both", g), ("L", g[g["direction"] == 1]),
                        ("S", g[g["direction"] == -1])):
            r = gg["net_ret"].to_numpy(float)
            if len(r) < 2:
                continue
            t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
            sym_pnl = gg.groupby("symbol")["net_ret"].sum()
            rows.append({"cell": cell, "leg": leg, "n": len(r),
                         "gross_bps": round(gg["gross_ret"].mean() * 1e4, 1),
                         "net_bps": round(r.mean() * 1e4, 1),
                         "t_net": round(t, 2),
                         "win": round((r > 0).mean(), 3),
                         "sym_pos": round((sym_pnl > 0).mean(), 2)})
    agg = pd.DataFrame(rows).sort_values(["leg", "t_net"], ascending=[True, False])
    agg.to_csv(RESULTS / "a7_ranking.csv", index=False)
    print("\n=== A7 coin-toss ranking (net 3bp, IS 2015-2021) ===")
    print(agg.to_string(index=False))
    print(f"\ndone in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
