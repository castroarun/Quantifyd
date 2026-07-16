"""EXP-G5: union portfolio of the morning-strength triggers (post-A4 breadth).

Book = F&O-only subset (liquid, shortable, futures-proxy cost story holds;
non-F&O breadth rows are evidence, not book members — pre-declared).
Sleeves:
  - index: NIFTY gap>=0.25% + ORB W12 long (locked, 1bp slip, 1.0% risk)
  - stocks: union of {gap-ORB W6, D2 O=L break, C2 CPR-open} long ts4
    (locked, 3bp slip, 0.5% risk), DEDUP one trade per (symbol, entry day)
    with priority gap-ORB > D2 > C2.
Construction locked from G4: cap 6 concurrent, 25% notional cap, daily MTM.
Window: IS+Val 2015-02..2023-12. OOS stays locked (separate runner, only
after user checkpoint).

Run: venv/bin/python3 research/81_swing_edge_discovery/scripts/run_g5_union_book.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
for p in (str(STUDY), str(ROOT), str(STUDY / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader, metrics                     # noqa: E402
from engine.backtester import BTConfig, run_symbol     # noqa: E402
from engine.costs import CostConfig                    # noqa: E402
from run_g3_orb_validation import orb_entries          # noqa: E402
from run_setups_battery import d2_entries, c2_entries  # noqa: E402

RESULTS = STUDY / "experiments" / "G5_union_book" / "results"
START, END = "2015-02-01", "2023-12-31"
RISK_IDX, RISK_STK = 0.010, 0.005
MAX_CONCURRENT = 6
MAX_NOTIONAL_PCT = 0.25
PRIORITY = {"ORB": 0, "D2": 1, "C2": 2}
CFG_IDX = BTConfig(cost=CostConfig(product="FUTURES_PROXY", slippage=0.0001),
                   time_stop_sessions=4)
CFG_STK = BTConfig(cost=CostConfig(product="FUTURES_PROXY", slippage=0.0003),
                   time_stop_sessions=4)
MIN_SESSIONS = 500


import os
ORB_ONLY = os.getenv("G5_ORB_ONLY") == "1"   # A4 gate: D2/C2 failed breadth
W_STOCKS = int(os.getenv("G5_W", "6"))       # A4 breadth: W12 dominant
CAP = int(os.getenv("G5_CAP", "6"))
RISK_STK_ENV = float(os.getenv("G5_RISK_STK", "0.005"))
BOOK_NOTIONAL_CAP = float(os.getenv("G5_BOOK_NOTIONAL", "99.0"))  # x equity
# equal-NOTIONAL sizing (fraction of equity per position); 0 = equal-risk mode.
# Rationale: A4 validated the EQUAL-WEIGHTED per-trade edge; 1/stop-width
# sizing overweights narrow-OR (losing) trades and inverts it.
EQUAL_NOTIONAL = float(os.getenv("G5_EQUAL_NOTIONAL", "0"))
# within-day intake ranked by gap size (validated monotone dose-response,
# F2/G3); gap is fixed at the 09:15 open -> causal ranking. 0 = clock order.
GAP_RANK_TOP = int(os.getenv("G5_GAP_RANK", "0"))


def collect() -> tuple[pd.DataFrame, dict]:
    from services.data_manager import FNO_LOT_SIZES
    fno = sorted(FNO_LOT_SIZES.keys())
    closes = {}
    rows = []

    def day_gaps(df5):
        d = loader.to_daily(df5)
        return (d["open"] / d["close"].shift(1) - 1)

    nifty = loader.load_bars("NIFTY50", "5minute", start=START, end=END)
    closes["NIFTY50"] = loader.to_daily(nifty)["close"]
    tri = run_symbol(nifty, orb_entries(nifty, 12, START, END, 0.0025),
                     CFG_IDX, symbol="NIFTY50")
    tri["trigger"], tri["risk_pct"] = "ORB", RISK_IDX
    g = day_gaps(nifty)
    tri["gap"] = pd.to_datetime(tri["entry_time"]).dt.normalize().map(g).values
    rows.append(tri)

    for sym in fno:
        df = loader.load_bars(sym, "5minute", start=START, end=END)
        if df.empty or df.index.normalize().nunique() < MIN_SESSIONS:
            continue
        closes[sym] = loader.to_daily(df)["close"]
        trig_list = [("ORB", orb_entries(df, W_STOCKS, START, END, 0.0025))]
        if not ORB_ONLY:
            trig_list += [("D2", d2_entries(df, "OL_long")),
                          ("C2", c2_entries(df, "long"))]
        g = day_gaps(df)
        for trig, ent in trig_list:
            t = run_symbol(df, ent, CFG_STK, symbol=sym)
            t["trigger"], t["risk_pct"] = trig, RISK_STK_ENV
            t["gap"] = pd.to_datetime(t["entry_time"]).dt.normalize().map(g).values
            rows.append(t)
    tr = pd.concat(rows, ignore_index=True)
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    tr["exit_time"] = pd.to_datetime(tr["exit_time"])
    # dedup: one trade per (symbol, entry day), trigger priority
    tr["prio"] = tr["trigger"].map(PRIORITY)
    tr["eday"] = tr["entry_time"].dt.normalize()
    tr = (tr.sort_values(["symbol", "eday", "prio"])
            .drop_duplicates(subset=["symbol", "eday"], keep="first"))
    return tr.sort_values("entry_time").reset_index(drop=True), closes


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trades, closes = collect()
    if GAP_RANK_TOP > 0:
        rank = trades.groupby("eday")["gap"].rank(ascending=False, method="first")
        before = len(trades)
        trades = trades[rank <= GAP_RANK_TOP].reset_index(drop=True)
        print(f"gap-rank intake: top {GAP_RANK_TOP}/day -> {len(trades)} of {before}")
    print(f"union trades: {len(trades)} across {trades['symbol'].nunique()} symbols")
    print(trades.groupby("trigger")["net_ret"].agg(["size", "mean"])
          .assign(mean_bps=lambda x: (x["mean"] * 1e4).round(1))[["size", "mean_bps"]]
          .to_string(), flush=True)

    cal_set = set()
    for c in closes.values():
        cal_set |= set(c.index)
    cal = pd.DatetimeIndex(sorted(d for d in cal_set
                                  if pd.Timestamp(START) <= d <= pd.Timestamp(END)))
    equity = pd.Series(np.nan, index=cal)
    eq, open_pos, ti = 1.0, [], 0
    n_taken = n_skipped = 0
    for d in cal:
        while ti < len(trades) and trades.loc[ti, "eday"] <= d:
            tr = trades.loc[ti]
            ti += 1
            if tr["eday"] != d:
                continue
            if len(open_pos) >= CAP:
                n_skipped += 1
                continue
            risk = abs(tr["entry_px"] - tr["stop"])
            if risk <= 0:
                continue
            book_notional = sum(p["qty"] * p["mark"] for p in open_pos)
            headroom = max(eq * BOOK_NOTIONAL_CAP - book_notional, 0.0)
            if EQUAL_NOTIONAL > 0:
                base_qty = (eq * EQUAL_NOTIONAL) / tr["entry_px"]
            else:
                base_qty = (eq * tr["risk_pct"]) / risk
            qty = min(base_qty, (eq * MAX_NOTIONAL_PCT) / tr["entry_px"],
                      headroom / tr["entry_px"])
            if qty * tr["entry_px"] < eq * 0.005:      # headroom too thin
                n_skipped += 1
                continue
            slipped = tr["direction"] * (tr["exit_px"] / tr["entry_px"] - 1)
            open_pos.append({"symbol": tr["symbol"], "qty": qty,
                             "dir": tr["direction"], "mark": tr["entry_px"],
                             "exit_day": tr["exit_time"].normalize(),
                             "exit_px": tr["exit_px"],
                             "charges": (slipped - tr["net_ret"]) * tr["entry_px"] * qty})
            n_taken += 1
        pnl, still = 0.0, []
        for p in open_pos:
            if d >= p["exit_day"]:
                pnl += p["qty"] * p["dir"] * (p["exit_px"] - p["mark"]) - p["charges"]
            else:
                c = closes[p["symbol"]].get(d, np.nan)
                if not np.isnan(c):
                    pnl += p["qty"] * p["dir"] * (c - p["mark"])
                    p["mark"] = c
                still.append(p)
        open_pos = still
        eq += pnl
        equity[d] = eq

    equity = equity.ffill()
    cs = metrics.curve_stats(equity)
    print(f"\ntaken={n_taken} skipped_at_cap={n_skipped}")
    print("\n=== G5 union book (F&O-only, 2015-02..2023-12) ===")
    for k, v in cs.items():
        print(f"{k:>18}: {v:.4f}" if isinstance(v, float) else f"{k:>18}: {v}")
    print("\nper-year:")
    print((metrics.per_year_table(equity.pct_change()) * 100).round(2).to_string())
    bench = closes["NIFTY50"].reindex(cal).ffill()
    bs = metrics.curve_stats(bench / bench.iloc[0])
    print(f"\nNIFTY B&H: CAGR {bs['cagr']:.2%} Sharpe {bs['sharpe']:.2f} "
          f"MaxDD {bs['max_dd']:.2%}")
    print(f"corr vs NIFTY: {equity.pct_change().corr(bench.pct_change()):.3f}")
    equity.to_csv(RESULTS / "g5_nav.csv")
    trades.to_csv(RESULTS / "g5_trades.csv", index=False)
    print(f"\ndone in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
