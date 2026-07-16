"""EXP-G6: THE ONE-TIME OOS TOUCH (2024-01-01 -> data end). User-authorized
2026-07-16. Everything here is locked; nothing is tuned on OOS; ALL results
reported regardless of outcome.

Reports:
  A. Trade-level OOS of the locked signal cells:
     - index: NIFTY W12 gap>=0.25% long ts4 @ 1bp and 3bp
     - stocks: 77-name F&O pooled, W6 and W12 (both pre-registered locks) @3bp
  B. G4 book OOS: 10 instruments (NIFTY W12 @1% risk + 9 megacaps W6 @0.5%),
     cap 6, per-position notional cap 25% (construction locked at G4).
  C. Breadth book v6 OOS: 77 F&O, W12, equal-notional 10%, cap 15,
     book notional cap 150%, gap-rank 15 (final construction, informational —
     failed IS gates, reported anyway per protocol).
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

RESULTS = STUDY / "experiments" / "G6_oos" / "results"
OOS_START = "2024-01-01"
OOS_END = "2026-07-15"
MEGA9 = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "TCS", "SBIN",
         "ITC", "HINDUNILVR", "BHARTIARTL"]

CFG1 = BTConfig(cost=CostConfig(product="FUTURES_PROXY", slippage=0.0001),
                time_stop_sessions=4)
CFG3 = BTConfig(cost=CostConfig(product="FUTURES_PROXY", slippage=0.0003),
                time_stop_sessions=4)


def tstats(tr, label):
    r = tr["net_ret"].to_numpy(float)
    if len(r) < 2:
        print(f"{label}: n={len(r)} (insufficient)")
        return
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    yr = pd.to_datetime(tr["entry_time"]).dt.year
    yr_net = tr.groupby(yr)["net_ret"].mean() * 1e4
    sym_pnl = tr.groupby("symbol")["net_ret"].sum() if "symbol" in tr else None
    extra = f" sym_pos={(sym_pnl > 0).mean():.2f}" if sym_pnl is not None and len(sym_pnl) > 1 else ""
    print(f"{label}: n={len(r)} gross={tr['gross_ret'].mean()*1e4:.1f}bps "
          f"net={r.mean()*1e4:.1f}bps t={t:.2f} win={(r > 0).mean():.3f}{extra}")
    print(f"   per-year net bps: {dict(yr_net.round(1))}")


def nav_sim(trades, closes, cap, equal_notional, book_cap, label):
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    trades["eday"] = pd.to_datetime(trades["entry_time"]).dt.normalize()
    cal_set = set()
    for c in closes.values():
        cal_set |= set(c.index)
    cal = pd.DatetimeIndex(sorted(d for d in cal_set
                                  if pd.Timestamp(OOS_START) <= d <= pd.Timestamp(OOS_END)))
    eq, open_pos, ti = 1.0, [], 0
    equity = pd.Series(np.nan, index=cal)
    for d in cal:
        while ti < len(trades) and trades.loc[ti, "eday"] <= d:
            tr = trades.loc[ti]
            ti += 1
            if tr["eday"] != d or len(open_pos) >= cap:
                continue
            risk = abs(tr["entry_px"] - tr["stop"])
            if risk <= 0:
                continue
            book_notional = sum(p["qty"] * p["mark"] for p in open_pos)
            head = max(eq * book_cap - book_notional, 0.0)
            if equal_notional > 0:
                base = (eq * equal_notional) / tr["entry_px"]
            else:
                base = (eq * tr["risk_pct"]) / risk
            qty = min(base, (eq * 0.25) / tr["entry_px"], head / tr["entry_px"])
            if qty * tr["entry_px"] < eq * 0.005:
                continue
            slipped = tr["direction"] * (tr["exit_px"] / tr["entry_px"] - 1)
            open_pos.append({"symbol": tr["symbol"], "qty": qty, "dir": tr["direction"],
                             "mark": tr["entry_px"],
                             "exit_day": pd.Timestamp(tr["exit_time"]).normalize(),
                             "exit_px": tr["exit_px"],
                             "charges": (slipped - tr["net_ret"]) * tr["entry_px"] * qty})
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
    print(f"\n--- {label} OOS book ---")
    for k, v in cs.items():
        print(f"{k:>18}: {v:.4f}" if isinstance(v, float) else f"{k:>18}: {v}")
    print((metrics.per_year_table(equity.pct_change()) * 100).round(2).to_string())
    equity.to_csv(RESULTS / f"g6_nav_{label}.csv")
    return cs


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from services.data_manager import FNO_LOT_SIZES
    fno = sorted(FNO_LOT_SIZES.keys())

    print(f"==== EXP-G6 OOS TOUCH {OOS_START}..{OOS_END} ====\n")
    nifty = loader.load_bars("NIFTY50", "5minute", start="2023-11-01", end=OOS_END)
    print(f"NIFTY 5m last bar: {nifty.index.max()}")

    # ---- A. trade-level locked cells ----
    print("\n=== A. trade-level OOS ===")
    ent = orb_entries(nifty, 12, OOS_START, OOS_END, 0.0025)
    tr_idx1 = run_symbol(nifty, ent, CFG1, symbol="NIFTY50")
    tr_idx3 = run_symbol(nifty, ent, CFG3, symbol="NIFTY50")
    tstats(tr_idx1, "index W12 gap25 @1bp")
    tstats(tr_idx3, "index W12 gap25 @3bp")

    stock_bars = {}
    trs_w6, trs_w12 = [], []
    for sym in fno:
        df = loader.load_bars(sym, "5minute", start="2023-11-01", end=OOS_END)
        if df.empty or df.index.normalize().nunique() < 100:
            continue
        stock_bars[sym] = df
        for w, acc in ((6, trs_w6), (12, trs_w12)):
            t = run_symbol(df, orb_entries(df, w, OOS_START, OOS_END, 0.0025),
                           CFG3, symbol=sym)
            t["risk_pct"] = 0.005
            acc.append(t)
    tr_w6 = pd.concat(trs_w6, ignore_index=True)
    tr_w12 = pd.concat(trs_w12, ignore_index=True)
    tstats(tr_w6, f"stocks W6 gap25 @3bp ({len(stock_bars)} F&O)")
    tstats(tr_w12, f"stocks W12 gap25 @3bp ({len(stock_bars)} F&O)")

    closes = {s: loader.to_daily(b)["close"] for s, b in stock_bars.items()}
    closes["NIFTY50"] = loader.to_daily(nifty)["close"]

    # ---- B. G4 book (locked construction) ----
    tr_idx1b = tr_idx1.copy()
    tr_idx1b["risk_pct"] = 0.010
    g4_tr = pd.concat([tr_idx1b]
                      + [tr_w6[tr_w6["symbol"] == s] for s in MEGA9],
                      ignore_index=True)
    cs_g4 = nav_sim(g4_tr, closes, cap=6, equal_notional=0.0, book_cap=99.0,
                    label="G4_10name")

    # ---- C. breadth v6 book (final construction, informational) ----
    tr_breadth = tr_w12.copy()
    gaps = {}
    for s, b in stock_bars.items():
        d = loader.to_daily(b)
        gaps[s] = d["open"] / d["close"].shift(1) - 1
    ed = pd.to_datetime(tr_breadth["entry_time"]).dt.normalize()
    tr_breadth["gap"] = [gaps[s].get(d, np.nan)
                         for s, d in zip(tr_breadth["symbol"], ed)]
    tr_breadth["eday"] = ed
    rank = tr_breadth.groupby("eday")["gap"].rank(ascending=False, method="first")
    tr_breadth = tr_breadth[rank <= 15]
    allb = pd.concat([tr_idx1b, tr_breadth], ignore_index=True)
    cs_v6 = nav_sim(allb, closes, cap=15, equal_notional=0.10, book_cap=1.5,
                    label="breadth_v6")

    print(f"\nOOS touch complete in {(time.time()-t0)/60:.1f}m — "
          "log this in the OOS ledger. No further OOS looks for these systems.")


if __name__ == "__main__":
    main()
