"""EXP-B1: z-score mean reversion, 2-4 session hold, F&O daily, IS 2005-2017.

Pre-registration: experiments/B1_zscore_daily/ZSCORE_REVERSION_DAILY_SWEEP_STATUS.md
Grid LOCKED: z_thr {2.0,2.5} x dir {L,S} x target {none,sma20} x ts {2,4} = 16 cells.
Stop FIXED at 2.5 x ATR14. Long only above SMA200; short only below (trend prior).

Run (VPS): cd /home/arun/quantifyd/research/81_swing_edge_discovery &&
           /home/arun/quantifyd/venv/bin/python3 scripts/run_b1_zscore_daily.py
"""
from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(ROOT))

from engine import loader                              # noqa: E402
from engine.backtester import BTConfig, run_symbol     # noqa: E402
from engine.costs import CostConfig                    # noqa: E402

logging.disable(logging.WARNING)

RESULTS = STUDY / "experiments" / "B1_zscore_daily" / "results"
CELLS_CSV = RESULTS / "b1_cells.csv"
TRADES_CSV = RESULTS / "b1_trades.csv"

IS_START, IS_END = "2005-01-01", "2017-12-31"
WARMUP_START = "2004-01-01"          # SMA200 warmup
K_ATR = 2.5
GRID = [(z, d, tgt, ts) for z in (2.0, 2.5) for d in (1, -1)
        for tgt in ("none", "sma20") for ts in (2, 4)]
CA_EXCLUDE_FLAGS = 5
CA_SKIP_SESSIONS = 3

CELL_FIELDS = ["cell", "symbol", "n_trades", "exp_gross_pct", "exp_net_pct",
               "win_rate", "pf_net", "avg_hold", "sum_net"]
TRADE_FIELDS = ["cell", "symbol", "entry_time", "direction", "gross_ret",
                "net_ret", "exit_reason", "hold_sessions"]


def cell_label(z, d, tgt, ts):
    return f"z{z}_{'L' if d == 1 else 'S'}_{tgt}_ts{ts}"


def atr14(df):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def build_entries(df, z_thr, d, tgt, ca_dates):
    sma20 = df["close"].rolling(20).mean()
    sd20 = df["close"].rolling(20).std()
    sma200 = df["close"].rolling(200).mean()
    a = atr14(df)
    z = (df["close"] - sma20) / sd20
    if d == 1:
        sigmask = (z <= -z_thr) & (df["close"] > sma200)
        stop = df["close"] - K_ATR * a
    else:
        sigmask = (z >= z_thr) & (df["close"] < sma200)
        stop = df["close"] + K_ATR * a
    sigmask &= a.notna() & sma200.notna() & (sd20 > 0)
    if len(ca_dates):
        bad = pd.Series(False, index=df.index)
        pos = df.index.get_indexer(ca_dates)
        for p in pos[pos >= 0]:
            bad.iloc[p:p + CA_SKIP_SESSIONS + 1] = True
        sigmask &= ~bad
    sig = df.index[sigmask]
    sig = sig[(sig >= pd.Timestamp(IS_START)) & (sig <= pd.Timestamp(IS_END))]
    target = sma20.loc[sig] if tgt == "sma20" else pd.Series(np.nan, index=sig)
    return pd.DataFrame({"direction": d, "stop": stop.loc[sig],
                         "target": target}, index=sig)


def append_rows(path, fields, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields)
        if new:
            wcsv.writeheader()
        wcsv.writerows(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    from services.data_manager import FNO_LOT_SIZES
    universe = sorted(FNO_LOT_SIZES.keys())

    done = set()
    if CELLS_CSV.exists():
        with open(CELLS_CSV) as f:
            done = {(r["cell"], r["symbol"]) for r in csv.DictReader(f)}
        print(f"resuming: {len(done)} rows present")

    cfg = {ts: BTConfig(cost=CostConfig(product="FUTURES_PROXY"),
                        time_stop_sessions=ts) for ts in (2, 4)}
    t0 = time.time()
    excluded = thin = 0
    for si, sym in enumerate(universe, 1):
        df = loader.load_bars(sym, "day", start=WARMUP_START, end=IS_END)
        if len(df) < 300:
            thin += 1
            continue
        ca = loader.ca_gap_flags(df)
        if len(ca) > CA_EXCLUDE_FLAGS:
            excluded += 1
            continue
        for (z, d, tgt, ts) in GRID:
            lbl = cell_label(z, d, tgt, ts)
            if (lbl, sym) in done:
                continue
            tr = run_symbol(df, build_entries(df, z, d, tgt, ca), cfg[ts], symbol=sym)
            if len(tr):
                wins = tr["net_ret"] > 0
                lossum = tr.loc[~wins, "net_ret"].sum()
                row = {"cell": lbl, "symbol": sym, "n_trades": len(tr),
                       "exp_gross_pct": round(tr["gross_ret"].mean() * 100, 4),
                       "exp_net_pct": round(tr["net_ret"].mean() * 100, 4),
                       "win_rate": round(wins.mean(), 4),
                       "pf_net": round(tr.loc[wins, "net_ret"].sum()
                                       / abs(lossum), 3) if lossum else np.inf,
                       "avg_hold": round(tr["hold_sessions"].mean(), 2),
                       "sum_net": round(tr["net_ret"].sum(), 4)}
                append_rows(TRADES_CSV, TRADE_FIELDS, [
                    {"cell": lbl, "symbol": sym, "entry_time": r.entry_time,
                     "direction": r.direction, "gross_ret": round(r.gross_ret, 6),
                     "net_ret": round(r.net_ret, 6), "exit_reason": r.exit_reason,
                     "hold_sessions": r.hold_sessions}
                    for r in tr.itertuples()])
            else:
                row = {"cell": lbl, "symbol": sym, "n_trades": 0,
                       "exp_gross_pct": 0, "exp_net_pct": 0, "win_rate": 0,
                       "pf_net": 0, "avg_hold": 0, "sum_net": 0}
            append_rows(CELLS_CSV, CELL_FIELDS, [row])
        if si % 20 == 0:
            print(f"[{si}/{len(universe)}] ({(time.time()-t0)/60:.1f}m)", flush=True)

    print(f"sweep done: excluded={excluded} thin={thin} "
          f"({(time.time()-t0)/60:.1f}m)")

    trades = pd.read_csv(TRADES_CSV)
    out = []
    for lbl, g in trades.groupby("cell"):
        r = g["net_ret"].to_numpy(float)
        gr = g["gross_ret"].to_numpy(float)
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan
        sym_pnl = g.groupby("symbol")["net_ret"].sum()
        out.append({"cell": lbl, "n": len(r),
                    "gross_bps": round(gr.mean() * 1e4, 2),
                    "net_bps": round(r.mean() * 1e4, 2),
                    "t_net": round(t, 2),
                    "win": round((r > 0).mean(), 3),
                    "pct_sym_pos": round((sym_pnl > 0).mean(), 3),
                    "n_sym": sym_pnl.size})
    agg = pd.DataFrame(out).sort_values("t_net", ascending=False)
    agg.to_csv(RESULTS / "b1_ranking.csv", index=False)
    print("\n=== B1 pooled ranking (net, IS 2005-2017) ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
