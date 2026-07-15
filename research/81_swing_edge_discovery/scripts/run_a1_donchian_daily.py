"""EXP-A1: Donchian N-day breakout, 2-4 session hold, F&O daily, IS 2005-2017.

Pre-registration: experiments/A1_donchian_daily/DONCHIAN_BREAKOUT_DAILY_SWEEP_STATUS.md
Grid LOCKED: N in {10,20,55} x dir {long,short} x k_ATR {1.5,2.5} x ts {2,4} = 24 cells.

Resumable: per-(cell,symbol) rows appended to results/a1_cells.csv immediately;
re-run skips completed pairs. Trades appended slim to results/a1_trades.csv.

Run (VPS): cd /home/arun/quantifyd/research/81_swing_edge_discovery &&
           /home/arun/quantifyd/venv/bin/python3 scripts/run_a1_donchian_daily.py
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from datetime import datetime
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

RESULTS = STUDY / "experiments" / "A1_donchian_daily" / "results"
STATUS_MD = STUDY / "experiments" / "A1_donchian_daily" / "DONCHIAN_BREAKOUT_DAILY_SWEEP_STATUS.md"
CELLS_CSV = RESULTS / "a1_cells.csv"
TRADES_CSV = RESULTS / "a1_trades.csv"

IS_START, IS_END = "2005-01-01", "2017-12-31"
WARMUP_START = "2004-06-01"
GRID = [(N, d, k, ts) for N in (10, 20, 55) for d in (1, -1)
        for k in (1.5, 2.5) for ts in (2, 4)]
CA_EXCLUDE_FLAGS = 5      # symbols with more CA-gap flags than this are excluded
CA_SKIP_SESSIONS = 3      # no entries within this many sessions after a flag

CELL_FIELDS = ["cell", "symbol", "n_trades", "exp_gross_pct", "exp_net_pct",
               "win_rate", "pf_net", "avg_hold", "sum_net"]
TRADE_FIELDS = ["cell", "symbol", "entry_time", "direction", "gross_ret",
                "net_ret", "exit_reason", "hold_sessions"]


def cell_label(N, d, k, ts):
    return f"N{N}_{'L' if d == 1 else 'S'}_k{k}_ts{ts}"


def atr14(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def build_entries(df: pd.DataFrame, N: int, d: int, k: float,
                  ca_dates: pd.DatetimeIndex) -> pd.DataFrame:
    a = atr14(df)
    if d == 1:
        ref = df["high"].rolling(N).max().shift(1)
        sigmask = df["close"] > ref
        stop = df["close"] - k * a
    else:
        ref = df["low"].rolling(N).min().shift(1)
        sigmask = df["close"] < ref
        stop = df["close"] + k * a
    sigmask &= a.notna()
    # corporate-action guard: no entry within CA_SKIP_SESSIONS after a flag
    if len(ca_dates):
        bad = pd.Series(False, index=df.index)
        pos = df.index.get_indexer(ca_dates)
        for p in pos[pos >= 0]:
            bad.iloc[p:p + CA_SKIP_SESSIONS + 1] = True
        sigmask &= ~bad
    sig = df.index[sigmask]
    sig = sig[(sig >= pd.Timestamp(IS_START)) & (sig <= pd.Timestamp(IS_END))]
    return pd.DataFrame({"direction": d, "stop": stop.loc[sig],
                         "target": np.nan}, index=sig)


def append_rows(path: Path, fields: list[str], rows: list[dict]):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields)
        if new:
            wcsv.writeheader()
        wcsv.writerows(rows)


def log_status(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    with open(STATUS_MD, "a", encoding="utf-8") as f:
        f.write(f"| {ts} | {msg} | |\n")
    print(msg, flush=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    from services.data_manager import FNO_LOT_SIZES
    universe = sorted(FNO_LOT_SIZES.keys())

    done = set()
    if CELLS_CSV.exists():
        with open(CELLS_CSV) as f:
            done = {(r["cell"], r["symbol"]) for r in csv.DictReader(f)}
        print(f"resuming: {len(done)} (cell,symbol) rows already present")

    cfg_cache = {}
    t0 = time.time()
    excluded, thin = [], []
    for si, sym in enumerate(universe, 1):
        df = loader.load_bars(sym, "day", start=WARMUP_START, end=IS_END)
        if len(df) < 300:
            thin.append(sym)
            continue
        ca = loader.ca_gap_flags(df)
        if len(ca) > CA_EXCLUDE_FLAGS:
            excluded.append(sym)
            continue
        for (N, d, k, ts) in GRID:
            lbl = cell_label(N, d, k, ts)
            if (lbl, sym) in done:
                continue
            if ts not in cfg_cache:
                cfg_cache[ts] = BTConfig(cost=CostConfig(product="FUTURES_PROXY"),
                                         time_stop_sessions=ts)
            entries = build_entries(df, N, d, k, ca)
            tr = run_symbol(df, entries, cfg_cache[ts], symbol=sym)
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
        if si % 10 == 0:
            print(f"[{si}/{len(universe)}] {sym} done "
                  f"({(time.time()-t0)/60:.1f}m elapsed)", flush=True)

    log_status(f"Sweep complete: {len(universe)-len(thin)-len(excluded)} symbols run, "
               f"{len(excluded)} CA-excluded ({','.join(excluded[:8])}...), "
               f"{len(thin)} too thin, {(time.time()-t0)/60:.1f} min")

    # ---------------- aggregation: pooled per-cell stats ----------------
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
    agg.to_csv(RESULTS / "a1_ranking.csv", index=False)
    print("\n=== A1 pooled ranking (net, IS 2005-2017) ===")
    print(agg.to_string(index=False))
    log_status("Aggregation done — ranking in results/a1_ranking.csv")


if __name__ == "__main__":
    main()
