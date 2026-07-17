"""Shared G1 screening harness for research/81 family experiments.

Each experiment script supplies: EXP id, GRID, cell_label(), build_entries(df, cell, ca)
and calls run_screen(...). Everything else (universe, resume, incremental CSV,
pooled aggregation incl. per-year split) is standardized here so all families
are compared apples-to-apples.
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
for p in (str(STUDY), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader                              # noqa: E402
from engine.backtester import BTConfig, run_symbol     # noqa: E402
from engine.costs import CostConfig                    # noqa: E402

logging.disable(logging.WARNING)

IS_START, IS_END = "2005-01-01", "2017-12-31"
WARMUP_START = "2004-01-01"
CA_EXCLUDE_FLAGS = 5
CA_SKIP_SESSIONS = 3

CELL_FIELDS = ["cell", "symbol", "n_trades", "exp_gross_pct", "exp_net_pct",
               "win_rate", "pf_net", "avg_hold", "sum_net"]
TRADE_FIELDS = ["cell", "symbol", "entry_time", "direction", "gross_ret",
                "net_ret", "exit_reason", "hold_sessions"]


def atr14(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def ca_mask(df: pd.DataFrame, ca_dates) -> pd.Series:
    """True where entries are allowed (not within CA_SKIP_SESSIONS after a flag)."""
    good = pd.Series(True, index=df.index)
    if len(ca_dates):
        pos = df.index.get_indexer(ca_dates)
        for p in pos[pos >= 0]:
            good.iloc[p:p + CA_SKIP_SESSIONS + 1] = False
    return good


def clip_is(sig: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return sig[(sig >= pd.Timestamp(IS_START)) & (sig <= pd.Timestamp(IS_END))]


def _append(path: Path, fields, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def run_screen(exp_id: str, results_dir: Path, grid: list, cell_label,
               build_entries, universe: list[str] | None = None,
               timestop_of=lambda cell: cell[-1]) -> pd.DataFrame:
    """Standard G1 loop: per symbol x cell, resumable, incremental, aggregated."""
    results_dir.mkdir(parents=True, exist_ok=True)
    cells_csv = results_dir / f"{exp_id}_cells.csv"
    trades_csv = results_dir / f"{exp_id}_trades.csv"

    if universe is None:
        from services.data_manager import FNO_LOT_SIZES
        universe = sorted(FNO_LOT_SIZES.keys())

    done = set()
    if cells_csv.exists():
        with open(cells_csv) as f:
            done = {(r["cell"], r["symbol"]) for r in csv.DictReader(f)}
        print(f"resuming: {len(done)} rows present")

    from engine.backtester import MAX_TIME_STOP
    cfg = {ts: BTConfig(cost=CostConfig(product="FUTURES_PROXY"),
                        time_stop_sessions=ts)
           for ts in range(1, MAX_TIME_STOP + 1)}
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
        allowed = ca_mask(df, ca)
        for cell in grid:
            lbl = cell_label(cell)
            if (lbl, sym) in done:
                continue
            entries = build_entries(df, cell, allowed)
            tr = run_symbol(df, entries, cfg[timestop_of(cell)], symbol=sym)
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
                _append(trades_csv, TRADE_FIELDS, [
                    {"cell": lbl, "symbol": sym, "entry_time": r.entry_time,
                     "direction": r.direction, "gross_ret": round(r.gross_ret, 6),
                     "net_ret": round(r.net_ret, 6), "exit_reason": r.exit_reason,
                     "hold_sessions": r.hold_sessions}
                    for r in tr.itertuples()])
            else:
                row = {"cell": lbl, "symbol": sym, "n_trades": 0,
                       "exp_gross_pct": 0, "exp_net_pct": 0, "win_rate": 0,
                       "pf_net": 0, "avg_hold": 0, "sum_net": 0}
            _append(cells_csv, CELL_FIELDS, [row])
        if si % 20 == 0:
            print(f"[{si}/{len(universe)}] ({(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"{exp_id} sweep done: excluded={excluded} thin={thin} "
          f"({(time.time()-t0)/60:.1f}m)", flush=True)
    return aggregate(exp_id, results_dir)


def aggregate(exp_id: str, results_dir: Path) -> pd.DataFrame:
    trades_csv = results_dir / f"{exp_id}_trades.csv"
    if not trades_csv.exists():
        print(f"{exp_id}: NO TRADES AT ALL")
        return pd.DataFrame()
    trades = pd.read_csv(trades_csv)
    out = []
    for lbl, g in trades.groupby("cell"):
        r = g["net_ret"].to_numpy(float)
        gr = g["gross_ret"].to_numpy(float)
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan
        sym_pnl = g.groupby("symbol")["net_ret"].sum()
        yr = pd.to_datetime(g["entry_time"]).dt.year
        yr_pnl = g.groupby(yr)["net_ret"].mean()
        out.append({"cell": lbl, "n": len(r),
                    "gross_bps": round(gr.mean() * 1e4, 2),
                    "net_bps": round(r.mean() * 1e4, 2),
                    "t_net": round(t, 2),
                    "win": round((r > 0).mean(), 3),
                    "pct_sym_pos": round((sym_pnl > 0).mean(), 3),
                    "pct_yr_pos": round((yr_pnl > 0).mean(), 3),
                    "n_sym": sym_pnl.size})
    agg = pd.DataFrame(out).sort_values("t_net", ascending=False)
    agg.to_csv(results_dir / f"{exp_id}_ranking.csv", index=False)
    print(f"\n=== {exp_id} pooled ranking (net, IS {IS_START}..{IS_END}) ===")
    print(agg.to_string(index=False), flush=True)
    return agg


def per_year_detail(exp_id: str, results_dir: Path, cell: str) -> pd.DataFrame:
    trades = pd.read_csv(results_dir / f"{exp_id}_trades.csv")
    g = trades[trades["cell"] == cell]
    yr = pd.to_datetime(g["entry_time"]).dt.year
    tab = g.groupby(yr).agg(n=("net_ret", "size"),
                            net_bps=("net_ret", lambda x: round(x.mean() * 1e4, 1)),
                            win=("net_ret", lambda x: round((x > 0).mean(), 3)))
    print(f"\n--- {exp_id} {cell} per-year ---")
    print(tab.to_string(), flush=True)
    return tab
