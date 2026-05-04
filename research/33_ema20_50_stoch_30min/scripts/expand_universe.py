"""Phase A: expand the top 5 cells from D+E to the full 79-stock 5-min universe.

Period: 2024-03-18 -> 2026-03-12 (~24 months, lower bound of common window).
Cost: 0.10% per trade (round-trip).

Top cells (from param_sweep_results.csv, ranked by net Sharpe):
  1. long stoch=(14,5,3) os=35 exit=X9_OR_X4   — Sharpe 1.60, DD 22.6%, TR +111.4%
  2. long stoch=(14,5,3) os=40 exit=X9_OR_X4   — Sharpe 1.40, DD 27.2%, TR +114.2%
  3. long stoch=(14,5,3) os=30 exit=X9_OR_X4   — Sharpe 1.39, DD 21.2%, TR  +69.3%
  4. long stoch=(14,3,3) os=30 exit=X9_OR_X4   — Sharpe 1.37, DD 22.1%, TR  +80.9%
  5. long stoch=(14,5,3) os=35 exit=X9_22_3.0  — Sharpe 1.32, DD 36.2%, TR +122.5%

Output:
  results/universe_expand_results.csv — per-cell × universe metrics
"""

from __future__ import annotations

import csv
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data30 import to_30min  # noqa: E402
from indicators_stoch import ema, stochastics, rsi, atr, supertrend  # noqa: E402
from param_sweep import gen_setups, sim, cell_metrics, COST_PCT  # noqa: E402

OUT_DIR = SCRIPT_DIR.parent / "results"
LOG_DIR = SCRIPT_DIR.parent / "logs"
DB_PATH = SCRIPT_DIR.parent.parent.parent / "backtest_data" / "market_data.db"

UNIVERSE_START = "2024-03-18"
UNIVERSE_END = "2026-03-12"

TOP_CELLS = [
    {"side": "long", "stoch": (14, 5, 3), "os": 35.0, "exit": "X9_OR_X4"},
    {"side": "long", "stoch": (14, 5, 3), "os": 40.0, "exit": "X9_OR_X4"},
    {"side": "long", "stoch": (14, 5, 3), "os": 30.0, "exit": "X9_OR_X4"},
    {"side": "long", "stoch": (14, 3, 3), "os": 30.0, "exit": "X9_OR_X4"},
    {"side": "long", "stoch": (14, 5, 3), "os": 35.0, "exit": "X9_22_3.0"},
]


def list_universe() -> list[str]:
    """Return all symbols with 5-min data covering the universe period."""
    con = sqlite3.connect(str(DB_PATH))
    sql = """
        SELECT symbol, MIN(date) as first_dt, MAX(date) as last_dt
        FROM market_data_unified
        WHERE timeframe='5minute'
        GROUP BY symbol
        HAVING first_dt <= ? AND last_dt >= ?
        ORDER BY symbol
    """
    df = pd.read_sql(sql, con, params=(UNIVERSE_START + " 23:59:59",
                                       UNIVERSE_END + " 00:00:00"))
    con.close()
    syms = df["symbol"].tolist()
    # Drop indices (NIFTY50, BANKNIFTY) — equity strategy, indices have 0 volume
    syms = [s for s in syms if s not in ("NIFTY50", "BANKNIFTY")]
    return syms


def load_30min_period(symbol: str) -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    sql = """
        SELECT date, open, high, low, close, volume
        FROM market_data_unified
        WHERE symbol = ? AND timeframe = '5minute'
          AND date >= ? AND date <= ?
        ORDER BY date
    """
    df5 = pd.read_sql(sql, con, params=(symbol, UNIVERSE_START,
                                        UNIVERSE_END + " 23:59:59"))
    con.close()
    if df5.empty:
        return df5
    df5["date"] = pd.to_datetime(df5["date"])
    df5 = df5.set_index("date")
    return to_30min(df5)


def prepare(symbol: str) -> pd.DataFrame:
    df = load_30min_period(symbol)
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
    st7 = supertrend(df["high"], df["low"], df["close"], 7, 3.0)
    df["st7_3_dir"] = st7["dir"]
    st10 = supertrend(df["high"], df["low"], df["close"], 10, 3.0)
    df["st10_3_dir"] = st10["dir"]
    df["hh22"] = df["high"].rolling(22, min_periods=22).max()
    df["ll22"] = df["low"].rolling(22, min_periods=22).min()
    df["hh15"] = df["high"].rolling(15, min_periods=15).max()
    df["ll15"] = df["low"].rolling(15, min_periods=15).min()
    return df.reset_index().rename(columns={"date": "dt"})


def main():
    log_path = LOG_DIR / "universe_expand.log"
    log = log_path.open("w")
    def say(msg: str):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    universe = list_universe()
    say(f"=== Phase A: 79-stock universe expansion ===")
    say(f"Period: {UNIVERSE_START} -> {UNIVERSE_END}")
    say(f"Universe size: {len(universe)} symbols")
    say(f"Top cells to test: {len(TOP_CELLS)}")
    say(f"Cost: {COST_PCT}% per trade")
    say("")

    # Pre-prepare per symbol (~1 min for 79)
    say("Loading + preparing all symbols...")
    t0 = time.time()
    prepped: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(universe, 1):
        df = prepare(sym)
        if not df.empty:
            prepped[sym] = df
        if i % 10 == 0:
            say(f"  {i}/{len(universe)}  ({time.time()-t0:.1f}s)")
    say(f"Prepared {len(prepped)} symbols in {time.time()-t0:.1f}s")
    say("")

    out_csv = OUT_DIR / "universe_expand_results.csv"
    fieldnames = [
        "side", "stoch_k", "stoch_sk", "stoch_sd", "stoch_os", "exit_id",
        "n_symbols", "trades", "win_rate", "profit_factor", "sharpe_ann",
        "max_dd_pct", "total_ret_pct", "mean_ret_pct", "avg_hold",
    ]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    # Also dump per-symbol breakdown for the top cell
    per_sym_csv = OUT_DIR / "universe_expand_per_symbol_top.csv"
    per_sym_rows = []

    for ci, cell in enumerate(TOP_CELLS, 1):
        sk, ssk, ssd = cell["stoch"]
        side = cell["side"]; os_ = cell["os"]; exit_id = cell["exit"]
        say(f"--- Cell {ci}/{len(TOP_CELLS)}: side={side} stoch=({sk},{ssk},{ssd}) os={os_:.0f} exit={exit_id} ---")
        t_cell = time.time()
        rets, holds = [], []
        sym_metrics = {}
        for sym, df in prepped.items():
            ss = gen_setups(df, side, sk, ssk, ssd, os_)
            sym_rets, sym_holds = [], []
            for s in ss:
                # Use s["df_ref"] — the post-dropna+reset_index df that
                # the setup's fill_idx is aligned to. The outer `df` has
                # an offset due to NaN warmup rows being dropped.
                _, _, _, held, ret = sim(
                    s["df_ref"], side, s["fill_idx"], s["fill_price"],
                    s["anchor_low"], s["anchor_high"], exit_id,
                )
                sym_rets.append(ret - COST_PCT); sym_holds.append(held)
            rets.extend(sym_rets); holds.extend(sym_holds)
            if sym_rets:
                sym_metrics[sym] = cell_metrics(sym_rets, sym_holds)

        m = cell_metrics(rets, holds)
        n_syms = len(sym_metrics)
        say(f"  trades={m.get('trades',0)}  syms_with_trades={n_syms}  "
            f"WR={m.get('win_rate','-')}%  PF={m.get('profit_factor','-')}  "
            f"Sharpe={m.get('sharpe_ann','-')}  DD={m.get('max_dd_pct','-')}%  "
            f"TR={m.get('total_ret_pct','-')}%  ({time.time()-t_cell:.1f}s)")

        row = {"side": side, "stoch_k": sk, "stoch_sk": ssk, "stoch_sd": ssd,
               "stoch_os": os_, "exit_id": exit_id, "n_symbols": n_syms, **m}
        for k in fieldnames:
            if k not in row: row[k] = ""
        with out_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        # Per-symbol breakdown for top cell only
        if ci == 1:
            for sym, mm in sym_metrics.items():
                per_sym_rows.append({
                    "symbol": sym,
                    "trades": mm.get("trades", 0),
                    "win_rate": mm.get("win_rate", 0),
                    "profit_factor": mm.get("profit_factor", 0),
                    "sharpe_ann": mm.get("sharpe_ann", 0),
                    "max_dd_pct": mm.get("max_dd_pct", 0),
                    "total_ret_pct": mm.get("total_ret_pct", 0),
                })

    if per_sym_rows:
        psdf = pd.DataFrame(per_sym_rows).sort_values("sharpe_ann", ascending=False)
        psdf.to_csv(per_sym_csv, index=False)
        say("")
        say(f"Per-symbol breakdown for TOP cell wrote to {per_sym_csv}")
        say("")
        say("=== Top 15 symbols by per-symbol Sharpe (top cell) ===")
        say(psdf.head(15).to_string(index=False))
        say("")
        say("=== Bottom 10 symbols by per-symbol Sharpe (top cell) ===")
        say(psdf.tail(10).to_string(index=False))

    say(f"\nFinal results -> {out_csv}")
    log.close()


if __name__ == "__main__":
    main()
