"""Phase-2 runner: apply 5 exit policies to every Phase-1 signal.

Reads:  results/phase1_signals.csv
        results/phase1_trajectories.csv
Writes: results/phase2_exits.csv  (append-only, incremental, resumable)

For each signal we walk the trajectory and emit one row per
(exit_policy × params) tuple. Columns:

  signal_id, path, variant, symbol, timeframe, direction, signal_price,
  exit_policy, exit_params, exit_time, exit_price, exit_reason, net_pts,
  hold_minutes, sl_hit_first

Exit policies
-------------
T0 — Time only: hold to last available candle
T1_SL{X}  — Time + hard SL of X pts (NIFTY) or X * ATR (stocks-style)
T2_TR{X}  — Time + trail: when +X pts in favour, raise SL to entry; hard SL
            still at -50 pts (NIFTY) by default for safety; backstop at EOD
T3_RSI    — Exit when RSI re-crosses 50 against direction; else EOD
T4_LVL    — Exit when price re-crosses signal's level_for_T4 against
            direction; else EOD

For stocks (E/F) we scale the SL/TR thresholds by an "ATR-equivalent"
proxy: median absolute pts move per 5-min candle over the trailing 30 days
on that symbol — pre-computed once per symbol from phase1_trajectories.

NIFTY thresholds: SL = {30, 50, 75}, TR = {10, 20, 30}.
Stock thresholds: SL_factor = {0.5, 1.0, 1.5}, TR_factor = {0.25, 0.5, 1.0}
applied to per-symbol-per-day median |close-prev_close| (call this `atr_proxy`).
"""

from __future__ import annotations

import csv
import logging
import sys
import time as _time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_loader import INDEX_SYMBOL, INTRADAY_STOCKS, load_5min, BT_START, BT_END  # noqa: E402

logging.disable(logging.WARNING)

RESULTS_DIR = _HERE.parent / "results"
SIGNALS_CSV = RESULTS_DIR / "phase1_signals.csv"
TRAJ_CSV = RESULTS_DIR / "phase1_trajectories.csv"
EXITS_CSV = RESULTS_DIR / "phase2_exits.csv"

EXIT_FIELDS = [
    "signal_id", "path", "variant", "symbol", "timeframe", "direction",
    "signal_price", "level_for_T4",
    "exit_policy", "exit_params",
    "exit_time", "exit_price", "exit_reason",
    "net_pts", "hold_minutes", "sl_hit_first",
]

# Per-instrument SL grids
NIFTY_SL_VALUES = (30.0, 50.0, 75.0)
NIFTY_TR_VALUES = (10.0, 20.0, 30.0)
STOCK_SL_FACTORS = (0.5, 1.0, 1.5)
STOCK_TR_FACTORS = (0.25, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Trajectory index — pre-loaded once
# ---------------------------------------------------------------------------

def load_trajectories() -> dict[int, pd.DataFrame]:
    """Group trajectory rows by signal_id into a per-id DataFrame.
    Trajectory rows are already in chronological order in the CSV.
    """
    print(f"Loading trajectories from {TRAJ_CSV}...", flush=True)
    df = pd.read_csv(TRAJ_CSV)
    df["t"] = pd.to_datetime(df["t"])
    df["close"] = df["close"].astype(float)
    df["pts_with"] = df["pts_with"].astype(float)
    # rsi may be empty string -> NaN
    df["rsi"] = pd.to_numeric(df["rsi"], errors="coerce")
    print(f"  {len(df)} trajectory rows for {df['signal_id'].nunique()} signals", flush=True)
    by_id: dict[int, pd.DataFrame] = {sid: g.sort_values("t").reset_index(drop=True)
                                       for sid, g in df.groupby("signal_id")}
    return by_id


def load_signals() -> pd.DataFrame:
    df = pd.read_csv(SIGNALS_CSV)
    df["signal_id"] = df["signal_id"].astype(int)
    df["signal_price"] = df["signal_price"].astype(float)
    df["level_for_T4"] = pd.to_numeric(df["level_for_T4"], errors="coerce")
    df["signal_time"] = pd.to_datetime(df["signal_time"])
    return df


# ---------------------------------------------------------------------------
# ATR-style scaling proxy per stock (for E/F SL/TR scaling)
# ---------------------------------------------------------------------------

def compute_stock_atr_proxy() -> dict[str, float]:
    """Per-symbol approx ATR: median absolute 5-min close-to-close pts
    move over the backtest period. Used to scale SL/TR for stocks.
    """
    proxy: dict[str, float] = {}
    for sym in INTRADAY_STOCKS:
        df = load_5min(sym, BT_START, BT_END)
        if df.empty:
            proxy[sym] = 1.0
            continue
        moves = df["close"].diff().abs().dropna()
        # use 75th percentile of |move| as a stable per-bar volatility proxy
        proxy[sym] = float(moves.quantile(0.75))
    print(f"Stock ATR proxy (75th-pct |close diff| over 5-min): "
          f"{ {k: round(v, 2) for k, v in proxy.items()} }", flush=True)
    return proxy


# ---------------------------------------------------------------------------
# Exit-policy logic
# ---------------------------------------------------------------------------

def _eod(traj: pd.DataFrame) -> tuple[pd.Timestamp, float, float]:
    """(time, close, pts_with) at last candle in trajectory."""
    last = traj.iloc[-1]
    return last["t"], float(last["close"]), float(last["pts_with"])


def exit_T0(traj: pd.DataFrame) -> dict:
    t, p, pts = _eod(traj)
    return dict(exit_time=t, exit_price=p, exit_reason="EOD",
                net_pts=pts, sl_hit_first=False)


def exit_T1(traj: pd.DataFrame, sl_pts: float) -> dict:
    """Hard stop at -sl_pts. If hit, exit at that candle's close.
    Hold to EOD otherwise.
    """
    for _, row in traj.iterrows():
        pts = float(row["pts_with"])
        if pts <= -sl_pts:
            return dict(exit_time=row["t"], exit_price=float(row["close"]),
                        exit_reason="SL", net_pts=pts, sl_hit_first=True)
    t, p, pts = _eod(traj)
    return dict(exit_time=t, exit_price=p, exit_reason="EOD",
                net_pts=pts, sl_hit_first=False)


def exit_T2(traj: pd.DataFrame, tr_pts: float, hard_sl_pts: float) -> dict:
    """Trail to entry once +tr_pts achieved. Hard SL still active.
    If trailing-SL is hit (pts < 0 after activation), exit.
    """
    activated = False
    sl_first_hit = False
    for _, row in traj.iterrows():
        pts = float(row["pts_with"])
        if not activated and pts >= tr_pts:
            activated = True
        # Hard stop
        if pts <= -hard_sl_pts:
            return dict(exit_time=row["t"], exit_price=float(row["close"]),
                        exit_reason="HARD_SL", net_pts=pts, sl_hit_first=True)
        # Trail-to-entry
        if activated and pts <= 0:
            return dict(exit_time=row["t"], exit_price=float(row["close"]),
                        exit_reason="TRAIL", net_pts=pts, sl_hit_first=False)
    t, p, pts = _eod(traj)
    return dict(exit_time=t, exit_price=p, exit_reason="EOD",
                net_pts=pts, sl_hit_first=False)


def exit_T3(traj: pd.DataFrame, direction: str) -> dict:
    """RSI-reversal: exit when RSI re-crosses 50 against signal direction.

    For longs: exit when RSI dips below 50 AFTER having been >= 50 at signal.
    For shorts: exit when RSI rises above 50 AFTER having been <= 50.

    If RSI never crosses, hold to EOD.
    """
    sign_long = direction == "long"
    saw_aligned = False  # RSI on correct side at least once after entry
    for _, row in traj.iterrows():
        r = row["rsi"]
        if pd.isna(r):
            continue
        rv = float(r)
        if sign_long:
            if rv >= 50:
                saw_aligned = True
            elif saw_aligned and rv < 50:
                return dict(exit_time=row["t"], exit_price=float(row["close"]),
                            exit_reason="RSI_REV", net_pts=float(row["pts_with"]),
                            sl_hit_first=float(row["pts_with"]) < 0)
        else:
            if rv <= 50:
                saw_aligned = True
            elif saw_aligned and rv > 50:
                return dict(exit_time=row["t"], exit_price=float(row["close"]),
                            exit_reason="RSI_REV", net_pts=float(row["pts_with"]),
                            sl_hit_first=float(row["pts_with"]) < 0)
    t, p, pts = _eod(traj)
    return dict(exit_time=t, exit_price=p, exit_reason="EOD",
                net_pts=pts, sl_hit_first=False)


def exit_T4(traj: pd.DataFrame, direction: str, level: float) -> dict:
    """Level-reversal: exit when CLOSE crosses back through `level`
    against direction.

    Long: exit when close < level
    Short: exit when close > level
    """
    if pd.isna(level):
        # No reference level — fall back to EOD
        t, p, pts = _eod(traj)
        return dict(exit_time=t, exit_price=p, exit_reason="EOD_no_level",
                    net_pts=pts, sl_hit_first=False)
    sign_long = direction == "long"
    for _, row in traj.iterrows():
        c = float(row["close"])
        if (sign_long and c < level) or (not sign_long and c > level):
            return dict(exit_time=row["t"], exit_price=c,
                        exit_reason="LVL_REV", net_pts=float(row["pts_with"]),
                        sl_hit_first=float(row["pts_with"]) < 0)
    t, p, pts = _eod(traj)
    return dict(exit_time=t, exit_price=p, exit_reason="EOD",
                net_pts=pts, sl_hit_first=False)


# ---------------------------------------------------------------------------
# Resume / dedupe
# ---------------------------------------------------------------------------

def _ensure_csv_header(path: Path, fields: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _load_done_keys() -> set[tuple]:
    """(signal_id, exit_policy, exit_params) already logged."""
    done: set[tuple] = set()
    if not EXITS_CSV.exists():
        return done
    with EXITS_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((int(row["signal_id"]), row["exit_policy"], row["exit_params"]))
    return done


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    _ensure_csv_header(EXITS_CSV, EXIT_FIELDS)
    done_keys = _load_done_keys()

    sig_df = load_signals()
    traj_by_id = load_trajectories()
    atr_proxy = compute_stock_atr_proxy()

    print(f"\nPhase 2 launching: {len(sig_df)} signals  "
          f"already_done_rows={len(done_keys)}", flush=True)

    written = 0
    skipped = 0
    last_print = _time.time()
    t_start = _time.time()

    with EXITS_CSV.open("a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=EXIT_FIELDS)
        for i, sig in sig_df.iterrows():
            sid = int(sig["signal_id"])
            traj = traj_by_id.get(sid)
            if traj is None or traj.empty:
                continue

            symbol = sig["symbol"]
            is_index = symbol == INDEX_SYMBOL
            sl_grid = NIFTY_SL_VALUES if is_index else tuple(
                round(f * atr_proxy.get(symbol, 1.0), 2) for f in STOCK_SL_FACTORS
            )
            tr_grid = NIFTY_TR_VALUES if is_index else tuple(
                round(f * atr_proxy.get(symbol, 1.0), 2) for f in STOCK_TR_FACTORS
            )
            hard_sl_for_T2 = sl_grid[1]  # use the middle SL value as the hard floor for T2

            base = {
                "signal_id": sid,
                "path": sig["path"],
                "variant": sig["variant"],
                "symbol": symbol,
                "timeframe": sig["timeframe"],
                "direction": sig["direction"],
                "signal_price": sig["signal_price"],
                "level_for_T4": sig["level_for_T4"],
            }
            sig_time = pd.Timestamp(sig["signal_time"])

            policy_outcomes = []

            # T0
            policy_outcomes.append(("T0", "time_only", exit_T0(traj)))
            # T1 — three SL values
            for sl_pts in sl_grid:
                tag = f"SL{sl_pts}"
                policy_outcomes.append((f"T1_{tag}", f"sl_pts={sl_pts}", exit_T1(traj, sl_pts)))
            # T2 — three TR values
            for tr_pts in tr_grid:
                tag = f"TR{tr_pts}"
                policy_outcomes.append((f"T2_{tag}", f"tr_pts={tr_pts};hard_sl={hard_sl_for_T2}",
                                        exit_T2(traj, tr_pts, hard_sl_for_T2)))
            # T3
            policy_outcomes.append(("T3_RSI", "rsi_cross_50", exit_T3(traj, sig["direction"])))
            # T4
            policy_outcomes.append(("T4_LVL", "level_for_T4_cross",
                                    exit_T4(traj, sig["direction"], float(sig["level_for_T4"]))))

            for policy, params, out in policy_outcomes:
                key = (sid, policy, params)
                if key in done_keys:
                    skipped += 1
                    continue
                exit_t = out["exit_time"]
                hold_min = (pd.Timestamp(exit_t) - sig_time).total_seconds() / 60.0
                w.writerow({
                    **base,
                    "exit_policy": policy,
                    "exit_params": params,
                    "exit_time": pd.Timestamp(exit_t).strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_price": round(float(out["exit_price"]), 4),
                    "exit_reason": out["exit_reason"],
                    "net_pts": round(float(out["net_pts"]), 4),
                    "hold_minutes": round(hold_min, 1),
                    "sl_hit_first": bool(out["sl_hit_first"]),
                })
                done_keys.add(key)
                written += 1
            fp.flush()

            if _time.time() - last_print > 5.0:
                pct = (i + 1) / len(sig_df) * 100
                print(f"  [{i+1}/{len(sig_df)} {pct:.1f}%]  written={written}  "
                      f"skipped={skipped}  elapsed={_time.time()-t_start:.0f}s",
                      flush=True)
                last_print = _time.time()

    print(f"\nDone. exits_csv={EXITS_CSV}  written={written}  skipped={skipped}  "
          f"elapsed={_time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
