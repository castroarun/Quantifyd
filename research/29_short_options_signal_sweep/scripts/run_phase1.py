"""Phase-1 runner: generate signals + log forward trajectories.

For each (path, variant, symbol, timeframe) cell:
  1. Generate Signal objects for every trading day in [BT_START, BT_END].
  2. For each Signal, walk forward 5-min candles from signal_time to the
     session's last candle and emit:
        - one signal-summary row to phase1_signals.csv
        - per-candle trajectory rows to phase1_trajectories.csv
  3. Resumable: re-running skips (cell_key, date) pairs already in
     phase1_signals.csv.

Smoke-test mode (`--smoke`) runs only Path A on NIFTY50 with one variant
over a one-month window so we can eyeball the output before launching
the full sweep.

Usage:
  python run_phase1.py --smoke
  python run_phase1.py --paths A B C D E F  (full sweep — wired up later)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

# Add this script's dir to path so sibling imports work when run directly
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_loader import (  # noqa: E402
    BT_END,
    BT_START,
    INDEX_SYMBOL,
    INTRADAY_STOCKS,
    load_5min,
    load_daily,
    resample,
    slice_session,
)
from indicators import rsi as rsi_func  # noqa: E402
from signals import (  # noqa: E402
    Signal,
    path_a_signals,
    path_b_signals,
    path_c_signals,
    path_d_signals,
    strategy_e_signals,
    strategy_f_signals,
)

logging.disable(logging.WARNING)


RESULTS_DIR = _HERE.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS_CSV = RESULTS_DIR / "phase1_signals.csv"
TRAJ_CSV = RESULTS_DIR / "phase1_trajectories.csv"

SIGNALS_FIELDS = [
    "signal_id",
    "path", "variant", "symbol", "timeframe",
    "date", "signal_time", "direction", "signal_price", "level_for_T4",
    "gap", "rsi_at_signal", "or_high", "or_low",
    "mae_against", "mae_against_time",
    "mfe_with", "mfe_with_time",
    "net_at_eod", "eod_time",
    "mae_before_mfe",
    "n_candles_forward",
]

TRAJ_FIELDS = [
    "signal_id",
    "t",          # candle close time
    "close",
    "rsi",
    "pts_with",   # signed: + with direction, - against direction
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_csv_header(path: Path, fields: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _load_done_keys() -> set[tuple]:
    """Set of (path, variant, symbol, timeframe, date_iso) already logged."""
    done: set[tuple] = set()
    if not SIGNALS_CSV.exists():
        return done
    with SIGNALS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(
                (
                    row["path"], row["variant"], row["symbol"],
                    row["timeframe"], row["date"][:10],
                )
            )
    return done


def _next_signal_id() -> int:
    """Highest existing signal_id + 1 (0-based start). O(N) per call — fine
    since we only call once per run launch."""
    if not SIGNALS_CSV.exists():
        return 0
    max_id = -1
    with SIGNALS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_id = max(max_id, int(row["signal_id"]))
            except (KeyError, ValueError):
                pass
    return max_id + 1


def _walk_forward_and_log(
    signal: Signal,
    df_5min: pd.DataFrame,
    rsi_full: pd.Series,
    signal_id: int,
    sig_writer: csv.DictWriter,
    sig_fp,
    traj_writer: csv.DictWriter,
    traj_fp,
) -> None:
    """Compute MAE/MFE/EOD on the forward path of one signal and append rows."""
    sess = slice_session(df_5min, signal.date)
    forward = sess.loc[sess.index > signal.signal_time]
    if forward.empty:
        # signal fired on the last candle of the day — no forward path
        sig_writer.writerow(
            {
                "signal_id": signal_id,
                "path": signal.path,
                "variant": signal.variant,
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "date": signal.date.strftime("%Y-%m-%d"),
                "signal_time": signal.signal_time.strftime("%Y-%m-%d %H:%M:%S"),
                "direction": signal.direction,
                "signal_price": signal.signal_price,
                "level_for_T4": signal.level_for_T4,
                "gap": signal.extras.get("gap"),
                "rsi_at_signal": signal.extras.get("rsi"),
                "or_high": signal.extras.get("or_high"),
                "or_low": signal.extras.get("or_low"),
                "mae_against": 0.0,
                "mae_against_time": signal.signal_time.strftime("%Y-%m-%d %H:%M:%S"),
                "mfe_with": 0.0,
                "mfe_with_time": signal.signal_time.strftime("%Y-%m-%d %H:%M:%S"),
                "net_at_eod": 0.0,
                "eod_time": signal.signal_time.strftime("%Y-%m-%d %H:%M:%S"),
                "mae_before_mfe": False,
                "n_candles_forward": 0,
            }
        )
        sig_fp.flush()
        return

    sign = 1.0 if signal.direction == "long" else -1.0

    mae_against = 0.0  # most-negative pts_with seen so far (stored as positive abs)
    mae_against_time = signal.signal_time
    mfe_with = 0.0     # most-positive pts_with seen so far
    mfe_with_time = signal.signal_time
    mae_first_candle: int | None = None
    mfe_first_candle: int | None = None

    for i, (ts, row) in enumerate(forward.iterrows(), start=1):
        close = float(row["close"])
        pts_with = sign * (close - signal.signal_price)
        r = float(rsi_full.loc[ts]) if ts in rsi_full.index else float("nan")

        if pts_with < -mae_against:
            mae_against = -pts_with
            mae_against_time = ts
            if mae_first_candle is None:
                mae_first_candle = i
        if pts_with > mfe_with:
            mfe_with = pts_with
            mfe_with_time = ts
            if mfe_first_candle is None:
                mfe_first_candle = i

        traj_writer.writerow(
            {
                "signal_id": signal_id,
                "t": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "close": close,
                "rsi": r if not np.isnan(r) else "",
                "pts_with": pts_with,
            }
        )

    last_ts = forward.index[-1]
    eod_close = float(forward["close"].iloc[-1])
    net_at_eod = sign * (eod_close - signal.signal_price)

    mae_before_mfe = (
        mae_first_candle is not None
        and (mfe_first_candle is None or mae_first_candle < mfe_first_candle)
    )

    def _safe_round(v, n):
        if v is None:
            return ""
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return ""
        if np.isnan(fv):
            return ""
        return round(fv, n)

    sig_writer.writerow(
        {
            "signal_id": signal_id,
            "path": signal.path,
            "variant": signal.variant,
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "date": signal.date.strftime("%Y-%m-%d"),
            "signal_time": signal.signal_time.strftime("%Y-%m-%d %H:%M:%S"),
            "direction": signal.direction,
            "signal_price": _safe_round(signal.signal_price, 4),
            "level_for_T4": _safe_round(signal.level_for_T4, 4),
            "gap": _safe_round(signal.extras.get("gap"), 6),
            "rsi_at_signal": _safe_round(signal.extras.get("rsi"), 2),
            "or_high": signal.extras.get("or_high"),
            "or_low": signal.extras.get("or_low"),
            "mae_against": _safe_round(mae_against, 4),
            "mae_against_time": mae_against_time.strftime("%Y-%m-%d %H:%M:%S"),
            "mfe_with": _safe_round(mfe_with, 4),
            "mfe_with_time": mfe_with_time.strftime("%Y-%m-%d %H:%M:%S"),
            "net_at_eod": _safe_round(net_at_eod, 4),
            "eod_time": last_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "mae_before_mfe": mae_before_mfe,
            "n_candles_forward": int(len(forward)),
        }
    )
    sig_fp.flush()
    traj_fp.flush()


# ---------------------------------------------------------------------------
# Generic cell runner — dispatches to any signal generator
# ---------------------------------------------------------------------------

def run_cell(
    *,
    sig_gen_callable,
    sig_gen_kwargs: dict,
    df_5min: pd.DataFrame,            # frame fed to signal generator (may be resampled)
    daily: pd.DataFrame,
    sig_writer,
    sig_fp,
    traj_writer,
    traj_fp,
    signal_id_seq: list[int],
    done_keys: set[tuple],
    df_5min_walk: pd.DataFrame | None = None,  # frame used for forward walk (always 5-min)
) -> int:
    """Run one (signal_generator × kwargs) cell.

    `df_5min` is what the signal generator sees. `df_5min_walk` is what's
    used to build the post-signal trajectory; defaults to `df_5min` for
    cells where the signal frame is already 5-min.
    """
    walk_df = df_5min_walk if df_5min_walk is not None else df_5min
    rsi_full = rsi_func(walk_df["close"], 14)
    n_logged = 0
    for sig in sig_gen_callable(df_5min, daily, **sig_gen_kwargs):
        key = (
            sig.path, sig.variant, sig.symbol, sig.timeframe,
            sig.date.strftime("%Y-%m-%d"),
        )
        if key in done_keys:
            continue
        sid = signal_id_seq[0]
        signal_id_seq[0] += 1
        _walk_forward_and_log(
            sig, walk_df, rsi_full, sid,
            sig_writer, sig_fp, traj_writer, traj_fp,
        )
        done_keys.add(key)
        n_logged += 1
    return n_logged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run only Path A on NIFTY for one month")
    parser.add_argument("--paths", nargs="+", default=None,
                        help="Subset of {A,B,C,D,E,F} to run (default: all)")
    parser.add_argument("--start", default=BT_START)
    parser.add_argument("--end", default=BT_END)
    args = parser.parse_args()

    selected_paths = set(args.paths) if args.paths else {"A", "B", "C", "D", "E", "F"}

    # Each cell: (label, callable, kwargs, data_key).
    # `data_key` identifies which (symbol, timeframe) DataFrame to use.
    # The runner looks it up in the `data_registry` built below.
    cells: list[tuple[str, callable, dict, tuple[str, str]]] = []

    if args.smoke:
        start, end = "2024-03-04", "2024-04-30"
        cells = [(
            "A_smoke",
            path_a_signals,
            dict(gap_threshold=0.005, rsi_low=40.0, rsi_high=60.0,
                 symbol=INDEX_SYMBOL, timeframe="5min"),
            (INDEX_SYMBOL, "5min"),
        )]
    else:
        start, end = args.start, args.end
        # --- Path A: gap × RSI grid (5 × 3 = 15 variants) ---
        if "A" in selected_paths:
            for gap in (0.003, 0.005, 0.007, 0.010, None):
                for r_lo, r_hi in ((40, 60), (35, 65), (30, 70)):
                    cells.append((
                        f"A_g{gap}_r{r_lo}{r_hi}",
                        path_a_signals,
                        dict(gap_threshold=gap, rsi_low=float(r_lo),
                             rsi_high=float(r_hi),
                             symbol=INDEX_SYMBOL, timeframe="5min"),
                        (INDEX_SYMBOL, "5min"),
                    ))
        # --- Path B: post-11 RSI zone-entry (3 variants) ---
        if "B" in selected_paths:
            for r_lo, r_hi in ((40, 60), (35, 65), (30, 70)):
                cells.append((
                    f"B_r{r_lo}{r_hi}",
                    path_b_signals,
                    dict(rsi_low=float(r_lo), rsi_high=float(r_hi),
                         symbol=INDEX_SYMBOL, timeframe="5min"),
                    (INDEX_SYMBOL, "5min"),
                ))
        # --- Path C: post-12 day-extreme break × range × RSI gate (5 × 2 = 10) ---
        if "C" in selected_paths:
            for rng in (0.004, 0.006, 0.008, 0.010, None):
                for use_rsi in (False, True):
                    cells.append((
                        f"C_rng{rng}_rsi{int(use_rsi)}",
                        path_c_signals,
                        dict(range_threshold=rng, use_rsi=use_rsi,
                             rsi_low=40.0, rsi_high=60.0,
                             symbol=INDEX_SYMBOL, timeframe="5min"),
                        (INDEX_SYMBOL, "5min"),
                    ))
        # --- Path D: post-12 RSI + CPR (2 conventions × 3 RSI = 6) ---
        if "D" in selected_paths:
            for conv in ("priceCPR", "cprDelta"):
                for r_lo, r_hi in ((40, 60), (35, 65), (30, 70)):
                    cells.append((
                        f"D_{conv}_r{r_lo}{r_hi}",
                        path_d_signals,
                        dict(cpr_convention=conv,
                             rsi_low=float(r_lo), rsi_high=float(r_hi),
                             symbol=INDEX_SYMBOL, timeframe="5min"),
                        (INDEX_SYMBOL, "5min"),
                    ))
        # --- Strategy E: 10 stocks × 3 timeframes × 4 filter modes ---
        if "E" in selected_paths:
            for sym in INTRADAY_STOCKS:
                for tf in ("5min", "10min", "15min"):
                    for mode in ("base", "cpr", "rsi", "cpr_rsi"):
                        cells.append((
                            f"E_{sym}_{tf}_{mode}",
                            strategy_e_signals,
                            dict(filter_mode=mode,
                                 rsi_low=40.0, rsi_high=60.0,
                                 cpr_convention="priceCPR",
                                 symbol=sym, timeframe=tf),
                            (sym, tf),
                        ))
        # --- Strategy F: 10 stocks × 5-min × priceCPR ---
        if "F" in selected_paths:
            for sym in INTRADAY_STOCKS:
                cells.append((
                    f"F_{sym}",
                    strategy_f_signals,
                    dict(cpr_convention="priceCPR",
                         symbol=sym, timeframe="5min"),
                    (sym, "5min"),
                ))

    _ensure_csv_header(SIGNALS_CSV, SIGNALS_FIELDS)
    _ensure_csv_header(TRAJ_CSV, TRAJ_FIELDS)

    done_keys = _load_done_keys()
    signal_id_seq = [_next_signal_id()]

    print(f"Phase 1 launching: {len(cells)} cell(s)  range={start} -> {end}  "
          f"already_done={len(done_keys)}  next_signal_id={signal_id_seq[0]}",
          flush=True)

    # ---- Build data registry for needed (symbol, timeframe) pairs ----
    needed_keys: set[tuple[str, str]] = {dk for *_, dk in cells}
    data_registry: dict[tuple[str, str], pd.DataFrame] = {}
    daily_registry: dict[str, pd.DataFrame] = {}

    needed_symbols = sorted({sym for sym, _ in needed_keys})
    print(f"Loading data for {len(needed_symbols)} symbols...", flush=True)
    for sym in needed_symbols:
        df5 = load_5min(sym, start, end)
        data_registry[(sym, "5min")] = df5
        daily_registry[sym] = load_daily(sym)
        # Resample needs
        for tf in ("10min", "15min"):
            if (sym, tf) in needed_keys:
                data_registry[(sym, tf)] = resample(df5, tf)
        print(f"  {sym:12s} 5min={len(df5):>6}", flush=True)

    with SIGNALS_CSV.open("a", newline="", encoding="utf-8") as sig_fp, \
         TRAJ_CSV.open("a", newline="", encoding="utf-8") as traj_fp:
        sig_w = csv.DictWriter(sig_fp, fieldnames=SIGNALS_FIELDS)
        traj_w = csv.DictWriter(traj_fp, fieldnames=TRAJ_FIELDS)

        # We always walk forward on 5-min data (trajectories are 5-min).
        # For E with 10/15-min signals we still log forward 5-min trajectory
        # because the signal_time aligns to a 5-min boundary by construction.
        for i, (label, sg, kw, dk) in enumerate(cells, 1):
            t0 = _time.time()
            print(f"[{i}/{len(cells)}] {label} ...", end="", flush=True)
            df_for_signal = data_registry[dk]
            df_5min_for_walk = data_registry[(dk[0], "5min")]
            daily = daily_registry[dk[0]]
            try:
                n = run_cell(
                    sig_gen_callable=sg,
                    sig_gen_kwargs=kw,
                    df_5min=df_for_signal,           # used by signal generator
                    daily=daily,
                    sig_writer=sig_w, sig_fp=sig_fp,
                    traj_writer=traj_w, traj_fp=traj_fp,
                    signal_id_seq=signal_id_seq,
                    done_keys=done_keys,
                    df_5min_walk=df_5min_for_walk,    # used for forward walk
                )
            except NotImplementedError as e:
                print(f" SKIPPED ({e})")
                continue
            except Exception as e:
                print(f" ERROR: {type(e).__name__}: {e}")
                continue
            elapsed = _time.time() - t0
            print(f" {n} new signals  ({elapsed:.1f}s)", flush=True)

    print(f"\nDone. signals_csv={SIGNALS_CSV}  traj_csv={TRAJ_CSV}")


if __name__ == "__main__":
    main()
