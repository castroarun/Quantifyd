"""Phase A: MST sweep on NIFTY50 + BANKNIFTY across 15/30/60-min.

Computes SuperTrend for 252 cells (2 indices × 3 TFs × 6 ATR periods × 7 multipliers),
scores each on stickiness composite, writes ranking CSV.
"""
from __future__ import annotations
import os
import sys
import sqlite3
import time
import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
sys.path.insert(0, str(SCRIPT_DIR))

from supertrend import supertrend, resample_5min_to  # noqa

DB = ROOT / "backtest_data" / "market_data.db"
INDICES = ["NIFTY50", "BANKNIFTY"]
TIMEFRAMES = ["15min", "30min", "60min"]
ATR_PERIODS = [7, 10, 14, 21, 30, 50]
MULTIPLIERS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def load_5min(symbol: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date",
        con, params=(symbol,), parse_dates=["date"],
    )
    con.close()
    return df.set_index("date")


def load_60min(symbol: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='60minute' ORDER BY date",
        con, params=(symbol,), parse_dates=["date"],
    )
    con.close()
    return df.set_index("date")


def load_stored_tf(symbol: str, kite_tf: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe=? ORDER BY date",
        con, params=(symbol, kite_tf), parse_dates=["date"],
    )
    con.close()
    return df.set_index("date")


def get_ohlc(symbol: str, tf: str) -> pd.DataFrame:
    if tf == "60min":
        df = load_60min(symbol)
        if not df.empty:
            return df
    elif tf == "30min":
        # Prefer stored 30minute if present (extended NIFTY history from 2020)
        df = load_stored_tf(symbol, "30minute")
        if not df.empty:
            return df
    df5 = load_5min(symbol)
    return resample_5min_to(df5, tf)


def compute_segments(df: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    """Slice into trend segments. Returns segment-level df."""
    n = len(df)
    valid_start = np.where(~np.isnan(direction))[0]
    if len(valid_start) == 0:
        return pd.DataFrame()
    s = valid_start[0]
    segs = []
    cur_dir = int(direction[s])
    seg_start = s
    seg_high = df["high"].iloc[s]
    seg_low = df["low"].iloc[s]
    for i in range(s + 1, n):
        d = int(direction[i])
        seg_high = max(seg_high, df["high"].iloc[i])
        seg_low = min(seg_low, df["low"].iloc[i])
        if d != cur_dir:
            segs.append({
                "start_idx": seg_start, "end_idx": i - 1,
                "start_dt": df.index[seg_start], "end_dt": df.index[i - 1],
                "direction": cur_dir,
                "entry_close": df["close"].iloc[seg_start],
                "exit_close": df["close"].iloc[i - 1],
                "seg_high": seg_high, "seg_low": seg_low,
                "bars": i - seg_start,
            })
            cur_dir = d
            seg_start = i
            seg_high = df["high"].iloc[i]
            seg_low = df["low"].iloc[i]
    # Final open segment
    segs.append({
        "start_idx": seg_start, "end_idx": n - 1,
        "start_dt": df.index[seg_start], "end_dt": df.index[n - 1],
        "direction": cur_dir,
        "entry_close": df["close"].iloc[seg_start],
        "exit_close": df["close"].iloc[n - 1],
        "seg_high": seg_high, "seg_low": seg_low,
        "bars": n - seg_start,
    })
    return pd.DataFrame(segs)


def cell_metrics(df: pd.DataFrame, segs: pd.DataFrame, tf: str) -> dict:
    if segs.empty:
        return {}
    # Closed segments only for stickiness stats (drop final open)
    closed = segs.iloc[:-1] if len(segs) > 1 else segs
    days_span = (df.index[-1] - df.index[0]).days or 1
    months = days_span / 30.4375

    # Calendar-day duration per closed segment
    closed = closed.copy()
    closed["cal_days"] = (closed["end_dt"] - closed["start_dt"]).dt.total_seconds() / 86400.0

    # MFE/MAE per segment (in points), in trend direction
    def mfe(row):
        if row["direction"] == 1:
            return row["seg_high"] - row["entry_close"]
        return row["entry_close"] - row["seg_low"]

    def mae(row):
        if row["direction"] == 1:
            return row["entry_close"] - row["seg_low"]
        return row["seg_high"] - row["entry_close"]

    closed["mfe"] = closed.apply(mfe, axis=1)
    closed["mae"] = closed.apply(mae, axis=1)
    closed["mfe"] = closed["mfe"].clip(lower=0)
    closed["mae"] = closed["mae"].clip(lower=0)

    n_flips = len(closed)
    total_bars = len(df)
    long_bars = (segs["direction"] == 1).pipe(
        lambda s: segs.loc[s, "bars"].sum() if s.any() else 0
    )
    short_bars = (segs["direction"] == -1).pipe(
        lambda s: segs.loc[s, "bars"].sum() if s.any() else 0
    )

    flips_per_month = n_flips / months if months > 0 else 0
    avg_trend_bars = closed["bars"].mean() if n_flips else 0
    avg_trend_cal_days = closed["cal_days"].mean() if n_flips else 0
    median_trend_cal_days = closed["cal_days"].median() if n_flips else 0

    pct_long = long_bars / total_bars if total_bars else 0
    pct_short = short_bars / total_bars if total_bars else 0
    dominant = max(pct_long, pct_short)

    mean_mae = closed["mae"].mean() if n_flips else 0
    mean_mfe = closed["mfe"].mean() if n_flips else 0
    mfe_mae_ratio = (mean_mfe / mean_mae) if mean_mae > 0 else 0

    # Weekly alignment: trends spanning >= 5 calendar days (covers a Thu-Thu cycle)
    weekly_alignment = (closed["cal_days"] >= 5).mean() if n_flips else 0

    return {
        "n_flips": n_flips,
        "flips_per_month": flips_per_month,
        "avg_trend_bars": avg_trend_bars,
        "avg_trend_cal_days": avg_trend_cal_days,
        "median_trend_cal_days": median_trend_cal_days,
        "pct_bars_long": pct_long,
        "pct_bars_short": pct_short,
        "dominant_direction_pct": dominant,
        "mean_mae_pts": mean_mae,
        "mean_mfe_pts": mean_mfe,
        "mfe_mae_ratio": mfe_mae_ratio,
        "weekly_alignment_pct": weekly_alignment,
        "total_bars": total_bars,
        "calendar_days_span": days_span,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "mst_ranking.csv"
    fieldnames = [
        "underlying", "timeframe", "atr_period", "multiplier", "label",
        "n_flips", "flips_per_month",
        "avg_trend_bars", "avg_trend_cal_days", "median_trend_cal_days",
        "pct_bars_long", "pct_bars_short", "dominant_direction_pct",
        "mean_mae_pts", "mean_mfe_pts", "mfe_mae_ratio",
        "weekly_alignment_pct", "total_bars", "calendar_days_span",
        "passed_gates",
    ]
    with open(out_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    # Pre-load OHLC per (underlying, tf)
    print("Loading data...", flush=True)
    ohlc_cache = {}
    for sym in INDICES:
        for tf in TIMEFRAMES:
            t0 = time.time()
            df = get_ohlc(sym, tf)
            ohlc_cache[(sym, tf)] = df
            print(f"  {sym:10} {tf:6} bars={len(df):>6}  {df.index.min()} -> {df.index.max()}  ({time.time()-t0:.1f}s)", flush=True)

    total = len(INDICES) * len(TIMEFRAMES) * len(ATR_PERIODS) * len(MULTIPLIERS)
    i = 0
    rows = []
    t_start = time.time()
    for sym in INDICES:
        for tf in TIMEFRAMES:
            df = ohlc_cache[(sym, tf)]
            high = df["high"].to_numpy()
            low = df["low"].to_numpy()
            close = df["close"].to_numpy()
            for p in ATR_PERIODS:
                for m in MULTIPLIERS:
                    i += 1
                    direction, _, _ = supertrend(high, low, close, p, m)
                    segs = compute_segments(df, direction)
                    metrics = cell_metrics(df, segs, tf)
                    if not metrics:
                        continue
                    # Quality gates
                    gates_ok = (
                        0.5 <= metrics["flips_per_month"] <= 12
                        and metrics["avg_trend_cal_days"] >= 3
                        and metrics["n_flips"] >= 6
                    )
                    label = f"{sym}_{tf}_p{p}_m{m}"
                    row = {
                        "underlying": sym, "timeframe": tf,
                        "atr_period": p, "multiplier": m, "label": label,
                        **metrics,
                        "passed_gates": int(gates_ok),
                    }
                    rows.append(row)
                    with open(out_csv, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
                    if i % 25 == 0 or i == total:
                        print(f"  [{i}/{total}] {label}: flips/mo={metrics['flips_per_month']:.2f} avg_days={metrics['avg_trend_cal_days']:.1f} mfe/mae={metrics['mfe_mae_ratio']:.2f} gates={int(gates_ok)}", flush=True)

    print(f"\nSweep done in {time.time()-t_start:.1f}s. {len(rows)} cells written.", flush=True)

    # Composite ranking
    df = pd.DataFrame(rows)
    df_pass = df[df["passed_gates"] == 1].copy()
    if df_pass.empty:
        print("WARNING: no cell passed quality gates.")
        return

    def norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / rng

    df_pass["score"] = (
        0.40 * norm(df_pass["mfe_mae_ratio"])
        + 0.25 * norm(df_pass["avg_trend_cal_days"])
        + 0.20 * norm(df_pass["dominant_direction_pct"])
        + 0.15 * norm(1.0 / df_pass["flips_per_month"])
    )

    df_pass = df_pass.sort_values("score", ascending=False)
    df_pass.to_csv(RESULTS_DIR / "mst_ranking_scored.csv", index=False)

    for sym in INDICES:
        top = df_pass[df_pass["underlying"] == sym].head(10)
        top.to_csv(RESULTS_DIR / f"mst_top10_{sym}.csv", index=False)
        print(f"\nTop 5 {sym}:")
        for _, r in top.head(5).iterrows():
            print(f"  {r['label']:40} score={r['score']:.3f} flips/mo={r['flips_per_month']:.2f} days={r['avg_trend_cal_days']:.1f} mfe/mae={r['mfe_mae_ratio']:.2f} dom={r['dominant_direction_pct']:.2f}")


if __name__ == "__main__":
    main()
