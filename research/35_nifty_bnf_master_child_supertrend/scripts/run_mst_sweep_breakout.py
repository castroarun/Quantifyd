"""Phase A-prime: MST sweep with BREAK-OF-EXTREME entry filter.

Same SuperTrend cells as run_mst_sweep.py, but the entry into each segment
requires a subsequent bar to break the flip-bar's high (long) or low (short).
- If the breakout never happens before the next flip, the segment is FILTERED OUT.
- Entry price = the break level (high of flip-bar for long, low for short).
- MFE/MAE recomputed from the breakout bar onwards.
"""
from __future__ import annotations
import os, sys, sqlite3, time, csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
sys.path.insert(0, str(SCRIPT_DIR))

from supertrend import supertrend  # noqa
from run_mst_sweep import get_ohlc, compute_segments, INDICES, TIMEFRAMES, ATR_PERIODS, MULTIPLIERS  # noqa


def apply_breakout_filter(df: pd.DataFrame, raw_segs: pd.DataFrame) -> pd.DataFrame:
    """For each raw close-flip segment, compute the break-of-extreme entry.
    Drops segments where the breakout never occurs before the next flip."""
    if raw_segs.empty:
        return raw_segs
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    new_segs = []
    for _, seg in raw_segs.iterrows():
        s, e, d = int(seg.start_idx), int(seg.end_idx), int(seg.direction)
        flip_high = high[s]
        flip_low = low[s]

        entry_idx = None
        for i in range(s + 1, e + 1):
            if d == 1 and high[i] > flip_high:
                entry_idx = i
                break
            if d == -1 and low[i] < flip_low:
                entry_idx = i
                break
        if entry_idx is None:
            continue

        entry_price = flip_high if d == 1 else flip_low

        if entry_idx <= e:
            seg_h = float(high[entry_idx : e + 1].max())
            seg_l = float(low[entry_idx : e + 1].min())
        else:
            seg_h = high[entry_idx]
            seg_l = low[entry_idx]

        new_segs.append({
            "start_idx": entry_idx, "end_idx": e,
            "start_dt": df.index[entry_idx], "end_dt": df.index[e],
            "direction": d,
            "entry_close": entry_price,
            "exit_close": close[e],
            "seg_high": seg_h, "seg_low": seg_l,
            "bars": e - entry_idx + 1,
            "orig_flip_idx": s,
            "entry_lag_bars": entry_idx - s,
        })

    return pd.DataFrame(new_segs)


def cell_metrics_breakout(df: pd.DataFrame, segs: pd.DataFrame, raw_n_flips: int) -> dict:
    if segs.empty:
        return {}
    closed = segs.iloc[:-1] if len(segs) > 1 else segs
    days_span = (df.index[-1] - df.index[0]).days or 1
    months = days_span / 30.4375

    closed = closed.copy()
    closed["cal_days"] = (closed["end_dt"] - closed["start_dt"]).dt.total_seconds() / 86400.0

    def mfe(row):
        if row["direction"] == 1:
            return row["seg_high"] - row["entry_close"]
        return row["entry_close"] - row["seg_low"]

    def mae(row):
        if row["direction"] == 1:
            return row["entry_close"] - row["seg_low"]
        return row["seg_high"] - row["entry_close"]

    closed["mfe"] = closed.apply(mfe, axis=1).clip(lower=0)
    closed["mae"] = closed.apply(mae, axis=1).clip(lower=0)

    n_trades = len(closed)
    total_bars = len(df)

    # How many raw flips were filtered out
    filter_rate = 1.0 - (n_trades / raw_n_flips) if raw_n_flips else 0

    # Direction time-in-market. Note: with breakout filter we're flat between
    # flip bar and breakout bar, so dominant_direction is computed on bars
    # within active segments only.
    bars_in_active = int(closed["bars"].sum())
    long_bars = int(closed[closed["direction"] == 1]["bars"].sum())
    short_bars = int(closed[closed["direction"] == -1]["bars"].sum())
    pct_long = long_bars / total_bars
    pct_short = short_bars / total_bars
    dominant = max(pct_long, pct_short)
    pct_in_market = bars_in_active / total_bars

    flips_per_month = n_trades / months if months > 0 else 0
    avg_trend_bars = closed["bars"].mean()
    avg_trend_cal_days = closed["cal_days"].mean()
    median_trend_cal_days = closed["cal_days"].median()

    mean_mae = closed["mae"].mean()
    mean_mfe = closed["mfe"].mean()
    mfe_mae_ratio = (mean_mfe / mean_mae) if mean_mae > 0 else 0
    weekly_alignment = (closed["cal_days"] >= 5).mean()
    avg_entry_lag = closed["bars"].rsub(closed["bars"]).mean()  # placeholder
    avg_entry_lag = float(closed.get("entry_lag_bars", pd.Series([0])).mean()) if "entry_lag_bars" in segs.columns else 0
    # entry_lag is on raw segs not closed; recompute from segs
    avg_entry_lag = float(segs["entry_lag_bars"].mean()) if "entry_lag_bars" in segs.columns else 0

    return {
        "n_trades": n_trades,
        "raw_n_flips": raw_n_flips,
        "filter_rate": filter_rate,
        "trades_per_month": flips_per_month,
        "avg_entry_lag_bars": avg_entry_lag,
        "avg_trend_bars": avg_trend_bars,
        "avg_trend_cal_days": avg_trend_cal_days,
        "median_trend_cal_days": median_trend_cal_days,
        "pct_in_market": pct_in_market,
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
    out_csv = RESULTS_DIR / "mst_ranking_breakout.csv"
    fields = [
        "underlying", "timeframe", "atr_period", "multiplier", "label",
        "n_trades", "raw_n_flips", "filter_rate", "trades_per_month",
        "avg_entry_lag_bars",
        "avg_trend_bars", "avg_trend_cal_days", "median_trend_cal_days",
        "pct_in_market", "pct_bars_long", "pct_bars_short", "dominant_direction_pct",
        "mean_mae_pts", "mean_mfe_pts", "mfe_mae_ratio",
        "weekly_alignment_pct", "total_bars", "calendar_days_span",
        "passed_gates",
    ]
    with open(out_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print("Loading data...", flush=True)
    ohlc_cache = {}
    for sym in INDICES:
        for tf in TIMEFRAMES:
            df = get_ohlc(sym, tf)
            ohlc_cache[(sym, tf)] = df
            print(f"  {sym:10} {tf:6} bars={len(df):>6}", flush=True)

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
                    raw_segs = compute_segments(df, direction)
                    raw_closed = max(0, len(raw_segs) - 1)
                    filt_segs = apply_breakout_filter(df, raw_segs)
                    metrics = cell_metrics_breakout(df, filt_segs, raw_closed)
                    if not metrics:
                        continue
                    gates_ok = (
                        0.5 <= metrics["trades_per_month"] <= 12
                        and metrics["avg_trend_cal_days"] >= 3
                        and metrics["n_trades"] >= 6
                    )
                    label = f"{sym}_{tf}_p{p}_m{m}"
                    row = {
                        "underlying": sym, "timeframe": tf,
                        "atr_period": p, "multiplier": m, "label": label,
                        **metrics, "passed_gates": int(gates_ok),
                    }
                    rows.append(row)
                    with open(out_csv, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=fields).writerow(row)
                    if i % 25 == 0 or i == total:
                        print(f"  [{i}/{total}] {label}: trades/mo={metrics['trades_per_month']:.2f} filt={metrics['filter_rate']:.0%} avg_days={metrics['avg_trend_cal_days']:.1f} mfe/mae={metrics['mfe_mae_ratio']:.2f} gates={int(gates_ok)}", flush=True)

    print(f"\nSweep done in {time.time()-t_start:.1f}s. {len(rows)} cells written.", flush=True)

    df_all = pd.DataFrame(rows)
    df_pass = df_all[df_all["passed_gates"] == 1].copy()
    if df_pass.empty:
        print("No cell passed gates.")
        return

    def norm(s):
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / rng

    df_pass["score"] = (
        0.40 * norm(df_pass["mfe_mae_ratio"])
        + 0.25 * norm(df_pass["avg_trend_cal_days"])
        + 0.20 * norm(df_pass["dominant_direction_pct"])
        + 0.15 * norm(1.0 / df_pass["trades_per_month"])
    )
    df_pass = df_pass.sort_values("score", ascending=False)
    df_pass.to_csv(RESULTS_DIR / "mst_ranking_breakout_scored.csv", index=False)

    for sym in INDICES:
        for tf in TIMEFRAMES:
            top = df_pass[(df_pass.underlying == sym) & (df_pass.timeframe == tf)].head(5)
            if top.empty:
                continue
            print(f"\n--- {sym} {tf} (BREAKOUT) top 5 ---")
            print(top[["label", "score", "trades_per_month", "filter_rate",
                       "avg_entry_lag_bars", "avg_trend_cal_days",
                       "mfe_mae_ratio", "dominant_direction_pct",
                       "weekly_alignment_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
