"""Phase B: CST/hedge-trigger evaluation.

For each MST winner, slice into trend segments. Within each segment, find the
MAE peak (max adverse excursion). Evaluate four CST families on whether they
fire BEFORE the MAE peak — measuring lead-time, coverage, MAE-avoided.
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

from supertrend import supertrend, resample_5min_to, rsi, stochastic, bollinger  # noqa
from run_mst_sweep import get_ohlc, compute_segments  # noqa

# MST winners — top 2 per index (30-min anchor + 60-min comparator)
MST_WINNERS = [
    ("NIFTY50",   "30min", 21, 5.0),
    ("NIFTY50",   "60min", 50, 5.0),
    ("BANKNIFTY", "30min", 50, 5.0),
    ("BANKNIFTY", "60min", 21, 4.0),
]

# CST family configs
CST_SUPERTREND = [(3, 1.0), (3, 1.5), (5, 1.0), (5, 1.5), (7, 1.0), (7, 1.5), (7, 2.0)]
CST_STOCH = [(5, 3, 3, 80, 20), (14, 3, 3, 80, 20)]  # k, d, smooth, ob, os
CST_RSI = [(14, 70, 30), (9, 70, 30)]
CST_BB = [(20, 2.0), (20, 2.5)]


def first_supertrend_flip_against(direction_cst, mst_dir, start_idx, end_idx):
    """Return first idx in [start_idx+1, end_idx] where CST direction != mst_dir."""
    for i in range(start_idx + 1, end_idx + 1):
        if not np.isnan(direction_cst[i]) and direction_cst[i] != mst_dir:
            return i
    return None


def first_stoch_against(k, d, mst_dir, start_idx, end_idx, ob, os_):
    """In uptrend (mst_dir=1): trigger when %K crosses BELOW %D from above OB.
    In downtrend (mst_dir=-1): trigger when %K crosses ABOVE %D from below OS."""
    for i in range(start_idx + 1, end_idx + 1):
        if np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(k[i - 1]):
            continue
        if mst_dir == 1:
            # bearish cross from OB
            if k[i - 1] >= d[i - 1] and k[i] < d[i] and k[i - 1] >= ob:
                return i
        else:
            if k[i - 1] <= d[i - 1] and k[i] > d[i] and k[i - 1] <= os_:
                return i
    return None


def first_rsi_against(rsi_arr, mst_dir, start_idx, end_idx, ob, os_):
    """Uptrend: RSI peaks above OB then turns down. Downtrend: RSI bottoms below OS then turns up."""
    peaked = False
    peak_val = None
    for i in range(start_idx + 1, end_idx + 1):
        if np.isnan(rsi_arr[i]):
            continue
        if mst_dir == 1:
            if rsi_arr[i] >= ob:
                peaked = True
                peak_val = max(peak_val or 0, rsi_arr[i])
            if peaked and rsi_arr[i] < (peak_val - 5):
                return i
        else:
            if rsi_arr[i] <= os_:
                peaked = True
                peak_val = min(peak_val or 100, rsi_arr[i])
            if peaked and rsi_arr[i] > (peak_val + 5):
                return i
    return None


def first_bb_against(close, upper, lower, mst_dir, start_idx, end_idx):
    """Uptrend: close tags upper band then prints a red candle. Downtrend: tags lower then green."""
    for i in range(start_idx + 1, end_idx + 1):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        if mst_dir == 1:
            tagged = close[i] >= upper[i]
            reversed_ = i + 1 <= end_idx and close[i + 1] < close[i]
            if tagged and reversed_:
                return i + 1
        else:
            tagged = close[i] <= lower[i]
            reversed_ = i + 1 <= end_idx and close[i + 1] > close[i]
            if tagged and reversed_:
                return i + 1
    return None


def evaluate_cst_for_mst(df, mst_segs, trigger_fn, label):
    """Run trigger_fn over each closed MST segment. Return aggregate metrics."""
    closed = mst_segs.iloc[:-1] if len(mst_segs) > 1 else mst_segs
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    leads = []
    mae_avoided_pct = []
    trigger_count = 0
    fp_count = 0  # false positive: trigger fired but trend ended in MFE direction with tiny MAE
    no_trigger = 0

    MAE_THRESHOLD_PTS = 0  # any MAE counts; we measure capture vs absolute MAE

    for _, seg in closed.iterrows():
        s, e, d = int(seg.start_idx), int(seg.end_idx), int(seg.direction)
        if e - s < 3:
            continue
        # Compute MAE path bar-by-bar
        if d == 1:
            entry = close[s]
            running_low = np.minimum.accumulate(low[s:e + 1])
            ae_series = entry - running_low  # adverse excursion
        else:
            entry = close[s]
            running_high = np.maximum.accumulate(high[s:e + 1])
            ae_series = running_high - entry
        mae_peak = ae_series.max()
        if mae_peak <= 0:
            continue  # trend went all our way

        mae_peak_idx = s + int(np.argmax(ae_series))

        trig = trigger_fn(s, e, d)
        if trig is None:
            no_trigger += 1
            continue
        trigger_count += 1
        ae_at_trig = ae_series[trig - s] if trig - s < len(ae_series) else mae_peak
        if mae_peak > 0:
            avoided = max(0.0, (mae_peak - ae_at_trig) / mae_peak)
        else:
            avoided = 0.0
        leads.append(mae_peak_idx - trig)
        mae_avoided_pct.append(avoided)
        # False positive: trigger but MAE was tiny (< 25% of seg MFE)
        seg_mfe = seg.seg_high - entry if d == 1 else entry - seg.seg_low
        if mae_peak < 0.25 * max(seg_mfe, 1e-9):
            fp_count += 1

    n_trends = len(closed)
    coverage = trigger_count / n_trends if n_trends else 0
    return {
        "cst_label": label,
        "n_trends": n_trends,
        "trigger_count": trigger_count,
        "no_trigger_count": no_trigger,
        "coverage": coverage,
        "avg_lead_bars": float(np.mean(leads)) if leads else 0,
        "median_lead_bars": float(np.median(leads)) if leads else 0,
        "avg_mae_avoided_pct": float(np.mean(mae_avoided_pct)) if mae_avoided_pct else 0,
        "false_positive_rate": fp_count / trigger_count if trigger_count else 0,
    }


def main():
    out_csv = RESULTS_DIR / "cst_evaluation.csv"
    fields = [
        "mst_underlying", "mst_timeframe", "mst_label",
        "cst_family", "cst_label",
        "n_trends", "trigger_count", "no_trigger_count", "coverage",
        "avg_lead_bars", "median_lead_bars", "avg_mae_avoided_pct", "false_positive_rate",
    ]
    with open(out_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for sym, tf, p, m in MST_WINNERS:
        print(f"\n=== MST {sym} {tf} p{p} m{m} ===", flush=True)
        df = get_ohlc(sym, tf)
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        mst_dir, _, _ = supertrend(high, low, close, p, m)
        mst_segs = compute_segments(df, mst_dir)
        print(f"  segments={len(mst_segs)}", flush=True)
        mst_label = f"{sym}_{tf}_p{p}_m{m}"

        # 1) Shorter SuperTrend
        for cp, cm in CST_SUPERTREND:
            cst_dir, _, _ = supertrend(high, low, close, cp, cm)
            label = f"ST_p{cp}_m{cm}"
            metrics = evaluate_cst_for_mst(
                df, mst_segs,
                lambda s, e, d: first_supertrend_flip_against(cst_dir, d, s, e),
                label,
            )
            row = {"mst_underlying": sym, "mst_timeframe": tf, "mst_label": mst_label,
                   "cst_family": "ShorterST", **metrics}
            with open(out_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(f"  ShorterST {label:15} cov={metrics['coverage']:.2f} lead={metrics['avg_lead_bars']:.1f} avoid={metrics['avg_mae_avoided_pct']:.2f} fp={metrics['false_positive_rate']:.2f}", flush=True)

        # 2) Stochastic
        for kp, dp, sm, ob, os_ in CST_STOCH:
            k, d_arr = stochastic(high, low, close, kp, dp, sm)
            label = f"Stoch_{kp}_{dp}_{sm}_OB{ob}_OS{os_}"
            metrics = evaluate_cst_for_mst(
                df, mst_segs,
                lambda s, e, mst_d, k=k, d_arr=d_arr, ob=ob, os_=os_:
                    first_stoch_against(k, d_arr, mst_d, s, e, ob, os_),
                label,
            )
            row = {"mst_underlying": sym, "mst_timeframe": tf, "mst_label": mst_label,
                   "cst_family": "Stoch", **metrics}
            with open(out_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(f"  Stoch     {label:25} cov={metrics['coverage']:.2f} lead={metrics['avg_lead_bars']:.1f} avoid={metrics['avg_mae_avoided_pct']:.2f} fp={metrics['false_positive_rate']:.2f}", flush=True)

        # 3) RSI
        for rp, ob, os_ in CST_RSI:
            r = rsi(close, rp)
            label = f"RSI{rp}_OB{ob}_OS{os_}"
            metrics = evaluate_cst_for_mst(
                df, mst_segs,
                lambda s, e, mst_d, r=r, ob=ob, os_=os_:
                    first_rsi_against(r, mst_d, s, e, ob, os_),
                label,
            )
            row = {"mst_underlying": sym, "mst_timeframe": tf, "mst_label": mst_label,
                   "cst_family": "RSI", **metrics}
            with open(out_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(f"  RSI       {label:20} cov={metrics['coverage']:.2f} lead={metrics['avg_lead_bars']:.1f} avoid={metrics['avg_mae_avoided_pct']:.2f} fp={metrics['false_positive_rate']:.2f}", flush=True)

        # 4) Bollinger Bands
        for bp, bm in CST_BB:
            u, mid, lwr = bollinger(close, bp, bm)
            label = f"BB_{bp}_{bm}"
            metrics = evaluate_cst_for_mst(
                df, mst_segs,
                lambda s, e, mst_d, u=u, lwr=lwr:
                    first_bb_against(close, u, lwr, mst_d, s, e),
                label,
            )
            row = {"mst_underlying": sym, "mst_timeframe": tf, "mst_label": mst_label,
                   "cst_family": "BB", **metrics}
            with open(out_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(f"  BB        {label:15} cov={metrics['coverage']:.2f} lead={metrics['avg_lead_bars']:.1f} avoid={metrics['avg_mae_avoided_pct']:.2f} fp={metrics['false_positive_rate']:.2f}", flush=True)

    print("\nPhase B done.")


if __name__ == "__main__":
    main()
