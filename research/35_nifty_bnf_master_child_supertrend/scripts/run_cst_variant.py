"""Phase B-prime: CST 'exit-the-zone' variant.

Variant rule: trigger fires only when %K both crosses %D AND closes back across
the OB/OS threshold — i.e. the cross has to actually escape the extreme zone,
not just occur while still inside it.

For LONG MST (bearish CST):
    K[i-1] >= D[i-1] AND K[i] < D[i] AND K[i-1] >= 80 AND K[i] < 80
For SHORT MST (bullish CST):
    K[i-1] <= D[i-1] AND K[i] > D[i] AND K[i-1] <= 20 AND K[i] > 20

Compared head-to-head with the original (no-exit-condition) variant.
"""
from __future__ import annotations
import os, sys, csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
sys.path.insert(0, str(SCRIPT_DIR))

from supertrend import supertrend, stochastic  # noqa
from run_mst_sweep import get_ohlc, compute_segments  # noqa
from run_cst_evaluation import evaluate_cst_for_mst, MST_WINNERS  # noqa


def first_stoch_original(k, d, mst_dir, start_idx, end_idx, ob, os_):
    """Original: cross from extreme, no exit-zone requirement."""
    for i in range(start_idx + 1, end_idx + 1):
        if np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(k[i - 1]):
            continue
        if mst_dir == 1:
            if k[i - 1] >= d[i - 1] and k[i] < d[i] and k[i - 1] >= ob:
                return i
        else:
            if k[i - 1] <= d[i - 1] and k[i] > d[i] and k[i - 1] <= os_:
                return i
    return None


def first_stoch_exit_zone(k, d, mst_dir, start_idx, end_idx, ob, os_):
    """Variant: cross AND %K must close back across the threshold."""
    for i in range(start_idx + 1, end_idx + 1):
        if np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(k[i - 1]):
            continue
        if mst_dir == 1:
            if (k[i - 1] >= d[i - 1] and k[i] < d[i]
                    and k[i - 1] >= ob and k[i] < ob):
                return i
        else:
            if (k[i - 1] <= d[i - 1] and k[i] > d[i]
                    and k[i - 1] <= os_ and k[i] > os_):
                return i
    return None


def main():
    out_csv = RESULTS_DIR / "cst_variant_evaluation.csv"
    fields = [
        "mst_underlying", "mst_timeframe", "mst_label",
        "variant", "stoch_config",
        "n_trends", "trigger_count", "no_trigger_count", "coverage",
        "avg_lead_bars", "median_lead_bars", "avg_mae_avoided_pct", "false_positive_rate",
    ]
    with open(out_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"{'MST':40} {'Variant':16} {'Stoch':12} {'Cov':>5} {'Lead':>6} {'Avoid':>6} {'FP':>5}")
    print("-" * 110)

    for sym, tf, p, m in MST_WINNERS:
        df = get_ohlc(sym, tf)
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        mst_dir, _, _ = supertrend(high, low, close, p, m)
        mst_segs = compute_segments(df, mst_dir)
        mst_label = f"{sym}_{tf}_p{p}_m{m}"

        for kp, dp, sm in [(14, 3, 3), (5, 3, 3)]:
            k, d_arr = stochastic(high, low, close, kp, dp, sm)
            for variant_name, fn in [("original", first_stoch_original),
                                     ("exit_zone", first_stoch_exit_zone)]:
                metrics = evaluate_cst_for_mst(
                    df, mst_segs,
                    lambda s, e, mst_d, k=k, d_arr=d_arr, ob=80, os_=20, fn=fn:
                        fn(k, d_arr, mst_d, s, e, ob, os_),
                    f"{variant_name}_{kp}_{dp}_{sm}",
                )
                row = {"mst_underlying": sym, "mst_timeframe": tf, "mst_label": mst_label,
                       "variant": variant_name, "stoch_config": f"({kp},{dp},{sm})",
                       "n_trends": metrics["n_trends"],
                       "trigger_count": metrics["trigger_count"],
                       "no_trigger_count": metrics["no_trigger_count"],
                       "coverage": metrics["coverage"],
                       "avg_lead_bars": metrics["avg_lead_bars"],
                       "median_lead_bars": metrics["median_lead_bars"],
                       "avg_mae_avoided_pct": metrics["avg_mae_avoided_pct"],
                       "false_positive_rate": metrics["false_positive_rate"]}
                with open(out_csv, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row)
                print(f"{mst_label:40} {variant_name:16} ({kp},{dp},{sm})    {metrics['coverage']:.2f}  {metrics['avg_lead_bars']:>5.1f}  {metrics['avg_mae_avoided_pct']:.2f}   {metrics['false_positive_rate']:.2f}")
        print()


if __name__ == "__main__":
    main()
