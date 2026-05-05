"""Re-run the full MST/CST/pyramid analysis on NIFTY 30-min from 2020-01-01.

Extended period: 6.3 years (vs 2 years in original research/35+36).

Three sub-analyses:
  A) MST cell re-rank — does p21,m5.0 still win on 6 years?
  B) CST continuation rate on extended data
  C) Pyramid trigger D-AND-B FP rate / coverage on extended data

All outputs go to results/extended_* CSVs and RESULTS_EXTENDED.md.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

sys.path.insert(0, str(ROOT / "research" / "35_nifty_bnf_master_child_supertrend" / "scripts"))
from supertrend import supertrend, stochastic, atr_wilder  # noqa
from run_mst_sweep import get_ohlc, compute_segments  # noqa
from run_mst_sweep_breakout import apply_breakout_filter, cell_metrics_breakout  # noqa
sys.path.insert(0, str(SCRIPT_DIR))
from test_reestablishment_triggers import (  # noqa
    trigger_stoch_return_to_extreme,
    trigger_break_post_cst_extreme,
    trigger_two_closes_beyond,
    trigger_new_n_bar_extreme,
)


ATR_PERIODS = [7, 10, 14, 21, 30, 50]
MULTIPLIERS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def phase_a_mst_rerank(df, high, low, close):
    """Re-rank all 42 NIFTY 30-min cells on extended data, with break-of-extreme entry."""
    print("\n=== PHASE A: MST cell re-rank (NIFTY 30-min, 42 cells, break-of-extreme) ===")
    rows = []
    for p in ATR_PERIODS:
        for m in MULTIPLIERS:
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
            row = {"atr_period": p, "multiplier": m,
                   "label": f"NIFTY50_30min_p{p}_m{m}",
                   **metrics, "passed_gates": int(gates_ok)}
            rows.append(row)

    df_all = pd.DataFrame(rows)
    df_pass = df_all[df_all.passed_gates == 1].copy()

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
    df_pass.to_csv(RESULTS_DIR / "extended_mst_ranking.csv", index=False)

    print("\nTop 10 cells (extended period 6.3 years):")
    print(df_pass.head(10)[
        ["label", "score", "trades_per_month", "filter_rate",
         "avg_trend_cal_days", "mfe_mae_ratio", "weekly_alignment_pct"]
    ].to_string(index=False))
    return df_pass


def phase_b_cst_continuation(df, high, low, close, k, d_arr, atr, p, m):
    """Run CST continuation labeling on the chosen cell."""
    print(f"\n=== PHASE B: CST continuation on cell p{p},m{m} (extended period) ===")
    direction, _, _ = supertrend(high, low, close, p, m)
    raw_segs = compute_segments(df, direction)
    bo_segs = apply_breakout_filter(df, raw_segs)
    closed = bo_segs.iloc[:-1]

    cst_rows = []
    for seg_idx, seg in closed.iterrows():
        s, e, dirn = int(seg.start_idx), int(seg.end_idx), int(seg.direction)
        seg_entry_close = close[s]

        for i in range(s + 1, e + 1):
            if np.isnan(k[i]) or np.isnan(d_arr[i]) or np.isnan(k[i - 1]):
                continue
            is_cst = ((dirn == 1 and k[i - 1] >= d_arr[i - 1] and k[i] < d_arr[i] and k[i - 1] >= 80) or
                      (dirn == -1 and k[i - 1] <= d_arr[i - 1] and k[i] > d_arr[i] and k[i - 1] <= 20))
            if not is_cst or i + 1 > e:
                continue

            cst_close = close[i]
            cst_atr = atr[i] if not np.isnan(atr[i]) else 0
            future_high = high[i + 1:e + 1].max()
            future_low = low[i + 1:e + 1].min()
            continuation_pts = (future_high - cst_close) if dirn == 1 else (cst_close - future_low)
            reversal_pts = (cst_close - future_low) if dirn == 1 else (future_high - cst_close)

            if continuation_pts >= cst_atr and continuation_pts > reversal_pts:
                outcome = "TREND_CONTINUED"
            elif reversal_pts >= cst_atr and reversal_pts >= continuation_pts:
                outcome = "CST_CORRECT"
            else:
                outcome = "NEUTRAL"

            cst_rows.append({
                "seg_idx": seg_idx, "seg_dir": dirn, "cst_idx": i,
                "cst_dt": df.index[i], "cst_close": cst_close, "cst_atr": cst_atr,
                "continuation_pts": continuation_pts, "reversal_pts": reversal_pts,
                "outcome": outcome,
            })

    df_csts = pd.DataFrame(cst_rows)
    df_csts.to_csv(RESULTS_DIR / "extended_cst_continuation.csv", index=False)
    df_first = df_csts.groupby("seg_idx").first().reset_index()
    df_first.to_csv(RESULTS_DIR / "extended_cst_first_per_trend.csv", index=False)

    print(f"  Total trends (post-breakout filter): {len(closed)}")
    print(f"  Total CSTs analyzed: {len(df_csts)}")
    print(f"  First-CSTs per trend: {len(df_first)}")
    print()
    print("All CSTs outcome distribution:")
    print(df_csts["outcome"].value_counts())
    print((df_csts["outcome"].value_counts(normalize=True) * 100).round(1))
    print()
    print("FIRST CST per trend outcome distribution (what condor experiences):")
    print(df_first["outcome"].value_counts())
    print((df_first["outcome"].value_counts(normalize=True) * 100).round(1))

    tc = df_csts[df_csts.outcome == "TREND_CONTINUED"]
    if not tc.empty:
        print(f"\n  Continuation magnitude (TREND_CONTINUED, n={len(tc)}):")
        print(f"    Median pts: {tc.continuation_pts.median():.0f}")
        print(f"    Mean pts:   {tc.continuation_pts.mean():.0f}")
        print(f"    Median ATR: {(tc.continuation_pts / tc.cst_atr).median():.2f}")

    return df_csts, closed, raw_segs


def phase_c_pyramid_trigger(df, high, low, close, k, d_arr, df_csts, closed):
    """Re-validate D AND B trigger on extended period."""
    print("\n=== PHASE C: Pyramid trigger D AND B on extended period ===")

    cst_outcomes = {(row.seg_idx, row.cst_idx): row.outcome for row in df_csts.itertuples()}

    results = {n: {"continued": [], "correct": []} for n in
               ["A", "B", "C", "D", "D_AND_B", "D_AND_C"]}

    for seg_idx, seg in closed.iterrows():
        s, e, dirn = int(seg.start_idx), int(seg.end_idx), int(seg.direction)
        for i in range(s + 1, e + 1):
            if np.isnan(k[i]) or np.isnan(d_arr[i]) or np.isnan(k[i - 1]):
                continue
            is_cst = ((dirn == 1 and k[i - 1] >= d_arr[i - 1] and k[i] < d_arr[i] and k[i - 1] >= 80) or
                      (dirn == -1 and k[i - 1] <= d_arr[i - 1] and k[i] > d_arr[i] and k[i - 1] <= 20))
            if not is_cst or i + 1 > e:
                continue
            outcome = cst_outcomes.get((seg_idx, i))
            if outcome not in ("TREND_CONTINUED", "CST_CORRECT"):
                continue

            if dirn == 1:
                mfe_idx = i + 1 + int(np.argmax(high[i + 1:e + 1]))
            else:
                mfe_idx = i + 1 + int(np.argmin(low[i + 1:e + 1]))

            level_D = high[i] if dirn == 1 else low[i]
            tA = trigger_new_n_bar_extreme(high, low, dirn, i, e, n=5)
            tB = trigger_stoch_return_to_extreme(k, dirn, i, e)
            tC = trigger_break_post_cst_extreme(high, low, dirn, i, e)
            tD = trigger_two_closes_beyond(close, dirn, i, e, level_D)

            def combo_and(t1, t2):
                if t1 is None or t2 is None:
                    return None
                return max(t1, t2)

            t_DB = combo_and(tD, tB)
            t_DC = combo_and(tD, tC)

            for name, t_idx in [("A", tA), ("B", tB), ("C", tC), ("D", tD),
                                ("D_AND_B", t_DB), ("D_AND_C", t_DC)]:
                rec = {"trigger_idx": t_idx,
                       "lead": (mfe_idx - t_idx) if t_idx is not None else None}
                if outcome == "TREND_CONTINUED":
                    results[name]["continued"].append(rec)
                else:
                    results[name]["correct"].append(rec)

    print(f"\n{'Trigger':12} {'Coverage':>10} {'AvgLead':>10} {'MedLead':>10} {'FP_rate':>10} {'Score':>10}")
    print("-" * 70)
    rows = []
    for name, data in results.items():
        cont = pd.DataFrame(data["continued"])
        corr = pd.DataFrame(data["correct"])
        flagged = cont[cont.trigger_idx.notna()]
        coverage = len(flagged) / len(cont) if len(cont) else 0
        flagged = flagged[flagged.lead.notna()]
        avg_lead = flagged.lead.mean() if len(flagged) else 0
        med_lead = flagged.lead.median() if len(flagged) else 0
        fp = corr[corr.trigger_idx.notna()].shape[0] if not corr.empty else 0
        fp_rate = fp / len(corr) if len(corr) else 0
        useful_lead = max(0, avg_lead)
        score = coverage * useful_lead * (1 - fp_rate)
        print(f"{name:12} {coverage:>10.2%} {avg_lead:>10.2f} {med_lead:>10.2f} {fp_rate:>10.2%} {score:>10.3f}")
        rows.append(dict(trigger=name, coverage=coverage, avg_lead=avg_lead,
                         median_lead=med_lead, fp_rate=fp_rate, score=score,
                         n_continued_total=len(cont), n_continued_flagged=len(flagged),
                         n_correct_total=len(corr), n_correct_flagged=fp))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "extended_trigger_scores.csv", index=False)


def main():
    df = get_ohlc("NIFTY50", "30min")
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    days_span = (df.index[-1] - df.index[0]).days
    print(f"Data: {len(df)} bars, {df.index.min()} -> {df.index.max()} ({days_span} cal days, ~{days_span/365.25:.1f} years)")

    # Phase A
    ranking = phase_a_mst_rerank(df, high, low, close)
    top = ranking.iloc[0]
    P, M = int(top["atr_period"]), float(top["multiplier"])
    print(f"\n>>> Top cell on extended data: p{P}, m{M} (score={top['score']:.3f})")

    # Phase B - keep using p21,m5.0 for continuity with research/35
    P_orig, M_orig = 21, 5.0
    print(f"\n>>> Re-running CST continuation on RECOMMENDED cell p{P_orig}, m{M_orig} for direct comparison")
    k, d_arr = stochastic(high, low, close, 14, 3, 3)
    atr = atr_wilder(high, low, close, 21)
    df_csts, closed, raw_segs = phase_b_cst_continuation(df, high, low, close, k, d_arr, atr, P_orig, M_orig)

    # Phase C
    phase_c_pyramid_trigger(df, high, low, close, k, d_arr, df_csts, closed)


if __name__ == "__main__":
    main()
