"""Test combinations of the top triggers from the prior round.

Goal: lower the false-positive rate while keeping high coverage. Pyramiding
on a false signal doubles directional exposure right before a real reversal,
so FP rate is more critical than coverage.

Combinations tested:
  D AND B         — 2 closes past CST bar AND stoch K back to ≥80/≤20
  D AND C         — 2 closes past CST bar AND break of post-CST extreme
  D OR B          — either fires (high coverage)
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
from run_mst_sweep_breakout import apply_breakout_filter  # noqa
sys.path.insert(0, str(SCRIPT_DIR))
from test_reestablishment_triggers import (  # noqa
    trigger_new_n_bar_extreme,
    trigger_stoch_return_to_extreme,
    trigger_break_post_cst_extreme,
    trigger_two_closes_beyond,
)


def main():
    df = get_ohlc("NIFTY50", "30min")
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    mst_dir, _, _ = supertrend(high, low, close, 21, 5.0)
    raw_segs = compute_segments(df, mst_dir)
    bo_segs = apply_breakout_filter(df, raw_segs)
    closed = bo_segs.iloc[:-1]
    k, d_arr = stochastic(high, low, close, 14, 3, 3)

    cst_outcomes = {}
    for _, row in pd.read_csv(RESULTS_DIR / "cst_continuation_per_trigger.csv").iterrows():
        cst_outcomes[(row["seg_idx"], row["cst_idx"])] = row["outcome"]

    combos = ["D_AND_B", "D_AND_C", "D_OR_B", "D_OR_C"]
    results = {c: {"continued": [], "correct": []} for c in combos}

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
            tB = trigger_stoch_return_to_extreme(k, dirn, i, e)
            tC = trigger_break_post_cst_extreme(high, low, dirn, i, e)
            tD = trigger_two_closes_beyond(close, dirn, i, e, level_D)

            # AND combinators take the LATER of the two trigger times (both must fire)
            def combo_and(t1, t2):
                if t1 is None or t2 is None:
                    return None
                return max(t1, t2)

            def combo_or(t1, t2):
                cands = [t for t in (t1, t2) if t is not None]
                return min(cands) if cands else None

            t_d_and_b = combo_and(tD, tB)
            t_d_and_c = combo_and(tD, tC)
            t_d_or_b = combo_or(tD, tB)
            t_d_or_c = combo_or(tD, tC)

            for combo_name, t_idx in [
                ("D_AND_B", t_d_and_b),
                ("D_AND_C", t_d_and_c),
                ("D_OR_B", t_d_or_b),
                ("D_OR_C", t_d_or_c),
            ]:
                rec = {"trigger_idx": t_idx,
                       "lead_to_mfe": (mfe_idx - t_idx) if t_idx is not None else None}
                if outcome == "TREND_CONTINUED":
                    results[combo_name]["continued"].append(rec)
                else:
                    results[combo_name]["correct"].append(rec)

    print("=" * 110)
    print(f"{'Combination':25} {'Coverage':>10} {'AvgLead':>10} {'MedLead':>10} {'FP_rate':>10} {'Score':>10}")
    print("=" * 110)
    rows = []
    for combo, data in results.items():
        cont = pd.DataFrame(data["continued"])
        corr = pd.DataFrame(data["correct"])
        flagged = cont[cont.trigger_idx.notna()]
        coverage = len(flagged) / len(cont) if len(cont) else 0
        flagged = flagged[flagged.lead_to_mfe.notna()]
        avg_lead = flagged.lead_to_mfe.mean() if len(flagged) else 0
        med_lead = flagged.lead_to_mfe.median() if len(flagged) else 0
        fp_count = corr[corr.trigger_idx.notna()].shape[0] if not corr.empty else 0
        fp_rate = fp_count / len(corr) if len(corr) else 0
        useful_lead = max(0, avg_lead)
        score = coverage * useful_lead * (1 - fp_rate)
        print(f"{combo:25} {coverage:>10.2%} {avg_lead:>10.2f} {med_lead:>10.2f} {fp_rate:>10.2%} {score:>10.3f}")
        rows.append(dict(combo=combo, coverage=coverage, avg_lead=avg_lead,
                         median_lead=med_lead, fp_rate=fp_rate, score=score))

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "combined_trigger_scores.csv", index=False)


if __name__ == "__main__":
    main()
