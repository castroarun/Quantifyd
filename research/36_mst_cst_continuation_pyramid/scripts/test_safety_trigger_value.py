"""How often would the wing-breach safety trigger fire as a NEW pyramid event?

For each first-CST in the extended sample:
  - Compute hypothetical condor strikes (entry_atm = round(seg_entry_close/50)*50)
    K3 (long MST) = entry_atm + 400; K4 = entry_atm + 600
    Safety threshold = entry_atm + 2.5 * 200 = entry_atm + 500 (midpoint of upper wing)
    Mirror for short MST.
  - Find:
    a) When D_cumulative AND B would fire (the new primary trigger)
    b) When the safety trigger would fire (spot reaches ATM ± 500)
  - Categorize:
    * Both fire — D_cumulative+B is fastest; safety is a backup confirmation
    * Only D_cumulative+B fires — safety not needed here
    * Only safety fires — THE VALUE-ADD: D_cumulative+B missed but safety would have caught
    * Neither fires — pure miss (these are typically CST-correct cases or trend reverses fast)
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
from test_reestablishment_triggers import trigger_stoch_return_to_extreme  # noqa
from test_d_variants import d_cumulative  # noqa


SPREAD_WIDTH = 200
SAFETY_WING_FRACTION = 0.5  # 50% into the wing past K3/K-3


def main():
    df = get_ohlc("NIFTY50", "30min")
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    mst_dir, _, _ = supertrend(high, low, close, 21, 5.0)
    raw_segs = compute_segments(df, mst_dir)
    bo_segs = apply_breakout_filter(df, raw_segs)
    closed = bo_segs.iloc[:-1]
    k, _ = stochastic(high, low, close, 14, 3, 3)
    d_arr_full = pd.Series(k).rolling(3).mean().values

    cst_outcomes = {}
    df_csts = pd.read_csv(RESULTS_DIR / "extended_cst_continuation.csv")
    for _, row in df_csts.iterrows():
        cst_outcomes[(row["seg_idx"], row["cst_idx"])] = row["outcome"]

    # Only test FIRST CST per trend (the one that would build the condor)
    seen_seg = set()
    rows = []
    for _, seg in closed.iterrows():
        s = int(seg.start_idx)
        e = int(seg.end_idx)
        dirn = int(seg.direction)
        seg_entry_close = close[s]
        entry_atm = round(seg_entry_close / 50) * 50

        # Find first CST
        first_cst = None
        for i in range(s + 1, e + 1):
            if np.isnan(k[i]) or np.isnan(d_arr_full[i]) or np.isnan(k[i - 1]):
                continue
            is_cst = ((dirn == 1 and k[i - 1] >= d_arr_full[i - 1] and k[i] < d_arr_full[i] and k[i - 1] >= 80) or
                      (dirn == -1 and k[i - 1] <= d_arr_full[i - 1] and k[i] > d_arr_full[i] and k[i - 1] <= 20))
            if is_cst and i + 1 <= e:
                first_cst = i
                break

        if first_cst is None:
            continue

        outcome = cst_outcomes.get((seg.name, first_cst))
        if outcome not in ("TREND_CONTINUED", "CST_CORRECT"):
            continue

        cst_high = high[first_cst]
        cst_low = low[first_cst]
        level_d = cst_high if dirn == 1 else cst_low

        # Time to D_cumulative AND B
        t_d = d_cumulative(close, dirn, first_cst, e, level_d)
        t_b = trigger_stoch_return_to_extreme(k, dirn, first_cst, e)
        t_db = max(t_d, t_b) if (t_d is not None and t_b is not None) else None

        # Time to safety trigger
        # Long MST: spot >= entry_atm + (2 + SAFETY_WING_FRACTION) * SPREAD_WIDTH = entry_atm + 500
        # Short MST: spot <= entry_atm - (2 + SAFETY_WING_FRACTION) * SPREAD_WIDTH
        threshold_offset = (2 + SAFETY_WING_FRACTION) * SPREAD_WIDTH
        t_safety = None
        for i in range(first_cst + 1, e + 1):
            if dirn == 1 and high[i] >= entry_atm + threshold_offset:
                t_safety = i
                break
            if dirn == -1 and low[i] <= entry_atm - threshold_offset:
                t_safety = i
                break

        # Categorize
        db_fired = t_db is not None
        safety_fired = t_safety is not None
        if db_fired and safety_fired:
            if t_db < t_safety:
                category = "DB_first_then_safety"
            elif t_safety < t_db:
                category = "safety_first_then_DB"
            else:
                category = "DB_and_safety_same_bar"
        elif db_fired and not safety_fired:
            category = "DB_only"
        elif safety_fired and not db_fired:
            category = "safety_only_VALUEADD"
        else:
            category = "neither"

        rows.append({
            "seg_idx": seg.name,
            "direction": dirn,
            "first_cst_idx": first_cst,
            "outcome": outcome,
            "entry_atm": entry_atm,
            "safety_threshold": entry_atm + threshold_offset if dirn == 1 else entry_atm - threshold_offset,
            "t_db": t_db,
            "t_safety": t_safety,
            "category": category,
            "db_lead_to_safety": (t_safety - t_db) if (t_db is not None and t_safety is not None) else None,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULTS_DIR / "safety_trigger_analysis.csv", index=False)
    print(f"\nFirst-CSTs analyzed: {len(out_df)}")
    print()
    print("Categories (across ALL first-CSTs):")
    print(out_df["category"].value_counts())
    print()
    print("--- Among TREND_CONTINUED first-CSTs only ---")
    cont = out_df[out_df.outcome == "TREND_CONTINUED"]
    print(f"n = {len(cont)}")
    print(cont["category"].value_counts())
    pct = (cont["category"].value_counts(normalize=True) * 100).round(1)
    print(pct)
    print()
    print("--- Among CST_CORRECT first-CSTs (where pyramiding is WRONG) ---")
    corr = out_df[out_df.outcome == "CST_CORRECT"]
    print(f"n = {len(corr)}")
    print(corr["category"].value_counts())
    print()
    # Key value-add metric:
    cont_safety_only = (cont["category"] == "safety_only_VALUEADD").sum()
    cont_db_only = (cont["category"] == "DB_only").sum()
    cont_both = ((cont["category"] == "DB_first_then_safety") |
                 (cont["category"] == "safety_first_then_DB") |
                 (cont["category"] == "DB_and_safety_same_bar")).sum()
    print(f"VALUE-ADD: {cont_safety_only} cases ({cont_safety_only/len(cont)*100:.1f}%) where ONLY safety would have fired")
    print(f"           This is incremental coverage from adding the safety trigger.")
    corr_safety_only = (corr["category"] == "safety_only_VALUEADD").sum()
    print(f"COST: {corr_safety_only} cases ({corr_safety_only/len(corr)*100:.1f}%) where safety fires in CST_CORRECT (false pyramid)")


if __name__ == "__main__":
    main()
