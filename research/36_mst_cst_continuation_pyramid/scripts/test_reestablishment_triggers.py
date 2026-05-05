"""Test candidate "trend re-establishment" triggers.

Goal: among CSTs that turned out to be false alarms (trend continued), which
indicator-based trigger would have flagged the resumption EARLY and reliably?

Triggers tested:
  A) New 5-bar HIGH (long) / 5-bar LOW (short) after CST
  B) Stoch %K crosses back above 80 (long) / below 20 (short) — momentum return
  C) Break of extreme: high of any bar between CST and now is broken
  D) Two consecutive closes above CST bar's high (long) / below low (short)
  E) Stoch %K crosses %D from below in NOT-OB zone (50<K<80) — bullish cross
  F) Price closes above an entry-time anchored level (ATM + 0.5×SPREAD_WIDTH)
     — uses the position's structural levels

Scoring per trigger:
  - Coverage: % of TREND_CONTINUED CSTs that the trigger correctly flags BEFORE
              the trend's MFE peak after CST
  - Avg lead time: bars from trigger to MFE peak (positive = leads)
  - False positive rate: triggers fired in CST_CORRECT trends (where we
                          should NOT have re-doubled)
"""
from __future__ import annotations
import sys, csv
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


def trigger_new_n_bar_extreme(high, low, dirn, cst_idx, end_idx, n=5):
    """A) After CST, find first bar that prints a new n-bar high/low."""
    for i in range(cst_idx + 1, end_idx + 1):
        if i - n + 1 < 0:
            continue
        if dirn == 1:
            if high[i] > high[i - n:i].max():
                return i
        else:
            if low[i] < low[i - n:i].min():
                return i
    return None


def trigger_stoch_return_to_extreme(k, dirn, cst_idx, end_idx, ob=80, os_=20):
    """B) Long: %K returns to >=80 after dipping below; Short: %K returns to <=20."""
    if cst_idx + 1 > end_idx:
        return None
    # Wait for K to leave the extreme zone first
    left_zone = False
    for i in range(cst_idx + 1, end_idx + 1):
        if np.isnan(k[i]):
            continue
        if dirn == 1:
            if k[i] < 70:
                left_zone = True
            if left_zone and k[i] >= ob:
                return i
        else:
            if k[i] > 30:
                left_zone = True
            if left_zone and k[i] <= os_:
                return i
    return None


def trigger_break_post_cst_extreme(high, low, dirn, cst_idx, end_idx):
    """C) Find the bar with highest high (long) / lowest low (short) between
    CST and current bar; trigger when a subsequent bar BREAKS that extreme."""
    if cst_idx + 1 > end_idx:
        return None
    if dirn == 1:
        running_high = high[cst_idx]
        for i in range(cst_idx + 1, end_idx + 1):
            if high[i] > running_high:
                # Don't fire on FIRST new high; wait for a SECOND breakout
                # of any local high (more selective).
                # Simplification: fire when high[i] breaks high[i-1] AND high[i-1]
                # was already above previous run.
                if i + 1 <= end_idx and high[i + 1] > high[i]:
                    return i + 1
                running_high = high[i]
    else:
        running_low = low[cst_idx]
        for i in range(cst_idx + 1, end_idx + 1):
            if low[i] < running_low:
                if i + 1 <= end_idx and low[i + 1] < low[i]:
                    return i + 1
                running_low = low[i]
    return None


def trigger_two_closes_beyond(close, dirn, cst_idx, end_idx, level):
    """D) Two consecutive closes above (long) / below (short) the given level."""
    streak = 0
    for i in range(cst_idx + 1, end_idx + 1):
        if (dirn == 1 and close[i] > level) or (dirn == -1 and close[i] < level):
            streak += 1
            if streak >= 2:
                return i
        else:
            streak = 0
    return None


def trigger_stoch_bullish_midzone(k, d, dirn, cst_idx, end_idx):
    """E) Long: %K crosses ABOVE %D where both are in [40, 80]; opposite for short."""
    for i in range(cst_idx + 2, end_idx + 1):
        if np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(k[i - 1]):
            continue
        if dirn == 1:
            if (k[i - 1] <= d[i - 1] and k[i] > d[i]
                    and 40 < k[i] < 80 and 40 < d[i] < 80):
                return i
        else:
            if (k[i - 1] >= d[i - 1] and k[i] < d[i]
                    and 20 < k[i] < 60 and 20 < d[i] < 60):
                return i
    return None


def trigger_close_above_anchor(close, dirn, cst_idx, end_idx, anchor):
    """F) First close past the anchor level after CST."""
    for i in range(cst_idx + 1, end_idx + 1):
        if (dirn == 1 and close[i] > anchor) or (dirn == -1 and close[i] < anchor):
            return i
    return None


def main():
    df = get_ohlc("NIFTY50", "30min")
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    mst_dir, _, _ = supertrend(high, low, close, 21, 5.0)
    raw_segs = compute_segments(df, mst_dir)
    bo_segs = apply_breakout_filter(df, raw_segs)
    closed = bo_segs.iloc[:-1]

    k, d = stochastic(high, low, close, 14, 3, 3)
    atr = atr_wilder(high, low, close, 21)

    # Reload CST results from prior measurement to know outcomes
    cst_outcomes = {}
    df_csts = pd.read_csv(RESULTS_DIR / "cst_continuation_per_trigger.csv")
    for _, row in df_csts.iterrows():
        cst_outcomes[(row["seg_idx"], row["cst_idx"])] = row["outcome"]

    trigger_results = {n: {"continued_flagged": [], "correct_flagged": []}
                       for n in ["A_new_5bar_extreme", "B_stoch_return_to_extreme",
                                "C_break_post_cst_extreme", "D_two_closes_beyond",
                                "E_stoch_midzone_cross",
                                "F_close_above_anchor_atm200",
                                "F2_close_above_anchor_atm100"]}

    SPREAD_WIDTH = 200

    for seg_idx, seg in closed.iterrows():
        s, e, dirn = int(seg.start_idx), int(seg.end_idx), int(seg.direction)
        seg_entry_close = close[s]

        # Anchor levels (entry-time ATM rounded to 50)
        entry_atm = round(seg_entry_close / 50) * 50
        anchor_atm200 = entry_atm + (SPREAD_WIDTH if dirn == 1 else -SPREAD_WIDTH)
        anchor_atm100 = entry_atm + (100 if dirn == 1 else -100)

        # Find each CST in this trend
        for i in range(s + 1, e + 1):
            if np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(k[i - 1]):
                continue
            is_cst = ((dirn == 1 and k[i - 1] >= d[i - 1] and k[i] < d[i] and k[i - 1] >= 80) or
                      (dirn == -1 and k[i - 1] <= d[i - 1] and k[i] > d[i] and k[i - 1] <= 20))
            if not is_cst or i + 1 > e:
                continue

            outcome = cst_outcomes.get((seg_idx, i))
            if outcome not in ("TREND_CONTINUED", "CST_CORRECT"):
                continue

            # MFE peak idx after this CST (only for measuring lead time)
            if dirn == 1:
                mfe_idx = i + 1 + int(np.argmax(high[i + 1:e + 1]))
            else:
                mfe_idx = i + 1 + int(np.argmin(low[i + 1:e + 1]))

            # Test each trigger
            tA = trigger_new_n_bar_extreme(high, low, dirn, i, e, n=5)
            tB = trigger_stoch_return_to_extreme(k, dirn, i, e)
            tC = trigger_break_post_cst_extreme(high, low, dirn, i, e)
            level_for_D = high[i] if dirn == 1 else low[i]
            tD = trigger_two_closes_beyond(close, dirn, i, e, level_for_D)
            tE = trigger_stoch_bullish_midzone(k, d, dirn, i, e)
            tF = trigger_close_above_anchor(close, dirn, i, e, anchor_atm200)
            tF2 = trigger_close_above_anchor(close, dirn, i, e, anchor_atm100)

            for trig_name, trig_idx in [
                ("A_new_5bar_extreme", tA),
                ("B_stoch_return_to_extreme", tB),
                ("C_break_post_cst_extreme", tC),
                ("D_two_closes_beyond", tD),
                ("E_stoch_midzone_cross", tE),
                ("F_close_above_anchor_atm200", tF),
                ("F2_close_above_anchor_atm100", tF2),
            ]:
                rec = {
                    "seg_idx": seg_idx,
                    "cst_idx": i,
                    "trigger_idx": trig_idx,
                    "lead_to_mfe": (mfe_idx - trig_idx) if trig_idx is not None else None,
                    "outcome": outcome,
                }
                if outcome == "TREND_CONTINUED":
                    trigger_results[trig_name]["continued_flagged"].append(rec)
                else:
                    trigger_results[trig_name]["correct_flagged"].append(rec)

    # Aggregate
    print("=" * 110)
    print(f"{'Trigger':35} {'Coverage':>10} {'AvgLead':>10} {'MedLead':>10} {'FP_rate':>10} {'Score':>10}")
    print("=" * 110)

    score_rows = []
    for trig_name, data in trigger_results.items():
        cont = pd.DataFrame(data["continued_flagged"])
        corr = pd.DataFrame(data["correct_flagged"])
        if cont.empty:
            continue
        # Coverage: in TREND_CONTINUED, did the trigger fire (not None)?
        flagged = cont[cont.trigger_idx.notna()]
        coverage = len(flagged) / len(cont)
        # Among flagged, lead time
        flagged = flagged[flagged.lead_to_mfe.notna()]
        avg_lead = flagged.lead_to_mfe.mean() if len(flagged) else 0
        med_lead = flagged.lead_to_mfe.median() if len(flagged) else 0
        # FP: in CST_CORRECT trends, did the trigger fire (false alarm — pyramiding when we shouldn't)
        fp_count = corr[corr.trigger_idx.notna()].shape[0] if not corr.empty else 0
        fp_rate = fp_count / len(corr) if len(corr) else 0
        # Composite score: coverage × avg_lead × (1 - fp_rate)
        # Higher is better. Lead must be positive to count.
        useful_lead = max(0, avg_lead)
        score = coverage * useful_lead * (1 - fp_rate)
        print(f"{trig_name:35} {coverage:>10.2%} {avg_lead:>10.2f} {med_lead:>10.2f} {fp_rate:>10.2%} {score:>10.3f}")
        score_rows.append({
            "trigger": trig_name,
            "coverage": coverage,
            "avg_lead_bars": avg_lead,
            "median_lead_bars": med_lead,
            "false_positive_rate": fp_rate,
            "composite_score": score,
            "continued_flagged_n": len(flagged),
            "continued_total_n": len(cont),
            "correct_flagged_n": fp_count,
            "correct_total_n": len(corr),
        })

    pd.DataFrame(score_rows).to_csv(RESULTS_DIR / "reestablishment_trigger_scores.csv", index=False)


if __name__ == "__main__":
    main()
