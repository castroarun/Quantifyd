"""Test 4 variants of the D trigger to address the staircase gap.

Variants:
  D_strict     — 2 consecutive closes beyond CST level (current production)
  D_2of3       — 2 of last 3 bars closed beyond CST level (allows 1 calm gap)
  D_cumulative — net (above_count - below_count) >= 3 within 6-bar lookback
  D_new_high   — any close beyond CST level AND price made new post-CST high/low

Each tested ALONE and AND'd with B (Stoch %K back to OB/OS).
Same 6.3-year extended dataset, same MST cell as research/36 production.
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


def d_strict(close, dirn, cst_idx, end_idx, level):
    """2 consecutive closes beyond level."""
    streak = 0
    for i in range(cst_idx + 1, end_idx + 1):
        if (dirn == 1 and close[i] > level) or (dirn == -1 and close[i] < level):
            streak += 1
            if streak >= 2:
                return i
        else:
            streak = 0
    return None


def d_2of3(close, dirn, cst_idx, end_idx, level):
    """2 of the last 3 bars closed beyond level (allows 1 calm gap)."""
    for i in range(cst_idx + 3, end_idx + 1):
        window = close[i - 2:i + 1]  # last 3 bars including current
        if dirn == 1:
            beyond = sum(1 for c in window if c > level)
        else:
            beyond = sum(1 for c in window if c < level)
        # Require current bar to be one of the beyond closes
        cur_beyond = (close[i] > level) if dirn == 1 else (close[i] < level)
        if beyond >= 2 and cur_beyond:
            return i
    return None


def d_cumulative(close, dirn, cst_idx, end_idx, level, lookback=6, threshold=3):
    """Net (above - below) within lookback >= threshold AND current bar is above."""
    for i in range(cst_idx + 1, end_idx + 1):
        start = max(cst_idx + 1, i - lookback + 1)
        window = close[start:i + 1]
        if dirn == 1:
            above = sum(1 for c in window if c > level)
            below = sum(1 for c in window if c < level)
        else:
            above = sum(1 for c in window if c < level)
            below = sum(1 for c in window if c > level)
        cur_beyond = (close[i] > level) if dirn == 1 else (close[i] < level)
        if cur_beyond and (above - below) >= threshold:
            return i
    return None


def d_new_high(close, high, low, dirn, cst_idx, end_idx, level):
    """Close beyond level AND price made a new post-CST high (long) / low (short).
    The new H/L is the strict max/min of all post-CST bars including current.
    """
    if cst_idx + 1 > end_idx:
        return None
    if dirn == 1:
        running_max = high[cst_idx]
        for i in range(cst_idx + 1, end_idx + 1):
            new_high = high[i] > running_max
            running_max = max(running_max, high[i])
            close_beyond = close[i] > level
            if new_high and close_beyond:
                return i
    else:
        running_min = low[cst_idx]
        for i in range(cst_idx + 1, end_idx + 1):
            new_low = low[i] < running_min
            running_min = min(running_min, low[i])
            close_beyond = close[i] < level
            if new_low and close_beyond:
                return i
    return None


def main():
    df = get_ohlc("NIFTY50", "30min")
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    days = (df.index[-1] - df.index[0]).days
    print(f"Data: {len(df)} bars, {df.index.min()} -> {df.index.max()} ({days} days, ~{days/365.25:.1f} yrs)")

    mst_dir, _, _ = supertrend(high, low, close, 21, 5.0)
    raw_segs = compute_segments(df, mst_dir)
    bo_segs = apply_breakout_filter(df, raw_segs)
    closed = bo_segs.iloc[:-1]
    k, _ = stochastic(high, low, close, 14, 3, 3)
    d_arr_full = pd.Series(k).rolling(3).mean().values  # %D = SMA of %K

    cst_outcomes = {}
    df_csts = pd.read_csv(RESULTS_DIR / "extended_cst_continuation.csv")
    for _, row in df_csts.iterrows():
        cst_outcomes[(row["seg_idx"], row["cst_idx"])] = row["outcome"]

    # Build the trigger map
    variants = {
        "D_strict":     lambda dirn, cst, end, level, _h, _l: d_strict(close, dirn, cst, end, level),
        "D_2of3":       lambda dirn, cst, end, level, _h, _l: d_2of3(close, dirn, cst, end, level),
        "D_cumulative": lambda dirn, cst, end, level, _h, _l: d_cumulative(close, dirn, cst, end, level),
        "D_new_high":   lambda dirn, cst, end, level, h_arr, l_arr: d_new_high(close, h_arr, l_arr, dirn, cst, end, level),
    }

    results = {f"{name}_AND_B": {"continued": [], "correct": []} for name in variants}
    for name in variants:
        results[name + "_alone"] = {"continued": [], "correct": []}

    # Build dict of (seg_idx, cst_idx) -> CST event row + segment end
    for _, seg in closed.iterrows():
        s = int(seg.start_idx)
        e = int(seg.end_idx)
        dirn = int(seg.direction)
        for i in range(s + 1, e + 1):
            if np.isnan(k[i]) or np.isnan(d_arr_full[i]) or np.isnan(k[i - 1]):
                continue
            is_cst = ((dirn == 1 and k[i - 1] >= d_arr_full[i - 1] and k[i] < d_arr_full[i] and k[i - 1] >= 80) or
                      (dirn == -1 and k[i - 1] <= d_arr_full[i - 1] and k[i] > d_arr_full[i] and k[i - 1] <= 20))
            if not is_cst or i + 1 > e:
                continue
            outcome = cst_outcomes.get((seg.name, i))
            if outcome not in ("TREND_CONTINUED", "CST_CORRECT"):
                continue

            level = high[i] if dirn == 1 else low[i]
            if dirn == 1:
                mfe_idx = i + 1 + int(np.argmax(high[i + 1:e + 1]))
            else:
                mfe_idx = i + 1 + int(np.argmin(low[i + 1:e + 1]))

            tB = trigger_stoch_return_to_extreme(k, dirn, i, e)

            for name, fn in variants.items():
                t_d = fn(dirn, i, e, level, high, low)
                # Alone
                rec = {"trigger_idx": t_d,
                       "lead": (mfe_idx - t_d) if t_d is not None else None}
                bucket = "continued" if outcome == "TREND_CONTINUED" else "correct"
                results[name + "_alone"][bucket].append(rec)
                # AND B
                if t_d is None or tB is None:
                    t_and = None
                else:
                    t_and = max(t_d, tB)
                rec_and = {"trigger_idx": t_and,
                           "lead": (mfe_idx - t_and) if t_and is not None else None}
                results[name + "_AND_B"][bucket].append(rec_and)

    # Score
    print("\n" + "=" * 110)
    print(f"{'Variant':30} {'Coverage':>10} {'AvgLead':>10} {'MedLead':>10} {'FP_rate':>10} {'Score':>10}")
    print("=" * 110)
    rows = []
    for name in list(variants.keys()):
        for suffix in ("_alone", "_AND_B"):
            label = name + suffix
            cont = pd.DataFrame(results[label]["continued"])
            corr = pd.DataFrame(results[label]["correct"])
            flagged = cont[cont.trigger_idx.notna()] if not cont.empty else cont
            cov = len(flagged) / len(cont) if len(cont) else 0
            f2 = flagged[flagged.lead.notna()] if not flagged.empty else flagged
            avg_lead = f2.lead.mean() if len(f2) else 0
            med_lead = f2.lead.median() if len(f2) else 0
            fp = corr[corr.trigger_idx.notna()].shape[0] if not corr.empty else 0
            fp_rate = fp / len(corr) if len(corr) else 0
            useful = max(0, avg_lead)
            score = cov * useful * (1 - fp_rate)
            print(f"{label:30} {cov:>10.2%} {avg_lead:>10.2f} {med_lead:>10.2f} {fp_rate:>10.2%} {score:>10.3f}")
            rows.append(dict(variant=label, coverage=cov, avg_lead=avg_lead,
                             median_lead=med_lead, fp_rate=fp_rate, score=score,
                             n_continued_total=len(cont),
                             n_continued_flagged=len(f2),
                             n_correct_total=len(corr),
                             n_correct_flagged=fp))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "d_variant_scores.csv", index=False)
    print(f"\nWritten to {RESULTS_DIR / 'd_variant_scores.csv'}")


if __name__ == "__main__":
    main()
