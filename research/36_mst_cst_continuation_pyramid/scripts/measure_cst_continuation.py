"""First-pass: how often does the trend continue meaningfully after a CST fires?

For each CST trigger inside an active MST trend on NIFTY 30-min p21,m5.0 (with
break-of-extreme entry confirmed), measure:

  - Continuation pts: max favorable price travel from CST close to trend end,
                       in the MST direction
  - Reversal pts: max adverse price travel from CST close to trend end
  - Outcome label:
      'TREND_CONTINUED'  if continuation > reversal AND continuation >= ATR
      'CST_CORRECT'      if reversal >= 1*ATR before any further continuation,
                          or trend ended within reversal_window
      'NEUTRAL'          otherwise (small move both ways, no clear signal)

Output: per-CST table + aggregate stats.
"""
from __future__ import annotations
import sys, sqlite3, csv
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

# Reuse research/35 helpers
sys.path.insert(0, str(ROOT / "research" / "35_nifty_bnf_master_child_supertrend" / "scripts"))
from supertrend import supertrend, stochastic, atr_wilder  # noqa
from run_mst_sweep import get_ohlc, compute_segments  # noqa
from run_mst_sweep_breakout import apply_breakout_filter  # noqa


def main():
    df = get_ohlc("NIFTY50", "30min")
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    # MST signals + break-of-extreme entries
    mst_dir, _, _ = supertrend(high, low, close, 21, 5.0)
    raw_segs = compute_segments(df, mst_dir)
    bo_segs = apply_breakout_filter(df, raw_segs)
    print(f"Total MST trends (close-based): {len(raw_segs)}")
    print(f"Trends after break-of-extreme filter: {len(bo_segs)}")

    # Stochastic
    k, d = stochastic(high, low, close, 14, 3, 3)

    # ATR for sizing thresholds
    atr = atr_wilder(high, low, close, 21)

    cst_rows = []
    closed = bo_segs.iloc[:-1] if len(bo_segs) > 1 else bo_segs

    for seg_idx, seg in closed.iterrows():
        s, e, dirn = int(seg.start_idx), int(seg.end_idx), int(seg.direction)
        seg_entry_close = close[s]
        seg_max_h = high[s:e + 1].max()
        seg_min_l = low[s:e + 1].min()

        for i in range(s + 1, e + 1):
            if np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(k[i - 1]):
                continue
            is_cst = False
            if dirn == 1:  # long MST
                is_cst = (k[i - 1] >= d[i - 1] and k[i] < d[i] and k[i - 1] >= 80)
            else:  # short MST
                is_cst = (k[i - 1] <= d[i - 1] and k[i] > d[i] and k[i - 1] <= 20)
            if not is_cst:
                continue

            cst_close = close[i]
            cst_atr = atr[i] if not np.isnan(atr[i]) else 0
            # From cst_idx+1 to end-of-trend
            if i + 1 > e:
                continue
            future_high = high[i + 1:e + 1].max()
            future_low = low[i + 1:e + 1].min()
            continuation_pts = (future_high - cst_close) if dirn == 1 else (cst_close - future_low)
            reversal_pts = (cst_close - future_low) if dirn == 1 else (future_high - cst_close)

            # Label
            if continuation_pts >= cst_atr and continuation_pts > reversal_pts:
                outcome = "TREND_CONTINUED"
            elif reversal_pts >= cst_atr and reversal_pts >= continuation_pts:
                outcome = "CST_CORRECT"
            else:
                outcome = "NEUTRAL"

            # MFE position: was the trend's overall MFE reached BEFORE or AFTER this CST?
            # If after CST, this CST was premature.
            if dirn == 1:
                mfe_before_cst = high[s:i + 1].max() - seg_entry_close
                mfe_after_cst = future_high - seg_entry_close
            else:
                mfe_before_cst = seg_entry_close - low[s:i + 1].min()
                mfe_after_cst = seg_entry_close - future_low
            mfe_after_cst = max(mfe_after_cst, mfe_before_cst)
            mfe_extension_pct = (mfe_after_cst - mfe_before_cst) / max(mfe_before_cst, 1e-9)

            cst_rows.append({
                "seg_idx": seg_idx,
                "seg_dir": dirn,
                "seg_entry_dt": seg.start_dt,
                "seg_end_dt": seg.end_dt,
                "seg_bars": seg.bars,
                "cst_idx": i,
                "cst_dt": df.index[i],
                "cst_close": cst_close,
                "cst_atr": cst_atr,
                "continuation_pts": continuation_pts,
                "reversal_pts": reversal_pts,
                "outcome": outcome,
                "mfe_before_cst": mfe_before_cst,
                "mfe_after_cst": mfe_after_cst,
                "mfe_extension_pct": mfe_extension_pct,
            })

    df_csts = pd.DataFrame(cst_rows)
    df_csts.to_csv(RESULTS_DIR / "cst_continuation_per_trigger.csv", index=False)
    print(f"\nTotal CST triggers: {len(df_csts)}")
    print()

    # Aggregate
    print("=== Outcome distribution (all CSTs in active MST trends) ===")
    print(df_csts["outcome"].value_counts())
    print(df_csts["outcome"].value_counts(normalize=True).round(3) * 100)
    print()

    print("=== Continuation magnitude (TREND_CONTINUED only) ===")
    tc = df_csts[df_csts.outcome == "TREND_CONTINUED"]
    if not tc.empty:
        print(f"  N = {len(tc)}")
        print(f"  Continuation pts: mean={tc.continuation_pts.mean():.1f} median={tc.continuation_pts.median():.1f} max={tc.continuation_pts.max():.1f}")
        print(f"  Continuation in ATR units: mean={(tc.continuation_pts / tc.cst_atr).mean():.2f} median={(tc.continuation_pts / tc.cst_atr).median():.2f}")
        print(f"  MFE extension after CST: mean={tc.mfe_extension_pct.mean()*100:.1f}% median={tc.mfe_extension_pct.median()*100:.1f}%")
    print()

    print("=== Continuation magnitude (CST_CORRECT only) ===")
    cc = df_csts[df_csts.outcome == "CST_CORRECT"]
    if not cc.empty:
        print(f"  N = {len(cc)}")
        print(f"  Reversal pts: mean={cc.reversal_pts.mean():.1f} median={cc.reversal_pts.median():.1f}")
        print(f"  Continuation pts (residual): mean={cc.continuation_pts.mean():.1f}")

    # First-CST-per-trend filter — most relevant since condor is built on first CST
    df_first = df_csts.groupby("seg_idx").first().reset_index()
    print(f"\n=== Outcome of FIRST CST per trend (n={len(df_first)}) — what condor experiences ===")
    print(df_first["outcome"].value_counts())
    print(df_first["outcome"].value_counts(normalize=True).round(3) * 100)
    df_first.to_csv(RESULTS_DIR / "cst_continuation_first_per_trend.csv", index=False)


if __name__ == "__main__":
    main()
