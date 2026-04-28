"""Per-variant aggregate stats from phase1_signals.csv.

Prints a markdown table ranked by `score = mean(net) - 0.5 * std(net)` and a
second table sorted by signal-favourability for shorting options
(% of signals with MAE_against <= SL pts).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "results" / "phase1_signals.csv"


def main(sl_pts: float = 30.0) -> None:
    df = pd.read_csv(CSV)
    if df.empty:
        print("phase1_signals.csv is empty — run run_phase1.py first.")
        return

    df["mae_against"] = df["mae_against"].astype(float)
    df["net_at_eod"] = df["net_at_eod"].astype(float)
    df["mfe_with"] = df["mfe_with"].astype(float)
    df["mae_before_mfe"] = df["mae_before_mfe"].astype(str).str.lower().isin({"true", "1"})

    g = df.groupby(["path", "variant"])
    out = pd.DataFrame({
        "n": g.size(),
        "n_long": g.apply(lambda x: (x["direction"] == "long").sum()),
        "n_short": g.apply(lambda x: (x["direction"] == "short").sum()),
        "win_rate%": g.apply(lambda x: 100.0 * (x["net_at_eod"] > 0).mean()),
        "mean_net": g["net_at_eod"].mean(),
        "median_net": g["net_at_eod"].median(),
        "std_net": g["net_at_eod"].std(ddof=0),
        "mean_mae": g["mae_against"].mean(),
        "p50_mae": g["mae_against"].median(),
        "p90_mae": g["mae_against"].quantile(0.9),
        "mean_mfe": g["mfe_with"].mean(),
        f"%maele{int(sl_pts)}": g.apply(lambda x: 100.0 * (x["mae_against"] <= sl_pts).mean()),
        "%adv_first": g["mae_before_mfe"].mean().mul(100.0),
    })
    out["score"] = out["mean_net"] - 0.5 * out["std_net"]
    out = out.round(2)

    print("\n=== Ranked by score = mean(net) - 0.5 * std(net) ===\n")
    by_score = out.sort_values("score", ascending=False).reset_index()
    print(by_score.to_string(index=False))

    print("\n=== Ranked by % signals with MAE_against le "
          f"{int(sl_pts)} pts (short-options friendliness) ===\n")
    by_friendly = out.sort_values(f"%maele{int(sl_pts)}", ascending=False).reset_index()
    print(by_friendly.to_string(index=False))


if __name__ == "__main__":
    main()
