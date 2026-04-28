"""Phase-3 ranking: aggregate Phase-2 exits and rank.

Reads:  results/phase2_exits.csv
Writes: results/phase3_ranking.csv  (full ranked grid)
        results/RESULTS.md          (top configurations per strategy)

Per (path, variant, symbol, timeframe, exit_policy, exit_params):
- n_signals
- mean(net_pts), std, median, p25, p75
- win_rate (% net_pts > 0)
- pct_mae_against_le_threshold — % of signals where the EXIT was triggered
  by hitting the hard SL first (sl_hit_first==True). For T0 this is 0; for
  T1 it directly counts hard-SL stops.
  We surface this as a stop-out proxy.
- composite_score = mean - 0.5 * std  (primary rank)
- tiebreak = pct_sl_first ascending (lower stop-out is better)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE.parent / "results"
EXITS_CSV = RESULTS_DIR / "phase2_exits.csv"
RANK_CSV = RESULTS_DIR / "phase3_ranking.csv"
RESULTS_MD = RESULTS_DIR / "RESULTS.md"


def _df_to_md(df: pd.DataFrame) -> str:
    """Manual markdown table writer (avoids tabulate dependency)."""
    if df.empty:
        return "_(no rows)_"
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(_fmt_cell(r[c]) for c in cols) + " |")
    return "\n".join([head, sep] + rows)


def _fmt_cell(v) -> str:
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        return f"{v:.3f}"
    return str(v)


def main():
    print(f"Loading exits from {EXITS_CSV}...", flush=True)
    df = pd.read_csv(EXITS_CSV)
    print(f"  {len(df)} rows  signals={df['signal_id'].nunique()}", flush=True)

    df["net_pts"] = df["net_pts"].astype(float)
    df["sl_hit_first"] = df["sl_hit_first"].astype(str).str.lower().isin(("true", "1"))

    group_keys = ["path", "variant", "symbol", "timeframe",
                  "exit_policy", "exit_params"]

    print("Aggregating...", flush=True)
    g = df.groupby(group_keys)
    agg = g["net_pts"].agg(
        n_signals="count",
        mean_net="mean",
        std_net="std",
        median_net="median",
        p25_net=lambda s: s.quantile(0.25),
        p75_net=lambda s: s.quantile(0.75),
        win_rate=lambda s: (s > 0).mean() * 100,
    )
    agg["pct_sl_first"] = g["sl_hit_first"].mean() * 100
    agg = agg.reset_index()
    agg["std_net"] = agg["std_net"].fillna(0.0)

    # Composite: penalised mean
    agg["composite_score"] = agg["mean_net"] - 0.5 * agg["std_net"]

    # Sort
    agg = agg.sort_values(["composite_score", "pct_sl_first"],
                          ascending=[False, True])

    # round display columns
    for c in ["mean_net", "std_net", "median_net", "p25_net", "p75_net",
              "win_rate", "pct_sl_first", "composite_score"]:
        agg[c] = agg[c].round(3)

    agg.to_csv(RANK_CSV, index=False)
    print(f"Wrote {RANK_CSV}  ({len(agg)} rows)", flush=True)

    # Build RESULTS.md
    lines = [
        "# Short-Options Signal Sweep — RESULTS",
        "",
        f"Total exit rows: **{len(df):,}**  •  Total signals: **{df['signal_id'].nunique():,}**",
        f"Total ranked configurations: **{len(agg):,}**",
        "",
        "Composite score = `mean(net_pts) - 0.5 × std(net_pts)`. ",
        "`pct_sl_first` = percent of signals where the policy's hard-SL was hit first (lower = fewer stop-outs).",
        "",
        "## Top 5 configurations across ALL strategies",
        "",
    ]
    cols = ["path", "variant", "symbol", "timeframe", "exit_policy", "exit_params",
            "n_signals", "mean_net", "std_net", "win_rate",
            "pct_sl_first", "composite_score"]
    top5 = agg.head(5)[cols]
    lines.append(_df_to_md(top5))
    lines.append("")

    # Per-path top 5
    for path in sorted(agg["path"].unique()):
        sub = agg[agg["path"] == path].head(5)
        if sub.empty:
            continue
        lines.append(f"## Top 5 — Path **{path}**")
        lines.append("")
        lines.append(_df_to_md(sub[cols]))
        lines.append("")

    # Per-path SUMMARY: best policy per (path, exit_policy)
    lines.append("## Best mean-net per (path × exit_policy)")
    lines.append("")
    by_pol = (agg.groupby(["path", "exit_policy"])["composite_score"].max()
              .reset_index().sort_values(["path", "exit_policy"]))
    lines.append(_df_to_md(by_pol))
    lines.append("")

    # Path-level T0 baseline (raw signal quality) summary
    lines.append("## Strategy quality at T0 (time-only) — best variant per path")
    lines.append("")
    t0 = agg[agg["exit_policy"] == "T0"].copy()
    if not t0.empty:
        # best per path by composite
        t0_best = t0.sort_values("composite_score", ascending=False).groupby("path").head(1)
        lines.append(_df_to_md(t0_best[cols]))
    lines.append("")

    # Honest read
    lines.append("## Honest read")
    lines.append("")
    lines.append(_interpret(agg))
    lines.append("")

    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS_MD}")


def _interpret(agg: pd.DataFrame) -> str:
    """Auto-generated honest read of the rankings."""
    lines = []
    # Best overall
    top = agg.iloc[0]
    lines.append(
        f"- **Best overall config:** Path {top['path']} / `{top['variant']}` / "
        f"{top['symbol']} {top['timeframe']} with `{top['exit_policy']}` "
        f"({top['exit_params']}) — n={int(top['n_signals'])}, "
        f"mean={top['mean_net']:.2f}, std={top['std_net']:.2f}, "
        f"win-rate={top['win_rate']:.1f}%, composite={top['composite_score']:.2f}."
    )

    # Per-path best mean at T0 (signal quality)
    t0 = agg[agg["exit_policy"] == "T0"]
    if not t0.empty:
        best_t0 = t0.sort_values("composite_score", ascending=False).groupby("path").head(1)
        lines.append("- **Raw signal quality (T0 time-only):**")
        for _, r in best_t0.iterrows():
            sign = "+" if r["mean_net"] >= 0 else ""
            lines.append(
                f"  - Path {r['path']} best: `{r['variant']}` / "
                f"{r['symbol']} {r['timeframe']} → mean={sign}{r['mean_net']:.2f}, "
                f"win-rate={r['win_rate']:.1f}%, n={int(r['n_signals'])}"
            )

    # Are exit policies actually helping?
    lines.append("- **Effect of exit policies:**")
    for path in sorted(agg["path"].unique()):
        sub = agg[agg["path"] == path]
        # Best policy for this path
        best = sub.sort_values("composite_score", ascending=False).iloc[0]
        # Best T0 for this path
        t0p = sub[sub["exit_policy"] == "T0"].sort_values("composite_score", ascending=False)
        if not t0p.empty:
            t0_best = t0p.iloc[0]
            delta = best["composite_score"] - t0_best["composite_score"]
            lines.append(
                f"  - Path {path}: best policy `{best['exit_policy']}` "
                f"({best['exit_params']}) lifts composite by {delta:.2f} vs T0 "
                f"(mean: {best['mean_net']:.2f} vs {t0_best['mean_net']:.2f})."
            )

    # Win-rate filter for short-options viability
    lines.append("- **Short-options viability proxy:**")
    lines.append(
        "  For shorting OTM options, we want LOW adverse excursion (signals "
        "that don't whip against direction) and HIGH probability the underlying "
        "stays with us through EOD. Configs with high `win_rate` and small "
        "absolute `mean_net` (drift) are *more* viable than configs with high "
        "mean_net and high std (large directional moves cut both ways for an "
        "option seller, but only one side benefits)."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
