"""Re-rank Phase 3 results with a Sharpe-style metric that's comparable
across asset scales (NIFTY pts vs stock pts).

Sharpe-style score: (mean_net / std_net) * win_rate_decimal

Why this works across scales:
- mean / std is unitless (Sharpe ratio core) — scale-invariant
- win_rate is a 0-1 fraction — also scale-invariant
- Product rewards configs that are BOTH directionally right (high WR)
  AND consistent (high mean/std)

Filters:
- min_n: drop configs with too few signals (statistically meaningless)
- mean_net > 0: only positive-edge configs (not interested in negative
  drift no matter how consistent)

Outputs:
- phase3_ranking_sharpe.csv (full re-rank)
- RESULTS_SHARPE.md (top 5 across all + per-strategy)
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Dict

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
RANKING_IN = ROOT / "results" / "phase3_ranking.csv"
RANKING_OUT = ROOT / "results" / "phase3_ranking_sharpe.csv"
RESULTS_OUT = ROOT / "results" / "RESULTS_SHARPE.md"


def load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                r["n_signals"] = int(r["n_signals"])
                r["mean_net"] = float(r["mean_net"])
                r["std_net"] = float(r["std_net"])
                r["win_rate"] = float(r["win_rate"])  # already 0-100
                r["pct_sl_first"] = float(r["pct_sl_first"])
                r["composite_score"] = float(r["composite_score"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows


def sharpe_score(r: Dict) -> float:
    """(mean / std) * win_rate_fraction. Penalises high-variance configs;
    rewards high-WR. Returns 0 for degenerate cases."""
    if r["std_net"] <= 0 or math.isnan(r["std_net"]):
        return 0.0
    mean_over_std = r["mean_net"] / r["std_net"]
    wr_frac = r["win_rate"] / 100.0
    return mean_over_std * wr_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=30,
                    help="Minimum signal count to be eligible (default 30)")
    ap.add_argument("--positive-only", action="store_true", default=True,
                    help="Require mean_net > 0 (default true)")
    args = ap.parse_args()

    rows = load_rows(RANKING_IN)
    print(f"Loaded {len(rows)} rows from {RANKING_IN.name}")

    eligible = []
    for r in rows:
        if r["n_signals"] < args.min_n:
            continue
        if args.positive_only and r["mean_net"] <= 0:
            continue
        r["sharpe_score"] = sharpe_score(r)
        eligible.append(r)
    print(f"Eligible after filters (n>={args.min_n}, mean>0): {len(eligible)}")

    eligible.sort(key=lambda r: r["sharpe_score"], reverse=True)

    # Write full re-ranked CSV
    fieldnames = [
        "path", "variant", "symbol", "timeframe", "exit_policy", "exit_params",
        "n_signals", "mean_net", "std_net", "win_rate", "pct_sl_first",
        "composite_score", "sharpe_score",
    ]
    with RANKING_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in eligible:
            w.writerow(r)
    print(f"Wrote {RANKING_OUT}")

    # Build markdown
    lines: List[str] = []
    lines.append("# Short-Options Signal Sweep — Re-ranked (Sharpe-style)\n")
    lines.append(f"Re-ranking metric: `(mean_net / std_net) × win_rate_fraction`. ")
    lines.append("Filters: n_signals ≥ {n}, mean_net > 0.\n".format(n=args.min_n))
    lines.append(f"Eligible configurations: **{len(eligible)}** of {len(rows)}.\n")

    def fmt_table(rows_subset: List[Dict]) -> str:
        out = []
        out.append("| path | variant | symbol | tf | exit | params | n | mean | std | WR% | Sharpe |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows_subset:
            out.append(
                f"| {r['path']} | {r['variant']} | {r['symbol']} | {r['timeframe']} "
                f"| {r['exit_policy']} | {r['exit_params']} "
                f"| {r['n_signals']} | {r['mean_net']:.2f} | {r['std_net']:.2f} "
                f"| {r['win_rate']:.1f} | {r['sharpe_score']:.4f} |"
            )
        return "\n".join(out)

    lines.append("\n## Top 10 across ALL strategies\n")
    lines.append(fmt_table(eligible[:10]))

    for path in ["A", "B", "C", "D", "E", "F"]:
        sub = [r for r in eligible if r["path"] == path][:5]
        if not sub:
            continue
        lines.append(f"\n## Top 5 — Path {path}\n")
        lines.append(fmt_table(sub))

    # Compare to composite-rank top
    lines.append("\n## Old composite-rank top 5 — for comparison\n")
    by_composite = sorted(rows, key=lambda r: r["composite_score"], reverse=True)
    sub = []
    for r in by_composite[:5]:
        r["sharpe_score"] = sharpe_score(r)
        sub.append(r)
    lines.append(fmt_table(sub))

    RESULTS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_OUT}")

    # Print top 5 to stdout
    print("\n=== Top 5 by Sharpe-style score ===")
    for i, r in enumerate(eligible[:5]):
        print(f"  {i+1}. {r['path']}/{r['variant']}/{r['symbol']}/{r['timeframe']} "
              f"{r['exit_policy']} {r['exit_params']} "
              f"n={r['n_signals']} mean={r['mean_net']:.2f} std={r['std_net']:.2f} "
              f"WR={r['win_rate']:.1f}% Sharpe={r['sharpe_score']:.4f}")


if __name__ == "__main__":
    main()
