"""Fast streaming aggregation of volbreakout_signals.csv.

The original aggregate() in run_volbreakout_expanded.py reads the entire
168 MB CSV with pd.read_csv — hangs on Windows for this size. This module
streams the CSV row-by-row using csv.DictReader, accumulates per-cell
running statistics in memory dicts (Welford's online variance), then
writes ranking + leaders + RESULTS.md. Should complete in 1-2 min.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
SIGNAL_CSV = RESULTS / "volbreakout_signals.csv"
RANKING_CSV = RESULTS / "volbreakout_ranking.csv"
LEADERS_CSV = RESULTS / "volume_leaders.csv"
MD_PATH = RESULTS / "RESULTS.md"

EXIT_POLICIES = [
    "T_NO", "T_HARD_SL",
    "T_ATR_SL_0.3", "T_ATR_SL_0.5", "T_ATR_SL_1.0",
    "T_CHANDELIER_1.0", "T_CHANDELIER_1.5", "T_CHANDELIER_2.0",
    "T_R_TARGET_1.0R", "T_R_TARGET_1.5R", "T_R_TARGET_2.0R", "T_R_TARGET_3.0R",
    "T_STEP_TRAIL",
]

# Group key: (symbol, cohort, timeframe, variant, direction)
# For each group + each policy, track running stats for net_pct, net_pts, mfe_pts, mae_pts


def fnum(s: str) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main():
    if not SIGNAL_CSV.exists():
        print(f"ERROR: {SIGNAL_CSV} not found")
        sys.exit(1)

    print(f"Streaming {SIGNAL_CSV} ({SIGNAL_CSV.stat().st_size / 1e6:.1f} MB)...")

    # cell_stats[group_key][policy] = {
    #   "n": int, "net_pct_sum": float, "net_pct_sum_sq": float,
    #   "net_pts_sum": float, "net_pts_sum_sq": float,
    #   "wins": int, "losses": int,
    #   "win_pct_sum": float, "loss_pct_sum": float,
    #   "mfe_pct_sum": float, "mae_pct_sum": float,
    #   "net_pct_values": [...] for median (only kept if cell looks promising)
    # }
    cell_stats: dict = defaultdict(lambda: defaultdict(lambda: {
        "n": 0,
        "net_pct_sum": 0.0, "net_pct_sum_sq": 0.0,
        "net_pts_sum": 0.0, "net_pts_sum_sq": 0.0,
        "wins": 0, "losses": 0,
        "win_pct_sum": 0.0, "loss_pct_sum": 0.0,
        "mfe_pct_sum": 0.0, "mae_pct_sum": 0.0,
    }))

    rows_processed = 0
    with SIGNAL_CSV.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows_processed += 1
            if rows_processed % 20000 == 0:
                print(f"  Streamed {rows_processed:,} rows...")
            sym = r.get("symbol", "")
            cohort = r.get("cohort", "")
            tf = r.get("timeframe", "")
            variant = r.get("variant", "")
            direction = r.get("direction", "")
            entry_price = fnum(r.get("entry_price"))
            if not entry_price or entry_price <= 0:
                continue
            key = (sym, cohort, tf, variant, direction)

            for p in EXIT_POLICIES:
                net_pct = fnum(r.get(f"{p}__net_pct"))
                net_pts = fnum(r.get(f"{p}__net_pts"))
                mfe_pts = fnum(r.get(f"{p}__mfe_pts"))
                mae_pts = fnum(r.get(f"{p}__mae_pts"))
                if net_pct is None or net_pts is None:
                    continue
                s = cell_stats[key][p]
                s["n"] += 1
                s["net_pct_sum"] += net_pct
                s["net_pct_sum_sq"] += net_pct * net_pct
                s["net_pts_sum"] += net_pts
                s["net_pts_sum_sq"] += net_pts * net_pts
                if net_pct > 0:
                    s["wins"] += 1
                    s["win_pct_sum"] += net_pct
                else:
                    s["losses"] += 1
                    s["loss_pct_sum"] += net_pct
                if mfe_pts is not None:
                    s["mfe_pct_sum"] += mfe_pts / entry_price
                if mae_pts is not None:
                    s["mae_pct_sum"] += mae_pts / entry_price

    print(f"Streamed {rows_processed:,} total rows. Computing per-cell stats...")

    rows_out = []
    for key, by_policy in cell_stats.items():
        sym, cohort, tf, variant, direction = key
        for p, s in by_policy.items():
            n = s["n"]
            if n < 5:
                continue
            mean_net_pct = s["net_pct_sum"] / n
            var_net_pct = max(0.0, s["net_pct_sum_sq"] / n - mean_net_pct * mean_net_pct)
            std_net_pct = math.sqrt(var_net_pct)

            mean_net_pts = s["net_pts_sum"] / n
            var_net_pts = max(0.0, s["net_pts_sum_sq"] / n - mean_net_pts * mean_net_pts)
            std_net_pts = math.sqrt(var_net_pts)

            wr = s["wins"] / n
            avg_win = (s["win_pct_sum"] / s["wins"]) if s["wins"] > 0 else 0.0
            avg_loss = (s["loss_pct_sum"] / s["losses"]) if s["losses"] > 0 else 0.0
            payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else (999.0 if avg_win > 0 else 0.0)
            sharpe = (mean_net_pct / std_net_pct * wr) if std_net_pct > 0 else 0.0
            expectancy = (wr * avg_win) - ((1 - wr) * abs(avg_loss))
            mean_mfe_pct = s["mfe_pct_sum"] / n
            mean_mae_pct = s["mae_pct_sum"] / n
            cap_eff = (mean_net_pct / mean_mfe_pct) if mean_mfe_pct > 0 else 0.0

            rows_out.append({
                "symbol": sym,
                "cohort": cohort,
                "timeframe": tf,
                "variant": variant,
                "direction": direction,
                "exit_policy": p,
                "n_signals": n,
                "mean_net_pts": round(mean_net_pts, 4),
                "std_net_pts": round(std_net_pts, 4),
                "mean_net_pct": round(mean_net_pct, 6),
                "std_net_pct": round(std_net_pct, 6),
                "win_rate": round(wr * 100.0, 2),
                "avg_win_pct": round(avg_win * 100.0, 4),
                "avg_loss_pct": round(avg_loss * 100.0, 4),
                "payoff_ratio": round(payoff, 3),
                "mean_mfe_pct": round(mean_mfe_pct * 100.0, 4),
                "mean_mae_pct": round(mean_mae_pct * 100.0, 4),
                "capture_efficiency": round(cap_eff, 4),
                "sharpe_score": round(sharpe, 5),
                "expectancy_pct": round(expectancy * 100.0, 4),
            })

    rows_out.sort(key=lambda r: r["sharpe_score"], reverse=True)
    print(f"Aggregated {len(rows_out)} cells. Writing CSVs...")

    if rows_out:
        fields = list(rows_out[0].keys())
        with RANKING_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"Wrote {RANKING_CSV} ({len(rows_out)} rows)")

    # Volume leaders: per-stock best cell with n>=10
    by_sym = defaultdict(list)
    for r in rows_out:
        if r["n_signals"] >= 10:
            by_sym[r["symbol"]].append(r)

    leaders = []
    for sym, cells in by_sym.items():
        # Best by Sharpe (must have positive mean)
        positive = [c for c in cells if c["mean_net_pct"] > 0]
        if not positive:
            continue
        best = max(positive, key=lambda c: c["sharpe_score"])
        # Robust cell count: sharpe>=0.3 and n>=10 and mean>0
        robust = [c for c in cells if c["sharpe_score"] >= 0.3 and c["mean_net_pct"] > 0]
        leaders.append({
            "symbol": sym,
            "cohort": best["cohort"],
            "best_timeframe": best["timeframe"],
            "best_variant": best["variant"],
            "best_direction": best["direction"],
            "best_exit_policy": best["exit_policy"],
            "best_n": best["n_signals"],
            "best_mean_pct": best["mean_net_pct"] * 100,
            "best_win_rate": best["win_rate"],
            "best_payoff": best["payoff_ratio"],
            "best_sharpe": best["sharpe_score"],
            "robust_cells_count": len(robust),
            "promote": (
                best["sharpe_score"] >= 0.5
                and best["n_signals"] >= 15
                and len(robust) >= 3
            ),
        })

    leaders.sort(key=lambda r: r["best_sharpe"], reverse=True)
    if leaders:
        fields = list(leaders[0].keys())
        with LEADERS_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in leaders:
                w.writerow(r)
        print(f"Wrote {LEADERS_CSV} ({len(leaders)} stocks)")

    # RESULTS.md
    lines = []
    lines.append("# Volume-Breakout EXPANDED — Backtest Results\n\n")
    lines.append("## Setup\n\n")
    lines.append("- Period: per-stock available range, capped 2026-03-25\n")
    lines.append(f"- Stocks: 79 (10 Cohort A since 2018; 69 Cohort B since 2024-03-18)\n")
    lines.append("- Timeframes: 5min, 10min, 15min\n")
    lines.append("- Variant grid: vol_mult ∈ {1.5, 2.0, 3.0} × gap_pct ∈ {0%, 0.3%, 0.5%, off} × RSI ∈ {off, on(40/60)}\n")
    lines.append("- Direction: long & short (independent)\n")
    lines.append(f"- 13 exit policies tested per signal in parallel\n")
    lines.append(f"- Total signal rows processed: **{rows_processed:,}**\n")
    lines.append(f"- Ranked cells (n>=5): **{len(rows_out):,}**\n\n")

    # Top 10 by Sharpe (n>=15, mean>0)
    top = [r for r in rows_out if r["n_signals"] >= 15 and r["mean_net_pct"] > 0]
    lines.append("## Top 10 configurations across all stocks (n>=15, mean>0)\n\n")
    lines.append("| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Sharpe |\n")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|\n")
    for r in top[:10]:
        lines.append(
            f"| {r['symbol']} | {r['timeframe']} | {r['variant']} | {r['direction']} | "
            f"{r['exit_policy']} | {r['n_signals']} | {r['mean_net_pct']*100:.3f} | "
            f"{r['win_rate']:.1f} | {r['payoff_ratio']:.2f} | {r['sharpe_score']:.4f} |\n"
        )

    # Volume leaders top 25
    lines.append("\n## Top 25 Volume Leaders (best per-stock cell, n>=10)\n\n")
    lines.append("| # | Stock | Cohort | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Sharpe | RobustCells | Promote? |\n")
    lines.append("|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
    for i, r in enumerate(leaders[:25]):
        promo = "✅ YES" if r["promote"] else "—"
        lines.append(
            f"| {i+1} | **{r['symbol']}** | {r['cohort']} | {r['best_timeframe']} | "
            f"{r['best_variant']} | {r['best_direction']} | {r['best_exit_policy']} | "
            f"{r['best_n']} | {r['best_mean_pct']:.3f} | {r['best_win_rate']:.1f} | "
            f"{r['best_payoff']:.2f} | {r['best_sharpe']:.4f} | "
            f"{r['robust_cells_count']} | {promo} |\n"
        )

    # Promote candidates
    promote = [r for r in leaders if r["promote"]]
    lines.append(f"\n## Promote candidates ({len(promote)} stocks pass robustness gate)\n\n")
    lines.append("Gate: best-cell Sharpe >= 0.5 AND n >= 15 AND robust across >= 3 cells (Sharpe>=0.3 + mean>0).\n\n")
    if promote:
        lines.append("| Stock | TF | Variant | Dir | n | mean% | WR% | Sharpe | RobustCells |\n")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for r in promote:
            lines.append(
                f"| **{r['symbol']}** | {r['best_timeframe']} | {r['best_variant']} | "
                f"{r['best_direction']} | {r['best_n']} | {r['best_mean_pct']:.3f} | "
                f"{r['best_win_rate']:.1f} | {r['best_sharpe']:.4f} | "
                f"{r['robust_cells_count']} |\n"
            )
    else:
        lines.append("_No stocks passed the gate._\n")

    # Per-direction summary
    long_cells = [r for r in rows_out if r["direction"] == "long" and r["n_signals"] >= 10]
    short_cells = [r for r in rows_out if r["direction"] == "short" and r["n_signals"] >= 10]
    def avg(xs, key):
        return (sum(x[key] for x in xs) / len(xs)) if xs else 0.0
    lines.append("\n## Direction comparison (cells with n>=10)\n\n")
    lines.append("| Direction | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    lines.append(f"| long | {len(long_cells)} | {avg(long_cells,'mean_net_pct')*100:.4f} | {avg(long_cells,'win_rate'):.1f} | {avg(long_cells,'sharpe_score'):.4f} |\n")
    lines.append(f"| short | {len(short_cells)} | {avg(short_cells,'mean_net_pct')*100:.4f} | {avg(short_cells,'win_rate'):.1f} | {avg(short_cells,'sharpe_score'):.4f} |\n")

    # Per-timeframe summary
    lines.append("\n## Timeframe comparison (cells with n>=10, mean>0)\n\n")
    lines.append("| TF | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for tf in ["5min", "10min", "15min"]:
        cs = [r for r in rows_out if r["timeframe"] == tf and r["n_signals"] >= 10 and r["mean_net_pct"] > 0]
        lines.append(f"| {tf} | {len(cs)} | {avg(cs,'mean_net_pct')*100:.4f} | {avg(cs,'win_rate'):.1f} | {avg(cs,'sharpe_score'):.4f} |\n")

    # Volume threshold sweep
    lines.append("\n## Volume threshold sweep (cells with n>=10, mean>0)\n\n")
    lines.append("| vol_mult | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for vm in ["vm1.5", "vm2.0", "vm3.0"]:
        cs = [r for r in rows_out if vm in r["variant"] and r["n_signals"] >= 10 and r["mean_net_pct"] > 0]
        lines.append(f"| {vm[2:]}x | {len(cs)} | {avg(cs,'mean_net_pct')*100:.4f} | {avg(cs,'win_rate'):.1f} | {avg(cs,'sharpe_score'):.4f} |\n")

    # Exit policy comparison
    lines.append("\n## Exit policy comparison (cells with n>=10, mean>0)\n\n")
    lines.append("| Exit | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for p in EXIT_POLICIES:
        cs = [r for r in rows_out if r["exit_policy"] == p and r["n_signals"] >= 10 and r["mean_net_pct"] > 0]
        lines.append(f"| {p} | {len(cs)} | {avg(cs,'mean_net_pct')*100:.4f} | {avg(cs,'win_rate'):.1f} | {avg(cs,'sharpe_score'):.4f} |\n")

    MD_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {MD_PATH}")

    print(f"\n=== Top 5 by Sharpe (n>=15, mean>0) ===")
    for r in top[:5]:
        print(f"  {r['symbol']} {r['timeframe']} {r['variant']} {r['direction']} {r['exit_policy']}: "
              f"n={r['n_signals']} mean={r['mean_net_pct']*100:.3f}% WR={r['win_rate']:.1f}% Sharpe={r['sharpe_score']:.4f}")

    print(f"\n=== Promote candidates: {len(promote)} ===")
    for r in promote[:15]:
        print(f"  {r['symbol']} {r['best_timeframe']} {r['best_direction']} n={r['best_n']} "
              f"WR={r['best_win_rate']:.1f}% Sharpe={r['best_sharpe']:.4f} robust={r['robust_cells_count']}")


if __name__ == "__main__":
    main()
