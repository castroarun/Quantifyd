"""CCRB — streaming aggregator.

Streams `results/ccrb_signals.csv` row-by-row (csv.DictReader),
accumulates per-cell stats with Welford-style running sums (mean, variance,
win_rate, MFE/MAE), and writes:

  - results/ccrb_ranking.csv  (per-cell aggregate; n>=5)
  - results/ccrb_leaders.csv  (per-stock best cell with n>=10 + robust count)
  - results/RESULTS.md        (final report + comparison to research/30b)

Sharpe-style score: (mean_net_pct / std_net_pct) * win_rate_fraction.

Cross-references against
  research/30b_volume_breakout_expanded/results/volume_leaders.csv
to identify CCRB-specific stocks vs volume-specific stocks vs robust-on-both.
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
SIGNAL_CSV = RESULTS / "ccrb_signals.csv"
RANKING_CSV = RESULTS / "ccrb_ranking.csv"
LEADERS_CSV = RESULTS / "ccrb_leaders.csv"
MD_PATH = RESULTS / "RESULTS.md"

VOL_LEADERS_30B = (ROOT.parent / "30b_volume_breakout_expanded"
                   / "results" / "volume_leaders.csv")

EXIT_POLICIES = [
    "T_NO", "T_HARD_SL",
    "T_ATR_SL_0.3", "T_ATR_SL_0.5", "T_ATR_SL_1.0",
    "T_CHANDELIER_1.0", "T_CHANDELIER_1.5", "T_CHANDELIER_2.0",
    "T_R_TARGET_1.0R", "T_R_TARGET_1.5R", "T_R_TARGET_2.0R", "T_R_TARGET_3.0R",
    "T_STEP_TRAIL",
]


def fnum(s):
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

    print(f"Streaming {SIGNAL_CSV} ({SIGNAL_CSV.stat().st_size / 1e6:.1f} MB)...",
          flush=True)

    cell_stats: dict = defaultdict(lambda: defaultdict(lambda: {
        "n": 0,
        "net_pct_sum": 0.0, "net_pct_sum_sq": 0.0,
        "net_pts_sum": 0.0, "net_pts_sum_sq": 0.0,
        "wins": 0, "losses": 0,
        "win_pct_sum": 0.0, "loss_pct_sum": 0.0,
        "mfe_pct_sum": 0.0, "mae_pct_sum": 0.0,
    }))

    rows_processed = 0
    n_long = 0
    n_short = 0
    by_cohort = {"A": 0, "B": 0}

    with SIGNAL_CSV.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows_processed += 1
            if rows_processed % 100000 == 0:
                print(f"  Streamed {rows_processed:,} rows...", flush=True)
            sym = r.get("symbol", "")
            cohort = r.get("cohort", "")
            tf = r.get("timeframe", "")
            variant = r.get("variant_tag", "")
            direction = r.get("direction", "")
            entry_price = fnum(r.get("entry_price"))
            if not entry_price or entry_price <= 0:
                continue
            if direction == "long":
                n_long += 1
            elif direction == "short":
                n_short += 1
            if cohort in by_cohort:
                by_cohort[cohort] += 1

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

    print(f"Streamed {rows_processed:,} total signal-row x policy entries.", flush=True)
    print(f"Computing per-cell stats over {len(cell_stats):,} cells...", flush=True)

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
    print(f"Aggregated {len(rows_out):,} cells (n>=5). Writing CSVs...", flush=True)

    if rows_out:
        fields = list(rows_out[0].keys())
        with RANKING_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"Wrote {RANKING_CSV} ({len(rows_out)} rows)")

    # Per-stock leaders: best cell with n>=10 (positive mean) + robust count
    by_sym = defaultdict(list)
    for r in rows_out:
        if r["n_signals"] >= 10:
            by_sym[r["symbol"]].append(r)

    leaders = []
    for sym, cells in by_sym.items():
        positive = [c for c in cells if c["mean_net_pct"] > 0]
        if not positive:
            continue
        best = max(positive, key=lambda c: c["sharpe_score"])
        robust = [c for c in cells
                  if c["sharpe_score"] >= 0.3 and c["mean_net_pct"] > 0]
        leaders.append({
            "symbol": sym,
            "cohort": best["cohort"],
            "best_timeframe": best["timeframe"],
            "best_variant": best["variant"],
            "best_direction": best["direction"],
            "best_exit_policy": best["exit_policy"],
            "best_n": best["n_signals"],
            "best_mean_pct": round(best["mean_net_pct"] * 100, 4),
            "best_win_rate": best["win_rate"],
            "best_payoff": best["payoff_ratio"],
            "best_sharpe": best["sharpe_score"],
            "robust_cells_count": len(robust),
            "promote": (best["sharpe_score"] >= 0.5
                        and best["n_signals"] >= 15
                        and len(robust) >= 3),
        })

    leaders.sort(key=lambda r: r["best_sharpe"], reverse=True)

    if leaders:
        fields = list(leaders[0].keys())
        with LEADERS_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in leaders:
                w.writerow(r)
        print(f"Wrote {LEADERS_CSV} ({len(leaders)} stocks)")

    # ----- Comparison to research/30b -----
    vol_leaders: dict[str, dict] = {}
    if VOL_LEADERS_30B.exists():
        with VOL_LEADERS_30B.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                vol_leaders[r["symbol"]] = r
    print(f"Loaded {len(vol_leaders)} stocks from research/30b vol leaders.")

    ccrb_leaders_by_sym = {r["symbol"]: r for r in leaders}

    comparison_rows = []
    in_both = sorted(set(ccrb_leaders_by_sym) & set(vol_leaders))
    only_ccrb = sorted(set(ccrb_leaders_by_sym) - set(vol_leaders))
    only_vol = sorted(set(vol_leaders) - set(ccrb_leaders_by_sym))
    for sym in in_both:
        c = ccrb_leaders_by_sym[sym]
        v = vol_leaders[sym]
        v_sharpe = float(v.get("best_sharpe", 0.0) or 0.0)
        c_sharpe = float(c["best_sharpe"])
        same_dir = (c["best_direction"] == v.get("best_direction", ""))
        same_tf = (c["best_timeframe"] == v.get("best_timeframe", ""))
        comparison_rows.append({
            "symbol": sym,
            "cohort": c["cohort"],
            "ccrb_tf": c["best_timeframe"],
            "ccrb_dir": c["best_direction"],
            "ccrb_n": c["best_n"],
            "ccrb_sharpe": c_sharpe,
            "vol_tf": v.get("best_timeframe", ""),
            "vol_dir": v.get("best_direction", ""),
            "vol_n": int(float(v.get("best_n", 0) or 0)),
            "vol_sharpe": v_sharpe,
            "same_direction": same_dir,
            "same_timeframe": same_tf,
            "ccrb_better": c_sharpe > v_sharpe,
            "delta": round(c_sharpe - v_sharpe, 4),
        })

    # ----- RESULTS.md -----
    lines = []
    lines.append("# CPR-Compression Range Breakout (CCRB) — Backtest Results\n\n")
    lines.append("## Setup\n\n")
    lines.append("- Universe: 79 stocks (10 Cohort A since 2018-01-01; 69 Cohort B since 2024-03-18)\n")
    lines.append("- Period end: 2026-03-25 (data cap)\n")
    lines.append("- Timeframes: 5min, 10min, 15min\n")
    lines.append("- Daily-bar setup filter:\n")
    lines.append("  - today_cpr_width / today_open <= today_narrow ∈ {0.30%, 0.40%, 0.50%}\n")
    lines.append("  - yesterday_ctx ∈ {W (wide CPR >= 0.50/0.65/0.80%), N (narrow range <= 0.50/0.70/0.90%), W_OR_N, W_AND_N}\n")
    lines.append("- Intraday trigger: first fresh transition past prev_day_high (long) / prev_day_low (short), 09:20-14:00 IST\n")
    lines.append("- Volume filter: off, vm1.5, vm2.0 (vs 20-day same-bar-position avg)\n")
    lines.append("- Direction: long & short (independent)\n")
    lines.append("- 13 exit policies tested per signal in parallel\n")
    lines.append(f"- Total signal rows: **{n_long + n_short:,}** (long: {n_long:,}, short: {n_short:,})\n")
    lines.append(f"- Cohort A signals: {by_cohort['A']:,}; Cohort B signals: {by_cohort['B']:,}\n")
    lines.append(f"- Ranked cells (n>=5): **{len(rows_out):,}**\n")
    lines.append(f"- Per-stock leaders found: **{len(leaders)}**\n\n")

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

    # Top 15 leaders
    lines.append("\n## Top 15 CPR-Compression Leaders (per-stock best cell, n>=10)\n\n")
    lines.append("| # | Stock | Cohort | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Sharpe | RobustCells | Promote? |\n")
    lines.append("|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
    for i, r in enumerate(leaders[:15]):
        promo = "YES" if r["promote"] else "-"
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
        lines.append("| Stock | TF | Variant | Dir | Exit | n | mean% | WR% | Sharpe | RobustCells |\n")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for r in promote:
            lines.append(
                f"| **{r['symbol']}** | {r['best_timeframe']} | {r['best_variant']} | "
                f"{r['best_direction']} | {r['best_exit_policy']} | {r['best_n']} | "
                f"{r['best_mean_pct']:.3f} | {r['best_win_rate']:.1f} | "
                f"{r['best_sharpe']:.4f} | {r['robust_cells_count']} |\n"
            )
    else:
        lines.append("_No stocks passed the gate._\n")

    # Direction summary
    long_cells = [r for r in rows_out if r["direction"] == "long" and r["n_signals"] >= 10]
    short_cells = [r for r in rows_out if r["direction"] == "short" and r["n_signals"] >= 10]
    def avg(xs, k):
        return (sum(x[k] for x in xs) / len(xs)) if xs else 0.0
    lines.append("\n## Direction comparison (cells with n>=10)\n\n")
    lines.append("| Direction | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    lines.append(f"| long | {len(long_cells)} | {avg(long_cells,'mean_net_pct')*100:.4f} | "
                 f"{avg(long_cells,'win_rate'):.1f} | {avg(long_cells,'sharpe_score'):.4f} |\n")
    lines.append(f"| short | {len(short_cells)} | {avg(short_cells,'mean_net_pct')*100:.4f} | "
                 f"{avg(short_cells,'win_rate'):.1f} | {avg(short_cells,'sharpe_score'):.4f} |\n")

    # Timeframe summary
    lines.append("\n## Timeframe comparison (cells with n>=10, mean>0)\n\n")
    lines.append("| TF | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for tf in ["5min", "10min", "15min"]:
        cs = [r for r in rows_out if r["timeframe"] == tf
              and r["n_signals"] >= 10 and r["mean_net_pct"] > 0]
        lines.append(f"| {tf} | {len(cs)} | {avg(cs,'mean_net_pct')*100:.4f} | "
                     f"{avg(cs,'win_rate'):.1f} | {avg(cs,'sharpe_score'):.4f} |\n")

    # today_narrow sweep
    lines.append("\n## today_narrow_threshold sweep (cells with n>=10, mean>0)\n\n")
    lines.append("| today_narrow | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for tn_label, tn_sub in [("0.30%", "t0.0030_"), ("0.40%", "t0.0040_"), ("0.50%", "t0.0050_")]:
        cs = [r for r in rows_out if r["variant"].startswith(tn_sub)
              and r["n_signals"] >= 10 and r["mean_net_pct"] > 0]
        lines.append(f"| {tn_label} | {len(cs)} | {avg(cs,'mean_net_pct')*100:.4f} | "
                     f"{avg(cs,'win_rate'):.1f} | {avg(cs,'sharpe_score'):.4f} |\n")

    # yesterday_ctx breakdown
    lines.append("\n## yesterday_ctx breakdown (cells with n>=10, mean>0)\n\n")
    lines.append("| ctx | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for ctx in ["ctxW_", "ctxN_", "ctxW_OR_N_", "ctxW_AND_N_"]:
        cs = [r for r in rows_out if ctx in r["variant"]
              and r["n_signals"] >= 10 and r["mean_net_pct"] > 0]
        lines.append(f"| {ctx[:-1]} | {len(cs)} | {avg(cs,'mean_net_pct')*100:.4f} | "
                     f"{avg(cs,'win_rate'):.1f} | {avg(cs,'sharpe_score'):.4f} |\n")

    # Volume mode breakdown
    lines.append("\n## Volume mode breakdown (cells with n>=10, mean>0)\n\n")
    lines.append("| vol_mode | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for vm in ["_off_", "_vm1.5_", "_vm2.0_"]:
        cs = [r for r in rows_out if vm in r["variant"]
              and r["n_signals"] >= 10 and r["mean_net_pct"] > 0]
        lines.append(f"| {vm.strip('_')} | {len(cs)} | {avg(cs,'mean_net_pct')*100:.4f} | "
                     f"{avg(cs,'win_rate'):.1f} | {avg(cs,'sharpe_score'):.4f} |\n")

    # Exit policy breakdown
    lines.append("\n## Exit policy comparison (cells with n>=10, mean>0)\n\n")
    lines.append("| Exit | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for p in EXIT_POLICIES:
        cs = [r for r in rows_out if r["exit_policy"] == p
              and r["n_signals"] >= 10 and r["mean_net_pct"] > 0]
        lines.append(f"| {p} | {len(cs)} | {avg(cs,'mean_net_pct')*100:.4f} | "
                     f"{avg(cs,'win_rate'):.1f} | {avg(cs,'sharpe_score'):.4f} |\n")

    # ----- Comparison vs research/30b -----
    lines.append("\n---\n\n## Comparison vs research/30b (volume-breakout)\n\n")
    lines.append(f"- Stocks in CCRB leaderboard: **{len(ccrb_leaders_by_sym)}**\n")
    lines.append(f"- Stocks in vol-breakout leaderboard: **{len(vol_leaders)}**\n")
    lines.append(f"- In both: **{len(in_both)}**  |  Only CCRB: {len(only_ccrb)}  |  Only vol-breakout: {len(only_vol)}\n\n")

    # CCRB-better
    ccrb_better = [c for c in comparison_rows if c["ccrb_better"]]
    vol_better = [c for c in comparison_rows if not c["ccrb_better"]]
    ccrb_better.sort(key=lambda c: c["delta"], reverse=True)
    vol_better.sort(key=lambda c: c["delta"])

    lines.append(f"### Where CCRB beats vol-breakout ({len(ccrb_better)} stocks)\n\n")
    lines.append("(These are CPR-compression specialists.)\n\n")
    lines.append("| Symbol | Coh | CCRB TF/Dir/n/Sharpe | Vol TF/Dir/n/Sharpe | SameDir | SameTF | Δ |\n")
    lines.append("|---|---|---|---|---|---|---:|\n")
    for c in ccrb_better[:30]:
        lines.append(
            f"| {c['symbol']} | {c['cohort']} | "
            f"{c['ccrb_tf']}/{c['ccrb_dir']}/{c['ccrb_n']}/{c['ccrb_sharpe']:.3f} | "
            f"{c['vol_tf']}/{c['vol_dir']}/{c['vol_n']}/{c['vol_sharpe']:.3f} | "
            f"{'Y' if c['same_direction'] else 'N'} | "
            f"{'Y' if c['same_timeframe'] else 'N'} | {c['delta']:+.3f} |\n"
        )

    lines.append(f"\n### Where vol-breakout beats CCRB ({len(vol_better)} stocks)\n\n")
    lines.append("(These are volume specialists.)\n\n")
    lines.append("| Symbol | Coh | CCRB TF/Dir/n/Sharpe | Vol TF/Dir/n/Sharpe | SameDir | SameTF | Δ |\n")
    lines.append("|---|---|---|---|---|---|---:|\n")
    for c in vol_better[:30]:
        lines.append(
            f"| {c['symbol']} | {c['cohort']} | "
            f"{c['ccrb_tf']}/{c['ccrb_dir']}/{c['ccrb_n']}/{c['ccrb_sharpe']:.3f} | "
            f"{c['vol_tf']}/{c['vol_dir']}/{c['vol_n']}/{c['vol_sharpe']:.3f} | "
            f"{'Y' if c['same_direction'] else 'N'} | "
            f"{'Y' if c['same_timeframe'] else 'N'} | {c['delta']:+.3f} |\n"
        )

    # Stocks with strong signals on BOTH
    both_strong = [c for c in comparison_rows
                   if c["ccrb_sharpe"] >= 0.4 and c["vol_sharpe"] >= 0.4]
    both_strong.sort(key=lambda c: c["ccrb_sharpe"] + c["vol_sharpe"], reverse=True)
    lines.append(f"\n### Stocks robust on BOTH (CCRB Sharpe >= 0.4 AND vol Sharpe >= 0.4): {len(both_strong)}\n\n")
    if both_strong:
        lines.append("| Symbol | Coh | CCRB Sharpe | Vol Sharpe | SameDir | SameTF |\n")
        lines.append("|---|---|---:|---:|---|---|\n")
        for c in both_strong:
            lines.append(
                f"| **{c['symbol']}** | {c['cohort']} | "
                f"{c['ccrb_sharpe']:.3f} | {c['vol_sharpe']:.3f} | "
                f"{'Y' if c['same_direction'] else 'N'} | "
                f"{'Y' if c['same_timeframe'] else 'N'} |\n"
            )

    # Only-CCRB stocks (in CCRB leaderboard but NOT in vol-breakout leaderboard)
    if only_ccrb:
        lines.append(f"\n### Stocks ONLY found by CCRB (not by vol-breakout): {len(only_ccrb)}\n\n")
        ccrb_only_rows = sorted(
            [ccrb_leaders_by_sym[s] for s in only_ccrb],
            key=lambda r: r["best_sharpe"], reverse=True
        )
        lines.append("| Symbol | Coh | TF | Dir | n | Sharpe | RobustCells |\n")
        lines.append("|---|---|---|---|---:|---:|---:|\n")
        for r in ccrb_only_rows[:20]:
            lines.append(
                f"| {r['symbol']} | {r['cohort']} | {r['best_timeframe']} | "
                f"{r['best_direction']} | {r['best_n']} | {r['best_sharpe']:.4f} | "
                f"{r['robust_cells_count']} |\n"
            )

    if only_vol:
        lines.append(f"\n### Stocks ONLY in vol-breakout leaderboard (no CCRB leader): {len(only_vol)}\n\n")
        lines.append(", ".join(only_vol[:30]) + ("..." if len(only_vol) > 30 else "") + "\n")

    # Honest read
    lines.append("\n---\n\n## Honest Read\n\n")
    if comparison_rows:
        # Mean Sharpe deltas
        mean_ccrb = sum(c["ccrb_sharpe"] for c in comparison_rows) / len(comparison_rows)
        mean_vol = sum(c["vol_sharpe"] for c in comparison_rows) / len(comparison_rows)
        # Promote counts
        ccrb_promote_n = sum(1 for r in leaders if r["promote"])
        # vol_promote_n = number of vol-leaders with promote==True (column may be string)
        vol_promote_n = 0
        for v in vol_leaders.values():
            p = v.get("promote", "False")
            if str(p).strip().lower() in ("true", "1", "yes"):
                vol_promote_n += 1
        lines.append(f"- Average best-cell Sharpe across stocks in BOTH leaderboards: "
                     f"CCRB={mean_ccrb:.3f} vs vol-breakout={mean_vol:.3f}\n")
        lines.append(f"- CCRB beats vol-breakout in {len(ccrb_better)} / {len(comparison_rows)} "
                     f"shared stocks ({len(ccrb_better)/max(1,len(comparison_rows))*100:.0f}%).\n")
        lines.append(f"- Promote-gate passes: CCRB={ccrb_promote_n}; vol-breakout={vol_promote_n}.\n")
        same_dir = sum(1 for c in comparison_rows if c["same_direction"])
        same_tf = sum(1 for c in comparison_rows if c["same_timeframe"])
        lines.append(f"- Direction agreement (same best dir): {same_dir} / {len(comparison_rows)} stocks.\n")
        lines.append(f"- Timeframe agreement (same best TF): {same_tf} / {len(comparison_rows)} stocks.\n\n")
    lines.append("Interpretation: CCRB and vol-breakout target different setups. CCRB requires a\n")
    lines.append("daily-bar geometric filter (CPR compression) that gates the day; vol-breakout\n")
    lines.append("triggers off the first candle's volume regardless of CPR. Stocks where they\n")
    lines.append("disagree on direction or timeframe are likely catching different regimes.\n")
    lines.append("Stocks robust on BOTH are the highest-conviction names — the daily geometric\n")
    lines.append("filter and the first-bar volume confirmation could be combined as a higher-bar\n")
    lines.append("entry rule.\n")

    MD_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {MD_PATH}")

    print(f"\n=== Top 5 by Sharpe (n>=15, mean>0) ===")
    for r in top[:5]:
        print(f"  {r['symbol']} {r['timeframe']} {r['variant']} {r['direction']} {r['exit_policy']}: "
              f"n={r['n_signals']} mean={r['mean_net_pct']*100:.3f}% WR={r['win_rate']:.1f}% "
              f"Sharpe={r['sharpe_score']:.4f}")
    print(f"\n=== Promote candidates: {sum(1 for r in leaders if r['promote'])} ===")
    for r in [r for r in leaders if r["promote"]][:15]:
        print(f"  {r['symbol']} {r['best_timeframe']} {r['best_direction']} n={r['best_n']} "
              f"WR={r['best_win_rate']:.1f}% Sharpe={r['best_sharpe']:.4f} "
              f"robust={r['robust_cells_count']}")


if __name__ == "__main__":
    main()
