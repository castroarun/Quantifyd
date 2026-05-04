"""Walk-forward selection on the locked spec — true out-of-sample test.

For each rebalance point T:
  1. Score each stock on the prior 6-month "lookback" window
  2. Apply the selection rule to pick the basket for the next quarter
  3. Trade that basket for the next 3-month "trade" window
  4. Accumulate the realized trades into the global equity curve
  5. Roll forward 3 months, repeat

This eliminates the in-sample selection bias of the headline 3.65 Sharpe.
With 24 months total data, 6mo lookback + 3mo trade gives 6 quarters of
true out-of-sample trading (months 7-24).

Selection rules tested in parallel:
  R1: Top 15 by trailing Sharpe (with min 5 trades)
  R2: Top 20 by trailing Sharpe (with min 5 trades)
  R3: Trailing Sharpe >= 1.0 AND >= 5 trades
  R4: Trailing Sharpe >= 1.5 AND >= 5 trades
  R5: Sticky — once selected, keep until trailing Sharpe < 0.5 (reduces churn)

Outputs:
  results/walkforward_summary.csv     — per-rule metrics
  results/walkforward_baskets.csv     — per-quarter basket per rule
  results/walkforward_equity.csv      — equity curves (one column per rule)
  results/walkforward_curves.png      — equity + drawdown chart per rule
  results/walkforward_summary.txt     — human-readable summary
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from final_run import (  # noqa: E402
    list_universe, prepare_30min, gen_setups_locked,
    UNIVERSE_START, UNIVERSE_END,
    SIDE, EXIT_ID, COST_PCT,
)
from param_sweep import sim  # noqa: E402

OUT_DIR = SCRIPT_DIR.parent / "results"
LOG_DIR = SCRIPT_DIR.parent / "logs"

LOOKBACK_MONTHS = 6
TRADE_MONTHS = 3
MIN_TRADES_LOOKBACK = 5

RULES = ["R1_top15", "R2_top20", "R3_sharpe1.0", "R4_sharpe1.5", "R5_sticky"]


def select_basket(rule: str, scores: pd.DataFrame, prev_basket: set[str]) -> list[str]:
    """Return the basket of stocks to trade given a per-stock score table."""
    qualified = scores[scores["trades"] >= MIN_TRADES_LOOKBACK].copy()
    if qualified.empty:
        return list(prev_basket)  # fall back to previous if nothing qualifies

    if rule == "R1_top15":
        return qualified.sort_values("sharpe", ascending=False).head(15)["symbol"].tolist()
    if rule == "R2_top20":
        return qualified.sort_values("sharpe", ascending=False).head(20)["symbol"].tolist()
    if rule == "R3_sharpe1.0":
        return qualified[qualified["sharpe"] >= 1.0]["symbol"].tolist()
    if rule == "R4_sharpe1.5":
        return qualified[qualified["sharpe"] >= 1.5]["symbol"].tolist()
    if rule == "R5_sticky":
        # Keep prev basket as long as their trailing Sharpe >= 0.5
        # Add new stocks meeting Sharpe >= 1.5
        keepers = set(qualified[qualified["sharpe"] >= 0.5]["symbol"].tolist()) & prev_basket
        new_picks = set(qualified[qualified["sharpe"] >= 1.5]["symbol"].tolist())
        return sorted(keepers | new_picks)
    raise ValueError(rule)


def per_stock_score(trades_in_window: list[float], avg_hold: float) -> dict:
    """Sharpe-style score on a list of per-trade returns."""
    n = len(trades_in_window)
    if n == 0:
        return {"trades": 0, "sharpe": 0.0, "total_ret": 0.0}
    arr = np.array(trades_in_window)
    mean_r = arr.mean()
    std_r = arr.std(ddof=1) if n > 1 else 0
    candles_per_year = 3125
    tpy = candles_per_year / max(avg_hold, 1)
    sharpe = (mean_r / std_r * math.sqrt(max(tpy, 1))) if std_r > 0 else 0
    return {"trades": n, "sharpe": float(sharpe), "total_ret": float(arr.sum())}


def main():
    log_path = LOG_DIR / "walkforward.log"
    log = log_path.open("w")
    def say(msg: str):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    universe = list_universe()
    say("=== WALK-FORWARD SELECTION ===")
    say(f"Period: {UNIVERSE_START} -> {UNIVERSE_END}")
    say(f"Universe: {len(universe)} stocks")
    say(f"Lookback: {LOOKBACK_MONTHS} months   Trade: {TRADE_MONTHS} months")
    say(f"Locked spec: gap=0, ADX>=25, Stoch(14,5,3) os=35, exit={EXIT_ID}")
    say(f"Cost: {COST_PCT}% per trade")
    say(f"Selection rules: {RULES}")
    say("")

    # Build rebalance schedule
    start = pd.Timestamp(UNIVERSE_START)
    end = pd.Timestamp(UNIVERSE_END)
    rebalance_points = []
    t = start + pd.DateOffset(months=LOOKBACK_MONTHS)
    while t < end:
        rebalance_points.append(t)
        t = t + pd.DateOffset(months=TRADE_MONTHS)
    say(f"Rebalance points: {len(rebalance_points)}")
    for rp in rebalance_points:
        trade_end = min(rp + pd.DateOffset(months=TRADE_MONTHS), end)
        say(f"  {rp.date()}  trade window -> {trade_end.date()}")
    say("")

    # ---- Pre-load all data + pre-generate ALL setups ----
    say("Phase 1: prepare data + pre-generate setups for all 79 stocks (one pass over 24 months)...")
    t0 = time.time()
    all_setups: dict[str, list[dict]] = {}
    prepped: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(universe, 1):
        df = prepare_30min(sym)
        if df.empty: continue
        prepped[sym] = df
        ss = gen_setups_locked(df)
        # Simulate each setup ONCE; store entry/exit + return
        sim_results = []
        for s in ss:
            ex_idx, ex_p, ex_r, held, ret = sim(
                s["df_ref"], SIDE, s["fill_idx"], s["fill_price"],
                s["anchor_low"], s["anchor_high"], EXIT_ID,
            )
            sim_results.append({
                "signal_dt": s["signal_dt"],
                "entry_dt": pd.Timestamp(s["df_ref"]["dt"].iloc[s["fill_idx"]]),
                "entry_price": s["fill_price"],
                "exit_dt": pd.Timestamp(s["df_ref"]["dt"].iloc[ex_idx]),
                "exit_price": ex_p,
                "exit_reason": ex_r,
                "candles_held": held,
                "net_ret": ret - COST_PCT,
            })
        all_setups[sym] = sim_results
        if i % 10 == 0:
            say(f"  {i}/{len(universe)}  ({time.time()-t0:.0f}s)")
    say(f"Pre-generated trades for {len(all_setups)} stocks in {time.time()-t0:.0f}s")
    n_total = sum(len(v) for v in all_setups.values())
    say(f"Total trades pre-generated across whole period: {n_total}")
    say("")

    # Convert each stock's trades to a DataFrame for fast filtering
    trades_dfs = {sym: pd.DataFrame(v) for sym, v in all_setups.items() if v}
    for sym, tdf in trades_dfs.items():
        tdf["entry_dt"] = pd.to_datetime(tdf["entry_dt"])
        tdf["exit_dt"] = pd.to_datetime(tdf["exit_dt"])

    # ---- Phase 2: walk-forward per rule ----
    say("Phase 2: walk-forward simulation per selection rule...")
    rule_results = {}    # rule -> dict
    rule_baskets = []    # list of dicts (for CSV)
    rule_trades = {}     # rule -> list of trade dicts (chronological)

    for rule in RULES:
        say(f"\n--- Rule: {rule} ---")
        rule_trades[rule] = []
        prev_basket: set[str] = set()
        for rp in rebalance_points:
            lookback_start = rp - pd.DateOffset(months=LOOKBACK_MONTHS)
            trade_end = min(rp + pd.DateOffset(months=TRADE_MONTHS), end)

            # Score each stock on lookback window (entry_dt within [lookback_start, rp))
            score_rows = []
            for sym, tdf in trades_dfs.items():
                lb = tdf[(tdf["entry_dt"] >= lookback_start) & (tdf["entry_dt"] < rp)]
                rets = lb["net_ret"].tolist()
                avg_hold = float(lb["candles_held"].mean()) if len(lb) > 0 else 1
                s = per_stock_score(rets, avg_hold)
                score_rows.append({"symbol": sym, **s})
            scores = pd.DataFrame(score_rows)

            # Apply selection rule
            basket = select_basket(rule, scores, prev_basket)

            # Trade this basket on [rp, trade_end)
            quarter_trades = []
            for sym in basket:
                tdf = trades_dfs.get(sym)
                if tdf is None: continue
                qt = tdf[(tdf["entry_dt"] >= rp) & (tdf["entry_dt"] < trade_end)]
                for _, row in qt.iterrows():
                    quarter_trades.append({
                        "symbol": sym,
                        "rebalance_dt": rp,
                        **row.to_dict(),
                    })
            rule_trades[rule].extend(quarter_trades)

            rule_baskets.append({
                "rule": rule,
                "rebalance_dt": rp,
                "trade_window_end": trade_end,
                "basket_size": len(basket),
                "stocks": ",".join(sorted(basket)),
                "quarter_trades": len(quarter_trades),
            })
            say(f"  {rp.date()} -> {trade_end.date()}: basket {len(basket):>2} stocks, "
                f"{len(quarter_trades):>3} trades   (score-window stocks scored: {len(scores)})")
            prev_basket = set(basket)

        # Aggregate metrics for this rule
        all_t = rule_trades[rule]
        if not all_t:
            say(f"  Rule {rule}: no trades")
            continue
        rets = [t["net_ret"] for t in all_t]
        holds = [t["candles_held"] for t in all_t]
        arr = np.array(rets)
        wins = arr > 0; losses = arr <= 0
        n = len(arr)
        pf = (arr[wins].sum() / -arr[losses].sum()) if losses.any() and arr[losses].sum() < 0 else float("inf")
        mean_r = float(arr.mean()); std_r = float(arr.std(ddof=1)) if n > 1 else 0
        avg_hold = float(np.mean(holds))
        tpy = 3125 / max(avg_hold, 1)
        sharpe = (mean_r / std_r * math.sqrt(max(tpy, 1))) if std_r > 0 else 0
        down = arr[arr < 0]
        sortino = (mean_r / down.std(ddof=1) * math.sqrt(max(tpy, 1))) if down.size > 1 else 0
        eq = np.cumsum(arr); peak = np.maximum.accumulate(eq); dd = peak - eq
        max_dd = float(dd.max()) if len(dd) else 0
        # OOS period = total - lookback = 24-6 = 18 months
        oos_years = (end - rebalance_points[0]).days / 365.25
        annual = arr.sum() / oos_years
        calmar = annual / max_dd if max_dd > 0 else float("inf")
        rule_results[rule] = {
            "trades": n,
            "trades_per_yr": round(n / oos_years, 0),
            "win_rate": round(float(wins.mean()*100), 2),
            "avg_win_pct": round(float(arr[wins].mean()), 4) if wins.any() else 0,
            "avg_loss_pct": round(float(arr[losses].mean()), 4) if losses.any() else 0,
            "profit_factor": round(pf, 3) if math.isfinite(pf) else "inf",
            "expectancy_pct": round(mean_r, 4),
            "annual_ret_pct": round(annual, 2),
            "total_ret_pct": round(float(arr.sum()), 2),
            "sharpe_ann": round(sharpe, 2),
            "sortino_ann": round(sortino, 2),
            "max_dd_pct": round(max_dd, 2),
            "calmar": round(calmar, 2) if math.isfinite(calmar) else "inf",
            "avg_hold": round(avg_hold, 1),
            "oos_years": round(oos_years, 2),
        }

    # ---- Phase 3: write outputs ----
    say("\n=== Writing outputs ===")
    # Summary CSV
    summary_df = pd.DataFrame([{"rule": r, **m} for r, m in rule_results.items()])
    summary_df = summary_df.sort_values("sharpe_ann", ascending=False)
    summary_df.to_csv(OUT_DIR / "walkforward_summary.csv", index=False)
    say(f"  walkforward_summary.csv  ({len(summary_df)} rules)")

    # Baskets CSV
    baskets_df = pd.DataFrame(rule_baskets)
    baskets_df.to_csv(OUT_DIR / "walkforward_baskets.csv", index=False)
    say(f"  walkforward_baskets.csv  ({len(baskets_df)} basket-quarters)")

    # Equity curves CSV (one row per trade, columns = rule equity)
    eq_data = {}
    for rule, all_t in rule_trades.items():
        if not all_t: continue
        sorted_t = sorted(all_t, key=lambda x: x["entry_dt"])
        eq = []
        cum = 0
        for t in sorted_t:
            cum += t["net_ret"]
            eq.append({"entry_dt": t["entry_dt"], "cum_ret_pct": cum, "rule": rule})
        eq_data[rule] = pd.DataFrame(eq)

    eq_long = pd.concat(eq_data.values(), ignore_index=True) if eq_data else pd.DataFrame()
    eq_long.to_csv(OUT_DIR / "walkforward_equity.csv", index=False)
    say(f"  walkforward_equity.csv  ({len(eq_long)} rows)")

    # ---- Phase 4: matplotlib chart ----
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                  gridspec_kw={"height_ratios": [2.5, 1]})
        ax1, ax2 = axes
        colors = ["#2E86DE", "#10AC84", "#EE5A24", "#A55EEA", "#FFC312"]

        for i, (rule, df) in enumerate(eq_data.items()):
            df = df.sort_values("entry_dt").reset_index(drop=True)
            df["peak"] = df["cum_ret_pct"].cummax()
            df["dd"] = df["peak"] - df["cum_ret_pct"]
            color = colors[i % len(colors)]
            ax1.plot(df["entry_dt"], df["cum_ret_pct"], label=f"{rule} (Sharpe {rule_results[rule]['sharpe_ann']})", color=color, linewidth=1.5)
            ax2.fill_between(df["entry_dt"], 0, -df["dd"], color=color, alpha=0.3)
            ax2.plot(df["entry_dt"], -df["dd"], color=color, linewidth=0.8, alpha=0.6)

        # Mark rebalance points
        for rp in rebalance_points:
            ax1.axvline(rp, color="grey", linestyle="--", alpha=0.3)
            ax2.axvline(rp, color="grey", linestyle="--", alpha=0.3)

        ax1.set_title(f"Walk-Forward Equity (gap=0 + ADX>=25, lookback={LOOKBACK_MONTHS}mo, trade={TRADE_MONTHS}mo, NET {COST_PCT}%)\n"
                      f"OOS period: {rebalance_points[0].date()} -> {end.date()} ({rule_results[list(rule_results.keys())[0]]['oos_years']}y)")
        ax1.set_ylabel("Cumulative net return (%, equal-risk per trade)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.axhline(0, color="black", linewidth=0.5)

        ax2.set_title("Drawdown (% from peak)")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(alpha=0.3)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        png_path = OUT_DIR / "walkforward_curves.png"
        plt.savefig(png_path, dpi=110, bbox_inches="tight")
        plt.close()
        say(f"  walkforward_curves.png  saved")
    except Exception as e:
        say(f"  Chart generation failed: {e}")

    # ---- Phase 5: human-readable summary ----
    summary_txt = OUT_DIR / "walkforward_summary.txt"
    lines = []
    lines.append("=== WALK-FORWARD SELECTION RESULTS ===")
    lines.append(f"OOS period: {rebalance_points[0].date()} -> {end.date()}")
    lines.append(f"Lookback: {LOOKBACK_MONTHS}mo   Trade: {TRADE_MONTHS}mo")
    lines.append(f"Cost: {COST_PCT}% per trade")
    lines.append("")
    lines.append("Rule ranking by Sharpe (annualized):")
    lines.append(summary_df[["rule","trades","trades_per_yr","win_rate","profit_factor",
                             "expectancy_pct","annual_ret_pct","sharpe_ann","calmar","max_dd_pct"]].to_string(index=False))
    lines.append("")
    lines.append("=== Per-quarter basket churn (top rule) ===")
    top_rule = summary_df.iloc[0]["rule"]
    rb = baskets_df[baskets_df["rule"] == top_rule].sort_values("rebalance_dt")
    lines.append(f"Top rule: {top_rule}")
    for _, row in rb.iterrows():
        lines.append(f"  {row['rebalance_dt']} -> {row['trade_window_end']}: "
                     f"{row['basket_size']} stocks, {row['quarter_trades']} trades")
        lines.append(f"    Stocks: {row['stocks']}")
    summary_txt.write_text("\n".join(lines))
    say(f"  walkforward_summary.txt")

    say("")
    say("=== FINAL RESULTS — per-rule summary ===")
    say(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
