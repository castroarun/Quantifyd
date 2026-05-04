"""Phase B + C: deep-dive on the top cell (LONG E1 X9) and cost-adjusted re-rank.

B — for the top cell, compute and dump:
  - Per-trade log to results/top_cell_trades.csv
  - Monthly P&L to results/top_cell_monthly.csv
  - Equity curve (cumulative pct) to results/top_cell_equity.csv
  - Worst 10 / Best 10 trades printed to log
  - Longest drawdown period (start, trough, recovery)

C — re-rank ALL 60 cells with a configurable round-trip cost
  (default 0.10% = STT + brokerage + slippage estimate)
  Output: results/sweep_results_with_costs.csv (sorted by net Sharpe)
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data30 import INTRADAY_STOCKS, BT_START, BT_END, load_30min  # noqa: E402
from indicators_stoch import ema, stochastics, rsi, atr, supertrend  # noqa: E402
from backtest import (  # noqa: E402
    _prepare, generate_setups, simulate_trade, cell_metrics,
    SIDES, ENTRY_VARIANTS, EXIT_VARIANTS,
)

OUT_DIR = SCRIPT_DIR.parent / "results"
LOG_DIR = SCRIPT_DIR.parent / "logs"

# Cost model: per-trade round-trip in PERCENT of price
# - Zerodha brokerage: ₹20 per executed order, both sides → effectively
#   round-trip ~₹40 on a ~₹1L trade = ~0.04%
# - STT (delivery sell): 0.025% on sell side only → 0.025%
# - Slippage on stop fill + market exit: ~0.04-0.06% round-trip
# Conservative estimate: 0.10% per trade
DEFAULT_COST_PCT = 0.10

TOP_SIDE, TOP_EV, TOP_XV = "long", "E1", "X9"


def _trade_to_dict(symbol: str, t, df_dt: pd.Series, side: str) -> dict:
    return {
        "symbol": symbol,
        "side": side,
        "entry_dt": pd.Timestamp(df_dt.iloc[t.entry_idx]).isoformat(),
        "entry_price": t.entry_price,
        "exit_dt": pd.Timestamp(df_dt.iloc[t.exit_idx]).isoformat(),
        "exit_price": t.exit_price,
        "exit_reason": t.exit_reason,
        "candles_held": t.candles_held,
        "return_pct": round(t.return_pct, 4),
    }


def main():
    log_path = LOG_DIR / "drill_top_cell.log"
    log = log_path.open("w")
    def say(msg: str):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    say("=== Phase B + C: Top-cell deep dive + cost-adjusted re-rank ===")
    say(f"Top cell: side={TOP_SIDE} entry={TOP_EV} exit={TOP_XV}")
    say(f"Default round-trip cost: {DEFAULT_COST_PCT}%")
    say("")

    # Re-load + re-prepare
    say("Pre-computing indicators...")
    t0 = time.time()
    prepared = {}
    for sym in INTRADAY_STOCKS:
        prepared[sym] = _prepare(sym)
    say(f"  done in {time.time()-t0:.1f}s")
    say("")

    # ----------------------------------------------------------
    # B — top cell drill
    # ----------------------------------------------------------
    say(f"--- B: extracting top-cell trades ({TOP_SIDE}/{TOP_EV}/{TOP_XV}) ---")
    all_trades: list[dict] = []
    for sym in INTRADAY_STOCKS:
        df = prepared[sym]
        if df.empty: continue
        setups = generate_setups(df, TOP_SIDE, TOP_EV)
        for s in setups:
            outcome = simulate_trade(
                df, TOP_SIDE, s["fill_idx"], s["fill_price"],
                s["anchor_low"], s["anchor_high"], TOP_XV,
            )
            all_trades.append(_trade_to_dict(sym, outcome, df["dt"], TOP_SIDE))

    if not all_trades:
        say("No trades.")
        log.close()
        return

    trades_df = pd.DataFrame(all_trades)
    trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_dt"])
    trades_df["exit_dt"] = pd.to_datetime(trades_df["exit_dt"])
    trades_df = trades_df.sort_values("entry_dt").reset_index(drop=True)

    trades_csv = OUT_DIR / "top_cell_trades.csv"
    trades_df.to_csv(trades_csv, index=False)
    say(f"Wrote {trades_csv}  ({len(trades_df)} trades)")

    # Equity curve (cumulative pct)
    trades_df["cum_pct"] = trades_df["return_pct"].cumsum()
    trades_df["peak"] = trades_df["cum_pct"].cummax()
    trades_df["dd_pct"] = trades_df["peak"] - trades_df["cum_pct"]

    eq_csv = OUT_DIR / "top_cell_equity.csv"
    trades_df[["entry_dt", "symbol", "return_pct", "cum_pct", "peak", "dd_pct"]].to_csv(eq_csv, index=False)
    say(f"Wrote {eq_csv}")

    # Monthly aggregation by entry month
    trades_df["entry_month"] = trades_df["entry_dt"].dt.to_period("M")
    monthly = trades_df.groupby("entry_month").agg(
        trades=("return_pct", "size"),
        wins=("return_pct", lambda s: int((s > 0).sum())),
        sum_ret_pct=("return_pct", "sum"),
        mean_ret_pct=("return_pct", "mean"),
        max_dd_in_month=("dd_pct", "max"),
    ).round(2)
    monthly["win_rate"] = (monthly["wins"] / monthly["trades"] * 100).round(1)
    monthly_csv = OUT_DIR / "top_cell_monthly.csv"
    monthly.to_csv(monthly_csv)
    say(f"Wrote {monthly_csv}  ({len(monthly)} months)")
    say("")
    say("Monthly P&L summary:")
    say(monthly[["trades", "win_rate", "sum_ret_pct", "mean_ret_pct"]].to_string())
    say("")

    # Worst / Best 10 trades
    say("Worst 10 trades:")
    say(trades_df.nsmallest(10, "return_pct")[
        ["entry_dt", "symbol", "entry_price", "exit_dt", "exit_price", "exit_reason", "candles_held", "return_pct"]
    ].to_string(index=False))
    say("")
    say("Best 10 trades:")
    say(trades_df.nlargest(10, "return_pct")[
        ["entry_dt", "symbol", "entry_price", "exit_dt", "exit_price", "exit_reason", "candles_held", "return_pct"]
    ].to_string(index=False))
    say("")

    # Longest drawdown period
    in_dd = trades_df["dd_pct"] > 0
    dd_segments = []
    start = None
    for i, x in enumerate(in_dd):
        if x and start is None:
            start = i
        elif not x and start is not None:
            seg_slice = trades_df.iloc[start:i]
            trough_idx = seg_slice["dd_pct"].idxmax()
            dd_segments.append({
                "start_dt": trades_df.iloc[start]["entry_dt"],
                "trough_dt": trades_df.loc[trough_idx]["entry_dt"],
                "end_dt": trades_df.iloc[i]["entry_dt"],
                "trades_in_dd": i - start,
                "max_dd": trades_df.loc[trough_idx]["dd_pct"],
            })
            start = None
    if dd_segments:
        dd_df = pd.DataFrame(dd_segments).sort_values("max_dd", ascending=False)
        say("Top 5 drawdown periods (by depth):")
        say(dd_df.head(5).to_string(index=False))
        say("")
        dd_df["duration_days"] = (dd_df["end_dt"] - dd_df["start_dt"]).dt.total_seconds() / 86400
        longest = dd_df.sort_values("duration_days", ascending=False).head(5)
        say("Top 5 drawdown periods (by duration):")
        say(longest[["start_dt", "trough_dt", "end_dt", "duration_days", "trades_in_dd", "max_dd"]].to_string(index=False))
    say("")

    # Headline metrics for top cell
    rets = trades_df["return_pct"].values
    say(f"=== Top-cell summary (gross, no costs) ===")
    say(f"  Trades:        {len(rets)}")
    say(f"  Win rate:      {(rets>0).mean()*100:.2f}%")
    say(f"  Total return:  {rets.sum():.2f}%")
    say(f"  Mean per tr:   {rets.mean():.4f}%")
    say(f"  Std per tr:    {rets.std(ddof=1):.4f}%")
    say(f"  Max trade DD:  {trades_df['dd_pct'].max():.2f}%")
    say(f"  Best trade:    {rets.max():.2f}%")
    say(f"  Worst trade:   {rets.min():.2f}%")

    # ----------------------------------------------------------
    # C — cost-adjusted re-rank for ALL 60 cells
    # ----------------------------------------------------------
    say("")
    say(f"--- C: cost-adjusted re-rank ({DEFAULT_COST_PCT}% round-trip) ---")
    rows = []
    for side in SIDES:
        for ev in ENTRY_VARIANTS:
            for xv in EXIT_VARIANTS:
                trades = []
                for sym in INTRADAY_STOCKS:
                    df = prepared[sym]
                    if df.empty: continue
                    setups = generate_setups(df, side, ev)
                    for s in setups:
                        outcome = simulate_trade(
                            df, side, s["fill_idx"], s["fill_price"],
                            s["anchor_low"], s["anchor_high"], xv,
                        )
                        trades.append(outcome)
                # apply cost: subtract DEFAULT_COST_PCT from each trade return
                for t in trades:
                    t.return_pct = t.return_pct - DEFAULT_COST_PCT
                m = cell_metrics(trades)
                rows.append({"side": side, "entry_variant": ev, "exit_variant": xv,
                             "cost_pct": DEFAULT_COST_PCT, **m})
    cdf = pd.DataFrame(rows)
    cdf = cdf.sort_values("sharpe_ann", ascending=False)
    out_csv = OUT_DIR / "sweep_results_with_costs.csv"
    cdf.to_csv(out_csv, index=False)
    say(f"Wrote {out_csv}")
    say("")
    say("=== TOP 15 CELLS NET OF COSTS (sorted by Sharpe) ===")
    show_cols = ["side", "entry_variant", "exit_variant", "trades", "win_rate",
                 "profit_factor", "sharpe_ann", "max_dd_pct", "total_ret_pct", "avg_hold_candles"]
    say(cdf.head(15)[show_cols].to_string(index=False))
    say("")

    # Pass-rate after costs
    gates = (cdf["win_rate"] >= 25) & (cdf["trades"] >= 50) & (cdf["profit_factor"] >= 1.0) & (cdf["max_dd_pct"] <= 50)
    say(f"Cells passing gates after costs: {int(gates.sum())} of 60 (was 22 pre-cost)")

    log.close()


if __name__ == "__main__":
    main()
