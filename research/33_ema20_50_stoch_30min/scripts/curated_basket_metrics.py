"""Compute proper portfolio-level metrics for the F2-curated 17-stock basket.

The per-symbol metrics in regime_filter_per_symbol_F2.csv are AVERAGES of
per-symbol Sharpes — NOT a portfolio-equivalent figure. To get the right
deployment-relevant numbers, we re-simulate just the 17 winners as a
single equity curve (equal risk per trade across all symbols).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from regime_filter_test import (  # noqa: E402
    prepare_30min, gen_setups, sim, COST_PCT, SIDE, EXIT_ID,
)

OUT_DIR = SCRIPT_DIR.parent / "results"
LOG_DIR = SCRIPT_DIR.parent / "logs"

CURATED_17 = [
    "ASIANPAINT", "ICICIBANK", "HDFCLIFE", "RELIANCE", "COLPAL",
    "BRITANNIA", "SBIN", "INFY", "MCX", "ITC", "HINDUNILVR", "M&M",
    "BEL", "BHARTIARTL", "FEDERALBNK",
    # Plus the 2 unnamed >2.0 stocks — derived dynamically from the F2 csv
]


def main():
    log_path = LOG_DIR / "curated_basket.log"
    log = log_path.open("w")
    def say(msg: str):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    # Pull the actual list dynamically — anyone with Sharpe > 2.0 under F2
    f2_csv = OUT_DIR / "regime_filter_per_symbol_F2.csv"
    f2 = pd.read_csv(f2_csv).sort_values("sharpe_ann", ascending=False)
    curated = f2[f2["sharpe_ann"] > 2.0]["symbol"].tolist()
    say(f"=== Curated basket portfolio metrics ===")
    say(f"Filter: F2 (ADX(14)>=25 at trigger) on the winning cell")
    say(f"Stocks (Sharpe>2.0 under F2): {len(curated)}")
    say(f"  {curated}")
    say("")

    say("Preparing 17 symbols (~2 min)...")
    t0 = time.time()
    prepped = {}
    for i, sym in enumerate(curated, 1):
        df30, df_daily = prepare_30min(sym)
        if not df30.empty:
            prepped[sym] = (df30, df_daily)
        say(f"  [{i}/{len(curated)}] {sym}  ({time.time()-t0:.0f}s)")
    say(f"Prepared in {time.time()-t0:.0f}s")
    say("")

    # Run the F2-filtered, top-cell strategy on each, collect per-trade data
    say("Simulating F2-filtered top cell on curated basket...")
    all_trades: list[dict] = []
    for sym, (df30, df_daily) in prepped.items():
        ss = gen_setups(df30, df_daily, SIDE, "F2")
        for s in ss:
            ex_idx, ex_p, ex_r, held, ret = sim(
                s["df_ref"], SIDE, s["fill_idx"], s["fill_price"],
                s["anchor_low"], s["anchor_high"], EXIT_ID,
            )
            df = s["df_ref"]
            all_trades.append({
                "symbol": sym,
                "entry_dt": pd.Timestamp(df["dt"].iloc[s["fill_idx"]]),
                "exit_dt": pd.Timestamp(df["dt"].iloc[ex_idx]),
                "entry_price": s["fill_price"],
                "exit_price": ex_p,
                "exit_reason": ex_r,
                "candles_held": held,
                "gross_return_pct": ret,
                "net_return_pct": ret - COST_PCT,
            })

    if not all_trades:
        say("No trades — bug?")
        log.close()
        return

    tdf = pd.DataFrame(all_trades).sort_values("entry_dt").reset_index(drop=True)
    tdf.to_csv(OUT_DIR / "curated_basket_trades.csv", index=False)
    say(f"Total trades across {len(prepped)} symbols: {len(tdf)}")
    say(f"Wrote {OUT_DIR / 'curated_basket_trades.csv'}")
    say("")

    # ---- Aggregate metrics (treating each trade as 1 unit of equal risk) ----
    rets = tdf["net_return_pct"].values
    holds = tdf["candles_held"].values

    n = len(rets)
    win_mask = rets > 0
    wins = rets[win_mask]
    losses = rets[~win_mask]

    total = float(rets.sum())
    mean_per_trade = float(rets.mean())
    std_per_trade = float(rets.std(ddof=1)) if n > 1 else 0
    avg_win = float(wins.mean()) if wins.size else 0
    avg_loss = float(losses.mean()) if losses.size else 0
    win_rate = float(win_mask.mean() * 100)
    profit_factor = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    expectancy = mean_per_trade  # already accounts for WR + sizes

    avg_hold = float(np.mean(holds)) if len(holds) else 0
    # 30-min candles per year: ~250 trading days × 12.5 candles/day = 3125
    candles_per_year = 3125
    trades_per_year = candles_per_year / max(avg_hold, 1)
    sharpe_per_trade = (mean_per_trade / std_per_trade) if std_per_trade > 0 else 0
    sharpe_ann = sharpe_per_trade * math.sqrt(max(trades_per_year, 1))

    # Sortino (downside-only std)
    down = rets[rets < 0]
    sortino_per_trade = (mean_per_trade / down.std(ddof=1)) if down.size > 1 else 0
    sortino_ann = sortino_per_trade * math.sqrt(max(trades_per_year, 1))

    # MaxDD on cumulative-pct equity
    eq = np.cumsum(rets); peak = np.maximum.accumulate(eq); dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0

    # Annualization: period = 24 months = 2 years
    period_years = 2.0
    actual_trades_per_year = n / period_years
    annual_ret = total / period_years  # linear annualization (no compounding)
    calmar = annual_ret / max_dd if max_dd > 0 else float("inf")

    say("=== PORTFOLIO METRICS — Curated basket (equal-risk per trade) ===")
    say(f"  Stocks            : {len(prepped)}")
    say(f"  Period            : 2024-03-18 -> 2026-03-12 (24 months)")
    say(f"  Trades            : {n}")
    say(f"  Trades / year     : {actual_trades_per_year:.0f}")
    say(f"  Win rate          : {win_rate:.2f}%")
    say(f"  Avg win / trade   : +{avg_win:.3f}%")
    say(f"  Avg loss / trade  : {avg_loss:.3f}%")
    say(f"  Profit factor     : {profit_factor:.3f}")
    say(f"  Expectancy / trade: {expectancy:+.4f}%  (== mean per trade, net of {COST_PCT}% cost)")
    say(f"  Total return      : {total:+.2f}%   (sum of per-trade returns, equal-risk)")
    say(f"  Annual return     : {annual_ret:+.2f}%   (linear annualized)")
    say(f"  Std per trade     : {std_per_trade:.3f}%")
    say(f"  Sharpe (ann)      : {sharpe_ann:.2f}")
    say(f"  Sortino (ann)     : {sortino_ann:.2f}")
    say(f"  Max drawdown      : {max_dd:.2f}%   (cumulative, equal-risk equity)")
    say(f"  Calmar (Ann/MDD)  : {calmar:.2f}")
    say(f"  Avg hold (candles): {avg_hold:.1f}  (~{avg_hold*30/60:.1f} hours session-time)")

    # Save summary to its own CSV for the comparison table
    pd.DataFrame([{
        "phase": "Curated 17 + F2",
        "universe": len(prepped),
        "trades": n,
        "trades_per_yr": round(actual_trades_per_year, 0),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "expectancy_pct": round(expectancy, 4),
        "annual_ret_pct": round(annual_ret, 2),
        "sharpe_ann": round(sharpe_ann, 2),
        "calmar": round(calmar, 2) if math.isfinite(calmar) else "inf",
        "max_dd_pct": round(max_dd, 2),
        "avg_hold_candles": round(avg_hold, 1),
    }]).to_csv(OUT_DIR / "curated_basket_summary.csv", index=False)

    log.close()


if __name__ == "__main__":
    main()
