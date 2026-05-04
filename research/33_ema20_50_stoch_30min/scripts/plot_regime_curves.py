"""Plot equity + drawdown for the top regime-gated cells.

Loads top cells from walkforward_regime_summary.csv and re-runs them
to capture per-trade equity curves, then plots.
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

from final_run import (  # noqa: E402
    list_universe, prepare_30min, gen_setups_locked,
    UNIVERSE_START, UNIVERSE_END,
    SIDE, EXIT_ID, COST_PCT,
)
from param_sweep import sim  # noqa: E402
from walkforward_regime_v2 import (  # noqa: E402
    load_nifty_with_regime, attribute_regime_vectorized,
    apply_gate, per_stock_score, select_basket,
    LOOKBACK_MONTHS, TRADE_MONTHS, MIN_TRADES_LOOKBACK,
)

OUT_DIR = SCRIPT_DIR.parent / "results"
LOG_DIR = SCRIPT_DIR.parent / "logs"

# Cells to plot (rule, gate, label)
CELLS = [
    ("R4_sh1.5", "G6_dailyvol20_above_0.8", "R4_sh1.5 + G6 (daily vol>0.8)"),
    ("R9_totret5", "G6_dailyvol20_above_0.8", "R9_totret5 + G6"),
    ("ALL", "G1_vol30m_above_0.20", "ALL + G1 (30m vol>0.20)"),
    ("ALL", "G6_dailyvol20_above_0.8", "ALL + G6"),
    ("ALL", "G0_no_gate", "ALL + no gate (baseline)"),
]


def main():
    print("Loading + generating all trades on full universe...")
    universe = list_universe()
    print(f"  Universe: {len(universe)} stocks")
    t0 = time.time()
    rows = []
    for i, sym in enumerate(universe, 1):
        df = prepare_30min(sym)
        if df.empty: continue
        ss = gen_setups_locked(df)
        for s in ss:
            ex_idx, ex_p, ex_r, held, ret = sim(
                s["df_ref"], SIDE, s["fill_idx"], s["fill_price"],
                s["anchor_low"], s["anchor_high"], EXIT_ID,
            )
            rows.append({
                "symbol": sym,
                "entry_dt": pd.Timestamp(s["df_ref"]["dt"].iloc[s["fill_idx"]]),
                "exit_dt": pd.Timestamp(s["df_ref"]["dt"].iloc[ex_idx]),
                "candles_held": held,
                "net_return_pct": ret - COST_PCT,
            })
        if i % 20 == 0: print(f"  {i}/{len(universe)}  ({time.time()-t0:.0f}s)")
    print(f"  Done: {time.time()-t0:.0f}s, {len(rows)} trades")

    trades = pd.DataFrame(rows)
    trades["entry_dt"] = pd.to_datetime(trades["entry_dt"])

    print("Loading NIFTY + attributing regime...")
    n30, nd = load_nifty_with_regime()
    trades = attribute_regime_vectorized(trades, n30, nd)
    print(f"  Done. Trades: {len(trades)}")

    # Rebalance schedule
    start = pd.Timestamp(UNIVERSE_START); end = pd.Timestamp(UNIVERSE_END)
    rebalance_points = []
    t = start + pd.DateOffset(months=LOOKBACK_MONTHS)
    while t < end:
        rebalance_points.append(t); t = t + pd.DateOffset(months=TRADE_MONTHS)
    oos_y = (end - rebalance_points[0]).days / 365.25
    print(f"OOS period: {rebalance_points[0].date()} -> {end.date()}")

    # Run each cell, collect chronological trades
    cell_equity = {}
    for rule, gate, label in CELLS:
        full_mask = apply_gate(trades, gate)
        gated = trades[full_mask].copy()
        all_t = []
        prev = set()
        for rp in rebalance_points:
            lb_start = rp - pd.DateOffset(months=LOOKBACK_MONTHS)
            te = min(rp + pd.DateOffset(months=TRADE_MONTHS), end)
            lb = gated[(gated["entry_dt"] >= lb_start) & (gated["entry_dt"] < rp)]
            score_rows = []
            for sym, g in lb.groupby("symbol"):
                s = per_stock_score(g["net_return_pct"].tolist(), g["candles_held"].tolist())
                score_rows.append({"symbol": sym, **s})
            seen = {r["symbol"] for r in score_rows}
            for sym in trades["symbol"].unique():
                if sym not in seen:
                    score_rows.append({"symbol": sym, "trades": 0, "sharpe": 0,
                                       "total_ret": 0, "expectancy": 0, "pf": 0})
            scores = pd.DataFrame(score_rows)
            basket = select_basket(rule, scores, prev)
            for sym in basket:
                st_trades = gated[(gated["symbol"] == sym)
                                   & (gated["entry_dt"] >= rp)
                                   & (gated["entry_dt"] < te)]
                for _, row in st_trades.iterrows():
                    all_t.append({"symbol": sym, **row.to_dict()})
            prev = set(basket)
        if not all_t:
            print(f"  {label}: no trades")
            continue
        df = pd.DataFrame(all_t).sort_values("entry_dt").reset_index(drop=True)
        df["cum_ret_pct"] = df["net_return_pct"].cumsum()
        df["peak"] = df["cum_ret_pct"].cummax()
        df["dd_pct"] = df["peak"] - df["cum_ret_pct"]
        cell_equity[label] = df
        print(f"  {label}: {len(df)} trades, end cum {df['cum_ret_pct'].iloc[-1]:.1f}%, max DD {df['dd_pct'].max():.1f}%")

    # Plot
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})
    ax1, ax2 = axes
    colors = ["#10AC84", "#2E86DE", "#EE5A24", "#A55EEA", "#999999"]

    for i, (label, df) in enumerate(cell_equity.items()):
        c = colors[i % len(colors)]
        ax1.plot(df["entry_dt"], df["cum_ret_pct"], label=label, color=c, linewidth=1.6)
        ax2.fill_between(df["entry_dt"], 0, -df["dd_pct"], color=c, alpha=0.25)
        ax2.plot(df["entry_dt"], -df["dd_pct"], color=c, linewidth=0.8, alpha=0.7)

    for rp in rebalance_points:
        ax1.axvline(rp, color="grey", linestyle="--", alpha=0.3, linewidth=0.6)
        ax2.axvline(rp, color="grey", linestyle="--", alpha=0.3, linewidth=0.6)

    ax1.set_title(f"Walk-forward equity — regime-gated (NET {COST_PCT}%, OOS {rebalance_points[0].date()} → {end.date()})")
    ax1.set_ylabel("Cumulative return (%, equal-risk per trade)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3); ax1.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Drawdown from peak (%)")
    ax2.set_ylabel("Drawdown (%)"); ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3); ax2.axhline(0, color="black", linewidth=0.5)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    out = OUT_DIR / "walkforward_regime_curves.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
