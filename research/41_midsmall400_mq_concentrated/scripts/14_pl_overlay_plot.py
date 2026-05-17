"""Phase 14 — overlay the equity (P/L) curves of the two final systems
(+ Nifty 50 reference) on one chart. Rebuilds NAV from the vetted
Phase-11 (SMOOTHEST) and Phase-10 (MAX-RETURN) backtests; no new data.
"""
from __future__ import annotations
import importlib.util, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
RES = HERE / "results"
SC = Path(__file__).resolve().parent


def _mod(n, f):
    s = importlib.util.spec_from_file_location(n, str(SC / f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


p11 = _mod("p11", "11_stocklevel_risk.py")
p10 = _mod("p10", "10_hedge_overlay.py")
rs2 = p11.rs2


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print("Rebuilding SMOOTHEST ...", flush=True)
    sm = p11.backtest(close, tv, True, True, 0.12)["nav"]
    print("Rebuilding SMOOTHEST-KT8 ...", flush=True)
    kr = p11.backtest(close, tv, True, True, 0.12, keep_top=8)
    kt8 = kr["nav"]
    print("Rebuilding MAX-RETURN ...", flush=True)
    mx = p10.backtest(close, tv, "beta", 1.0)["nav"]

    nb = pd.read_sql("SELECT date,close FROM market_data_unified WHERE "
                     "timeframe='day' AND symbol='NIFTYBEES' AND close "
                     "IS NOT NULL ORDER BY date",
                     sqlite3.connect(ROOT / "backtest_data" /
                                     "market_data.db"),
                     parse_dates=["date"]).set_index("date")["close"]
    nb = nb.reindex(nb.index.union(sm.index)).ffill().reindex(sm.index)
    nb = nb / nb.iloc[0]                       # normalise to 1.0 at start

    def dd(s):
        return (s - s.cummax()) / s.cummax() * 100

    fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(sm.index, sm.values, color="#1f77b4", lw=2,
               label=f"SMOOTHEST  ({sm.iloc[-1]:.0f}x | 35.6% CAGR | "
                     f"-15.1% maxDD)")
    ax[0].plot(kt8.index, kt8.values, color="#2ca02c", lw=1.7,
               ls=(0, (5, 1.5)),
               label=f"SMOOTHEST-KT8 ({kt8.iloc[-1]:.0f}x | "
                     f"{kr['cagr']:.1f}% CAGR | {kr['dd']:.1f}% maxDD)")
    ax[0].plot(mx.index, mx.values, color="#d62728", lw=2,
               label=f"MAX-RETURN ({mx.iloc[-1]:.0f}x | 42.8% CAGR | "
                     f"-22.7% maxDD)")
    ax[0].plot(nb.index, nb.values, color="#888888", lw=1.3, ls="--",
               label=f"Nifty 50 ref ({nb.iloc[-1]:.1f}x | 13.1% CAGR)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Growth of 1 (log scale)")
    ax[0].set_title("research/41 — Final Systems Equity Overlay  "
                    "(PIT mid-cap band, 2014–2026, gross)")
    ax[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax[0].grid(True, which="both", alpha=0.25)

    ax[1].plot(nb.index, dd(nb).values, color="#888888", lw=1.1,
               ls="--", label="Nifty 50")
    ax[1].plot(sm.index, dd(sm).values, color="#1f77b4", lw=1.2,
               label="SMOOTHEST")
    ax[1].plot(kt8.index, dd(kt8).values, color="#2ca02c", lw=1.1,
               ls=(0, (5, 1.5)), label="SMOOTHEST-KT8")
    ax[1].plot(mx.index, dd(mx).values, color="#d62728", lw=1.2,
               label="MAX-RETURN")
    ax[1].fill_between(nb.index, dd(nb).values, 0, color="#888888",
                       alpha=0.10)
    ax[1].fill_between(sm.index, dd(sm).values, 0, color="#1f77b4",
                       alpha=0.15)
    ax[1].fill_between(mx.index, dd(mx).values, 0, color="#d62728",
                       alpha=0.12)
    ax[1].legend(loc="lower left", fontsize=8, ncol=3, framealpha=0.9)
    ax[1].set_ylabel("Drawdown %")
    ax[1].set_xlabel("Year")
    ax[1].grid(True, alpha=0.25)
    fig.tight_layout()
    out = RES / "final_systems_pl_overlay.png"
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")
    print(f"SMOOTHEST final {sm.iloc[-1]:.1f}x ; MAX-RETURN "
          f"{mx.iloc[-1]:.1f}x ; Nifty50 {nb.iloc[-1]:.2f}x")


if __name__ == "__main__":
    main()
