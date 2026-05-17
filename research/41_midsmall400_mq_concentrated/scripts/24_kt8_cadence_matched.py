"""Phase 24 — keep-top8 vs base, CADENCE-MATCHED (the honest re-test).

The Phase-23 month-end comparison was unfair: it changed two things at
once (keep-8-vs-cash AND weekly→monthly regime). The locked SMOOTHEST
gate is WEEKLY (Phase-15). This re-tests keep-top8 vs base-cash with
BOTH on the locked weekly cadence, on the SAME validated daily-marked
engine used by Phase-22 (`22_smoothest_variants.run`), on fresh data.

Outputs the decision-grade numbers (gross + post-tax@20% + strict daily
MaxDD + Sharpe + Calmar + per-year) and a clean 2-line equity + drawdown
PNG so the comparison is visible on the correct engine.
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
RES = HERE / "results"; RES.mkdir(exist_ok=True)
SC = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location(
    "p22", str(SC / "22_smoothest_variants.py"))
p22 = importlib.util.module_from_spec(_s); _s.loader.exec_module(p22)
rs2 = p22.rs2


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print("BASE SMOOTHEST (allcash, weekly) ...", flush=True)
    bg = p22.run(close, tv, "allcash", 0.0, 0, 100)
    bt = p22.run(close, tv, "allcash", 0.0, 0, 100, stcg=0.20)
    print("keep-top8 (keeptop K=8, weekly) ...", flush=True)
    kg = p22.run(close, tv, "keeptop", 0.0, 8, 100)
    kt = p22.run(close, tv, "keeptop", 0.0, 8, 100, stcg=0.20)

    summ = pd.DataFrame([
        dict(system="BASE SMOOTHEST (allcash)", cagr=round(bg["cagr"], 1),
             post_tax20=round(bt["cagr"], 1), maxdd_daily=round(bg["dd"], 1),
             sharpe=round(bg["sh"], 2), calmar=round(bg["cal"], 2)),
        dict(system="SMOOTHEST keep-top8", cagr=round(kg["cagr"], 1),
             post_tax20=round(kt["cagr"], 1), maxdd_daily=round(kg["dd"], 1),
             sharpe=round(kg["sh"], 2), calmar=round(kg["cal"], 2)),
    ])
    summ.to_csv(RES / "phase24_kt8_cadence.csv", index=False)
    py = pd.DataFrame({"BASE SMOOTHEST": bg["py"],
                       "keep-top8": kg["py"]}).reindex(range(2014, 2027))
    py.to_csv(RES / "phase24_kt8_peryear.csv")
    print("\n=== CADENCE-MATCHED (weekly regime, daily-marked) ===")
    print(summ.to_string(index=False))
    print("\n=== PER-YEAR % ===\n" + py.to_string())

    sm, kt8 = bg["nav"], kg["nav"]
    sm.to_csv(RES / "nav_weekly_smoothest_base.csv")
    kt8.to_csv(RES / "nav_weekly_smoothest_kt8.csv")
    nb = pd.read_sql("SELECT date,close FROM market_data_unified WHERE "
                     "timeframe='day' AND symbol='NIFTYBEES' AND close "
                     "IS NOT NULL ORDER BY date",
                     sqlite3.connect(ROOT / "backtest_data" /
                                     "market_data.db"),
                     parse_dates=["date"]).set_index("date")["close"]
    nb = nb.reindex(nb.index.union(sm.index)).ffill().reindex(sm.index)
    nb = nb / nb.iloc[0]

    def dd(s):
        return (s - s.cummax()) / s.cummax() * 100

    fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(sm.index, sm.values, color="#1f77b4", lw=2,
               label=f"SMOOTHEST base — risk-off→cash  "
                     f"({sm.iloc[-1]:.0f}× | {bg['cagr']:.1f}% CAGR | "
                     f"{bt['cagr']:.1f}% post-tax | {bg['dd']:.1f}% DD | "
                     f"Cal {bg['cal']:.2f})")
    ax[0].plot(kt8.index, kt8.values, color="#2ca02c", lw=2,
               label=f"SMOOTHEST keep-top8  "
                     f"({kt8.iloc[-1]:.0f}× | {kg['cagr']:.1f}% CAGR | "
                     f"{kt['cagr']:.1f}% post-tax | {kg['dd']:.1f}% DD | "
                     f"Cal {kg['cal']:.2f})")
    ax[0].plot(nb.index, nb.values, color="#888888", lw=1.3, ls="--",
               label=f"Nifty 50 ref ({nb.iloc[-1]:.1f}×)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Growth of 1 (log scale)")
    ax[0].set_title("research/41 — keep-top8 vs base, CADENCE-MATCHED "
                    "(locked weekly regime, daily-marked, PIT mid-cap, "
                    "2014–2026)")
    ax[0].legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax[0].grid(True, which="both", alpha=0.25)
    ax[1].plot(nb.index, dd(nb).values, color="#888888", lw=1.0, ls="--",
               label="Nifty 50")
    ax[1].plot(sm.index, dd(sm).values, color="#1f77b4", lw=1.2,
               label="SMOOTHEST base")
    ax[1].plot(kt8.index, dd(kt8).values, color="#2ca02c", lw=1.2,
               label="keep-top8")
    ax[1].fill_between(sm.index, dd(sm).values, 0, color="#1f77b4",
                       alpha=0.13)
    ax[1].fill_between(kt8.index, dd(kt8).values, 0, color="#2ca02c",
                       alpha=0.13)
    ax[1].legend(loc="lower left", fontsize=8, ncol=3, framealpha=0.9)
    ax[1].set_ylabel("Drawdown %"); ax[1].set_xlabel("Year")
    ax[1].grid(True, alpha=0.25)
    fig.tight_layout()
    out = RES / "smoothest_vs_kt8_weekly.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\nsaved -> {out.name}")


if __name__ == "__main__":
    main()
