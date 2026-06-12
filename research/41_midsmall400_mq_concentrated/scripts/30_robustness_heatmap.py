"""Phase 30 — Per-year + sub-period robustness + yearly-vs-Nifty50 heatmap for
the two live finalists (keep-top8, all-cash+weekly) on the daily-marked engine.

Outputs:
  - results/phase30_peryear.csv         per-year % for each variant + Nifty 50
  - results/phase30_subperiod.csv       full / H1 / H2 / thirds: CAGR, MaxDD, Sharpe
  - results/phase30_yearly_heatmap.png  heatmap (variants + Nifty 50) x years
All gross (matches the existing app heatmap); post-tax CAGR shown in sub-period table.
"""
from __future__ import annotations
import importlib.util, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
SCR = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("rs2", str(SCR / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)
_p = importlib.util.spec_from_file_location(
    "p22", str(SCR / "22_smoothest_variants.py"))
p22 = importlib.util.module_from_spec(_p); _p.loader.exec_module(p22)
_w = importlib.util.spec_from_file_location(
    "p27", str(SCR / "27_weekly_reentry.py"))
p27 = importlib.util.module_from_spec(_w); _w.loader.exec_module(p27)

BENCH = rs2.NIFTY_SYM
START = pd.Timestamp("2014-01-01")


def metrics(n, lo=None, hi=None):
    if lo is not None:
        n = n[(n.index >= lo) & (n.index <= hi)]
    n = n.dropna()
    if len(n) < 2:
        return dict(cagr=np.nan, dd=np.nan, sh=np.nan)
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh)


def peryear(n):
    yl = n.groupby(n.index.year).last()
    py = yl.pct_change()
    py.iloc[0] = yl.iloc[0] / n.iloc[0] - 1
    return (py * 100).round(1)


def main():
    t0 = time.time()
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print(f"  loaded {time.time()-t0:.0f}s", flush=True)
    idx = close.index[close.index >= START]

    navs, post = {}, {}
    print("  running keep-top8 ...", flush=True)
    navs["keep-top8"] = p22.run(close, tv, "keeptop", 0.0, 8, 100)["nav"]
    post["keep-top8"] = p22.run(close, tv, "keeptop", 0.0, 8, 100, stcg=0.20)["cagr"]
    print("  running all-cash+weekly ...", flush=True)
    navs["all-cash+weekly"] = p27.run(close, tv, "allcash", 0, True)["nav"]
    post["all-cash+weekly"] = p27.run(close, tv, "allcash", 0, True, stcg=0.20)["cagr"]
    print("  running all-cash base ...", flush=True)
    navs["all-cash base"] = p22.run(close, tv, "allcash", 0.0, 0, 100)["nav"]
    post["all-cash base"] = p22.run(close, tv, "allcash", 0.0, 0, 100, stcg=0.20)["cagr"]
    # Nifty 50 benchmark (NIFTYBEES buy & hold)
    bser = close[BENCH].reindex(idx).ffill().dropna()
    navs["Nifty 50"] = bser / bser.iloc[0]
    post["Nifty 50"] = np.nan

    order = ["all-cash+weekly", "keep-top8", "all-cash base", "Nifty 50"]

    # ---- per-year ----
    py = pd.DataFrame({k: peryear(navs[k]) for k in order}).reindex(
        range(2014, 2027))
    py.to_csv(RES / "phase30_peryear.csv")
    print("\n=== PER-YEAR RETURNS (%) ===\n" + py.to_string())

    # beats-index count
    for k in order[:-1]:
        wins = int((py[k] > py["Nifty 50"]).sum())
        tot = int(py[k].notna().sum())
        print(f"  {k:18} beats Nifty50 in {wins}/{tot} years", flush=True)

    # ---- sub-period robustness ----
    wins = [("Full 2014-2026", START, idx[-1]),
            ("H1 2014-2019", pd.Timestamp("2014-01-01"), pd.Timestamp("2019-12-31")),
            ("H2 2020-2026", pd.Timestamp("2020-01-01"), idx[-1]),
            ("T1 2014-2017", pd.Timestamp("2014-01-01"), pd.Timestamp("2017-12-31")),
            ("T2 2018-2021", pd.Timestamp("2018-01-01"), pd.Timestamp("2021-12-31")),
            ("T3 2022-2026", pd.Timestamp("2022-01-01"), idx[-1])]
    rows = []
    for k in order:
        for wlbl, lo, hi in wins:
            m = metrics(navs[k], lo, hi)
            rows.append(dict(variant=k, window=wlbl,
                             cagr=round(m["cagr"], 1), maxdd=round(m["dd"], 1),
                             sharpe=round(m["sh"], 2)))
    sp = pd.DataFrame(rows)
    sp.to_csv(RES / "phase30_subperiod.csv", index=False)
    print("\n=== SUB-PERIOD ROBUSTNESS (gross) ===\n" + sp.to_string(index=False))
    print("\nPost-tax@20% full-period CAGR: " +
          ", ".join(f"{k}={post[k]:.1f}%" for k in order if not np.isnan(post[k])))

    # ---- heatmap ----
    yrs = list(range(2014, 2027))
    M = py.T.reindex(order)[yrs].values.astype(float)
    fig, ax = plt.subplots(figsize=(14, 4.2))
    norm = TwoSlopeNorm(vmin=np.nanmin(M), vcenter=0, vmax=np.nanmax(M))
    im = ax.imshow(M, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(yrs))); ax.set_xticklabels(yrs)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=11)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=9, color="black")
    full = {k: metrics(navs[k])["cagr"] for k in order}
    ylabels = [f"{k}\n(CAGR {full[k]:.1f}%)" for k in order]
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_title("Midcap RS-120 — yearly returns vs Nifty 50 (gross, daily-marked, "
                 "2014-2026)\nall-cash+weekly & keep-top8 finalists", fontsize=12)
    fig.colorbar(im, ax=ax, label="annual return %", shrink=0.8)
    plt.tight_layout()
    fig.savefig(RES / "phase30_yearly_heatmap.png", dpi=130)
    print("\nSaved phase30_yearly_heatmap.png")
    print(f"Total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
