"""Phase 16 — monthly-returns heatmap (year x month) for the headline
systems, styled like the user's sample (diverging red->white->green,
colorbar, % cell labels, years ascending, months 1-12).

Monthly return = month-over-month change of the system's month-end NAV.
SMOOTHEST = Phase-11 (gate+perSMA+12%trail); MAX-RETURN = Phase-10
beta-hedge hr1.0. Window 2014-01..2026-03 (2026 partial -> blank cells).
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
SC = Path(__file__).resolve().parent


def _mod(n, f):
    s = importlib.util.spec_from_file_location(n, str(SC / f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


p11 = _mod("p11", "11_stocklevel_risk.py")
p10 = _mod("p10", "10_hedge_overlay.py")
rs2 = p11.rs2


def monthly_matrix(nav):
    r = nav.pct_change() * 100
    r.iloc[0] = (nav.iloc[0] / 1.0 - 1) * 100
    df = pd.DataFrame({"y": r.index.year, "m": r.index.month,
                       "ret": r.values})
    piv = df.pivot_table(index="y", columns="m", values="ret",
                         aggfunc="last")
    return piv.reindex(columns=range(1, 13))


def draw(piv, title, out):
    yrs = list(piv.index)
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(yrs) + 2.4))
    data = piv.values.astype(float)
    vmax = float(np.nanmax(np.abs(data)))
    vmax = max(8.0, np.ceil(vmax / 5) * 5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad("#f5f5f5")
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(range(1, 13))
    ax.set_yticks(range(len(yrs)))
    ax.set_yticklabels(yrs)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title(title, fontsize=12, pad=10)
    ax.invert_yaxis()                       # earliest year at bottom
    for i in range(len(yrs)):
        for j in range(12):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=7.2,
                        color="#222" if abs(v) < 0.62 * vmax else "white")
    ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(yrs), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.3)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Monthly return %")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved -> {out.name}  ({piv.index.min()}..{piv.index.max()})")


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print("SMOOTHEST nav ...", flush=True)
    sm = p11.backtest(close, tv, True, True, 0.12)["nav"]
    print("MAX-RETURN nav ...", flush=True)
    mx = p10.backtest(close, tv, "beta", 1.0)["nav"]
    draw(monthly_matrix(sm),
         "SMOOTHEST — monthly returns (mid-cap RS-120 + SMA100 + ATH"
         " + per-stock-SMA + 12% trail)",
         RES / "monthly_heatmap_smoothest.png")
    draw(monthly_matrix(mx),
         "MAX-RETURN — monthly returns (mid-cap RS-120 + SMA100"
         " regime-hedge, short 1x Nifty in risk-off)",
         RES / "monthly_heatmap_maxreturn.png")
    monthly_matrix(sm).round(2).to_csv(RES / "monthly_returns_smoothest.csv")
    monthly_matrix(mx).round(2).to_csv(RES / "monthly_returns_maxreturn.csv")
    print("done.")


if __name__ == "__main__":
    main()
