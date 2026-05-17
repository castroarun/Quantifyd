"""Phase 17 — yearly-returns heatmap, systems vs benchmarks.

Layout per user: rows = YEARS (yearly returns on Y axis), columns in
this exact order:
  SMOOTHEST | MAX-RETURN | FORTIFIED | Nifty 50 | Nifty Midcap 150 |
  Nifty Smallcap 250
Mid/Small shown "as available" — missing years (niftyindices gaps,
deliberately NOT re-fetched, NOT fabricated) render as blank grey cells.
Diverging RdYlGn colour scale centred at 0, % per cell, colorbar.

SMOOTHEST = Phase-11 (gate+perSMA+12%trail); MAX-RETURN = Phase-10
beta-hedge hr1.0; FORTIFIED = Phase-13 (beta-hedge + perSMA + 12%trail).
Nifty 50 = NIFTYBEES (full, reliable). Mid150/Small250 = on-disk
idx_*.csv only.
"""
from __future__ import annotations
import importlib.util, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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
p13 = _mod("p13", "13_maxret_plus_stocklevel.py")
rs2 = p11.rs2

YEARS = list(range(2014, 2027))


def yr_ret_nav(nav):
    y = nav.groupby(nav.index.year).last()
    r = y.pct_change()
    r.iloc[0] = y.iloc[0] / nav.iloc[0] - 1
    return (r * 100).round(1)


def yr_ret_idx(csv):
    p = RES / csv
    if not p.exists():
        return pd.Series(dtype=float)
    s = pd.read_csv(p, parse_dates=[0], index_col=0).iloc[:, 0].sort_index()
    out = {}
    for y, g in s.groupby(s.index.year):
        if len(g) >= 2:
            out[y] = round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1)
    return pd.Series(out)


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print("SMOOTHEST ...", flush=True)
    sm = yr_ret_nav(p11.backtest(close, tv, True, True, 0.12)["nav"])
    print("MAX-RETURN ...", flush=True)
    mx = yr_ret_nav(p10.backtest(close, tv, "beta", 1.0)["nav"])
    print("FORTIFIED ...", flush=True)
    ft = p13.backtest(close, tv, True)["pyr"]            # already %/yr
    nb = pd.read_sql("SELECT date,close FROM market_data_unified WHERE "
                     "timeframe='day' AND symbol='NIFTYBEES' AND close "
                     "IS NOT NULL ORDER BY date",
                     sqlite3.connect(ROOT / "backtest_data" /
                                     "market_data.db"),
                     parse_dates=["date"]).set_index("date")["close"]
    n50 = yr_ret_nav(nb)

    cols = ["SMOOTHEST", "MAX-RETURN", "FORTIFIED", "Nifty 50"]
    M = pd.DataFrame(index=YEARS, columns=cols, dtype=float)
    M["SMOOTHEST"] = sm.reindex(YEARS)
    M["MAX-RETURN"] = mx.reindex(YEARS)
    M["FORTIFIED"] = ft.reindex(YEARS)
    M["Nifty 50"] = n50.reindex(YEARS)
    M.to_csv(RES / "yearly_matrix.csv")

    data = M.values.astype(float)
    fig, ax = plt.subplots(figsize=(9.2, 0.5 * len(YEARS) + 2))
    vmax = float(np.nanmax(np.abs(data)))
    vmax = max(20.0, np.ceil(vmax / 10) * 10)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn"); cmap.set_bad("#ededed")
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=0, ha="center", fontsize=9.5)
    ax.xaxis.set_ticks_position("top")          # headers at TOP
    ax.xaxis.set_label_position("top")
    ax.set_yticks(range(len(YEARS)))
    ax.set_yticklabels(YEARS)
    ax.set_ylabel("Year (annual return %)")
    ax.set_title("research/41 — Yearly returns: systems vs Nifty 50 "
                 "(gross). PIT mid-cap band, 2014–2026.",
                 fontsize=10, pad=34)
    ax.invert_yaxis()
    for i in range(len(YEARS)):
        for j in range(len(cols)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8,
                        color="#1a1a1a" if abs(v) < 0.6 * vmax
                        else "white")
            else:
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=7, color="#999")
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(YEARS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="minor", length=0)
    # colorbar intentionally omitted (cells are % -annotated + colour-coded)
    fig.tight_layout()
    out = RES / "yearly_matrix_heatmap.png"
    fig.savefig(out, dpi=145); plt.close(fig)
    print("\n=== YEARLY MATRIX ===\n" + M.to_string())
    print(f"\nsaved -> {out.name}")


if __name__ == "__main__":
    main()
