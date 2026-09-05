"""research/155 — factsheet. Four panels: growth of Rs 100 (log), drawdown, how much of the
IPO sleeve each mechanism can actually redeploy per year, and the paired verdict against the
pre-registered adoption bar."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
P = pd.read_csv(RES / "paths.csv").drop_duplicates(["cell", "path"], keep="last")
PY = pd.read_csv(RES / "peryear.csv", header=[0, 1], index_col=0)

NAVS = {
    "IPO idle -> CASH (incumbent)": ("nav_A_incumbent.csv", "#1f77b4", "-"),
    "GATED: park only when no candidate can exist (Arun's proposal)":
        ("nav_P3b_E25_OA_r0_monthly.csv", "#d62728", "-"),
    "CONTINUOUS: park all idle cash in OA, daily (naive)":
        ("nav_P1_B_OA.csv", "#2ca02c", "-"),
}
oa = pd.read_csv("/home/arun/quantifyd/research/154_multi_system_blends/results/oa_navs30.csv",
                 index_col=0, parse_dates=True)
tn = pd.read_csv("/home/arun/quantifyd/research/154_multi_system_blends/results/tn_navs12.csv",
                 index_col=0, parse_dates=True)


def blend_med(navfile):
    nav = pd.read_csv(RES / navfile, index_col=0, parse_dates=True)
    curves = []
    for p, c in enumerate(nav.columns):
        idx = nav.index.intersection(oa.index).intersection(tn.index)
        n_ = nav[c].loc[idx]; n_ = n_ / n_.iloc[0]
        o_ = oa[f"s{p+1}"].loc[idx]; o_ = o_ / o_.iloc[0]
        t_ = tn[f"off{p%12}"].loc[idx]; t_ = t_ / t_.iloc[0]
        m = lambda x: x.resample("ME").last().pct_change().fillna(0.0)  # noqa: E731
        curves.append((1 + 0.4 * m(o_) + 0.4 * m(t_) + 0.2 * m(n_)).cumprod())
    C = pd.concat(curves, axis=1)
    return C.median(axis=1)


fig = plt.figure(figsize=(14.5, 11))
gs = GridSpec(3, 2, height_ratios=[2.5, 1.0, 1.25], hspace=0.30, wspace=0.22)
ax0, ax1 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
ax2, ax3 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

for lab, (f, col, ls) in NAVS.items():
    med = blend_med(f)
    med = 100 * med / med.iloc[0]
    ax0.plot(med.index, med, color=col, ls=ls, lw=1.7, label=lab)
    ax1.plot(med.index, 100 * (med / med.cummax() - 1), color=col, ls=ls, lw=1.1)
ax0.set_yscale("log")
ax0.set_ylabel("growth of Rs 100 (log)")
ax0.grid(alpha=.25)
ax0.legend(fontsize=9, loc="upper left", framealpha=.92)
ax0.set_title("research/155 — redeploying the IPO sleeve's idle cash into Open Alpha\n"
              "40/40/20 TN+OA+IPO blend · median of 30 paired paths · after tax · "
              "25 bps per side · monthly rebalanced · 2006-2026",
              fontsize=12, pad=10)
ax1.set_ylabel("drawdown %"); ax1.grid(alpha=.25)
ax1.axhline(0, color="#888", lw=.6)

# ── panel 3: how much of the sleeve each mechanism can actually redeploy ──
yrs = PY.index.values
w = 0.38
ax2.bar(yrs - w / 2, PY[("E_gate_OA", "parked_pct")], w, color="#d62728",
        label="GATED (Arun's proposal)")
ax2.bar(yrs + w / 2, PY[("CONT_OA_best_T1", "parked_pct")], w, color="#2ca02c",
        label="CONTINUOUS (best mechanic: 2-slot reserve, monthly)")
ax2.set_ylabel("% of IPO sleeve NAV parked")
ax2.set_title("How much of the sleeve the mechanism can actually move", fontsize=10)
ax2.legend(fontsize=8); ax2.grid(alpha=.2, axis="y")
ax2.set_xticks(yrs[::3])
ax2.tick_params(labelsize=8)

# ── panel 4: the paired verdict ──
base = P[P.cell == "A_incumbent"].set_index("path")
arms = [("GATED, 25 bps", "P5_E25_OA_c25_full"),
        ("GATED, 40 bps", "P5_E25_OA_c40_full"),
        ("GATED, 60 bps", "P5_E25_OA_c60_full"),
        ("GATED, frictionless", "P5_E25_OA_frictionless"),
        ("CONTINUOUS, best T+1", "P4_OA_c25_full"),
        ("CONTINUOUS, T+0", "P2_OA_s0_r1_weekly_prorata"),
        ("CONTINUOUS, frictionless", "P1_B_OA_frictionless")]
basecost = {"P5_E25_OA_c40_full": "P4_A_c40", "P5_E25_OA_c60_full": "P4_A_c60"}
labs, vals, wins = [], [], []
for lab, c in arms:
    a = P[P.cell == c].set_index("path")
    b = P[P.cell == basecost.get(c, "A_incumbent")].set_index("path")
    ix = a.index.intersection(b.index)
    d = (a.loc[ix, "b_calmar"] - b.loc[ix, "b_calmar"])
    labs.append(lab); vals.append(float(d.median())); wins.append(int((d > 0).sum()))
y = np.arange(len(labs))
ax3.barh(y, vals, color=["#d62728"] * 4 + ["#2ca02c"] * 3, alpha=.85)
ax3.axvline(0.10, color="#111", ls="--", lw=1.3)
ax3.text(0.105, 0.1, "pre-registered\nbar: +0.10", fontsize=8, va="top")
ax3.axvline(0, color="#666", lw=.8)
ax3.set_yticks(y); ax3.set_yticklabels(labs, fontsize=8)
ax3.invert_yaxis()
ax3.set_xlabel("median paired change in blend Calmar vs the incumbent")
ax3.set_title("Every arm falls short of the bar — most fall below zero", fontsize=10)
ax3.grid(alpha=.2, axis="x")
for i, (v, wn) in enumerate(zip(vals, wins)):
    ax3.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.3f}  ({wn}/30)",
             va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
ax3.set_xlim(min(vals) * 1.5 - 0.05, 0.20)

fig.savefig(RES / "ipo_cash_redeployment_research155.png", dpi=125,
            bbox_inches="tight", facecolor="white")
print("wrote", RES / "ipo_cash_redeployment_research155.png")
