"""Phase A baseline: replay the 6 live ATM NAS systems as-configured over all recorded
days, emit P&L-curve CSVs + a factsheet PNG + per-system/combined stats.

Baseline params: premium 1.3x per-leg stop, ATM strikes, 15:15 squareoff, lots=2 (norm).
Also reports a LIVE-LOTS combined book (916 family halved to 1 lot) = real-money expectation.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import sim_day, SYSTEMS_6, days, oc  # noqa

OUT = Path(__file__).resolve().parents[1] / "results"; OUT.mkdir(parents=True, exist_ok=True)

# ---------- run ----------
rows = []
for name, em, mg, live_lots in SYSTEMS_6:
    n = 0
    for day in days:
        try:
            for r in sim_day(day, em, mg, stop_mode="premium", strike_offset=0):
                r["system"] = name; r["live_lots"] = live_lots; rows.append(r); n += 1
        except Exception as e:
            print(f"ERR {name} {day}: {e}")
    print(f"  {name}: {n} legs", flush=True)
legs = pd.DataFrame(rows)
legs.to_csv(OUT / "baseline_legs.csv", index=False)
legs["day"] = pd.to_datetime(legs["day"])
order = [s[0] for s in SYSTEMS_6]
present = [s for s in order if s in set(legs["system"])]
dates = sorted(legs["day"].unique())

# per-day per-system (lots=2 normalized)
daily = legs.groupby(["system", "day"])["pnl"].sum().reset_index()
pv = daily.pivot(index="system", columns="day", values="pnl").reindex(present).reindex(columns=dates).fillna(0.0)
cum = pv.cumsum(axis=1)
# live-lots scaling: 916 family -> x0.5 (1 lot vs normalized 2)
scale = {s[0]: (s[3] / 2.0) for s in SYSTEMS_6}
pv_live = pv.mul([scale[s] for s in present], axis=0)
comb = pv.sum(axis=0).cumsum()               # combined, all lots=2
comb_live = pv_live.sum(axis=0).cumsum()      # combined, live lots
pv.T.to_csv(OUT / "baseline_per_day.csv")
pd.DataFrame({"combined_lots2": comb, "combined_livelots": comb_live}).to_csv(OUT / "baseline_equity.csv")


def mdd(s):
    return float((s - s.cummax()).min())


def sharpe(daily_pnl):
    d = daily_pnl.values.astype(float)
    return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0


def pf(daily_pnl):
    g = daily_pnl[daily_pnl > 0].sum(); l = -daily_pnl[daily_pnl < 0].sum()
    return float(g / l) if l > 0 else float("inf")


summ = []
for s in present:
    g = legs[legs["system"] == s]
    bd = g.groupby("day")["pnl"].sum()           # day pnl (lots=2)
    nd = g["day"].nunique()
    legwin = float((g["pnl"] > 0).mean() * 100)
    summ.append(dict(System=s, Days=nd, Legs=len(g), Net=round(g["pnl"].sum()),
                     PerDay=round(g["pnl"].sum() / max(nd, 1)),
                     DayWin=round(100 * (bd > 0).mean()), LegWin=round(legwin),
                     Sharpe=round(sharpe(bd), 2), PF=round(pf(bd), 2),
                     MaxDD=round(mdd(cum.loc[s])), Best=round(bd.max()), Worst=round(bd.min())))
summ = pd.DataFrame(summ)
summ.to_csv(OUT / "baseline_summary.csv", index=False)

# combined stats
comb_day = pv.sum(axis=0); combL_day = pv_live.sum(axis=0)
comb_stats = dict(
    net2=round(comb.iloc[-1]), perday2=round(comb.iloc[-1] / len(dates)),
    dd2=round(mdd(comb)), sharpe2=round(sharpe(comb_day), 2), pf2=round(pf(comb_day), 2),
    daywin2=round(100 * (comb_day > 0).mean()),
    netL=round(comb_live.iloc[-1]), perdayL=round(comb_live.iloc[-1] / len(dates)),
    ddL=round(mdd(comb_live)), sharpeL=round(sharpe(combL_day), 2))

# DTE aggregates
dte_comb = legs.groupby("dte")["pnl"].sum().sort_index()
dte_sys = legs.pivot_table(index="system", columns="dte", values="pnl", aggfunc="sum").reindex(present).fillna(0.0)
dte_sys = dte_sys.reindex(columns=sorted(dte_sys.columns))
tot = legs["pnl"].sum()

# ---------- factsheet ----------
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.facecolor": "white"})
fig = plt.figure(figsize=(15, 19)); gs = GridSpec(5, 2, height_ratios=[.4, 1.1, 1.1, 1.0, 1.0], hspace=.55, wspace=.18)
ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
ax0.text(0, .78, "NAS 6 ATM Systems — REPLAY on recorded NIFTY chain", fontsize=15, fontweight="bold")
ax0.text(0, .42, f"{pd.to_datetime(dates[0]).date()} → {pd.to_datetime(dates[-1]).date()} · {len(dates)} days · "
         f"lots=2 norm · combined net ₹{comb_stats['net2']:,.0f} (₹{comb_stats['perday2']:,.0f}/day, "
         f"Sharpe {comb_stats['sharpe2']}, MaxDD ₹{comb_stats['dd2']:,.0f})", fontsize=10)
ax0.text(0, .12, f"LIVE-LOTS book (916=1 lot): net ₹{comb_stats['netL']:,.0f}  (₹{comb_stats['perdayL']:,.0f}/day, "
         f"MaxDD ₹{comb_stats['ddL']:,.0f}, Sharpe {comb_stats['sharpeL']})", fontsize=10, color="#06c")
ax0.text(0, -.18, "True replay on real premiums. 41-day single regime → SIGNAL/AUDIT, not validation. "
         "9:16 exact; squeeze approx; 1-min SL/ST; LTP, no slippage (optimistic tail).", fontsize=8.5, color="#a00", style="italic")
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(comb.index, comb.values, color="#06c", lw=2, label="lots=2")
ax1.plot(comb_live.index, comb_live.values, color="#0a6", lw=1.6, ls="--", label="live lots")
ax1.axhline(0, color="#999", lw=.8); ax1.legend(fontsize=8)
ax1.set_title("Combined book — cumulative net ₹ (P&L curve)"); ax1.tick_params(axis="x", rotation=45)
ax2 = fig.add_subplot(gs[1, 1])
for s in present:
    ax2.plot(cum.columns, cum.loc[s].values, lw=1.3, label=s)
ax2.axhline(0, color="#999", lw=.8); ax2.legend(fontsize=7, ncol=2)
ax2.set_title("Per-system cumulative net ₹"); ax2.tick_params(axis="x", rotation=45)
ax3 = fig.add_subplot(gs[2, :])
M = pv.values.astype(float); vmax = np.nanpercentile(np.abs(M), 95) or 1
im = ax3.imshow(M, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
ax3.set_yticks(range(len(present))); ax3.set_yticklabels(present, fontsize=8)
ax3.set_xticks(range(len(dates))); ax3.set_xticklabels([pd.to_datetime(d).strftime("%m-%d") for d in dates], rotation=90, fontsize=6.5)
ax3.set_title("Per-day × per-system net ₹ (lots=2)")
for i in range(len(present)):
    for j in range(len(dates)):
        if abs(M[i, j]) > 1:
            ax3.text(j, i, f"{M[i, j] / 1000:.1f}", ha="center", va="center", fontsize=5, color="black")
fig.colorbar(im, ax=ax3, fraction=.025, pad=.01)
axd = fig.add_subplot(gs[3, 0])
axd.bar([str(int(d)) for d in dte_comb.index], dte_comb.values,
        color=["#0a6" if v >= 0 else "#d33" for v in dte_comb.values])
axd.axhline(0, color="#333", lw=.8); axd.set_xlabel("DTE at entry"); axd.set_title("Combined net ₹ by DTE")
axe = fig.add_subplot(gs[3, 1])
Md = dte_sys.values.astype(float); vd = np.nanpercentile(np.abs(Md), 95) or 1
imd = axe.imshow(Md, aspect="auto", cmap="RdYlGn", vmin=-vd, vmax=vd)
axe.set_yticks(range(len(present))); axe.set_yticklabels(present, fontsize=7)
axe.set_xticks(range(len(dte_sys.columns))); axe.set_xticklabels([f"{int(d)}DTE" for d in dte_sys.columns], fontsize=8)
axe.set_title("Net ₹ by system × DTE")
for i in range(len(present)):
    for j in range(len(dte_sys.columns)):
        if abs(Md[i, j]) > 1:
            axe.text(j, i, f"{Md[i, j] / 1000:.1f}", ha="center", va="center", fontsize=6, color="black")
ax4 = fig.add_subplot(gs[4, :]); ax4.axis("off")
disp = summ.copy()
for cc in ["Net", "PerDay", "MaxDD", "Best", "Worst"]:
    disp[cc] = disp[cc].map(lambda v: f"{v:,.0f}")
disp["DayWin"] = disp["DayWin"].map(lambda v: f"{v:.0f}%"); disp["LegWin"] = disp["LegWin"].map(lambda v: f"{v:.0f}%")
tb = ax4.table(cellText=disp.values, colLabels=disp.columns, loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.6)
for j in range(len(disp.columns)):
    tb[0, j].set_facecolor("#222"); tb[0, j].set_text_props(color="white", fontweight="bold")
ax4.set_title("Per-system replay stats (net ₹, lots=2)", y=.92, fontweight="bold")
fig.savefig(OUT / "nas6_baseline.png", dpi=110, bbox_inches="tight")
print("WROTE", OUT / "nas6_baseline.png")
print("\n=== COMBINED (lots=2) ===")
print(comb_stats)
print("\n=== PER-SYSTEM ===")
print(summ.to_string(index=False))
print("\n=== NET BY DTE (combined) ===")
print(dte_comb.to_string())
oc.close()
