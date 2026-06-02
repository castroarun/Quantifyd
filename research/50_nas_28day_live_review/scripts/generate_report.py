"""NAS 8-system 28-day live/paper performance review from REAL recorded trades.

Reads the 8 nas*trading.db trade tables (since START), aggregates per system and
per day, and renders a single comprehensive factsheet PNG plus RESULTS.md.

NOT a replay/backtest: these are the systems' actual recorded fills on real
markets. See the STATUS doc for caveats (small sample, lot-size change, messy tail).
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path("/home/arun/quantifyd")
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"
OUT.mkdir(parents=True, exist_ok=True)
START = "2026-04-20"

SYSTEMS = [
    ("nas_trading.db",          "nas_trades",     "Squeeze OTM"),
    ("nas_atm_trading.db",      "nas_atm_trades", "Squeeze ATM"),
    ("nas_atm2_trading.db",     "nas_atm_trades", "Squeeze ATM2"),
    ("nas_atm4_trading.db",     "nas_atm_trades", "Squeeze ATM4"),
    ("nas_916_otm_trading.db",  "nas_trades",     "916 OTM"),
    ("nas_916_atm_trading.db",  "nas_atm_trades", "916 ATM"),
    ("nas_916_atm2_trading.db", "nas_atm_trades", "916 ATM2"),
    ("nas_916_atm4_trading.db", "nas_atm_trades", "916 ATM4"),
]
ORDER = [s[2] for s in SYSTEMS]

# ---- load ------------------------------------------------------------------
frames = []
for db, tbl, label in SYSTEMS:
    p = ROOT / "backtest_data" / db
    if not p.exists():
        continue
    cx = sqlite3.connect(str(p))
    try:
        df = pd.read_sql_query(
            f"SELECT trade_date, net_pnl, gross_pnl, lots, adjustments, exit_reason "
            f"FROM {tbl} WHERE trade_date >= ?", cx, params=[START])
    except Exception as e:
        print(f"skip {db}: {e}"); cx.close(); continue
    cx.close()
    if df.empty:
        continue
    df["system"] = label
    frames.append(df)

allt = pd.concat(frames, ignore_index=True)
allt["trade_date"] = pd.to_datetime(allt["trade_date"])
allt["net_pnl"] = pd.to_numeric(allt["net_pnl"], errors="coerce").fillna(0.0)
allt["gross_pnl"] = pd.to_numeric(allt["gross_pnl"], errors="coerce").fillna(0.0)
allt["lots"] = pd.to_numeric(allt["lots"], errors="coerce")
allt["pnl_per_lot"] = allt["net_pnl"] / allt["lots"].replace(0, np.nan)
allt = allt.sort_values("trade_date")
allt.to_csv(OUT / "per_trade.csv", index=False)

dates = sorted(allt["trade_date"].unique())
present = [s for s in ORDER if s in set(allt["system"])]

def daily_pivot(col):
    d = allt.groupby(["system", "trade_date"])[col].sum().reset_index()
    return d.pivot(index="system", columns="trade_date", values=col).reindex(present).reindex(columns=dates)

pv_net = daily_pivot("net_pnl").fillna(0.0)
pv_lot = daily_pivot("pnl_per_lot").fillna(0.0)
cum_net = pv_net.cumsum(axis=1)
cum_lot = pv_lot.cumsum(axis=1)
combined_daily = pv_net.sum(axis=0)
combined_cum = combined_daily.cumsum()

def max_drawdown(series):
    peak = series.cummax()
    dd = series - peak
    return dd.min(), dd

# ---- per-system stats ------------------------------------------------------
rows = []
for s in present:
    g = allt[allt["system"] == s]
    cum = cum_net.loc[s]
    mdd, _ = max_drawdown(cum)
    rows.append(dict(
        System=s, Trades=len(g), Days=g["trade_date"].nunique(),
        WinPct=round(100 * (g["net_pnl"] > 0).mean(), 1),
        NetPnL=round(g["net_pnl"].sum()),
        PerLot=round(g["pnl_per_lot"].mean()) if g["pnl_per_lot"].notna().any() else 0,
        AvgTrade=round(g["net_pnl"].mean()),
        MaxDD=round(mdd),
        BestDay=round(pv_net.loc[s].max()), WorstDay=round(pv_net.loc[s].min()),
        Adj=int(g["adjustments"].fillna(0).sum()),
    ))
stats = pd.DataFrame(rows)
cmdd, cdd = max_drawdown(combined_cum)

# ---- figure ----------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.edgecolor": "#888", "figure.facecolor": "white"})
fig = plt.figure(figsize=(15, 19))
gs = GridSpec(5, 2, height_ratios=[0.5, 1.1, 1.1, 1.3, 1.0], hspace=0.42, wspace=0.18)

tot_net = allt["net_pnl"].sum()
tot_tr = len(allt)
win = 100 * (allt["net_pnl"] > 0).mean()
ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
ax0.text(0, 0.7, "NAS 8-System Performance — Real Recorded Trades", fontsize=18, fontweight="bold")
ax0.text(0, 0.32, f"{pd.to_datetime(dates[0]).date()} → {pd.to_datetime(dates[-1]).date()}  ·  "
                  f"{len(dates)} trading days  ·  {tot_tr} trades  ·  "
                  f"net ₹{tot_net:,.0f}  ·  win {win:.0f}%  ·  combined MaxDD ₹{cmdd:,.0f}",
         fontsize=11, color="#333")
ax0.text(0, -0.05, "Actual live/paper fills (not synthetic, not a replay). 28-day single-regime sample — "
                   "behaviour audit, NOT strategy validation. Lot size changed 10→2 mid-window (see per-lot panel).",
         fontsize=8.5, color="#a00", style="italic")

# combined equity + drawdown
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(combined_cum.index, combined_cum.values, color="#0a6", lw=2)
ax1.axhline(0, color="#999", lw=0.8)
ax1.set_title("Combined book — cumulative net P&L (₹)"); ax1.tick_params(axis="x", rotation=45)
ax2 = fig.add_subplot(gs[1, 1])
ax2.fill_between(cdd.index, cdd.values, 0, color="#d33", alpha=0.5)
ax2.set_title(f"Combined drawdown (₹)  ·  MaxDD ₹{cmdd:,.0f}"); ax2.tick_params(axis="x", rotation=45)

# per-system cumulative raw
ax3 = fig.add_subplot(gs[2, 0])
for s in present:
    ax3.plot(cum_net.columns, cum_net.loc[s].values, lw=1.4, label=s)
ax3.axhline(0, color="#999", lw=0.8)
ax3.set_title("Per-system cumulative net P&L — RAW ₹ (lot-size change distorts)")
ax3.legend(fontsize=7, ncol=2); ax3.tick_params(axis="x", rotation=45)

# per-system cumulative per-lot
ax4 = fig.add_subplot(gs[2, 1])
for s in present:
    ax4.plot(cum_lot.columns, cum_lot.loc[s].values, lw=1.4, label=s)
ax4.axhline(0, color="#999", lw=0.8)
ax4.set_title("Per-system cumulative — PER-LOT normalized ₹ (fair comparison)")
ax4.legend(fontsize=7, ncol=2); ax4.tick_params(axis="x", rotation=45)

# heatmap system x day
ax5 = fig.add_subplot(gs[3, :])
M = pv_net.values.astype(float)
vmax = np.nanpercentile(np.abs(M), 95) or 1
im = ax5.imshow(M, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
ax5.set_yticks(range(len(present))); ax5.set_yticklabels(present, fontsize=8)
ax5.set_xticks(range(len(dates)))
ax5.set_xticklabels([pd.to_datetime(d).strftime("%m-%d") for d in dates], rotation=90, fontsize=7)
ax5.set_title("Per-day × per-system net P&L (₹)  —  green=profit, red=loss, blank=no trade")
for i in range(len(present)):
    for j in range(len(dates)):
        v = M[i, j]
        if abs(v) > 1:
            ax5.text(j, i, f"{v/1000:.1f}", ha="center", va="center", fontsize=5.5,
                     color="black")
fig.colorbar(im, ax=ax5, fraction=0.025, pad=0.01, label="₹ net P&L")

# stats table
ax6 = fig.add_subplot(gs[4, :]); ax6.axis("off")
disp = stats.copy()
for c in ["NetPnL", "PerLot", "AvgTrade", "MaxDD", "BestDay", "WorstDay"]:
    disp[c] = disp[c].map(lambda x: f"{x:,.0f}")
disp["WinPct"] = disp["WinPct"].map(lambda x: f"{x:.0f}%")
tbl = ax6.table(cellText=disp.values, colLabels=disp.columns, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
for j in range(len(disp.columns)):
    tbl[0, j].set_facecolor("#222"); tbl[0, j].set_text_props(color="white", fontweight="bold")
ax6.set_title("Per-system stats (net ₹, since 2026-04-20)", y=0.92, fontsize=11, fontweight="bold")

fig.savefig(OUT / "nas_28day_review.png", dpi=110, bbox_inches="tight")
print("WROTE", OUT / "nas_28day_review.png")

# ---- RESULTS.md ------------------------------------------------------------
lines = []
lines.append("# NAS 8-System 28-Day Live/Paper Review — RESULTS\n")
lines.append(f"**Window:** {pd.to_datetime(dates[0]).date()} → {pd.to_datetime(dates[-1]).date()} "
             f"({len(dates)} trading days) · **{tot_tr} trades** · "
             f"**net ₹{tot_net:,.0f}** · win {win:.0f}% · combined MaxDD ₹{cmdd:,.0f}\n")
lines.append("**VERDICT: SIGNAL/BEHAVIOUR AUDIT ONLY** — 28-day single-regime sample on actual "
             "(not replayed) fills. Real per-trade economics are informative; cross-system ranking is "
             "indicative at best. NOT strategy validation.\n")
lines.append("## Per-system (net ₹ since 2026-04-20)\n")
lines.append("| " + " | ".join(stats.columns) + " |")
lines.append("|" + "|".join(["---"] * len(stats.columns)) + "|")
for _, r in stats.iterrows():
    lines.append("| " + " | ".join(str(r[c]) for c in stats.columns) + " |")
lines.append("\n## Caveats\n")
lines.append("- Lot size 10→2 mid-window → RAW cumulative not comparable over time; use per-lot panel.")
lines.append("- Mixed paper/live; restarts; 2026-06-02 trades distorted by exit/adjustment bugs, ghost "
             "legs and MANUAL_FLATTEN_RECONCILE handled that day.")
lines.append("- Exit-reason mix (all systems): " +
             ", ".join(f"{k}={v}" for k, v in allt["exit_reason"].fillna("NA").value_counts().head(8).items()))
(OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
print("WROTE", OUT / "RESULTS.md")
