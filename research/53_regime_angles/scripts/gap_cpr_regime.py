"""Layer 1: do GAP size and CPR WIDTH predict a range (short-vol-friendly) day?
NIFTY 30-min 2020-2026 (~1,565 days). Per-year stability is the test.

CPR (from PRIOR day H/L/C): P=(H+L+C)/3, BC=(H+L)/2, TC=2P-BC, width=|TC-BC|/C.
Theory: WIDE CPR -> sideways/range day (good for short-vol); NARROW CPR -> trend day.
Range day = small intraday |close-open|.
"""
import sqlite3
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/arun/quantifyd"); BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"; OUT.mkdir(parents=True, exist_ok=True)
cx = sqlite3.connect(str(ROOT / "backtest_data" / "market_data.db"))
df = pd.read_sql_query("SELECT date,open,high,low,close FROM market_data_unified "
                       "WHERE symbol='NIFTY50' AND timeframe='30minute' ORDER BY date", cx); cx.close()
df["dt"] = pd.to_datetime(df["date"]); df["day"] = df["dt"].dt.date
g = df.groupby("day")
d = pd.DataFrame({"open": g["open"].first(), "close": g["close"].last(), "high": g["high"].max(), "low": g["low"].min()})
d.index = pd.to_datetime(d.index); d = d[d.index.weekday < 5]
# prior-day H/L/C for gap + CPR
d["pc"] = d["close"].shift(1); d["ph"] = d["high"].shift(1); d["pl"] = d["low"].shift(1)
d = d.dropna()
d["gap_pct"] = (d["open"] - d["pc"]) / d["pc"] * 100
P = (d["ph"] + d["pl"] + d["pc"]) / 3; BC = (d["ph"] + d["pl"]) / 2; TC = 2 * P - BC
d["cpr_pct"] = (TC - BC).abs() / d["pc"] * 100
d["move_pct"] = (d["close"] - d["open"]).abs() / d["open"] * 100
d["year"] = d.index.year
d = d[(d["cpr_pct"] > 0) & (d["cpr_pct"] < 3)]
med = d["move_pct"].median(); d["range_day"] = (d["move_pct"] < med).astype(int)

# GAP buckets
d["gap_b"] = pd.cut(d["gap_pct"], [-99, -0.5, -0.15, 0.15, 0.5, 99],
                    labels=["gap dn >0.5%", "dn 0.15-0.5", "flat ±0.15", "up 0.15-0.5", "gap up >0.5%"])
by_gap = d.groupby("gap_b", observed=True).agg(n=("move_pct", "size"), avg_move=("move_pct", "mean"), range_rate=("range_day", "mean"))
# CPR width quartiles (narrow -> wide)
d["cpr_q"] = pd.qcut(d["cpr_pct"], 4, labels=["Q1-narrow", "Q2", "Q3", "Q4-wide"])
by_cpr = d.groupby("cpr_q", observed=True).agg(n=("move_pct", "size"), avg_move=("move_pct", "mean"), range_rate=("range_day", "mean"))
# per-year CPR narrow vs wide range-rate (stability)
yrs = sorted(d["year"].unique()); pyr = {}
for y in yrs:
    dd = d[d.year == y]
    if len(dd) < 40: continue
    q = pd.qcut(dd["cpr_pct"], 4, labels=False, duplicates="drop"); m = dd["move_pct"].median(); rd = (dd["move_pct"] < m).astype(int)
    pyr[y] = (rd[q == 0].mean(), rd[q == 3].mean())  # narrow, wide range-rate
corr_cpr_yr = {y: d[d.year == y]["cpr_pct"].corr(d[d.year == y]["move_pct"]) for y in yrs}

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
ax[0, 0].bar(by_gap.index.astype(str), by_gap["range_rate"], color="#369"); ax[0, 0].axhline(.5, color="#999")
ax[0, 0].set_title("Range-day rate by GAP bucket"); ax[0, 0].tick_params(axis="x", rotation=20)
ax[0, 1].bar(by_gap.index.astype(str), by_gap["avg_move"], color="#36a"); ax[0, 1].set_title("Avg intraday |move|%% by GAP bucket"); ax[0, 1].tick_params(axis="x", rotation=20)
ax[1, 0].bar(by_cpr.index.astype(str), by_cpr["range_rate"], color="#0a6"); ax[1, 0].axhline(.5, color="#999")
ax[1, 0].set_title("Range-day rate by CPR width (narrow->wide)  [theory: wide=range]")
xs = list(pyr); w = .38
ax[1, 1].bar([i - w/2 for i in range(len(xs))], [pyr[y][0] for y in xs], w, label="narrow CPR", color="#d33")
ax[1, 1].bar([i + w/2 for i in range(len(xs))], [pyr[y][1] for y in xs], w, label="wide CPR", color="#0a6")
ax[1, 1].set_xticks(range(len(xs))); ax[1, 1].set_xticklabels(xs); ax[1, 1].axhline(.5, color="#999"); ax[1, 1].legend(fontsize=8)
ax[1, 1].set_title("Range-day rate: narrow vs wide CPR, PER YEAR")
fig.suptitle(f"Gap & CPR regimes — NIFTY 30-min {d.index.min().date()}->{d.index.max().date()} ({len(d)} days)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "gap_cpr_regime.png", dpi=110, bbox_inches="tight"); print("WROTE gap_cpr_regime.png")

L = [f"# Gap & CPR regimes — NIFTY 30-min ({d.index.min().date()}->{d.index.max().date()}, {len(d)} days)\n",
     "Range day = small intraday |close-open| (good for short-vol). Higher range-rate = more short-vol-friendly.\n",
     "## Range-day rate + avg move by GAP bucket\n| gap bucket | n | avg move %% | range-day rate |\n|---|---|---|---|"]
for q in by_gap.index: L.append(f"| {q} | {int(by_gap.loc[q,'n'])} | {by_gap.loc[q,'avg_move']:.2f} | {by_gap.loc[q,'range_rate']:.2f} |")
L.append("\n## Range-day rate by CPR width (narrow->wide)\n| CPR width | n | avg move %% | range-day rate |\n|---|---|---|---|")
for q in by_cpr.index: L.append(f"| {q} | {int(by_cpr.loc[q,'n'])} | {by_cpr.loc[q,'avg_move']:.2f} | {by_cpr.loc[q,'range_rate']:.2f} |")
L.append("\n## CPR: narrow vs wide range-day rate, PER YEAR (corr cpr_width vs |move| in parens)")
L.append("| year | narrow CPR range-rate | wide CPR range-rate | corr |\n|---|---|---|---|")
for y in pyr: L.append(f"| {y} | {pyr[y][0]:.2f} | {pyr[y][1]:.2f} | {corr_cpr_yr[y]:.2f} |")
L.append("\n- CPR theory holds if WIDE CPR range-rate > NARROW consistently. Gap: large gaps (up or dn) should "
         "show higher move / lower range-rate if gaps drive trend; if symmetric gap-fill, range-rate stays ~flat.")
(OUT / "RESULTS_gap_cpr.md").write_text("\n".join(L), encoding="utf-8"); print("WROTE RESULTS_gap_cpr.md")
