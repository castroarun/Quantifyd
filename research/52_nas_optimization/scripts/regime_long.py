"""Validate the opening-range -> range-day finding over MORE years.

Part (c) used NIFTY 5-min (2024-26, 451 days). We also have NIFTY50 30-min back to
2020 (~6 years incl. 2020 COVID crash, 2021 bull, 2022 bear) — a much harder OOS test.
Opening range = first 30-min bar's (high-low). Same question: does a TIGHT open predict
a RANGE day (small |close-open|)? Per-year stability across 2020-2026 is the real test.
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
                       "WHERE symbol='NIFTY50' AND timeframe='30minute' ORDER BY date", cx)
cx.close()
df["dt"] = pd.to_datetime(df["date"]); df["day"] = df["dt"].dt.date
g = df.groupby("day")
daily = pd.DataFrame({"open": g["open"].first(), "close": g["close"].last(),
                      "high": g["high"].max(), "low": g["low"].min(),
                      "f_hi": g["high"].first(), "f_lo": g["low"].first()})  # first 30-min bar
daily.index = pd.to_datetime(daily.index)
daily = daily[daily.index.weekday < 5]
daily["open30_pct"] = (daily["f_hi"] - daily["f_lo"]) / daily["open"] * 100
daily["move_pct"] = (daily["close"] - daily["open"]).abs() / daily["open"] * 100
daily["year"] = daily.index.year
daily = daily.dropna()
daily = daily[(daily["open30_pct"] > 0) & (daily["open30_pct"] < 5)]  # drop degenerate

corr_all = daily["open30_pct"].corr(daily["move_pct"])
yrs = sorted(daily["year"].unique())
corr_yr = {y: daily[daily.year == y]["open30_pct"].corr(daily[daily.year == y]["move_pct"]) for y in yrs}

# range-day rate (move below that year's median) by opening-range quartile, pooled + per year
daily["range_day"] = (daily["move_pct"] < daily["move_pct"].median()).astype(int)
daily["q"] = pd.qcut(daily["open30_pct"], 4, labels=["Q1-tight", "Q2", "Q3", "Q4-wide"])
by_q = daily.groupby("q", observed=True).agg(n=("move_pct", "size"), avg_move=("move_pct", "mean"), range_rate=("range_day", "mean"))
# per-year tight(Q1) vs wide(Q4) range-rate to show stability
pyr = {}
for y in yrs:
    d = daily[daily.year == y]
    if len(d) < 40: continue
    qs = pd.qcut(d["open30_pct"], 4, labels=False, duplicates="drop")
    med = d["move_pct"].median(); rd = (d["move_pct"] < med).astype(int)
    pyr[y] = (rd[qs == 0].mean(), rd[qs == 3].mean())  # tight, wide range-day rate

fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
ax[0].bar([str(y) for y in yrs], [corr_yr[y] for y in yrs], color="#e8821e")
ax[0].axhline(0, color="#333"); ax[0].set_title(f"open30 vs |move| corr per year (all={corr_all:.2f})\npositive & stable = real predictor")
ax[1].bar(by_q.index.astype(str), by_q["range_rate"], color="#0a6"); ax[1].axhline(.5, color="#999")
ax[1].set_title("Range-day rate by opening-30min-range quartile (pooled 2020-26)")
xs = list(pyr.keys()); w = .38
ax[2].bar([i - w/2 for i in range(len(xs))], [pyr[y][0] for y in xs], w, label="tight (Q1)", color="#0a6")
ax[2].bar([i + w/2 for i in range(len(xs))], [pyr[y][1] for y in xs], w, label="wide (Q4)", color="#d33")
ax[2].set_xticks(range(len(xs))); ax[2].set_xticklabels(xs); ax[2].axhline(.5, color="#999"); ax[2].legend(fontsize=8)
ax[2].set_title("Range-day rate: tight vs wide open, PER YEAR")
fig.suptitle(f"Opening-range validation over NIFTY 30-min {daily.index.min().date()}->{daily.index.max().date()} ({len(daily)} days)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "regime_long.png", dpi=110, bbox_inches="tight")
print("WROTE", OUT / "regime_long.png")

L = [f"# Opening-range validation over 6 years (NIFTY 30-min, {daily.index.min().date()}->{daily.index.max().date()}, {len(daily)} days)\n",
     f"Tighter open should predict a smaller-move (range) day. **all-period corr(open30, |move|) = {corr_all:.2f}** "
     "(positive = tight open -> calmer day). Per-year stability across COVID(2020)/bull(2021)/bear(2022)/2023-26 "
     "is the real test.\n",
     "## Per-year correlation (open30 vs |move|)\n", "| year | corr |", "|---|---|"]
for y in yrs: L.append(f"| {y} | {corr_yr[y]:.2f} |")
L.append("\n## Range-day rate by opening-30min quartile (pooled)\n| quartile | n | avg move %% | range-day rate |\n|---|---|---|---|")
for q in by_q.index: L.append(f"| {q} | {int(by_q.loc[q,'n'])} | {by_q.loc[q,'avg_move']:.2f} | {by_q.loc[q,'range_rate']:.2f} |")
L.append("\n## Tight (Q1) vs Wide (Q4) range-day rate — PER YEAR")
L.append("| year | tight-open range-rate | wide-open range-rate |\n|---|---|---|")
for y in pyr: L.append(f"| {y} | {pyr[y][0]:.2f} | {pyr[y][1]:.2f} |")
L.append("\n- If tight>wide every year, the opening-range filter is robust across regimes (not a 2024-26 fluke).")
(OUT / "RESULTS_regime_long.md").write_text("\n".join(L), encoding="utf-8")
print("WROTE", OUT / "RESULTS_regime_long.md")
