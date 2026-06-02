"""Part (c): regime research on LONG underlying history (NIFTY50 5-min, 2024-03→2026-03,
~500 days). Short premium-selling profits on RANGE days (small close-to-open move) and
loses on TREND days. We can't replay option P&L over years (no historical premiums), so
we use the underlying realised move as a proxy and ask: which CAUSAL morning features
predict a range day? Per-year stability included (the robust part of this project).
"""
import sqlite3
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/arun/quantifyd"); BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"; OUT.mkdir(parents=True, exist_ok=True)
cx = sqlite3.connect(str(ROOT / "backtest_data" / "market_data.db"))
df = pd.read_sql_query("SELECT date, open, high, low, close FROM market_data_unified "
                       "WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date", cx)
cx.close()
df["dt"] = pd.to_datetime(df["date"]); df["day"] = df["dt"].dt.date
g = df.groupby("day")
daily = pd.DataFrame({
    "open": g["open"].first(), "close": g["close"].last(),
    "high": g["high"].max(), "low": g["low"].min(),
})
# first-15-min (first 3 bars) range
first15 = df[df["dt"].dt.time <= pd.to_datetime("09:30").time()].groupby("day").agg(
    f_hi=("high", "max"), f_lo=("low", "min"))
daily = daily.join(first15)
daily.index = pd.to_datetime(daily.index)
daily["weekday"] = daily.index.weekday
daily = daily[daily["weekday"] < 5]  # drop rare Sat special sessions (muhurat/budget)
daily["pc"] = daily["close"].shift(1); daily["po"] = daily["open"].shift(1)
daily["ph"] = daily["high"].shift(1); daily["pl"] = daily["low"].shift(1)
daily = daily.dropna()
# causal morning features (known ~09:30)
daily["gap_pct"] = (daily["open"] - daily["pc"]) / daily["pc"] * 100
daily["open15_pct"] = (daily["f_hi"] - daily["f_lo"]) / daily["open"] * 100
daily["prevrng_pct"] = (daily["ph"] - daily["pl"]) / daily["pc"] * 100
daily["prevret_pct"] = (daily["pc"] - daily["po"]) / daily["po"] * 100
# target: short-vol favourability = -|close-open| move (small move = range day = good)
daily["move_pct"] = (daily["close"] - daily["open"]).abs() / daily["open"] * 100
daily["rng_pct"] = (daily["high"] - daily["low"]) / daily["open"] * 100
daily["year"] = daily.index.year

feats = ["gap_pct", "open15_pct", "prevrng_pct", "prevret_pct"]
# correlation of features with the day's move (positive corr => feature predicts a TREND/move day)
corr_all = {f: daily[f].abs().corr(daily["move_pct"]) for f in feats}
corr_yr = {y: {f: d[f].abs().corr(d["move_pct"]) for f in feats} for y, d in daily.groupby("year")}

# range-day rate by weekday + by gap/open15 buckets
wd_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
move_thr = daily["move_pct"].median()
daily["range_day"] = (daily["move_pct"] < move_thr).astype(int)
by_wd = daily.groupby("weekday").agg(n=("move_pct", "size"), avg_move=("move_pct", "mean"),
                                     range_rate=("range_day", "mean"))
# open15 bucket: small opening range -> range day?
daily["o15_q"] = pd.qcut(daily["open15_pct"], 4, labels=["Q1-tight", "Q2", "Q3", "Q4-wide"])
by_o15 = daily.groupby("o15_q", observed=True).agg(n=("move_pct", "size"), avg_move=("move_pct", "mean"),
                                                   range_rate=("range_day", "mean"))
daily["gap_q"] = pd.qcut(daily["gap_pct"].abs(), 4, labels=["Q1-flat", "Q2", "Q3", "Q4-biggap"])
by_gap = daily.groupby("gap_q", observed=True).agg(n=("move_pct", "size"), avg_move=("move_pct", "mean"),
                                                   range_rate=("range_day", "mean"))

# ---- figure ----
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
ax[0, 0].bar([wd_names[w] for w in by_wd.index], by_wd["avg_move"], color="#369")
ax[0, 0].set_title("Avg intraday |close-open| %% by weekday (lower = better for short-vol)")
ax[0, 1].bar(by_o15.index.astype(str), by_o15["range_rate"], color="#0a6")
ax[0, 1].axhline(0.5, color="#999", lw=.8); ax[0, 1].set_title("Range-day rate by opening-15min range quartile")
ax[1, 0].bar(by_gap.index.astype(str), by_gap["range_rate"], color="#a60")
ax[1, 0].axhline(0.5, color="#999", lw=.8); ax[1, 0].set_title("Range-day rate by |overnight gap| quartile")
yrs = sorted(corr_yr); xw = 0.2
for i, f in enumerate(feats):
    ax[1, 1].bar([y + i*xw for y in range(len(yrs))], [corr_yr[y][f] for y in yrs], xw, label=f)
ax[1, 1].set_xticks([y + 1.5*xw for y in range(len(yrs))]); ax[1, 1].set_xticklabels(yrs)
ax[1, 1].axhline(0, color="#333", lw=.8); ax[1, 1].legend(fontsize=7)
ax[1, 1].set_title("Feature vs move corr — per-year stability")
fig.suptitle(f"Part (c) — NIFTY range/trend regime ({daily.index.min().date()}→{daily.index.max().date()}, "
             f"{len(daily)} days). Underlying proxy for short-vol favourability.", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "regime.png", dpi=110, bbox_inches="tight")
print("WROTE", OUT / "regime.png")

L = [f"# Part (c) — NIFTY range/trend regime ({daily.index.min().date()}→{daily.index.max().date()}, {len(daily)} days)\n",
     "Proxy: short premium-selling profits on small-|close-open| (range) days. Causal morning "
     "features only. **~500 days, per-year stability shown — the most robust part of this study, "
     "but it's an UNDERLYING proxy (no historical option P&L).**\n",
     "## Feature → |move| correlation (positive = feature flags a TREND/move day)\n",
     "| feature | all | " + " | ".join(str(y) for y in yrs) + " |",
     "|" + "|".join(["---"]*(len(yrs)+2)) + "|"]
for f in feats:
    L.append(f"| {f} | {corr_all[f]:.2f} | " + " | ".join(f"{corr_yr[y][f]:.2f}" for y in yrs) + " |")
L.append("\n## Avg intraday move %% + range-day rate by weekday")
L.append("| weekday | n | avg move %% | range-day rate |"); L.append("|---|---|---|---|")
for w in by_wd.index: L.append(f"| {wd_names[w]} | {int(by_wd.loc[w,'n'])} | {by_wd.loc[w,'avg_move']:.2f} | {by_wd.loc[w,'range_rate']:.2f} |")
L.append("\n## Range-day rate by opening-15min-range quartile")
L.append("| quartile | n | avg move %% | range-day rate |"); L.append("|---|---|---|---|")
for q in by_o15.index: L.append(f"| {q} | {int(by_o15.loc[q,'n'])} | {by_o15.loc[q,'avg_move']:.2f} | {by_o15.loc[q,'range_rate']:.2f} |")
L.append("\n## Range-day rate by |overnight gap| quartile")
L.append("| quartile | n | avg move %% | range-day rate |"); L.append("|---|---|---|---|")
for q in by_gap.index: L.append(f"| {q} | {int(by_gap.loc[q,'n'])} | {by_gap.loc[q,'avg_move']:.2f} | {by_gap.loc[q,'range_rate']:.2f} |")
L.append("\n- Read: features with a STABLE per-year correlation are usable filters; flickering-sign ones are noise.")
L.append("- Expiry-day shifted historically (Thu->Tue) so weekday!=DTE across all years; use as regime hint, "
         "combine with the 0/1-DTE options finding.")
(OUT / "RESULTS_regime.md").write_text("\n".join(L), encoding="utf-8")
print("WROTE", OUT / "RESULTS_regime.md")
