"""Weekly CPR regime: does the PRIOR WEEK's CPR width predict calm (range) DAYS this
week? Textbook: WIDE weekly CPR -> sideways week (calm days, good short-vol); NARROW ->
trending week. Tested over NIFTY 30-min 2020-2026, per-year. Frame stays INTRADAY
(each day's same-day |close-open|); weekly CPR is just the week-level label.
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
d["move_pct"] = (d["close"] - d["open"]).abs() / d["open"] * 100
d["year"] = d.index.year
d["wk"] = d.index.to_period("W-FRI")

# weekly OHLC -> weekly CPR -> PRIOR week's CPR applies to current week
wk = d.groupby("wk").agg(H=("high", "max"), L=("low", "min"), C=("close", "last"))
P = (wk["H"] + wk["L"] + wk["C"]) / 3; BC = (wk["H"] + wk["L"]) / 2; TC = 2 * P - BC
wk["cpr_pct"] = (TC - BC).abs() / wk["C"] * 100
wk["cpr_prior"] = wk["cpr_pct"].shift(1)        # prior week's CPR -> this week
wk["wk_range_pct"] = (wk["H"] - wk["L"]) / wk["C"] * 100   # this week's realised range
d = d.join(wk[["cpr_prior"]], on="wk").dropna(subset=["cpr_prior"])
d = d[(d["cpr_prior"] > 0) & (d["cpr_prior"] < 5)]
med = d["move_pct"].median(); d["range_day"] = (d["move_pct"] < med).astype(int)
d["cpr_q"] = pd.qcut(d["cpr_prior"], 4, labels=["Q1-narrow", "Q2", "Q3", "Q4-wide"])
by_q = d.groupby("cpr_q", observed=True).agg(n=("move_pct", "size"), avg_move=("move_pct", "mean"), range_rate=("range_day", "mean"))

# week-level: does prior-week CPR predict THIS week's realised range? (textbook framing)
wk2 = wk.dropna(subset=["cpr_prior"]).copy()
wk_corr = wk2["cpr_prior"].corr(wk2["wk_range_pct"])

yrs = sorted(d["year"].unique()); pyr = {}
for y in yrs:
    dd = d[d.year == y]
    if len(dd) < 40: continue
    q = pd.qcut(dd["cpr_prior"], 4, labels=False, duplicates="drop"); m = dd["move_pct"].median(); rd = (dd["move_pct"] < m).astype(int)
    pyr[y] = (rd[q == 0].mean(), rd[q == 3].mean())  # narrow, wide range-rate

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].bar(by_q.index.astype(str), by_q["range_rate"], color="#0a6"); ax[0].axhline(.5, color="#999")
for i, (idx, r) in enumerate(by_q.iterrows()): ax[0].text(i, r["range_rate"], f"mv {r['avg_move']:.2f}%", ha="center", va="bottom", fontsize=8)
ax[0].set_title("Range-day rate by PRIOR-WEEK CPR width (narrow->wide)\n[textbook: wide=calm/range]")
xs = list(pyr); w = .38
ax[1].bar([i - w/2 for i in range(len(xs))], [pyr[y][0] for y in xs], w, label="narrow wkCPR", color="#d33")
ax[1].bar([i + w/2 for i in range(len(xs))], [pyr[y][1] for y in xs], w, label="wide wkCPR", color="#0a6")
ax[1].set_xticks(range(len(xs))); ax[1].set_xticklabels(xs); ax[1].axhline(.5, color="#999"); ax[1].legend(fontsize=8)
ax[1].set_title("Range-day rate: narrow vs wide weekly CPR, PER YEAR")
fig.suptitle(f"Weekly CPR regime — NIFTY 30-min {d.index.min().date()}->{d.index.max().date()} ({len(d)} days, {len(wk2)} weeks)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "weekly_cpr.png", dpi=110, bbox_inches="tight"); print("WROTE weekly_cpr.png")

L = [f"# Weekly CPR regime — NIFTY 30-min ({d.index.min().date()}->{d.index.max().date()}, {len(d)} days)\n",
     f"Prior-WEEK CPR width -> this week's days. Textbook: WIDE weekly CPR = calmer (range) week.\n",
     f"**Week-level: corr(prior-week CPR, this-week realised range) = {wk_corr:.2f}** "
     f"(positive => WIDE prior-CPR predicts a BIGGER-range week = OPPOSITE of textbook; negative => textbook holds).\n",
     "## Daily range-day rate by prior-week CPR width\n| weekly CPR | n | avg daily move %% | range-day rate |\n|---|---|---|---|"]
for q in by_q.index: L.append(f"| {q} | {int(by_q.loc[q,'n'])} | {by_q.loc[q,'avg_move']:.2f} | {by_q.loc[q,'range_rate']:.2f} |")
L.append("\n## Narrow vs wide weekly-CPR range-day rate, PER YEAR\n| year | narrow | wide |\n|---|---|---|")
for y in pyr: L.append(f"| {y} | {pyr[y][0]:.2f} | {pyr[y][1]:.2f} |")
L.append("\n- If WIDE weekly-CPR range-rate > NARROW consistently AND week-corr < 0, the textbook (wide=calm) holds.")
(OUT / "RESULTS_weekly_cpr.md").write_text("\n".join(L), encoding="utf-8"); print("WROTE RESULTS_weekly_cpr.md")
