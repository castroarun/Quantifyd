"""research/153 — client chart: growth of Rs 100 (log) vs NIFTY 50 / MIDCAP150 / SMLCAP250,
with a drawdown panel beneath. Median across the 30 selection seeds.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
R146 = Path("/home/arun/quantifyd/research/146_complementary_third_sleeve/results")
DB = Path("/home/arun/quantifyd/backtest_data/market_data.db")

ipo = pd.read_csv(RES / "ipo_equity_seeds.csv", index_col=0, parse_dates=True)
oa = pd.read_csv(R146 / "oa_navs.csv", index_col=0, parse_dates=True)
tn = pd.concat([pd.read_csv(R146 / f"tn_nav_off{o}.csv", index_col=0,
                            parse_dates=True).rename(columns={"0": f"off{o}"})
                for o in (0, 4, 8)], axis=1)

con = sqlite3.connect(str(DB))
idxs = {}
for sym, lab in (("NIFTY50", "NIFTY 50"), ("NIFTYMIDCAP150", "Midcap 150"),
                 ("NIFTYSMLCAP250", "Smallcap 250"), ("NIFTYBEES", "NIFTYBEES")):
    d = pd.read_sql_query("select date, close from market_data_unified where symbol=? "
                          "and timeframe='day' order by date", con, params=(sym,))
    if len(d):
        d["date"] = pd.to_datetime(d["date"].str[:10])
        idxs[lab] = d.set_index("date")["close"].astype(float)
        print(f"{lab}: {d['date'].min().date()} -> {d['date'].max().date()} ({len(d)} rows)")
con.close()

common = ipo.index.intersection(oa.index).intersection(tn.index)
ipo, oa, tn = ipo.loc[common], oa.loc[common], tn.loc[common]
ipo_m = (ipo / ipo.iloc[0]).resample("ME").last().pct_change().fillna(0)
oa_m = (oa / oa.iloc[0]).resample("ME").last().pct_change().fillna(0)
tn_m = (tn / tn.iloc[0]).resample("ME").last().pct_change().fillna(0)

series = {}
series["IPO Base sleeve (median of 30 seeds)"] = (ipo / ipo.iloc[0]).median(axis=1) * 100
b2 = pd.DataFrame({f"{o}|{t}": (1 + .5 * oa_m[o] + .5 * tn_m[t]).cumprod()
                   for o in oa_m.columns for t in tn_m.columns}).median(axis=1) * 100
b3 = pd.DataFrame({f"{o}|{t}|{i}": (1 + .4 * oa_m[o] + .4 * tn_m[t] + .2 * ipo_m[i]).cumprod()
                   for o in oa_m.columns for t in tn_m.columns
                   for i in list(ipo_m.columns)[:10]}).median(axis=1) * 100
series["TN+OA 50-50 (deployed baseline)"] = b2.reindex(common).ffill()
series["TN+OA+IPO 40/40/20 (candidate)"] = b3.reindex(common).ffill()
for lab, s in idxs.items():
    if lab == "NIFTYBEES":
        continue
    ss = s.reindex(common).ffill().dropna()
    if len(ss) > 100:
        series[lab] = ss / ss.iloc[0] * 100

fig, (ax, axd) = plt.subplots(2, 1, figsize=(13.5, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.15]})
colors = {"IPO Base sleeve (median of 30 seeds)": "#e8833a",
          "TN+OA 50-50 (deployed baseline)": "#7f8c9b",
          "TN+OA+IPO 40/40/20 (candidate)": "#2e86de",
          "NIFTY 50": "#9b59b6", "Midcap 150": "#27ae60", "Smallcap 250": "#c0392b"}
for lab, s in series.items():
    lw = 2.4 if "IPO" in lab or "40/40/20" in lab else 1.5
    ls = "-" if "NIFTY" not in lab and "cap" not in lab else "--"
    ax.plot(s.index, s.values, label=lab, lw=lw, ls=ls, color=colors.get(lab))
ax.set_yscale("log")
ax.set_ylabel("Growth of Rs 100 (log scale)")
ax.set_title("research/153 — IPO Base breakout sleeve vs the deployed book and the indices\n"
             "after-tax (20% STCG / 12.5% LTCG, FY netting), 25 bps/side, 5% idle-cash yield; "
             "medians across 30 selection seeds", fontsize=11)
ax.grid(alpha=.25, which="both")
ax.legend(fontsize=9, loc="upper left")

for lab in ("IPO Base sleeve (median of 30 seeds)", "TN+OA 50-50 (deployed baseline)",
            "TN+OA+IPO 40/40/20 (candidate)"):
    s = series[lab]
    axd.fill_between(s.index, (s / s.cummax() - 1) * 100, 0, alpha=.35,
                     color=colors.get(lab), label=lab)
axd.set_ylabel("Drawdown %")
axd.grid(alpha=.25)
axd.legend(fontsize=8, loc="lower left")
plt.tight_layout()
out = RES / "ipo_base_research153.png"
plt.savefig(out, dpi=125)
print("wrote", out)
