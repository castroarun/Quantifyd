"""research/155 — quick console summary of results/paths.csv (medians + paired stats)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 60)
RES = Path(__file__).resolve().parents[1] / "results"
p = pd.read_csv(RES / "paths.csv")
p = p.drop_duplicates(subset=["cell", "path"], keep="last")

cols = ["s_cagr", "s_dd", "s_calmar", "s_invested", "s_parked_pct", "s_trades",
        "b_cagr", "b_dd", "b_calmar", "b_dd_0809", "b_dd_1214", "b_dd_1314",
        "n_pull", "n_missed", "n_park", "pull_cost_x", "pull_tax_x",
        "corr_d_oa", "corr_d_tn", "corr_m_oa", "corr_m_tn"]
g = p.groupby("cell")[cols].median().round(3)
n = p.groupby("cell").size().rename("n")
g = g.join(n)

base = p[p.cell == "A_incumbent"].set_index("path")
rows = []
for c, sub in p.groupby("cell"):
    s = sub.set_index("path")
    common = base.index.intersection(s.index)
    if len(common) < 5:
        continue
    d_cagr = s.loc[common, "b_cagr"] - base.loc[common, "b_cagr"]
    d_cal = s.loc[common, "b_calmar"] - base.loc[common, "b_calmar"]
    d_dd = s.loc[common, "b_dd"] - base.loc[common, "b_dd"]
    rows.append(dict(cell=c, n=len(common),
                     d_cagr=round(float(d_cagr.median()), 3),
                     win_cagr=int((d_cagr > 0).sum()),
                     d_calmar=round(float(d_cal.median()), 3),
                     win_calmar=int((d_cal > 0).sum()),
                     d_dd=round(float(d_dd.median()), 3),
                     win_dd=int((d_dd > 0).sum())))
pr = pd.DataFrame(rows).set_index("cell")

out = g.join(pr[["d_cagr", "win_cagr", "d_calmar", "win_calmar", "d_dd", "win_dd"]])
if len(sys.argv) > 1:
    out = out[out.index.str.contains(sys.argv[1])]
print(out.sort_values("d_calmar", ascending=False).to_string())
