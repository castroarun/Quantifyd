"""research/75 Phase 2 — predictability: can the morning (open-burst), the gap, DTE, or the prior
day forecast whether TODAY's 11:15-13:00 lull will stay calm? Exploratory (n~34, small). Reads
section_metrics.csv + fullday_minute_paths.csv."""
import pandas as pd, numpy as np
from pathlib import Path

OUT = Path("/home/arun/quantifyd/research/75_intraday_time_sections/results")
S = pd.read_csv(OUT / "section_metrics.csv")
P = pd.read_csv(OUT / "fullday_minute_paths.csv", dtype={"hm": str})


def dayfeat(g):
    g = g.sort_values("hm")
    mo = g[(g.hm >= "09:15") & (g.hm < "10:00")]
    lull = g[(g.hm >= "11:15") & (g.hm < "13:00")]           # the DATA-DRIVEN true lull
    return pd.Series(dict(open=g.spot.iloc[0], close=g.spot.iloc[-1], hi=g.spot.max(), lo=g.spot.min(),
        open_rng=(mo.spot.max() - mo.spot.min()) if len(mo) > 2 else np.nan,
        open_net=abs(mo.spot.iloc[-1] - mo.spot.iloc[0]) if len(mo) > 2 else np.nan,
        lull_rng=(lull.spot.max() - lull.spot.min()) if len(lull) > 2 else np.nan))


F = P.groupby(["day", "dte"]).apply(dayfeat, include_groups=False).reset_index().sort_values("day").reset_index(drop=True)
F["prev_close"] = F["close"].shift(1)
F["gap_abs"] = ((F["open"] - F["prev_close"]) / F["prev_close"] * 100).abs()
F["prev_range"] = (F["hi"].shift(1) - F["lo"].shift(1))
F["prev_lull"] = F["lull_rng"].shift(1)
F = F.dropna(subset=["lull_rng"])
print("=== Phase 2 — predicting TODAY's 11:15-13:00 lull range (lower = calmer)  n=%d ===" % len(F))
print("lull_range: mean %.0f  median %.0f  (calm<=median vs active>median)" % (F.lull_rng.mean(), F.lull_rng.median()))

print("\n-- linear correlation of each predictor with today's lull_range --")
for feat in ["open_rng", "open_net", "gap_abs", "prev_range", "prev_lull", "dte"]:
    sub = F.dropna(subset=[feat, "lull_rng"])
    if len(sub) > 6:
        r = np.corrcoef(sub[feat], sub["lull_rng"])[0, 1]
        flag = "  <-- notable" if abs(r) >= 0.35 else ""
        print("  %-11s corr %+.2f  (n=%d)%s" % (feat, r, len(sub), flag))

print("\n-- split: does a predictor separate calm vs active lulls? (mean lull_range) --")
for feat in ["open_rng", "gap_abs", "prev_range", "prev_lull", "dte"]:
    sub = F.dropna(subset=[feat]).copy()
    if len(sub) < 10:
        continue
    med = sub[feat].median()
    lo = sub[sub[feat] <= med]["lull_rng"].mean()
    hi = sub[sub[feat] > med]["lull_rng"].mean()
    print("  low %-10s -> lull %3.0f   |   high %-10s -> lull %3.0f   (Δ %+.0f)" % (feat, lo, feat, hi, hi - lo))

# hit-rate: does a quiet open predict a quiet lull?
q = F.dropna(subset=["open_rng"]).copy()
q["open_quiet"] = q["open_rng"] <= q["open_rng"].median()
q["lull_quiet"] = q["lull_rng"] <= q["lull_rng"].median()
ct = pd.crosstab(q["open_quiet"], q["lull_quiet"])
print("\n-- quiet OPEN -> quiet LULL? (contingency) --\n", ct.to_string())
agree = (q["open_quiet"] == q["lull_quiet"]).mean()
print("  open-mood matches lull-mood on %.0f%% of days (50%% = no signal)" % (100 * agree))
