"""research/54 Stage 6 — multi-year CALM-DAY classifier (robust day-selection layer).

Short-vol wants CALM (low intraday-move) days. research/52 validated opening-range alone (6yr).
Q: does stacking causal 09:20-known features beat opening-range alone at predicting a calm day?
Features (all known by 09:20 -> tradeable, no look-ahead):
  fc_range  = first 5-min candle (09:15-09:20) high-low / open %
  gap       = |today 09:15 open - prev close| / prev close %
  pd_range  = prev-day (high-low)/prev close %
  pd_move   = prev-day |close-open|/open %
  wd        = weekday 0-4
Target: today_move = |close-open|/open %  (SMALLER = calmer = better to sell).
Model: linear (numpy lstsq), OOS per-year (train on all OTHER years, predict held-out year).
Compare multi-feature vs fc_range-alone: corr(pred, actual |move|) per year (LOWER pred -> sell).
NIFTY50 5-min from market_data.db (2018-2026). VALIDATION layer (years of real paths).
"""
import sqlite3
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/arun/quantifyd"); BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"; OUT.mkdir(parents=True, exist_ok=True)
cx = sqlite3.connect(str(ROOT / "backtest_data" / "market_data.db"))
df = pd.read_sql_query("SELECT date,open,high,low,close FROM market_data_unified "
                       "WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date", cx)
cx.close()
df["dt"] = pd.to_datetime(df["date"]); df["day"] = df["dt"].dt.date
print("5-min rows:", len(df), " days:", df["day"].nunique(), flush=True)

rows = []
prev = None
for day, g in df.groupby("day"):
    g = g.sort_values("dt")
    op = g["open"].iloc[0]; hi = g["high"].max(); lo = g["low"].min(); cl = g["close"].iloc[-1]
    fc = g[g["dt"].dt.time <= pd.to_datetime("09:20").time()]
    if len(fc) == 0 or op == 0:
        prev = dict(close=cl, hi=hi, lo=lo, op=op); continue
    fc_range = (fc["high"].max() - fc["low"].min()) / op * 100
    today_move = abs(cl - op) / op * 100
    rec = dict(day=day, wd=pd.Timestamp(day).weekday(), fc_range=fc_range, today_move=today_move,
               year=pd.Timestamp(day).year)
    if prev and prev["close"]:
        rec["gap"] = abs(op - prev["close"]) / prev["close"] * 100
        rec["pd_range"] = (prev["hi"] - prev["lo"]) / prev["close"] * 100
        rec["pd_move"] = abs(prev["close"] - prev["op"]) / prev["op"] * 100 if prev["op"] else np.nan
    rows.append(rec)
    prev = dict(close=cl, hi=hi, lo=lo, op=op)

D = pd.DataFrame(rows).dropna()
D = D[(D["wd"] < 5)]
print("usable days:", len(D), " years:", sorted(D['year'].unique()), flush=True)

FEATS = ["fc_range", "gap", "pd_range", "pd_move", "wd"]
def fit_pred(train, test, feats):
    X = train[feats].values; X = np.c_[np.ones(len(X)), X]; y = train["today_move"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    Xt = np.c_[np.ones(len(test)), test[feats].values]
    return Xt @ beta

years = sorted(D["year"].unique())
res = []
for yr in years:
    tr = D[D["year"] != yr]; te = D[D["year"] == yr]
    if len(te) < 20 or len(tr) < 50: continue
    pred_multi = fit_pred(tr, te, FEATS)
    pred_fc = fit_pred(tr, te, ["fc_range"])
    cm = np.corrcoef(pred_multi, te["today_move"])[0, 1]
    cf = np.corrcoef(pred_fc, te["today_move"])[0, 1]
    # decile test: of the 30% LOWEST predicted-move days, what is avg actual move vs overall?
    te = te.copy(); te["pm"] = pred_multi
    calm = te.nsmallest(max(int(len(te) * .3), 5), "pm")["today_move"].mean()
    res.append(dict(year=yr, n=len(te), corr_multi=round(cm, 2), corr_fc=round(cf, 2),
                    calm30_move=round(calm, 2), all_move=round(te["today_move"].mean(), 2)))
RES = pd.DataFrame(res)
print(RES.to_string(), flush=True)

# pooled feature correlations (sign + strength)
fcorr = {f: round(D[f].corr(D["today_move"]), 2) for f in FEATS}

L = ["# research/54 Stage 6 — multi-year calm-day classifier (NIFTY 5-min, %d days, OOS per-year)\n" % len(D),
     "Predict today's intraday move %% from 09:20-known features; LOWER predicted move => sell. "
     "OOS = train on all OTHER years, test on held-out year. **VALIDATION layer (years of real paths).**\n",
     "## Per-year OOS: corr(predicted, actual move) — multi-feature vs opening-range alone",
     "| year | n | corr multi | corr fc-only | calm30 avg move %% | all-day avg move %% |",
     "|---|---|---|---|---|---|"]
for _, r in RES.iterrows():
    L.append("| %d | %d | %.2f | %.2f | %.2f | %.2f |" % (r.year, r.n, r.corr_multi, r.corr_fc, r.calm30_move, r.all_move))
L += ["\n## Pooled feature corr with today's move (sign/strength)",
      "| feature | corr |", "|---|---|"] + ["| %s | %.2f |" % (f, fcorr[f]) for f in FEATS]
L += ["\n## Read",
      "- If `corr multi` > `corr fc-only` in MOST years -> stacking adds robust predictive power over opening-range alone.",
      "- `calm30 avg move` << `all-day avg move` every year => the classifier's calmest-30%% days really are calmer (tradeable filter).",
      "- Causal features only (known by 09:20). Linear model = conservative; a tree could do better but risks overfit on this n."]
(OUT / "RESULTS_calm_classifier.md").write_text("\n".join(L), encoding="utf-8")
print("DONE  feat_corr=%s" % fcorr, flush=True)
