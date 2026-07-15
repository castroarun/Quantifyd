"""EXP-F2: conditioning filters on the LOCKED F1 cell (NIFTY ORB W12 long ts4),
then a single pre-declared Validation-split confirmation.

Pre-registration: experiments/F2_nifty_orb_filters/NIFTY_ORB_FILTERS_5MIN_SWEEP_STATUS.md
IS = 2015-02-01..2021-09-30 (marginal filter analysis).
Val = 2021-10-01..2023-12-31 (ONE touch, locked config only, printed last).

Usage:
  run_f2_orb_filters.py            # IS filter table only
  run_f2_orb_filters.py --val CFG  # ONE Val confirmation of locked config
                                   # CFG in {base, gap_up, gap_up25, gap_down,
                                   #         prev_up, prev_down, vix_low, vix_mid, vix_high}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
for p in (str(STUDY), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader                              # noqa: E402
from engine.backtester import BTConfig, run_symbol     # noqa: E402
from engine.costs import CostConfig                    # noqa: E402

RESULTS = STUDY / "experiments" / "F2_nifty_orb_filters" / "results"
IS_START, IS_END = "2015-02-01", "2021-09-30"
VAL_START, VAL_END = "2021-10-01", "2023-12-31"
W = 12          # locked from F1
TS = 4          # locked from F1


def base_entries(df5: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    sess = df5.index.normalize()
    g = df5.groupby(sess)
    pos = g.cumcount()
    or_high = df5["high"].where(pos < W).groupby(sess).transform("max")
    or_low = df5["low"].where(pos < W).groupby(sess).transform("min")
    raw = (pos >= W) & (df5["close"] > or_high)
    first = raw & (raw.groupby(sess).cumsum() == 1)
    sig = df5.index[first]
    sig = sig[(sig >= pd.Timestamp(lo)) & (sig <= pd.Timestamp(hi + " 23:59:59"))]
    return pd.DataFrame({"direction": 1, "stop": or_low.loc[sig],
                         "target": np.nan}, index=sig)


def session_features(df5: pd.DataFrame) -> pd.DataFrame:
    """Per-session causal features: gap, prev-day trend, VIX tercile."""
    d = loader.to_daily(df5)
    feats = pd.DataFrame(index=d.index)
    feats["gap"] = d["open"] / d["close"].shift(1) - 1
    feats["prev_up"] = (d["close"] > d["open"]).shift(1)
    vix5 = loader.load_bars("INDIAVIX", "5minute", start="2014-01-01")
    vixd = loader.to_daily(vix5)["close"]
    lo_q = vixd.rolling(252).quantile(0.333).shift(1)
    hi_q = vixd.rolling(252).quantile(0.667).shift(1)
    vprev = vixd.shift(1)
    vix_bucket = pd.Series("mid", index=vixd.index)
    vix_bucket[vprev <= lo_q] = "low"
    vix_bucket[vprev >= hi_q] = "high"
    vix_bucket[vprev.isna() | lo_q.isna()] = "na"
    feats["vix"] = vix_bucket.reindex(feats.index)
    return feats


def allowed_sessions(feats: pd.DataFrame, cfg: str) -> pd.Index:
    if cfg == "base":
        m = pd.Series(True, index=feats.index)
    elif cfg == "gap_up":
        m = feats["gap"] > 0
    elif cfg == "gap_up25":
        m = feats["gap"] >= 0.0025
    elif cfg == "gap_down":
        m = feats["gap"] < 0
    elif cfg == "prev_up":
        m = feats["prev_up"] == True          # noqa: E712
    elif cfg == "prev_down":
        m = feats["prev_up"] == False         # noqa: E712
    elif cfg.startswith("vix_"):
        m = feats["vix"] == cfg.split("_")[1]
    else:
        raise ValueError(cfg)
    return feats.index[m.fillna(False)]


def evaluate(df5, entries, slip) -> dict:
    cfg = BTConfig(cost=CostConfig(product="FUTURES_PROXY", slippage=slip),
                   time_stop_sessions=TS)
    tr = run_symbol(df5, entries, cfg, symbol="NIFTY50")
    r = tr["net_ret"].to_numpy(float)
    if len(r) < 2:
        return {"n": len(r)}
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    yr = pd.to_datetime(tr["entry_time"]).dt.year
    yr_net = tr.groupby(yr)["net_ret"].mean()
    return {"n": len(r), "gross_bps": round(tr["gross_ret"].mean() * 1e4, 2),
            "net_bps": round(r.mean() * 1e4, 2), "t": round(t, 2),
            "win": round((r > 0).mean(), 3),
            "pct_yr_pos": round((yr_net > 0).mean(), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="", help="locked config for the ONE Val touch")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.val:
        df5 = loader.load_bars("NIFTY50", "5minute", start="2021-06-01", end=VAL_END)
        feats = session_features(loader.load_bars("NIFTY50", "5minute",
                                                  start="2015-02-01", end=VAL_END))
        ent = base_entries(df5, VAL_START, VAL_END)
        keep = ent.index.normalize().isin(allowed_sessions(feats, args.val))
        ent = ent[keep]
        print(f"=== VAL TOUCH (ONE-TIME) config={args.val} "
              f"{VAL_START}..{VAL_END} ===")
        for slip in (0.0001, 0.0003):
            print(f"slip {slip*1e4:.0f}bp:", evaluate(df5, ent, slip))
        return

    df5 = loader.load_bars("NIFTY50", "5minute", start="2015-02-01", end=IS_END)
    feats = session_features(df5)
    ent_all = base_entries(df5, IS_START, IS_END)
    rows = []
    for cfg in ("base", "gap_up", "gap_up25", "gap_down", "prev_up",
                "prev_down", "vix_low", "vix_mid", "vix_high"):
        keep = ent_all.index.normalize().isin(allowed_sessions(feats, cfg))
        ent = ent_all[keep]
        r1 = evaluate(df5, ent, 0.0001)
        r3 = evaluate(df5, ent, 0.0003)
        rows.append({"config": cfg, "n": r1.get("n"),
                     "net1bp": r1.get("net_bps"), "t1bp": r1.get("t"),
                     "net3bp": r3.get("net_bps"), "t3bp": r3.get("t"),
                     "win": r1.get("win"), "pct_yr_pos": r1.get("pct_yr_pos")})
        print(rows[-1], flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / "f2_is_filters.csv", index=False)
    print("\n=== F2 IS filter table ===")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
