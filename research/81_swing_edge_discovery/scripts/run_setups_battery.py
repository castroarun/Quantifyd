"""Classic setups battery: EXP-D2 (O=H/O=L), EXP-C2 (CPR open), EXP-A5
(PDH/PWH break), EXP-A6 (MA cross multi-TF). Locked grids — see
experiments/SETUPS_BATTERY/CLASSIC_SETUPS_BATTERY_5MIN_SWEEP_STATUS.md.
Universe: NIFTY + 9 deep names. IS 2015-02-01..2021-09-30. Costs 3bp.
"""
from __future__ import annotations

import sys
import time
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

RESULTS = STUDY / "experiments" / "SETUPS_BATTERY" / "results"
IS_START, IS_END = "2015-02-01", "2021-09-30"
SYMS = ["NIFTY50", "HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "TCS",
        "SBIN", "ITC", "HINDUNILVR", "BHARTIARTL"]
CFG = {ts: BTConfig(cost=CostConfig(product="FUTURES_PROXY"),
                    time_stop_sessions=ts) for ts in (1, 2, 4)}
TOL = 0.001


def clip(sig):
    return sig[(sig >= pd.Timestamp(IS_START))
               & (sig <= pd.Timestamp(IS_END + " 23:59:59"))]


def ent_df(sig, d, stop):
    return pd.DataFrame({"direction": d, "stop": stop.loc[sig],
                         "target": np.nan}, index=sig)


# ---------------- D2: open=low / open=high ----------------
def d2_entries(df5, kind):
    sess = df5.index.normalize()
    g = df5.groupby(sess)
    pos = g.cumcount()
    w = 6
    w_hi = df5["high"].where(pos < w).groupby(sess).transform("max")
    w_lo = df5["low"].where(pos < w).groupby(sess).transform("min")
    s_open = df5["open"].groupby(sess).transform("first")
    if kind == "OL_long":
        setup = (s_open <= w_lo * (1 + TOL))
        raw = setup & (pos >= w) & (df5["close"] > w_hi)
        d, stop = 1, w_lo
    else:  # OH_short
        setup = (s_open >= w_hi * (1 - TOL))
        raw = setup & (pos >= w) & (df5["close"] < w_lo)
        d, stop = -1, w_hi
    first = raw & (raw.groupby(sess).cumsum() == 1)
    return ent_df(clip(df5.index[first]), d, stop)


# ---------------- C2: CPR opening candle ----------------
def c2_entries(df5, kind):
    d1 = loader.to_daily(df5)
    P = ((d1["high"] + d1["low"] + d1["close"]) / 3).shift(1)
    BC = ((d1["high"] + d1["low"]) / 2).shift(1)
    TC = 2 * P - BC
    sess = df5.index.normalize()
    g = df5.groupby(sess)
    pos = g.cumcount()
    first_bar = pos == 0
    tc = pd.Series(TC.reindex(sess).values, index=df5.index)
    bc = pd.Series(BC.reindex(sess).values, index=df5.index)
    piv = pd.Series(P.reindex(sess).values, index=df5.index)
    lo_band, hi_band = pd.concat([tc, bc], axis=1).min(axis=1), \
        pd.concat([tc, bc], axis=1).max(axis=1)
    if kind == "long":
        raw = first_bar & (df5["close"] > df5["open"]) & (df5["close"] > hi_band)
        d, stop = 1, piv
    else:
        raw = first_bar & (df5["close"] < df5["open"]) & (df5["close"] < lo_band)
        d, stop = -1, piv
    raw &= piv.notna()
    return ent_df(clip(df5.index[raw]), d, stop)


# ---------------- A5: prev day / week range break ----------------
def a5_entries(df5, kind):
    d1 = loader.to_daily(df5)
    sess = df5.index.normalize()
    if kind.startswith("PD"):
        hi = d1["high"].shift(1)
        lo = d1["low"].shift(1)
        off = 0.5
    else:  # PW
        wk = d1.resample("W").agg(high=("high", "max"), low=("low", "min"))
        hi = wk["high"].shift(1).reindex(d1.index, method="ffill")
        lo = wk["low"].shift(1).reindex(d1.index, method="ffill")
        off = 0.33
    rng = hi - lo
    hi5 = pd.Series(hi.reindex(sess).values, index=df5.index)
    lo5 = pd.Series(lo.reindex(sess).values, index=df5.index)
    rng5 = pd.Series(rng.reindex(sess).values, index=df5.index)
    if kind.endswith("H_L"):
        raw = df5["close"] > hi5
        d, stop = 1, hi5 - off * rng5
    else:  # L_S
        raw = df5["close"] < lo5
        d, stop = -1, lo5 + off * rng5
    raw &= hi5.notna() & (rng5 > 0)
    first = raw & (raw.groupby(sess).cumsum() == 1)
    return ent_df(clip(df5.index[first]), d, stop)


# ---------------- A6: MA crossover multi-TF ----------------
def a6_entries(df5, tf, d):
    if tf == "15m":
        bars, f, s = loader.resample_intraday(df5, 15), 9, 21
    elif tf == "60m":
        bars, f, s = loader.resample_intraday(df5, 60), 9, 21
    else:
        bars, f, s = loader.to_daily(df5), 10, 20
    ef = bars["close"].ewm(span=f, adjust=False).mean()
    es = bars["close"].ewm(span=s, adjust=False).mean()
    up = (ef > es) & (ef.shift(1) <= es.shift(1))
    dn = (ef < es) & (ef.shift(1) >= es.shift(1))
    raw = up if d == 1 else dn
    sig = bars.index[raw & es.notna()]
    stop = es
    if tf == "daily":
        # map daily signal to the session's LAST 5-min bar for next-open entry
        last_bar = df5.groupby(df5.index.normalize()).tail(1).index
        m = {b.normalize(): b for b in last_bar}
        sig5 = pd.DatetimeIndex([m[t] for t in sig if t in m])
        stop5 = pd.Series(stop.reindex([t.normalize() for t in sig5]).values,
                          index=sig5)
        return pd.DataFrame({"direction": d, "stop": stop5,
                             "target": np.nan}, index=clip(sig5))
    # intraday TF: signal bar timestamp = bucket start; entry next 5-min bar
    # after bucket END -> use last 5-min bar inside the bucket as signal bar
    width = 15 if tf == "15m" else 60
    sig5, stops = [], []
    idx5 = df5.index
    for t in sig:
        end = t + pd.Timedelta(minutes=width - 5)
        j = idx5.searchsorted(end)
        if j < len(idx5):
            b = idx5[min(j, len(idx5) - 1)]
            if b.normalize() == t.normalize():
                sig5.append(b)
                stops.append(stop.loc[t])
    sig5 = pd.DatetimeIndex(sig5)
    return pd.DataFrame({"direction": d, "stop": stops,
                         "target": np.nan}, index=sig5).loc[clip(sig5)]


CELLS = (
    [("D2", f"{k}_ts{ts}", (d2_entries, (k,)), ts)
     for k in ("OL_long", "OH_short") for ts in (1, 2, 4)]
    + [("C2", f"cpr_{k}_ts{ts}", (c2_entries, (k,)), ts)
       for k in ("long", "short") for ts in (1, 2, 4)]
    + [("A5", f"{k}_ts{ts}", (a5_entries, (k,)), ts)
       for k in ("PDH_L", "PDL_S", "PWH_L", "PWL_S") for ts in (2, 4)]
    + [("A6", f"ma_{tf}_{'L' if d == 1 else 'S'}_ts{ts}", (a6_entries, (tf, d)), ts)
       for tf in ("15m", "60m", "daily") for d in (1, -1) for ts in (2, 4)]
)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_tr = []
    for sym in SYMS:
        df5 = loader.load_bars(sym, "5minute", start=IS_START, end=IS_END)
        if df5.empty:
            continue
        for exp, lbl, (fn, args), ts in CELLS:
            try:
                ent = fn(df5, *args)
                tr = run_symbol(df5, ent, CFG[ts], symbol=sym)
                tr["cell"], tr["exp"] = lbl, exp
                all_tr.append(tr)
            except Exception as e:
                print(f"{sym} {lbl} ERROR: {e}", flush=True)
        print(f"{sym} done ({(time.time()-t0)/60:.1f}m)", flush=True)

    trades = pd.concat(all_tr, ignore_index=True)
    trades.to_csv(RESULTS / "battery_trades.csv", index=False)
    rows = []
    for (exp, lbl), g in trades.groupby(["exp", "cell"]):
        r = g["net_ret"].to_numpy(float)
        if len(r) < 2:
            rows.append({"exp": exp, "cell": lbl, "n": len(r)})
            continue
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        sym_pnl = g.groupby("symbol")["net_ret"].sum()
        yr = pd.to_datetime(g["entry_time"]).dt.year
        yr_net = g.groupby(yr)["net_ret"].mean()
        rows.append({"exp": exp, "cell": lbl, "n": len(r),
                     "gross_bps": round(g["gross_ret"].mean() * 1e4, 1),
                     "net_bps": round(r.mean() * 1e4, 1),
                     "t_net": round(t, 2), "win": round((r > 0).mean(), 3),
                     "sym_pos": round((sym_pnl > 0).mean(), 2),
                     "yr_pos": round((yr_net > 0).mean(), 2)})
    agg = pd.DataFrame(rows).sort_values(["exp", "t_net"], ascending=[True, False])
    agg.to_csv(RESULTS / "battery_ranking.csv", index=False)
    print("\n=== SETUPS BATTERY ranking (net 3bp, IS 2015-2021) ===")
    print(agg.to_string(index=False))
    print(f"\ndone in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
