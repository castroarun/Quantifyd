#!/usr/bin/env python3
"""
research/89 — G1 PROBE: is a low-vol regime harvestable by a short monthly straddle?

Cheapest gate. For every trading-day entry across Tier A (NIFTY50, BANKNIFTY;
REAL IV = INDIAVIX/100) and Tier C (F&O-liquid stocks; MODELED IV = RV20 x VRP):
  1. Persistence table by detector decile  -> must be MONOTONIC (calm predicts calm).
  2. Gross short-straddle-to-expiry EV by decile, conditional vs unconditional,
     with t-stat + non-overlapping (independent) subsample check.
  3. VRP calibration: mean INDIAVIX / NIFTY-RV20  -> sets the stock IV multiplier.

Short straddle to expiry: gross = premium0 - |S_H - S0|  (intrinsic at T=0).
Writes results/ incrementally. Prints progress per symbol.
"""
import os, sys, math, csv, sqlite3, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

H = 20                     # holding window (trading days ~ 1 month)
VRP_MULT = 1.15            # stock IV = RV20 * this (calibrated below, baseline)
COST_FRAC = 0.0017         # 0.17% of premium turned over (baseline, NIFTY-measured)
START = "2010-01-01"
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def calibrate_vrp(conn):
    """Mean ratio of real 30-day IV (INDIAVIX) to NIFTY trailing RV20."""
    nifty = E.load_daily("NIFTY50", conn)
    vix = E.load_daily("INDIAVIX", conn)
    if nifty.empty or vix.empty:
        return None
    rv = E.annualized_rv(nifty["close"], 20)
    iv = vix["close"] / 100.0
    j = pd.concat([rv.rename("rv"), iv.rename("iv")], axis=1).dropna()
    j = j[(j["rv"] > 0.02) & (j["iv"] > 0.02)]
    ratio = (j["iv"] / j["rv"])
    return dict(n=len(j), mean_ratio=float(ratio.mean()),
                median_ratio=float(ratio.median()),
                mean_iv=float(j["iv"].mean()), mean_rv=float(j["rv"].mean()),
                span=f"{j.index.min().date()}..{j.index.max().date()}")


def build_entries(symbol, tier, conn, vix_iv=None, vrp_mult=VRP_MULT):
    df = E.load_daily(symbol, conn)
    if df.empty or len(df) < 300:
        return None
    df = df[df.index >= START]
    if len(df) < 300:
        return None
    feat = E.add_features(df)
    close = feat["close"]
    # IV series
    if tier == "A" and vix_iv is not None:
        iv_ser = vix_iv.reindex(feat.index).ffill(limit=3)
    else:
        iv_ser = E.stock_iv_series(feat["rv20"], vrp_mult)
    rows = []
    idx = feat.index
    n = len(feat)
    for i in range(n - H - 1):
        rank = feat["rv_rank"].iloc[i]
        rv20 = feat["rv20"].iloc[i]
        iv0 = iv_ser.iloc[i]
        if not (np.isfinite(rank) and np.isfinite(rv20) and np.isfinite(iv0) and iv0 > 0):
            continue
        # skip entry the day AFTER a corporate-action-suspect gap
        if abs(feat["gap"].iloc[i]) > 0.20:
            continue
        S0 = float(close.iloc[i])
        K = S0
        T0 = H / E.TRADING_DAYS
        prem0 = E.straddle_value(S0, K, T0, float(iv0))
        if prem0 <= 0:
            continue
        S_H = float(close.iloc[i + H])
        intrinsic = abs(S_H - S0)
        gross = prem0 - intrinsic
        cost = COST_FRAC * (prem0 + intrinsic)
        net = gross - cost
        fwd_rv = E.forward_rv(close, i, H)
        rows.append(dict(
            symbol=symbol, tier=tier, date=idx[i].date().isoformat(),
            rv20=round(float(rv20), 4), rv_rank=round(float(rank), 4),
            bbw_rank=round(float(feat["bbw_rank"].iloc[i]), 4) if np.isfinite(feat["bbw_rank"].iloc[i]) else np.nan,
            iv0=round(float(iv0), 4), prem0_pct=round(prem0 / S0, 5),
            fwd_move_pct=round((S_H - S0) / S0, 5),
            fwd_rv=round(float(fwd_rv), 4) if np.isfinite(fwd_rv) else np.nan,
            gross_pct=round(gross / S0, 5), net_pct=round(net / S0, 5),
            win=1 if net > 0 else 0,
            entry_i=i,
        ))
    return pd.DataFrame(rows)


def decile_table(d, col_rank="rv_rank"):
    d = d.dropna(subset=[col_rank, "gross_pct"]).copy()
    d["decile"] = (d[col_rank] * 10).clip(0, 9).astype(int)
    g = d.groupby("decile")
    tab = pd.DataFrame({
        "n": g.size(),
        "entry_rv20": g["rv20"].mean().round(3),
        "fwd_rv": g["fwd_rv"].mean().round(3),
        "prem_pct": (g["prem0_pct"].mean() * 100).round(2),
        "abs_move_pct": (d.assign(a=d["fwd_move_pct"].abs()).groupby("decile")["a"].mean() * 100).round(2),
        "gross_bps": (g["gross_pct"].mean() * 10000).round(1),
        "net_bps": (g["net_pct"].mean() * 10000).round(1),
        "win_rate": (g["win"].mean() * 100).round(1),
    })
    # persistence: P(forward realized vol stays <= entry rv20 * 1.20)
    dd = d.dropna(subset=["fwd_rv"])
    persist = dd.assign(stay=(dd["fwd_rv"] <= dd["rv20"] * 1.20).astype(int)) \
                .groupby("decile")["stay"].mean() * 100
    tab["P_stay_calm"] = persist.round(1)
    return tab


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std() == 0:
        return float("nan")
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


def nonoverlap(d):
    """Keep one entry per symbol every H rows (independent samples)."""
    keep = []
    for sym, g in d.sort_values(["symbol", "entry_i"]).groupby("symbol"):
        last = -10 ** 9
        for _, r in g.iterrows():
            if r["entry_i"] - last >= H:
                keep.append(r)
                last = r["entry_i"]
    return pd.DataFrame(keep)


def main():
    t0 = time.time()
    conn = sqlite3.connect(E.db_path())
    log(f"DB: {E.db_path()}")

    vrp = calibrate_vrp(conn)
    if vrp:
        log(f"VRP calibration (INDIAVIX/NIFTY-RV20): mean={vrp['mean_ratio']:.3f} "
            f"median={vrp['median_ratio']:.3f}  meanIV={vrp['mean_iv']:.3f} "
            f"meanRV={vrp['mean_rv']:.3f}  n={vrp['n']} span={vrp['span']}")
        with open(RESULTS / "vrp_calibration.txt", "w") as f:
            f.write(str(vrp))
    vix = E.load_daily("INDIAVIX", conn)
    vix_iv = (vix["close"] / 100.0) if not vix.empty else None

    all_rows = []
    universe = [(s, "A") for s in E.TIER_A] + [(s, "C") for s in E.DEFAULT_STOCKS]
    per_csv = RESULTS / "g1_entries.csv"
    first = True
    for sym, tier in universe:
        d = build_entries(sym, tier, conn, vix_iv=vix_iv)
        if d is None or d.empty:
            log(f"  {sym:12s} ({tier}) : NO DATA / too short — skipped")
            continue
        d.to_csv(per_csv, mode="w" if first else "a", header=first, index=False)
        first = False
        all_rows.append(d)
        log(f"  {sym:12s} ({tier}) : {len(d):5d} entries | "
            f"gross {d['gross_pct'].mean()*10000:6.1f} bps | "
            f"net {d['net_pct'].mean()*10000:6.1f} bps | win {d['win'].mean()*100:4.1f}%")

    if not all_rows:
        log("No entries built — aborting."); return
    full = pd.concat(all_rows, ignore_index=True)
    log(f"TOTAL entries: {len(full)}")

    # ---- decile tables ----
    for tier_label, sub in [("ALL", full), ("A_index", full[full.tier == "A"]),
                            ("C_stocks", full[full.tier == "C"])]:
        if sub.empty:
            continue
        tab = decile_table(sub)
        tab.to_csv(RESULTS / f"g1_deciles_{tier_label}.csv")
        log(f"\n===== DECILE TABLE [{tier_label}] (decile 0 = calmest) =====")
        print(tab.to_string(), flush=True)

    # ---- conditional vs unconditional (calm = rv_rank<=0.3) ----
    log("\n===== CONDITIONAL vs UNCONDITIONAL (net bps to expiry) =====")
    summ = []
    for tier_label, sub in [("ALL", full), ("A_index", full[full.tier == "A"]),
                            ("C_stocks", full[full.tier == "C"])]:
        if sub.empty:
            continue
        calm = sub[sub.rv_rank <= 0.30]
        loud = sub[sub.rv_rank >= 0.70]
        no = nonoverlap(sub)
        no_calm = no[no.rv_rank <= 0.30]
        row = dict(
            tier=tier_label, n_all=len(sub), n_calm=len(calm),
            uncond_net_bps=round(sub.net_pct.mean() * 10000, 1),
            calm_net_bps=round(calm.net_pct.mean() * 10000, 1) if len(calm) else float("nan"),
            loud_net_bps=round(loud.net_pct.mean() * 10000, 1) if len(loud) else float("nan"),
            calm_gross_bps=round(calm.gross_pct.mean() * 10000, 1) if len(calm) else float("nan"),
            calm_win=round(calm.win.mean() * 100, 1) if len(calm) else float("nan"),
            uncond_win=round(sub.win.mean() * 100, 1),
            t_calm_overlap=round(tstat(calm.net_pct), 2) if len(calm) else float("nan"),
            t_calm_indep=round(tstat(no_calm.net_pct), 2) if len(no_calm) else float("nan"),
            n_calm_indep=len(no_calm),
        )
        summ.append(row)
        print(row, flush=True)
    pd.DataFrame(summ).to_csv(RESULTS / "g1_conditional_summary.csv", index=False)

    log(f"\nDONE in {time.time()-t0:.0f}s. Artifacts in {RESULTS}")


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)
    main()
