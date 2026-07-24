#!/usr/bin/env python3
"""
research/89 — G3 FLIP: short vol is a MEAN-REVERSION trade, so sell premium AFTER
a vol spike (not into calm), with a DEFINED-RISK iron fly.

Decision-grade leg = NIFTY50/BANKNIFTY with REAL IV (INDIAVIX). Stocks (modeled IV)
reported for breadth but flagged. Conditionings:
  loud   : rv_rank >= 0.70  (currently high-vol regime; G1 showed fwd vol falls)
  spike  : rv20 > 1.5 x rv20[t-20] AND rv20 now <= rv20[t-5]  (spiked THEN cooling)
Structures: naked + ironfly.  Exits: expiry / dte5 / target0.5 / stop_move1.5.
Non-overlapping entries. Reports per-tier table, per-year, and OOS (train<=2020 / test>=2021).
"""
import os, sys, math, sqlite3, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

H = 20
VRP_MULT = 1.15
COST_FRAC = 0.0017
START = "2010-01-01"
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

POLICIES = [("expiry", ("expiry",)), ("dte5", ("dte", 5)),
            ("target0.5", ("target", 0.5)), ("stop_move1.5", ("stop_move", 1.5))]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std() == 0: return float("nan")
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


def prep(symbol, tier, conn, vix_iv):
    df = E.load_daily(symbol, conn)
    if df.empty or len(df) < 300: return None
    df = df[df.index >= START]
    if len(df) < 300: return None
    feat = E.add_features(df)
    feat["iv"] = (vix_iv.reindex(feat.index).ffill(limit=3)
                  if (tier == "A" and vix_iv is not None)
                  else E.stock_iv_series(feat["rv20"], VRP_MULT))
    # conditioning signals
    feat["loud"] = feat["rv_rank"] >= 0.70
    rv = feat["rv20"]
    feat["spike"] = (rv > 1.5 * rv.shift(20)) & (rv <= rv.shift(5))
    return feat


def entries(feat, mask):
    idx, last, vals = [], -10**9, feat.reset_index(drop=True)
    for i in range(len(vals) - H - 1):
        if not bool(mask.iloc[i]): continue
        if i - last < H: continue
        r = vals.iloc[i]
        if not (np.isfinite(r["rv_rank"]) and np.isfinite(r["rv20"]) and np.isfinite(r["iv"]) and r["iv"] > 0):
            continue
        if abs(r["gap"]) > 0.20: continue
        idx.append(i); last = i
    return idx


def sim_all(symbol, tier, feat, ei):
    out = []
    dates = feat.index
    for i in ei:
        sub = feat.iloc[i:i + H + 1][["close", "high", "low"]]
        ivp = feat["iv"].iloc[i:i + H + 1]
        for pname, pol in POLICIES:
            for struct in ("naked", "ironfly"):
                res = E.simulate_trade(sub, ivp, H=H, structure=struct, wing_mult=1.0,
                                       cost_frac=COST_FRAC, exit_policy=pol)
                if res:
                    out.append(dict(symbol=symbol, tier=tier, policy=pname, structure=struct,
                                    year=dates[i].year, net_pct=res["pnl_net_pct"],
                                    win=res["win"], held=res["held_days"]))
    return out


def main():
    t0 = time.time()
    conn = sqlite3.connect(E.db_path())
    log(f"DB: {E.db_path()}")
    vix = E.load_daily("INDIAVIX", conn)
    vix_iv = (vix["close"] / 100.0) if not vix.empty else None
    universe = [(s, "A") for s in E.TIER_A] + [(s, "C") for s in E.DEFAULT_STOCKS]

    for cond in ("loud", "spike"):
        rows = []
        for sym, tier in universe:
            feat = prep(sym, tier, conn, vix_iv)
            if feat is None: continue
            ei = entries(feat, feat[cond])
            rows += sim_all(sym, tier, feat, ei)
        d = pd.DataFrame(rows)
        if d.empty:
            log(f"[{cond}] no entries"); continue
        d.to_csv(RESULTS / f"g3_flip_{cond}_trades.csv", index=False)
        log(f"\n########## CONDITION = {cond.upper()} ##########")
        for tier_label, sub in [("A_index_REALIV", d[d.tier == "A"]),
                                ("C_stocks_MODELED", d[d.tier == "C"])]:
            if sub.empty: continue
            agg = []
            for (pol, struct), g in sub.groupby(["policy", "structure"]):
                agg.append(dict(policy=pol, structure=struct, n=len(g),
                                net_bps=round(g.net_pct.mean()*10000, 1),
                                win=round(g.win.mean()*100, 1),
                                t=round(tstat(g.net_pct), 2)))
            adf = pd.DataFrame(agg).sort_values("net_bps", ascending=False)
            log(f"--- {tier_label} (n_trades total {len(sub)}) ---")
            print(adf.to_string(index=False), flush=True)
            adf.to_csv(RESULTS / f"g3_flip_{cond}_{tier_label}.csv", index=False)

        # headline cell = ironfly + expiry: per-year + OOS on the REAL-IV index
        idx = d[(d.tier == "A") & (d.structure == "ironfly") & (d.policy == "expiry")]
        if not idx.empty:
            log(f"[{cond}] INDEX ironfly/expiry — per-year net bps:")
            py = idx.groupby("year")["net_pct"].agg(["count", "mean"])
            py["net_bps"] = (py["mean"]*10000).round(1)
            print(py[["count", "net_bps"]].to_string(), flush=True)
            tr = idx[idx.year <= 2020]["net_pct"]; te = idx[idx.year >= 2021]["net_pct"]
            log(f"[{cond}] INDEX ironfly/expiry OOS: train<=2020 net={tr.mean()*10000:.1f}bps "
                f"(n={len(tr)},t={tstat(tr):.2f}) | test>=2021 net={te.mean()*10000:.1f}bps "
                f"(n={len(te)},t={tstat(te):.2f})")

    log(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import logging; logging.disable(logging.WARNING)
    main()
