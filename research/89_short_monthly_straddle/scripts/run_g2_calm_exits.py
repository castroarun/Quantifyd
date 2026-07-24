#!/usr/bin/env python3
"""
research/89 — G2-lite: can EXITS rescue the calm-entry straddle, and is the
CORRECTLY-SPECIFIED version (sell only when real IV is rich vs realized) better?

Given G1 showed calm entries lose (vol mean-reverts UP after quiet), test:
  (A) Calm entries (rv_rank<=0.30), NAKED + IRONFLY, across exit policies:
        expiry / stop_move 1.0x / stop_move 1.5x / target 0.5 / dte 5
      -> does any exit turn calm-selling positive net?
  (B) INDEX real-IV VRP conditioning: entries where iv0/rv20 (IV richness vs
      trailing realized) is in the TOP tercile -> net edge? (stocks excluded:
      their modeled IV/RV is constant by construction.)

Non-overlapping entries (independent samples) for honest t-stats.
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

POLICIES = [
    ("expiry", ("expiry",)),
    ("stop_move_1.0", ("stop_move", 1.0)),
    ("stop_move_1.5", ("stop_move", 1.5)),
    ("target_0.5", ("target", 0.5)),
    ("dte_5", ("dte", 5)),
]


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
    if tier == "A" and vix_iv is not None:
        iv_ser = vix_iv.reindex(feat.index).ffill(limit=3)
    else:
        iv_ser = E.stock_iv_series(feat["rv20"], VRP_MULT)
    feat["iv"] = iv_ser
    return feat


def calm_indices(feat, mask):
    """Non-overlapping entry indices where mask true and features valid."""
    idx = []
    last = -10**9
    vals = feat.reset_index(drop=True)
    n = len(vals)
    for i in range(n - H - 1):
        if not mask.iloc[i]: continue
        if i - last < H: continue
        r = vals.iloc[i]
        if not (np.isfinite(r["rv_rank"]) and np.isfinite(r["rv20"]) and np.isfinite(r["iv"]) and r["iv"] > 0):
            continue
        if abs(r["gap"]) > 0.20: continue
        idx.append(i); last = i
    return idx


def run_policies(symbol, tier, feat, entry_idx):
    out = []
    for i in entry_idx:
        sub = feat.iloc[i:i + H + 1][["close", "high", "low"]]
        iv_path = feat["iv"].iloc[i:i + H + 1]
        for pname, pol in POLICIES:
            for struct in ("naked", "ironfly"):
                res = E.simulate_trade(sub, iv_path, H=H, structure=struct,
                                       wing_mult=1.0, cost_frac=COST_FRAC, exit_policy=pol)
                if res is None: continue
                out.append(dict(symbol=symbol, tier=tier, policy=pname, structure=struct,
                                net_pct=res["pnl_net_pct"], gross_pct=res["pnl_gross_pct"],
                                win=res["win"], held=res["held_days"], reason=res["exit_reason"]))
    return out


def main():
    t0 = time.time()
    conn = sqlite3.connect(E.db_path())
    log(f"DB: {E.db_path()}")
    vix = E.load_daily("INDIAVIX", conn)
    vix_iv = (vix["close"] / 100.0) if not vix.empty else None

    universe = [(s, "A") for s in E.TIER_A] + [(s, "C") for s in E.DEFAULT_STOCKS]

    # ---------- (A) calm-entry exit bake-off ----------
    rows = []
    for sym, tier in universe:
        feat = prep(sym, tier, conn, vix_iv)
        if feat is None:
            log(f"  {sym} skip"); continue
        calm_mask = feat["rv_rank"] <= 0.30
        ei = calm_indices(feat, calm_mask)
        rows += run_policies(sym, tier, feat, ei)
        log(f"  {sym:12s} ({tier}) calm entries={len(ei)}")
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "g2_calm_exit_trades.csv", index=False)

    log("\n===== (A) CALM-ENTRY EXIT BAKE-OFF (net bps, non-overlapping) =====")
    for tier_label, sub in [("A_index", d[d.tier == "A"]), ("C_stocks", d[d.tier == "C"])]:
        if sub.empty: continue
        log(f"--- {tier_label} ---")
        agg = []
        for (pol, struct), g in sub.groupby(["policy", "structure"]):
            agg.append(dict(policy=pol, structure=struct, n=len(g),
                            net_bps=round(g.net_pct.mean()*10000, 1),
                            win=round(g.win.mean()*100, 1),
                            med_held=int(g.held.median()),
                            t=round(tstat(g.net_pct), 2)))
        adf = pd.DataFrame(agg).sort_values(["structure", "policy"])
        print(adf.to_string(index=False), flush=True)
        adf.to_csv(RESULTS / f"g2_calm_exits_{tier_label}.csv", index=False)

    # ---------- (B) INDEX real-IV VRP-richness conditioning ----------
    log("\n===== (B) INDEX real-IV: sell only when IV is RICH vs realized =====")
    rowsB = []
    for sym in E.TIER_A:
        feat = prep(sym, sym and "A", conn, vix_iv)
        if feat is None: continue
        rich = feat["iv"] / feat["rv20"]           # IV richness vs trailing realized
        thr = rich.quantile(0.667)                 # top tercile of richness
        for tag, mask in [("all", pd.Series(True, index=feat.index)),
                          ("rich_IV", rich >= thr),
                          ("cheap_IV", rich <= rich.quantile(0.333))]:
            ei = calm_indices(feat, mask & feat["rv_rank"].notna())
            for i in ei:
                sub = feat.iloc[i:i + H + 1][["close", "high", "low"]]
                iv_path = feat["iv"].iloc[i:i + H + 1]
                res = E.simulate_trade(sub, iv_path, H=H, structure="naked",
                                       cost_frac=COST_FRAC, exit_policy=("expiry",))
                if res: rowsB.append(dict(symbol=sym, cond=tag, net_pct=res["pnl_net_pct"], win=res["win"]))
    db = pd.DataFrame(rowsB)
    if not db.empty:
        for cond, g in db.groupby("cond"):
            log(f"  {cond:9s} n={len(g):4d}  net={g.net_pct.mean()*10000:7.1f} bps  "
                f"win={g.win.mean()*100:4.1f}%  t={tstat(g.net_pct):5.2f}")
        db.to_csv(RESULTS / "g2_index_vrp_conditioning.csv", index=False)

    log(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import logging; logging.disable(logging.WARNING)
    main()
