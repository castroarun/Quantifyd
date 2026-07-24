#!/usr/bin/env python3
"""
research/89 — G4 REAL-IV: re-run the straddle study on REAL option prices from
nse_options_bhav (no modeled IV). Entry premium = actual ATM CE+PE close; entry IV
= BS inversion of that real premium; settle to expiry on the real underlying close.

Compares calm-entry vs after-spike-entry, naked vs iron fly, NET of cost, per tier
(index / stocks), with per-year + OOS. This is the decision-grade test the modeled
engine could not do.

Usage: run_g4_realiv_flip.py [--symbols NIFTY,BANKNIFTY]   (default = all in table)
"""
import os, sys, math, sqlite3, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

DTE_LO, DTE_HI, DTE_TGT = 22, 45, 30     # monthly expiry window (calendar days)
H_TD = 20                                # max holding (trading days) if exiting pre-expiry
COST_FRAC = 0.0017
RESULTS = Path(__file__).resolve().parent.parent / "results"
INDEX_SYMS = {"NIFTY", "BANKNIFTY"}
UMAP = {"NIFTY": "NIFTY50", "BANKNIFTY": "BANKNIFTY"}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std() == 0: return float("nan")
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


def invert_iv(premium, S, K, T):
    """Bisection: sigma s.t. BS straddle == premium."""
    if premium <= 0 or T <= 0: return None
    lo, hi = 0.01, 3.0
    flo = E.straddle_value(S, K, T, lo) - premium
    fhi = E.straddle_value(S, K, T, hi) - premium
    if flo * fhi > 0: return None
    for _ in range(50):
        mid = (lo + hi) / 2
        fm = E.straddle_value(S, K, T, mid) - premium
        if abs(fm) < 1e-3: return mid
        if flo * fm < 0: hi = mid
        else: lo, flo = mid, fm
    return (lo + hi) / 2


def load_chain(conn, symbol):
    q = ("SELECT trade_date, expiry_date, strike, option_type, close, settle_price, open_interest "
         "FROM nse_options_bhav WHERE symbol=? AND close>0 ORDER BY trade_date")
    df = pd.read_sql_query(q, conn, params=(symbol,))
    return df


def build_real_trades(conn, symbol):
    und = UMAP.get(symbol, symbol)
    spot_df = E.load_daily(und, conn)
    if spot_df.empty:
        return None
    feat = E.add_features(spot_df)
    closes = feat["close"]
    chain = load_chain(conn, symbol)
    if chain.empty:
        return None
    chain["td"] = pd.to_datetime(chain["trade_date"])
    chain["ed"] = pd.to_datetime(chain["expiry_date"])
    chain["dte"] = (chain["ed"] - chain["td"]).dt.days
    by_date = {d: g for d, g in chain.groupby("td")}

    tier = "A" if symbol in INDEX_SYMS else "C"
    rows = []
    dates = feat.index
    last = -10**9
    for i in range(len(feat) - 1):
        dt = dates[i]
        if i - last < H_TD:   # non-overlapping
            continue
        rank = feat["rv_rank"].iloc[i]; rv20 = feat["rv20"].iloc[i]
        if not (np.isfinite(rank) and np.isfinite(rv20)): continue
        if abs(feat["gap"].iloc[i]) > 0.20: continue
        g = by_date.get(dt)
        if g is None: continue
        gm = g[(g.dte >= DTE_LO) & (g.dte <= DTE_HI)]
        if gm.empty: continue
        # choose expiry closest to target DTE
        exp = gm.iloc[(gm.dte - DTE_TGT).abs().argmin()]["ed"]
        ge = gm[gm.ed == exp]
        S0 = float(closes.iloc[i])
        # ATM strike = nearest to spot with BOTH CE & PE
        piv = ge.pivot_table(index="strike", columns="option_type", values="close", aggfunc="last")
        piv = piv.dropna(subset=[c for c in ("CE", "PE") if c in piv.columns])
        if not {"CE", "PE"}.issubset(piv.columns) or piv.empty: continue
        strikes = piv.index.values
        K = float(strikes[np.argmin(np.abs(strikes - S0))])
        if abs(K / S0 - 1) > 0.06: continue
        premium0 = float(piv.loc[K, "CE"] + piv.loc[K, "PE"])
        if premium0 <= 0: continue
        dte = int((exp - dt).days)
        T0 = dte / 365.0
        ivr = invert_iv(premium0, S0, K, T0)
        # settle to expiry on real underlying close
        exp_key = exp
        if exp_key in closes.index:
            S_exp = float(closes.loc[exp_key])
        else:
            fut = closes[closes.index >= exp]
            if fut.empty: continue
            S_exp = float(fut.iloc[0])
        intrinsic = abs(S_exp - K)
        gross_naked = premium0 - intrinsic
        # iron fly: buy wings ~1x premium away, from REAL chain
        wing = premium0
        Kc = float(strikes[np.argmin(np.abs(strikes - (K + wing)))])
        Kp = float(strikes[np.argmin(np.abs(strikes - (K - wing)))])
        wc = piv.loc[Kc, "CE"] if Kc in piv.index and not np.isnan(piv.loc[Kc, "CE"]) else 0.0
        wp = piv.loc[Kp, "PE"] if Kp in piv.index and not np.isnan(piv.loc[Kp, "PE"]) else 0.0
        wing_cost0 = float(wc + wp)
        wing_val = max(0.0, S_exp - Kc) + max(0.0, Kp - S_exp)
        gross_if = gross_naked + (wing_val - wing_cost0)
        cost_naked = COST_FRAC * (premium0 + intrinsic)
        cost_if = COST_FRAC * (premium0 + intrinsic + wing_cost0 + wing_val)
        rows.append(dict(
            symbol=symbol, tier=tier, date=dt.date().isoformat(), year=dt.year,
            rv_rank=float(rank), rv20=float(rv20), dte=dte, S0=S0, K=K,
            premium0=premium0, prem_pct=premium0 / S0, real_iv=ivr,
            net_naked_pct=(gross_naked - cost_naked) / S0,
            net_if_pct=(gross_if - cost_if) / S0,
            win_naked=1 if gross_naked - cost_naked > 0 else 0,
            win_if=1 if gross_if - cost_if > 0 else 0,
        ))
        last = i
    return pd.DataFrame(rows)


def report(d, label):
    log(f"\n########## {label}  (n={len(d)}) ##########")
    for cond, sub in [("calm rv_rank<=0.3", d[d.rv_rank <= 0.30]),
                      ("mid 0.3-0.7", d[(d.rv_rank > 0.3) & (d.rv_rank < 0.7)]),
                      ("after-spike rv_rank>=0.7", d[d.rv_rank >= 0.70])]:
        if sub.empty: continue
        log(f"  [{cond:26s}] n={len(sub):4d}  "
            f"NAKED net={sub.net_naked_pct.mean()*10000:7.1f}bps t={tstat(sub.net_naked_pct):5.2f} win={sub.win_naked.mean()*100:4.1f}%  |  "
            f"IRONFLY net={sub.net_if_pct.mean()*10000:7.1f}bps t={tstat(sub.net_if_pct):5.2f} win={sub.win_if.mean()*100:4.1f}%")
    # OOS on after-spike iron fly
    sp = d[d.rv_rank >= 0.70]
    if len(sp) > 6:
        tr = sp[sp.year <= 2021]["net_if_pct"]; te = sp[sp.year >= 2022]["net_if_pct"]
        log(f"  after-spike IRONFLY OOS: train<=2021 {tr.mean()*10000:.1f}bps(n={len(tr)},t={tstat(tr):.2f}) "
            f"| test>=2022 {te.mean()*10000:.1f}bps(n={len(te)},t={tstat(te):.2f})")
    if d["real_iv"].notna().any():
        log(f"  mean REAL entry IV={d.real_iv.mean():.3f}  vs mean RV20={d.rv20.mean():.3f}  "
            f"(VRP ratio {d.real_iv.mean()/d.rv20.mean():.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    args = ap.parse_args()
    conn = sqlite3.connect(E.db_path())
    if args.symbols:
        syms = args.symbols.split(",")
    else:
        syms = [r[0] for r in conn.execute(
            "SELECT symbol, COUNT(*) c FROM nse_options_bhav GROUP BY symbol HAVING c>500 ORDER BY symbol")]
    log(f"symbols: {len(syms)}")
    alld = []
    for s in syms:
        d = build_real_trades(conn, s)
        if d is None or d.empty:
            log(f"  {s:12s}: no trades"); continue
        alld.append(d)
        log(f"  {s:12s}: {len(d):4d} real trades  net_if={d.net_if_pct.mean()*10000:6.1f}bps")
    if not alld:
        log("no trades built"); return
    full = pd.concat(alld, ignore_index=True)
    full.to_csv(RESULTS / "g4_realiv_trades.csv", index=False)
    report(full[full.tier == "A"], "TIER A — INDEX (REAL chain IV)")
    report(full[full.tier == "C"], "TIER C — STOCKS (REAL chain IV)")
    log(f"\nDONE. {len(full)} trades -> results/g4_realiv_trades.csv")


if __name__ == "__main__":
    import logging; logging.disable(logging.WARNING)
    main()
