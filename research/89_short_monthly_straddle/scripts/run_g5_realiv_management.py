#!/usr/bin/env python3
"""
research/89 — G5 REAL-IV ACTIVE MANAGEMENT bake-off.

Marks the ATM straddle DAILY at REAL option closes (nse_options_bhav) and applies
best-practice management exits. Answers the user's reframe: don't hold a full month;
take profit / cut on criteria for a BETTER-PROBABILITY-of-calm shorter hold.

Entry conditions (each tested): calm (rv_rank<=0.3), after-spike (rv_rank>=0.7), all.
Exit policies (best practice): HTE, TP25, TP50, TP50_SL200, TP50_SL150_T10, EXIT_21DTE, MOVE5.
Structures: naked + iron fly (real wings). Net of cost. Per-tier, per-year, OOS.

Usage: run_g5_realiv_management.py [--symbols A,B]  (default all in table with >500 rows)
"""
import os, sys, math, sqlite3, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

DTE_LO, DTE_HI, DTE_TGT = 22, 45, 30
COST_FRAC = 0.0017
RESULTS = Path(__file__).resolve().parent.parent / "results"
INDEX_SYMS = {"NIFTY", "BANKNIFTY"}
UMAP = {"NIFTY": "NIFTY50", "BANKNIFTY": "BANKNIFTY"}

# policy = (name, tp, sl, tdays, dte_exit, movepct)  ; None = disabled
POLICIES = [
    ("HTE",            None, None, None, None, None),
    ("TP25",           0.25, None, None, None, None),
    ("TP50",           0.50, None, None, None, None),
    ("TP50_SL200",     0.50, 2.0,  None, None, None),
    ("TP50_SL150_T10", 0.50, 1.5,  10,   None, None),
    ("EXIT_21DTE",     None, None, None, 21,   None),
    ("TP50_MOVE5",     0.50, None, None, None, 0.05),
]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std() == 0: return float("nan")
    return x.mean()/(x.std(ddof=1)/math.sqrt(len(x)))


def leg_series(exp_piv, strike, otype):
    """Return dict date->close for a contract, from a per-expiry pivot; nearest strike."""
    if (strike, otype) in exp_piv.columns:
        s = exp_piv[(strike, otype)].dropna()
        return s
    return pd.Series(dtype=float)


def run_symbol(conn, symbol):
    und = UMAP.get(symbol, symbol)
    spot_df = E.load_daily(und, conn)
    if spot_df.empty: return None
    feat = E.add_features(spot_df); close = feat["close"]
    q = ("SELECT trade_date, expiry_date, strike, option_type, close FROM nse_options_bhav "
         "WHERE symbol=? AND close>0")
    ch = pd.read_sql_query(q, conn, params=(symbol,))
    if ch.empty: return None
    ch["td"] = pd.to_datetime(ch["trade_date"]); ch["ed"] = pd.to_datetime(ch["expiry_date"])
    # per-expiry pivot cache: index=td, columns=(strike,otype), values=close
    exp_pivots = {}
    for ed, g in ch.groupby("ed"):
        exp_pivots[ed] = g.pivot_table(index="td", columns=["strike", "option_type"],
                                       values="close", aggfunc="last")
    # entry-date -> available (expiry, dte)
    ch["dte"] = (ch["ed"] - ch["td"]).dt.days
    by_td = {d: g for d, g in ch.groupby("td")}
    tier = "A" if symbol in INDEX_SYMS else "C"
    dates = feat.index; last = -10**9
    trades = []
    for i in range(len(feat) - 1):
        dt = dates[i]
        if i - last < 20: continue
        rank = feat["rv_rank"].iloc[i]; rv0 = feat["rv20"].iloc[i]
        if not (np.isfinite(rank) and np.isfinite(rv0)): continue
        if abs(feat["gap"].iloc[i]) > 0.20: continue
        g = by_td.get(dt)
        if g is None: continue
        gm = g[(g.dte >= DTE_LO) & (g.dte <= DTE_HI)]
        if gm.empty: continue
        exp = gm.iloc[(gm.dte - DTE_TGT).abs().argmin()]["ed"]
        piv = exp_pivots.get(exp)
        if piv is None or dt not in piv.index: continue
        S0 = float(close.iloc[i])
        strikes = sorted({s for (s, ot) in piv.columns})
        strikes = np.array(strikes)
        K = float(strikes[np.argmin(np.abs(strikes - S0))])
        if abs(K/S0 - 1) > 0.06: continue
        try:
            ce0 = piv.loc[dt, (K, "CE")]; pe0 = piv.loc[dt, (K, "PE")]
        except KeyError:
            continue
        if not (np.isfinite(ce0) and np.isfinite(pe0)): continue
        premium0 = float(ce0 + pe0)
        if premium0 <= 0: continue
        # wing strikes for iron fly (~1x premium OTM)
        Kc = float(strikes[np.argmin(np.abs(strikes - (K + premium0)))])
        Kp = float(strikes[np.argmin(np.abs(strikes - (K - premium0)))])
        try:
            wc0 = float(piv.loc[dt, (Kc, "CE")]); wp0 = float(piv.loc[dt, (Kp, "PE")])
        except KeyError:
            wc0 = wp0 = np.nan
        wingcost0 = (wc0 + wp0) if (np.isfinite(wc0) and np.isfinite(wp0)) else None
        # forward path of this expiry's pivot from entry to expiry -> build REAL daily marks
        path = piv.loc[(piv.index > dt) & (piv.index <= exp)]
        marks = []   # list of (date, straddle_val, wing_val_real, Sx, dte_now)
        for pd_ in path.index:
            row = path.loc[pd_]
            ce = row.get((K, "CE"), np.nan); pe = row.get((K, "PE"), np.nan)
            if not (np.isfinite(ce) and np.isfinite(pe)):
                continue
            sv = float(ce + pe)
            wc = row.get((Kc, "CE"), np.nan); wp = row.get((Kp, "PE"), np.nan)
            wv = float((wc if np.isfinite(wc) else 0.0) + (wp if np.isfinite(wp) else 0.0))
            Sx = float(close.loc[pd_]) if pd_ in close.index else S0
            marks.append((pd_, sv, wv, Sx, (exp - pd_).days))
        S_exp = float(close.loc[exp]) if exp in close.index else (marks[-1][3] if marks else S0)
        base = dict(symbol=symbol, tier=tier, date=dt.date().isoformat(), year=dt.year,
                    rv_rank=float(rank), premium0=premium0, prem_pct=premium0/S0, S0=S0, K=K,
                    entry_dte=int((exp-dt).days))
        for (pname, tp, sl, tdays, dte_exit, movepct) in POLICIES:
            for struct in ("naked", "ironfly"):
                if struct == "ironfly" and wingcost0 is None: continue
                credit0 = premium0 if struct == "naked" else (premium0 - wingcost0)
                exited = False; held = 0
                for j, (pd_, sv, wv, Sx, dte_now) in enumerate(marks, 1):
                    posval = sv if struct == "naked" else (sv - wv)   # cost to close position
                    mtm = credit0 - posval
                    held = j
                    hit = False
                    if tp is not None and mtm >= tp*abs(credit0): hit = True
                    elif sl is not None and (-mtm) >= sl*abs(credit0): hit = True
                    elif tdays is not None and j >= tdays: hit = True
                    elif dte_exit is not None and dte_now <= dte_exit: hit = True
                    elif movepct is not None and abs(Sx-S0)/S0 >= movepct: hit = True
                    if hit:
                        pnl = credit0 - posval
                        turnover = premium0 + sv + (wv + wingcost0 if struct == "ironfly" else 0.0)
                        exited = True; break
                if not exited:   # settle at expiry intrinsic (both straddle & wings)
                    sv_e = abs(S_exp - K)
                    wv_e = max(0.0, S_exp - Kc) + max(0.0, Kp - S_exp)
                    posval = sv_e if struct == "naked" else (sv_e - wv_e)
                    pnl = credit0 - posval
                    turnover = premium0 + sv_e + (wv_e + wingcost0 if struct == "ironfly" else 0.0)
                    held = len(marks)
                net = pnl - COST_FRAC*turnover
                trades.append({**base, "policy": pname, "structure": struct,
                               "net_pct": net/S0, "win": 1 if net > 0 else 0, "held": held})
        last = i
    return pd.DataFrame(trades)


def report(d, tier_label):
    if d.empty: return
    log(f"\n########## {tier_label}  (entries~{d[(d.policy=='HTE')&(d.structure=='naked')].shape[0]}) ##########")
    for cond, sub in [("ALL", d), ("calm rv<=0.3", d[d.rv_rank <= 0.3]),
                      ("after-spike rv>=0.7", d[d.rv_rank >= 0.7])]:
        if sub.empty: continue
        log(f"--- entry: {cond} ---")
        agg = []
        for (pol, st), g in sub.groupby(["policy", "structure"]):
            agg.append(dict(policy=pol, struct=st, n=len(g),
                            net_bps=round(g.net_pct.mean()*10000, 1),
                            win=round(g.win.mean()*100, 1),
                            med_held=int(g.held.median()), t=round(tstat(g.net_pct), 2)))
        a = pd.DataFrame(agg).sort_values("net_bps", ascending=False)
        print(a.to_string(index=False), flush=True)
    # per-year + OOS for TP50 naked and TP50_SL200 ironfly on ALL entries
    for pol, st in [("TP50", "naked"), ("TP50_SL200", "ironfly")]:
        g = d[(d.policy == pol) & (d.structure == st)]
        if len(g) < 8: continue
        py = g.groupby("year")["net_pct"].agg(["count", "mean"])
        py["net_bps"] = (py["mean"]*10000).round(0)
        log(f"  [{pol}/{st}] per-year net bps: " +
            " ".join(f"{int(y)}:{int(r.net_bps)}({int(r['count'])})" for y, r in py.iterrows()))
        tr = g[g.year <= 2021]["net_pct"]; te = g[g.year >= 2022]["net_pct"]
        log(f"  [{pol}/{st}] OOS train<=2021 {tr.mean()*10000:.0f}bps(n={len(tr)},t={tstat(tr):.2f}) "
            f"| test>=2022 {te.mean()*10000:.0f}bps(n={len(te)},t={tstat(te):.2f})  "
            f"worst-yr {int(py.net_bps.min())}bps")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", default="")
    args = ap.parse_args()
    conn = sqlite3.connect(E.db_path())
    if args.symbols:
        syms = args.symbols.split(",")
    else:
        syms = [r[0] for r in conn.execute(
            "SELECT symbol, COUNT(*) c FROM nse_options_bhav GROUP BY symbol HAVING c>500 ORDER BY symbol")]
    log(f"symbols: {len(syms)}")
    allt = []
    for s in syms:
        d = run_symbol(conn, s)
        if d is None or d.empty:
            log(f"  {s:12s}: no trades"); continue
        allt.append(d); log(f"  {s:12s}: {d[(d.policy=='HTE')&(d.structure=='naked')].shape[0]} entries")
    if not allt: log("nothing"); return
    full = pd.concat(allt, ignore_index=True)
    full.to_csv(RESULTS/"g5_management_trades.csv", index=False)
    report(full[full.tier == "A"], "TIER A INDEX (real chain, daily-marked)")
    report(full[full.tier == "C"], "TIER C STOCKS (real chain, daily-marked)")
    log("\nDONE -> results/g5_management_trades.csv")

if __name__ == "__main__":
    import logging; logging.disable(logging.WARNING); main()
