#!/usr/bin/env python3
"""
research/127 — per-symbol daily ATM straddle IV series from nse_options_bhav.

For each trade_date: nearest monthly expiry with DTE in [25,60], ATM strike
(nearest to spot, both legs close>0), BS-invert the straddle price -> IV.
Causal: uses only that day's chain. Output: results/iv_daily.csv
(symbol,date,dte,K,iv). Rank/VRP computed at analysis time.
"""
import sys, math, sqlite3, time, csv
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "89_short_monthly_straddle" / "scripts"))
import engine as E

RESULTS = HERE.parent / "results"
OUT = RESULTS / "iv_daily.csv"
DTE_LO, DTE_HI = 25, 60

def straddle_iv(S, K, T, price):
    lo, hi = 1e-3, 3.0
    if not (E.straddle_value(S, K, T, lo) < price < E.straddle_value(S, K, T, hi)):
        return np.nan
    for _ in range(40):
        mid = 0.5*(lo+hi)
        if E.straddle_value(S, K, T, mid) < price: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def run_symbol(conn, symbol):
    spot = E.load_daily(symbol, conn)
    if spot.empty: return []
    close = spot["close"]
    ch = pd.read_sql_query(
        "SELECT trade_date,expiry_date,strike,option_type,close FROM nse_options_bhav "
        "WHERE symbol=? AND close>0", conn, params=(symbol,))
    if ch.empty: return []
    ch["td"] = pd.to_datetime(ch["trade_date"]); ch["ed"] = pd.to_datetime(ch["expiry_date"])
    ch["dte"] = (ch["ed"] - ch["td"]).dt.days
    ch = ch[(ch.dte >= DTE_LO) & (ch.dte <= DTE_HI)]
    rows = []
    for dt, g in ch.groupby("td"):
        if dt not in close.index: continue
        S = float(close.loc[dt])
        # nearest expiry to 45 dte among those present
        ed = g.iloc[(g.dte - 45).abs().argsort()].iloc[0]["ed"]
        gg = g[g.ed == ed]
        piv = gg.pivot_table(index="strike", columns="option_type", values="close", aggfunc="last")
        if "CE" not in piv or "PE" not in piv: continue
        both = piv.dropna()
        if both.empty: continue
        K = float(both.index[np.argmin(np.abs(both.index.values - S))])
        if abs(K/S - 1) > 0.06: continue
        price = float(both.loc[K, "CE"] + both.loc[K, "PE"])
        T = (ed - dt).days / 365.0
        iv = straddle_iv(S, K, T, price)
        if np.isfinite(iv):
            rows.append((symbol, dt.date().isoformat(), int((ed-dt).days), K, round(iv, 4)))
    return rows

def main():
    conn = sqlite3.connect(E.db_path())
    syms = [r[0] for r in conn.execute(
        "SELECT symbol, COUNT(*) c FROM nse_options_bhav "
        "WHERE symbol NOT IN ('NIFTY','BANKNIFTY') GROUP BY symbol HAVING c>500 ORDER BY symbol")]
    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, usecols=["symbol"])["symbol"].unique())
    hdr = not OUT.exists()
    for i, s in enumerate(syms):
        if s in done: continue
        t0 = time.time()
        rows = run_symbol(conn, s)
        with open(OUT, "a", newline="") as f:
            w = csv.writer(f)
            if hdr: w.writerow(["symbol","date","dte","K","iv"]); hdr = False
            w.writerows(rows)
        print(f"[{i+1}/{len(syms)}] {s}: {len(rows)} days ({time.time()-t0:.0f}s)", flush=True)
    print("DONE ->", OUT, flush=True)

if __name__ == "__main__":
    import logging; logging.disable(logging.WARNING)
    main()
