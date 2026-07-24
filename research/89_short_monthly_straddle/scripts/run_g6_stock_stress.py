#!/usr/bin/env python3
"""
research/89 — G6 ROBUSTNESS/ADVERSARIAL on the Tier-C stock edge.

The G5 stock edge (+188bps iron fly, every year +) is suspiciously clean. Stress it:
  - record GROSS pnl + turnover + entry ATM liquidity (OI, volume) per trade,
  - so we can post-process a realistic COST SWEEP and a LIQUIDITY filter,
  - per-name concentration + per-year + OOS.

Two headline policies only: TP50 naked, TP50_SL200 iron fly. Stocks only.
Writes results/g6_stock_stress_trades.csv (gross+turnover+liquidity, cost applied later).
"""
import os, sys, math, sqlite3, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

DTE_LO, DTE_HI, DTE_TGT = 22, 45, 30
RESULTS = Path(__file__).resolve().parent.parent / "results"
INDEX_SYMS = {"NIFTY", "BANKNIFTY"}
UMAP = {"NIFTY": "NIFTY50", "BANKNIFTY": "BANKNIFTY"}
# (name, tp, sl, struct)
POLICIES = [("TP50", 0.50, None, "naked"), ("TP50_SL200", 0.50, 2.0, "ironfly")]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def run_symbol(conn, symbol):
    und = UMAP.get(symbol, symbol)
    spot_df = E.load_daily(und, conn)
    if spot_df.empty: return None
    feat = E.add_features(spot_df); close = feat["close"]
    q = ("SELECT trade_date, expiry_date, strike, option_type, close, open_interest, contracts "
         "FROM nse_options_bhav WHERE symbol=? AND close>0")
    ch = pd.read_sql_query(q, conn, params=(symbol,))
    if ch.empty: return None
    ch["td"] = pd.to_datetime(ch["trade_date"]); ch["ed"] = pd.to_datetime(ch["expiry_date"])
    ch["dte"] = (ch["ed"] - ch["td"]).dt.days
    pv_close, pv_oi, pv_vol = {}, {}, {}
    for ed, g in ch.groupby("ed"):
        pv_close[ed] = g.pivot_table(index="td", columns=["strike","option_type"], values="close", aggfunc="last")
        pv_oi[ed]    = g.pivot_table(index="td", columns=["strike","option_type"], values="open_interest", aggfunc="last")
        pv_vol[ed]   = g.pivot_table(index="td", columns=["strike","option_type"], values="contracts", aggfunc="last")
    by_td = {d: g for d, g in ch.groupby("td")}
    tier = "A" if symbol in INDEX_SYMS else "C"
    dates = feat.index; last = -10**9; out = []
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
        piv = pv_close.get(exp)
        if piv is None or dt not in piv.index: continue
        S0 = float(close.iloc[i]); strikes = np.array(sorted({s for (s,ot) in piv.columns}))
        K = float(strikes[np.argmin(np.abs(strikes - S0))])
        if abs(K/S0 - 1) > 0.06: continue
        try:
            ce0 = piv.loc[dt,(K,"CE")]; pe0 = piv.loc[dt,(K,"PE")]
        except KeyError: continue
        if not (np.isfinite(ce0) and np.isfinite(pe0)): continue
        premium0 = float(ce0+pe0)
        if premium0 <= 0: continue
        # entry liquidity
        def lk(pv, s, ot):
            try:
                v = pv.loc[dt,(s,ot)]; return float(v) if np.isfinite(v) else 0.0
            except KeyError: return 0.0
        atm_oi = lk(pv_oi[exp],K,"CE")+lk(pv_oi[exp],K,"PE")
        atm_vol = lk(pv_vol[exp],K,"CE")+lk(pv_vol[exp],K,"PE")
        Kc = float(strikes[np.argmin(np.abs(strikes-(K+premium0)))])
        Kp = float(strikes[np.argmin(np.abs(strikes-(K-premium0)))])
        try:
            wc0=float(piv.loc[dt,(Kc,"CE")]); wp0=float(piv.loc[dt,(Kp,"PE")])
        except KeyError: wc0=wp0=np.nan
        wingcost0 = (wc0+wp0) if (np.isfinite(wc0) and np.isfinite(wp0)) else None
        path = piv.loc[(piv.index>dt)&(piv.index<=exp)]
        marks=[]
        for pd_ in path.index:
            row=path.loc[pd_]; ce=row.get((K,"CE"),np.nan); pe=row.get((K,"PE"),np.nan)
            if not (np.isfinite(ce) and np.isfinite(pe)): continue
            wc=row.get((Kc,"CE"),np.nan); wp=row.get((Kp,"PE"),np.nan)
            wv=float((wc if np.isfinite(wc) else 0.0)+(wp if np.isfinite(wp) else 0.0))
            Sx=float(close.loc[pd_]) if pd_ in close.index else S0
            marks.append((sv:=float(ce+pe), wv, Sx))
        S_exp = float(close.loc[exp]) if exp in close.index else (marks[-1][2] if marks else S0)
        rec=dict(symbol=symbol,tier=tier,date=dt.date().isoformat(),year=dt.year,rv_rank=float(rank),
                 prem_pct=premium0/S0,atm_oi=atm_oi,atm_vol=atm_vol,S0=S0)
        for (pname,tp,sl,struct) in POLICIES:
            if struct=="ironfly" and wingcost0 is None:
                rec[f"gross_{pname}"]=np.nan; rec[f"turn_{pname}"]=np.nan; continue
            credit0 = premium0 if struct=="naked" else (premium0-wingcost0)
            exited=False
            for (sv,wv,Sx) in marks:
                posval = sv if struct=="naked" else (sv-wv)
                mtm = credit0-posval
                if (tp is not None and mtm>=tp*abs(credit0)) or (sl is not None and (-mtm)>=sl*abs(credit0)):
                    pnl=credit0-posval
                    turn=premium0+sv+((wv+wingcost0) if struct=="ironfly" else 0.0)
                    exited=True; break
            if not exited:
                sv_e=abs(S_exp-K); wv_e=max(0.0,S_exp-Kc)+max(0.0,Kp-S_exp)
                posval=sv_e if struct=="naked" else (sv_e-wv_e)
                pnl=credit0-posval; turn=premium0+sv_e+((wv_e+wingcost0) if struct=="ironfly" else 0.0)
            rec[f"gross_{pname}"]=pnl/S0; rec[f"turn_{pname}"]=turn/S0
        out.append(rec); last=i
    return pd.DataFrame(out)

def main():
    conn=sqlite3.connect(E.db_path())
    syms=[r[0] for r in conn.execute(
        "SELECT symbol,COUNT(*) c FROM nse_options_bhav WHERE symbol NOT IN ('NIFTY','BANKNIFTY') "
        "GROUP BY symbol HAVING c>500 ORDER BY symbol")]
    log(f"stock symbols: {len(syms)}")
    allt=[]
    for s in syms:
        d=run_symbol(conn,s)
        if d is None or d.empty: log(f"  {s}: none"); continue
        allt.append(d); log(f"  {s:12s}: {len(d)}")
    full=pd.concat(allt,ignore_index=True)
    full.to_csv(RESULTS/"g6_stock_stress_trades.csv",index=False)
    log(f"DONE {len(full)} trades -> results/g6_stock_stress_trades.csv")

if __name__=="__main__":
    import logging; logging.disable(logging.WARNING); main()
