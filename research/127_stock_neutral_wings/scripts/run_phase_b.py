#!/usr/bin/env python3
"""
research/127 — Phase B (G2): pure options-EOD parametric sweep.

One-axis-at-a-time around the Phase A base (E45/X21/W5/SL200/TP50/ATM).
No stock-price indicators involved — options chain only (spot used solely for
ATM pick + %S0 normalization). Per (config,symbol) resume-safe incremental CSV.

Configs: DTE_entry {30,40,50,60}, DTE_exit {10,15,28}, wing {3,7,10}%,
SL {150,300,none}/TP{none}, strangle offset {2.5,5}% — 17 total incl. base.

Run on VPS: /home/arun/quantifyd/venv/bin/python3 run_phase_b.py
"""
import os, sys, math, sqlite3, time, csv
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "89_short_monthly_straddle" / "scripts"))
import engine as E

RESULTS = HERE.parent / "results"
OUT_CSV = RESULTS / "phase_b_trades.csv"
ATM_BAND = 0.06
WING_MIN_PCT = 0.02

BASE = dict(dte_entry=45, dte_exit=21, tp=0.50, sl=2.00, wing_pct=0.05, k=0.0)
def cfgs():
    out = [("BASE", dict(BASE))]
    for v in [30, 40, 50, 60]: out.append((f"E{v}", dict(BASE, dte_entry=v)))
    for v in [10, 15, 28]:     out.append((f"X{v}", dict(BASE, dte_exit=v)))
    for v in [0.03, 0.07, 0.10]: out.append((f"W{int(v*100)}", dict(BASE, wing_pct=v)))
    out.append(("SL150", dict(BASE, sl=1.50)))
    out.append(("SL300", dict(BASE, sl=3.00)))
    out.append(("SLnone", dict(BASE, sl=None)))
    out.append(("TPnone", dict(BASE, tp=None)))
    out.append(("K2.5", dict(BASE, k=0.025)))
    out.append(("K5", dict(BASE, k=0.05)))
    return out

FIELDS = ["config","symbol","expiry","entry_date","exit_date","exit_reason","year",
          "S0","Ks_ce","Ks_pe","Kc","Kp","dte_actual","hold_days",
          "prem_pct","wing_debit_pct","credit_pct","gross_pct","turnover_pct",
          "atm_vol","atm_oi","wing_vol_min"]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def load_chain(conn, symbol):
    ch = pd.read_sql_query(
        "SELECT trade_date,expiry_date,strike,option_type,close,open_interest,contracts "
        "FROM nse_options_bhav WHERE symbol=? AND close>0", conn, params=(symbol,))
    if ch.empty: return None
    ch["td"] = pd.to_datetime(ch["trade_date"]); ch["ed"] = pd.to_datetime(ch["expiry_date"])
    piv = {}
    for exp, g in ch.groupby("ed"):
        piv[exp] = (g.pivot_table(index="td", columns=["strike","option_type"], values="close", aggfunc="last"),
                    g.pivot_table(index="td", columns=["strike","option_type"], values="contracts", aggfunc="last"),
                    g.pivot_table(index="td", columns=["strike","option_type"], values="open_interest", aggfunc="last"))
    return piv

def run_cfg_symbol(piv, close, symbol, label, p):
    rows = []
    for exp in sorted(piv):
        pv_c, pv_v, pv_o = piv[exp]
        tgt = exp - pd.Timedelta(days=p["dte_entry"])
        cands = pv_c.index[pv_c.index <= tgt]
        if len(cands) == 0: continue
        dt = cands.max(); dte = (exp - dt).days
        if not (p["dte_entry"] - 1 <= dte <= p["dte_entry"] + 5): continue
        lag = p.get("lag", 0)
        if lag:
            loc = pv_c.index.get_loc(dt) + lag
            if loc >= len(pv_c.index): continue
            dt = pv_c.index[loc]; dte = (exp - dt).days
            if dte <= p["dte_exit"] + 3: continue
        if dt not in close.index: continue
        S0 = float(close.loc[dt])
        row_c, row_v = pv_c.loc[dt], pv_v.loc[dt]
        def val(K, ot):
            v = row_c.get((K, ot), np.nan); return float(v) if np.isfinite(v) else np.nan
        def traded(K, ot):
            v = row_v.get((K, ot), np.nan); return np.isfinite(v) and v > 0
        strikes = np.array(sorted({s for (s, ot) in row_c.index if np.isfinite(row_c.get((s, ot), np.nan))}))
        if len(strikes) < 5: continue
        def pick_short(side):
            t_ = S0 * (1 + p["k"]) if side == "CE" else S0 * (1 - p["k"])
            cand = [s for s in strikes if traded(s, side) and val(s, side) > 0]
            if not cand: return None
            s_ = min(cand, key=lambda s: abs(s - t_))
            return s_ if abs(s_/S0 - (t_/S0)) <= ATM_BAND else None
        Ks_ce, Ks_pe = pick_short("CE"), pick_short("PE")
        if Ks_ce is None or Ks_pe is None or Ks_ce < Ks_pe: continue
        ce0, pe0 = val(Ks_ce,"CE"), val(Ks_pe,"PE")
        if not (ce0 + pe0 > 0): continue
        def pick_wing(side, Kshort):
            t_ = Kshort + p["wing_pct"]*S0 if side == "CE" else Kshort - p["wing_pct"]*S0
            cand = [s for s in strikes
                    if (s > Kshort if side == "CE" else s < Kshort)
                    and traded(s, side) and val(s, side) > 0]
            if not cand: return None
            w = min(cand, key=lambda s: abs(s - t_))
            return w if abs(w - Kshort)/S0 >= WING_MIN_PCT else None
        Kc, Kp = pick_wing("CE", Ks_ce), pick_wing("PE", Ks_pe)
        if Kc is None or Kp is None: continue
        wc0, wp0 = val(Kc,"CE"), val(Kp,"PE")
        prem0, wing0 = ce0 + pe0, wc0 + wp0
        credit0 = prem0 - wing0
        if credit0 <= 0: continue
        atm_vol = float((row_v.get((Ks_ce,"CE"),0) or 0) + (row_v.get((Ks_pe,"PE"),0) or 0))
        atm_oi = float((pv_o.loc[dt].get((Ks_ce,"CE"),0) or 0) + (pv_o.loc[dt].get((Ks_pe,"PE"),0) or 0))
        wing_vol_min = float(min(row_v.get((Kc,"CE"),0) or 0, row_v.get((Kp,"PE"),0) or 0))
        path = pv_c.loc[(pv_c.index > dt) & (pv_c.index <= exp)]
        time_exit_date = exp - pd.Timedelta(days=p["dte_exit"])
        exit_reason = exit_dt = posval_x = None
        sv_x = wv_x = 0.0
        for d_ in path.index:
            r = path.loc[d_]
            ce, pe = r.get((Ks_ce,"CE"), np.nan), r.get((Ks_pe,"PE"), np.nan)
            if not (np.isfinite(ce) and np.isfinite(pe)): continue
            wc, wp = r.get((Kc,"CE"), np.nan), r.get((Kp,"PE"), np.nan)
            wv = float((wc if np.isfinite(wc) else 0.0) + (wp if np.isfinite(wp) else 0.0))
            sv = float(ce + pe); posval = sv - wv
            mtm = credit0 - posval
            if p["tp"] is not None and mtm >= p["tp"]*credit0: exit_reason = "target"
            elif p["sl"] is not None and posval >= p["sl"]*credit0: exit_reason = "stop"
            elif d_ >= time_exit_date: exit_reason = "time"
            if exit_reason:
                exit_dt, posval_x, sv_x, wv_x = d_, posval, sv, wv
                break
        if exit_reason is None:
            S_exp = float(close.loc[exp]) if exp in close.index else S0
            sv_x = max(0.0, S_exp-Ks_ce) + max(0.0, Ks_pe-S_exp)
            wv_x = max(0.0, S_exp-Kc) + max(0.0, Kp-S_exp)
            posval_x = sv_x - wv_x; exit_dt = exp; exit_reason = "expiry"
        gross = credit0 - posval_x
        turnover = prem0 + wing0 + sv_x + wv_x
        rows.append(dict(
            config=label, symbol=symbol, expiry=exp.date().isoformat(),
            entry_date=dt.date().isoformat(), exit_date=exit_dt.date().isoformat(),
            exit_reason=exit_reason, year=dt.year, S0=round(S0,2),
            Ks_ce=Ks_ce, Ks_pe=Ks_pe, Kc=Kc, Kp=Kp, dte_actual=dte,
            hold_days=(exit_dt-dt).days,
            prem_pct=round(prem0/S0,5), wing_debit_pct=round(wing0/S0,5),
            credit_pct=round(credit0/S0,5), gross_pct=round(gross/S0,5),
            turnover_pct=round(turnover/S0,5),
            atm_vol=atm_vol, atm_oi=atm_oi, wing_vol_min=wing_vol_min))
    return rows

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(E.db_path())
    syms = [r[0] for r in conn.execute(
        "SELECT symbol, COUNT(*) c FROM nse_options_bhav "
        "WHERE symbol NOT IN ('NIFTY','BANKNIFTY') GROUP BY symbol HAVING c>500 ORDER BY symbol")]
    done = set()
    if OUT_CSV.exists():
        d = pd.read_csv(OUT_CSV, usecols=["config","symbol"]).drop_duplicates()
        done = set(zip(d["config"], d["symbol"]))
        log(f"resume: {len(done)} (config,symbol) pairs done")
    write_header = not OUT_CSV.exists()
    CFGS = cfgs()
    log(f"{len(syms)} symbols x {len(CFGS)} configs")
    for si, s in enumerate(syms):
        todo = [(l, p) for (l, p) in CFGS if (l, s) not in done]
        if not todo: continue
        t0 = time.time()
        piv = load_chain(conn, s)
        spot = E.load_daily(s, conn)
        if piv is None or spot.empty:
            log(f"  {s}: no data"); continue
        close = spot["close"]
        n = 0
        with open(OUT_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header: w.writeheader(); write_header = False
            for label, p in todo:
                try:
                    rows = run_cfg_symbol(piv, close, s, label, p)
                except Exception as ex:
                    log(f"  {s}/{label}: ERROR {ex}"); rows = []
                for r_ in rows: w.writerow(r_)
                n += len(rows)
        log(f"  [{si+1}/{len(syms)}] {s:12s}: {n:5d} trades over {len(todo)} cfgs ({time.time()-t0:.0f}s)")
    log(f"DONE -> {OUT_CSV}")

if __name__ == "__main__":
    import logging; logging.disable(logging.WARNING)
    main()
