#!/usr/bin/env python3
"""
research/128 G1 — stock ±2.5% short strangle (45→21 DTE, TP50, no stop) with
INDEX wings (NIFTY; BANKNIFTY for banks) notional-matched, vs NAKED control.
Same entries/liquidity gate as r/127 C1. Output: results/g1_trades.csv
"""
import sqlite3, time, csv, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "127_stock_neutral_wings" / "scripts"))
import run_phase_b as B          # engine helpers (load_chain, E, constants)

RESULTS = HERE.parent / "results"
OUT = RESULTS / "g1_trades.csv"
DTE_IN, DTE_OUT, TP, K_OFF = 45, 21, 0.50, 0.025
ATM_BAND, ATM_VOL_MIN = 0.06, 100
SLIP_STOCK, SLIP_IDX = 0.005, 0.0025
BANKS = {"AXISBANK","BANKBARODA","FEDERALBNK","HDFCBANK","ICICIBANK",
         "IDFCFIRSTB","INDUSINDBK","KOTAKBANK","PNB","SBIN"}
CFGS = [("NAKED", None), ("NW3", 0.03), ("NW5", 0.05), ("NW7", 0.07)]
FIELDS = ["config","symbol","expiry","entry_date","exit_date","exit_reason","year",
          "S0","Ks_ce","Ks_pe","idx","idx_spot","wce","wpe","units_ratio",
          "prem_pct","wing_debit_pct","credit_pct","gross_pct","stock_turn_pct",
          "idx_turn_pct","net_pct","atm_vol","hold_days"]

def log(m): print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

IDX_CACHE = {}
def idx_pivots(conn, idx, expiry):
    key = (idx, expiry)
    if key not in IDX_CACHE:
        g = pd.read_sql_query(
            "SELECT trade_date,strike,option_type,close,contracts FROM nse_options_bhav "
            "WHERE symbol=? AND expiry_date=? AND close>0", conn, params=(idx, expiry))
        if g.empty:
            IDX_CACHE[key] = None
        else:
            g["td"] = pd.to_datetime(g["trade_date"])
            IDX_CACHE[key] = (
                g.pivot_table(index="td", columns=["strike","option_type"], values="close", aggfunc="last"),
                g.pivot_table(index="td", columns=["strike","option_type"], values="contracts", aggfunc="last"))
    return IDX_CACHE[key]

IDX_SPOT = {}
def idx_spot(conn, idx, dt):
    if idx not in IDX_SPOT:
        und = "NIFTY50" if idx == "NIFTY" else "BANKNIFTY"
        d = B.E.load_daily(und, conn)
        IDX_SPOT[idx] = d["close"] if not d.empty else None
    s = IDX_SPOT[idx]
    return float(s.loc[dt]) if s is not None and dt in s.index else None

def run_symbol(conn, sym, close):
    idx = "BANKNIFTY" if sym in BANKS else "NIFTY"
    piv = B.load_chain(conn, sym)
    if piv is None:
        return []
    rows = []
    for exp in sorted(piv):
        pv_c, pv_v, pv_o = piv[exp]
        tgt = exp - pd.Timedelta(days=DTE_IN)
        cands = pv_c.index[pv_c.index <= tgt]
        if len(cands) == 0:
            continue
        dt = cands.max(); dte = (exp - dt).days
        if not (DTE_IN - 1 <= dte <= DTE_IN + 5) or dt not in close.index:
            continue
        S0 = float(close.loc[dt])
        row_c, row_v = pv_c.loc[dt], pv_v.loc[dt]
        def val(K, ot):
            v = row_c.get((K, ot), np.nan); return float(v) if np.isfinite(v) else np.nan
        def traded(K, ot):
            v = row_v.get((K, ot), np.nan); return np.isfinite(v) and v > 0
        strikes = np.array(sorted({s for (s, ot) in row_c.index
                                   if np.isfinite(row_c.get((s, ot), np.nan))}))
        if len(strikes) < 5:
            continue
        def pick(side, t_):
            cand = [s for s in strikes if traded(s, side) and val(s, side) > 0]
            if not cand: return None
            s_ = min(cand, key=lambda x: abs(x - t_))
            return s_ if abs(s_ - t_) / S0 <= ATM_BAND else None
        kce = pick("CE", S0 * (1 + K_OFF)); kpe = pick("PE", S0 * (1 - K_OFF))
        if kce is None or kpe is None or kce < kpe:
            continue
        sce0, spe0 = val(kce, "CE"), val(kpe, "PE")
        atm_vol = float((row_v.get((kce, "CE"), 0) or 0) + (row_v.get((kpe, "PE"), 0) or 0))
        if atm_vol < ATM_VOL_MIN or sce0 + spe0 <= 0:
            continue
        exp_s = exp.date().isoformat()
        ip = idx_pivots(conn, idx, exp_s)
        Sn = idx_spot(conn, idx, dt)
        if ip is None or Sn is None or dt not in ip[0].index:
            continue
        ipc, ipv = ip
        irow_c, irow_v = ipc.loc[dt], ipv.loc[dt]
        istrikes = np.array(sorted({s for (s, ot) in irow_c.index
                                    if np.isfinite(irow_c.get((s, ot), np.nan))}))
        units = S0 / Sn                       # per stock-share: index units for notional match
        prem0 = sce0 + spe0
        xcal = exp - pd.Timedelta(days=DTE_OUT)
        path = pv_c.loc[(pv_c.index > dt) & (pv_c.index <= exp)]
        for label, w in CFGS:
            wce = wpe = None; iwc0 = iwp0 = 0.0
            if w is not None:
                def ipick(side, t_):
                    cand = [s for s in istrikes
                            if (irow_v.get((s, side), 0) or 0) > 0
                            and np.isfinite(irow_c.get((s, side), np.nan))
                            and irow_c.get((s, side), 0) > 0]
                    return min(cand, key=lambda x: abs(x - t_)) if cand else None
                wce = ipick("CE", Sn * (1 + w)); wpe = ipick("PE", Sn * (1 - w))
                if wce is None or wpe is None:
                    continue
                iwc0 = float(irow_c[(wce, "CE")]); iwp0 = float(irow_c[(wpe, "PE")])
            wing0 = (iwc0 + iwp0) * units     # per stock-share
            credit0 = prem0 - wing0
            if credit0 <= 0:
                continue
            exit_dt = exit_reason = None; sv_x = wv_x = None
            for d_ in path.index:
                r_ = path.loc[d_]
                ce, pe = r_.get((kce, "CE"), np.nan), r_.get((kpe, "PE"), np.nan)
                if not (np.isfinite(ce) and np.isfinite(pe)):
                    continue
                sv = float(ce + pe)
                wv = 0.0
                if w is not None and d_ in ipc.index:
                    ir = ipc.loc[d_]
                    iwc = ir.get((wce, "CE"), np.nan); iwp = ir.get((wpe, "PE"), np.nan)
                    wv = ((iwc if np.isfinite(iwc) else 0.0) +
                          (iwp if np.isfinite(iwp) else 0.0)) * units
                posval = sv - wv
                why = None
                if credit0 - posval >= TP * credit0: why = "target"
                elif d_ >= xcal: why = "time"
                if why:
                    exit_dt, exit_reason, sv_x, wv_x = d_, why, sv, wv
                    break
            if exit_reason is None:
                S_exp = float(close.loc[exp]) if exp in close.index else S0
                sv_x = max(0.0, S_exp - kce) + max(0.0, kpe - S_exp)
                Sn_x = idx_spot(conn, idx, exp) or Sn
                wv_x = 0.0 if w is None else (max(0.0, Sn_x - wce) + max(0.0, wpe - Sn_x)) * units
                exit_dt, exit_reason = exp, "expiry"
            gross = credit0 - (sv_x - wv_x)
            stock_turn = prem0 + sv_x
            idx_turn = wing0 + wv_x
            net = gross - SLIP_STOCK * stock_turn - SLIP_IDX * idx_turn \
                - 0.0010 * prem0 - 0.0005 * (stock_turn + idx_turn)
            rows.append(dict(
                config=label, symbol=sym, expiry=exp_s, entry_date=dt.date().isoformat(),
                exit_date=exit_dt.date().isoformat(), exit_reason=exit_reason, year=dt.year,
                S0=round(S0, 2), Ks_ce=kce, Ks_pe=kpe, idx=idx, idx_spot=round(Sn, 1),
                wce=wce or "", wpe=wpe or "", units_ratio=round(units, 4),
                prem_pct=round(prem0 / S0, 5), wing_debit_pct=round(wing0 / S0, 5),
                credit_pct=round(credit0 / S0, 5), gross_pct=round(gross / S0, 5),
                stock_turn_pct=round(stock_turn / S0, 5), idx_turn_pct=round(idx_turn / S0, 5),
                net_pct=round(net / S0, 5), atm_vol=atm_vol,
                hold_days=(exit_dt - dt).days))
    return rows

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(B.E.db_path())
    syms = [r[0] for r in conn.execute(
        "SELECT symbol, COUNT(*) c FROM nse_options_bhav "
        "WHERE symbol NOT IN ('NIFTY','BANKNIFTY') GROUP BY symbol HAVING c>500 ORDER BY symbol")]
    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, usecols=["symbol"])["symbol"].unique())
        log("resume: %d symbols done" % len(done))
    hdr = not OUT.exists()
    for i, s in enumerate(syms):
        if s in done:
            continue
        t0 = time.time()
        spot = B.E.load_daily(s, conn)
        if spot.empty:
            continue
        try:
            rows = run_symbol(conn, s, spot["close"])
        except Exception as ex:
            log("%s ERROR %s" % (s, ex)); continue
        with open(OUT, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if hdr: w.writeheader(); hdr = False
            for r_ in rows: w.writerow(r_)
        log("[%d/%d] %s: %d rows (%.0fs)" % (i + 1, len(syms), s, len(rows), time.time() - t0))
    log("DONE -> %s" % OUT)

if __name__ == "__main__":
    import logging; logging.disable(logging.WARNING)
    main()
