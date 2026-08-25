#!/usr/bin/env python3
"""
research/127 — Phase A (G1 probe): stock neutral-phase winged short straddle.

Base config, ONE ruleset for all stocks (no per-stock tuning):
  - monthly expiry cycle, entry at expiry-45cd (rolled back to a session, tol ±5d)
  - SELL ATM CE+PE, BUY wings ~5%-of-spot away (snapped to traded strikes)
  - exits: target 50% of net credit / stop at -100% of net credit
           (structure cost-to-close >= 2x credit) / time at DTE<=21
  - all 4 legs must have contracts>0 and close>0 at entry (r/89 lesson)
  - marks daily (bhav close); missing wing marks valued 0 (pessimistic for us)
  - records GROSS pnl + turnover (cost sweep post-hoc) + entry liquidity +
    causal neutral-phase features (gates analysed post-hoc, no re-runs)

Incremental per-symbol CSV append; resume-safe (skips completed symbols).
Run on VPS: /home/arun/quantifyd/venv/bin/python3 run_phase_a.py
"""
import os, sys, math, sqlite3, time, csv
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "89_short_monthly_straddle" / "scripts"))
import engine as E  # db_path, load_daily, add_features (rv20, rv_rank, gap)

DTE_ENTRY, DTE_TOL = 45, 5          # entry at expiry-45cd, roll back up to 5d
DTE_EXIT = 21                        # time exit when calendar DTE <= 21
TP, SL = 0.50, 2.00                  # on net credit; SL: posval >= SL*credit
WING_PCT = 0.05                      # target wing distance, fraction of spot
WING_MIN_PCT = 0.025                 # reject if snapping brings wing closer than this
ATM_BAND = 0.06                      # ATM strike must be within 6% of spot
RESULTS = HERE.parent / "results"
OUT_CSV = RESULTS / "phase_a_trades.csv"

FIELDS = ["symbol","expiry","entry_date","exit_date","exit_reason","year",
          "S0","K","Kc","Kp","dte_actual","hold_days",
          "prem_pct","wing_debit_pct","credit_pct","be_width_pct",
          "gross_pct","turnover_pct","move_pct","maxmove_pct",
          "atm_vol","atm_oi","wing_vol_min","chain_strikes",
          "rv20","rv_rank","hv30_rank","gapflag",
          "bb_bw","bb_bw_rank","atr_ratio","adx14","chop14","rsi14",
          "trend_dist_atr","cpr_width_pct","nr7","inside_day"]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---------- causal indicator features (all computed on data <= t) ----------
def wilder(s, n): return s.ewm(alpha=1.0/n, adjust=False).mean()

def features(df):
    o,h,l,c = df["open"],df["high"],df["low"],df["close"]
    out = pd.DataFrame(index=df.index)
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr14 = wilder(tr,14)
    out["atr14"] = atr14
    out["atr_ratio"] = atr14 / atr14.rolling(28).mean()
    up, dn = h.diff(), -l.diff()
    plus  = wilder(pd.Series(np.where((up>dn)&(up>0),up,0.0),index=df.index),14)
    minus = wilder(pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=df.index),14)
    atrw = wilder(tr,14)
    pdi, mdi = 100*plus/atrw, 100*minus/atrw
    dx = 100*(pdi-mdi).abs()/(pdi+mdi)
    out["adx14"] = wilder(dx.fillna(0),14)
    out["chop14"] = 100*np.log10(tr.rolling(14).sum()/(h.rolling(14).max()-l.rolling(14).min()))/np.log10(14)
    d = c.diff(); g = wilder(d.clip(lower=0),14); ls = wilder((-d).clip(lower=0),14)
    out["rsi14"] = 100 - 100/(1+g/ls)
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_bw"] = 100*(4*sd20)/sma20
    out["bb_bw_rank"] = out["bb_bw"].rolling(252).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    out["trend_dist_atr"] = (c-sma20).abs()/atr14
    # CPR width from PREVIOUS day HLC (causal for an entry at today's close)
    ph,pl,pc = h.shift(),l.shift(),c.shift()
    piv = (ph+pl+pc)/3; bc = (ph+pl)/2; tc = 2*piv-bc
    out["cpr_width_pct"] = 100*(tc-bc).abs()/pc
    rng = h-l
    out["nr7"] = (rng == rng.rolling(7).min()).astype(int)
    out["inside_day"] = ((h<h.shift())&(l>l.shift())).astype(int)
    lr = np.log(c/c.shift())
    hv30 = lr.rolling(30).std()*math.sqrt(252)
    out["hv30_rank"] = hv30.rolling(252).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    return out

# ---------- per-symbol backtest ----------
def run_symbol(conn, symbol):
    spot = E.load_daily(symbol, conn)
    if spot is None or spot.empty: return []
    feat = E.add_features(spot)          # rv20, rv_rank, gap
    ind = features(spot)
    close = spot["close"]
    ch = pd.read_sql_query(
        "SELECT trade_date,expiry_date,strike,option_type,close,open_interest,contracts "
        "FROM nse_options_bhav WHERE symbol=? AND close>0", conn, params=(symbol,))
    if ch.empty: return []
    ch["td"] = pd.to_datetime(ch["trade_date"]); ch["ed"] = pd.to_datetime(ch["expiry_date"])
    rows = []
    for exp, g in sorted(ch.groupby("ed"), key=lambda kv: kv[0]):
        pv_c = g.pivot_table(index="td", columns=["strike","option_type"], values="close", aggfunc="last")
        pv_v = g.pivot_table(index="td", columns=["strike","option_type"], values="contracts", aggfunc="last")
        pv_o = g.pivot_table(index="td", columns=["strike","option_type"], values="open_interest", aggfunc="last")
        # entry session: last chain session <= expiry-45cd, within tolerance
        tgt = exp - pd.Timedelta(days=DTE_ENTRY)
        cands = pv_c.index[(pv_c.index <= tgt)]
        if len(cands) == 0: continue
        dt = cands.max()
        dte = (exp - dt).days
        if not (DTE_ENTRY - 1 <= dte <= DTE_ENTRY + DTE_TOL): continue
        if dt not in close.index: continue
        S0 = float(close.loc[dt])
        row_c, row_v = pv_c.loc[dt], pv_v.loc[dt]
        def val(K, ot, r=row_c):
            v = r.get((K, ot), np.nan); return float(v) if np.isfinite(v) else np.nan
        def traded(K, ot):
            v = row_v.get((K, ot), np.nan); return np.isfinite(v) and v > 0
        strikes = np.array(sorted({s for (s, ot) in row_c.index if np.isfinite(row_c.get((s, ot), np.nan))}))
        if len(strikes) < 5: continue
        K = float(strikes[np.argmin(np.abs(strikes - S0))])
        if abs(K/S0 - 1) > ATM_BAND: continue
        if not (traded(K,"CE") and traded(K,"PE")): continue
        ce0, pe0 = val(K,"CE"), val(K,"PE")
        if not (np.isfinite(ce0) and np.isfinite(pe0) and ce0+pe0 > 0): continue
        # wings: nearest TRADED strike to K +/- 5% of spot
        def pick_wing(side):
            targetK = K + WING_PCT*S0 if side=="CE" else K - WING_PCT*S0
            cand = [s for s in strikes
                    if (s > K if side=="CE" else s < K)
                    and traded(s, side) and np.isfinite(val(s, side)) and val(s, side) > 0]
            if not cand: return None
            w = min(cand, key=lambda s: abs(s-targetK))
            return w if abs(w-K)/S0 >= WING_MIN_PCT else None
        Kc, Kp = pick_wing("CE"), pick_wing("PE")
        if Kc is None or Kp is None: continue
        wc0, wp0 = val(Kc,"CE"), val(Kp,"PE")
        prem0 = ce0+pe0; wing0 = wc0+wp0
        credit0 = prem0 - wing0
        if credit0 <= 0: continue
        atm_vol = float((row_v.get((K,"CE"),0) or 0) + (row_v.get((K,"PE"),0) or 0))
        atm_oi  = float((pv_o.loc[dt].get((K,"CE"),0) or 0) + (pv_o.loc[dt].get((K,"PE"),0) or 0))
        wing_vol_min = float(min(row_v.get((Kc,"CE"),0) or 0, row_v.get((Kp,"PE"),0) or 0))
        # walk the path
        path = pv_c.loc[(pv_c.index > dt) & (pv_c.index <= exp)]
        time_exit_date = exp - pd.Timedelta(days=DTE_EXIT)
        exit_reason, exit_dt, posval_x, sv_x, wv_x = None, None, None, None, None
        maxmove = 0.0
        for d_ in path.index:
            r = path.loc[d_]
            ce, pe = r.get((K,"CE"),np.nan), r.get((K,"PE"),np.nan)
            if d_ in close.index:
                maxmove = max(maxmove, abs(float(close.loc[d_])/S0 - 1))
            if not (np.isfinite(ce) and np.isfinite(pe)):
                if d_ >= time_exit_date and exit_reason is None:
                    continue    # roll forward to next day with marks
                continue
            wc, wp = r.get((Kc,"CE"),np.nan), r.get((Kp,"PE"),np.nan)
            wv = float((wc if np.isfinite(wc) else 0.0) + (wp if np.isfinite(wp) else 0.0))
            sv = float(ce+pe); posval = sv - wv
            mtm = credit0 - posval
            if mtm >= TP*credit0: exit_reason="target"
            elif posval >= SL*credit0: exit_reason="stop"
            elif d_ >= time_exit_date: exit_reason="time"
            if exit_reason:
                exit_dt, posval_x, sv_x, wv_x = d_, posval, sv, wv
                break
        if exit_reason is None:
            # no marks after time-exit: settle at intrinsic on expiry spot
            S_exp = float(close.loc[exp]) if exp in close.index else S0*(1+np.sign(0)*0)
            sv_x = abs(S_exp-K); wv_x = max(0.0,S_exp-Kc)+max(0.0,Kp-S_exp)
            posval_x = sv_x - wv_x; exit_dt = exp; exit_reason = "expiry"
        gross = credit0 - posval_x
        turnover = prem0 + wing0 + sv_x + wv_x     # sum |premium traded|, for cost sweep
        Sx = float(close.loc[exit_dt]) if exit_dt in close.index else S0
        fi = feat.index.get_indexer([dt])[0]
        rows.append(dict(
            symbol=symbol, expiry=exp.date().isoformat(), entry_date=dt.date().isoformat(),
            exit_date=exit_dt.date().isoformat(), exit_reason=exit_reason, year=dt.year,
            S0=round(S0,2), K=K, Kc=Kc, Kp=Kp, dte_actual=dte,
            hold_days=(exit_dt-dt).days,
            prem_pct=round(prem0/S0,5), wing_debit_pct=round(wing0/S0,5),
            credit_pct=round(credit0/S0,5), be_width_pct=round(credit0/S0,5),
            gross_pct=round(gross/S0,5), turnover_pct=round(turnover/S0,5),
            move_pct=round(Sx/S0-1,5), maxmove_pct=round(maxmove,5),
            atm_vol=atm_vol, atm_oi=atm_oi, wing_vol_min=wing_vol_min,
            chain_strikes=len(strikes),
            rv20=round(float(feat["rv20"].iloc[fi]),4) if np.isfinite(feat["rv20"].iloc[fi]) else "",
            rv_rank=round(float(feat["rv_rank"].iloc[fi]),3) if np.isfinite(feat["rv_rank"].iloc[fi]) else "",
            hv30_rank=round(float(ind["hv30_rank"].loc[dt]),3) if np.isfinite(ind["hv30_rank"].loc[dt]) else "",
            gapflag=int(abs(feat["gap"].iloc[fi])>0.20) if np.isfinite(feat["gap"].iloc[fi]) else "",
            bb_bw=round(float(ind["bb_bw"].loc[dt]),3) if np.isfinite(ind["bb_bw"].loc[dt]) else "",
            bb_bw_rank=round(float(ind["bb_bw_rank"].loc[dt]),3) if np.isfinite(ind["bb_bw_rank"].loc[dt]) else "",
            atr_ratio=round(float(ind["atr_ratio"].loc[dt]),3) if np.isfinite(ind["atr_ratio"].loc[dt]) else "",
            adx14=round(float(ind["adx14"].loc[dt]),2) if np.isfinite(ind["adx14"].loc[dt]) else "",
            chop14=round(float(ind["chop14"].loc[dt]),2) if np.isfinite(ind["chop14"].loc[dt]) else "",
            rsi14=round(float(ind["rsi14"].loc[dt]),2) if np.isfinite(ind["rsi14"].loc[dt]) else "",
            trend_dist_atr=round(float(ind["trend_dist_atr"].loc[dt]),3) if np.isfinite(ind["trend_dist_atr"].loc[dt]) else "",
            cpr_width_pct=round(float(ind["cpr_width_pct"].loc[dt]),3) if np.isfinite(ind["cpr_width_pct"].loc[dt]) else "",
            nr7=int(ind["nr7"].loc[dt]), inside_day=int(ind["inside_day"].loc[dt])))
    return rows

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(E.db_path())
    syms = [r[0] for r in conn.execute(
        "SELECT symbol, COUNT(*) c FROM nse_options_bhav "
        "WHERE symbol NOT IN ('NIFTY','BANKNIFTY') GROUP BY symbol HAVING c>500 ORDER BY symbol")]
    done = set()
    if OUT_CSV.exists():
        done = set(pd.read_csv(OUT_CSV, usecols=["symbol"])["symbol"].unique())
        log(f"resume: {len(done)} symbols already done")
    write_header = not OUT_CSV.exists()
    log(f"stock symbols: {len(syms)}  (skipping {len(done)})")
    total = 0
    for s in syms:
        if s in done: continue
        t0 = time.time()
        try:
            rows = run_symbol(conn, s)
        except Exception as ex:
            log(f"  {s}: ERROR {ex}"); continue
        with open(OUT_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header: w.writeheader(); write_header = False
            for r in rows: w.writerow(r)
        total += len(rows)
        log(f"  {s:12s}: {len(rows):4d} trades  ({time.time()-t0:.0f}s)  running total {total}")
    log(f"DONE -> {OUT_CSV}")

if __name__ == "__main__":
    import logging; logging.disable(logging.WARNING)
    main()
