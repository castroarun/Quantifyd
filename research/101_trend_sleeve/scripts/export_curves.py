# -*- coding: utf-8 -*-
"""Export monthly cumulative P&L + drawdown series for the cushion tearsheet: 10-lot DTE-28 NIFTY
straddle alone vs +BankNifty cushion (w3, w8) vs +SBI cushion (w5), 2016-2026. All at 10 lots."""
import sqlite3, datetime as dt, json
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP, NSTRAD_QTY = 160, 0.003, 500
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
def spot(sym): return {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0 AND date>='2015-06-01'", (sym,))}
NF, BN, SB = spot("NIFTY50"), spot("BANKNIFTY"), spot("SBIN")
def mexp(sym, frm="2016-01-01"):
    ex = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol=? AND expiry_date>=?", (sym, frm))})
    bm = defaultdict(list)
    for e in ex: bm[e[:7]].append(e)
    return sorted(max(v) for v in bm.values())
tdays = sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2016-01-01'")}); tset = set(tdays)
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def leg(sym, d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol=? AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (sym, d, E, K, ot)).fetchone()
    return (r[0] if r and r[0] else (r[1] if r else None), r[1] if r else None, r[2] or 0 if r else 0) if r else None
strad = {}; trades = []
for E in mexp("NIFTY"):
    tgt = dt.date.fromisoformat(E) - dt.timedelta(days=28); d = None
    for k in range(6):
        cd = (tgt - dt.timedelta(days=k)).isoformat()
        if cd in tset and cd in NF: d = cd; break
    if not d or dte(E, d) < 1: continue
    K = round(NF[d][0]/50)*50
    ce = leg("NIFTY", d, E, K, "CE"); pe = leg("NIFTY", d, E, K, "PE")
    if not ce or not pe or ce[2] < 50 or pe[2] < 50: continue
    hold = [x for x in tdays if d <= x <= E and dte(E, x) >= 1]; xd = hold[-1] if hold else d
    cf = leg("NIFTY", xd, E, K, "CE"); pf = leg("NIFTY", xd, E, K, "PE")
    if not cf or not pf: continue
    tot = round((ce[0]+pe[0]-(cf[1]+pf[1]))*NSTRAD_QTY - COST*10 - SLIP*(ce[0]+pe[0]+cf[1]+pf[1])*NSTRAD_QTY)
    strad[E[:7]] = tot
    trades.append({"m": E[:7], "entry": d, "exit": xd, "expiry": E, "K": int(K),
                   "ceIn": round(ce[0], 1), "ceOut": round(cf[1], 1), "cePL": round((ce[0]-cf[1])*NSTRAD_QTY),
                   "peIn": round(pe[0], 1), "peOut": round(pf[1], 1), "pePL": round((pe[0]-pf[1])*NSTRAD_QTY),
                   "uEntry": round(NF[d][1]), "uExit": round(NF[xd][1]) if xd in NF else round(NF[d][1]), "total": tot})
def strangle(sym, sp, step, lot, pct):
    out = defaultdict(float); legs = {}
    for E in mexp(sym):
        if E not in sp: continue
        d = None
        for k in range(8):
            cd = (dt.date.fromisoformat(E)-dt.timedelta(days=28+k)).isoformat()
            if cd in tset and cd in sp: d = cd; break
        if not d: continue
        atm = round(sp[d][0]/step)*step; Kc = round(atm*(1+pct/100)/step)*step; Kp = round(atm*(1-pct/100)/step)*step
        ce = leg(sym, d, E, Kc, "CE"); pe = leg(sym, d, E, Kp, "PE")
        if not ce or not pe or ce[2] < 20 or pe[2] < 20: continue
        ceOut = max(sp[E][1]-Kc, 0); peOut = max(Kp-sp[E][1], 0)
        out[E[:7]] = (ceOut + peOut - (ce[0]+pe[0]) - SLIP*(ce[0]+pe[0]))*lot
        legs[E[:7]] = {"d": d, "Kc": int(Kc), "Kp": int(Kp), "ceIn": round(ce[0], 1), "ceOut": round(ceOut, 1),
                       "peIn": round(pe[0], 1), "peOut": round(peOut, 1), "lot": lot}
    return out, legs
BNFc, BNFlegs = strangle("BANKNIFTY", BN, 100, 15, 1); SBIc, _ = strangle("SBIN", SB, 10, 750, 5)
CW = 8  # BankNifty cushion weight for the blotter (the featured ×8 book)
for t in trades:
    bl = BNFlegs.get(t["m"])
    if bl:
        t["bnf"] = {"Kc": bl["Kc"], "Kp": bl["Kp"], "entry": bl["d"],
                    "ceIn": bl["ceIn"], "ceOut": bl["ceOut"], "cePL": round((bl["ceOut"]-bl["ceIn"])*bl["lot"]*CW),
                    "peIn": bl["peIn"], "peOut": bl["peOut"], "pePL": round((bl["peOut"]-bl["peIn"])*bl["lot"]*CW)}
        t["grptotal"] = t["total"] + round(BNFc.get(t["m"], 0)*CW)
    else:
        t["bnf"] = None; t["grptotal"] = t["total"]
mm = sorted(strad)
def series(fn):
    cum = 0; out = []
    for m in mm: cum += fn(m); out.append([m, round(cum)])
    return out
variants = {
    "Straddle alone": lambda m: strad[m],
    "+ BankNifty cushion (×3)": lambda m: strad[m] + 3*BNFc.get(m, 0),
    "+ BankNifty cushion (×8, best)": lambda m: strad[m] + 8*BNFc.get(m, 0),
    "+ SBI cushion (×5, best)": lambda m: strad[m] + 5*SBIc.get(m, 0),
}
data = {"months": mm, "curves": {k: series(v) for k, v in variants.items()}}
# KPIs
MARGIN = 2000000  # Rs20L working capital for 10-lot NIFTY short straddle
kpi = {}
for k, v in variants.items():
    pl = [v(m) for m in mm]; tot = sum(pl); cum=pk=mdd=0
    for x in pl: cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    yrs = len(mm)/12
    kpi[k] = {"total": tot, "annual": round(tot/yrs), "cagr": round(100*(tot/yrs)/MARGIN, 1),
              "calmar": round((tot/yrs)/abs(mdd), 2) if mdd else 0, "mdd": round(mdd),
              "win": round(100*sum(1 for x in pl if x>0)/len(pl))}
data["kpi"] = kpi; data["margin"] = MARGIN; data["trades"] = trades
json.dump(data, open("/tmp/cushion_curves.json", "w"))
print("exported", len(mm), "months,", len(variants), "variants -> /tmp/cushion_curves.json")
for k in variants: print(f"  {k}: total ₹{kpi[k]['total']:+,} CAGR {kpi[k]['cagr']}% Calmar {kpi[k]['calmar']} maxDD ₹{kpi[k]['mdd']:+,}")
c.close()
