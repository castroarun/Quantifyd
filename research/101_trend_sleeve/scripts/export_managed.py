# -*- coding: utf-8 -*-
"""4 books for the tearsheet (10-lot DTE-28 monthly straddle, 2016-2026):
 1) Straddle alone (hold DTE-1)          2) + BankNifty cushion (x8)
 3) Managed·VIX15 + cushion  (VIX>=15 gate + 40% PT + premium-2.5x stop + DTE-5, then cushion)
 4) Managed·noVIX + cushion  (same management but NO VIX gate — trades every month)
Managed exits via daily-close path. Blotter = the VIX>=15 managed trades + BankNifty legs."""
import sqlite3, datetime as dt, json
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP, QTY, CW, MARGIN = 160, 0.003, 500, 8, 2000000
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
NF = {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 AND date>='2015-06-01'")}
BN = {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol='BANKNIFTY' AND timeframe='day' AND close>0 AND date>='2015-06-01'")}
VIX = {d: cl for d, cl in c.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day' AND close>0")}
def mexp(sym):
    ex = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol=? AND expiry_date>='2016-01-01'", (sym,))})
    bm = defaultdict(list)
    for e in ex: bm[e[:7]].append(e)
    return sorted(max(v) for v in bm.values())
tset = {r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>='2016-01-01'")}; tdays = sorted(tset)
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def openleg(sym, d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol=? AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (sym, d, E, K, ot)).fetchone()
    return (r[0] if r and r[0] else (r[1] if r else None), r[2] or 0 if r else 0) if r else None
def cser(E, K, ot, d):
    return {td: cl for td, cl in c.execute("SELECT trade_date,close FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND strike=? AND option_type=? AND trade_date>=? AND trade_date<=? AND close>0", (E, K, ot, d, E))}
alone = {}; mvix = {}; mnovix = {}; blot = []
for E in mexp("NIFTY"):
    tgt = dt.date.fromisoformat(E) - dt.timedelta(days=28); d = None
    for k in range(6):
        cd = (tgt - dt.timedelta(days=k)).isoformat()
        if cd in tset and cd in NF: d = cd; break
    if not d or dte(E, d) < 1: continue
    K = round(NF[d][0]/50)*50
    ce = openleg("NIFTY", d, E, K, "CE"); pe = openleg("NIFTY", d, E, K, "PE")
    if not ce or not pe or ce[1] < 50 or pe[1] < 50: continue
    credit = ce[0] + pe[0]; ecl = cser(E, K, "CE", d); pcl = cser(E, K, "PE", d)
    hold = [x for x in tdays if d <= x <= E and dte(E, x) >= 1]; xd = hold[-1] if hold else d
    if xd not in ecl or xd not in pcl: continue
    alone[E[:7]] = round((credit-(ecl[xd]+pcl[xd]))*QTY - COST*10 - SLIP*(credit+ecl[xd]+pcl[xd])*QTY)
    exd, emark, ecep, epep, reason = xd, ecl[xd]+pcl[xd], ecl[xd], pcl[xd], "DTE-1"
    for x in hold:
        if x not in ecl or x not in pcl: continue
        cev, pev = ecl[x], pcl[x]; mark = cev+pev
        if mark <= 0.6*credit: reason, exd, emark, ecep, epep = "PT 40%", x, mark, cev, pev; break
        if mark >= 2.5*credit: reason, exd, emark, ecep, epep = "prem 2.5x", x, mark, cev, pev; break
        if dte(E, x) <= 5: reason, exd, emark, ecep, epep = "DTE-5", x, mark, cev, pev; break
    mpl = round((credit-emark)*QTY - COST*10 - SLIP*(credit+emark)*QTY)
    mnovix[E[:7]] = mpl
    vx = VIX.get(d); gated = bool(vx and vx >= 15)
    mvix[E[:7]] = mpl if gated else 0
    if gated:
        blot.append({"m": E[:7], "entry": d, "exit": exd, "reason": reason, "vix": round(vx, 1), "K": int(K),
                     "ceIn": round(ce[0], 1), "ceOut": round(ecep, 1), "cePL": round((ce[0]-ecep)*QTY),
                     "peIn": round(pe[0], 1), "peOut": round(epep, 1), "pePL": round((pe[0]-epep)*QTY), "total": mpl})
# BankNifty cushion x8
bnf = defaultdict(float); bnflegs = {}
for E in mexp("BANKNIFTY"):
    if E not in BN: continue
    d = None
    for k in range(8):
        cd = (dt.date.fromisoformat(E)-dt.timedelta(days=28+k)).isoformat()
        if cd in tset and cd in BN: d = cd; break
    if not d: continue
    atm = round(BN[d][0]/100)*100; Kc = round(atm*1.01/100)*100; Kp = round(atm*0.99/100)*100
    ce = openleg("BANKNIFTY", d, E, Kc, "CE"); pe = openleg("BANKNIFTY", d, E, Kp, "PE")
    if not ce or not pe or ce[1] < 20 or pe[1] < 20: continue
    ceOut = max(BN[E][1]-Kc, 0); peOut = max(Kp-BN[E][1], 0)
    bnf[E[:7]] = ((ceOut+peOut)-(ce[0]+pe[0]) - SLIP*(ce[0]+pe[0]))*15*CW
    bnflegs[E[:7]] = {"Kc": int(Kc), "Kp": int(Kp), "ceIn": round(ce[0], 1), "ceOut": round(ceOut, 1), "cePL": round((ceOut-ce[0])*15*CW), "peIn": round(pe[0], 1), "peOut": round(peOut, 1), "pePL": round((peOut-pe[0])*15*CW)}
mm = sorted(alone)
def cum(fn):
    c0 = 0; out = []
    for m in mm: c0 += fn(m); out.append([m, round(c0)])
    return out
V = {"Straddle alone": lambda m: alone[m],
     "+ BankNifty cushion": lambda m: alone[m]+bnf.get(m, 0),
     "Managed·VIX15 + cushion": lambda m: mvix[m]+(bnf.get(m, 0) if mvix[m] != 0 else 0),
     "Managed·noVIX + cushion": lambda m: mnovix[m]+bnf.get(m, 0)}
kpi = {}
for k, fn in V.items():
    pl = [fn(m) for m in mm]; tot = sum(pl); cum2 = pk = mdd = 0
    for x in pl: cum2 += x; pk = max(pk, cum2); mdd = min(mdd, cum2-pk)
    tr = [x for x in pl if x != 0]; nn = len(tr) or 1
    kpi[k] = {"total": round(tot), "cagr": round(100*(tot/(len(mm)/12))/MARGIN, 1),
              "calmar": round((tot/(len(mm)/12))/abs(mdd), 2) if mdd else 0, "mdd": round(mdd),
              "win": round(100*sum(1 for x in tr if x > 0)/nn), "trades": nn}
for t in blot:
    bl = bnflegs.get(t["m"]); t["bnf"] = bl; t["grptotal"] = t["total"]+round(bnf.get(t["m"], 0)) if bl else t["total"]
# ---- liquid-yield version: 6%/yr; ~Rs7.5L liquid-pledged while trading, Rs10L cash->liquid when idle ----
TRADE_INT, IDLE_INT = round(0.06/12*750000), round(0.06/12*1000000)   # 3750 / 5000 per month
def traded(k, m): return (mvix[m] != 0) if k == "Managed·VIX15 + cushion" else True
def cumY(fn, k):
    c0 = 0; out = []
    for m in mm: c0 += fn(m) + (TRADE_INT if traded(k, m) else IDLE_INT); out.append([m, round(c0)])
    return out
curvesY = {k: cumY(fn, k) for k, fn in V.items()}
kpiY = {}
for k, fn in V.items():
    pl = [fn(m) + (TRADE_INT if traded(k, m) else IDLE_INT) for m in mm]; tot = sum(pl); cum2 = pk = mdd = 0
    for x in pl: cum2 += x; pk = max(pk, cum2); mdd = min(mdd, cum2-pk)
    kpiY[k] = {"total": round(tot), "cagr": round(100*(tot/(len(mm)/12))/MARGIN, 1),
               "calmar": round((tot/(len(mm)/12))/abs(mdd), 2) if mdd else 0, "mdd": round(mdd),
               "win": kpi[k]["win"], "trades": kpi[k]["trades"]}
json.dump({"months": mm, "curves": {k: cum(fn) for k, fn in V.items()}, "kpi": kpi,
           "curvesY": curvesY, "kpiY": kpiY, "margin": MARGIN, "trades": blot,
           "yieldnote": f"+6% liquid: Rs{TRADE_INT}/mo trading, Rs{IDLE_INT}/mo idle"}, open("/tmp/managed_curves.json", "w"))
print("with-yield KPIs:")
for k in V: print(f"  {k}: base CAGR {kpi[k]['cagr']}% -> +yield {kpiY[k]['cagr']}% | Calmar {kpi[k]['calmar']}->{kpiY[k]['calmar']}")
# managed ALONE (no cushion) vs +cushion, same calendar basis
print("\ncushion vs no-cushion on the MANAGED book (same calendar basis):")
for lbl, fn in [("Managed alone (NO cushion)", lambda m: mvix[m]), ("Managed + BankNifty cushion", lambda m: mvix[m]+(bnf.get(m, 0) if mvix[m] != 0 else 0))]:
    pl = [fn(m) for m in mm]; tot = sum(pl); a=p=md=0
    for x in pl: a+=x; p=max(p, a); md=min(md, a-p)
    ply = [fn(m)+(TRADE_INT if traded("Managed·VIX15 + cushion", m) else IDLE_INT) for m in mm]; toty = sum(ply); b=q=m2=0
    for x in ply: b+=x; q=max(q, b); m2=min(m2, b-q)
    yr = len(mm)/12
    print(f"  {lbl:>28}: total ₹{tot:+,} Calmar {(tot/yr)/abs(md):.2f} DD ₹{md:+,} | +yield Calmar {(toty/yr)/abs(m2):.2f} DD ₹{m2:+,}")
print("exported", len(mm), "months, 4 books,", len(blot), "managed(VIX15) trades")
for k in V: print(f"  {k}: CAGR {kpi[k]['cagr']}% total ₹{kpi[k]['total']:+,} Calmar {kpi[k]['calmar']} maxDD ₹{kpi[k]['mdd']:+,} win {kpi[k]['win']}% (n={kpi[k]['trades']})")
c.close()
