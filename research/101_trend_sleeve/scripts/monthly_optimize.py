# -*- coding: utf-8 -*-
"""Optimize the CUSHIONED monthly NIFTY straddle (2011-2026, 15y): straddle DTE-entry × BNF cushion
width × cushion weight → best combined Calmar. Then report the winner's full stats + per-year."""
import sqlite3, datetime as dt, json
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
COST, SLIP, QTY = 160, 0.003, 50
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=30000")
NF = {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day' AND close>0")}
BN = {d: (o, cl) for d, o, cl in c.execute("SELECT date,open,close FROM market_data_unified WHERE symbol='BANKNIFTY' AND timeframe='day' AND close>0")}
def mexp(sym):
    ex = sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol=? AND expiry_date>='2011-01-01'", (sym,))})
    bm = defaultdict(list)
    for e in ex: bm[e[:7]].append(e)
    return sorted(max(v) for v in bm.values())
tdays = sorted({r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY'")}); tset = set(tdays)
def dte(E, d): return (dt.date.fromisoformat(E) - dt.date.fromisoformat(d)).days
def leg(sym, d, E, K, ot):
    r = c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol=? AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?", (sym, d, E, K, ot)).fetchone()
    return (r[0] if r and r[0] else (r[1] if r else None), r[1] if r else None, r[2] or 0 if r else 0) if r else None
NEXP = mexp("NIFTY"); BEXP = mexp("BANKNIFTY")
def straddle(N):
    out = {}
    for E in NEXP:
        tgt = dt.date.fromisoformat(E) - dt.timedelta(days=N); d = None
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
        out[E[:7]] = round((ce[0]+pe[0]-(cf[1]+pf[1]))*QTY - COST - SLIP*(ce[0]+pe[0]+cf[1]+pf[1])*QTY)
    return out
def bnf(pct):
    out = defaultdict(float)
    for E in BEXP:
        if E not in BN: continue
        d = None
        for k in range(8):
            cd = (dt.date.fromisoformat(E)-dt.timedelta(days=28+k)).isoformat()
            if cd in tset and cd in BN: d = cd; break
        if not d: continue
        atm = round(BN[d][0]/100)*100; Kc = round(atm*(1+pct/100)/100)*100; Kp = round(atm*(1-pct/100)/100)*100
        ce = leg("BANKNIFTY", d, E, Kc, "CE"); pe = leg("BANKNIFTY", d, E, Kp, "PE")
        if not ce or not pe or ce[2] < 30 or pe[2] < 30: continue
        out[E[:7]] = (max(BN[E][1]-Kc,0)+max(Kp-BN[E][1],0) - (ce[0]+pe[0]) - SLIP*(ce[0]+pe[0]))*15
    return out
def stats(series):
    mm = sorted(series); pl = [series[m] for m in mm]; n = len(pl); tot = sum(pl)
    cum = pk = mdd = 0
    for x in pl: cum += x; pk = max(pk, cum); mdd = min(mdd, cum-pk)
    negy = sum(1 for y in {m[:4] for m in mm} if sum(series[m] for m in mm if m[:4]==y) < 0)
    return dict(n=n, tot=tot, cal=(tot/(n/12))/abs(mdd) if mdd else 0, mdd=mdd,
                win=100*sum(1 for x in pl if x>0)/n, worst=min(pl), negy=negy)
DTES = [1, 3, 5, 7, 10, 15, 20, 28]
STR = {N: straddle(N) for N in DTES}
CUSH = {p: bnf(p) for p in [1, 2, 3]}
print("MONTHLY NIFTY STRADDLE + BankNifty long-vol cushion (2011-2026, 15y)")
print("Sizing: straddle = 50 units (~1 lot); each cushion lot = 15 units. ₹ scale linearly with lots.\n")
print(f"{'DTE':>5} | {'STRADDLE ALONE  Calmar/total/DD/negYr':>42} | {'BEST CUSHIONED  Calmar/total/DD (cfg)':>44}")
grid = []
for N in DTES:
    mm = sorted(STR[N]); su = stats(STR[N])
    best = (su['cal'], 0, 1, su)
    for p in [1, 2, 3]:
        for w in [0, 1, 2, 3]:
            comb = {m: STR[N][m] + w*CUSH[p].get(m, 0) for m in mm}; s = stats(comb)
            grid.append((s['cal'], N, p, w, s))
            if w > 0 and s['cal'] > best[0]: best = (s['cal'], w, p, s)
    bcal, bw, bp, bs = best
    print(f"DTE-{N:<2} | {su['cal']:>6.2f} {su['tot']:>+10,} {su['mdd']:>+9,} {su['negy']:>3}yr      | {bcal:>6.2f} {bs['tot']:>+10,} {bs['mdd']:>+9,}  ({bw}x BNF-{bp}%)")
grid.sort(reverse=True)
cal, N, p, w, s = grid[0]
print(f"\nWINNER: monthly straddle DTE-{N} + {w}x BankNifty {p}% cushion")
best = {m: STR[N][m] + w*CUSH[p].get(m, 0) for m in sorted(STR[N])}
by = defaultdict(int)
for m, v in best.items(): by[m[:4]] += v
print("  per-year:", " ".join(f"{y}:{by[y]//1000:+}k" for y in sorted(by)))
print(f"  vs straddle-alone DTE-{N}: Calmar {stats(STR[N])['cal']:.2f} -> cushioned {cal:.2f}")
json.dump([[m, best[m]] for m in sorted(best)], open("/tmp/cushioned_straddle.json", "w"))
c.close()
