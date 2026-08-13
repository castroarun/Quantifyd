# -*- coding: utf-8 -*-
"""PHASE 2 · HARDEN — adversarial checks on the 2 survivors.
1) GOLD weekly trend: param sensitivity (plateau vs peak), per-year, walk-forward OOS.
2) BNF long-strangle cushion: is the straddle-Calmar lift COVID-2020 only?
3) 3-sleeve (straddle + BNF cushion + gold trend): gold-straddle corr + combined-Calmar optimization."""
import sqlite3, json, math, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=20000")
def bars(sym, frm="2016-01-01"):
    return c.execute("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0 AND date>=? ORDER BY date", (sym, frm)).fetchall()
def weekly(sym, frm="2016-01-01"):
    wk = {}
    for d, o, h, l, cl in bars(sym, frm):
        y, w, _ = dt.date.fromisoformat(d).isocalendar(); key = (y, w)
        if key not in wk: wk[key] = [d, o, h, l, cl]
        else: wk[key][2] = max(wk[key][2], h); wk[key][3] = min(wk[key][3], l); wk[key][4] = cl; wk[key][0] = d
    return [wk[k] for k in sorted(wk)]
def donch(b, N):
    out = []; pos = 0
    for i in range(N, len(b) - 1):
        hh = max(x[2] for x in b[i-N:i]); ll = min(x[3] for x in b[i-N:i]); cl = b[i][4]
        if cl > hh: pos = 1
        elif cl < ll: pos = -1
        out.append((b[i+1][0], pos * (b[i+1][4] / b[i][4] - 1)))
    return out
def sharpe(r):
    n=len(r); mu=sum(r)/n; sd=(sum((x-mu)**2 for x in r)/n)**0.5; return mu/sd*math.sqrt(52) if sd else 0

GW = weekly("GOLDBEES")
print("== 1) GOLD weekly trend robustness ==")
print("  param sensitivity (Donchian N):", "  ".join(f"N{N}:{sharpe([x[1] for x in donch(GW,N)]):+.2f}" for N in [6,8,10,13,20]))
# per-year
s = donch(GW, 10); yr = defaultdict(list)
for d, r in s: yr[d[:4]].append(r)
print("  per-year return:", " ".join(f"{y}:{sum(yr[y])*100:+.0f}%" for y in sorted(yr)))
# walk-forward: train<=2021 pick best N, test>=2022
train = [x for x in GW if x[0] < "2022-01-01"]; test = [x for x in GW if x[0] >= "2021-06-01"]
bestN = max([6,8,10,13,20], key=lambda N: sharpe([x[1] for x in donch(train, N)]))
oos = [x[1] for x in donch(test, bestN) if x[0] >= "2022-01-01"]
print(f"  walk-forward: train picks N={bestN}; OOS(2022-26) Sharpe {sharpe(oos):+.2f}, ann {sum(oos)/(len(oos)/52)*100:+.1f}%")

# straddle + BNF cushion
strad = defaultdict(float)
for t in json.load(open("/tmp/realistic_trades.json"))["dte3"]: strad[t["exit"][:7]] += t["naked"]
def bnf_strangle():
    sym="BANKNIFTY"; spot={d:(o,cl) for d,o,h,l,cl in bars(sym,"2016-06-01")}
    allexp=sorted({r[0] for r in c.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol=? AND expiry_date>='2016-06-01'",(sym,))})
    bymon=defaultdict(list)
    for e in allexp: bymon[e[:7]].append(e)
    monthly=sorted(max(v) for v in bymon.values()); tset=set(spot); out=defaultdict(float)
    def leg(d,E,K,ot):
        r=c.execute("SELECT open,close,open_interest FROM nse_options_bhav WHERE symbol=? AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?",(sym,d,E,K,ot)).fetchone()
        return (r[0] if r and r[0] else (r[1] if r else None), r[2] if r else 0) if r else None
    for E in monthly:
        if E not in spot: continue
        d=None
        for k in range(8):
            cd=(dt.date.fromisoformat(E)-dt.timedelta(days=28+k)).isoformat()
            if cd in tset: d=cd; break
        if not d: continue
        atm=round(spot[d][0]/100)*100; Kc=round(atm*1.01/100)*100; Kp=round(atm*0.99/100)*100
        ce=leg(d,E,Kc,"CE"); pe=leg(d,E,Kp,"PE")
        if not ce or not pe or ce[1]<50 or pe[1]<50: continue
        cost=ce[0]+pe[0]; sexp=spot[E][1]
        out[E[:7]]+=(max(sexp-Kc,0)+max(Kp-sexp,0)-cost-0.003*cost)*15
    return out
bnf=bnf_strangle()
def calmar(pl):
    cum=pk=mdd=0
    for x in pl: cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    return (sum(pl)/(len(pl)/12))/abs(mdd) if mdd else 0
print("\n== 2) BNF cushion attribution (is the lift COVID-only?) ==")
for lbl, mm in [("full 2018-26", sorted(strad)), ("EX-2020", [m for m in sorted(strad) if m[:4]!='2020'])]:
    c0=calmar([strad[m] for m in mm]); c3=calmar([strad[m]+3*bnf.get(m,0) for m in mm])
    print(f"  {lbl}: straddle-alone {c0:.2f} -> +3 BNF cushion {c3:.2f}")

print("\n== 3) 3-sleeve: straddle + 3×BNF cushion + gold trend ==")
# gold trend monthly ₹ on ₹5L notional
gmon=defaultdict(float)
for d,r in donch(GW,10): gmon[d[:7]]+=r*500000
months=sorted(set(strad)&set(gmon))
xs=[gmon[m] for m in months]; ys=[strad[m] for m in months]; n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
cov=sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/n; sx=(sum((x-mx)**2 for x in xs)/n)**0.5; sy=(sum((y-my)**2 for y in ys)/n)**0.5
print(f"  gold-trend vs straddle corr = {cov/(sx*sy) if sx and sy else 0:+.2f}")
best=(0,0,0)
for g in [0,0.5,1,1.5,2,3]:
    pl=[strad[m]+3*bnf.get(m,0)+g*gmon[m] for m in months]; cal=calmar(pl)
    if cal>best[0]: best=(cal,g)
    print(f"  gold weight {g} (×₹5L): combined Calmar {cal:.2f}")
print(f"  -> BEST 3-sleeve: 3×BNF + {best[1]}×gold → Calmar {best[0]:.2f} (straddle-alone {calmar([strad[m] for m in months]):.2f})")
c.close()
