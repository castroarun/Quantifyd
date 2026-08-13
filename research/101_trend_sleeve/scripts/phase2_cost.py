# -*- coding: utf-8 -*-
"""Cost of the BNF long-strangle cushion: how much does 3× BankNifty long 1% strangle drag the
naked NIFTY straddle's annualized P&L (full period vs ex-COVID)?"""
import sqlite3, json, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=20000")
strad = defaultdict(float)
for t in json.load(open("/tmp/realistic_trades.json"))["dte3"]: strad[t["exit"][:7]] += t["naked"]
def bars(sym, frm): return c.execute("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0 AND date>=? ORDER BY date", (sym, frm)).fetchall()
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
def yrs(mm): return len(mm)/12
for lbl, mm in [("FULL (2018-26)", sorted(set(strad))), ("EX-2020 (no COVID)", [m for m in sorted(set(strad)) if m[:4]!="2020"])]:
    st_tot=sum(strad[m] for m in mm); st_yr=st_tot/yrs(mm)
    cu_tot=sum(3*bnf.get(m,0) for m in mm); cu_yr=cu_tot/yrs(mm)     # 3 lots of BNF strangle
    print(f"{lbl}: {yrs(mm):.1f}y")
    print(f"  NIFTY straddle:        {st_yr:>+12,.0f}/yr  (total {st_tot:>+13,.0f})")
    print(f"  3× BNF cushion (net):  {cu_yr:>+12,.0f}/yr  (total {cu_tot:>+13,.0f})")
    print(f"  → cushion drag on straddle P&L: {cu_yr/st_yr*100:>+.1f}%  (net-of-cushion PL {st_yr+cu_yr:>+,.0f}/yr)\n")
# also 1-lot cushion cost per year, and the max single-year payoff
print("BNF 1% strangle per-year P&L (1 lot):")
yr=defaultdict(float)
for m,v in bnf.items(): yr[m[:4]]+=v
for y in sorted(yr): print(f"   {y}: {yr[y]:>+9,.0f}")
c.close()
