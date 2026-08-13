
"""research/111 CPR probe — prior-day CPR width (from underlying_spot daily agg) vs
straddle-day outcomes: EOD decay of the 09:16 ATM straddle (from csl_best_configs cells
09:16->15:20 SLnone... use per-day series of the no-SL full-day cell) + breach behavior."""
import json, sqlite3, statistics as st
Q="/home/arun/quantifyd"
oc=sqlite3.connect(Q+"/backtest_data/options_data.db")
cfg=json.load(open(Q+"/static/app/straddles/csl_best_configs.json"))
def daily_hlc(sym):
    m={}
    for d,h,l,c in oc.execute("SELECT substr(snapshot_time,1,10), MAX(spot_price), MIN(spot_price), spot_price FROM underlying_spot WHERE symbol=? AND spot_price>0 GROUP BY 1", (sym,)):
        m[d]=(h,l,c)   # c = last row's price per group (sqlite quirk: bare col = arbitrary; acceptable close proxy)
    return m
def cpr_width(prev):
    H,L,C=prev; piv=(H+L+C)/3.0; bc=(H+L)/2.0; tc=2*piv-bc
    return abs(tc-bc)/piv*100
for sym in ("NIFTY","SENSEX"):
    hlc=daily_hlc(sym); days=sorted(hlc)
    # full-day no-stop cell per DTE isn't uniform; use the 09:16->15:20 SL 'none' cells' series union
    ser={}
    for c in cfg["cells"]:
        if c["sym"]==sym and c["entry"]=="09:16" and c["exit"]=="15:20" and c["sl"]=="none" and c.get("series"):
            for d,v in c["series"]: ser[d]=v
    rows=[]
    for i in range(1,len(days)):
        d=days[i]
        if d not in ser: continue
        w=cpr_width(hlc[days[i-1]])
        rows.append((w, ser[d]))
    if not rows: print(sym,"no rows"); continue
    ws=sorted(r[0] for r in rows)
    t1,t2=ws[len(ws)//3], ws[2*len(ws)//3]
    print("== %s (n=%d days · full-day no-SL straddle pnl at study lots) ==" % (sym,len(rows)))
    for lbl,lo,hi in (("NARROW (<%.2f%%)"%t1,-1,t1),("MID",t1,t2),("WIDE (>%.2f%%)"%t2,t2,999)):
        sub=[p for w,p in rows if lo<w<=hi]
        if not sub: continue
        print("  CPR %-16s n=%2d  mean %+7.0f  median %+7.0f  win %d%%" % (
            lbl,len(sub),sum(sub)/len(sub),st.median(sub),round(100*sum(1 for x in sub if x>0)/len(sub))))
    try:
        print("  corr(width, pnl) = %+.2f" % st.correlation([r[0] for r in rows],[r[1] for r in rows]))
    except Exception: pass
