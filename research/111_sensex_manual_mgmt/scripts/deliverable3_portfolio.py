
"""deliverable 3 — portfolio weight scan over 4 sleeves (all per-6-lot daily streams):
NIFTY-CSL(best book replay), NIFTY-NAS(916x3 live), SENSEX-CSL(replay), SENSEX-NAS(live).
Grid weights 0/0.5/1/1.5/2 units of 6 lots each; rank by return/maxDD; also per-sleeve stats."""
import json, sqlite3, itertools
Q="/home/arun/quantifyd"
LOTS={"2026-03":1,"2026-04":5,"2026-05":1,"2026-06":1,"2026-07":2,"2026-08":3}
cfg=json.load(open(Q+"/static/app/straddles/csl_best_configs.json"))
def csl(sym,scale):
    m={}
    for k,b in cfg["best"][sym].items():
        for d,v in b.get("series",[]): m[d]=m.get(d,0)+v*scale
    return m
def nifty_nas():
    out={}
    for n in ("nas_916_atm","nas_916_atm2","nas_916_atm4"):
        c=sqlite3.connect(Q+"/backtest_data/%s_trading.db"%n); c.row_factory=None
        cols=[d[1] for d in c.execute("PRAGMA table_info(nas_atm_trades)")]
        pcol="pnl" if "pnl" in cols else "net_pnl"; dcol="entry_time" if "entry_time" in cols else "created_at"
        for d,p in c.execute("SELECT %s,%s FROM nas_atm_trades WHERE %s IS NOT NULL"%(dcol,pcol,pcol)):
            d=str(d)[:10]; out[d]=out.get(d,0)+(p/LOTS.get(d[:7],1))*2
    return out
def sensex_nas():
    c=sqlite3.connect(Q+"/backtest_data/sensex_atm2_trading.db")
    out={}
    for d,p,l in c.execute("SELECT trade_date,net_pnl,lots FROM nas_atm_trades WHERE net_pnl IS NOT NULL"):
        d=str(d)[:10]; out[d]=out.get(d,0)+(p/(l or 1))*6
    return out
S4={"N_CSL":csl("NIFTY",0.6),"N_NAS":nifty_nas(),"S_CSL":csl("SENSEX",1.2),"S_NAS":sensex_nas()}
days=sorted(set().union(*[set(v) for v in S4.values()]))
def agg(f):
    c=pk=dd=0
    for v in f: c+=v; pk=max(pk,c); dd=min(dd,c-pk)
    return round(sum(f)), round(dd), (round(sum(f)/abs(dd),1) if dd<0 else 99.0)
print("sleeves (per 1.0 unit = 6 lots):")
for k,v in S4.items():
    t,dd,r=agg([v.get(d,0) for d in days]); print("  %-6s n=%2d tot %+9d dd %+8d ratio %5.1f"%(k,len(v),t,dd,r))
best=[]
W=(0,0.5,1,1.5,2)
for w in itertools.product(W,repeat=4):
    if sum(w)==0: continue
    f=[sum(wi*S4[k].get(d,0) for wi,k in zip(w,S4)) for d in days]
    t,dd,r=agg(f)
    best.append((r,t,dd,w))
best.sort(reverse=True)
print("\nTOP 8 weight mixes (units of 6 lots: N_CSL, N_NAS, S_CSL, S_NAS):")
for r,t,dd,w in best[:8]:
    print("  w=%-18s tot %+9d dd %+8d ratio %5.1f"%(str(w),t,dd,r))
# equal-weight + all-CSL + all-NAS reference rows
for lbl,w in (("equal 1,1,1,1",(1,1,1,1)),("CSL-only 1,0,1,0",(1,0,1,0)),("NAS-only 0,1,0,1",(0,1,0,1))):
    f=[sum(wi*S4[k].get(d,0) for wi,k in zip(w,S4)) for d in days]
    t,dd,r=agg(f); print("  ref %-14s tot %+9d dd %+8d ratio %5.1f"%(lbl,t,dd,r))
json.dump({"days":len(days),"top":[{"w":list(w),"total":t,"dd":dd,"ratio":r} for r,t,dd,w in best[:20]]},
          open(Q+"/research/111_sensex_manual_mgmt/results/deliverable3_portfolio.json","w"))
print("DONE")
