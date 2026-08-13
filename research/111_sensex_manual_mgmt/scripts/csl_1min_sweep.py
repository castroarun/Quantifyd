
"""research/111 - CSL sweep x DTE on 1-MIN marks from the raw chain, NIFTY + SENSEX."""
import json, os, sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta
DB="/home/arun/quantifyd/backtest_data/options_data.db"
OUT="/home/arun/quantifyd/research/111_sensex_manual_mgmt/results/csl_1min_sweep.json"
CFG={"NIFTY":{"step":50,"qty":650,"lots":10},"SENSEX":{"step":100,"qty":100,"lots":5}}
SLS=(15,20,25,30,40,50,999); COST=160
oc=sqlite3.connect(DB)
GRID=[]
t=datetime.strptime("09:20","%H:%M")
while t<=datetime.strptime("15:20","%H:%M"):
    GRID.append(t.strftime("%H:%M")); t+=timedelta(minutes=1)
def tdte(E,day):
    dd=datetime.strptime(day,"%Y-%m-%d").date(); ed=datetime.strptime(E,"%Y-%m-%d").date()
    n=0; cur=dd+timedelta(days=1)
    while cur<=ed:
        if cur.weekday()<5: n+=1
        cur+=timedelta(days=1)
    return n
def agg(f):
    c=pk=dd=0
    for v in f: c+=v; pk=max(pk,c); dd=min(dd,c-pk)
    n=len(f)
    return dict(total=round(sum(f)),mean=round(sum(f)/n),win=round(100*sum(1 for v in f if v>0)/n),
                maxdd=round(dd),n=n,ratio=(round(sum(f)/abs(dd),1) if dd<0 else 99))
out={}
for SYM,cfg in CFG.items():
    days=[r[0] for r in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10) FROM underlying_spot WHERE symbol=? AND spot_price>0 ORDER BY 1",(SYM,))]
    rows=[]
    for i,day in enumerate(days):
        sp=[(r[0][11:16],float(r[1])) for r in oc.execute("SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol=? AND substr(snapshot_time,1,10)=? AND spot_price>0 ORDER BY snapshot_time",(SYM,day))]
        if not sp: continue
        ts=[a for a,_ in sp]
        def spot_at(h):
            j=bisect_right(ts,h); return sp[j-1][1] if j else None
        exps=sorted({r[0] for r in oc.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=? AND substr(snapshot_time,1,10)=? AND expiry_date>=?",(SYM,day,day))})
        if not exps: continue
        E=exps[0]; s0=spot_at("09:20")
        if not s0: continue
        K=round(s0/cfg["step"])*cfg["step"]
        ser={}
        for ty in ("CE","PE"):
            s=[(r[0][11:16],float(r[1])) for r in oc.execute("SELECT snapshot_time,ltp FROM option_chain WHERE symbol=? AND expiry_date=? AND strike=? AND instrument_type=? AND substr(snapshot_time,1,10)=? AND ltp>0 ORDER BY snapshot_time",(SYM,E,K,ty,day))]
            ser[ty]=s
        if not (ser["CE"] and ser["PE"]): continue
        def at(s,h):
            k=[a for a,_ in s]; j=bisect_right(k,h); return s[j-1][1] if j else None
        comb=[]
        for h in GRID:
            c_=at(ser["CE"],h); p_=at(ser["PE"],h)
            if c_ and p_: comb.append((h,c_+p_))
        if len(comb)<10: continue
        ent=comb[0][1]
        rec={"day":day,"dte":tdte(E,day),"ent":ent}
        for sl in SLS:
            thr=(1+sl/100.0)*ent; done=False
            for h,p in comb:
                if sl<900 and p>=thr:
                    rec[str(sl)]=round((ent-p)*cfg["qty"]-COST); done=True; break
            if not done: rec[str(sl)]=round((ent-comb[-1][1])*cfg["qty"]-COST)
        rows.append(rec)
        if i%10==0: print("%s %d/%d"%(SYM,i,len(days)),flush=True)
    out[SYM]={"qty":cfg["qty"],"lots":cfg["lots"],"rows":rows,"by_dte":{}}
    print("== %s (qty %d = %d lots) =="%(SYM,cfg["qty"],cfg["lots"]),flush=True)
    for k in range(5):
        sub=[r for r in rows if r["dte"]==k]
        if not sub: continue
        print("DTE%d n=%d:"%(k,len(sub)),flush=True)
        for sl in SLS:
            a=agg([r[str(sl)] for r in sub])
            out[SYM]["by_dte"].setdefault(k,{})[str(sl)]=a
            print("  SL%-4s tot %+9d mean %+7d win %2d%% dd %+8d ratio %s"%("none" if sl==999 else str(sl)+"%",a["total"],a["mean"],a["win"],a["maxdd"],a["ratio"]),flush=True)
json.dump(out,open(OUT,"w"))
print("DONE ->",OUT)
