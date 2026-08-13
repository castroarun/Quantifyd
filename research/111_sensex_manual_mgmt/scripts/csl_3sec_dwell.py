
"""research/111 - CSL replay on RAW ~3-sec snaps with the ACCEPTED live mechanic:
breach must persist >=2 consecutive snaps, then MARKET exit at the NEXT snap's price.
Entry 09:16 first snap; else hold to last snap <=15:20. NIFTY + SENSEX, by DTE."""
import json, sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta
DB="/home/arun/quantifyd/backtest_data/options_data.db"
OUT="/home/arun/quantifyd/research/111_sensex_manual_mgmt/results/csl_3sec_dwell.json"
CFG={"NIFTY":{"step":50,"qty":650,"lots":10},"SENSEX":{"step":100,"qty":100,"lots":5}}
SLS=(20,25,30,40,999); COST=160
oc=sqlite3.connect(DB)
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
        exps=sorted({r[0] for r in oc.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=? AND substr(snapshot_time,1,10)=? AND expiry_date>=?",(SYM,day,day))})
        if not exps: continue
        E=exps[0]
        sp=[(r[0][11:19],float(r[1])) for r in oc.execute("SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol=? AND substr(snapshot_time,1,10)=? AND spot_price>0 ORDER BY snapshot_time",(SYM,day))]
        if not sp: continue
        ts=[a for a,_ in sp]; j=bisect_right(ts,"09:16:30")
        if not j: continue
        K=round(sp[j-1][1]/cfg["step"])*cfg["step"]
        legs={}
        for ty in ("CE","PE"):
            legs[ty]=[(r[0][11:19],float(r[1])) for r in oc.execute("SELECT snapshot_time,ltp FROM option_chain WHERE symbol=? AND expiry_date=? AND strike=? AND instrument_type=? AND substr(snapshot_time,1,10)=? AND ltp>0 ORDER BY snapshot_time",(SYM,E,K,ty,day))]
        if not (legs["CE"] and legs["PE"]): continue
        ck=[a for a,_ in legs["CE"]]; pk_=[a for a,_ in legs["PE"]]
        cv=[v for _,v in legs["CE"]]; pv=[v for _,v in legs["PE"]]
        # clock = CE snap times within session; PE looked up at-or-before
        comb=[]
        for idx,t in enumerate(ck):
            if t<"09:16:00" or t>"15:20:00": continue
            jp=bisect_right(pk_,t)
            if not jp: continue
            comb.append((t,cv[idx]+pv[jp-1]))
        if len(comb)<50: continue
        ent=comb[0][1]
        rec={"day":day,"dte":tdte(E,day)}
        for sl in SLS:
            thr=ent*(1+sl/100.0); streak=0; pnl=None
            if sl<900:
                for m in range(1,len(comb)):
                    if comb[m][1]>=thr:
                        streak+=1
                        if streak>=2:                      # dwell confirmed
                            nx=comb[m+1][1] if m+1<len(comb) else comb[m][1]
                            pnl=(ent-nx)*cfg["qty"]-COST   # market exit at NEXT snap
                            break
                    else: streak=0
            if pnl is None: pnl=(ent-comb[-1][1])*cfg["qty"]-COST
            rec[str(sl)]=round(pnl)
        rows.append(rec)
        if i%10==0: print("%s %d/%d"%(SYM,i,len(days)),flush=True)
    out[SYM]={"rows":rows,"qty":cfg["qty"],"lots":cfg["lots"]}
    print("== %s 3-SEC DWELL (qty %d = %d lots) =="%(SYM,cfg["qty"],cfg["lots"]),flush=True)
    for k in range(5):
        sub=[r for r in rows if r["dte"]==k]
        if not sub: continue
        print("DTE%d n=%d:"%(k,len(sub)),flush=True)
        for sl in SLS:
            a=agg([r[str(sl)] for r in sub])
            print("  SL%-4s tot %+9d mean %+7d win %2d%% dd %+8d ratio %s"%("none" if sl==999 else str(sl)+"%",a["total"],a["mean"],a["win"],a["maxdd"],a["ratio"]),flush=True)
json.dump(out,open(OUT,"w"))
print("DONE")
