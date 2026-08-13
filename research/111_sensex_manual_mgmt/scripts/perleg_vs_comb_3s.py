
"""research/111 - PER-LEG vs COMBINED 30% SL on RAW ~3-sec snaps, entry 09:16, dwell
(breach persists >=2 consecutive snaps -> market exit at NEXT snap). NIFTY, by DTE.
Per-leg: each leg has its own 30% dwell-stop (buy back that leg; other runs to close).
Combined: netted premium 30% dwell-stop closes both. None: hold to last snap <=15:20."""
import json, sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
DB="/home/arun/quantifyd/backtest_data/options_data.db"; SYM="NIFTY"; STEP=50; QTY=650; COST=160
oc=sqlite3.connect(DB)
def tdte(E,day):
    dd=datetime.strptime(day,"%Y-%m-%d").date(); ed=datetime.strptime(E,"%Y-%m-%d").date()
    n=0; cur=dd+timedelta(days=1)
    while cur<=ed:
        if cur.weekday()<5: n+=1
        cur+=timedelta(days=1)
    return n
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
    K=round(sp[j-1][1]/STEP)*STEP
    L={}
    for ty in ("CE","PE"):
        L[ty]=[(r[0][11:19],float(r[1])) for r in oc.execute("SELECT snapshot_time,ltp FROM option_chain WHERE symbol=? AND expiry_date=? AND strike=? AND instrument_type=? AND substr(snapshot_time,1,10)=? AND ltp>0 ORDER BY snapshot_time",(SYM,E,K,ty,day))]
    if not (L["CE"] and L["PE"]): continue
    pkt=[a for a,_ in L["PE"]]; pvv=[v for _,v in L["PE"]]
    path=[]
    for t,cv in L["CE"]:
        if t<"09:16:00" or t>"15:20:00": continue
        jp=bisect_right(pkt,t)
        if jp: path.append((t,cv,pvv[jp-1]))
    if len(path)<50: continue
    ce0,pe0=path[0][1],path[0][2]; ent=ce0+pe0
    # COMBINED with dwell
    thr=1.3*ent; streak=0; comb=None
    for m in range(1,len(path)):
        if path[m][1]+path[m][2]>=thr:
            streak+=1
            if streak>=2:
                nx=path[m+1] if m+1<len(path) else path[m]
                comb=(ent-(nx[1]+nx[2]))*QTY-COST; break
        else: streak=0
    if comb is None: comb=(ent-(path[-1][1]+path[-1][2]))*QTY-COST
    # PER-LEG with dwell per leg
    pl=0.0; st={"CE":0,"PE":0}; alive={"CE":True,"PE":True}; e0={"CE":ce0,"PE":pe0}
    for m in range(1,len(path)):
        for gi,ty in ((1,"CE"),(2,"PE")):
            if not alive[ty]: continue
            if path[m][gi]>=1.3*e0[ty]:
                st[ty]+=1
                if st[ty]>=2:
                    nx=path[m+1] if m+1<len(path) else path[m]
                    pl+=(e0[ty]-nx[gi])*QTY; alive[ty]=False
            else: st[ty]=0
        if not (alive["CE"] or alive["PE"]): break
    if alive["CE"]: pl+=(ce0-path[-1][1])*QTY
    if alive["PE"]: pl+=(pe0-path[-1][2])*QTY
    pl-=COST
    none=(ent-(path[-1][1]+path[-1][2]))*QTY-COST
    rows.append({"day":day,"dte":tdte(E,day),"perleg":round(pl),"comb":round(comb),"none":round(none)})
    if i%10==0: print("%d/%d"%(i,len(days)),flush=True)
def agg(f):
    c=pk=dd=0
    for v in f: c+=v; pk=max(pk,c); dd=min(dd,c-pk)
    n=len(f)
    return dict(total=round(sum(f)),mean=round(sum(f)/n),win=round(100*sum(1 for v in f if v>0)/n),maxdd=round(dd),ratio=(round(sum(f)/abs(dd),1) if dd<0 else 99))
print("\n== NIFTY 09:16 straddle, RAW 3-sec + dwell(2), qty 650 = 10 lots, %d days =="%len(rows))
for m in ("perleg","comb","none"):
    a=agg([r[m] for r in rows]); print("%-7s tot %+9d mean %+7d win %2d%% dd %+9d ratio %s"%(m,a["total"],a["mean"],a["win"],a["maxdd"],a["ratio"]))
print("by DTE:")
for k in range(5):
    sub=[r for r in rows if r["dte"]==k]
    if not sub: continue
    parts=["%s %+d r%s"%(m[0].upper(),agg([r[m] for r in sub])["total"],agg([r[m] for r in sub])["ratio"]) for m in ("perleg","comb","none")]
    print("  DTE%d n=%d: %s"%(k,len(sub)," | ".join(parts)))
json.dump(rows,open("/home/arun/quantifyd/research/111_sensex_manual_mgmt/results/perleg_vs_comb_3s.json","w"))
def eq(f):
    c=0;o=[]
    for v in f: c+=v;o.append(c)
    return o
def ddc(e):
    p=0;o=[]
    for v in e: p=max(p,v);o.append(v-p)
    return o
fig,ax=plt.subplots(2,1,figsize=(13,8),sharex=True,gridspec_kw={"height_ratios":[2.2,1]})
X=range(len(rows))
for m,lbl,cl in [("perleg","Per-leg 30% SL (NAS mechanic)","#d62728"),("comb","Combined 30% SL (CSL mechanic)","#2ca02c"),("none","No stop","#7f7f7f")]:
    e=eq([r[m] for r in rows]); ax[0].plot(X,e,color=cl,lw=1.9,label="{} (Rs{:+,})".format(lbl,round(e[-1])))
    ax[1].plot(X,ddc(e),color=cl,lw=1.5); ax[1].fill_between(X,ddc(e),0,color=cl,alpha=0.08)
ax[0].set_title("Per-leg vs Combined 30% SL - NIFTY 09:16 ATM short straddle, RAW 3-sec + 2-snap dwell, 10 lots",fontsize=11)
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25); ax[0].axhline(0,color="#888",lw=0.6)
ax[1].set_title("Drawdown (Rs)",fontsize=10); ax[1].grid(alpha=0.25)
tick=[0,len(rows)//2,len(rows)-1]; ax[1].set_xticks(tick); ax[1].set_xticklabels([rows[i]["day"] for i in tick])
plt.tight_layout()
for p in ("static/app/perleg_vs_comb.png","frontend/public/perleg_vs_comb.png"):
    fig.savefig("/home/arun/quantifyd/"+p,dpi=110)
print("DONE 3s + chart")
