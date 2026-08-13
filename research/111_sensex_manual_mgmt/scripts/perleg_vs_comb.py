
"""research/111 - per-leg 30% SL vs combined 30% SL vs none: NIFTY 09:20 ATM straddle, 1-min."""
import json, sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
DB="/home/arun/quantifyd/backtest_data/options_data.db"; SYM="NIFTY"; STEP=50; QTY=650; COST=160
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
days=[r[0] for r in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10) FROM underlying_spot WHERE symbol=? AND spot_price>0 ORDER BY 1",(SYM,))]
rows=[]
for i,day in enumerate(days):
    exps=sorted({r[0] for r in oc.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=? AND substr(snapshot_time,1,10)=? AND expiry_date>=?",(SYM,day,day))})
    if not exps: continue
    E=exps[0]
    sp=[(r[0][11:16],float(r[1])) for r in oc.execute("SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol=? AND substr(snapshot_time,1,10)=? AND spot_price>0 ORDER BY snapshot_time",(SYM,day))]
    if not sp: continue
    k=[a for a,_ in sp]; j=bisect_right(k,"09:20")
    if not j: continue
    K=round(sp[j-1][1]/STEP)*STEP
    ser={}
    for ty in ("CE","PE"):
        ser[ty]=[(r[0][11:16],float(r[1])) for r in oc.execute("SELECT snapshot_time,ltp FROM option_chain WHERE symbol=? AND expiry_date=? AND strike=? AND instrument_type=? AND substr(snapshot_time,1,10)=? AND ltp>0 ORDER BY snapshot_time",(SYM,E,K,ty,day))]
    if not (ser["CE"] and ser["PE"]): continue
    def at(s,h):
        kk=[a for a,_ in s]; j2=bisect_right(kk,h); return s[j2-1][1] if j2 else None
    path=[]
    for h in GRID:
        c_=at(ser["CE"],h); p_=at(ser["PE"],h)
        if c_ and p_: path.append((h,c_,p_))
    if len(path)<10: continue
    ce0,pe0=path[0][1],path[0][2]; ent=ce0+pe0
    # NONE
    none=(ent-(path[-1][1]+path[-1][2]))*QTY-COST
    # COMB30
    comb=None
    for h,c_,p_ in path:
        if c_+p_>=1.3*ent: comb=(ent-(c_+p_))*QTY-COST; break
    if comb is None: comb=none
    # PERLEG30: each leg stopped individually at 1.3x its entry; other leg runs to close
    pl=0; ce_al=pe_al=True
    for h,c_,p_ in path:
        if ce_al and c_>=1.3*ce0: pl+=(ce0-c_)*QTY; ce_al=False
        if pe_al and p_>=1.3*pe0: pl+=(pe0-p_)*QTY; pe_al=False
        if not (ce_al or pe_al): break
    if ce_al: pl+=(ce0-path[-1][1])*QTY
    if pe_al: pl+=(pe0-path[-1][2])*QTY
    pl-=COST
    rows.append({"day":day,"dte":tdte(E,day),"none":round(none),"comb":round(comb),"perleg":round(pl)})
    if i%10==0: print("%d/%d"%(i,len(days)),flush=True)
def agg(f):
    c=pk=dd=0
    for v in f: c+=v; pk=max(pk,c); dd=min(dd,c-pk)
    n=len(f)
    return dict(total=round(sum(f)),mean=round(sum(f)/n),win=round(100*sum(1 for v in f if v>0)/n),maxdd=round(dd),
                ratio=(round(sum(f)/abs(dd),1) if dd<0 else 99),n=n)
print("\n== NIFTY 09:20 ATM straddle, 1-min, qty 650 = 10 lots, %d days =="%len(rows))
for m in ("perleg","comb","none"):
    a=agg([r[m] for r in rows]); print("%-7s tot %+9d mean %+7d win %2d%% dd %+9d ratio %s"%(m,a["total"],a["mean"],a["win"],a["maxdd"],a["ratio"]))
print("by DTE (perleg | comb | none: total,ratio):")
for kdte in range(5):
    sub=[r for r in rows if r["dte"]==kdte]
    if not sub: continue
    parts=[]
    for m in ("perleg","comb","none"):
        a=agg([r[m] for r in sub]); parts.append("%s %+d r%s"%(m[0].upper(),a["total"],a["ratio"]))
    print("  DTE%d n=%d: %s"%(kdte,len(sub)," | ".join(parts)))
json.dump(rows,open("/home/arun/quantifyd/research/111_sensex_manual_mgmt/results/perleg_vs_comb.json","w"))
def eq(f):
    c=0; o=[]
    for v in f: c+=v; o.append(c)
    return o
def ddc(e):
    pk=0; o=[]
    for v in e: pk=max(pk,v); o.append(v-pk)
    return o
fig,ax=plt.subplots(2,1,figsize=(13,8),sharex=True,gridspec_kw={"height_ratios":[2.2,1]})
X=range(len(rows))
for m,lbl,cl in [("perleg","Per-leg 30% SL (NAS mechanic)","#d62728"),("comb","Combined 30% SL (CSL mechanic)","#2ca02c"),("none","No stop","#7f7f7f")]:
    e=eq([r[m] for r in rows]); ax[0].plot(X,e,color=cl,lw=1.9,label="{} (Rs{:+,})".format(lbl,round(e[-1])))
    ax[1].plot(X,ddc(e),color=cl,lw=1.5); ax[1].fill_between(X,ddc(e),0,color=cl,alpha=0.08)
ax[0].set_title("Per-leg vs Combined 30% SL - same NIFTY 09:20 ATM short straddle, 1-min marks, 10 lots",fontsize=11)
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25); ax[0].axhline(0,color="#888",lw=0.6)
ax[1].set_title("Drawdown (Rs)",fontsize=10); ax[1].grid(alpha=0.25)
tick=[0,len(rows)//2,len(rows)-1]; ax[1].set_xticks(tick); ax[1].set_xticklabels([rows[i]["day"] for i in tick])
plt.tight_layout()
for p in ("static/app/perleg_vs_comb.png","frontend/public/perleg_vs_comb.png"):
    fig.savefig("/home/arun/quantifyd/"+p,dpi=110)
print("DONE + chart perleg_vs_comb.png")
