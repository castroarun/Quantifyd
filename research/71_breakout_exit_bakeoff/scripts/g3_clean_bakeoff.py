"""G3: clean liquidity + entry-fill realism, then re-run top exits.
Changes vs G2:
 - liquidity gate = 20d MEDIAN turnover >= TMED cr (persistent), not mean spike.
 - entry-fill guard: skip if entry (next-open) bar is circuit-locked ((h-l)/o < 1%)
   or gaps > 15% above signal close (unfillable chase).
 - report clean vs old population, named-stock inclusion, edge on clean set."""
import sqlite3, pandas as pd, numpy as np, os, time
BASE="/home/arun/quantifyd/research/71_breakout_exit_bakeoff"; RES=os.path.join(BASE,"results")
DB="/home/arun/quantifyd/backtest_data/market_data.db"
TMED=float(os.environ.get("TMED","5.0")); GAP_MAX=0.15; MAXBARS=120; CATA=0.20
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def macd(c): return ema(c,12)-ema(c,26)
def atr(df,n=22):
    h,l,c=df["high"],df["low"],df["close"]; pc=c.shift()
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def supertrend(df,period,mult):
    a=atr(df,period).values; hl2=((df["high"]+df["low"])/2).values; c=df["close"].values; n=len(c)
    ub=hl2+mult*a; lb=hl2-mult*a; fub=ub.copy(); flb=lb.copy(); dir_=np.ones(n)
    for i in range(1,n):
        fub[i]=ub[i] if (ub[i]<fub[i-1] or c[i-1]>fub[i-1]) else fub[i-1]
        flb[i]=lb[i] if (lb[i]>flb[i-1] or c[i-1]<flb[i-1]) else flb[i-1]
        dir_[i]=1 if c[i]>fub[i-1] else (-1 if c[i]<flb[i-1] else dir_[i-1])
    return dir_<0
CONFIGS=[("ST_10_3","st",dict(p=10,m=3.0)),("DONCH_20","donch",dict(N=20)),("ST_7_3","st",dict(p=7,m=3.0)),
 ("CHAND_atr22_m4","chand",dict(m=4.0)),("EMA_50","ema",dict(N=50)),("TIME_40","time",dict(N=40)),
 ("FIX_sl15_noTgt","fix",dict(sl=.15,tgt=None)),("FIX_sl5_tgt10","fix",dict(sl=.05,tgt=.10))]
def fta(b,s):
    sub=b[s:]; w=np.argmax(sub)
    return s+w if sub.size and sub[w] else -1
con=sqlite3.connect(DB)
syms=[r[0] for r in con.execute("select distinct symbol from market_data_unified where timeframe='day'")]
NAMED={"KAMOPAINTS","OSIAHYPER","SUPRIYA","MAZDOCK","MARINE","IRFC"}; named_hits={}
res={n:{"ret":[],"hold":[],"year":[]} for n,_,_ in CONFIGS}
old_ct=clean_ct=fill_skip=0
for si,sym in enumerate(syms):
    df=pd.read_sql("select date,open,high,low,close,volume from market_data_unified where symbol=? and timeframe='day' order by date",con,params=(sym,),parse_dates=["date"]).set_index("date")
    if len(df)<300: continue
    c=df["close"]; o=df["open"].values; hi=df["high"].values; lo=df["low"].values; cl=c.values; yrs=df.index.year.values
    md=macd(c); mw=macd(c.resample("W-FRI").last()).reindex(df.index,method="ffill"); mm=macd(c.resample("ME").last()).reindex(df.index,method="ffill")
    volspike=df["volume"]/df["volume"].rolling(20).mean(); hi252=c.rolling(252,min_periods=200).max()
    tv=c*df["volume"]; turn_mean=tv.rolling(20).mean()/1e7; turn_med=tv.rolling(20).median()/1e7
    base=(md>0)&(mw>0)&(mm>0)&(volspike>=2.0)&(c>=0.98*hi252)&(c>=20)
    old=(base&(turn_mean>=1.0)).values
    clean=(base&(turn_med>=TMED)).values
    old_ct+=int(old.sum())
    A=atr(df,22).values
    st={pm:supertrend(df,*pm) for pm in [(10,3.0),(7,3.0)]}
    donch={N:(cl<pd.Series(lo,index=df.index).rolling(N).min().shift(1).values) for N in [20]}
    emax={N:(cl<ema(c,N).values) for N in [50]}
    n=len(df)
    qpos=np.where(clean)[0]; qpos=qpos[qpos+1<n]
    for i in qpos:
        e=i+1; ep=o[e]
        if ep<=0 or np.isnan(ep): continue
        # entry-fill guard
        if (hi[e]-lo[e])/ep < 0.01 or (ep/cl[i]-1) > GAP_MAX:
            fill_skip+=1; continue
        clean_ct+=1
        if sym in NAMED: named_hits.setdefault(sym,[]).append(str(df.index[i].date()))
        cap=ep*(1-CATA); end=min(n-1,e+MAXBARS)
        for name,kind,p in CONFIGS:
            if kind=="fix":
                sl=ep*(1-p["sl"]); tg=ep*(1+p["tgt"]) if p["tgt"] else None; xi=None;xp=None
                for j in range(e,end+1):
                    if lo[j]<=sl: xi=j; xp=min(o[j],sl); break
                    if tg and hi[j]>=tg: xi=j; xp=max(o[j],tg); break
                if xi is None: xi=end; xp=cl[end]
            elif kind=="time":
                j=min(e+p["N"],n-1); hit=fta(lo<=cap,e)
                if 0<=hit<=j: xi,xp=hit,min(o[hit],cap)
                else: xi,xp=j,(o[j] if j<n else cl[-1])
            else:
                if kind=="chand":
                    hh=np.maximum.accumulate(hi[e:end+1]); trig=cl[e:end+1]<(hh-p["m"]*A[e:end+1])
                    w=fta(trig,0); tj=e+w if w>=0 else -1
                elif kind=="st": tj=fta(st[(p["p"],p["m"])],e); tj=tj if 0<=tj<=end else -1
                elif kind=="donch": tj=fta(donch[p["N"]],e); tj=tj if 0<=tj<=end else -1
                elif kind=="ema": tj=fta(emax[p["N"]],e); tj=tj if 0<=tj<=end else -1
                catj=fta(lo<=cap,e); catj=catj if 0<=catj<=end else -1
                cands=[x for x in [tj,catj] if x>=0]
                if not cands: xi,xp=end,cl[end]
                elif catj>=0 and (tj<0 or catj<=tj): xi,xp=catj,min(o[catj],cap)
                else: xi=min(tj+1,n-1); xp=o[tj+1] if tj+1<n else cl[tj]
            res[name]["ret"].append((xp/ep-1)*100); res[name]["hold"].append(xi-e); res[name]["year"].append(int(yrs[i]))
    if (si+1)%300==0: log(f"{si+1}/{len(syms)} old_sig={old_ct} clean_sig={clean_ct} fillskip={fill_skip}")
con.close()
log(f"POPULATION: old(mean>=1cr)={old_ct}  clean(med>={TMED}cr + fillguard)={clean_ct}  entry-fill skips={fill_skip}")
log("NAMED-STOCK inclusion after clean filter:")
for s in sorted(NAMED): print(f"   {s:12s}: {named_hits.get(s,['(excluded)'])}",flush=True)
def agg(r,h,cost):
    r=np.array(r)-cost; w=r[r>0]; l=r[r<0]   # cost already in % (0.20 = 0.20% round-trip)
    return dict(n=len(r),mean=round(r.mean(),2),median=round(np.median(r),2),win=round((r>0).mean()*100,1),
                pf=round(w.sum()/abs(l.sum()),2) if l.sum()!=0 else 0,avghold=round(np.mean(h),1),
                tail50=round((r>50).mean()*100,1),maxloss=round(r.min(),1),total=round(r.sum(),0))
log("=== CLEAN population, exits net@0.20% ===")
rows=[dict(config=n,**agg(res[n]["ret"],res[n]["hold"],0.20)) for n,_,_ in CONFIGS]
print(pd.DataFrame(rows).sort_values("mean",ascending=False).to_string(index=False),flush=True)
log("=== cost sensitivity (net mean/trade) ===")
cs=[dict(config=n,c0=agg(res[n]["ret"],res[n]["hold"],0)["mean"],c010=agg(res[n]["ret"],res[n]["hold"],0.10)["mean"],
         c020=agg(res[n]["ret"],res[n]["hold"],0.20)["mean"],c035=agg(res[n]["ret"],res[n]["hold"],0.35)["mean"]) for n,_,_ in CONFIGS]
print(pd.DataFrame(cs).to_string(index=False),flush=True)
log("=== per-year mean net@0.20 (ST_10_3, DONCH_20, TIME_40) ===")
py={}
for n in ["ST_10_3","DONCH_20","TIME_40"]:
    d=pd.DataFrame({"ret":np.array(res[n]["ret"])-0.20,"y":res[n]["year"]}); py[n]=d.groupby("y")["ret"].mean().round(2)
print(pd.DataFrame(py).to_string(),flush=True)
log("DONE")
