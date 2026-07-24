"""G2 exit bake-off: replay the MTF-bullish volume-breakout entry population through
every exit family. Per-trade % returns (gross), aggregated net at 3 cost levels.
Entry = next-day open after signal. Close-based exits fill next open (no look-ahead).
All non-fixed families carry a 20% catastrophic hard SL floor; 120-bar backstop on all."""
import sqlite3, pandas as pd, numpy as np, os, time, csv
BASE="/home/arun/quantifyd/research/71_breakout_exit_bakeoff"
RES=os.path.join(BASE,"results"); os.makedirs(RES,exist_ok=True)
DB="/home/arun/quantifyd/backtest_data/market_data.db"
TRADES=os.path.join(RES,"g2_trades.csv")
SUMMARY=os.path.join(RES,"g2_summary.csv")
MAXBARS=120; CATASTROPHE=0.20
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def macd(c): return ema(c,12)-ema(c,26)
def atr(df,n=22):
    h,l,c=df["high"],df["low"],df["close"]; pc=c.shift()
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def supertrend(df,period,mult):
    a=atr(df,period); hl2=(df["high"]+df["low"])/2
    ub=hl2+mult*a; lb=hl2-mult*a; c=df["close"].values
    ub=ub.values; lb=lb.values; n=len(c)
    fub=ub.copy(); flb=lb.copy(); st=np.zeros(n); dir_=np.ones(n)
    for i in range(1,n):
        fub[i]=ub[i] if (ub[i]<fub[i-1] or c[i-1]>fub[i-1]) else fub[i-1]
        flb[i]=lb[i] if (lb[i]>flb[i-1] or c[i-1]<flb[i-1]) else flb[i-1]
        if c[i]>fub[i-1]: dir_[i]=1
        elif c[i]<flb[i-1]: dir_[i]=-1
        else: dir_[i]=dir_[i-1]
    return dir_   # +1 up, -1 down

# exit configs: (name, kind, params)
CONFIGS=[
 ("FIX_sl8_tgtNone","fix",dict(sl=.08,tgt=None)),
 ("FIX_sl8_tgt15","fix",dict(sl=.08,tgt=.15)),
 ("FIX_sl8_tgt20","fix",dict(sl=.08,tgt=.20)),
 ("FIX_sl10_tgt20","fix",dict(sl=.10,tgt=.20)),
 ("FIX_sl10_tgt30","fix",dict(sl=.10,tgt=.30)),
 ("FIX_sl5_tgt10","fix",dict(sl=.05,tgt=.10)),
 ("FIX_sl15_tgtNone","fix",dict(sl=.15,tgt=None)),
 ("CHAND_atr22_m2","chand",dict(m=2.0)),
 ("CHAND_atr22_m2.5","chand",dict(m=2.5)),
 ("CHAND_atr22_m3","chand",dict(m=3.0)),
 ("CHAND_atr22_m4","chand",dict(m=4.0)),
 ("ST_10_3","st",dict(p=10,m=3.0)),
 ("ST_10_2","st",dict(p=10,m=2.0)),
 ("ST_7_3","st",dict(p=7,m=3.0)),
 ("ST_7_2","st",dict(p=7,m=2.0)),
 ("DONCH_10","donch",dict(N=10)),
 ("DONCH_15","donch",dict(N=15)),
 ("DONCH_20","donch",dict(N=20)),
 ("EMA_10","ema",dict(N=10)),
 ("EMA_20","ema",dict(N=20)),
 ("EMA_21","ema",dict(N=21)),
 ("EMA_50","ema",dict(N=50)),
 ("TIME_5","time",dict(N=5)),
 ("TIME_10","time",dict(N=10)),
 ("TIME_15","time",dict(N=15)),
 ("TIME_20","time",dict(N=20)),
 ("TIME_40","time",dict(N=40)),
 ("HYB_sl8_chand3","hyb",dict(sl=.08,m=3.0)),
 ("HYB_sl8_st10_3","hyb_st",dict(sl=.08,p=10,m=3.0)),
 ("HYB_sl8_donch15","hyb_donch",dict(sl=.08,N=15)),
 ("HYB_sl8_ema20","hyb_ema",dict(sl=.08,N=20)),
]

con=sqlite3.connect(DB)
syms=[r[0] for r in con.execute("select distinct symbol from market_data_unified where timeframe='day'")]
log(f"{len(syms)} symbols")

# per-config accumulators
res={name:{"ret":[],"hold":[],"year":[]} for name,_,_ in CONFIGS}

def first_true_after(arr_bool, start):
    """index of first True at position >= start, else -1"""
    sub=arr_bool[start:]
    w=np.argmax(sub)
    if sub.size and sub[w]: return start+w
    return -1

nsig=0
for si,sym in enumerate(syms):
    df=pd.read_sql("select date,open,high,low,close,volume from market_data_unified where symbol=? and timeframe='day' order by date",
                   con,params=(sym,),parse_dates=["date"]).set_index("date")
    if len(df)<300: continue
    c=df["close"]; o=df["open"].values; hi=df["high"].values; lo=df["low"].values; cl=c.values
    yrs=df.index.year.values
    # entry signal (same as G1)
    md=macd(c); mw=macd(c.resample("W-FRI").last()).reindex(df.index,method="ffill")
    mm=macd(c.resample("ME").last()).reindex(df.index,method="ffill")
    vol20=df["volume"].rolling(20).mean(); volspike=df["volume"]/vol20
    hi252=c.rolling(252,min_periods=200).max()
    turn=(c*df["volume"]).rolling(20).mean()/1e7
    qual=((md>0)&(mw>0)&(mm>0)&(volspike>=2.0)&(c>=0.98*hi252)&(c>=20)&(turn>=1.0)).values
    qpos=np.where(qual)[0]
    qpos=qpos[qpos+1<len(df)]
    if len(qpos)==0: continue
    # symbol-wide indicator exit-trigger series (close-based) -> exit at NEXT open
    A=atr(df,22).values
    hh=pd.Series(hi,index=df.index)
    st_series={}
    for p,m in [(10,3.0),(10,2.0),(7,3.0),(7,2.0)]:
        d=supertrend(df,p,m); st_series[(p,m)]=(d<0)  # flip down = exit
    donch={}
    for N in [10,15,20]:
        low_ch=pd.Series(lo,index=df.index).rolling(N).min().shift(1).values
        donch[N]=cl<low_ch
    emax={}
    for N in [10,20,21,50]:
        emax[N]=(cl<ema(c,N).values)
    n=len(df)

    for i in qpos:
        e=i+1; ep=o[e]
        if ep<=0 or np.isnan(ep): continue
        cap_stop=ep*(1-CATASTROPHE)
        end=min(n-1,e+MAXBARS)
        for name,kind,p in CONFIGS:
            xi=None; xp=None
            if kind=="fix":
                sl_lvl=ep*(1-p["sl"]); tgt_lvl=ep*(1+p["tgt"]) if p["tgt"] else None
                for j in range(e,end+1):
                    if lo[j]<=sl_lvl:
                        xi=j; xp=min(o[j],sl_lvl) if o[j]<sl_lvl else sl_lvl; break
                    if tgt_lvl and hi[j]>=tgt_lvl:
                        xi=j; xp=max(o[j],tgt_lvl) if o[j]>tgt_lvl else tgt_lvl; break
                if xi is None: xi=end; xp=cl[end]
            elif kind=="time":
                j=min(e+p["N"],n-1)
                # catastrophe check within window
                hit=first_true_after(lo<=cap_stop, e)
                if 0<=hit<=j: xi=hit; xp=min(o[hit],cap_stop)
                else: xi=j; xp=o[j] if j<n else cl[-1]
            else:
                # trailing families: build trigger index
                if kind=="chand":
                    hh_run=np.maximum.accumulate(hi[e:end+1])
                    stop=hh_run - p["m"]*A[e:end+1]
                    trig=cl[e:end+1] < stop
                    w=first_true_after(trig,0); tj=(e+w) if w>=0 else -1
                elif kind in ("st",):
                    trig=st_series[(p["p"],p["m"])]; tj=first_true_after(trig,e); tj=tj if tj<=end else -1
                elif kind=="donch":
                    tj=first_true_after(donch[p["N"]],e); tj=tj if tj<=end else -1
                elif kind=="ema":
                    tj=first_true_after(emax[p["N"]],e); tj=tj if tj<=end else -1
                elif kind=="hyb":
                    hh_run=np.maximum.accumulate(hi[e:end+1]); stop=hh_run-p["m"]*A[e:end+1]
                    trig=cl[e:end+1]<stop; w=first_true_after(trig,0); tj=(e+w) if w>=0 else -1
                elif kind=="hyb_st":
                    tj=first_true_after(st_series[(p["p"],p["m"])],e); tj=tj if tj<=end else -1
                elif kind=="hyb_donch":
                    tj=first_true_after(donch[p["N"]],e); tj=tj if tj<=end else -1
                elif kind=="hyb_ema":
                    tj=first_true_after(emax[p["N"]],e); tj=tj if tj<=end else -1
                # catastrophe SL (all trailing + hybrids)
                catj=first_true_after(lo<=cap_stop,e); catj=catj if catj<=end else -1
                # hybrid initial hard SL
                hardj=-1
                if kind.startswith("hyb"):
                    sl_lvl=ep*(1-p["sl"]); hj=first_true_after(lo<=sl_lvl,e)
                    hardj=hj if hj<=end else -1
                # earliest trigger
                cands=[x for x in [tj,catj,hardj] if x is not None and x>=0]
                if not cands:
                    xi=end; xp=cl[end]
                else:
                    xi=min(cands)
                    if xi==catj and (tj<0 or catj<=tj) and (hardj<0 or catj<=hardj):
                        xp=min(o[xi],cap_stop)
                    elif kind.startswith("hyb") and xi==hardj and (tj<0 or hardj<tj):
                        sl_lvl=ep*(1-p["sl"]); xp=min(o[xi],sl_lvl) if o[xi]<sl_lvl else sl_lvl
                    else:
                        # close-based trailing trigger at xi -> fill NEXT open
                        xp=o[xi+1] if xi+1<n else cl[xi]
                        xi=min(xi+1,n-1)
            ret=(xp/ep-1)*100
            res[name]["ret"].append(ret); res[name]["hold"].append(xi-e); res[name]["year"].append(int(yrs[i]))
        nsig+=1
    if (si+1)%200==0: log(f"{si+1}/{len(syms)} symbols, {nsig} signals simulated")
con.close()
log(f"simulated {nsig} signals x {len(CONFIGS)} configs")

# aggregate net at cost levels
def agg(rets,holds,cost):
    r=np.array(rets)-cost*100  # cost as % round trip
    if len(r)==0: return None
    wins=r[r>0]; losses=r[r<0]
    pf=wins.sum()/abs(losses.sum()) if losses.sum()!=0 else np.inf
    return dict(n=len(r),mean=r.mean(),median=np.median(r),win=(r>0).mean()*100,
                pf=pf,avgwin=wins.mean() if len(wins) else 0,avgloss=losses.mean() if len(losses) else 0,
                std=r.std(),sharpe_t=r.mean()/(r.std()+1e-9)*np.sqrt(len(r)),
                avghold=np.mean(holds),tail50=(r>50).mean()*100,maxloss=r.min(),total=r.sum())
rows=[]
for name,_,_ in CONFIGS:
    d=res[name]
    for cost in [0.0,0.10,0.20,0.35]:
        a=agg(d["ret"],d["hold"],cost/100)
        if a: rows.append(dict(config=name,cost_pct=cost,**{k:round(v,3) for k,v in a.items()}))
sm=pd.DataFrame(rows); sm.to_csv(SUMMARY,index=False)

# print net@0.20 ranked by mean
net=sm[sm.cost_pct==0.20].sort_values("mean",ascending=False)
log("=== NET @0.20% round-trip, ranked by mean per-trade return ===")
print(net[["config","n","mean","median","win","pf","avgwin","avgloss","avghold","tail50","maxloss","total","sharpe_t"]].to_string(index=False),flush=True)

# per-year mean for top-6 configs (net@0.20)
top6=net.head(6)["config"].tolist()
log("=== per-year mean (net@0.20) for top-6 ===")
py={}
for name in top6:
    d=res[name]; dfp=pd.DataFrame({"ret":np.array(d["ret"])-0.20,"year":d["year"]})
    py[name]=dfp.groupby("year")["ret"].mean().round(2)
print(pd.DataFrame(py).to_string(),flush=True)
log("DONE")
