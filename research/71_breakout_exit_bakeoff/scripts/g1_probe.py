import sqlite3, pandas as pd, numpy as np, sys, os, time
BASE=os.path.dirname(os.path.abspath(__file__))
RES=os.path.join(BASE,"..","results"); os.makedirs(RES,exist_ok=True)
ENTRIES=os.path.join(RES,"entries.csv")
DB="/home/arun/quantifyd/backtest_data/market_data.db"
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def macd(c): return ema(c,12)-ema(c,26)

con=sqlite3.connect(DB)
syms=[r[0] for r in con.execute("select distinct symbol from market_data_unified where timeframe='day'")]
log(f"{len(syms)} daily symbols")
HZ=[5,10,20,40]
rows=[]
n=0
for sym in syms:
    df=pd.read_sql("select date,open,high,low,close,volume from market_data_unified where symbol=? and timeframe='day' order by date",
                   con,params=(sym,),parse_dates=["date"])
    if len(df)<300: continue
    df=df.set_index("date")
    c=df["close"]
    md=macd(c)
    mw=macd(c.resample("W-FRI").last()).reindex(df.index,method="ffill")
    mm=macd(c.resample("ME").last()).reindex(df.index,method="ffill")
    vol20=df["volume"].rolling(20).mean()
    volspike=df["volume"]/vol20
    hi252=c.rolling(252,min_periods=200).max()
    near=c>=0.98*hi252
    turn=(c*df["volume"]).rolling(20).mean()/1e7   # in Rs cr
    pct=c.pct_change()*100
    qual=(md>0)&(mw>0)&(mm>0)&(volspike>=2.0)&near&(c>=20)&(turn>=1.0)
    o=df["open"].values; cl=c.values; idx=df.index
    qpos=np.where(qual.values)[0]
    for i in qpos:
        if i+1>=len(df): continue   # need next-open entry
        ep=o[i+1]
        if ep<=0 or np.isnan(ep): continue
        fr={}
        for h in HZ:
            j=i+1+h
            fr[h]=(cl[j]/ep-1)*100 if j<len(df) else np.nan
        rows.append((sym,str(idx[i].date()),round(pct.values[i],2),round(turn.values[i],2),
                     round(ep,2),*[round(fr[h],2) if not np.isnan(fr[h]) else "" for h in HZ]))
    n+=1
    if n%200==0: log(f"scanned {n} symbols, {len(rows)} qual signals")
con.close()
cols=["symbol","sig_date","pct_run","turn_cr","entry_open"]+[f"fwd{h}" for h in HZ]
ent=pd.DataFrame(rows,columns=cols)
ent.to_csv(ENTRIES,index=False)
log(f"TOTAL qualifying signals: {len(ent)} across {ent['symbol'].nunique()} names -> {ENTRIES}")

# forward-return summary: take-all vs top-K per day by pct_run
ent["sig_date"]=pd.to_datetime(ent["sig_date"])
ent["year"]=ent["sig_date"].dt.year
def summ(d,label):
    out=[]
    for h in HZ:
        v=pd.to_numeric(d[f"fwd{h}"],errors="coerce").dropna()
        out.append((label,h,len(v),round(v.mean(),2),round(v.median(),2),round((v>0).mean()*100,1)))
    return out
S=[]
S+=summ(ent,"ALL")
for K in [3,5,10]:
    topk=ent.sort_values("pct_run",ascending=False).groupby(ent["sig_date"]).head(K)
    S+=summ(topk,f"TOP{K}")
sm=pd.DataFrame(S,columns=["set","horizon_d","n","mean_ret%","median_ret%","win%"])
sm.to_csv(os.path.join(RES,"g1_forward_returns.csv"),index=False)
log("FORWARD-RETURN SUMMARY (gross, entry=next open):")
print(sm.to_string(index=False),flush=True)
# per-year mean fwd20 for TOP5
top5=ent.sort_values("pct_run",ascending=False).groupby(ent["sig_date"]).head(5)
py=top5.groupby("year").apply(lambda d: pd.Series({
    "n":len(d),"mean_fwd20":round(pd.to_numeric(d["fwd20"],errors="coerce").mean(),2),
    "win20%":round((pd.to_numeric(d["fwd20"],errors="coerce")>0).mean()*100,1)}))
print("\nTOP5 per-year fwd20:\n",py.to_string(),flush=True)
log("DONE")
