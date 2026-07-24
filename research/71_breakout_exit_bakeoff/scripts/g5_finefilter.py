"""G5 fine-filter: on the winning book (Donchian-20 + NIFTY>200DMA gate + 8 concurrent),
sweep trade-frequency controls and measure the actual entry CADENCE (trades/week, busiest
day, % days with an entry) alongside performance. Answers 'is it tradeable day-to-day?'."""
import sqlite3, pandas as pd, numpy as np, os, time
from collections import defaultdict
RES="/home/arun/quantifyd/research/71_breakout_exit_bakeoff/results"
DB="/home/arun/quantifyd/backtest_data/market_data.db"
CAP=1_000_000.0; COST=0.0020; TMED=5.0; GAP_MAX=0.15; MAXBARS=120; CATA=0.20; START="2006-01-01"; MAX_OPEN=8
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def macd(c): return ema(c,12)-ema(c,26)
def atr(df,n):
    h,l,c=df["high"],df["low"],df["close"]; pc=c.shift()
    return pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1).ewm(alpha=1/n,adjust=False).mean()
def fta(b,s):
    sub=b[s:]; w=np.argmax(sub); return s+w if sub.size and sub[w] else -1
con=sqlite3.connect(DB)
bn=pd.read_sql("select date,close from market_data_unified where symbol='NIFTYBEES' and timeframe='day' order by date",con,parse_dates=["date"]).set_index("date")
bn=bn[~bn.index.duplicated()]; bn["ma200"]=bn["close"].rolling(200).mean()
cal=bn.index[bn.index>=START]
regime_ok={d.toordinal():(bn.loc[d,"close"]>bn.loc[d,"ma200"]) if not np.isnan(bn.loc[d,"ma200"]) else False for d in cal}
syms=[r[0] for r in con.execute("select distinct symbol from market_data_unified where timeframe='day'")]
symclose={}; trades=[]
for sym in syms:
    df=pd.read_sql("select date,open,high,low,close,volume from market_data_unified where symbol=? and timeframe='day' order by date",con,params=(sym,),parse_dates=["date"]).set_index("date")
    df=df[df.index.notna()]
    if len(df)<300: continue
    c=df["close"]; o=df["open"].values; hi=df["high"].values; lo=df["low"].values; cl=c.values
    ords=np.array([d.toordinal() for d in df.index])
    md=macd(c); mw=macd(c.resample("W-FRI").last()).reindex(df.index,method="ffill"); mm=macd(c.resample("ME").last()).reindex(df.index,method="ffill")
    volspike=df["volume"]/df["volume"].rolling(20).mean(); hi252=c.rolling(252,min_periods=200).max()
    turn_med=(c*df["volume"]).rolling(20).median()/1e7
    clean=((md>0)&(mw>0)&(mm>0)&(volspike>=2.0)&(c>=0.98*hi252)&(c>=20)&(turn_med>=TMED)).values
    dn=cl<pd.Series(lo,index=df.index).rolling(20).min().shift(1).values
    n=len(df); symclose[sym]=(ords,cl)
    for i in np.where(clean)[0]:
        if i+1>=n: continue
        e=i+1; ep=o[e]
        if ep<=0 or np.isnan(ep) or (hi[e]-lo[e])/ep<0.01 or (ep/cl[i]-1)>GAP_MAX: continue
        if ords[e]<cal[0].toordinal(): continue
        cs=ep*(1-CATA); end=min(n-1,e+MAXBARS)
        tj=fta(dn,e); tj=tj if 0<=tj<=end else -1
        catj=fta(lo<=cs,e); catj=catj if 0<=catj<=end else -1
        cands=[x for x in [tj,catj] if x>=0]
        if not cands: xi,xp=end,cl[end]
        elif catj>=0 and (tj<0 or catj<=tj): xi,xp=catj,min(o[catj],cs)
        else: xi=min(tj+1,n-1); xp=(o[tj+1] if tj+1<n else cl[tj])
        trades.append(dict(sym=sym,eo=int(ords[e]),ep=float(ep),run=float((cl[i]/cl[i-1]-1)*100),xo=int(ords[xi]),xp=float(xp)))
con.close()
trades.sort(key=lambda x:x["eo"]); by_entry=defaultdict(list)
for tr in trades: by_entry[tr["eo"]].append(tr)
log(f"{len(trades)} candidate trades loaded")
def close_at(sym,to):
    ords,cl=symclose[sym]; k=np.searchsorted(ords,to,side="right")-1; return cl[k] if k>=0 else np.nan
def run(max_per_day,min_run):
    realized=0.0; open_pos=[]; peak=CAP; maxdd=0.0; eq=CAP; curve=[]; entry_days=defaultdict(int)
    for d in cal:
        to=d.toordinal()
        still=[]
        for p in open_pos:
            if to>=p["xo"]: realized+=p["notional"]*(p["xp"]/p["ep"]-1)-p["notional"]*COST
            else: still.append(p)
        open_pos=still
        unreal=sum(p["notional"]*(close_at(p["sym"],to)/p["ep"]-1) for p in open_pos) if open_pos else 0.0
        eq=CAP+realized+unreal
        if regime_ok.get(to,False) and len(open_pos)<MAX_OPEN and to in by_entry:
            held={p["sym"] for p in open_pos}
            cands=sorted([x for x in by_entry[to] if x["sym"] not in held and x["run"]>=min_run],key=lambda x:-x["run"])
            added=0
            for x in cands:
                if len(open_pos)>=MAX_OPEN or added>=max_per_day: break
                open_pos.append(dict(sym=x["sym"],ep=x["ep"],xo=x["xo"],xp=x["xp"],notional=eq/MAX_OPEN)); held.add(x["sym"]); added+=1
            if added: entry_days[to]=added
        peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1); curve.append((d,eq))
    cur=pd.Series({d:e for d,e in curve}); yrs=(cur.index[-1]-cur.index[0]).days/365.25
    cagr=(cur.iloc[-1]/CAP)**(1/yrs)-1; dr=cur.pct_change().dropna(); sharpe=dr.mean()/(dr.std()+1e-9)*np.sqrt(252)
    ntr=sum(entry_days.values()); wks=(cur.index[-1]-cur.index[0]).days/7
    active=len(entry_days); busiest=max(entry_days.values()) if entry_days else 0
    return dict(max_per_day=max_per_day,min_run=min_run,cagr=round(cagr*100,1),sharpe=round(sharpe,2),
                maxdd=round(maxdd*100,1),calmar=round(cagr/abs(maxdd),2) if maxdd<0 else 0,
                trades=ntr,tr_per_wk=round(ntr/wks,2),busiest_day=busiest,pct_days_entry=round(active/len(cal)*100,1)),cur
rows=[]; curves={}
for mpd in [99,3,2,1]:
    for mr in [0.0,2.0,4.0]:
        r,cur=run(mpd,mr); rows.append(r); curves[(mpd,mr)]=cur; log(f"done mpd={mpd} min_run={mr}: {r}")
sm=pd.DataFrame(rows); sm.to_csv(os.path.join(RES,"g5_finefilter.csv"),index=False)
log("=== G5 fine-filter (Donchian-20 + gate + 8 concurrent) ===")
print(sm.to_string(index=False),flush=True)
# recommended config = max 1 entry/day, no run gate -> save curve + tearsheet
best=curves[(1,0.0)]; best.to_csv(os.path.join(RES,"g5_recommended_curve.csv"))
import sys; sys.path.insert(0,"/home/arun/quantifyd/research/_utilities")
from tearsheet import generate_tearsheet
bref=bn["close"].reindex(best.index).ffill()
meta={"Strategy":"MTF-bullish volume breakout | Donchian-20 trail, no target, NIFTY>200DMA gate, 8 concurrent, MAX 1 new entry/day",
      "Period":f"{best.index.min().date()} to {best.index.max().date()}","Universe":"NSE, median 20d turnover >= Rs.5cr",
      "Costs":"0.20% round-trip, gross of tax","Benchmark":"NIFTYBEES"}
p=generate_tearsheet(best,bref,"Breakout Swing Book (Donchian trail + regime gate + 1/day cap)",meta=meta,out_dir=RES)
log(f"recommended tearsheet: {p}")
log("DONE")
