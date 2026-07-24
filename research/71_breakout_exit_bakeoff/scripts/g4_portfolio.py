"""G4 portfolio equity curves: MTF-bullish volume-breakout book.
Sweep exit {ST(10,3), Donchian-20} x regime {none, NIFTYBEES>200DMA} x max_open {5,8,10,15}.
Compounding, mark-to-market daily equity + drawdown. Benchmark = NIFTYBEES.
Entry selection when > slots free: rank day's qualifiers by today's %-run (user's workflow).
Cost 0.20% round-trip on notional. Capital Rs.10L. Window 2006-01-01+ (200DMA warmup)."""
import sqlite3, pandas as pd, numpy as np, os, time, json
BASE="/home/arun/quantifyd/research/71_breakout_exit_bakeoff"; RES=os.path.join(BASE,"results")
DB="/home/arun/quantifyd/backtest_data/market_data.db"
CAP=1_000_000.0; COST=0.0020; TMED=5.0; GAP_MAX=0.15; MAXBARS=120; CATA=0.20; START="2006-01-01"
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def macd(c): return ema(c,12)-ema(c,26)
def atr(df,n):
    h,l,c=df["high"],df["low"],df["close"]; pc=c.shift()
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def supertrend_down(df,period,mult):
    a=atr(df,period).values; hl2=((df["high"]+df["low"])/2).values; c=df["close"].values; n=len(c)
    ub=hl2+mult*a; lb=hl2-mult*a; fub=ub.copy(); flb=lb.copy(); dr=np.ones(n)
    for i in range(1,n):
        fub[i]=ub[i] if (ub[i]<fub[i-1] or c[i-1]>fub[i-1]) else fub[i-1]
        flb[i]=lb[i] if (lb[i]>flb[i-1] or c[i-1]<flb[i-1]) else flb[i-1]
        dr[i]=1 if c[i]>fub[i-1] else (-1 if c[i]<flb[i-1] else dr[i-1])
    return dr<0
def fta(b,s):
    sub=b[s:]; w=np.argmax(sub); return s+w if sub.size and sub[w] else -1

con=sqlite3.connect(DB)
# benchmark + regime
bn=pd.read_sql("select date,close from market_data_unified where symbol='NIFTYBEES' and timeframe='day' order by date",con,parse_dates=["date"]).set_index("date")
bn=bn[~bn.index.duplicated()]; bn["ma200"]=bn["close"].rolling(200).mean()
cal=bn.index[bn.index>=START]
regime_ok={d.toordinal():(bn.loc[d,"close"]>bn.loc[d,"ma200"]) if not np.isnan(bn.loc[d,"ma200"]) else False for d in cal}
log(f"benchmark NIFTYBEES {bn.index.min().date()}..{bn.index.max().date()} calendar={len(cal)}")

syms=[r[0] for r in con.execute("select distinct symbol from market_data_unified where timeframe='day'")]
symclose={}   # sym -> (ord array, close array) for MTM
trades=[]     # dicts
for si,sym in enumerate(syms):
    df=pd.read_sql("select date,open,high,low,close,volume from market_data_unified where symbol=? and timeframe='day' order by date",con,params=(sym,),parse_dates=["date"]).set_index("date")
    df=df[df.index.notna()]
    if len(df)<300: continue
    c=df["close"]; o=df["open"].values; hi=df["high"].values; lo=df["low"].values; cl=c.values
    ords=np.array([d.toordinal() for d in df.index])
    md=macd(c); mw=macd(c.resample("W-FRI").last()).reindex(df.index,method="ffill"); mm=macd(c.resample("ME").last()).reindex(df.index,method="ffill")
    volspike=df["volume"]/df["volume"].rolling(20).mean(); hi252=c.rolling(252,min_periods=200).max()
    tv=c*df["volume"]; turn_med=tv.rolling(20).median()/1e7
    clean=((md>0)&(mw>0)&(mm>0)&(volspike>=2.0)&(c>=0.98*hi252)&(c>=20)&(turn_med>=TMED)).values
    A=atr(df,22).values
    st=supertrend_down(df,10,3.0)
    dn=cl<pd.Series(lo,index=df.index).rolling(20).min().shift(1).values
    n=len(df)
    symclose[sym]=(ords,cl)
    qpos=np.where(clean)[0]; qpos=qpos[qpos+1<n]
    for i in qpos:
        e=i+1; ep=o[e]
        if ep<=0 or np.isnan(ep): continue
        if (hi[e]-lo[e])/ep<0.01 or (ep/cl[i]-1)>GAP_MAX: continue
        if df.index[e].toordinal()<cal[0].toordinal(): continue
        cap_stop=ep*(1-CATA); end=min(n-1,e+MAXBARS)
        exq={}
        for rule,trig in (("st",st),("dn",dn)):
            tj=fta(trig,e); tj=tj if 0<=tj<=end else -1
            catj=fta(lo<=cap_stop,e); catj=catj if 0<=catj<=end else -1
            cands=[x for x in [tj,catj] if x>=0]
            if not cands: xi,xp=end,cl[end]
            elif catj>=0 and (tj<0 or catj<=tj): xi,xp=catj,min(o[catj],cap_stop)
            else: xi=min(tj+1,n-1); xp=(o[tj+1] if tj+1<n else cl[tj])
            exq[rule]=(int(ords[xi]),float(xp))
        trades.append(dict(sym=sym,entry_ord=int(ords[e]),entry_px=float(ep),pct_run=float((cl[i]/cl[i-1]-1)*100),
                           st_xo=exq["st"][0],st_xp=exq["st"][1],dn_xo=exq["dn"][0],dn_xp=exq["dn"][1]))
    if (si+1)%400==0: log(f"{si+1}/{len(syms)} loaded, {len(trades)} trades")
con.close()
log(f"total tradeable signals in-window: {len(trades)}")
trades.sort(key=lambda x:x["entry_ord"])
from collections import defaultdict
by_entry=defaultdict(list)
for tr in trades: by_entry[tr["entry_ord"]].append(tr)

def close_at(sym,t_ord):
    ords,cl=symclose[sym]; k=np.searchsorted(ords,t_ord,side="right")-1
    return cl[k] if k>=0 else np.nan

def run(rule,gate,max_open):
    xo=f"{rule}_xo"; xp=f"{rule}_xp"
    equity=CAP; realized=0.0; open_pos=[]; curve=[]; peak=CAP; maxdd=0.0
    for d in cal:
        to=d.toordinal()
        # close
        still=[]
        for p in open_pos:
            if to>=p["xo"]:
                realized+=p["notional"]*(p["xp"]/p["entry_px"]-1) - p["notional"]*COST
            else: still.append(p)
        open_pos=still
        # mark to market
        unreal=sum(p["notional"]*(close_at(p["sym"],to)/p["entry_px"]-1) for p in open_pos) if open_pos else 0.0
        equity=CAP+realized+unreal
        # open new
        if (gate=="none" or regime_ok.get(to,False)) and len(open_pos)<max_open and to in by_entry:
            held={p["sym"] for p in open_pos}
            cands=sorted([x for x in by_entry[to] if x["sym"] not in held],key=lambda x:-x["pct_run"])
            for x in cands:
                if len(open_pos)>=max_open: break
                notional=equity/max_open
                open_pos.append(dict(sym=x["sym"],entry_px=x["entry_px"],xo=x[xo],xp=x[xp],notional=notional))
                held.add(x["sym"])
        peak=max(peak,equity); maxdd=min(maxdd,equity/peak-1)
        curve.append((d,equity))
    cur=pd.Series({d:e for d,e in curve})
    yrs=(cur.index[-1]-cur.index[0]).days/365.25
    cagr=(cur.iloc[-1]/CAP)**(1/yrs)-1
    dr=cur.pct_change().dropna(); sharpe=dr.mean()/(dr.std()+1e-9)*np.sqrt(252)
    calmar=cagr/abs(maxdd) if maxdd<0 else np.nan
    ntr=sum(1 for _ in trades)  # placeholder
    return dict(rule=rule,gate=gate,max_open=max_open,cagr=round(cagr*100,2),sharpe=round(sharpe,2),
                maxdd=round(maxdd*100,2),calmar=round(calmar,2),final=round(cur.iloc[-1]/1e5,2)),cur

rows=[]; curves={}
for rule in ["st","dn"]:
    for gate in ["none","nifty"]:
        for mo in [5,8,10,15]:
            r,cur=run(rule,gate,mo); rows.append(r); curves[(rule,gate,mo)]=cur
            log(f"{rule} gate={gate} open={mo}: CAGR={r['cagr']} DD={r['maxdd']} Calmar={r['calmar']} Sharpe={r['sharpe']} final={r['final']}xL")
sm=pd.DataFrame(rows).sort_values("calmar",ascending=False)
sm.to_csv(os.path.join(RES,"g4_portfolio_summary.csv"),index=False)
log("=== G4 portfolio summary (ranked by Calmar) ===")
print(sm.to_string(index=False),flush=True)
# save best curve + benchmark for tearsheet
best=sm.iloc[0]; bkey=(best["rule"],best["gate"],int(best["max_open"]))
bc=curves[bkey]; bc.to_csv(os.path.join(RES,"g4_best_curve.csv"))
bn_al=bn["close"].reindex(bc.index).ffill(); (bn_al/bn_al.iloc[0]*CAP).to_csv(os.path.join(RES,"g4_bench_curve.csv"))
json.dump({"best":bkey},open(os.path.join(RES,"g4_best.json"),"w"))
# per-year for best + a no-gate sibling
for key in [bkey,(best["rule"],"none",int(best["max_open"]))]:
    cur=curves[key]; yr=cur.resample("YE").last().pct_change().dropna()*100
    log(f"per-year % {key}: "+", ".join(f"{d.year}:{v:.0f}" for d,v in yr.items()))
log("DONE")
