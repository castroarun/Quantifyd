"""INTRADAY 'safe to sell' setups (research/67 extended). Target = the REST-OF-DAY move from 09:45 stays
CONTAINED (max excursion < stop) -> a short intraday straddle survives (not catching trends, selling calm).
Predictors known by 09:45: daily CPR (width/position), weekly CPR position, weekly+daily AGREE/DISAGREE,
1st-30min range, gap, ORB (broke the 09:15-09:30 range or not). NIFTY 5min 2015-26. Price-action proxy."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
df["d"]=df.index.normalize(); df["t"]=df.index.strftime("%H:%M")
rows=[]
for day,g in df.groupby("d"):
    g=g.sort_index()
    o=g["open"].iloc[0]
    f30=g[g["t"]<="09:40"]; orb=g[g["t"]<="09:25"]            # 1st 30-min ; opening range (first 15m)
    if len(f30)<4 or len(orb)<2: continue
    ref=f30["close"].iloc[-1]                                  # 09:45 close = entry ref
    f30h,f30l=f30["high"].max(),f30["low"].min()
    orbh,orbl=orb["high"].max(),orb["low"].min()
    rest=g[g["t"]>"09:45"]                                     # rest of day after entry
    if len(rest)<10: continue
    rh,rl=rest["high"].max(),rest["low"].min()
    maxexc=max(rh-ref,ref-rl)/ref*100                          # rest-of-day max excursion from entry
    rows.append(dict(day=day,o=o,ref=ref,f30rng=(f30h-f30l)/o*100,
                     orb_break=("up" if ref>orbh else("dn" if ref<orbl else "in")),maxexc=maxexc))
P=pd.DataFrame(rows).set_index("day")
# daily + weekly CPR (from prior day/week)
dd=df.groupby("d").agg(h=("high","max"),l=("low","min"),cl=("close","last"))
dpH,dpL,dpC=dd["h"].shift(1),dd["l"].shift(1),dd["cl"].shift(1)
dP=(dpH+dpL+dpC)/3;dBC=(dpH+dpL)/2;dTC=2*dP-dBC
dd["dlo"]=np.minimum(dBC,dTC);dd["dhi"]=np.maximum(dBC,dTC);dd["dcprw"]=(dTC-dBC).abs()/dpC*100
dd["pc"]=dpC
dd["wk"]=dd.index.to_period("W-FRI")
wk=dd.groupby("wk").agg(H=("h","max"),L=("l","min"),C=("cl","last"))
pH,pL,pC=wk["H"].shift(1),wk["L"].shift(1),wk["C"].shift(1)
P_=(pH+pL+pC)/3;BC=(pH+pL)/2;TC=2*P_-BC
wk["wlo"]=np.minimum(BC,TC);wk["whi"]=np.maximum(BC,TC)
dd["wlo"]=dd["wk"].map(wk["wlo"]);dd["whi"]=dd["wk"].map(wk["whi"])
X=P.join(dd[["dlo","dhi","dcprw","wlo","whi","pc"]]).dropna(subset=["dlo","wlo"])
def side(px,lo,hi): return "above" if px>hi else("below" if px<lo else "inside")
X["dpos"]=[side(a,b,c2) for a,b,c2 in zip(X["ref"],X["dlo"],X["dhi"])]
X["wpos"]=[side(a,b,c2) for a,b,c2 in zip(X["ref"],X["wlo"],X["whi"])]
X["gap"]=(X["o"]-X["pc"])/X["pc"]*100
X["agree"]=np.where((X.dpos==X.wpos)&X.dpos.isin(["above","below"]),"AGREE",np.where(X.dpos.isin(["above","below"])|X.wpos.isin(["above","below"]),"DISAGREE","inside"))

for THR in (0.5,0.8):
    X["calm"]=(X["maxexc"]<THR)
    base=X["calm"].mean()*100
    print(f"\n===== SAFE-TO-SELL = rest-of-day move < {THR}%  (BASE calm-rate {base:.0f}%, n={len(X)}) =====")
    def cut(name,col,order=None):
        print(f"  -- {name} --")
        vals=order or sorted(X[col].dropna().unique())
        for v in vals:
            s=X[X[col]==v]
            if len(s)<20: continue
            print(f"     {str(v):16s} n={len(s):>4}  calm {s['calm'].mean()*100:>3.0f}%  (lift {s['calm'].mean()*100-base:+.0f})")
    X["dcprw_q"]=pd.qcut(X["dcprw"],4,labels=["Q1 narrow","Q2","Q3","Q4 wide"])
    X["f30_q"]=pd.qcut(X["f30rng"],4,labels=["Q1 tiny","Q2","Q3","Q4 big"])
    X["gap_t"]=pd.cut(X["gap"].abs(),[0,0.3,0.7,99],labels=["flat","small","big"])
    cut("daily CPR width","dcprw_q",["Q1 narrow","Q2","Q3","Q4 wide"])
    cut("09:45 vs DAILY CPR","dpos",["inside","above","below"])
    cut("09:45 vs WEEKLY CPR","wpos",["inside","above","below"])
    cut("weekly+daily","agree",["AGREE","DISAGREE","inside"])
    cut("1st-30min range","f30_q",["Q1 tiny","Q2","Q3","Q4 big"])
    cut("gap","gap_t",["flat","small","big"])
    cut("ORB break by 09:45","orb_break",["in","up","dn"])
    if THR==0.5:
        # best combos
        print("  -- COMBOS (safe-to-sell hunters) --")
        for nm,m in [("inside DAILY CPR + tiny 1st candle",(X.dpos=="inside")&(X.f30_q=="Q1 tiny")),
                     ("inside daily + inside weekly CPR",(X.dpos=="inside")&(X.wpos=="inside")),
                     ("narrow daily CPR + flat gap",(X.dcprw_q=="Q1 narrow")&(X.gap_t=="flat")),
                     ("inside daily + ORB not broken",(X.dpos=="inside")&(X.orb_break=="in")),
                     ("inside daily + tiny candle + flat gap",(X.dpos=="inside")&(X.f30_q=="Q1 tiny")&(X.gap_t=="flat"))]:
            s=X[m]
            if len(s)>=20: print(f"     {nm:42s} n={len(s):>4} calm {s['calm'].mean()*100:>3.0f}% (lift {s['calm'].mean()*100-base:+.0f})")
