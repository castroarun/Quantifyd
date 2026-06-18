"""Multi-timeframe CPR confluence: at Monday's 1st-30min (09:45), is price above/below BOTH the WEEKLY CPR
and Monday's DAILY CPR? When the two timeframes AGREE, is the week's directional follow-through stronger?
All causal (weekly CPR from prior week; Monday's daily CPR from prior trading day). NIFTY 5min, 2015-2026."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
df["hhmm"]=df.index.strftime("%H:%M"); df["d"]=df.index.normalize()
# DAILY OHLC -> daily CPR band (drawn for day X from day X-1)
dd=df.groupby("d").agg(o=("open","first"),h=("high","max"),l=("low","min"),cl=("close","last"))
dpH,dpL,dpC=dd["h"].shift(1),dd["l"].shift(1),dd["cl"].shift(1)
dP=(dpH+dpL+dpC)/3; dBC=(dpH+dpL)/2; dTC=2*dP-dBC
dd["dlo"]=np.minimum(dBC,dTC); dd["dhi"]=np.maximum(dBC,dTC)
# WEEKLY bars + weekly CPR + Monday 09:45 price (f30) + which day Monday is
def pw(g):
    o=g["open"].iloc[0];cl=g["close"].iloc[-1];d0=g["d"].min();day0=g[g["d"]==d0]
    f=day0[day0["hhmm"]<="09:40"]
    return pd.Series({"open":o,"close":cl,"d0":d0,"f30":(f["close"].iloc[-1] if len(f) else o)})
wk=df.groupby(pd.Grouper(freq="W-FRI")).apply(pw).dropna()
pC=df.groupby(pd.Grouper(freq="W-FRI"))["close"].last().shift(1)
pH=df.groupby(pd.Grouper(freq="W-FRI"))["high"].max().shift(1)
pL=df.groupby(pd.Grouper(freq="W-FRI"))["low"].min().shift(1)
P=(pH+pL+pC)/3; BC=(pH+pL)/2; TC=2*P-BC
wk["wlo"]=np.minimum(BC,TC); wk["whi"]=np.maximum(BC,TC)
wk["dlo"]=wk["d0"].map(dd["dlo"]); wk["dhi"]=wk["d0"].map(dd["dhi"])
def sd(px,lo,hi): return "above" if px>hi else("below" if px<lo else "inside")
wk["wk_side"]=[sd(a,b,c2) for a,b,c2 in zip(wk["f30"],wk["wlo"],wk["whi"])]
wk["dy_side"]=[sd(a,b,c2) for a,b,c2 in zip(wk["f30"],wk["dlo"],wk["dhi"])]
wk["cl_side"]=[sd(a,b,c2) for a,b,c2 in zip(wk["close"],wk["wlo"],wk["whi"])]
wk["net"]=(wk["close"]-wk["open"])/wk["open"]*100
W=wk.dropna(subset=["wlo","dlo"]).copy()

def show(d,label):
    print(f"\n===== {label}  n={len(d)} =====")
    print(f"{'weekly x daily (1st-30min)':30s} {'n':>4} {'%wks':>5} | {'week closed ABOVE wCPR':>22} {'net% signed avg':>16}")
    combos=[("above","above","BOTH ABOVE (bull confluence)"),("above","below","split (wk+ / dy-)"),
            ("below","below","BOTH BELOW (bear confluence)"),("below","above","split (wk- / dy+)"),
            ("above","inside","wk+ / dy inside"),("below","inside","wk- / dy inside")]
    for ws,ds,lab in combos:
        s=d[(d["wk_side"]==ws)&(d["dy_side"]==ds)]
        if len(s)<5: continue
        print(f"{lab:30s} {len(s):>4} {len(s)/len(d)*100:>4.0f}% | {(s['cl_side']=='above').mean()*100:>21.0f}% {s['net'].mean():>+15.2f}")
    # collapsed agree vs disagree (ignoring inside)
    agree_bull=d[(d["wk_side"]=="above")&(d["dy_side"]=="above")]
    agree_bear=d[(d["wk_side"]=="below")&(d["dy_side"]=="below")]
    disagree=d[((d["wk_side"]=="above")&(d["dy_side"]=="below"))|((d["wk_side"]=="below")&(d["dy_side"]=="above"))]
    print("  -- multi-TF agreement (held the signalled direction at week close) --")
    print(f"  BOTH agree BULL : n={len(agree_bull):>3}  closed above wCPR {(agree_bull['cl_side']=='above').mean()*100:>3.0f}%  net {agree_bull['net'].mean():+.2f}%")
    print(f"  BOTH agree BEAR : n={len(agree_bear):>3}  closed below wCPR {(agree_bear['cl_side']=='below').mean()*100:>3.0f}%  net {agree_bear['net'].mean():+.2f}%")
    held_dis=(((disagree['wk_side']=='above')&(disagree['cl_side']=='above'))|((disagree['wk_side']=='below')&(disagree['cl_side']=='below'))).mean()*100
    print(f"  DISAGREE (split): n={len(disagree):>3}  held weekly-side {held_dis:>3.0f}%  (vs weekly-alone base above 69% / below 58%)")

show(W,"FULL 2015-2026")
show(W[W.index.year>=2023],"RECENT 2023-2026")
