"""Intra-week evolution: the daily CPR re-draws each day, so the weekly-pos x daily-CPR confluence state
changes day to day. Re-classify at EACH day's close and measure the REST-OF-WEEK move -> does a fresh
confluence read mid-week give a cue to adjust the options position? NIFTY 5min->daily, 2015-26."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
df["d"]=df.index.normalize()
dd=df.groupby("d").agg(o=("open","first"),h=("high","max"),l=("low","min"),cl=("close","last"))
# daily CPR band (from prior day)
dpH,dpL,dpC=dd["h"].shift(1),dd["l"].shift(1),dd["cl"].shift(1)
dP=(dpH+dpL+dpC)/3; dBC=(dpH+dpL)/2; dTC=2*dP-dBC
dd["dlo"]=np.minimum(dBC,dTC); dd["dhi"]=np.maximum(dBC,dTC)
# assign week + weekly band (from prior week)
dd["wk"]=dd.index.to_period("W-FRI")
wkb=dd.groupby("wk").agg(H=("h","max"),L=("l","min"),C=("cl","last"))
pH,pL,pC=wkb["H"].shift(1),wkb["L"].shift(1),wkb["C"].shift(1)
P=(pH+pL+pC)/3;BC=(pH+pL)/2;TC=2*P-BC
wkb["wlo"]=np.minimum(BC,TC);wkb["whi"]=np.maximum(BC,TC)
dd["wlo"]=dd["wk"].map(wkb["wlo"]);dd["whi"]=dd["wk"].map(wkb["whi"]);dd["wkclose"]=dd["wk"].map(wkb["C"])
dd["didx"]=dd.groupby("wk").cumcount()
def sd(px,lo,hi): return "above" if px>hi else("below" if px<lo else "inside")
dd["wk_side"]=[sd(a,b,c2) for a,b,c2 in zip(dd["cl"],dd["wlo"],dd["whi"])]
dd["dy_side"]=[sd(a,b,c2) for a,b,c2 in zip(dd["cl"],dd["dlo"],dd["dhi"])]
dd["cl_side"]=[sd(a,b,c2) for a,b,c2 in zip(dd["wkclose"],dd["wlo"],dd["whi"])]
dd["conf"]=(dd["wk_side"]==dd["dy_side"])&dd["wk_side"].isin(["above","below"])
dd["rest"]=(dd["wkclose"]-dd["cl"])/dd["cl"]*100
D=dd.dropna(subset=["wlo","dlo"]).copy()

# 1) does the weekly-side flip during the week?
flip=D.groupby("wk")["wk_side"].nunique()
print(f"weeks where the weekly-side classification CHANGES intra-week: {(flip>1).mean()*100:.0f}%  (so yes, it evolves daily)")

# 2) rest-of-week by (day, weekly-side, confluence) — does a fresh mid-week read predict the rest?
labels={0:"Mon",1:"Tue",2:"Wed",3:"Thu"}
print(f"\n{'day(close)':10s} {'state':26s} {'n':>4} {'rest-of-wk holds side':>21} {'rest net% avg':>14}")
for di in range(4):
    sub=D[D["didx"]==di]
    for ws in ("above","below"):
        for cf,tag in [(True,"confluence (wk=daily)"),(False,"coin-flip (wk!=daily)")]:
            s=sub[(sub["wk_side"]==ws)&(sub["conf"]==cf)]
            if len(s)<8: continue
            held=(s["cl_side"]==ws).mean()*100
            print(f"{labels[di]:10s} {ws+' '+tag:26s} {len(s):>4} {held:>20.0f}% {s['rest'].mean():>+13.2f}")
    print()

# 3) the transition you asked about: Monday coin-flip -> resolves to confluence by Wed -> rest-of-week?
mon=D[D["didx"]==0][["wk","wk_side","conf"]].rename(columns={"wk_side":"mws","conf":"mcf"})
wed=D[D["didx"]==2][["wk","wk_side","conf","cl_side","rest"]].rename(columns={"wk_side":"wws","conf":"wcf"})
mw=mon.merge(wed,on="wk")
for ws in ("above","below"):
    base=mw[(mw["mws"]==ws)&(~mw["mcf"])]                      # Monday coin-flip on this side
    res=base[(base["wws"]==ws)&(base["wcf"])]                  # ...that became confluence (same side) by Wed
    rev=base[(base["wws"]==ws)&(~base["wcf"])]                 # ...still coin-flip Wed
    if len(base)<6: continue
    print(f"Mon {ws} COIN-FLIP (n={len(base)}): -> Wed CONFLUENCE same side n={len(res)} -> rest holds {(res['cl_side']==ws).mean()*100:.0f}% net {res['rest'].mean():+.2f}%"
          f"  |  still coin-flip n={len(rev)} -> holds {(rev['cl_side']==ws).mean()*100 if len(rev) else 0:.0f}% net {rev['rest'].mean() if len(rev) else 0:+.2f}%")
