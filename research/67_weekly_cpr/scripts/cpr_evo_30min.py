"""Intra-week re-check on EACH day's 1st-30-min candle (09:45) — morning variant (vs the day-close version).
Per day: 09:45 close vs weekly CPR + that day's daily CPR (daily CPR from prior day) -> rest-of-week to
the week's final close. NIFTY 5min->daily, 11y 2015-26."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
df["d"]=df.index.normalize(); df["hhmm"]=df.index.strftime("%H:%M")
day=df.groupby("d").agg(h=("high","max"),l=("low","min"),cl=("close","last"))
day["f30"]=df[df["hhmm"]<="09:40"].groupby("d")["close"].last()   # ~09:45 close per day
day=day.dropna(subset=["f30"])
# daily CPR from prior day
dpH,dpL,dpC=day["h"].shift(1),day["l"].shift(1),day["cl"].shift(1)
dP=(dpH+dpL+dpC)/3; dBC=(dpH+dpL)/2; dTC=2*dP-dBC
day["dlo"]=np.minimum(dBC,dTC); day["dhi"]=np.maximum(dBC,dTC)
day["wk"]=day.index.to_period("W-FRI")
wkb=day.groupby("wk").agg(H=("h","max"),L=("l","min"),C=("cl","last"))
pH,pL,pC=wkb["H"].shift(1),wkb["L"].shift(1),wkb["C"].shift(1)
P=(pH+pL+pC)/3;BC=(pH+pL)/2;TC=2*P-BC
wkb["wlo"]=np.minimum(BC,TC);wkb["whi"]=np.maximum(BC,TC)
day["wlo"]=day["wk"].map(wkb["wlo"]);day["whi"]=day["wk"].map(wkb["whi"]);day["wkclose"]=day["wk"].map(wkb["C"])
day["didx"]=day.groupby("wk").cumcount()
def sd(px,lo,hi): return "above" if px>hi else("below" if px<lo else "inside")
day["wk_side"]=[sd(a,b,c2) for a,b,c2 in zip(day["f30"],day["wlo"],day["whi"])]   # 09:45 vs weekly
day["dy_side"]=[sd(a,b,c2) for a,b,c2 in zip(day["f30"],day["dlo"],day["dhi"])]   # 09:45 vs daily
day["cl_side"]=[sd(a,b,c2) for a,b,c2 in zip(day["wkclose"],day["wlo"],day["whi"])]
D=day.dropna(subset=["wlo","dlo"]).copy()
lab={0:"Mon",1:"Tue",2:"Wed",3:"Thu"}
print("MORNING (each day 1st-30min @09:45) intra-week re-check vs DAY-CLOSE version")
print(f"{'Day(09:45)':10s} {'weekly':7s} {'daily':7s} {'n':>4} {'rest holds side':>16}")
for di in range(4):
    sub=D[D["didx"]==di]
    for ws in ("above","below"):
        for dyv in ("above","below"):
            s=sub[(sub["wk_side"]==ws)&(sub["dy_side"]==dyv)]
            if len(s)<8: continue
            held=(s["cl_side"]==ws).mean()*100
            print(f"{lab[di]:10s} {ws:7s} {dyv:7s} {len(s):>4} {held:>15.0f}%")
    print()
# how often does the 09:45 read flip the weekly side during the week
flip=D.groupby("wk")["wk_side"].nunique()
print(f"weeks where the 09:45 weekly-side flips intra-week: {(flip>1).mean()*100:.0f}%")
