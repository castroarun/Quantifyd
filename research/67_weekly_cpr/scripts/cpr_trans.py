"""Why transition n was thin + a robust pooled estimate. We have 11y, but coin-flips are ~10% of weeks,
so 'Mon-coinflip -> Wed-confluence' is a rare sub-cell. Pool ALL resolutions: weeks NOT confluence on Mon
that LATER form confluence -> measure rest-of-week hold from the resolution day. Pool both sides/all days."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
df["d"]=df.index.normalize()
dd=df.groupby("d").agg(h=("high","max"),l=("low","min"),cl=("close","last"))
dpH,dpL,dpC=dd["h"].shift(1),dd["l"].shift(1),dd["cl"].shift(1)
dP=(dpH+dpL+dpC)/3; dBC=(dpH+dpL)/2; dTC=2*dP-dBC
dd["dlo"]=np.minimum(dBC,dTC); dd["dhi"]=np.maximum(dBC,dTC)
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

# how common is each Monday state? (shows coin-flips are rare -> why the sub-cell was thin)
mon=D[D["didx"]==0]
print("Monday state mix (why the transition cell is small):")
print(f"  confluence {mon['conf'].mean()*100:.0f}%  | coin-flip/inside {100-mon['conf'].mean()*100:.0f}%  (total Mondays n={len(mon)})")

# POOLED transition: weeks NOT confluence on Monday, that LATER form confluence (any day Tue-Thu, either side)
rows=[]
for wk,g in D.groupby("wk"):
    g=g.sort_values("didx")
    if g.empty or g.iloc[0]["conf"]: continue          # only weeks that started NON-confluence
    later=g[(g["didx"]>=1)&(g["conf"])]
    if later.empty: continue
    r=later.iloc[0]                                     # first day confluence forms
    rows.append({"side":r["wk_side"],"held":r["cl_side"]==r["wk_side"],"rest":r["rest"],"didx":r["didx"]})
T=pd.DataFrame(rows)
print(f"\nPOOLED 'resolves to confluence after a non-confluence Monday' (all days/both sides): n={len(T)}")
print(f"  rest-of-week HOLDS the new confluence side: {T['held'].mean()*100:.0f}%   rest net avg {T['rest'].mean():+.2f}%")
for sidev in ("above","below"):
    s=T[T["side"]==sidev]
    print(f"   {sidev}: n={len(s)}  holds {s['held'].mean()*100:.0f}%  net {s['rest'].mean():+.2f}%")

# for contrast: confluence present on ANY day (the LARGE-n robust base — you don't need the thin transition cell)
print("\nRobust base — ANY day with confluence present -> rest-of-week holds (large n, this is the usable rule):")
for di in range(4):
    s=D[(D["didx"]==di)&(D["conf"])]
    print(f"  {['Mon','Tue','Wed','Thu'][di]}: n={len(s):>3}  holds {(s['cl_side']==s['wk_side']).mean()*100:.0f}%")
