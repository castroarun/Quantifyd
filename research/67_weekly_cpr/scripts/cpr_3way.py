"""Does combining DAILY-CPR confluence + CANDLE COLOR on top of the weekly-CPR break improve predictability?
Progression: weekly-alone -> +daily confluence -> +candle color -> all three. NIFTY 5min->weekly 2015-26."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
df["hhmm"]=df.index.strftime("%H:%M"); df["d"]=df.index.normalize()
dd=df.groupby("d").agg(o=("open","first"),h=("high","max"),l=("low","min"),cl=("close","last"))
dpH,dpL,dpC=dd["h"].shift(1),dd["l"].shift(1),dd["cl"].shift(1)
dP=(dpH+dpL+dpC)/3; dBC=(dpH+dpL)/2; dTC=2*dP-dBC
dd["dlo"]=np.minimum(dBC,dTC); dd["dhi"]=np.maximum(dBC,dTC)
def pw(g):
    o=g["open"].iloc[0];h=g["high"].max();l=g["low"].min();cl=g["close"].iloc[-1];d0=g["d"].min()
    day0=g[g["d"]==d0]; f=day0[day0["hhmm"]<="09:40"]
    return pd.Series({"open":o,"high":h,"low":l,"close":cl,"d0":d0,"f30o":day0["open"].iloc[0],"f30":(f["close"].iloc[-1] if len(f) else o)})
wk=df.groupby(pd.Grouper(freq="W-FRI")).apply(pw).dropna()
H,L,C=wk["high"],wk["low"],wk["close"];pH,pL,pC=H.shift(1),L.shift(1),C.shift(1)
P=(pH+pL+pC)/3;BC=(pH+pL)/2;TC=2*P-BC
wk["wlo"]=np.minimum(BC,TC);wk["whi"]=np.maximum(BC,TC)
wk["dlo"]=wk["d0"].map(dd["dlo"]); wk["dhi"]=wk["d0"].map(dd["dhi"])
def sd(px,lo,hi): return "above" if px>hi else("below" if px<lo else "inside")
wk["wk"]=[sd(a,b,c2) for a,b,c2 in zip(wk["f30"],wk["wlo"],wk["whi"])]
wk["dy"]=[sd(a,b,c2) for a,b,c2 in zip(wk["f30"],wk["dlo"],wk["dhi"])]
wk["cl_"]=[sd(a,b,c2) for a,b,c2 in zip(wk["close"],wk["wlo"],wk["whi"])]
wk["green"]=wk["f30"]>wk["f30o"]
wk["net"]=(C-wk["open"])/wk["open"]*100
wk["mb"]=(H-wk["f30"])/wk["f30"]*100; wk["mr"]=(wk["f30"]-L)/wk["f30"]*100
W=wk.dropna(subset=["wlo","dlo"]).copy()

def row(name,mask,side):
    s=W[mask]
    if len(s)<8: print(f"{name:46s} n={len(s):>3} (thin)"); return
    want="above" if side=="bull" else "below"
    held=(s["cl_"]==want).mean()*100
    print(f"{name:46s} {len(s):>4} | held {held:>3.0f}% | net {s['net'].mean():>+5.2f}% | maxBull {s['mb'].mean():>4.2f} | maxBear {s['mr'].mean():>4.2f}")

print("===== BULLISH progression (weekly close ABOVE band) — FULL 2015-26 =====")
row("1. weekly above (alone)", W.wk=="above","bull")
row("2. + GREEN candle", (W.wk=="above")&W.green,"bull")
row("3. + daily confluence (daily also above)", (W.wk=="above")&(W.dy=="above"),"bull")
row("4. + GREEN + daily confluence (ALL THREE)", (W.wk=="above")&W.green&(W.dy=="above"),"bull")
row("   [contrast] above + RED + daily conf", (W.wk=="above")&(~W.green)&(W.dy=="above"),"bull")
row("   [contrast] above + GREEN + daily DISagree", (W.wk=="above")&W.green&(W.dy!="above"),"bull")
print("\n===== BEARISH progression (weekly close BELOW band) — FULL 2015-26 =====")
row("1. weekly below (alone)", W.wk=="below","bear")
row("2. + RED candle", (W.wk=="below")&(~W.green),"bear")
row("3. + daily confluence (daily also below)", (W.wk=="below")&(W.dy=="below"),"bear")
row("4. + RED + daily confluence (ALL THREE)", (W.wk=="below")&(~W.green)&(W.dy=="below"),"bear")
row("   [contrast] below + GREEN + daily conf", (W.wk=="below")&W.green&(W.dy=="below"),"bear")
