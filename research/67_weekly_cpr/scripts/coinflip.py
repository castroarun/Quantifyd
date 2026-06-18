"""Coin-flip weeks (weekly CPR & daily CPR DISAGREE at Mon 09:45) — comprehensive structure-planning stats.
Split by candle close ABOVE vs BELOW the weekly CPR. Max bull/bear excursion FROM ENTRY (f30 close), and
weekly pivot-level hit rates (R1/R2/S1/S2 from prior week) to place condor/fly/jade wings. NIFTY 5min 2015-26.
Contrast vs the CONFLUENCE weeks (both agree). Causal."""
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
    return pd.Series({"open":o,"high":h,"low":l,"close":cl,"d0":d0,"f30":(f["close"].iloc[-1] if len(f) else o)})
wk=df.groupby(pd.Grouper(freq="W-FRI")).apply(pw).dropna()
H,L,C=wk["high"],wk["low"],wk["close"];pH,pL,pC=H.shift(1),L.shift(1),C.shift(1)
P=(pH+pL+pC)/3;BC=(pH+pL)/2;TC=2*P-BC
wk["wlo"]=np.minimum(BC,TC);wk["whi"]=np.maximum(BC,TC)
wk["R1"]=2*P-pL; wk["S1"]=2*P-pH; wk["R2"]=P+(pH-pL); wk["S2"]=P-(pH-pL)
wk["dlo"]=wk["d0"].map(dd["dlo"]); wk["dhi"]=wk["d0"].map(dd["dhi"])
def sd(px,lo,hi): return "above" if px>hi else("below" if px<lo else "inside")
wk["wk_side"]=[sd(a,b,c2) for a,b,c2 in zip(wk["f30"],wk["wlo"],wk["whi"])]
wk["dy_side"]=[sd(a,b,c2) for a,b,c2 in zip(wk["f30"],wk["dlo"],wk["dhi"])]
wk["maxbull"]=(wk["high"]-wk["f30"])/wk["f30"]*100      # max up excursion FROM ENTRY (f30)
wk["maxbear"]=(wk["f30"]-wk["low"])/wk["f30"]*100       # max down excursion FROM ENTRY
wk["netf"]=(wk["close"]-wk["f30"])/wk["f30"]*100
wk["hitR1"]=(H>=wk["R1"]); wk["hitR2"]=(H>=wk["R2"]); wk["hitS1"]=(L<=wk["S1"]); wk["hitS2"]=(L<=wk["S2"])
W=wk.dropna(subset=["wlo","dlo","S2"]).copy()

def grp(name, mask):
    s=W[mask]
    if len(s)<5: print(f"{name:32s} n={len(s)} (thin)"); return
    print(f"{name:32s} {len(s):>4} | {s['netf'].mean():>+6.2f} | "
          f"{s['maxbull'].mean():>4.2f}/{s['maxbull'].quantile(.9):>4.2f} | {s['maxbear'].mean():>4.2f}/{s['maxbear'].quantile(.9):>4.2f} | "
          f"{s['hitR1'].mean()*100:>3.0f}% {s['hitR2'].mean()*100:>3.0f}% | {s['hitS1'].mean()*100:>3.0f}% {s['hitS2'].mean()*100:>3.0f}%")

print("group                            n   | netf% | maxBull avg/p90 | maxBear avg/p90 | R1%  R2% | S1%  S2%")
print("-- CONFLUENCE (both timeframes agree) --")
grp("ABOVE conf (wk+ & daily+)", (W.wk_side=="above")&(W.dy_side=="above"))
grp("BELOW conf (wk- & daily-)", (W.wk_side=="below")&(W.dy_side=="below"))
print("-- COIN-FLIP (weekly & daily DISAGREE) --")
grp("ABOVE coinflip (wk+ daily not+)", (W.wk_side=="above")&(W.dy_side!="above"))
grp("BELOW coinflip (wk- daily not-)", (W.wk_side=="below")&(W.dy_side!="below"))
print("\nKEY (your ask): in ABOVE-CPR coin-flip -> S1 hit% (downside risk); BELOW-CPR coin-flip -> R1 hit% (upside risk)")
ca=W[(W.wk_side=="above")&(W.dy_side!="above")]; cb=W[(W.wk_side=="below")&(W.dy_side!="below")]
print(f"  ABOVE coin-flip: S1 met {ca['hitS1'].mean()*100:.0f}% | R1 met {ca['hitR1'].mean()*100:.0f}% | both-sides whip (R1&S1) {((ca['hitR1'])&(ca['hitS1'])).mean()*100:.0f}%")
print(f"  BELOW coin-flip: R1 met {cb['hitR1'].mean()*100:.0f}% | S1 met {cb['hitS1'].mean()*100:.0f}% | both-sides whip (R1&S1) {((cb['hitR1'])&(cb['hitS1'])).mean()*100:.0f}%")
