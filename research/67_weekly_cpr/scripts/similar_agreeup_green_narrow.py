"""Find PAST weeks matching today's fingerprint: AGREE-UP (1st-30m close above BOTH weekly & daily CPR)
+ GREEN candle + NARROW weekly CPR. Report how they played out, esp. the downside (the wide put-leg risk).
Condor frame: short call ~+2%, short put ~-2%, breakevens +2.2% / -2.0% (matching the user's trade)."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
df["d"]=df.index.normalize(); df["t"]=df.index.strftime("%H:%M")
dd=df.groupby("d").agg(h=("high","max"),l=("low","min"),cl=("close","last"))
dpH,dpL,dpC=dd["h"].shift(1),dd["l"].shift(1),dd["cl"].shift(1)
dP=(dpH+dpL+dpC)/3;dBC=(dpH+dpL)/2;dTC=2*dP-dBC
dd["dlo"]=np.minimum(dBC,dTC);dd["dhi"]=np.maximum(dBC,dTC)
def pw(g):
    o=g["open"].iloc[0];h=g["high"].max();l=g["low"].min();cl=g["close"].iloc[-1];d0=g["d"].min()
    f=g[(g["d"]==d0)&(g["t"]<="09:40")]
    return pd.Series({"open":o,"high":h,"low":l,"close":cl,"d0":d0,"f30o":g[g["d"]==d0]["open"].iloc[0],"f30":(f["close"].iloc[-1] if len(f) else o)})
wk=df.groupby(pd.Grouper(freq="W-FRI")).apply(pw).dropna()
H,L,C=wk["high"],wk["low"],wk["close"];pH,pL,pC=H.shift(1),L.shift(1),C.shift(1)
P=(pH+pL+pC)/3;BC=(pH+pL)/2;TC=2*P-BC
wk["wlo"]=np.minimum(BC,TC);wk["whi"]=np.maximum(BC,TC);wk["cprw"]=(TC-BC).abs()/pC*100
wk["S1"]=2*P-pH;wk["S2"]=P-(pH-pL);wk["R1"]=2*P-pL;wk["R2"]=P+(pH-pL)
wk["dlo"]=wk["d0"].map(dd["dlo"]);wk["dhi"]=wk["d0"].map(dd["dhi"])
def sd(px,lo,hi): return "ABOVE" if px>hi else("BELOW" if px<lo else "INSIDE")
wk["wpos"]=[sd(a,b,c2) for a,b,c2 in zip(wk["f30"],wk["wlo"],wk["whi"])]
wk["dpos"]=[sd(a,b,c2) for a,b,c2 in zip(wk["f30"],wk["dlo"],wk["dhi"])]
wk["green"]=wk["f30"]>wk["f30o"]
W=wk.dropna(subset=["wlo","dlo","S1"]).copy()
# match today's fingerprint
m=(W.wpos=="ABOVE")&(W.dpos=="ABOVE")&(W.green)
def report(sub,label):
    if len(sub)<4: print(f"\n{label}: n={len(sub)} (too thin)"); return
    e=sub["f30"]
    mu=(sub["high"]-e)/e*100; mdn=(e-sub["low"])/e*100; net=(sub["close"]-sub["open"])/sub["open"]*100
    hold=(sub["close"]>sub["whi"]).mean()*100
    cpr_dn = (sub["low"]<sub["wlo"]).mean()*100   # traded below weekly CPR (whip toward put leg)
    print(f"\n=== {label}  n={len(sub)} ===")
    print(f"  week CLOSES above weekly CPR (bull held): {hold:.0f}%   | net move avg {net.mean():+.2f}% (median {net.median():+.2f})")
    print(f"  MAX UP excursion  avg {mu.mean():.2f}% / p90 {mu.quantile(.9):.2f}%   MAX DOWN avg {mdn.mean():.2f}% / p90 {mdn.quantile(.9):.2f}%")
    print(f"  CALL side risk: up-move > +2.2% (your call BE): {(mu>2.2).mean()*100:.0f}%   reached R1 {(sub['high']>=sub['R1']).mean()*100:.0f}% / R2 {(sub['high']>=sub['R2']).mean()*100:.0f}%")
    print(f"  PUT side risk (the wide leg): down-move > 2.0% (your put BE): {(mdn>2.0).mean()*100:.0f}%   reached S1 {(sub['low']<=sub['S1']).mean()*100:.0f}% / S2 {(sub['low']<=sub['S2']).mean()*100:.0f}%")
    print(f"  WHIP: traded BELOW the weekly CPR intraweek: {cpr_dn:.0f}%  | stayed within +/-2% both sides (condor body safe): {((mu<2.2)&(mdn<2.0)).mean()*100:.0f}%")
report(W[m],"AGREE-UP + GREEN (any width)")
report(W[m & (W.cprw<=0.40)],"AGREE-UP + GREEN + NARROW CPR (<=0.40%)")
report(W[m & (W.cprw<=0.20)],"AGREE-UP + GREEN + VERY-NARROW CPR (<=0.20%, ~ today 0.11%)")
print(f"\n(baseline: all weeks within +/-2% both sides = {(((W.high-W.f30)/W.f30*100<2.2)&((W.f30-W.low)/W.f30*100<2.0)).mean()*100:.0f}%)")
