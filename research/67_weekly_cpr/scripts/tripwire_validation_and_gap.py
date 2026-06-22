"""(A) VALIDATE the trip-wire: for AGREE-UP weeks, given a candle of TF X closes BELOW the weekly CPR during
the week, what % does the week actually END below the CPR (bias flipped)? 30m vs 1h vs 2h vs daily.
(B) Today's context: AGREE-UP+green+narrow split by GAP (up/flat/down) + prior-week character. NIFTY 2015-26."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
ORIG=df.index[0]; df["d"]=df.index.normalize(); df["t"]=df.index.strftime("%H:%M")
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
wk["wlo"]=np.minimum(BC,TC);wk["whi"]=np.maximum(BC,TC);wk["cprw"]=(TC-BC).abs()/pC*100;wk["pcprw"]=wk["cprw"].shift(1)
wk["dlo"]=wk["d0"].map(dd["dlo"]);wk["dhi"]=wk["d0"].map(dd["dhi"])
def sd(px,lo,hi): return "ABOVE" if px>hi else("BELOW" if px<lo else "INSIDE")
wk["wpos"]=[sd(a,b,c2) for a,b,c2 in zip(wk["f30"],wk["wlo"],wk["whi"])]
wk["dpos"]=[sd(a,b,c2) for a,b,c2 in zip(wk["f30"],wk["dlo"],wk["dhi"])]
wk["green"]=wk["f30"]>wk["f30o"]; wk["gap"]=(wk["open"]-pC)/pC*100
W=wk.dropna(subset=["wlo","dlo"]).copy()

print("===== (A) TRIP-WIRE VALIDATION — AGREE-UP weeks: a TF candle closes BELOW weekly CPR -> does the week END below it? =====")
au=W[(W.wpos=="ABOVE")&(W.dpos=="ABOVE")]   # AGREE-UP weeks
auwk=set(au.index)
endsbelow={p:(au.loc[p,"close"]<au.loc[p,"wlo"]) for p in au.index}   # week ended below the band
lo_by={p:au.loc[p,"wlo"] for p in au.index}
print(f"AGREE-UP weeks n={len(au)} | base rate the week ends below CPR = {np.mean(list(endsbelow.values()))*100:.0f}%")
print(f"{'trigger TF':10s} {'weeks w/ a close<CPR':>20} {'-> week ENDS below CPR':>24}")
for nm,fr in [("30-min","30min"),("1h","60min"),("2h","120min"),("daily","1D")]:
    b=df.resample(fr,origin=ORIG).agg(cl=("close","last")).dropna() if fr!="1D" else df.resample("1D").agg(cl=("close","last")).dropna()
    b["wk"]=b.index.to_period("W-FRI")
    fired=[];
    for p in au.index:
        g=b[b["wk"]==p]
        if len(g) and (g["cl"]<lo_by[p]).any(): fired.append(p)
    if fired:
        rate=np.mean([endsbelow[p] for p in fired])*100
        print(f"{nm:10s} {len(fired)/len(au)*100:>18.0f}% {rate:>23.0f}%")
print("  -> a 30-min close below CPR is an EARLY/NOISY warning; the DAILY close is the validated flip.")

print("\n===== (B) AGREE-UP+GREEN+NARROW(<=0.4%) split by GAP (today gap ~+0.3% = small-up/flat) =====")
sub=W[(W.wpos=="ABOVE")&(W.dpos=="ABOVE")&(W.green)&(W.cprw<=0.40)].copy()
sub["mu"]=(sub["high"]-sub["f30"])/sub["f30"]*100; sub["mdn"]=(sub["f30"]-sub["low"])/sub["f30"]*100
sub["g"]=pd.cut(sub["gap"],[-9,-0.1,0.3,9],labels=["gap-DOWN","flat/small-up","gap-UP>0.3%"])
print(f"{'gap bucket':16s} {'n':>4} {'bull held':>10} {'net%':>6} {'putBE>2%':>9} {'whip<CPR':>9} {'body safe':>10}")
for gname,s in sub.groupby("g",observed=True):
    if len(s)<5: print(f"{str(gname):16s} {len(s):>4} (thin)"); continue
    print(f"{str(gname):16s} {len(s):>4} {(s['close']>s['whi']).mean()*100:>9.0f}% {(s['close']-s['open']).div(s['open']).mul(100).mean():>+5.2f} {(s['mdn']>2).mean()*100:>8.0f}% {(s['low']<s['wlo']).mean()*100:>8.0f}% {((s['mu']<2.2)&(s['mdn']<2)).mean()*100:>9.0f}%")
print("\n===== prior-week CPR character (was last week itself narrow?) for this setup =====")
print(f"  median prior-week CPR width on these weeks: {sub['pcprw'].median():.2f}%  | this-week narrow after a narrow prior week happens {((sub['pcprw']<=0.4)).mean()*100:.0f}% of the time")
