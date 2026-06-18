"""Validate the refined view: (1) TOO-narrow weekly CPR -> volatile/whippy week (skip);
(2) 1st-30min candle POSITION (above/below weekly CPR) x COLOR (green=close>open / red) ->
    same-side -> strong bias, opposite -> mild/neutral. Within NARROW-CPR weeks. NIFTY 5min 2015-26."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
df["hhmm"]=df.index.strftime("%H:%M"); df["d"]=df.index.normalize()
def pw(g):
    o=g["open"].iloc[0];h=g["high"].max();l=g["low"].min();cl=g["close"].iloc[-1];d0=g["d"].min()
    day0=g[g["d"]==d0]; f=day0[day0["hhmm"]<="09:40"]
    fo=day0["open"].iloc[0]; fc=(f["close"].iloc[-1] if len(f) else o)
    return pd.Series({"open":o,"high":h,"low":l,"close":cl,"f30o":fo,"f30c":fc})
wk=df.groupby(pd.Grouper(freq="W-FRI")).apply(pw).dropna()
H,L,C,O=wk["high"],wk["low"],wk["close"],wk["open"];pH,pL,pC=H.shift(1),L.shift(1),C.shift(1)
P=(pH+pL+pC)/3;BC=(pH+pL)/2;TC=2*P-BC
wk["lo"]=np.minimum(BC,TC);wk["hi"]=np.maximum(BC,TC);wk["cprw"]=(TC-BC).abs()/pC*100
def sd(px,lo,hi): return "above" if px>hi else("below" if px<lo else "inside")
wk["pos"]=[sd(a,b,c2) for a,b,c2 in zip(wk["f30c"],wk["lo"],wk["hi"])]
wk["color"]=np.where(wk["f30c"]>wk["f30o"],"green","red")
wk["clpos"]=[sd(a,b,c2) for a,b,c2 in zip(wk["close"],wk["lo"],wk["hi"])]
wk["net"]=(C-O)/O*100
wk["range"]=(H-L)/pC*100
wk["trend_eff"]=(C-O).abs()/(H-L)
wk["whip"]=((H>wk["hi"])&(L<wk["lo"])).astype(float)   # touched BOTH sides of the band = whippy
W=wk.dropna(subset=["cprw"]).copy()

print("===== (1) TOO-NARROW = volatile? weekly CPR width deciles (FULL 2015-26) =====")
W["dec"]=pd.qcut(W["cprw"],10,labels=False)
print(f"{'decile':7s} {'CPRw band':>14} {'n':>4} | {'range%':>7} {'trend_eff':>9} {'whip%(both sides)':>17} {'|net|%':>7}")
for q in range(10):
    s=W[W["dec"]==q]
    print(f"D{q+1:<6d} {s['cprw'].min():>5.2f}-{s['cprw'].max():<6.2f} {len(s):>4} | {s['range'].mean():>6.2f} | {s['trend_eff'].mean():>9.3f} | {s['whip'].mean()*100:>16.0f}% | {s['net'].abs().mean():>6.2f}")

def combos(d,label):
    print(f"\n===== (2) position x candle-color, NARROW-CPR weeks — {label}  n={len(d)} =====")
    print(f"{'1st-30min candle':28s} {'n':>4} | {'held its side at wk close':>25} {'net% signed':>12}")
    specs=[("above","green","ABOVE + GREEN (your: strong bull)"),("above","red","ABOVE + RED (your: mild/neutral)"),
           ("below","red","BELOW + RED (your: strong bear)"),("below","green","BELOW + GREEN (your: mild/neutral)")]
    for pos,col,lab in specs:
        s=d[(d["pos"]==pos)&(d["color"]==col)]
        if len(s)<5: print(f"{lab:28s} {len(s):>4} | (thin)"); continue
        want="above" if pos=="above" else "below"
        held=(s["clpos"]==want).mean()*100
        print(f"{lab:28s} {len(s):>4} | {held:>24.0f}% {s['net'].mean():>+11.2f}")

narrow=W[W["cprw"]<=W["cprw"].quantile(0.40)]   # narrow = bottom 40%
combos(narrow,"NARROW (bottom 40%) FULL 2015-26")
combos(W,"ALL weeks (any CPR) FULL 2015-26 — for contrast")
nr=narrow[narrow.index.year>=2023]
combos(nr,"NARROW (bottom 40%) RECENT 2023-26 (thin)")
