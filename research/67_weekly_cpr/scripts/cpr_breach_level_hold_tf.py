"""After a weekly CPR breach, does S1/R1 HOLD on a 30m / 1h / 2h / daily / weekly CLOSING basis?
'Holds' = NO candle of that timeframe CLOSES beyond the level (S1 down / R1 up) during the week.
Finer TF = more candle closes = more breaks registered = lower hold rate. NIFTY 5min 2015-2026."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
ORIG=df.index[0]
dd=df.resample("D",origin=ORIG).agg(h=("high","max"),l=("low","min"),cl=("close","last")).dropna()
dd["wk"]=dd.index.to_period("W-FRI")
wk=dd.groupby("wk").agg(H=("h","max"),L=("l","min"),C=("cl","last"))
pH,pL,pC=wk["H"].shift(1),wk["L"].shift(1),wk["C"].shift(1)
P=(pH+pL+pC)/3;BC=(pH+pL)/2;TC=2*P-BC
wk["wlo"]=np.minimum(BC,TC);wk["whi"]=np.maximum(BC,TC); wk["S1"]=2*P-pH; wk["R1"]=2*P-pL
wk=wk.dropna(subset=["wlo","S1"])
# which weeks breached (daily close beyond the band)
dd=dd.join(wk[["wlo","whi","S1","R1"]],on="wk")
brk=dd.dropna(subset=["wlo"]).groupby("wk").apply(lambda g: pd.Series({"dn":(g["cl"]<g["wlo"]).any(),"up":(g["cl"]>g["whi"]).any()}))
DN=set(brk[brk.dn].index); UP=set(brk[brk.up].index)
S1=wk["S1"].to_dict(); R1=wk["R1"].to_dict()
def holdrate(freq):
    if freq=="W": b=df.resample("W-FRI",origin=ORIG).agg(cl=("close","last")).dropna(); b["wk"]=b.index.to_period("W-FRI")
    else: b=df.resample(freq,origin=ORIG).agg(cl=("close","last")).dropna(); b["wk"]=b.index.to_period("W-FRI")
    # per week: did any candle close below S1 (down) / above R1 (up)?
    dn_break={}; up_break={}
    for w,g in b.groupby("wk"):
        if w in S1: dn_break[w]=(g["cl"]<S1[w]).any(); up_break[w]=(g["cl"]>R1[w]).any()
    dn_hold=np.mean([not dn_break[w] for w in DN if w in dn_break])*100
    up_hold=np.mean([not up_break[w] for w in UP if w in up_break])*100
    return dn_hold,up_hold
print(f"down-breach weeks {len(DN)} | up-breach weeks {len(UP)}\n")
print(f"{'closing TF':10s} {'S1 holds (down)':>16} {'R1 holds (up)':>14}")
for nm,fr in [("30-min","30min"),("1h","60min"),("2h","120min"),("daily","D"),("weekly","W")]:
    dh,uh=holdrate(fr)
    print(f"{nm:10s} {dh:>15.0f}% {uh:>13.0f}%")
print("\n'Holds' = no candle of that TF closes beyond S1/R1 all week. Finer TF -> stricter -> lower hold.")
