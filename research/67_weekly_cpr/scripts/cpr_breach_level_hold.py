"""After a WEEKLY CPR breach (a daily close crosses the band), does the next pivot (S1 down / R1 up) HOLD
or BREAK? Helps the trader who exits on the CPR breach decide what to expect next. NIFTY 5min->daily,
weekly CPR + pivots from prior week, 2015-2026."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
df["d"]=df.index.normalize()
dd=df.groupby("d").agg(h=("high","max"),l=("low","min"),cl=("close","last"))
dd["wk"]=dd.index.to_period("W-FRI")
wk=dd.groupby("wk").agg(H=("h","max"),L=("l","min"),C=("cl","last"))
pH,pL,pC=wk["H"].shift(1),wk["L"].shift(1),wk["C"].shift(1)
P=(pH+pL+pC)/3;BC=(pH+pL)/2;TC=2*P-BC
wk["wlo"]=np.minimum(BC,TC);wk["whi"]=np.maximum(BC,TC)
wk["R1"]=2*P-pL; wk["S1"]=2*P-pH; wk["R2"]=P+(pH-pL); wk["S2"]=P-(pH-pL)
wk=wk.dropna(subset=["wlo","S1"])
dd=dd.join(wk[["wlo","whi","S1","S2","R1","R2"]],on="wk")
rows=[]
for w,g in dd.dropna(subset=["wlo"]).groupby("wk"):
    wlo,whi,S1,S2,R1,R2=g[["wlo","whi","S1","S2","R1","R2"]].iloc[0]
    lo,hi,clo=g["l"].min(),g["h"].max(),g["cl"].iloc[-1]
    dn_breach=(g["cl"]<wlo).any(); up_breach=(g["cl"]>whi).any()
    rows.append(dict(dn=dn_breach,up=up_breach,lo=lo,hi=hi,clo=clo,S1=S1,S2=S2,R1=R1,R2=R2))
T=pd.DataFrame(rows)
dn=T[T.dn]; up=T[T.up]
print(f"weeks total {len(T)} | down-breach (a daily close < weekly CPR) {len(dn)} | up-breach {len(up)}\n")
print("=== AFTER a DOWN-breach of the weekly CPR — where does it go, does S1 hold? ===")
reachS1=(dn.lo<=dn.S1); reachS2=(dn.lo<=dn.S2); closeBelowS1=(dn.clo<dn.S1)
print(f"  reached S1 (low <= S1):        {reachS1.mean()*100:>3.0f}%   -> {100-reachS1.mean()*100:.0f}% STALLED before S1 (S1 never tested)")
print(f"  of those that reached S1, BROKE to S2: {(reachS2[reachS1]).mean()*100:>3.0f}%   (so S1 HELD the extension {100-(reachS2[reachS1]).mean()*100:.0f}%)")
print(f"  week CLOSED below S1:          {closeBelowS1.mean()*100:>3.0f}%")
print(f"  -> S1 'holds' (week does NOT close below S1): {100-closeBelowS1.mean()*100:.0f}%")
print("\n=== AFTER an UP-breach of the weekly CPR — does R1 hold? ===")
reachR1=(up.hi>=up.R1); reachR2=(up.hi>=up.R2); closeAboveR1=(up.clo>up.R1)
print(f"  reached R1 (high >= R1):       {reachR1.mean()*100:>3.0f}%   -> {100-reachR1.mean()*100:.0f}% STALLED before R1")
print(f"  of those that reached R1, BROKE to R2: {(reachR2[reachR1]).mean()*100:>3.0f}%   (R1 HELD {100-(reachR2[reachR1]).mean()*100:.0f}%)")
print(f"  week CLOSED above R1:          {closeAboveR1.mean()*100:>3.0f}%")
print(f"  -> R1 'holds' (week does NOT close above R1): {100-closeAboveR1.mean()*100:.0f}%")
