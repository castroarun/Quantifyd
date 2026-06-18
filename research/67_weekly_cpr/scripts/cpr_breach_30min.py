"""Breach/cross trigger at 30-MIN granularity (user's intent), vs the daily-close version.
Entry = 1st 30-min candle (09:15-09:45) vs weekly CPR. Trigger = any LATER 30-min candle CLOSING
below the weekly CPR / below S1 (above entries); above CPR / R1 (below entries). Outcome = week close
vs CPR band and vs S1/R1. NIFTY 5min->30min, weekly band from prior week. 11y 2015-26."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
# 30-min candles aligned to 09:15 (offset 15min so bins are :15/:45)
m30=df.resample("30min",offset="15min").agg(o=("open","first"),h=("high","max"),l=("low","min"),cl=("close","last")).dropna()
m30["wk"]=m30.index.to_period("W-FRI")
# weekly band + S1/R1 from prior week (weekly bars from 5min)
w=df.resample("W-FRI").agg(H=("high","max"),L=("low","min"),C=("close","last")).dropna()
pH,pL,pC=w["H"].shift(1),w["L"].shift(1),w["C"].shift(1)
P=(pH+pL+pC)/3;BC=(pH+pL)/2;TC=2*P-BC
w["lo"]=np.minimum(BC,TC);w["hi"]=np.maximum(BC,TC);w["R1"]=2*P-pL;w["S1"]=2*P-pH
w["wc"]=w["C"]; w=w.dropna(subset=["lo","S1"])
wp={p:(r.lo,r.hi,r.S1,r.R1,r.wc) for p,r in w.iterrows()}

rA=[]; rB=[]
for wk,g in m30.groupby("wk"):
    if wk not in wp: continue
    lo,hi,S1,R1,wc=wp[wk]; g=g.sort_index()
    if len(g)<4: continue
    ec=g["cl"].iloc[0]                       # first 30-min candle close
    later=g.iloc[1:]
    if ec>hi:   rA.append((later["cl"].lt(lo).any(), later["cl"].lt(S1).any(), wc<lo, wc<S1))
    elif ec<lo: rB.append((later["cl"].gt(hi).any(), later["cl"].gt(R1).any(), wc>hi, wc>R1))
A=pd.DataFrame(rA,columns=["xdn","bS1","endLo","endS1"]); B=pd.DataFrame(rB,columns=["xup","bR1","endHi","endR1"])
print(f"30-MIN granularity (entry=1st 30m candle). ABOVE entries n={len(A)}, BELOW entries n={len(B)}")
print("\nABOVE entries (bull) — a LATER 30-min candle:")
e=A[A.xdn]; print(f"  closes BELOW CPR: occurs {len(e)/len(A)*100:>3.0f}% (n={len(e)}) -> week ends below CPR {e.endLo.mean()*100:>3.0f}% | below S1 {e.endS1.mean()*100:>3.0f}%")
e=A[A.bS1]; print(f"  closes BELOW S1 : occurs {len(e)/len(A)*100:>3.0f}% (n={len(e)}) -> week ends below CPR {e.endLo.mean()*100:>3.0f}% | below S1 {e.endS1.mean()*100:>3.0f}%")
print(f"  baseline ABOVE-entry ends below CPR: {(A.endLo.mean()*100):.0f}%")
print("\nBELOW entries (bear) — a LATER 30-min candle:")
e=B[B.xup]; print(f"  closes ABOVE CPR: occurs {len(e)/len(B)*100:>3.0f}% (n={len(e)}) -> week ends above CPR {e.endHi.mean()*100:>3.0f}% | above R1 {e.endR1.mean()*100:>3.0f}%")
e=B[B.bR1]; print(f"  closes ABOVE R1 : occurs {len(e)/len(B)*100:>3.0f}% (n={len(e)}) -> week ends above CPR {e.endHi.mean()*100:>3.0f}% | above R1 {e.endR1.mean()*100:>3.0f}%")
print(f"  baseline BELOW-entry ends above CPR: {(B.endHi.mean()*100):.0f}%")
print("\n(contrast — DAILY-close triggers were: above->S1 breach 88% end below CPR/68% below S1; below->R1 breach 91%/79%)")
