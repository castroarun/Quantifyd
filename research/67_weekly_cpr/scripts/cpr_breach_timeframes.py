"""Breach-trigger reliability vs candle timeframe (30m/1h/2h/3h/4h/daily). Entry = 1st-30-min vs weekly CPR.
Trigger = any LATER candle (excluding the week's opening candle at that TF) closing below CPR / below S1
(above entries) or above CPR / R1 (below entries). Outcome = week close vs CPR band & S1/R1. NIFTY 2015-26.
All TFs aligned to 09:15 via origin (24h divisible by each TF -> daily 09:15 alignment)."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
ORIG=df.index[0]   # a 09:15 -> all TFs align to 09:15 each day
# weekly band + S1/R1 (period-keyed)
w=df.resample("W-FRI").agg(H=("high","max"),L=("low","min"),C=("close","last")).dropna()
pH,pL,pC=w["H"].shift(1),w["L"].shift(1),w["C"].shift(1)
P=(pH+pL+pC)/3;BC=(pH+pL)/2;TC=2*P-BC
w["lo"]=np.minimum(BC,TC);w["hi"]=np.maximum(BC,TC);w["R1"]=2*P-pL;w["S1"]=2*P-pH;w["wc"]=w["C"]
w=w.dropna(subset=["lo","S1"]); w.index=w.index.to_period("W-FRI")
wp={p:(r.lo,r.hi,r.S1,r.R1,r.wc) for p,r in w.iterrows()}
# entry side from 1st 30-min candle
m30=df.resample("30min",origin=ORIG).agg(cl=("close","last")).dropna(); m30["wk"]=m30.index.to_period("W-FRI")
entry={}
for wk,g in m30.groupby("wk"):
    if wk in wp:
        lo,hi,_,_,_=wp[wk]; ec=g["cl"].iloc[0]
        entry[wk]="above" if ec>hi else ("below" if ec<lo else "inside")

def run(freq):
    b=df.resample(freq,origin=ORIG).agg(cl=("close","last")).dropna(); b["wk"]=b.index.to_period("W-FRI")
    A=[];B=[]
    for wk,g in b.groupby("wk"):
        if wk not in wp or wk not in entry or len(g)<2: continue
        lo,hi,S1,R1,wc=wp[wk]; e=entry[wk]; later=g["cl"].iloc[1:]   # exclude opening candle at this TF
        if e=="above": A.append((later.lt(lo).any(),later.lt(S1).any(),wc<lo,wc<S1))
        elif e=="below": B.append((later.gt(hi).any(),later.gt(R1).any(),wc>hi,wc>R1))
    return pd.DataFrame(A,columns=["x","b","el","es"]), pd.DataFrame(B,columns=["x","b","eh","er"])

tfs=[("30min","30min"),("1h","60min"),("2h","120min"),("3h","180min"),("4h","240min"),("daily","1D")]
print("ABOVE entries (bull) — LATER candle closes below S1 (the strong trigger) / below CPR (early):")
print(f"{'TF':6s} {'n':>4} | {'S1-breach occ% / endCPR% / endS1%':>34} | {'CPR-cross occ% / endCPR%':>26}")
for name,fr in tfs:
    A,_=run(fr); ab=A[A.b]; ax=A[A.x]
    s1=f"{len(ab)/len(A)*100:>3.0f}% / {ab.el.mean()*100:>3.0f}% / {ab.es.mean()*100:>3.0f}%" if len(ab) else "-"
    cx=f"{len(ax)/len(A)*100:>3.0f}% / {ax.el.mean()*100:>3.0f}%" if len(ax) else "-"
    print(f"{name:6s} {len(A):>4} | {s1:>34} | {cx:>26}")
print(f"  baseline ABOVE-entry ends below CPR ~20%")
print("\nBELOW entries (bear) — LATER candle closes above R1 / above CPR:")
print(f"{'TF':6s} {'n':>4} | {'R1-breach occ% / endCPR% / endR1%':>34} | {'CPR-cross occ% / endCPR%':>26}")
for name,fr in tfs:
    _,B=run(fr); br=B[B.b]; bx=B[B.x]
    r1=f"{len(br)/len(B)*100:>3.0f}% / {br.eh.mean()*100:>3.0f}% / {br.er.mean()*100:>3.0f}%" if len(br) else "-"
    cx=f"{len(bx)/len(B)*100:>3.0f}% / {bx.eh.mean()*100:>3.0f}%" if len(bx) else "-"
    print(f"{name:6s} {len(B):>4} | {r1:>34} | {cx:>26}")
print(f"  baseline BELOW-entry ends above CPR ~32%")
