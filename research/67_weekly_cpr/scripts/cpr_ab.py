"""(a) Weekly CPR as a FLY filter (proxy: 2%/1.5% move-stop breach rate from week open, by prev-week
        CPR quintile). NOTE: signal-only (breach/calm proxy) — real ₹ needs option premiums (AlgoTest).
   (b) DAILY CPR sign check: does narrow prior-DAY CPR predict a CALM next day (our live gate assumes
        narrow=calm)? Compare sign to the weekly (narrow=trend) finding."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")

def cpr_study(bars, freq, label, breach_levels):
    b=bars.resample(freq).agg(open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last")).dropna()
    H,L,C,O=b["high"],b["low"],b["close"],b["open"]; pH,pL,pC=H.shift(1),L.shift(1),C.shift(1)
    P=(pH+pL+pC)/3; BC=(pH+pL)/2; TC=2*P-BC
    b["cprw"]=(TC-BC).abs()/pC*100
    b["net"]=(C-O).abs()/O*100; b["trend_eff"]=(C-O).abs()/(H-L)
    b["maxside"]=np.maximum((H-O),(O-L))/O*100
    b["inside"]=((C>=np.minimum(BC,TC))&(C<=np.maximum(BC,TC))).astype(float)
    b=b.dropna(subset=["cprw"]).copy(); b["q"]=pd.qcut(b["cprw"],5,labels=False)
    print(f"\n=== {label}  n={len(b)}  corr(CPRw,net)={b['cprw'].corr(b['net']):+.3f}  corr(CPRw,trendEff)={b['cprw'].corr(b['trend_eff']):+.3f} ===")
    hdr=" ".join(f"breach>{x}%" for x in breach_levels)
    print(f"{'bucket':8s} {'CPRw band':>14} {'n':>4} | {'net abs%':>8} {'trendEff':>8} {'inside%':>7} | {hdr}")
    for q in range(5):
        s=b[b["q"]==q]; nm=["Q1 narrow","Q2","Q3","Q4","Q5 WIDE"][q]
        br=" ".join(f"{(s['maxside']>=x).mean()*100:>7.0f}%" for x in breach_levels)
        print(f"{nm:8s} {s['cprw'].min():>5.2f}-{s['cprw'].max():<6.2f} {len(s):>4} | {s['net'].mean():>7.2f} | {s['trend_eff'].mean():>7.3f} | {s['inside'].mean()*100:>6.0f}% | {br}")
    return b

print("########## (a) WEEKLY CPR as fly filter — breach/calm proxy ##########")
wkF=cpr_study(df,"W-FRI","WEEKLY FULL 2015-2026",[1.5,2,3])
wkR=cpr_study(df[df.index.year>=2023],"W-FRI","WEEKLY RECENT 2023-2026",[1.5,2,3])
print("\n  -> fly calm-rate = 100 - breach%. WIDE weekly CPR should give a HIGHER calm-rate (lower breach) if classic holds.")

print("\n########## (b) DAILY CPR sign check (our live gate assumes narrow=calm) ##########")
dyF=cpr_study(df,"D","DAILY FULL 2015-2026",[0.5,1,1.5])
dyR=cpr_study(df[df.index.year>=2023],"D","DAILY RECENT 2023-2026",[0.5,1,1.5])
print("\n  -> If DAILY shows narrow->LOW movement (calm), our gate sign is RIGHT. If narrow->HIGH movement (like weekly), the gate is BACKWARDS.")
