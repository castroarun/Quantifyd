"""Validate the CLASSIC CPR rule on NIFTY weekly (5-min->weekly, 2015-2026):
   narrow prior-week CPR -> TRENDING week ; wide prior-week CPR -> SIDEWAYS/contained week.
CPR is drawn FOR the week from the PRIOR week's H/L/C (exactly as on the chart).
Right metrics for 'trending vs sideways':
   net%      = |close-open|/open*100         (directional travel)
   trend_eff = |close-open|/(high-low)        (0=pure chop/sideways, 1=pure one-way trend)
   inside_cpr= did the week's CLOSE finish INSIDE the projected CPR band [BC,TC]? (contained)
   range%    = (high-low)/prevClose*100        (what I wrongly used before)"""
import sqlite3, numpy as np, pandas as pd
c = sqlite3.connect("backtest_data/market_data.db")
df5 = pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date", c, parse_dates=["date"]).set_index("date")
wk = df5.resample("W-FRI").agg(open=("open","first"), high=("high","max"), low=("low","min"), close=("close","last")).dropna()
H,L,C,O = wk["high"],wk["low"],wk["close"],wk["open"]
pH,pL,pC = H.shift(1),L.shift(1),C.shift(1)
# CPR drawn FOR this week from prior week
P  = (pH+pL+pC)/3; BC=(pH+pL)/2; TC=2*P-BC
lo_cpr, hi_cpr = np.minimum(BC,TC), np.maximum(BC,TC)
wk["prev_cpr"] = (TC-BC).abs()/pC*100
wk["range"]    = (H-L)/pC*100
wk["net"]      = (C-O).abs()/O*100
wk["trend_eff"]= (C-O).abs()/(H-L)
wk["close_inside_cpr"] = ((C>=lo_cpr)&(C<=hi_cpr)).astype(float)   # finished contained within the band
W = wk.dropna(subset=["prev_cpr","net"]).copy()

def buckets(d, label):
    d = d.copy(); d["q"]=pd.qcut(d["prev_cpr"],5,labels=["Q1 narrow","Q2","Q3","Q4","Q5 WIDE"])
    print(f"\n=== {label}  (n={len(d)}) ===")
    print(f"  corr(prevCPR, net move)   = {d['prev_cpr'].corr(d['net']):+.3f}")
    print(f"  corr(prevCPR, trend_eff)  = {d['prev_cpr'].corr(d['trend_eff']):+.3f}")
    print(f"  corr(prevCPR, range)      = {d['prev_cpr'].corr(d['range']):+.3f}")
    print(f"  {'bucket':10s} {'cpr%':>6} | {'net%':>10} | {'trend_eff':>10} | {'range%':>8} | {'closeInCPR%':>11}")
    for q,s in d.groupby("q",observed=True):
        print(f"  {q:10s} {s['prev_cpr'].mean():>6.2f} | {s['net'].mean():>5.2f} (med {s['net'].median():>4.2f}) | "
              f"{s['trend_eff'].mean():>10.3f} | {s['range'].mean():>8.2f} | {s['close_inside_cpr'].mean()*100:>10.0f}%")

buckets(W, "FULL 2015-2026")
buckets(W[W.index.year>=2023], "RECENT 2023-2026")
