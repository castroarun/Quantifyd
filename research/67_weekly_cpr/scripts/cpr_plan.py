"""Trade-planning stats for weekly CPR on NIFTY (5-min, focus 2023-2026).
CPR drawn FOR the week from the PRIOR week's H/L/C (as on chart). For each week we measure, off the
intraday 5-min path: net move, MAX one-side excursion, where the 1st-30min closed vs the CPR band,
and whether the week HELD that side or crossed back through the CPR."""
import sqlite3, numpy as np, pandas as pd
c = sqlite3.connect("backtest_data/market_data.db")
df = pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date", c, parse_dates=["date"]).set_index("date")
df["hhmm"] = df.index.strftime("%H:%M"); df["d"] = df.index.normalize()

def per_week(g):
    o=g["open"].iloc[0]; h=g["high"].max(); l=g["low"].min(); cl=g["close"].iloc[-1]
    d0=g["d"].min(); day0=g[g["d"]==d0]
    f30=day0[day0["hhmm"]<="09:40"]               # 1st 30 min = 09:15..09:45 -> last bar starts 09:40
    f30c=f30["close"].iloc[-1] if len(f30) else o
    return pd.Series({"open":o,"high":h,"low":l,"close":cl,"f30":f30c})

wk=df.groupby(pd.Grouper(freq="W-FRI")).apply(per_week).dropna()
pH,pL,pC=wk["high"].shift(1),wk["low"].shift(1),wk["close"].shift(1)
P=(pH+pL+pC)/3; BC=(pH+pL)/2; TC=2*P-BC
wk["lo"]=np.minimum(BC,TC); wk["hi"]=np.maximum(BC,TC); wk["piv"]=P
wk["cprw"]=(TC-BC).abs()/pC*100
wk["net"]=(wk["close"]-wk["open"])/wk["open"]*100
wk["up"]=(wk["high"]-wk["open"])/wk["open"]*100
wk["dn"]=(wk["open"]-wk["low"])/wk["open"]*100
wk["maxside"]=np.maximum(wk["up"],wk["dn"])
wk["range"]=(wk["high"]-wk["low"])/pC*100
def side(px,lo,hi): return "above" if px>hi else ("below" if px<lo else "inside")
wk["f30side"]=[side(a,b,c2) for a,b,c2 in zip(wk["f30"],wk["lo"],wk["hi"])]
wk["clside"]=[side(a,b,c2) for a,b,c2 in zip(wk["close"],wk["lo"],wk["hi"])]
# crossed = price visited the OPPOSITE side of the band intraweek
wk["touch_below"]=wk["low"]<wk["lo"]; wk["touch_above"]=wk["high"]>wk["hi"]
W=wk.dropna(subset=["cprw"]).copy()
R=W[W.index.year>=2023].copy()

def qtable(d,label):
    d=d.copy(); d["q"]=pd.qcut(d["cprw"],5,labels=False)
    print(f"\n=== {label}  n={len(d)} ===")
    print(f"{'bucket':8s} {'CPRwidth% band':>16} {'n':>4} | {'net%(avg|abs)':>14} {'maxSide% avg/med/p90':>20} {'range%avg':>9} {'closeInCPR%':>11}")
    for q in range(5):
        s=d[d["q"]==q]; nm=["Q1 narrow","Q2","Q3","Q4","Q5 WIDE"][q]
        print(f"{nm:8s} {s['cprw'].min():>6.2f}-{s['cprw'].max():<6.2f} {len(s):>4} | "
              f"{s['net'].mean():>+6.2f} | {s['net'].abs().mean():>5.2f} | "
              f"{s['maxside'].mean():>5.2f}/{s['maxside'].median():>4.2f}/{s['maxside'].quantile(.9):>5.2f} | "
              f"{s['range'].mean():>8.2f} | {(s['clside']=='inside').mean()*100:>10.0f}%")

def f30table(d,label):
    print(f"\n=== 1st-30min vs weekly CPR -> did the week HOLD that side?  {label}  n={len(d)} ===")
    print(f"{'1st30 close':12s} {'n':>4} {'%wks':>5} | {'closed ABOVE':>12} {'closed INSIDE':>13} {'closed BELOW':>12} | {'crossed to far side intrawk':>28}")
    for fs in ("above","inside","below"):
        s=d[d["f30side"]==fs]
        if not len(s): continue
        ca=(s["clside"]=="above").mean()*100; ci=(s["clside"]=="inside").mean()*100; cb=(s["clside"]=="below").mean()*100
        cross=(s["touch_below"].mean()*100) if fs=="above" else ((s["touch_above"].mean()*100) if fs=="below" else ((s["touch_above"]|s["touch_below"]).mean()*100))
        print(f"{fs:12s} {len(s):>4} {len(s)/len(d)*100:>4.0f}% | {ca:>11.0f}% {ci:>12.0f}% {cb:>11.0f}% | {cross:>27.0f}%")

qtable(R,"RECENT 2023-2026  (CPR width bands + movement)")
f30table(R,"RECENT 2023-2026")
print("\n(clarify: net% = END-TO-END week move open->close, signed avg & |abs| avg. "
      "maxSide% = the larger of (high-open) / (open-low) = the MAX one-direction excursion from the week open.)")
qtable(W,"FULL 2015-2026 (robustness)")
f30table(W,"FULL 2015-2026")
