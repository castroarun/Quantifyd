"""EXTENDED NIFTY-vs-BANKNIFTY beta (now real BNF daily 2015-2026, incl. 2018/2020/2022 stress).
Does the cross-hedge thesis (BNF moves MORE, esp. in collapses) hold once stress regimes are in-sample?"""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
def dly(sym): return pd.read_sql("SELECT date,close FROM market_data_unified WHERE symbol=? AND timeframe='day' ORDER BY date",c,params=[sym],parse_dates=["date"]).set_index("date")["close"]
d=pd.concat({"N":dly("NIFTY50"),"B":dly("BANKNIFTY")},axis=1).dropna()
d["rN"]=d["N"].pct_change()*100; d["rB"]=d["B"].pct_change()*100; d=d.dropna()
beta=lambda x,y: np.polyfit(x,y,1)[0]
print(f"FULL daily {d.index[0].date()}->{d.index[-1].date()}  n={len(d)}")
print(f"corr {d.rN.corr(d.rB):.3f} | beta ALL {beta(d.rN,d.rB):.2f} | DOWN {beta(d[d.rN<0].rN,d[d.rN<0].rB):.2f} | UP {beta(d[d.rN>0].rN,d[d.rN>0].rB):.2f} | BIG-DOWN(<-1%) {beta(d[d.rN<-1].rN,d[d.rN<-1].rB):.2f}")
print("\nCRASH days — does BNF fall MORE? (the collapse thesis)")
for thr in (1,2,3):
    s=d[d.rN<-thr]
    if len(s)<3: print(f"  NIFTY<-{thr}%: n={len(s)} thin"); continue
    print(f"  NIFTY<-{thr}% (n={len(s)}): NIFTY {s.rN.mean():.2f}% | BNF {s.rB.mean():.2f}% | ratio {(s.rB/s.rN).mean():.2f} | BNF fell MORE {((s.rB<s.rN)).mean()*100:.0f}% of the time")
print("\nPER-YEAR beta + corr (was BNF higher-beta in stress years?):")
print(f"{'yr':6s} {'n':>4} {'beta':>5} {'down-beta':>9} {'corr':>5} {'NIFTY ann%':>10} {'BNF ann%':>9}")
for y in range(2015,2027):
    s=d[d.index.year==y]
    if len(s)<30: continue
    db=beta(s[s.rN<0].rN,s[s.rN<0].rB) if (s.rN<0).sum()>5 else float("nan")
    print(f"{y:<6d} {len(s):>4} {beta(s.rN,s.rB):>5.2f} {db:>9.2f} {s.rN.corr(s.rB):>5.2f} {((s.N.iloc[-1]/s.N.iloc[0])-1)*100:>9.1f}% {((s.B.iloc[-1]/s.B.iloc[0])-1)*100:>8.1f}%")
print("\nWEEKLY:")
w=pd.concat({"rN":d["N"].resample("W-FRI").last().pct_change()*100,"rB":d["B"].resample("W-FRI").last().pct_change()*100},axis=1).dropna()
print(f"  corr {w.rN.corr(w.rB):.3f} beta {beta(w.rN,w.rB):.2f} down-beta {beta(w[w.rN<0].rN,w[w.rN<0].rB):.2f} | weekly |move| p90 N {w.rN.abs().quantile(.9):.2f}% B {w.rB.abs().quantile(.9):.2f}%")
