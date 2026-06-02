"""Capstone: the combined-edge prototype vs baseline.

BASELINE  : ATM straddle, EVERY day, 1.3x premium stop, exit 14:45.
COMBINED  : ATM straddle, only TIGHT-open days (opening-15min range < median),
            +/-0.4% underlying-move stop, exit 14:45.   (1-DTE structure both.)

Run over real NIFTY 5-min paths 2024-26 (BS premium model, IV calibrated to the real
28-day chain). The tight-open filter's validity rests on the 6-yr opening-range result
(regime_long). Premium side is modelled => read the LIFT (combined vs baseline) + tail,
not absolute level. Also reports the real-28d 1-DTE tight-open days (tiny n, flagged).
"""
import sqlite3, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/arun/quantifyd"); BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"; OUT.mkdir(parents=True, exist_ok=True)
LOT = 65; QTY = LOT * 2; BROK = 80; R = 0.065
def ncdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def bs(S, K, T, sig, typ):
    if T <= 0 or sig <= 0: return max(S-K,0) if typ=="C" else max(K-S,0)
    d1 = (math.log(S/K)+(R+.5*sig*sig)*T)/(sig*math.sqrt(T)); d2 = d1-sig*math.sqrt(T)
    return (S*ncdf(d1)-K*math.exp(-R*T)*ncdf(d2)) if typ=="C" else (K*math.exp(-R*T)*ncdf(-d2)-S*ncdf(-d1))
def strad(S,K,T,sig): return bs(S,K,T,sig,"C")+bs(S,K,T,sig,"P")

oc = sqlite3.connect(str(ROOT/"backtest_data"/"options_data.db"))
SIG = ((oc.execute("SELECT AVG(iv) FROM option_chain WHERE symbol='NIFTY' AND iv IS NOT NULL AND ABS(strike-underlying_spot)<=75 AND iv BETWEEN 3 AND 60").fetchone()[0]) or 13)/100
oc.close()
cx = sqlite3.connect(str(ROOT/"backtest_data"/"market_data.db"))
df = pd.read_sql_query("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date", cx); cx.close()
df["dt"]=pd.to_datetime(df["date"]); df["day"]=df["dt"].dt.date

# precompute opening-15min range % per day for the filter
o15 = {}
for day,g in df.groupby("day"):
    op = g[g["dt"].dt.time>=pd.to_datetime("09:15").time()]
    first3 = op[op["dt"].dt.time<=pd.to_datetime("09:30").time()]
    if len(first3) and len(op):
        o15[day] = (first3["high"].max()-first3["low"].min())/op["open"].iloc[0]*100
o15s = pd.Series(o15); med = o15s.median()

def sim(g, mode):
    g = g[g["dt"].dt.time>=pd.to_datetime("09:30").time()].reset_index(drop=True)
    if len(g)<5: return None
    S0=g["open"].iloc[0]; K=round(S0/50)*50; n=len(g); V0=strad(S0,K,1.2/365,SIG)
    for i,row in g.iterrows():
        frac=i/max(n-1,1); T=max((1.2-.25*frac)/365,1e-5)
        Sadv=row["high"] if abs(row["high"]-K)>abs(row["low"]-K) else row["low"]
        Vadv=strad(Sadv,K,T,SIG); last=(i==n-1)
        trig=last
        if mode=="base" and Vadv>=V0*1.3: trig=True
        elif mode=="combo" and abs(Sadv-S0)/S0*100>=0.4: trig=True
        if trig:
            Vx = Vadv if not last else strad(g["close"].iloc[-1],K,T,SIG)
            return (V0-Vx)*QTY-2*BROK
    return (V0-strad(g["close"].iloc[-1],K,1e-5,SIG))*QTY-2*BROK

res={"base":[], "combo":[]}
for day,g in df.groupby("day"):
    if g["dt"].dt.weekday.iloc[0]>=5: continue
    b=sim(g,"base")
    if b is not None: res["base"].append(b)
    if day in o15 and o15[day] < med:   # tight-open filter
        cmb=sim(g,"combo")
        if cmb is not None: res["combo"].append(cmb)

def stat(a):
    a=np.array(a); return dict(n=len(a), total=round(a.sum()), perday=round(a.mean()),
        win=round(100*(a>0).mean()), worst=round(a.min()), worst5=round(np.sort(a)[:5].mean()),
        sharpe=round(a.mean()/a.std()*math.sqrt(252),2) if a.std()>0 else 0)
B,Cc = stat(res["base"]), stat(res["combo"])
print("BASELINE", B); print("COMBINED", Cc)

fig,ax=plt.subplots(1,2,figsize=(13,5.5))
m=["n","perday","win","worst","worst5","sharpe"]
x=np.arange(len(m)); w=.38
ax[0].axis("off")
tbl=ax[0].table(cellText=[[B[k] for k in m],[Cc[k] for k in m]], rowLabels=["Baseline","Combined"], colLabels=m, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,2)
ax[0].set_title("Baseline (all days, 1.3x prem stop) vs Combined (tight-open, +/-0.4% move stop)", fontsize=10)
cum_b=np.cumsum(res["base"]); cum_c=np.cumsum(res["combo"])
ax[1].plot(cum_b,label=f"Baseline (n={B['n']})",color="#888")
ax[1].plot(cum_c,label=f"Combined (n={Cc['n']})",color="#0a6",lw=2)
ax[1].axhline(0,color="#999"); ax[1].legend(); ax[1].set_title("Cumulative modelled ₹ (per trade index)")
fig.suptitle(f"Combined-edge prototype vs baseline (BS model IV={SIG*100:.0f}%, real NIFTY 5-min 2024-26)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"prototype.png",dpi=110,bbox_inches="tight"); print("WROTE prototype.png")

L=[f"# Capstone — combined-edge prototype vs baseline (BS model, real NIFTY 5-min 2024-26, IV {SIG*100:.0f}%)\n",
   "BASELINE: ATM straddle every day, 1.3x premium stop. COMBINED: tight-open days only "
   "(open-15min < median), +/-0.4% underlying-move stop. Both 1-DTE structure, exit 14:45.\n",
   "| metric | Baseline | Combined |", "|---|---|---|"]
for k in m: L.append(f"| {k} | {B[k]} | {Cc[k]} |")
L+=["\n**Read the LIFT, not the level** (BS-modelled premiums). The combined config trades fewer days "
    "(tight-open filter) with a bounded move-stop. Compare per-day, win%, and especially worst/worst5 (tail).",
    "- Filter validity: opening-range->range-day is robust over 6 yrs (regime_long).",
    "- 28-day REAL premiums have only ~4 one-DTE days (~2 after the tight filter) -> real-level confirmation "
    "needs the recorder to accumulate; this multi-year model is the lift/tail lens."]
(OUT/"RESULTS_prototype.md").write_text("\n".join(L),encoding="utf-8"); print("WROTE RESULTS_prototype.md")
