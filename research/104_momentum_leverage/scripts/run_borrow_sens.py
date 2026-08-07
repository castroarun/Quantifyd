"""Borrow-rate sensitivity on the leverage frontier — 8% (futures-basis proxy) vs 10.5% (real MTF)."""
import importlib.util
from pathlib import Path
import pandas as pd
P = Path("/home/arun/quantifyd/research/104_momentum_leverage/scripts/run_lev_conc.py")
_s = importlib.util.spec_from_file_location("lc", str(P)); lc = importlib.util.module_from_spec(_s); _s.loader.exec_module(lc)
mom = lc.mom
close, tv = mom.load()
ema = {p: close.ewm(span=p, adjust=False, min_periods=p).mean() for p in (50,100,200)}
nbx = close[lc.BENCH].ffill().ewm(span=100, adjust=False, min_periods=100).mean()
print(f"{'config':>18} {'borrow':>6} {'CAGR':>6} {'MaxDD':>7} {'Calmar':>6}")
rows=[]
for score, N in [("radj_z",8),("ret252",8)]:
    for lev in [1.0,1.3,1.6,2.0]:
        for br in [0.08,0.105]:
            r = lc.run_lev(close, tv, ema, nbx, score, N, lev, borrow=br)
            cal = round(r['calmar'],2) if pd.notna(r['calmar']) else 0
            print(f"{score+'_N8_L'+str(lev):>18} {br*100:>5.1f}% {r['cagr']:>5.1f}% {r['dd']:>6.1f}% {cal:>6}", flush=True)
            rows.append((score,lev,br,round(r['cagr'],1),round(r['dd'],1),cal))
pd.DataFrame(rows,columns=["score","lev","borrow","cagr","maxdd","calmar"]).to_csv("/home/arun/quantifyd/research/104_momentum_leverage/results/borrow_sens.csv",index=False)
print("wrote borrow_sens.csv")
