"""Export leverage-frontier NAV curves (radj_z N8, borrow 10.5%) + B&H for the /app tearsheet."""
import importlib.util, json
from pathlib import Path
import pandas as pd, numpy as np
P = Path("/home/arun/quantifyd/research/104_momentum_leverage/scripts/run_lev_conc.py")
_s = importlib.util.spec_from_file_location("lc", str(P)); lc = importlib.util.module_from_spec(_s); _s.loader.exec_module(lc)
mom = lc.mom; START = mom.START
close, tv = mom.load()
ema = {p: close.ewm(span=p, adjust=False, min_periods=p).mean() for p in (50,100,200)}
nbx = close[lc.BENCH].ffill().ewm(span=100, adjust=False, min_periods=100).mean()
navs = {}; out = {"curves":{}, "kpi":{}, "peryear":{}}
for name, lev in [("Own capital · 1.0×",1.0),("Levered · 1.3×",1.3),("Levered · 1.6×",1.6),("Levered · 2.0×",2.0)]:
    r = lc.run_lev(close, tv, ema, nbx, "radj_z", 8, lev, borrow=0.105)
    navs[name] = r['nav']
    out['kpi'][name] = dict(cagr=round(r['cagr'],1), dd=round(r['dd'],1), sharpe=round(r['sharpe'],2),
        calmar=round(r['calmar'],2) if pd.notna(r['calmar']) else 0, mult=round(r['mult'],1))
bn = close[lc.BENCH].loc[close.index >= START].dropna(); bn = bn/bn.iloc[0]
bstat = mom._stats(list(zip(bn.index, bn.values)), dict(fills=0, turn_sum=0.0, gate_off=0, mcalls=0), START)
navs["NIFTY B&H"] = bstat['nav']
out['kpi']["NIFTY B&H"] = dict(cagr=round(bstat['cagr'],1), dd=round(bstat['dd'],1), sharpe=round(bstat['sharpe'],2),
    calmar=round(bstat['calmar'],2) if pd.notna(bstat['calmar']) else 0, mult=round(bstat['mult'],1))
months = None
for name, n in navs.items():
    m = n.resample("ME").last(); m = m/m.iloc[0]
    out['curves'][name] = [[d.strftime("%Y-%m"), round(float(v),4)] for d, v in m.items()]
    ms = [d.strftime("%Y-%m") for d in m.index]
    if months is None or len(ms) > len(months): months = ms
    yr = n.groupby(n.index.year).last(); py = yr.pct_change(); py.iloc[0] = yr.iloc[0]/n.iloc[0]-1
    out['peryear'][name] = {int(y): round(float(v)*100,1) for y, v in py.items()}
out['months'] = months
json.dump(out, open("/home/arun/quantifyd/research/104_momentum_leverage/results/lev_curves.json","w"))
print("wrote lev_curves.json —", len(months), "months,", len(navs), "series")
for name in navs: print(f"  {name:20} CAGR {out['kpi'][name]['cagr']}% DD {out['kpi'][name]['dd']}% Calmar {out['kpi'][name]['calmar']} {out['kpi'][name]['mult']}x")
