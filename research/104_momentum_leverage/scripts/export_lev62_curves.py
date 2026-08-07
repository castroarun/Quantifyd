"""Export EXISTING-BOOK (rsblend+buffer+Donchian) leverage-frontier curves + borrow sensitivity for the
amended /app tearsheet. Full 2006-2026 cycle, MTF financing. Reuses run_lev62 engine."""
import importlib.util, json
from pathlib import Path
import pandas as pd
R = Path("/home/arun/quantifyd/research/104_momentum_leverage/scripts/run_lev62.py")
_s = importlib.util.spec_from_file_location("R", str(R)); Rm = importlib.util.module_from_spec(_s); _s.loader.exec_module(Rm)
r62, BENCH, START = Rm.r62, Rm.BENCH, Rm.START
close, tv = Rm.mom75.load()
out = {"curves": {}, "kpi": {}, "peryear": {}}
navs = {}
for name, lev in [("Own capital · 1.0×", 1.0), ("Levered · 1.3×", 1.3), ("Levered · 1.6×", 1.6), ("Levered · 2.0×", 2.0)]:
    r = Rm.run_lev62(close, tv, "rsblend", 8, 22, 100, 15, 30, lev, borrow=0.105)
    navs[name] = r['nav']
    out['kpi'][name] = dict(cagr=round(r['cagr'],1), dd=round(r['dd'],1), sharpe=round(r['sharpe'],2),
        calmar=round(r['calmar'],2) if pd.notna(r['calmar']) else 0, mult=round(r['nav'].iloc[-1]/r['nav'].iloc[0],1))
bn = close[BENCH].loc[close.index >= START].dropna(); bn = bn/bn.iloc[0]
bs = r62.stats_from_nav(bn); navs["NIFTY B&H"] = bs['nav']
out['kpi']["NIFTY B&H"] = dict(cagr=round(bs['cagr'],1), dd=round(bs['dd'],1), sharpe=round(bs['sharpe'],2),
    calmar=round(bs['calmar'],2) if pd.notna(bs['calmar']) else 0, mult=round(bs['nav'].iloc[-1]/bs['nav'].iloc[0],1))
months = None
for name, n in navs.items():
    m = n.resample("ME").last(); m = m/m.iloc[0]
    out['curves'][name] = [[d.strftime("%Y-%m"), round(float(v),4)] for d, v in m.items()]
    ms = [d.strftime("%Y-%m") for d in m.index]
    if months is None or len(ms) > len(months): months = ms
    yr = n.groupby(n.index.year).last(); py = yr.pct_change(); py.iloc[0] = yr.iloc[0]/n.iloc[0]-1
    out['peryear'][name] = {int(y): round(float(v)*100,1) for y, v in py.items()}
out['months'] = months
json.dump(out, open("/home/arun/quantifyd/research/104_momentum_leverage/results/lev_curves.json", "w"))
print("wrote lev_curves.json (EXISTING book) —", len(months), "months")
for name in navs: print(f"  {name:20} CAGR {out['kpi'][name]['cagr']}% DD {out['kpi'][name]['dd']}% Calmar {out['kpi'][name]['calmar']} {out['kpi'][name]['mult']}x")
# borrow sensitivity 8% vs 10.5%
print("\nBORROW SENSITIVITY (existing book):")
print(f"{'lev':>5} {'CAGR@8%':>8} {'CAGR@10.5%':>11} {'DD@10.5%':>9} {'Calmar@10.5%':>12}")
for lev in [1.0, 1.3, 1.6, 2.0]:
    r8 = Rm.run_lev62(close, tv, "rsblend", 8, 22, 100, 15, 30, lev, borrow=0.08)
    r10 = out['kpi'][{1.0:"Own capital · 1.0×",1.3:"Levered · 1.3×",1.6:"Levered · 1.6×",2.0:"Levered · 2.0×"}[lev]]
    print(f"{lev:>4}x {r8['cagr']:>7.1f}% {r10['cagr']:>10.1f}% {r10['dd']:>8.1f}% {r10['calmar']:>12}")
