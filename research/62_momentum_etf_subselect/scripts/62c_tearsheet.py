"""Research 62 — client tearsheet for the G2 winner.
Winner = rsblend N8 buf22 gate100 Donch15 (net-of-cost daily NAV vs NIFTYBEES).
Generates the one-page factsheet PNG + HTML via research/_utilities/tearsheet.py.
"""
import importlib.util, sys, logging
from pathlib import Path
logging.disable(logging.WARNING)

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
ENG = HERE / "scripts" / "62_mom30_subselect.py"
_s = importlib.util.spec_from_file_location("m62", str(ENG))
m = importlib.util.module_from_spec(_s); _s.loader.exec_module(m)
sys.path.insert(0, "/home/arun/quantifyd/research/_utilities")
import tearsheet as TS

print("Loading ...", flush=True)
close, tv = m.rs2.load()
# winner — net of cost (0.4% RT already in engine), pre-tax NAV for the factsheet curve
g = m.run(close, tv, "rsblend", 8, 22, 100, 15)
t = m.run(close, tv, "rsblend", 8, 22, 100, 15, stcg=0.20)
strat = g["nav"]
bench = m.bench_nav(close)
print(f"strat CAGR(gross-of-tax,net-of-cost)={g['cagr']:.1f}%  net-tax={t['cagr']:.1f}%  "
      f"DD={g['dd']:.1f}%  Sharpe={g['sharpe']:.2f}", flush=True)

meta = dict(bench="NIFTY-50 (NIFTYBEES)", tag="RECONSTRUCTED · NET OF 0.4% COST",
            note=f"Post-tax CAGR {t['cagr']:.1f}% (STCG 20%). Reconstructed Nifty 200 "
                 f"Momentum 30 from methodology; backtest, not live.")
TS.generate_tearsheet(strat, bench,
                      "Momentum-30 Sub-Selection (Top-8 + Gate + Donchian-15)",
                      meta=meta, out_dir=str(RES))
print("TEARSHEET_DONE", flush=True)
for p in RES.glob("*.png"):
    print("  PNG:", p, p.stat().st_size, flush=True)
for p in RES.glob("*.html"):
    print("  HTML:", p, flush=True)
