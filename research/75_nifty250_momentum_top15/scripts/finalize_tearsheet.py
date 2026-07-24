"""Research 75 finalize: dump the faithful-base NAV, run the owed smallcap/combo cost-stress,
and generate the client tearsheet PNG+HTML for the /app publish."""
import importlib.util, sys, logging
from pathlib import Path
import pandas as pd
logging.disable(logging.WARNING)

HERE = Path("/home/arun/quantifyd/research/75_nifty250_momentum_top15")
RES = HERE / "results"; RES.mkdir(exist_ok=True)


def imp(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


p1 = imp("p1", str(HERE / "scripts" / "run_nifty250_momentum.py"))
p2 = imp("p2", str(HERE / "scripts" / "run_variants_phase2.py"))
sys.path.insert(0, "/home/arun/quantifyd/research/_utilities")
import tearsheet as TS
rs2 = p1.rs2; BENCH = p1.BENCH; RT = p1.RT; START = p1.START

print("Loading ...", flush=True)
close, tv = p1.load()
ema = {p: close.ewm(span=p, adjust=False, min_periods=p).mean() for p in (50, 100, 200)}
nbx_ema100 = close[BENCH].ffill().ewm(span=100, adjust=False, min_periods=100).mean()

# ---- faithful base = the video, exactly: n250, 12m momentum, EMA-stack ON, gate ON, monthly
base = p1.run(close, tv, ema, nbx_ema100, "ret252", 15, True, True, RT, 0.0, START, "monthly")
baset = p1.run(close, tv, ema, nbx_ema100, "ret252", 15, True, True, RT, 0.20, START, "monthly")
print(f"FAITHFUL BASE: net={base['cagr']:.1f}%  post-tax={baset['cagr']:.1f}%  "
      f"DD={base['dd']:.1f}%  Sharpe={base['sharpe']:.2f}  Cal={base['calmar']:.2f}  "
      f"mult={base['mult']:.0f}x", flush=True)
strat = base["nav"]; strat.to_csv(RES / "nav_faithful.csv", header=["nav"])
bench = close[BENCH].loc[close.index >= START].dropna(); bench = bench / bench.iloc[0]

# ---- owed cost-stress on the smallcap-heavy cells (they were optimistic at 0.3%)
print("COST-STRESS (combo & smallcap, ret252, gate ON, no stack):", flush=True)
cs = {}
for band in ("combo", "small"):
    for rt in (0.003, 0.005, 0.007):
        r = p2.run(close, tv, ema, nbx_ema100, band, "ret252", 15, False, True, rt, 0.0, 0.065)
        cs[(band, rt)] = (r['cagr'], r['dd'], r['calmar'])
        print(f"  {band:5} rt={rt*100:.1f}%  net={r['cagr']:5.1f}%  DD={r['dd']:6.1f}%  "
              f"Cal={r['calmar']:.2f}", flush=True)
pd.DataFrame([{"band": b, "rt_pct": rt * 100, "net_cagr": round(v[0], 1),
               "maxdd": round(v[1], 1), "calmar": round(v[2], 2)}
              for (b, rt), v in cs.items()]).to_csv(RES / "cost_stress.csv", index=False)

# ---- client tearsheet for the faithful base
meta = dict(bench="NIFTY-50 (NIFTYBEES)", tag="SURVIVORSHIP-FREE · NET 0.3% COST · DAILY-MARKED",
            note=f"Faithful replication of the Quantinuous Nifty-250 momentum video: top-15 by 12-month "
                 f"momentum, per-stock 50>100>200 EMA filter, NIFTYBEES>100-EMA cash gate, monthly. "
                 f"Post-tax CAGR {baset['cagr']:.1f}% (STCG 20%). Backtest 2006-2026, not live.")
TS.generate_tearsheet(strat, bench, "Nifty-250 Momentum Top-15 (Video Replication)",
                      meta=meta, out_dir=str(RES))
print("TEARSHEET_DONE", flush=True)
for p in RES.glob("*.png"):
    print("  PNG:", p.name, p.stat().st_size, flush=True)
