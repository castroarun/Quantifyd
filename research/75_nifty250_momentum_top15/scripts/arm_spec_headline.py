"""Measure the EXACT Aurum-armed spec (GATE-A locked decisions) so the registry headline
is honest: top-250 · 12m return · NO EMA-stack (Dec A) · buffer-22 (Dec C) · NIFTYBEES-100EMA
gate · N15 · monthly · daily-marked (Dec E honest basis). Also dump its NAV + a tearsheet."""
import importlib.util, sys, logging
from pathlib import Path
import pandas as pd
logging.disable(logging.WARNING)
HERE = Path("/home/arun/quantifyd/research/75_nifty250_momentum_top15")
RES = HERE / "results"


def imp(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m


p1 = imp("p1", str(HERE / "scripts" / "run_nifty250_momentum.py"))
p3 = imp("p3", str(HERE / "scripts" / "run_phase3_combos.py"))
sys.path.insert(0, "/home/arun/quantifyd/research/_utilities")
import tearsheet as TS
BENCH = p1.BENCH; RT = p1.RT; START = p1.START

print("Loading ...", flush=True)
close, tv = p1.load()
ema = {p: close.ewm(span=p, adjust=False, min_periods=p).mean() for p in (50, 100, 200)}
nbx_ema100 = close[BENCH].ffill().ewm(span=100, adjust=False, min_periods=100).mean()
ind = dict(ath=close.cummax(), sma100=close.rolling(100, min_periods=100).mean(),
           low15=close.rolling(15, min_periods=15).min().shift(1), r21=close.pct_change(21))

# EXACT armed spec: n250, ret252, NO stack (phase-3 run has no stack), gate ON, buffer ~22, no per-stock exit
def run(stcg):
    return p3.run(close, tv, ema, nbx_ema100, ind, band="n250", score="ret252", N=15,
                  use_gate=True, use_quality=False, use_ath=False, buffer_mult=22/15,
                  exit_mode="none", rt=RT, stcg=stcg)

net = run(0.0); tax = run(0.20)
m = net["modern"]
print("=== ARMED SPEC (n250 · 12m · NO-stack · buffer22 · gate · N15 · monthly) ===", flush=True)
print(f"  net CAGR   = {net['cagr']:.1f}%", flush=True)
print(f"  post-tax20 = {tax['cagr']:.1f}%", flush=True)
print(f"  MaxDD      = {net['dd']:.1f}%  (daily-marked)", flush=True)
print(f"  Calmar     = {net['calmar']:.2f}", flush=True)
print(f"  Sharpe     = {net['sharpe']:.2f}", flush=True)
print(f"  mult       = {net['mult']:.0f}x", flush=True)
print(f"  turnover   = {net['st']['turn_sum']/max(1,net['st']['fills']):.3f}", flush=True)
print(f"  modern 2014+ : CAGR {m['cagr']:.1f}%  DD {m['dd']:.1f}%  Cal {m['calmar']:.2f}", flush=True)

strat = net["nav"]; strat.to_csv(RES / "nav_armed_spec.csv", header=["nav"])
bench = close[BENCH].loc[close.index >= START].dropna(); bench = bench / bench.iloc[0]
meta = dict(bench="NIFTY-50 (NIFTYBEES)", tag="AURUM-ARMED SPEC · NET 0.3% · DAILY-MARKED",
            note=f"Aurum nifty250_momentum (n250-v1): top-250 by traded value, top-15 by 12-month return, "
                 f"NIFTYBEES-100EMA cash gate, retention buffer 22, monthly, NO per-stock EMA filter "
                 f"(dropped per GATE-A: inert-to-harmful). Post-tax {tax['cagr']:.1f}% (STCG 20%). Backtest 2006-2026.")
TS.generate_tearsheet(strat, bench, "Nifty-250 Momentum (Aurum n250-v1)", meta=meta, out_dir=str(RES))
# rename so it doesn't clobber the faithful-base tearsheet
import os
if (RES / "tearsheet.png").exists():
    os.replace(RES / "tearsheet.png", RES / "tearsheet_armed.png")
print("TEARSHEET_DONE -> tearsheet_armed.png", flush=True)
