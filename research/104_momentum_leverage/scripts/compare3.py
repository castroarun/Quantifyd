"""3-way, NO leverage, same engine (research/62), same universe (Nifty-200), same period, rt=0.3%:
 1) Existing live book  = rsblend score + top-30 pool/top-22 buffer + weekly gate + 15d Donchian
 2) Existing + RISK-ADJUSTED score (mom30) = only the score changes
 3) Plain risk-adj top-8 (study style) = mom30, no buffer pool, no Donchian
Isolates the effect of the risk-adjusted score and of the buffer/Donchian refinements."""
import importlib.util, logging
from pathlib import Path
import pandas as pd
logging.disable(logging.WARNING)
P = Path("/home/arun/quantifyd/research/62_momentum_etf_subselect/scripts/62_mom30_subselect.py")
_s = importlib.util.spec_from_file_location("r62", str(P)); r62 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r62)
close, tv = r62.rs2.load()
print(f"data {close.index.min().date()}..{close.index.max().date()}  period from {r62.START.date()}", flush=True)
bn = r62.bench_nav(close); bm = r62.stats_from_nav(bn)
print(f"\n{'BOOK':>44} {'CAGR':>6} {'MaxDD':>7} {'Sharpe':>6} {'Sortino':>7} {'Calmar':>6} {'donchX':>6} {'turn':>5}")
print(f"{'NIFTYBEES buy-hold':>44} {bm['cagr']:>5.1f}% {bm['dd']:>6.1f}% {bm['sharpe']:>6.2f} {bm['sortino']:>7.2f} {bm['calmar']:>6.2f}")
cfgs = [
    ("1) Existing live (rsblend+buffer+Donchian)", "rsblend", 8, 22, 100, 15, 30),
    ("2) Existing + RISK-ADJUSTED score",          "mom30",   8, 22, 100, 15, 30),
    ("3) Plain risk-adj top-8 (study style)",      "mom30",   8,  8, 100, None, 8),
]
rows = []
for lbl, score, N, buf, gate, donch, etf in cfgs:
    g = r62.run(close, tv, score_fn=score, N=N, buffer=buf, gate=gate, donchian=donch, etf_size=etf, rt=0.003)
    st = g['st']
    fills = st.get('fills', 0) or 1
    turn = st['turn_sum']/fills
    print(f"{lbl:>44} {g['cagr']:>5.1f}% {g['dd']:>6.1f}% {g['sharpe']:>6.2f} "
          f"{(g['sortino'] if pd.notna(g['sortino']) else 0):>7.2f} {(g['calmar'] if pd.notna(g['calmar']) else 0):>6.2f} "
          f"{st['donchian_exits']:>6} {turn:>5.2f}", flush=True)
    rows.append((lbl, round(g['cagr'],1), round(g['dd'],1), round(g['sharpe'],2), round(g['calmar'],2) if pd.notna(g['calmar']) else 0))
import csv
with open("/home/arun/quantifyd/research/104_momentum_leverage/results/compare3_nolev.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["book","cagr","maxdd","sharpe","calmar"]); w.writerows(rows)
print("\nwrote compare3_nolev.csv")
