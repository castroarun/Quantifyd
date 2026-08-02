"""
Run the research/62 WINNER (rsblend, N=8, buffer=22, gate=100, Donchian=15) on
research/41's mid/small-cap universe instead of the Nifty-200 pool. Reuses r62's
engine wholesale; only the universe band is swapped by monkey-patching eligible_etf.

Bands: nifty200=(0,200) [r62 original], combo=(100,500) [r41 MidSmallcap-400],
       mid=(100,250) [r41 winner band].

Run on VPS: venv/bin/python research/62_momentum_etf_subselect/scripts/compare_universes.py
"""
import importlib.util, csv
from pathlib import Path
import numpy as np, pandas as pd

M62 = Path("/home/arun/quantifyd/research/62_momentum_etf_subselect/scripts/62_mom30_subselect.py")
spec = importlib.util.spec_from_file_location("m62", str(M62)); m62 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m62)
rs2 = m62.rs2
RES = Path("/home/arun/quantifyd/research/62_momentum_etf_subselect/results")
CFG = {"band": "nifty200"}


def eligible_etf(close, tv, d, score_fn, etf_size):
    uni = rs2.pit_universe(tv, close, d, CFG["band"])
    if not uni:
        return None
    if score_fn == "mom30":
        sc = m62.mom30_score(close, d, uni)
    else:  # rsblend = r41 6m/12m relative strength
        sc = rs2.rs_scores(close, d, None)
        if sc is not None:
            sc = sc[[s for s in sc.index if s in uni and s != m62.BENCH]]
    if sc is None or len(sc) == 0:
        return None
    sc = sc[[s for s in sc.index if s != m62.BENCH]].sort_values(ascending=False)
    return list(sc.index[:etf_size])


def _breadth(close, tv, asof):
    uni = rs2.pit_universe(tv, close, asof, CFG["band"]); h = close.loc[:asof]; ab = tot = 0
    for s in uni:
        cs = h[s].dropna()
        if len(cs) >= 100:
            tot += 1
            if cs.iloc[-1] > cs.tail(100).mean():
                ab += 1
    return ab / tot if tot else 1.0


m62.eligible_etf = eligible_etf
m62._breadth = _breadth

print("Loading price+tv from VPS DB ...", flush=True)
close, tv = rs2.load()
print(f"  {close.shape[1]} symbols, {close.index.min().date()}..{close.index.max().date()}", flush=True)
print(f"  bands available: {rs2.BANDS}")

# how many names in each band (at a mid-sample date) for context
d0 = close.index[close.index >= "2020-01-01"][0]
for b in ["nifty200", "mid", "combo"]:
    try:
        u = rs2.pit_universe(tv, close, d0, b); print(f"  band {b:9s} -> {len(u)} names @ {d0.date()}")
    except Exception as e:
        print(f"  band {b}: {e}")

rows = []
WINNER = dict(score_fn="rsblend", N=8, buffer=22, gate=100, donchian=15)
for band in ["nifty200", "combo", "mid"]:
    CFG["band"] = band
    g = m62.run(close, tv, **WINNER)
    t = m62.run(close, tv, stcg=0.20, **WINNER)
    r = dict(universe=band,
             cagr=round(g["cagr"], 1), post_tax20=round(t["cagr"], 1),
             maxdd=round(g["dd"], 1), sharpe=round(g["sharpe"], 2),
             calmar=round(g["calmar"], 2) if pd.notna(g["calmar"]) else "",
             fills=g["st"]["fills"], donch_exits=g["st"]["donchian_exits"],
             gate_derisk=g["st"]["gate_derisk"])
    rows.append(r)
    print(f"  UNIVERSE={band:9s} CAGR={r['cagr']:5.1f}% net20={r['post_tax20']:5.1f}% "
          f"DD={r['maxdd']:6.1f}% Sharpe={r['sharpe']:.2f} Calmar={r['calmar']} "
          f"fills={r['fills']} donchX={r['donch_exits']}", flush=True)

# benchmark
bm = m62.stats_from_nav(m62.bench_nav(close))
print(f"  NIFTYBEES        CAGR={bm['cagr']:.1f}% DD={bm['dd']:.1f}% Sharpe={bm['sharpe']:.2f} Calmar={bm['calmar']:.2f}")

with open(RES / "compare_universes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print("\nwrote results/compare_universes.csv")
