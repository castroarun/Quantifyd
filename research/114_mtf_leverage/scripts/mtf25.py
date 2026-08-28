# -*- coding: utf-8 -*-
"""Extend the leverage frontier to the MTF levels Arun can actually take, at the REAL MTF rate.

research/104 already answered the leverage question for this exact book (rsblend, top-8, buffer-22,
Donchian-15, weekly index gate) over 2006-2026 including 2008 — but it stopped at 2.0x and financed
at 10.5%. Two gaps:

  1. Arun asks about 2.5x, the conservative end of Zerodha's per-stock MTF factors. Untested.
  2. docs/MOMENTUM_GOLIVE_RUNBOOK_3L.md puts real Zerodha MTF interest at ~14.6%/yr (0.04%/day),
     not 10.5%. On borrowed money that is a 4.1pp understatement of the cost, and it compounds
     with leverage: at 2.5x you borrow 1.5x equity, so the gap is ~6pp of equity per year.

Both matter most at exactly the leverage being asked about, so this re-runs the identical engine
(run_lev62) across a wider grid at both rates and reports margin calls, which is the ruin question.
MAINT = 0.25 in that engine: a margin call fires when equity/gross < 25%, and liquidates.
"""
import csv, importlib.util, sys
from pathlib import Path
import pandas as pd

R104 = Path("/home/arun/quantifyd/research/104_momentum_leverage/scripts/run_lev62.py")
OUT = Path("/home/arun/quantifyd/research/114_mtf_leverage/results")
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("lev62", str(R104))
lev62 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lev62)

run, r62, mom75, BENCH = lev62.run_lev62, lev62.r62, lev62.mom75, lev62.BENCH

LEVS = [1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
RATES = [("10.5% (r/104 assumption)", 0.105), ("14.6% (real Zerodha MTF)", 0.146)]


def main():
    close, tv = mom75.load()
    print(f"data {close.index.min().date()}..{close.index.max().date()}, frontier from "
          f"{lev62.START.date()}, MAINT={lev62.MAINT}", flush=True)
    bn = close[BENCH].loc[close.index >= lev62.START].dropna(); bn = bn / bn.iloc[0]
    bm = r62.stats_from_nav(bn)
    print(f"NIFTYBEES B&H: CAGR {bm['cagr']:.1f}%  DD {bm['dd']:.1f}%  Calmar {bm['calmar']:.2f}\n", flush=True)

    rows = []
    f = open(OUT / "mtf_frontier.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["borrow_rate", "lev", "cagr", "maxdd", "sharpe", "calmar", "donch_exits", "mcalls"])
    for label, rate in RATES:
        print(f"--- financing at {label} ---", flush=True)
        print(f"{'leverage':>9} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'mcalls':>7}", flush=True)
        for lev in LEVS:
            g = run(close, tv, "rsblend", 8, 22, 100, 15, 30, lev, borrow=rate)
            cal = g["calmar"] if pd.notna(g["calmar"]) else 0.0
            mc = g["st"]["mcalls"]
            flag = "  <-- MARGIN CALLS" if mc else ""
            print(f"{lev:>8}x {g['cagr']:>6.1f}% {g['dd']:>7.1f}% {g['sharpe']:>7.2f} {cal:>7.2f} "
                  f"{mc:>7}{flag}", flush=True)
            w.writerow([rate, lev, round(g["cagr"], 1), round(g["dd"], 1), round(g["sharpe"], 2),
                        round(cal, 2), g["st"]["donchian_exits"], mc])
            f.flush()
            rows.append(dict(rate=rate, label=label, lev=lev, cagr=g["cagr"], dd=g["dd"],
                             sharpe=g["sharpe"], calmar=cal, mcalls=mc))
        print("", flush=True)
    f.close()

    print("=" * 84)
    print("WHAT THE RATE COSTS (same leverage, 10.5% vs 14.6%)")
    print(f"{'leverage':>9} {'CAGR @10.5%':>12} {'CAGR @14.6%':>12} {'give-up':>9} {'DD':>8} {'mcalls':>7}")
    lo = {r["lev"]: r for r in rows if r["rate"] == 0.105}
    hi = {r["lev"]: r for r in rows if r["rate"] == 0.146}
    for lev in LEVS:
        a, b = lo[lev], hi[lev]
        print(f"{lev:>8}x {a['cagr']:>11.1f}% {b['cagr']:>11.1f}% {b['cagr']-a['cagr']:>8.1f}pp "
              f"{b['dd']:>7.1f}% {b['mcalls']:>7}")
    print("=" * 84)
    base = hi[1.0]
    for lev in (2.5, 3.0):
        r = hi[lev]
        print(f"  {lev}x at the real rate: {r['cagr']:.1f}% CAGR ({r['cagr']-base['cagr']:+.1f}pp over 1.0x) "
              f"for a {r['dd']:.1f}% drawdown (1.0x is {base['dd']:.1f}%), Calmar {r['calmar']:.2f} "
              f"vs {base['calmar']:.2f}, margin calls {r['mcalls']}")


if __name__ == "__main__":
    main()
