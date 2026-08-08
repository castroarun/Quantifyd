"""Hedge economics normalised to the BOOK AT THE TIME (not % of starting capital, which inflates
later years ~50x on a compounding book). This is the honest read of what the hedge actually costs."""
import json
from collections import defaultdict
from pathlib import Path
RES = Path("/home/arun/quantifyd/research/105_momentum_put_hedge/results")
V = json.load(open(RES / "hedge_viz.json"))
for key in ("monthly_full", "weekly_2019"):
    eps = V[key]["episodes"]
    yrs = sorted({e["entry"][:4] for e in eps})
    n_years = len(yrs)
    print(f"\n=== {V[key]['label']} ===")
    print(f"{'year':>6} {'puts':>5} {'premium % of book':>18} {'net P&L % of book':>19}")
    tc = tp = 0.0
    by = defaultdict(lambda: [0.0, 0.0, 0])
    for e in eps:
        y = e["entry"][:4]
        by[y][0] += e["cost_book"]; by[y][1] += e["pnl_book"]; by[y][2] += 1
        tc += e["cost_book"]; tp += e["pnl_book"]
    for y in yrs:
        c, p, n = by[y]
        print(f"{y:>6} {n:>5} {c:>17.1f}% {p:>+18.1f}%")
    print(f"{'TOTAL':>6} {len(eps):>5} {tc:>17.1f}% {tp:>+18.1f}%")
    print(f"  -> average {tc/n_years:.1f}% of book per year in premium, "
          f"{tp/n_years:+.1f}% per year net")
    print(f"  -> per put: avg cost {tc/len(eps):.2f}% of book, avg P&L {tp/len(eps):+.2f}%")
    print(f"  -> the hedge recovered {100*(tp+tc)/tc:.0f}% of every rupee of premium "
          f"(100% = free, below = a net drag)")
