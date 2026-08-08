"""EX-COVID TEST: does the put hedge only pay because of one once-in-a-decade crash?

Reports (a) hedge P&L per year so the concentration is visible, (b) hedge economics with COVID episodes
removed, and (c) the three books re-run over a clean post-COVID window (2021-2026) — i.e. what the hedge
actually costs in a NORMAL market with no once-in-a-decade crash.
"""
import importlib.util, json
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
RES = Path("/home/arun/quantifyd/research/105_momentum_put_hedge/results")
V = json.load(open(RES / "hedge_viz.json"))

for key in ("weekly_2019", "monthly_full"):
    eps = V[key]["episodes"]
    print(f"\n=== {V[key]['label']} — hedge P&L by year (% of starting capital) ===")
    by = defaultdict(lambda: [0.0, 0.0, 0, 0])
    for e in eps:
        y = e["entry"][:4]
        by[y][0] += e["cost_pct"]; by[y][1] += e["pnl_pct"]; by[y][2] += 1
        by[y][3] += 1 if e["pnl_pct"] > 0 else 0
    print(f"{'year':>6} {'puts':>5} {'wins':>5} {'premium':>10} {'net P&L':>11}")
    for y in sorted(by):
        c, p, n, w = by[y]
        print(f"{y:>6} {n:>5} {w:>5} {c:>9.1f}% {p:>+10.1f}%")
    # COVID = any episode touching 2020-02-01 .. 2020-05-31
    cov = [e for e in eps if not (e["exit"] < "2020-02-01" or e["entry"] > "2020-05-31")]
    non = [e for e in eps if (e["exit"] < "2020-02-01" or e["entry"] > "2020-05-31")]
    cc, cp = sum(e["cost_pct"] for e in cov), sum(e["pnl_pct"] for e in cov)
    nc, np_ = sum(e["cost_pct"] for e in non), sum(e["pnl_pct"] for e in non)
    print(f"  COVID window ({len(cov)} puts): premium {cc:.1f}%  net P&L {cp:+.1f}%")
    print(f"  ALL OTHER    ({len(non)} puts): premium {nc:.1f}%  net P&L {np_:+.1f}%")
    if nc:
        print(f"  -> outside COVID the hedge returned {100*(np_+nc)/nc:.0f}% of the premium paid "
              f"(100% = broke even, less = a net cost)")

# ── (c) clean post-COVID window ──
spec = importlib.util.spec_from_file_location(
    "hs", "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts/run_hedge_sweep.py")
hs = importlib.util.module_from_spec(spec); spec.loader.exec_module(hs)
spec2 = importlib.util.spec_from_file_location(
    "g5", "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts/run_hedge_g5.py")


def stats_from(pairs):
    n = pd.DataFrame(pairs, columns=["d", "v"]).set_index("d")["v"]
    y = (n.index[-1] - n.index[0]).days / 365.25
    c = (n.iloc[-1] / n.iloc[0]) ** (1 / y) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    return c * 100, dd * 100, (c / abs(dd) if dd < 0 else 0)


print("\n\n=== CLEAN POST-COVID WINDOW (2021-01 -> 2026): what the hedge costs in a normal market ===")
for lbl, mode, tenor in (("Cash exit (live rule)", "cash_exit", "weekly"),
                         ("Hold naked (no puts)", "hold", "weekly"),
                         ("Hold + WEEKLY puts 2x", "hedge", "weekly"),
                         ("Hold + MONTHLY puts 2x", "hedge", "monthly")):
    r = hs.run(mode=mode, struct="long", moneyness=0.0, tenor=tenor, ratio=2.0,
               start=pd.Timestamp("2021-01-01"))
    print(f"  {lbl:>24}: net CAGR {r['net_cagr']:>5.1f}%  MaxDD {r['maxdd']:>6.1f}%  "
          f"net Calmar {r['net_calmar']:>4.2f}  premium {r['prem']:>6.1f}%  hedgeP&L {r['hpnl']:>+7.1f}%")
print("\n(the 'hedge cost' you care about = the CAGR give-up of 'Hold + puts' vs 'Hold naked')")
