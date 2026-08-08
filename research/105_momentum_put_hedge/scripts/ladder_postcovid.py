"""Close out the hedge analysis: the tenor ladder on the POST-COVID window (2021-2026), where hedging
led, using the best tenor (bi-weekly). Also re-states the COVID-inclusive window side by side."""
import importlib.util
import pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "g5", "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts/run_hedge_g5.py")
g5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g5)
ARMS = [("Cash exit (LIVE rule)", dict(mode="cash_exit")),
        ("Hold naked (no puts)", dict(mode="hold")),
        ("WEEKLY 2x   (~7 DTE)", dict(mode="hedge", tenor="weekly", ratio=2.0)),
        ("BI-WEEKLY 1x (~14 DTE)", dict(mode="hedge", tenor="biweekly", ratio=1.0)),
        ("BI-WEEKLY 2x (~14 DTE)", dict(mode="hedge", tenor="biweekly", ratio=2.0)),
        ("TRI-WEEKLY 2x (~21 DTE)", dict(mode="hedge", tenor="triweekly", ratio=2.0)),
        ("MONTHLY 2x  (~30 DTE)", dict(mode="hedge", tenor="monthly", ratio=2.0))]
WINDOWS = [("FULL 2019-2026 (includes COVID)", pd.Timestamp("2019-02-01")),
           ("POST-COVID 2021-2026 (normal market)", pd.Timestamp("2021-01-01"))]
store = {}
for wlbl, st in WINDOWS:
    print(f"\n=== {wlbl} ===")
    print(f"{'book':>26} {'netCAGR':>8} {'MaxDD':>7} {'netCal':>7} {'premium':>9} {'#hdg':>6}")
    for lbl, kw in ARMS:
        r = g5.run3(start=st, **kw)
        store[(wlbl, lbl)] = r
        pm = f"{r['prem']:.1f}%" if r['prem'] else "-"
        hg = r['hedges'] if r['hedges'] else "-"
        print(f"{lbl:>26} {r['net_cagr']:>7.1f}% {r['maxdd']:>6.1f}% {r['net_calmar']:>7.2f} {pm:>9} {hg:>6}")
    b = max(((l, store[(wlbl, l)]) for l, _ in ARMS), key=lambda x: x[1]['net_calmar'])
    print(f"  -> best risk-adjusted: {b[0]}  (netCalmar {b[1]['net_calmar']:.2f})")
print("\n=== VERDICT TABLE: bi-weekly 2x vs the live cash-exit rule ===")
for wlbl, _ in WINDOWS:
    c = store[(wlbl, "Cash exit (LIVE rule)")]; h = store[(wlbl, "BI-WEEKLY 2x (~14 DTE)")]
    print(f"  {wlbl}")
    print(f"     return  {h['net_cagr']-c['net_cagr']:+.1f}pp   drawdown {h['maxdd']-c['maxdd']:+.1f}pp   "
          f"Calmar {h['net_calmar']:.2f} vs {c['net_calmar']:.2f} "
          f"({'HEDGE WINS' if h['net_calmar']>c['net_calmar'] else 'CASH EXIT WINS'})")
