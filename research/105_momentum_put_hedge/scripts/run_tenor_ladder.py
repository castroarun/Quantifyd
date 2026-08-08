"""Tenor ladder for the put hedge: does buying the NEXT week's expiry (~14 DTE) beat the current
week's (~7 DTE)? Slower theta decay and half the rolls, versus less gamma. Corrected engine
(daily re-sizing + unified gate rule). Same 2019-2026 window so all tenors are comparable."""
import importlib.util, json
import pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "g5", "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts/run_hedge_g5.py")
g5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g5)
st = pd.Timestamp("2019-02-01")
print(f"\n{'book':>34} {'netCAGR':>8} {'MaxDD':>7} {'netCal':>7} {'#hedges':>8} {'hdays':>7} {'premium':>9}")
base = g5.run3(mode="cash_exit", start=st)
print(f"{'Cash exit (LIVE rule)':>34} {base['net_cagr']:>7.1f}% {base['maxdd']:>6.1f}% "
      f"{base['net_calmar']:>7.2f} {'-':>8} {'-':>7} {'-':>9}")
naked = g5.run3(mode="hold", start=st)
print(f"{'Hold naked (no puts)':>34} {naked['net_cagr']:>7.1f}% {naked['maxdd']:>6.1f}% "
      f"{naked['net_calmar']:>7.2f} {'-':>8} {'-':>7} {'-':>9}")
rows = {}
for tenor, lbl in (("weekly", "WEEKLY  (~7 DTE, current week)"),
                   ("biweekly", "BI-WEEKLY (~14 DTE, next week)"),
                   ("triweekly", "TRI-WEEKLY (~21 DTE)"),
                   ("monthly", "MONTHLY (~30 DTE)")):
    for ratio in (1.0, 2.0):
        r = g5.run3(mode="hedge", tenor=tenor, ratio=ratio, start=st)
        rows[(tenor, ratio)] = r
        print(f"{lbl + f'  {ratio}x':>34} {r['net_cagr']:>7.1f}% {r['maxdd']:>6.1f}% "
              f"{r['net_calmar']:>7.2f} {r['hedges']:>8} {r['hdays']:>7} {r['prem']:>8.1f}%")
best = max(rows.items(), key=lambda kv: kv[1]['net_calmar'])
print(f"\nBEST by net Calmar: {best[0][0]} at {best[0][1]}x -> netCalmar {best[1]['net_calmar']:.2f}, "
      f"netCAGR {best[1]['net_cagr']:.1f}%, MaxDD {best[1]['maxdd']:.1f}%")
print(f"(cash-exit baseline netCalmar {base['net_calmar']:.2f}; naked {naked['net_calmar']:.2f})")
