"""EX-COVID, re-run with the CORRECTED engine (daily re-sizing + unified gate rule).
The first attempt used the original engine which still had the over-hedge bug, so its numbers were wrong.
Question: without a once-in-a-decade crash, does the put hedge cost meaningfully?"""
import importlib.util
import pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "g5", "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts/run_hedge_g5.py")
g5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g5)

WINDOWS = [("FULL weekly-era 2019-2026", pd.Timestamp("2019-02-01")),
           ("POST-COVID 2021-2026 (no once-in-a-decade crash)", pd.Timestamp("2021-01-01")),
           ("RECENT 2022-2026", pd.Timestamp("2022-01-01"))]
for wlbl, st in WINDOWS:
    print(f"\n=== {wlbl} ===")
    print(f"{'book':>26} {'netCAGR':>8} {'MaxDD':>7} {'netCal':>7} {'premium':>9} {'hedge days':>11}")
    rows = {}
    for lbl, kw in (("Cash exit (LIVE rule)", dict(mode="cash_exit", start=st)),
                    ("Hold naked (no puts)", dict(mode="hold", start=st)),
                    ("Hold + WEEKLY puts 2x", dict(mode="hedge", tenor="weekly", ratio=2.0, start=st)),
                    ("Hold + WEEKLY puts 1x", dict(mode="hedge", tenor="weekly", ratio=1.0, start=st)),
                    ("Hold + MONTHLY puts 2x", dict(mode="hedge", tenor="monthly", ratio=2.0, start=st))):
        r = g5.run3(**kw)
        rows[lbl] = r
        print(f"{lbl:>26} {r['net_cagr']:>7.1f}% {r['maxdd']:>6.1f}% {r['net_calmar']:>7.2f} "
              f"{r['prem']:>8.1f}% {r['hdays']:>11}")
    a, b = rows["Hold naked (no puts)"], rows["Hold + WEEKLY puts 2x"]
    print(f"  -> cost of the weekly hedge vs holding naked: "
          f"{b['net_cagr']-a['net_cagr']:+.1f}pp CAGR for {a['maxdd']-b['maxdd']:+.1f}pp less drawdown")
    c = rows["Cash exit (LIVE rule)"]
    print(f"  -> weekly hedge vs YOUR LIVE RULE:            "
          f"{b['net_cagr']-c['net_cagr']:+.1f}pp CAGR, drawdown {b['maxdd']-c['maxdd']:+.1f}pp, "
          f"Calmar {b['net_calmar']:.2f} vs {c['net_calmar']:.2f}")
