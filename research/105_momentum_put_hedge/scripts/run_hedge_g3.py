"""DECISIVE TEST: is the weekly-hedge win a TENOR effect or a REGIME (window) effect?
Run the MONTHLY hedge over the SAME 2019-2026 window as the weekly arms."""
import importlib.util
import pandas as pd
spec = importlib.util.spec_from_file_location("hs", "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts/run_hedge_sweep.py")
hs = importlib.util.module_from_spec(spec); spec.loader.exec_module(hs)
st19 = pd.Timestamp("2019-02-01")
print(f"\n{'config (all 2019-02 -> 2026)':>34} {'CAGR':>6} {'netCAGR':>8} {'MaxDD':>7} {'netCal':>7} {'prem':>7}", flush=True)
for lbl, kw in [
    ("A0_cash_exit (baseline)",      dict(mode="cash_exit", start=st19)),
    ("A1_hold_naked",                dict(mode="hold", start=st19)),
    ("MONTHLY hedge ATM r1.0",       dict(mode="hedge", struct="long", moneyness=0.0, tenor="monthly", ratio=1.0, start=st19)),
    ("MONTHLY hedge ATM r2.0",       dict(mode="hedge", struct="long", moneyness=0.0, tenor="monthly", ratio=2.0, start=st19)),
    ("WEEKLY hedge ATM r2.0",        dict(mode="hedge", struct="long", moneyness=0.0, tenor="weekly", ratio=2.0, start=st19)),
]:
    r = hs.run(**kw)
    print(f"{lbl:>34} {r['cagr']:>5.1f}% {r['net_cagr']:>7.1f}% {r['maxdd']:>6.1f}% {r['net_calmar']:>7.2f} {r['prem']:>7.1f}", flush=True)
# and the reverse: weekly-equivalent check is impossible pre-2019 (no weekly options) - state it
print("\n(weekly options do not exist before 2019, so the reverse test is impossible)", flush=True)
