"""Debug: why is avg invested fraction low for the incumbent? Dump yearly invested
fraction, risk-off share, held count, and Donchian exit counts."""
import sys, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tn", str(HERE / "tn_sweep.py"))
tn = importlib.util.module_from_spec(spec); spec.loader.exec_module(tn)

ctx = tn.Ctx()
dates = ctx.dates
garr = ctx.gate_arr("NIFTYBEES", "sma100")
i0 = ctx.i0
print("risk-off share by year (gate raw):")
s = pd.Series(garr[i0:], index=dates[i0:])
print((s.groupby(s.index.year).mean() * 100).round(0).to_dict())

# monkey-patch a tracking run
inv = []
heldn = []
orig_run = tn.run

# rerun incumbent with per-day tracking by copying run's internals lightly:
# simplest: temporarily wrap by re-executing run but sampling nav pieces via closure is
# invasive; instead re-run with exit=None to isolate donchian effect, and gate=NONE.
for label, kw in [
    ("incumbent", dict(series="NIFTYBEES", cons="sma100", exit=("donch", 15))),
    ("gate_only_nodonch", dict(series="NIFTYBEES", cons="sma100", exit=None)),
    ("donch_only_nogate", dict(series="NONE", cons="none", exit=("donch", 15))),
]:
    r = tn.run(ctx, **kw)
    print(label, "wa_cagr", r["wa_cagr"], "wa_dd", r["wa_dd"], "avg_inv", r["avg_inv"],
          "donch", r["donch_exits"], "gate_ev", r["gate_events"])
