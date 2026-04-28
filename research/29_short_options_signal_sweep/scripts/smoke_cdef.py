"""One-shot smoke test for new signal generators C/D/E/F.

Prints signal counts over a 30-day window so we can sanity-check before the
full sweep. Not committed to results CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import logging
logging.disable(logging.WARNING)

from data_loader import load_5min, load_daily, INDEX_SYMBOL
from signals import (
    path_c_signals,
    path_d_signals,
    strategy_e_signals,
    strategy_f_signals,
)

START = "2024-03-04"
END = "2024-04-30"

nifty_5m = load_5min(INDEX_SYMBOL, START, END)
nifty_d = load_daily(INDEX_SYMBOL)
n_sessions = len(set(nifty_5m.index.normalize()))

print(f"Window {START} -> {END}  sessions={n_sessions}\n")

# Path C variants
print("--- Path C (range gate × RSI gate) ---")
for rng in (0.004, 0.006, 0.008, 0.010, None):
    for use_rsi in (False, True):
        sigs = list(path_c_signals(nifty_5m, nifty_d, range_threshold=rng, use_rsi=use_rsi))
        rng_tag = "off" if rng is None else f"{rng:.3f}"
        print(f"  rng={rng_tag:>5}  rsi={'Y' if use_rsi else 'N'}  -> {len(sigs)} signals")

# Path D variants
print("\n--- Path D (CPR convention × RSI thresholds) ---")
for conv in ("priceCPR", "cprDelta"):
    for r_lo, r_hi in ((40, 60), (35, 65), (30, 70)):
        sigs = list(path_d_signals(nifty_5m, nifty_d,
                                   cpr_convention=conv,
                                   rsi_low=float(r_lo), rsi_high=float(r_hi)))
        print(f"  {conv:>10s}  rsi={r_lo}/{r_hi}  -> {len(sigs)} signals")

# Strategy E on RELIANCE
print("\n--- Strategy E (RELIANCE 5-min, 4 filter modes) ---")
rel_5m = load_5min("RELIANCE", START, END)
rel_d = load_daily("RELIANCE")
for mode in ("base", "cpr", "rsi", "cpr_rsi"):
    sigs = list(strategy_e_signals(rel_5m, rel_d, filter_mode=mode, symbol="RELIANCE", timeframe="5min"))
    print(f"  {mode:>10s}  -> {len(sigs)} signals")

# Strategy F on RELIANCE
print("\n--- Strategy F (RELIANCE 5-min, both CPR conventions) ---")
for conv in ("priceCPR", "cprDelta"):
    sigs = list(strategy_f_signals(rel_5m, rel_d, cpr_convention=conv, symbol="RELIANCE", timeframe="5min"))
    print(f"  {conv:>10s}  -> {len(sigs)} signals")

print("\nSmoke OK")
