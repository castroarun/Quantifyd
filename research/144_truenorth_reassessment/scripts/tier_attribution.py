"""Which cap-tier made True North's money, per year? (Arun 2026-09-04)

Runs the deployed-faithful incumbent (r/144 engine) with the daily holdings
recorder patched to capture VALUES, classifies each held name Nifty50 /
Next50 / Midcap via the official constituent CSVs, and attributes daily P&L
per tier per year. CAVEAT: constituent lists are CURRENT (point-in-time
membership history is not available) — past members of the 50 that have
since dropped out get tagged Midcap, so the Nifty-50 share is a floor,
slightly understated for older years.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# patched copy: record (index, ((sym, value), ...)) instead of just symbols
src = (HERE / 'tn_sweep.py').read_text()
src = src.replace('record.append((i, tuple(held)))',
                  'record.append((i, tuple((s, v[0]) for s, v in held.items())))')
(HERE / 'tn_attrib_engine.py').write_text(src)
import tn_attrib_engine as eng

ctx = eng.Ctx()
rec = []
row = eng.run(ctx, record=rec)   # incumbent defaults (NIFTYBEES sma100 weekly cash n=8)
print(f"incumbent run done: w0 CAGR {row['w0_cagr']}% dd {row['w0_dd']}%")

import csv as _csv
def members(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return {(r.get('Symbol') or '').strip() for r in _csv.DictReader(f)}
n50 = members('/home/arun/quantifyd/backtest_data/nifty50_official.csv')
nxt = members('/home/arun/quantifyd/backtest_data/niftynext50_official.csv')
def tier(s):
    return 'N50' if s in n50 else ('NEXT50' if s in nxt else 'MID')

C, sidx, dates = ctx.C, ctx.sidx, ctx.dates
pnl = {}      # year -> tier -> pnl (Rs on the engine's capital scale)
expo = {}     # year -> tier -> position-day value sum
for i, holds in rec:
    yr = dates[i].year
    prow, crow = C[i-1], C[i]
    for s, v in holds:
        j = sidx[s]
        p0, p1 = prow[j], crow[j]
        t = tier(s)
        expo.setdefault(yr, {}).setdefault(t, 0.0)
        pnl.setdefault(yr, {}).setdefault(t, 0.0)
        expo[yr][t] += v
        if p0 == p0 and p1 == p1 and p1 > 0:
            pnl[yr][t] += v * (1 - p0 / p1)

print(f"\n{'year':>5} | {'exposure share %':^24} | {'P&L share of year %':^26} | year P&L (Rs)")
print(f"{'':>5} | {'N50':>7}{'NEXT50':>8}{'MID':>7} | {'N50':>8}{'NEXT50':>9}{'MID':>8} |")
tot = {'N50': 0.0, 'NEXT50': 0.0, 'MID': 0.0}
for yr in sorted(pnl):
    e, p = expo[yr], pnl[yr]
    te = sum(e.values()) or 1.0
    tp = sum(p.values())
    for t in tot:
        tot[t] += p.get(t, 0.0)
    def sh(d, t, base):
        return 100 * d.get(t, 0.0) / base if base else 0.0
    pb = abs(tp) if abs(tp) > 1e-9 else 1.0
    sign = 1.0 if tp >= 0 else -1.0
    print(f"{yr:>5} | {sh(e,'N50',te):7.1f}{sh(e,'NEXT50',te):8.1f}{sh(e,'MID',te):7.1f} | "
          f"{sign*sh(p,'N50',pb):8.1f}{sign*sh(p,'NEXT50',pb):9.1f}{sign*sh(p,'MID',pb):8.1f} | "
          f"{tp:>13,.0f}")
tt = sum(tot.values())
print(f"\nTOTAL P&L split: N50 {100*tot['N50']/tt:.1f}%  NEXT50 {100*tot['NEXT50']/tt:.1f}%  "
      f"MID {100*tot['MID']/tt:.1f}%   (current-membership tiers — N50 share is a floor)")
