#!/usr/bin/env python3
"""research/109: aggregate wave-1 trades. Costs 10bps RT; excess vs
time-matched slot baseline; gates; IS-half stability."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
RES = STUDY / "results"
COST = 0.0010

tr = pd.read_csv(RES / "wave1_trades.csv")
sl = pd.read_csv(RES / "wave1_slots.csv")
tr["date"] = pd.to_datetime(tr["date"])
print(f"trades: {len(tr):,} | baseline samples: {len(sl):,}")

# time-matched baseline per slot (pooled across names/days)
base = sl.groupby("slot")["ret_to_eod"].mean()
tr["dir"] = np.where(tr["cell"].str.endswith("_L"), 1, -1)
tr["gross_dir"] = tr["dir"] * tr["gross_ret"]
tr["bench"] = tr["dir"] * tr["slot"].map(base).fillna(0.0)
tr["net"] = tr["gross_dir"] - COST
tr["excess"] = tr["gross_dir"] - tr["bench"] - COST

IS_END = pd.Timestamp("2021-09-30")
H1_END = pd.Timestamp("2018-06-30")
tr["window"] = np.where(tr["date"] <= IS_END, "IS", "VAL")
is_tr = tr[tr["window"] == "IS"]

rows = []
for cell, g in is_tr.groupby("cell"):
    ex = g["excess"].to_numpy()
    if len(ex) < 500:
        continue
    t = ex.mean() / (ex.std() / np.sqrt(len(ex)) + 1e-12)
    names_pos = g.groupby("symbol")["excess"].mean()
    h1 = g[g["date"] <= H1_END]["excess"].mean()
    h2 = g[g["date"] > H1_END]["excess"].mean()
    rows.append(dict(cell=cell, n=len(ex),
                     net_bps=round(g["net"].mean() * 1e4, 1),
                     excess_bps=round(ex.mean() * 1e4, 1), t=round(t, 2),
                     names_pos=round((names_pos > 0).mean(), 2),
                     h1_bps=round((h1 or 0) * 1e4, 1),
                     h2_bps=round((h2 or 0) * 1e4, 1),
                     win=round((g["net"] > 0).mean(), 3)))
out = pd.DataFrame(rows).sort_values("t", ascending=False)
out.to_csv(RES / "wave1_cells.csv", index=False)
print(out.to_string(index=False))
gate = out[(out.excess_bps > 0) & (out.t >= 3.0) & (out.names_pos >= 0.55)
           & (np.sign(out.h1_bps) == np.sign(out.h2_bps))]
print(f"\nIS GATE PASSERS: {len(gate)}")
for _, r in gate.iterrows():
    print(f"  {r.cell:14s} n={r.n} net={r.net_bps} excess={r.excess_bps} "
          f"t={r.t} names+={r.names_pos} halves=({r.h1_bps},{r.h2_bps})")
# Val check for passers
print("\n=== VAL (2021-10..2023-12) for IS passers ===")
val = tr[tr["window"] == "VAL"]
for cell in gate.cell:
    g = val[val.cell == cell]
    if len(g) < 100:
        print(f"  {cell}: n={len(g)} too few")
        continue
    ex = g["excess"].to_numpy()
    t = ex.mean() / (ex.std() / np.sqrt(len(ex)) + 1e-12)
    print(f"  {cell:14s} n={len(g)} net={g['net'].mean()*1e4:.1f} "
          f"excess={ex.mean()*1e4:.1f} t={t:.2f}")
print("AGG DONE")
