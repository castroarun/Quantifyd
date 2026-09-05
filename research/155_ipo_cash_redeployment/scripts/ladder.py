"""research/155 — cost-ladder pairing done CORRECTLY: each redeployment arm is compared
against the incumbent AT THE SAME COST, path by path (a 40 bps arm must not be scored against
a 25 bps incumbent)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
p = pd.read_csv(RES / "paths.csv").drop_duplicates(subset=["cell", "path"], keep="last")

BASE = {25: "P4_A_c25", 40: "P4_A_c40", 60: "P4_A_c60"}
ARMS = {
    "GATED (E25, OA, reserve 0, monthly, LIFO)": {25: "P5_E25_OA_c25_%s", 40: "P5_E25_OA_c40_%s",
                                                  60: "P5_E25_OA_c60_%s"},
    "GATED (E25, OA, reserve 2, monthly, LIFO)": {25: "P4_E25_OA_c25_%s", 40: "P4_E25_OA_c40_%s",
                                                  60: "P4_E25_OA_c60_%s"},
    "CONTINUOUS (OA, reserve 2, monthly, LIFO)": {25: "P4_OA_c25_%s", 40: "P4_OA_c40_%s",
                                                  60: "P4_OA_c60_%s"},
}
print(f"{'arm':<44} {'tax':>5} {'bps':>4} | {'blend CAGR':>10} {'MaxDD':>8} {'Calmar':>7} | "
      f"{'dCAGR':>7} {'dCalmar':>8} | {'winCAGR':>8} {'winCal':>7}")
out = []
for name, m in ARMS.items():
    for tax in ("full", "txn"):
        for bps in (25, 40, 60):
            c = m[bps] % tax
            a = p[p.cell == c].set_index("path")
            b = p[p.cell == BASE[bps]].set_index("path")
            if not len(a) or not len(b):
                print(f"  MISSING {c}")
                continue
            ix = a.index.intersection(b.index)
            dc = a.loc[ix, "b_cagr"] - b.loc[ix, "b_cagr"]
            dk = a.loc[ix, "b_calmar"] - b.loc[ix, "b_calmar"]
            print(f"{name:<44} {tax:>5} {bps:>4} | {a.b_cagr.median():10.3f} "
                  f"{a.b_dd.median():8.3f} {a.b_calmar.median():7.3f} | {dc.median():+7.3f} "
                  f"{dk.median():+8.4f} | {int((dc>0).sum()):5d}/{len(ix)} "
                  f"{int((dk>0).sum()):4d}/{len(ix)}")
            out.append(dict(arm=name, tax=tax, bps=bps, cagr=a.b_cagr.median(),
                            dd=a.b_dd.median(), calmar=a.b_calmar.median(),
                            d_cagr=dc.median(), d_calmar=dk.median(),
                            win_cagr=int((dc > 0).sum()), win_calmar=int((dk > 0).sum()),
                            n=len(ix)))
    print()
print(f"{'INCUMBENT (idle -> cash)':<44} {'-':>5} {'':>4} | " + " " * 8)
for bps in (25, 40, 60):
    b = p[p.cell == BASE[bps]]
    print(f"{'  incumbent':<44} {'-':>5} {bps:>4} | {b.b_cagr.median():10.3f} "
          f"{b.b_dd.median():8.3f} {b.b_calmar.median():7.3f}")
pd.DataFrame(out).round(4).to_csv(RES / "cost_ladder.csv", index=False)
