"""research/154 P7 - the weight frontier, enumerated rather than cherry-picked.

Every weight vector on a 5% grid over {OA, TN, IPO, GOLD} (and a 10% grid over
{OA, TN, IPO, GOLD, MYB}) is evaluated on ALL THREE panels, paired on the same 360 paths,
against three nulls:

  1. the deployed TN+OA 50-50 pair          (is it better than what we run today?)
  2. a plain CASH null at the same weight   (is the "diversification" just de-levering?)
  3. an IPO BETA-MATCHED null               (IPO is only ~19.6% invested - replace it with
                                             19.6% OA + 80.4% cash at the same weight)

A vector is ADMITTED only if, on ALL THREE panels: CAGR >= the deployed pair's CAGR,
Calmar > the pair's on >= 288/360 paths, and it beats BOTH nulls on >= 288/360 paths.
That is the pre-registered bar applied literally, with no post-hoc softening.
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from blend_matrix import PANELS, NPATH, Panel, build_monthly, load_daily, path_stats, RES

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 300)

IPO_INVESTED = 0.196
WIN = 288          # 80% of 360 paired paths


def grid(names, step):
    n = len(names)
    units = int(round(1 / step))
    for comb in itertools.product(range(units + 1), repeat=n - 1):
        if sum(comb) > units:
            continue
        w = list(comb) + [units - sum(comb)]
        yield {names[i]: w[i] * step for i in range(n) if w[i] > 0}


def evaluate(family, step, tag):
    d = load_daily()
    mm = build_monthly(d)
    pans, base = {}, {}
    for k in PANELS:
        if not set(family) <= set(PANELS[k]["members"]):
            continue
        pn = Panel(k, mm)
        pans[k] = pn
        base[k] = path_stats(pn.blend({"OA": .5, "TN": .5}), pn.years)
    vecs = [w for w in grid(family, step) if len(w) >= 2]
    print(f"\n{'='*118}\n{tag}: {len(vecs)} weight vectors x {len(pans)} panels "
          f"x {NPATH} paired paths = {len(vecs)*len(pans)} cells\n{'='*118}", flush=True)

    rows = []
    for w in vecs:
        rec = dict(weights="|".join(f"{s}:{v:.2f}" for s, v in sorted(w.items())),
                   n=len(w), **{f"w_{s}": round(w.get(s, 0.0), 3) for s in family})
        ok = True
        for k, pn in pans.items():
            bc, bdd, bk = base[k]
            c, dd, kk = path_stats(pn.blend(w), pn.years)
            cn = {}
            for s, v in w.items():
                cn[s if s in ("OA", "TN") else "CASH"] = \
                    cn.get(s if s in ("OA", "TN") else "CASH", 0) + v
            _, _, ck = path_stats(pn.blend(cn), pn.years)
            bm = dict(w)
            wi = bm.pop("IPO", 0.0)
            if wi:
                bm["OA"] = bm.get("OA", 0) + wi * IPO_INVESTED
                bm["CASH"] = bm.get("CASH", 0) + wi * (1 - IPO_INVESTED)
                _, _, mk = path_stats(pn.blend(bm), pn.years)
            else:
                mk = ck
            wb = int(np.nansum(kk > bk)); wc = int(np.nansum(kk > ck)); wm_ = int(np.nansum(kk > mk))
            rec.update({f"{k}_cagr": round(float(np.median(c)), 2),
                        f"{k}_dcagr": round(float(np.median(c - bc)), 2),
                        f"{k}_dd": round(float(np.median(dd)), 2),
                        f"{k}_calmar": round(float(np.nanmedian(kk)), 3),
                        f"{k}_dcalmar": round(float(np.nanmedian(kk - bk)), 3),
                        f"{k}_winb": wb, f"{k}_winc": wc, f"{k}_winm": wm_})
            if not (np.median(c) >= np.median(bc) and wb >= WIN and wc >= WIN and wm_ >= WIN):
                ok = False
        rec["ADMITTED"] = ok
        rec["minCalmar"] = round(min(rec[f"{k}_calmar"] for k in pans), 3)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"p7_frontier_{tag}.csv", index=False)
    adm = df[df.ADMITTED].sort_values("minCalmar", ascending=False)
    print(f"ADMITTED: {len(adm)} of {len(df)} weight vectors clear the pre-registered bar "
          f"on ALL {len(pans)} panels")
    show = ["weights"] + [c for k in pans for c in
                          (f"{k}_cagr", f"{k}_dcagr", f"{k}_dd", f"{k}_calmar",
                           f"{k}_winb", f"{k}_winc", f"{k}_winm")]
    print(adm[show].head(30).to_string(index=False))
    if len(adm) == 0:
        print("\n-- nothing admitted; the 12 highest-minCalmar vectors and WHY they fail --")
        near = df.sort_values("minCalmar", ascending=False).head(12)
        print(near[show + ["ADMITTED"]].to_string(index=False))
    return df, adm


if __name__ == "__main__":
    evaluate(["OA", "TN", "IPO", "GOLD"], 0.05, "OA_TN_IPO_GOLD")
    evaluate(["OA", "TN", "IPO", "GOLD", "MYB"], 0.10, "plusMYB")
    print("\nFRONTIER DONE")
