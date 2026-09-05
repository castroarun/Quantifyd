"""research/154 P4-P5 - the honest controls.

1. Sleeve exposure audit: how much of each sleeve's "diversification" is just idle cash.
2. BETA-MATCHED NULL for the IPO sleeve (the control the plain cash-null cannot give):
   IPO is only ~19.6% invested, so replace it with (19.6% OA + 80.4% cash) at the SAME
   weight. If the combination stops winning, IPO's blend value was de-levering.
3. Baseline drawdown timing - is the DD improvement one window or many?
4. Per-window rows (2008 / 2020 crash, 2018 / 2022H1 grind) for the finalists.
5. The two registered open questions from r/152.
6. Daily-marked robustness (honest intra-month drawdown).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from blend_matrix import (PANELS, NPATH, NSEED, NOFF, Panel, build_monthly, load_daily,
                          path_stats, cumnav, RES, WINDOWS)

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)

# IPO's measured average invested fraction of NAV (research/153 G3 print)
IPO_INVESTED = 0.196

FINALISTS = [
    ("TN+OA 50-50 (DEPLOYED BASELINE)", {"OA": .50, "TN": .50}),
    ("OA+TN+IPO+GOLD 25/25/25/25", {"OA": .25, "TN": .25, "IPO": .25, "GOLD": .25}),
    ("OA+TN+IPO+GOLD 30/30/20/20", {"OA": .30, "TN": .30, "IPO": .20, "GOLD": .20}),
    ("OA+TN+IPO+GOLD 40/40/10/10", {"OA": .40, "TN": .40, "IPO": .10, "GOLD": .10}),
    ("OA+IPO+GOLD 33/33/33", {"OA": 1 / 3, "IPO": 1 / 3, "GOLD": 1 / 3}),
    ("OA+TN+MYB+IPO+GOLD 20 each", {"OA": .20, "TN": .20, "MYB": .20, "IPO": .20, "GOLD": .20}),
    ("OA+TN+GOLD 45/45/10 (r147 rec)", {"OA": .45, "TN": .45, "GOLD": .10}),
    ("OA+TN+GOLD 40/40/20", {"OA": .40, "TN": .40, "GOLD": .20}),
    ("OA+MYB 50-50 (r152 question)", {"OA": .50, "MYB": .50}),
    ("TN+OA 80 / GOLD 10 / MYB 10", {"OA": .40, "TN": .40, "GOLD": .10, "MYB": .10}),
    ("OA+TN+VCP 33 each", {"OA": 1 / 3, "TN": 1 / 3, "VCP": 1 / 3}),
]


def summ(pn, wm):
    nav = pn.blend(wm)
    c, dd, k = path_stats(nav, pn.years)
    return nav, c, dd, k


def line(name, c, dd, k, bc=None, bdd=None, bk=None):
    s = (f"{name:<38} CAGR {np.median(c):6.2f} [{c.min():6.2f}..{c.max():6.2f}]  "
         f"DD {np.median(dd):7.2f} (worst {dd.min():7.2f})  Calmar {np.nanmedian(k):5.3f}")
    if bc is not None:
        s += (f"  | vs base dCAGR {np.median(c-bc):+6.2f}  dDD {np.median(dd-bdd):+6.2f}  "
              f"dCalmar {np.nanmedian(k-bk):+6.3f}  wins {int(np.nansum(k>bk)):3d}/{NPATH}")
    return s


def main():
    d = load_daily()
    mm = build_monthly(d)

    # ---------------------------------------------------------------- 1. exposure audit
    print("=" * 110)
    print("1. SLEEVE EXPOSURE AUDIT - how much of the 'diversification' is idle cash?")
    print("   (a year that returns ~5.0% with ~0 intra-year drawdown is 100% CASH at the "
          "5% idle yield)")
    print("=" * 110)
    rows = []
    for s in ("OA", "TN", "VCP", "MYB", "IPO"):
        nav = mm[s]
        yr = nav.groupby(nav.index.year).last()
        r = (yr.pct_change() * 100).median(axis=1)
        rows.append(pd.Series(r, name=s))
    gy = mm["GOLD"].groupby(mm["GOLD"].index.year).last().pct_change().iloc[:, 0] * 100
    rows.append(pd.Series(gy, name="GOLD"))
    ex = pd.concat(rows, axis=1).round(1)
    ex["IPO_cashlike"] = np.where(ex["IPO"].between(3.0, 7.0), "<-- CASH", "")
    print(ex.to_string())
    ex.to_csv(RES / "p4_yearly_sleeves.csv")

    # ------------------------------------------------- 2/3/4. panels, controls, windows
    allrows, winrows = [], []
    for key in PANELS:
        pn = Panel(key, mm)
        bnav, bc, bdd, bk = summ(pn, {"OA": .5, "TN": .5})
        print("\n" + "=" * 110)
        print(f"PANEL {key}: {pn.months[0]} -> {pn.months[-1]}  ({pn.years:.1f}y)  "
              f"members {pn.cfg['members']}")
        print("=" * 110)
        # drawdown timing of the baseline (median path)
        med = np.median(bnav, axis=1)
        run = np.maximum.accumulate(med)
        ddv = med / run - 1
        i = int(np.argmin(ddv))
        peak = int(np.argmax(run[:i + 1] == run[i])) if i else 0
        print(f"baseline TN+OA 50-50 max drawdown {ddv[i]*100:.2f}% : peak "
              f"{pn.months[peak]} -> trough {pn.months[i]}")
        # top 5 worst 12-month rolling stretches of the baseline
        print(line("  TN+OA 50-50 (baseline)", bc, bdd, bk))

        for name, wm in FINALISTS:
            if not set(wm) <= set(pn.cfg["members"]):
                continue
            nav, c, dd, k = summ(pn, wm)
            print(line("  " + name, c, dd, k, bc, bdd, bk))
            # standard cash-null
            cn = {}
            for s, w in wm.items():
                key2 = s if s in ("OA", "TN") else "CASH"
                cn[key2] = cn.get(key2, 0) + w
            _, cc, cdd, ck = summ(pn, cn)
            # beta-matched null: IPO -> 19.6% OA + 80.4% cash (MYB/GOLD unchanged)
            bm = dict(wm)
            wipo = bm.pop("IPO", 0.0)
            if wipo:
                bm["OA"] = bm.get("OA", 0) + wipo * IPO_INVESTED
                bm["CASH"] = bm.get("CASH", 0) + wipo * (1 - IPO_INVESTED)
                _, mc, mdd, mk = summ(pn, bm)
            else:
                mc = mdd = mk = None
            print(f"{'':40}   cash-null Calmar {np.nanmedian(ck):5.3f} "
                  f"(dCalmar {np.nanmedian(k-ck):+6.3f}, wins {int(np.nansum(k>ck)):3d}/{NPATH})"
                  + (f"  | IPO beta-matched null Calmar {np.nanmedian(mk):5.3f} "
                     f"(dCalmar {np.nanmedian(k-mk):+6.3f}, "
                     f"wins {int(np.nansum(k>mk)):3d}/{NPATH})" if mk is not None else ""))
            allrows.append(dict(
                panel=key, book=name,
                weights="|".join(f"{s}:{w:.2f}" for s, w in sorted(wm.items())),
                cagr=round(float(np.median(c)), 2), cagr_lo=round(float(c.min()), 2),
                cagr_hi=round(float(c.max()), 2), dd=round(float(np.median(dd)), 2),
                dd_worst=round(float(dd.min()), 2), calmar=round(float(np.nanmedian(k)), 3),
                d_cagr=round(float(np.median(c - bc)), 2),
                d_dd=round(float(np.median(dd - bdd)), 2),
                d_calmar=round(float(np.nanmedian(k - bk)), 3),
                wins=f"{int(np.nansum(k>bk))}/{NPATH}",
                cashnull_calmar=round(float(np.nanmedian(ck)), 3),
                d_vs_cashnull=round(float(np.nanmedian(k - ck)), 3),
                cashnull_wins=f"{int(np.nansum(k>ck))}/{NPATH}",
                betanull_calmar=(round(float(np.nanmedian(mk)), 3) if mk is not None else None),
                d_vs_betanull=(round(float(np.nanmedian(k - mk)), 3) if mk is not None else None),
                betanull_wins=(f"{int(np.nansum(k>mk))}/{NPATH}" if mk is not None else None)))
            row = dict(panel=key, book=name)
            for wn, (a, b) in WINDOWS.items():
                r_, dd_ = pn.window(nav, a, b)
                row[f"{wn} ret"] = None if r_ != r_ else round(r_, 1)
                row[f"{wn} dd"] = None if dd_ != dd_ else round(dd_, 1)
            winrows.append(row)
        # single sleeves for the window table
        for s in pn.cfg["members"]:
            nav, c, dd, k = summ(pn, {s: 1.0})
            row = dict(panel=key, book=f"{s} alone")
            for wn, (a, b) in WINDOWS.items():
                r_, dd_ = pn.window(nav, a, b)
                row[f"{wn} ret"] = None if r_ != r_ else round(r_, 1)
                row[f"{wn} dd"] = None if dd_ != dd_ else round(dd_, 1)
            winrows.append(row)
            if key == "B":
                print(line("  [single] " + s, c, dd, k, bc, bdd, bk))

    pd.DataFrame(allrows).to_csv(RES / "p4_finalists.csv", index=False)
    wdf = pd.DataFrame(winrows)
    wdf.to_csv(RES / "p5_windows.csv", index=False)
    print("\n" + "=" * 110)
    print("4. PER-WINDOW behaviour (median across 360 paths; monthly marks)")
    print("   MYB cannot be evaluated in 2008 at all (its history starts 2010-01);")
    print("   GOLD pre-2015 is the labelled reconstruction, not the real instrument.")
    print("=" * 110)
    for key in PANELS:
        print(f"\n--- panel {key} ---")
        print(wdf[wdf.panel == key].drop(columns=["panel"]).to_string(index=False))

    # -------------------------------------------------- 5. registered question 2
    print("\n" + "=" * 110)
    print("5. REGISTERED QUESTION (r/152 exploratory): 80% TN+OA / 10% GOLD / 10% MYB")
    print("   tested against a GOLD-ONLY null at the same total satellite weight.")
    print("=" * 110)
    for key in ("A", "C"):
        pn = Panel(key, mm)
        _, bc, bdd, bk = summ(pn, {"OA": .5, "TN": .5})
        tests = [
            ("80/10/10  OA .40 TN .40 GOLD .10 MYB .10", {"OA": .40, "TN": .40, "GOLD": .10, "MYB": .10}),
            ("GOLD-ONLY null: OA .40 TN .40 GOLD .20", {"OA": .40, "TN": .40, "GOLD": .20}),
            ("GOLD-ONLY at same gold wt: OA .45 TN .45 GOLD .10", {"OA": .45, "TN": .45, "GOLD": .10}),
            ("MYB-ONLY null: OA .40 TN .40 MYB .20", {"OA": .40, "TN": .40, "MYB": .20}),
            ("CASH null: OA .40 TN .40 CASH .20", {"OA": .40, "TN": .40, "CASH": .20}),
        ]
        print(f"\n--- panel {key} ({pn.months[0]} -> {pn.months[-1]}) ---")
        ref = None
        for nm, wm in tests:
            _, c, dd, k = summ(pn, wm)
            extra = ""
            if ref is not None:
                extra = (f"   vs the 80/10/10: dCalmar {np.nanmedian(ref[2]-k):+6.3f} "
                         f"(80/10/10 wins {int(np.nansum(ref[2]>k)):3d}/{NPATH})")
            print(line("  " + nm, c, dd, k, bc, bdd, bk) + extra)
            if ref is None:
                ref = (c, dd, k)

    # -------------------------------------------------- 6. registered question 1
    print("\n" + "=" * 110)
    print("6. REGISTERED QUESTION (r/152): does MYB+OA beat the deployed TN+OA pair?")
    print("   The 2010+ comparison is re-run, and then the SAME question is asked of the")
    print("   sleeves that CAN be tested through 2008, to see whether dropping TN is safe.")
    print("=" * 110)
    for key in PANELS:
        pn = Panel(key, mm)
        _, bc, bdd, bk = summ(pn, {"OA": .5, "TN": .5})
        print(f"\n--- panel {key} ({pn.months[0]} -> {pn.months[-1]}) ---")
        for nm, wm in [("OA+MYB 50-50", {"OA": .5, "MYB": .5}),
                       ("OA+IPO 50-50", {"OA": .5, "IPO": .5}),
                       ("OA+GOLD 50-50", {"OA": .5, "GOLD": .5}),
                       ("OA alone (no TN at all)", {"OA": 1.0}),
                       ("OA+TN 50-50 (deployed)", {"OA": .5, "TN": .5})]:
            if not set(wm) <= set(pn.cfg["members"]):
                print(f"  {nm:<38} NOT TESTABLE on this panel "
                      f"({[s for s in wm if s not in pn.cfg['members']]} absent)")
                continue
            _, c, dd, k = summ(pn, wm)
            print(line("  " + nm, c, dd, k, bc, bdd, bk))

    print("\nANALYZE DONE")


if __name__ == "__main__":
    main()
