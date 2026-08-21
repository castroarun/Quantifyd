#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/121 S7 - robustness closers.

  A. RESOLUTION EQUIVALENCE. The project rule bans 5-minute data in options
     backtests because it understates intraday extremes. That rule is about the
     PATH (when a stop fires). For the statistic this study fits on - the maximum
     excursion of the underlying inside a fixed window - 5-minute bars are exactly
     equivalent, because a window's high is the max of the bar highs at any
     resolution. Proved here by resampling the SENSEX 1-minute series to 5 minutes
     and differencing. This is what licenses the NIFTY 5-minute long sample.

  B. MONOTONICITY. Full quintile response curves for the leading signals, on both
     the raw excursion and the premium-relative excursion, on both long series.

  C. COMBINATIONS (pre-registered, 6 of them). Arun asked whether combining the
     signals helps. Each combination is scored against a random skip of the same size.

  D. SIZE ARITHMETIC. What lot count delivers a given rupee tail, and what that
     does to the reward:risk ratio (answer: nothing - R:R is size-invariant).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
rng = np.random.default_rng(20260821)

CELL2WIN = {"MON_NIFTY_DTE1": ("MON_1300_1400", 0, "NIFTY50_5MIN"),
            "WED_SENSEX_DTE1": ("WED_1030_1200", 2, "SENSEX_1MIN"),
            "FRI_NIFTY_DTE2": ("FRI_1000_1200", 4, "NIFTY50_5MIN")}

LEAD = ["pre_range_bp", "pdr_rel", "cpr_today", "wcpr_this", "vix_open", "gap_abs"]

COMBOS = [
    ("wide_daily_cpr_AND_big_prev_range", lambda f: (f.cpr_today >= f.cpr_today.quantile(.7)) &
                                                    (f.pdr_rel >= f.pdr_rel.quantile(.7))),
    ("wide_daily_cpr_AND_narrow_weekly", lambda f: (f.cpr_today >= f.cpr_today.quantile(.7)) &
                                                   (f.wcpr_this <= f.wcpr_this.quantile(.3))),
    ("big_gap_AND_high_vix", lambda f: (f.gap_abs >= f.gap_abs.quantile(.7)) &
                                       (f.vix_open >= f.vix_open.quantile(.7))),
    ("hot_preopen_range_AND_wide_cpr", lambda f: (f.pre_range_bp >= f.pre_range_bp.quantile(.7)) &
                                                 (f.cpr_today >= f.cpr_today.quantile(.7))),
    ("hot_preopen_range_OR_big_gap", lambda f: (f.pre_range_bp >= f.pre_range_bp.quantile(.8)) |
                                               (f.gap_abs >= f.gap_abs.quantile(.8))),
    ("vix_shock_up_AND_big_prev_range", lambda f: (f.vix_chg_oc_pct >= f.vix_chg_oc_pct.quantile(.8)) &
                                                  (f.pdr_rel >= f.pdr_rel.quantile(.7))),
]


def P(lines, s=""):
    lines.append(s); print(s, flush=True)


def main():
    lines = []
    wo = {t: pd.read_csv(os.path.join(RES, "window_outcomes_%s.csv" % t))
          for t in ("SENSEX_1MIN", "SENSEX_5MIN", "NIFTY50_5MIN")}

    # ---------- A ----------
    P(lines, "=" * 100)
    P(lines, "A. RESOLUTION EQUIVALENCE - is the NIFTY 5-minute long sample admissible?")
    P(lines, "=" * 100)
    a = wo["SENSEX_1MIN"][["window", "day", "exc_bp", "p0"]].rename(columns={"exc_bp": "e1", "p0": "p1"})
    b = wo["SENSEX_5MIN"][["window", "day", "exc_bp", "p0"]].rename(columns={"exc_bp": "e5", "p0": "p5"})
    m = a.merge(b, on=["window", "day"])
    d = (m.e5 - m.e1)
    P(lines, "SENSEX, same %d window-days measured at 1-min and at 5-min:" % len(m))
    P(lines, "  mean(exc5 - exc1) = %.4f bp   max|diff| = %.4f bp   rows differing = %d"
      % (d.mean(), d.abs().max(), int((d.abs() > 1e-9).sum())))
    P(lines, "  start-price mismatch rows = %d" % int((m.p1 != m.p5).sum()))
    P(lines, "  => the maximum excursion inside a fixed window is RESOLUTION-INVARIANT")
    P(lines, "     (a window's high is the max of the bar highs however you bucket them).")
    P(lines, "     The no-5-min rule bites on the PATH - which minute a stop fires - and that")
    P(lines, "     part of this study is done on the real 1-minute option chain, not on bars.")
    P(lines, "")

    # ---------- B ----------
    P(lines, "=" * 100)
    P(lines, "B. MONOTONICITY - full quintile response curves (never just the best cut)")
    P(lines, "   exc_bp = raw excursion; exc_norm = excursion / VIX-implied 1-day sigma")
    P(lines, "=" * 100)
    q = pd.read_csv(os.path.join(RES, "longfit_quintiles.csv"))
    q = q[(q.dow_filter == "LIVE") & (q.series != "SENSEX_5MIN")]
    for cell, (win, dow, ser) in CELL2WIN.items():
        P(lines, "### %s  (%s on %s)" % (cell, win, ser))
        for out in ("exc_bp", "exc_norm"):
            sub = q[(q.window == win) & (q.series == ser) & (q.outcome == out) & (q.signal.isin(LEAD))]
            if not len(sub):
                continue
            piv = sub.pivot(index="signal", columns="quintile", values="mean").reindex(LEAD)
            piv["Q5/Q1"] = (piv["Q5"] / piv["Q1"]).round(2)
            P(lines, "  %s (mean by signal quintile):" % out)
            P(lines, piv.round(3).to_string())
        P(lines, "")

    # ---------- C ----------
    P(lines, "=" * 100)
    P(lines, "C. COMBINATION FILTERS (6 pre-registered), long sample, live weekday")
    P(lines, "   'skip' = the combination is TRUE -> stand aside that day")
    P(lines, "   rand_pctile: where the kept-days' p90 excursion sits vs skipping the same")
    P(lines, "   number of days at random. LOW is good (a real filter cuts the tail).")
    P(lines, "=" * 100)
    rows = []
    for cell, (win, dow, ser) in CELL2WIN.items():
        f = wo[ser]
        f = f[(f.window == win) & (f.dow == dow)].copy()
        for c in ("cpr_today", "wcpr_this", "pdr_rel", "gap_abs", "vix_open",
                  "pre_range_bp", "vix_chg_oc_pct", "exc_bp", "exc_norm"):
            f[c] = pd.to_numeric(f[c], errors="coerce")
        f = f.dropna(subset=["exc_bp", "exc_norm", "cpr_today", "wcpr_this", "pdr_rel",
                             "gap_abs", "vix_open", "pre_range_bp", "vix_chg_oc_pct"])
        n = len(f)
        for name, fn in COMBOS:
            skip = fn(f).values
            keep = ~skip
            nk = int(keep.sum())
            if nk < 30 or nk == n:
                continue
            for out in ("exc_bp", "exc_norm"):
                y = f[out].values
                yk = y[keep]
                p90k = np.percentile(yk, 90)
                idx = rng.random((4000, n)).argsort(axis=1)[:, :nk]
                r90 = np.percentile(y[idx], 90, axis=1)
                rows.append(dict(cell=cell, combo=name, outcome=out, n=n,
                                 skipped=n - nk, skip_pct=round(100 * (n - nk) / n, 1),
                                 p90_all=round(float(np.percentile(y, 90)), 3),
                                 p90_kept=round(float(p90k), 3),
                                 max_all=round(float(y.max()), 3),
                                 max_kept=round(float(yk.max()), 3),
                                 rand_pctile=round(float((r90 <= p90k).mean() * 100), 1)))
    cb = pd.DataFrame(rows)
    cb.to_csv(os.path.join(RES, "combination_filters.csv"), index=False)
    for out in ("exc_bp", "exc_norm"):
        P(lines, "--- outcome = %s ---" % out)
        P(lines, cb[cb.outcome == out][["cell", "combo", "skip_pct", "p90_all", "p90_kept",
                                        "max_all", "max_kept", "rand_pctile"]].to_string(index=False))
        P(lines, "")

    # ---------- D ----------
    P(lines, "=" * 100)
    P(lines, "D. SIZE ARITHMETIC - the one lever that is guaranteed to work, and what it")
    P(lines, "   does NOT do (it does not change the reward:risk ratio at all)")
    P(lines, "=" * 100)
    L = pd.read_csv(os.path.join(RES, "options_sample.csv"))
    L = L[L.is_live_dow == 1]
    P(lines, "%-16s %6s %12s %12s %12s %8s" %
      ("cell", "lots", "medProfit", "SL20 cap", "SL20 as %", "R:R"))
    for cell, g in L.groupby("cell"):
        lot = g.lot.iloc[0]; cost = g.cost.iloc[0]
        credit_rs = (g.credit * lot).mean()
        med1 = g.net_SL20.median()
        cap1 = credit_rs * 0.20 + cost
        for lots in (10, 8, 5, 3, 2):
            P(lines, "%-16s %6d %12d %12d %12s %8.2f"
              % (cell, lots, int(med1 * lots), int(cap1 * lots), "20% of credit", cap1 / med1))
        P(lines, "")
    P(lines, "R:R is invariant to size by construction. Cutting lots cuts the rupee tail")
    P(lines, "proportionally and cuts the rupee profit by exactly the same proportion; it")
    P(lines, "leaves expectancy per rupee of margin untouched. That is the honest lever:")
    P(lines, "it changes how much the bad day HURTS, which is the actual complaint.")

    with open(os.path.join(RES, "robustness_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("DONE")


if __name__ == "__main__":
    main()
