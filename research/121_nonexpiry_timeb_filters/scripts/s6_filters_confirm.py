#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/121 S6 - ROUTE A on the rupee sample: does any pre-registered filter
improve the live cells, and does it beat randomly skipping the same number of days?

Nothing is FITTED here. The candidate list and the thresholds were pre-registered in
the STATUS-MD and fitted (where a long sample exists) in S3. This script only asks
whether the same rules, applied to real option P&L, do anything - and every accepted
rule must clear a random-skip control of identical frequency.

Three families are tested:
  1. LONG-FITTABLE day signals (CPR daily/weekly, gap, prev-day range, ATR, VIX level
     and VIX shocks, plus the same-session pre-window range).
  2. OPTION-BOOK day signals (PCR OI / volume, ATM-vs-wing OI share, OI walls,
     max-pain drift, overnight OI and PCR changes). These have NO long sample - the
     chain recorder starts 2026-04-20 - so they can only ever be tested on ~16 days
     per cell, which is exactly the regime that manufactures winners. They are
     therefore ALSO tested pooled across cells and against placebos, per research/115.
  3. PLACEBOS (research/115's lesson): `placebo_noise` carries no information at all;
     `placebo_prepath` is the signed pre-window index return - zero option information,
     but mechanically embedded in any PCR/OI measurement taken intraday. A real option
     signal must beat both.

Outputs
  results/filter_confirm_pooled.csv   IC of every signal vs P&L (% of credit) and vs
                                      the premium-relative excursion, pooled + per cell
  results/filter_confirm_skip.csv     skip rules vs an exact random-skip null
  results/filter_report.txt
"""
import os, csv, itertools, math
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
rng = np.random.default_rng(20260821)

LONG_SIGS = ["cpr_today", "cpr_prev", "wcpr_this", "wcpr_prev", "gap_abs", "gap_pct",
             "pdr_pct", "pdr_rel", "atr14_pct", "vix_open", "vix_chg_oc_pct",
             "vix_chg_oc_pts", "vix_chg_cc_pct", "vix_chg_cc_pts",
             "pre_move_bp", "pre_range_bp"]
BOOK_SIGS = ["pcr_oi_all", "pcr_oi_atm", "pcr_vol_all", "oi_atm_share",
             "dist_ce_wall_bp", "dist_pe_wall_bp", "wall_span_bp", "maxpain_drift_bp",
             "d_pcr_oi_all", "d_oi_total_pct", "d_oi_atm_pct", "d_oi_wing_pct"]
PLACEBOS = ["placebo_noise", "placebo_prepath"]
SKIP_FRACS = [0.25, 0.33, 0.50]
NDRAW = 20000


def main():
    lines = []

    def P(s=""):
        lines.append(s); print(s, flush=True)

    d = pd.read_csv(os.path.join(RES, "options_sample.csv"))
    L = d[d.is_live_dow == 1].copy()
    # outcome 1: net P&L under the LIVE rule, as % of the credit collected
    L["credit_rs"] = L.credit * L.lot
    L["pnl_pct_credit"] = L.net_SL20 / L.credit_rs * 100.0
    # outcome 2: how big the move was relative to the premium sold (the R:R driver)
    L["exc_over_credit"] = L.und_exc_bp / (L.credit_pct_spot * 100.0)
    # placebos
    L["placebo_noise"] = rng.normal(size=len(L))
    L["placebo_prepath"] = L.und_net_bp * 0  # filled below per row from the pre-window path
    L["placebo_prepath"] = L["pre_move_bp"].fillna(0) * np.sign(L["gap_pct"].fillna(0))

    ALL = LONG_SIGS + BOOK_SIGS + PLACEBOS

    P("=" * 110)
    P("DATA - the three live non-expiry cells on the recorded chain (live weekday only)")
    P("=" * 110)
    for cell, g in L.groupby("cell"):
        P("%-16s n=%2d  %s .. %s   median P&L %.2f%% of credit   worst %.2f%%   win %.0f%%"
          % (cell, len(g), g.day.min(), g.day.max(), g.pnl_pct_credit.median(),
             g.pnl_pct_credit.min(), (g.pnl_pct_credit > 0).mean() * 100))
    P("")
    P("Coverage of the option-book signals (non-null share):")
    for s in BOOK_SIGS:
        P("   %-18s %.0f%%" % (s, 100.0 * L[s].notna().mean()))
    P("")

    # ---------------- information coefficients ----------------
    rows = []
    for scope, sub in [("POOLED", L)] + [(c, g) for c, g in L.groupby("cell")]:
        for out in ("pnl_pct_credit", "exc_over_credit"):
            y = sub[out].values
            for s in ALL:
                x = sub[s].values.astype(float)
                m = ~(np.isnan(x) | np.isnan(y))
                if m.sum() < 10:
                    continue
                rho, p = stats.spearmanr(x[m], y[m])
                rows.append(dict(scope=scope, outcome=out, signal=s, n=int(m.sum()),
                                 spearman=round(float(rho), 4), p=round(float(p), 5)))
    ic = pd.DataFrame(rows)
    ic.to_csv(os.path.join(RES, "filter_confirm_pooled.csv"), index=False)
    P("=" * 110)
    P("INFORMATION COEFFICIENTS on the options sample (pooled across the 3 cells, n=48)")
    P("  a positive IC vs pnl_pct_credit  = higher signal -> BETTER day  (so you would skip LOW)")
    P("  a positive IC vs exc_over_credit = higher signal -> BIGGER move relative to premium")
    P("=" * 110)
    t = ic[ic.scope == "POOLED"].pivot(index="signal", columns="outcome",
                                       values="spearman").reindex(ALL)
    tp = ic[ic.scope == "POOLED"].pivot(index="signal", columns="outcome",
                                        values="p").reindex(ALL)
    t.columns = [c + "_rho" for c in t.columns]
    tp.columns = [c + "_p" for c in tp.columns]
    P(pd.concat([t, tp], axis=1).round(3).to_string())
    P("")

    # ---------------- skip rules vs an EXACT random-skip null ----------------
    srows = []
    for cell, g in L.groupby("cell"):
        g = g.reset_index(drop=True)
        n = len(g)
        tot_all = g.net_SL20.sum() * 10
        med_all = g.net_SL20.median() * 10
        worst_all = g.net_SL20.min() * 10
        for frac in SKIP_FRACS:
            k = max(1, int(round(n * frac)))
            nk = n - k
            # exact null over all C(n,nk) subsets if small, else sampled
            combs = list(itertools.combinations(range(n), nk))
            if len(combs) > NDRAW:
                combs = [tuple(rng.choice(n, size=nk, replace=False)) for _ in range(NDRAW)]
            arr = g.net_SL20.values
            nulls_tot = np.array([arr[list(c)].sum() for c in combs]) * 10
            nulls_worst = np.array([arr[list(c)].min() for c in combs]) * 10
            for s in ALL:
                x = g[s].values.astype(float)
                if np.isnan(x).sum() > n // 3:
                    continue
                xf = np.where(np.isnan(x), np.nanmedian(x), x)
                for side in ("skip_high", "skip_low"):
                    order = np.argsort(-xf) if side == "skip_high" else np.argsort(xf)
                    keep = np.sort(order[k:])
                    v = arr[keep]
                    tot = v.sum() * 10
                    srows.append(dict(
                        cell=cell, signal=s, side=side, n=n, skipped=k,
                        total_all=int(tot_all), total_kept=int(tot),
                        pnl_retained_pct=round(100.0 * tot / tot_all, 1) if tot_all else None,
                        median_all=int(med_all), median_kept=int(np.median(v) * 10),
                        worst_all=int(worst_all), worst_kept=int(v.min() * 10),
                        rand_total_mean=int(nulls_tot.mean()),
                        rand_total_pctile=round(100.0 * (nulls_tot <= tot).mean(), 1),
                        rand_worst_mean=int(nulls_worst.mean()),
                        rand_worst_pctile=round(100.0 * (nulls_worst <= v.min() * 10).mean(), 1)))
    sk = pd.DataFrame(srows)
    sk.to_csv(os.path.join(RES, "filter_confirm_skip.csv"), index=False)

    P("=" * 110)
    P("SKIP RULES vs an EXACT RANDOM-SKIP NULL (skip 33% of the days of each cell)")
    P("  rand_total_pctile = where the filter's kept-P&L sits in the distribution of")
    P("  randomly skipping the same number of days. >95 would be a real filter;")
    P("  ~50 means it is doing nothing; <50 means it is worse than random.")
    P("=" * 110)
    s33 = sk[sk.skipped == sk.groupby("cell").skipped.transform(lambda v: sorted(set(v))[1])]
    for cell, g in s33.groupby("cell"):
        g = g.sort_values("rand_total_pctile", ascending=False)
        P("### " + cell)
        P(g[["signal", "side", "skipped", "total_all", "total_kept", "pnl_retained_pct",
             "worst_all", "worst_kept", "rand_total_pctile", "rand_worst_pctile"]]
          .head(8).to_string(index=False))
        P("   ... worst 3:")
        P(g[["signal", "side", "total_kept", "pnl_retained_pct", "rand_total_pctile"]]
          .tail(3).to_string(index=False))
        P("")

    # how many rules cleared the 95th percentile anywhere?
    n_tests = len(sk)
    winners = sk[(sk.rand_total_pctile >= 95) & (sk.pnl_retained_pct >= 100)]
    P("Rules tried in this confirmation stage: %d  (signals %d x sides 2 x skip-fracs %d x cells 3)"
      % (n_tests, len(ALL), len(SKIP_FRACS)))
    P("Rules whose kept-P&L beat 95%% of random skips AND retained >=100%% of total P&L: %d"
      % len(winners))
    if len(winners):
        P(winners[["cell", "signal", "side", "skipped", "pnl_retained_pct",
                   "worst_all", "worst_kept", "rand_total_pctile"]].to_string(index=False))
    P("")
    P("Expected number of such rules under the pure null (5%% x 2 outcomes-ish): ~%.0f"
      % (0.05 * n_tests))

    with open(os.path.join(RES, "filter_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("DONE")


if __name__ == "__main__":
    main()
