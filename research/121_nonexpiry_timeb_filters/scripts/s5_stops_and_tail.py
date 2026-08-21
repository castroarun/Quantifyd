#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/121 S5 - Route B (the stop ladder) and the SIZE OF THE REAL TAIL.

Two things this script establishes:

1. The 14-17 recorded option days for each cell NEVER SAW the tail. The maximum
   underlying excursion inside the live windows over the recorded sample is 39-52 bp;
   over 274-548 comparable days in the long sample the same windows reach 97-215 bp.
   So the observed "worst day" understates the real worst day by 2-4x, and any R:R
   computed from the recorded sample alone is optimistic. (research/118's exact mistake.)

2. Because a combined-% stop CAPS the loss by construction, the reachability of a
   1:2.5 reward:risk is arithmetic - but the cost is expectancy. This script
   translates each stop level into the underlying move that triggers it (fitted on
   the real chain), then reads the TRUE firing frequency off the long sample.

Method for the translation:
  On the recorded chain, for every window-day, the running peak of
     rise% = (combined premium - credit) / credit * 100
  is paired with the running peak underlying excursion in bp. Normalising the move
  by the credit -   xrel = und_exc_bp / (credit_pct_spot * 100)   - makes the map
  comparable across venues and vol regimes (gamma scales inversely with credit).
  A robust (Theil-Sen) line is fitted through those pairs per cell.

Outputs
  results/stop_ladder.csv       per cell x arm: total, mean, median, worst, R:R, fire%
  results/tail_translation.csv  stop% -> triggering move (bp and xrel) -> long-sample
                                exceedance rate, observed-sample exceedance rate
  results/tail_report.txt
"""
import csv, os, math
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

CELL2WIN = {"MON_NIFTY_DTE1": ("MON_1300_1400", 0),
            "WED_SENSEX_DTE1": ("WED_1030_1200", 2),
            "FRI_NIFTY_DTE2": ("FRI_1000_1200", 4)}
# long-sample series used for each cell's excursion distribution.
# NIFTY has NO 1-minute series (research/120), so the NIFTY cells are read off the
# NIFTY50 5-minute clock AND off the SENSEX 1-minute clock as a same-shape control;
# the SENSEX 5-min resample quantifies how much 5-minute data understates extremes.
CELL2SERIES = {"MON_NIFTY_DTE1": ["NIFTY50_5MIN", "SENSEX_1MIN"],
               "WED_SENSEX_DTE1": ["SENSEX_1MIN", "NIFTY50_5MIN"],
               "FRI_NIFTY_DTE2": ["NIFTY50_5MIN", "SENSEX_1MIN"]}

PCT_STOPS = [25, 20, 15, 12, 10, 8, 6]
RS_CAPS = [2500, 2000, 1600, 1400, 1200, 1000, 800]
LOTS = 10


def theilsen(x, y):
    r = stats.theilslopes(y, x, 0.90)
    return r[0], r[1]


def main():
    lines = []

    def P(s=""):
        lines.append(s); print(s, flush=True)

    d = pd.read_csv(os.path.join(RES, "options_sample.csv"))
    pv = pd.read_csv(os.path.join(RES, "prem_vs_move.csv"))
    L = d[d.is_live_dow == 1].copy()

    # ---------- 1. the stop ladder on the recorded sample ----------
    rows = []
    for cell, g in L.groupby("cell"):
        base_med = g["net_SL20"].median()
        for arm in ["NOSTOP"] + ["SL%d" % s for s in PCT_STOPS] + ["RC%d" % r for r in RS_CAPS]:
            v = g["net_" + arm]
            fired = g["fired_" + arm].mean() * 100 if ("fired_" + arm) in g else 0.0
            med = v.median(); worst = v.min()
            rows.append(dict(cell=cell, n=len(g), arm=arm,
                             total_rs_10lot=int(v.sum() * LOTS),
                             mean_rs_10lot=int(v.mean() * LOTS),
                             median_rs_10lot=int(med * LOTS),
                             worst_rs_10lot=int(worst * LOTS),
                             win_pct=round((v > 0).mean() * 100, 1),
                             fire_pct=round(fired, 1),
                             RR_observed=round(abs(worst) / med, 2) if med > 0 else None,
                             pnl_vs_SL20_pct=round((v.sum() / g["net_SL20"].sum() - 1) * 100, 1)
                             if g["net_SL20"].sum() != 0 else None))
    lad = pd.DataFrame(rows)
    lad.to_csv(os.path.join(RES, "stop_ladder.csv"), index=False)
    P("=" * 100)
    P("ROUTE B - STOP LADDER on the recorded chain (live weekday only, Rs at 10 lots, NET of cost)")
    P("=" * 100)
    for cell, g in lad.groupby("cell"):
        P("### %s   n=%d days" % (cell, g.n.iloc[0]))
        P(g[["arm", "total_rs_10lot", "mean_rs_10lot", "median_rs_10lot", "worst_rs_10lot",
             "win_pct", "fire_pct", "RR_observed", "pnl_vs_SL20_pct"]].to_string(index=False))
        P("")

    # ---------- 2. the move -> premium map ----------
    P("=" * 100)
    P("THE MOVE -> PREMIUM MAP (fitted on the real chain; xrel = excursion bp / (credit% x 100))")
    P("=" * 100)
    cred = L.groupby("cell").credit_pct_spot.mean().to_dict()
    maps = {}
    tr_rows = []
    for cell, g in pv.groupby("cell"):
        base = L[L.cell == cell]
        cmap = dict(zip(base.day, base.credit_pct_spot))
        g = g[g.day.isin(cmap)].copy()
        if len(g) < 30:
            g = pv[pv.cell == cell].copy()
            g["cps"] = cred[cell]
        else:
            g["cps"] = g.day.map(cmap)
        g = g[(g.und_exc_bp > 0) & (g.cps > 0)]
        g["xrel"] = g.und_exc_bp / (g.cps * 100.0)
        sl, ic = theilsen(g.xrel.values, g.prem_rise_pct.values)
        r = np.corrcoef(g.xrel, g.prem_rise_pct)[0, 1]
        maps[cell] = (sl, ic, cred[cell])
        P("%-16s n=%5d pairs  rise%% = %.3f * xrel %+0.3f   (Theil-Sen, pearson r=%.3f)"
          % (cell, len(g), sl, ic, r))
    P("")

    # ---------- 3. what move does each stop level need, and how often does it happen? ----------
    P("=" * 100)
    P("STOP LEVEL -> TRIGGERING MOVE -> TRUE FIRING FREQUENCY")
    P("  observed  = the 14-17 recorded option days")
    P("  long      = the same window on hundreds of comparable days (the real clock)")
    P("=" * 100)
    wo = {}
    for tag in ("SENSEX_1MIN", "NIFTY50_5MIN", "SENSEX_5MIN"):
        wo[tag] = pd.read_csv(os.path.join(RES, "window_outcomes_%s.csv" % tag))
    for cell, (sl, ic, cps) in maps.items():
        win, dow = CELL2WIN[cell]
        obs = L[L.cell == cell]
        P("### %s   window %s   mean credit %.3f%% of spot" % (cell, win, cps))
        P("    recorded-sample excursion bp : med %.1f  p90 %.1f  MAX %.1f  (n=%d)"
          % (obs.und_exc_bp.median(), obs.und_exc_bp.quantile(.9), obs.und_exc_bp.max(), len(obs)))
        for tag in CELL2SERIES[cell] + ["SENSEX_5MIN"]:
            s = wo[tag]
            s = s[(s.window == win) & (s.dow == dow)]
            P("    long %-13s excursion bp : med %.1f  p90 %.1f  p99 %.1f  MAX %.1f  (n=%d, %s..%s)"
              % (tag, s.exc_bp.median(), s.exc_bp.quantile(.9), s.exc_bp.quantile(.99),
                 s.exc_bp.max(), len(s), s.day.min(), s.day.max()))
        prim = CELL2SERIES[cell][0]
        sp = wo[prim]; sp = sp[(sp.window == win) & (sp.dow == dow)]
        for st in PCT_STOPS + [4, 3, 2]:
            xrel_need = (st - ic) / sl if sl > 0 else np.nan
            bp_need = xrel_need * cps * 100.0
            f_obs = float((obs.und_exc_bp >= bp_need).mean() * 100)
            f_long = float((sp.exc_bp >= bp_need).mean() * 100)
            tr_rows.append(dict(cell=cell, stop_pct=st, xrel_trigger=round(xrel_need, 3),
                                move_bp_trigger=round(bp_need, 1),
                                fire_pct_observed=round(f_obs, 1),
                                fire_pct_long=round(f_long, 1), long_series=prim, long_n=len(sp)))
        t = pd.DataFrame([r for r in tr_rows if r["cell"] == cell])
        P(t[["stop_pct", "move_bp_trigger", "fire_pct_observed", "fire_pct_long"]].to_string(index=False))
        P("")
    pd.DataFrame(tr_rows).to_csv(os.path.join(RES, "tail_translation.csv"), index=False)

    # ---------- 4. the honest R:R arithmetic ----------
    P("=" * 100)
    P("THE R:R ARITHMETIC - what max loss does each stop actually promise, and what is")
    P("the implied reward:risk against the cell's OBSERVED median profit?")
    P("(loss cap = credit x stop%% x lot + round-trip cost; slippage on the stop leg is")
    P(" NOT modelled and makes the real cap worse.)")
    P("=" * 100)
    P("%-16s %7s %10s %10s %10s %9s %9s %9s" %
      ("cell", "stop%", "medProfit", "lossCap", "R:R", "fire_obs", "fire_long", "P&Lcost%"))
    for cell, (sl, ic, cps) in maps.items():
        g = L[L.cell == cell]
        lot = g.lot.iloc[0]; cost = g.cost.iloc[0]
        credit_rs = (g.credit * lot).mean()
        med = g["net_SL20"].median() * LOTS
        tt = pd.DataFrame(tr_rows); tt = tt[tt.cell == cell]
        for st in PCT_STOPS + [4, 3, 2]:
            cap = (credit_rs * st / 100.0 + cost) * LOTS
            rr = cap / med if med > 0 else float("nan")
            row = tt[tt.stop_pct == st].iloc[0]
            arm = "net_SL%d" % st
            cost_pct = round((g[arm].sum() / g["net_SL20"].sum() - 1) * 100, 1) if arm in g else None
            P("%-16s %7d %10d %10d %10.2f %9.1f %9.1f %9s"
              % (cell, st, int(med), int(cap), rr, row.fire_pct_observed, row.fire_pct_long,
                 ("%+.1f" % cost_pct) if cost_pct is not None else "-"))
        P("")

    with open(os.path.join(RES, "tail_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("DONE")


if __name__ == "__main__":
    main()
