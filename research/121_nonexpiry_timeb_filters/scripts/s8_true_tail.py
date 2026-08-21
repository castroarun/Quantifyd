#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/121 S8 - what the tail of each cell ACTUALLY looks like once it is priced
off hundreds of comparable days instead of the 14-17 recorded ones.

MODELLED, and flagged as such (research/103 precedent). Construction:
  1. credit% of spot is predicted from the day's VIX open by the regression fitted on
     the recorded chain (r = 0.65 - 0.91).
  2. the day's maximum excursion inside the window comes from the long index series.
  3. the peak combined-premium rise is read off the Theil-Sen map fitted on the real
     1-minute chain, in normalised units xrel = excursion_bp / (credit% x 100).
  4. the loss is capped at the deployed stop (20% of credit) and floored at 0.

The map is CONSERVATIVE: on the recorded sample the real premium path fires stops more
often than the map predicts (IV pops and spread widen; the map only knows the underlying).
So every firing frequency below is a LOWER BOUND.

Writes results/true_tail.csv and prints the headline table.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

CELL2WIN = {"MON_NIFTY_DTE1": ("MON_1300_1400", 0, "NIFTY50_5MIN", "SENSEX_1MIN"),
            "WED_SENSEX_DTE1": ("WED_1030_1200", 2, "SENSEX_1MIN", "NIFTY50_5MIN"),
            "FRI_NIFTY_DTE2": ("FRI_1000_1200", 4, "NIFTY50_5MIN", "SENSEX_1MIN")}
LOTS = 10


def main():
    L = pd.read_csv(os.path.join(RES, "options_sample.csv"))
    L = L[L.is_live_dow == 1]
    pv = pd.read_csv(os.path.join(RES, "prem_vs_move.csv"))
    wo = {t: pd.read_csv(os.path.join(RES, "window_outcomes_%s.csv" % t))
          for t in ("SENSEX_1MIN", "NIFTY50_5MIN")}

    out = []
    print("=" * 118)
    print("THE REAL TAIL - modelled loss distribution of each live cell over the LONG sample")
    print("  (Rs at %d lots, net of the round-trip cost, loss capped at the deployed SL20)" % LOTS)
    print("=" * 118)
    print("%-16s %-13s %5s %9s %9s %9s %9s %10s %10s %8s" %
          ("cell", "long series", "n", "medProfit", "loss_p90", "loss_p99", "loss_max",
           "obs_worst", "SL20cap", "R:R_true"))
    for cell, (win, dow, prim, sec) in CELL2WIN.items():
        g = L[L.cell == cell]
        lot = g.lot.iloc[0]; cost = g.cost.iloc[0]
        # credit% ~ VIX
        gg = g.dropna(subset=["vix_open"])
        sl_c, ic_c, r_c, _, _ = stats.linregress(gg.vix_open, gg.credit_pct_spot)
        # move -> premium map
        cmap = dict(zip(g.day, g.credit_pct_spot))
        pp = pv[(pv.cell == cell) & (pv.day.isin(cmap))].copy()
        pp["cps"] = pp.day.map(cmap)
        pp = pp[(pp.und_exc_bp > 0) & (pp.cps > 0)]
        pp["xrel"] = pp.und_exc_bp / (pp.cps * 100.0)
        ts = stats.theilslopes(pp.prem_rise_pct.values, pp.xrel.values, 0.90)
        m_sl, m_ic = ts[0], ts[1]
        med_profit = g.net_SL20.median() * LOTS
        obs_worst = g.net_SL20.min() * LOTS
        for tag in (prim, sec):
            s = wo[tag]
            s = s[(s.window == win) & (s.dow == dow)].copy()
            s["vix_open"] = pd.to_numeric(s.vix_open, errors="coerce")
            s = s.dropna(subset=["vix_open", "exc_bp", "p0"])
            cps = (sl_c * s.vix_open + ic_c).clip(lower=0.15)
            xrel = s.exc_bp / (cps * 100.0)
            rise = (m_sl * xrel + m_ic).clip(lower=0.0, upper=20.0)
            credit_rs = cps / 100.0 * s.p0 * lot
            # for the NIFTY cells read off the SENSEX clock the index level differs, so
            # scale the credit by the recorded cell's own mean credit instead of p0
            credit_rs = credit_rs / credit_rs.mean() * (g.credit * lot).mean()
            loss = (credit_rs * rise / 100.0 + cost) * LOTS
            cap = ((g.credit * lot).mean() * 0.20 + cost) * LOTS
            out.append(dict(cell=cell, series=tag, n=len(s),
                            med_profit=int(med_profit),
                            loss_p90=int(np.percentile(loss, 90)),
                            loss_p99=int(np.percentile(loss, 99)),
                            loss_max=int(loss.max()), obs_worst=int(obs_worst),
                            sl20_cap=int(cap),
                            rr_true=round(float(np.percentile(loss, 99)) / med_profit, 2)
                            if med_profit > 0 else None,
                            fire_sl20_pct=round(float((rise >= 20).mean() * 100), 2)))
            o = out[-1]
            print("%-16s %-13s %5d %9d %9d %9d %9d %10d %10d %8.2f" %
                  (cell, tag, o["n"], o["med_profit"], o["loss_p90"], o["loss_p99"],
                   o["loss_max"], o["obs_worst"], o["sl20_cap"], o["rr_true"]))
        print("")
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(RES, "true_tail.csv"), index=False)
    print("Note: loss_* are the MODELLED bad-day losses; obs_worst is what the 14-17")
    print("recorded days actually delivered. R:R_true uses the 99th-percentile modelled loss.")

    # what does the best single filter cost in P&L on the rupee sample?
    print("")
    print("=" * 118)
    print("COST OF SKIPPING - the leading filter (pre_range_bp, skip the hottest days)")
    print("=" * 118)
    sk = pd.read_csv(os.path.join(RES, "filter_confirm_skip.csv"))
    s = sk[(sk.signal.isin(["pre_range_bp", "cpr_today", "vix_open", "gap_abs", "pdr_rel"]))
           & (sk.side == "skip_high")]
    print(s[["cell", "signal", "skipped", "n", "total_all", "total_kept", "pnl_retained_pct",
             "worst_all", "worst_kept", "rand_total_pctile"]].to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
