#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/122 - assemble the Window Risk Atlas from Stage A (options rupee truth)
and Stage B (long-sample violent-move clock), through the credit-ladder bridge.

THE BRIDGE (the weakest link, stated explicitly):
  Stage B gives move percentiles in bp of spot. They are converted to rupee outcomes via
    adverse_Rs(p) = credit_pts x b x exc_bp(p) x lot x 10lots
  where b = the observed premium-rise slope: median over 2026 recorded days of
    (max combined-premium rise as fraction of credit) / (max underlying excursion in bp)
  fitted per (venue, DTE) pooled across grid cells, refined per cell where n>=8 and the
  cell saw real moves. Credit is a LADDER: the p25/median/p75 of credits observed in 2026
  for that exact cell. Assumptions this bakes in:
    * the premium's adverse extreme is proportional to the underlying's adverse extreme
      with the 2026-observed gamma/vega mix (IV pops beyond 2026's are NOT modelled ->
      conservative-side understatement on true tails);
    * a straddle's loss at the adverse extreme ~ premium rise, i.e. remaining premium is
      carried at entry value (decay earned before the spike offsets, spike-driven IV adds;
      both ignored);
    * 2026 credit levels (in pts) scale to other regimes only through the ladder - a
      high-VIX regime would sit above the p75 rung.
READ-ONLY inputs; writes results/atlas.csv, results/percentiles_long.csv,
results/bridge_map.csv, results/reconciliation.txt.
"""
import csv, os, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOT = {"NIFTY": 65, "SENSEX": 20}
COST = {"NIFTY": 250.0, "SENSEX": 200.0}   # LEGACY constant, reporting only -- net now carries the MEASURED outcome-aware cost charged in stage A (2026-08-25)

DEPLOYED = {  # name -> (venue, dte, cell, arm, sl_frac)
    "MON_NIFTY_DTE1":  ("NIFTY", 1, "DEP_1300_1400", "SL20", 0.20),
    "TUE_NIFTY_DTE0":  ("NIFTY", 0, "DEP_0930_1100", "SL25", 0.25),
    "WED_SENSEX_DTE1": ("SENSEX", 1, "DEP_1030_1200", "SL20", 0.20),
    "THU_SENSEX_DTE0": ("SENSEX", 0, "DEP_1300_1520", "NOSTOP", None),
    "FRI_NIFTY_DTE2":  ("NIFTY", 2, "DEP_1000_1200", "SL20", 0.20),
}


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


def load_a():
    rows = []
    with open(os.path.join(RES, "stage_a_alldays.csv")) as f:
        for r in csv.DictReader(f):
            r["dte_trd"] = int(r["dte_trd"])
            for k in ("credit", "net", "gross", "mae_full_rs", "mae_full_pct",
                      "und_exc_bp", "term_move_bp", "spot0"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def load_b():
    rows = []
    with open(os.path.join(RES, "stage_b_window_days.csv")) as f:
        for r in csv.DictReader(f):
            r["exc_bp"] = float(r["exc_bp"])
            r["term_bp"] = float(r["term_bp"])
            r["dte_trd"] = None if r["dte_trd"] == "" else int(r["dte_trd"])
            rows.append(r)
    return rows


def main():
    A = load_a()
    B = load_b()
    out = []

    # ---------- reconciliation vs r/120 and r/121 ----------
    rec = []
    def sel(venue, dte, cell, arm, upto=None):
        return [r for r in A if r["venue"] == venue and r["dte_trd"] == dte
                and r["cell"] == cell and r["arm"] == arm
                and (upto is None or r["day"] <= upto)]
    fri = sel("NIFTY", 2, "DEP_1000_1200", "SL20", upto="2026-08-14")
    if fri:
        nets = [r["net"] for r in fri]
        rec.append("FRI NIFTY DTE2 10:00-12:00 SL20 (days<=08-14): n=%d mean=%+.0f/lot "
                   "median=%+.0f worst=%+.0f wins=%d/%d   [r/120: n=14 mean +400 worst -344 13/14]"
                   % (len(nets), st.mean(nets), st.median(nets), min(nets),
                      sum(1 for x in nets if x > 0), len(nets)))
    mon = sel("NIFTY", 1, "DEP_1300_1400", "SL20", upto="2026-08-17")
    if mon:
        nets = [r["net"] * 10 for r in mon]
        rec.append("MON NIFTY DTE1 13:00-14:00 SL20 (days<=08-17): n=%d median@10=%+.0f "
                   "mean@10=%+.0f worst@10=%+.0f win%%=%.0f   [r/121: n=17 median +1240 mean +998 worst -4840 71%%]"
                   % (len(nets), st.median(nets), st.mean(nets), min(nets),
                      100.0 * sum(1 for x in nets if x > 0) / len(nets)))
    wed = sel("SENSEX", 1, "DEP_1030_1200", "SL20", upto="2026-08-19")
    if wed:
        nets = [r["net"] * 10 for r in wed]
        rec.append("WED SENSEX DTE1 10:30-12:00 SL20 (days<=08-19): n=%d median@10=%+.0f "
                   "mean@10=%+.0f worst@10=%+.0f win%%=%.0f   [r/121: n=17 median +3830 mean +1922 worst -11440 71%%]"
                   % (len(nets), st.median(nets), st.mean(nets), min(nets),
                      100.0 * sum(1 for x in nets if x > 0) / len(nets)))
    with open(os.path.join(RES, "reconciliation.txt"), "w") as f:
        f.write("\n".join(rec) + "\n")
    print("\n".join(rec))

    # ---------- bridge slopes ----------
    # b = median over NOSTOP grid rows with und_exc_bp>=20 of (mae_frac / exc_bp)
    slope_pool = defaultdict(list)          # (venue,dte) -> ratios
    slope_cell = defaultdict(list)          # (venue,dte,cell) -> ratios
    for r in A:
        if r["arm"] != "NOSTOP" or r["und_exc_bp"] < 20:
            continue
        ratio = (r["mae_full_pct"] / 100.0) / r["und_exc_bp"]
        slope_pool[(r["venue"], r["dte_trd"])].append(ratio)
        slope_cell[(r["venue"], r["dte_trd"], r["cell"])].append(ratio)

    def get_b(venue, dte, cell):
        v = slope_cell.get((venue, dte, cell), [])
        if len(v) >= 8:
            return st.median(v), "cell(n=%d)" % len(v)
        v = slope_pool.get((venue, dte), [])
        if v:
            return st.median(v), "pooled(n=%d)" % len(v)
        return None, "none"

    with open(os.path.join(RES, "bridge_map.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["venue", "dte", "cell", "b_frac_per_bp", "source",
                    "sl20_fires_at_bp", "sl25_fires_at_bp", "sl50_fires_at_bp"])
        seen = set()
        for r in A:
            key = (r["venue"], r["dte_trd"], r["cell"])
            if key in seen:
                continue
            seen.add(key)
            b, src = get_b(*key)
            if b:
                w.writerow([key[0], key[1], key[2], round(b, 6), src,
                            round(0.20 / b, 1), round(0.25 / b, 1), round(0.50 / b, 1)])

    # ---------- stage A aggregates per (venue,dte,cell,arm) ----------
    agg = defaultdict(list)
    for r in A:
        agg[(r["venue"], r["dte_trd"], r["cell"], r["arm"])].append(r)

    # ---------- stage B samples per (venue,dte,cell) ----------
    bs = defaultdict(list)       # DTE-matched
    bs_all = defaultdict(list)   # all days any label (deep-tail reference)
    for r in B:
        k_all = (r["venue"], r["cell"])
        bs_all[k_all].append(r)
        if r["dte_trd"] is not None:
            bs[(r["venue"], r["dte_trd"], r["cell"])].append(r)

    # ---------- long percentile table ----------
    fpc = open(os.path.join(RES, "percentiles_long.csv"), "w", newline="")
    wpc = csv.writer(fpc)
    wpc.writerow(["venue", "dte", "cell", "scope", "n_days",
                  "exc_p50", "exc_p90", "exc_p95", "exc_p99", "exc_max",
                  "term_p50", "term_p90", "term_p95", "term_p99", "term_max"])

    def write_pct(venue, dte, cell, scope, rows):
        if not rows:
            return None
        ex = [r["exc_bp"] for r in rows]
        tm = [r["term_bp"] for r in rows]
        wpc.writerow([venue, dte, cell, scope, len(ex)] +
                     [round(pct(ex, p), 1) for p in (0.5, 0.9, 0.95, 0.99)] + [round(max(ex), 1)] +
                     [round(pct(tm, p), 1) for p in (0.5, 0.9, 0.95, 0.99)] + [round(max(tm), 1)])
        return ex, tm

    # ---------- atlas ----------
    fat = open(os.path.join(RES, "atlas.csv"), "w", newline="")
    wat = csv.writer(fat)
    wat.writerow(["deployed_name", "venue", "dte", "cell", "start", "end", "arm",
                  "n_opt", "n_px", "n_px_ex2122",
                  "credit_p25", "credit_med", "credit_p75",
                  "med_net_10L", "mean_net_10L", "win_pct", "worst_opt_10L",
                  "b_slope", "b_src", "sl_cap_rs_10L",
                  "exc_p90_bp", "exc_p95_bp", "exc_p99_bp", "exc_max_bp",
                  "adv_p90_10L_c25", "adv_p90_10L_cmed", "adv_p90_10L_c75",
                  "adv_p95_10L_cmed", "adv_p99_10L_cmed", "adv_max_10L_cmed",
                  "adv_p90_capped", "adv_p95_capped",
                  "p_loss_opt", "p_loss_model", "p_sl20_hit", "p_sl_deployed_hit",
                  "rr_p90", "rr_p95",
                  "exc_p95_ex2122", "exc_p99_ex2122", "be_term_bp"])

    for (venue, dte, cell, arm), rows in sorted(agg.items()):
        if arm not in ("SL20", "SL25", "NOSTOP"):
            continue
        lot = LOT[venue]
        cost = COST[venue]
        nets10 = [r["net"] * 10 for r in rows]
        credits = [r["credit"] for r in rows]
        n_opt = len(rows)
        med_net = st.median(nets10)
        win = 100.0 * sum(1 for x in nets10 if x > 0) / n_opt
        worst = min(nets10)
        c25, cmed, c75 = pct(credits, 0.25), st.median(credits), pct(credits, 0.75)
        b, bsrc = get_b(venue, dte, cell)
        start, end = rows[0]["start"], rows[0]["end"]

        long_rows = bs.get((venue, dte, cell), [])
        r_pct = write_pct(venue, dte, cell, "dte_matched", long_rows)
        ex2 = [r for r in long_rows if r["day"] >= "2023-01-01"]
        r_pct2 = write_pct(venue, dte, cell, "dte_matched_ex2122", ex2)
        write_pct(venue, dte, cell, "all_days", bs_all.get((venue, cell), []))

        dep_name = ""
        sl_frac = {"SL20": 0.20, "SL25": 0.25, "NOSTOP": None}[arm]
        for nm, (v2, d2, c2, a2, s2) in DEPLOYED.items():
            if (v2, d2, c2, a2) == (venue, dte, cell, arm):
                dep_name = nm

        if not r_pct or b is None:
            wat.writerow([dep_name, venue, dte, cell, start, end, arm, n_opt, 0, ""] +
                         [round(x, 1) for x in (c25, cmed, c75)] +
                         [round(med_net), round(st.mean(nets10)), round(win, 1), round(worst)] +
                         [""] * 20)
            continue
        ex, tm = r_pct
        n_px = len(ex)
        e90, e95, e99, emax = pct(ex, 0.9), pct(ex, 0.95), pct(ex, 0.99), max(ex)

        def adv(exc_bp, credit_pts):
            return credit_pts * b * exc_bp * lot * 10 + cost * 10

        adv90 = {c: adv(e90, cv) for c, cv in (("c25", c25), ("cmed", cmed), ("c75", c75))}
        adv95 = adv(e95, cmed)
        adv99 = adv(e99, cmed)
        advmax = adv(emax, cmed)
        sl_cap = (cmed * sl_frac * lot * 10 + cost * 10) if sl_frac else None
        adv90_cap = min(adv90["cmed"], sl_cap) if sl_cap else adv90["cmed"]
        adv95_cap = min(adv95, sl_cap) if sl_cap else adv95

        # P(loss day), modelled on the long sample: a Theil-Sen line net/credit vs
        # |terminal move| fitted on the cell's own 2026 booked outcomes gives the
        # breakeven move; a long-sample day is a modelled loss when its terminal move
        # exceeds that breakeven, or (if a stop is deployed) its excursion trips the stop.
        pairs = [(r["term_move_bp"] if "term_move_bp" in r else 0.0,
                  r["net"] / (r["credit"] * lot)) for r in rows]
        xs = [abs(x) for x, _ in pairs]
        ys = [y for _, y in pairs]
        slopes = []
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                if abs(xs[i] - xs[j]) > 5.0:
                    slopes.append((ys[i] - ys[j]) / (xs[i] - xs[j]))
        be_term_bp = None
        if slopes:
            cts = st.median(slopes)
            ats = st.median([ys[i] - cts * xs[i] for i in range(len(xs))])
            if cts < 0 and ats > 0:
                be_term_bp = ats / (-cts)
        if be_term_bp is None:  # fallback: crude decay-vs-burn arithmetic
            med_gross_pts = st.median([r["gross"] for r in rows]) / lot
            be_term_bp = med_gross_pts / (b * cmed) if b * cmed > 0 else 1e9
        sl_bp = (sl_frac / b) if sl_frac else None
        n_loss = 0
        for r2 in long_rows:
            hit = r2["term_bp"] > be_term_bp
            if sl_bp is not None:
                hit = hit or (r2["exc_bp"] > sl_bp)
            if hit:
                n_loss += 1
        p_loss_model = 100.0 * n_loss / n_px
        p_loss_opt = 100.0 - win
        p_sl20 = 100.0 * sum(1 for r2 in long_rows if r2["exc_bp"] > 0.20 / b) / n_px
        p_sl_dep = (100.0 * sum(1 for r2 in long_rows if r2["exc_bp"] > sl_bp) / n_px) if sl_bp else ""

        rr90 = med_net / adv90_cap if adv90_cap > 0 else ""
        rr95 = med_net / adv95_cap if adv95_cap > 0 else ""
        e95x = pct([r["exc_bp"] for r in ex2], 0.95) if ex2 else ""
        e99x = pct([r["exc_bp"] for r in ex2], 0.99) if ex2 else ""

        wat.writerow([dep_name, venue, dte, cell, start, end, arm, n_opt, n_px, len(ex2),
                      round(c25, 1), round(cmed, 1), round(c75, 1),
                      round(med_net), round(st.mean(nets10)), round(win, 1), round(worst),
                      round(b, 6), bsrc, round(sl_cap) if sl_cap else "",
                      round(e90, 1), round(e95, 1), round(e99, 1), round(emax, 1),
                      round(adv90["c25"]), round(adv90["cmed"]), round(adv90["c75"]),
                      round(adv95), round(adv99), round(advmax),
                      round(adv90_cap), round(adv95_cap),
                      round(p_loss_opt, 1), round(p_loss_model, 1),
                      round(p_sl20, 1), round(p_sl_dep, 1) if p_sl_dep != "" else "",
                      round(rr90, 3) if rr90 != "" else "", round(rr95, 3) if rr95 != "" else "",
                      e95x if e95x == "" else round(e95x, 1),
                      e99x if e99x == "" else round(e99x, 1),
                      round(be_term_bp, 1)])
    fat.close()
    fpc.close()
    print("atlas written")


if __name__ == "__main__":
    main()
