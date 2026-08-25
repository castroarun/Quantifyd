#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/124 - Monday atlas + gates G1-G8.

Merges Stage A (options rupee truth, ALL days, Monday = analysis cut) with Stage B
(long-sample excursion clock, era-labelled) through r/122's credit-ladder bridge, then
applies the pre-registered gates:

  G1 med_net@8L>0 AND observed win>=60%
  G2 R:R@p95 < 1:3  (bridged p95 adverse, credit_med rung, SL-capped, over median net)
  G3 modelled P(loss) on the long sample <= 40% (Theil-Sen breakeven + stop-trip, r/122)
  G4 plateau: >=3 window-neighbours (same arm, start/end +-15min) pass G1+G2; adjacent
     stop rungs keep med_net>0
  G5 Westfall-Young sign-flip: cell |t| >= null-95 of family max|t| (2,000 draws)
  G6 label-shuffle null: Monday best cell metric >= 95th pct of best-cell metric over
     2,000 random 17-day draws from ALL recorded days (exact, same grid)
  G7 era consistency: R:R gate uses the WORST of dte-matched / all-era-Monday /
     current-era-Monday p95 excursion
  G8 (reported in RESULTS): vs putting the margin on TUE/FRI cells

Reads results/stage_a_monday.csv + results/stage_b_window_days.csv (+ r/122 atlas for
reconciliation). Writes results/monday_atlas.csv, results/percentiles_long.csv,
results/calmness_clock.csv, results/gates_report.txt.
"""
import csv, os, math, statistics as st
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
R122 = os.path.join(HERE, "..", "..", "122_window_risk_atlas", "results")
LOT = {"NIFTY": 65, "SENSEX": 20}
COST = {"NIFTY": 250.0, "SENSEX": 200.0}   # LEGACY constant, reporting only -- net now carries the MEASURED outcome-aware cost charged in stage A (2026-08-25)
NLOTS = 8
CUR_ERA = {"NIFTY": "era2_expT", "SENSEX": "era3_expT"}
MON_DTE = {"NIFTY": 1, "SENSEX": 3}
RNG = np.random.default_rng(124)
NDRAW = 2000


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def load_a():
    rows = []
    with open(os.path.join(RES, "stage_a_monday.csv")) as f:
        for r in csv.DictReader(f):
            r["dte_trd"] = int(r["dte_trd"])
            for k in ("credit", "net", "gross", "mae_full_rs", "mae_full_pct",
                      "und_exc_bp", "term_move_bp", "spot0", "exit_comb"):
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
    rep = []

    def say(m):
        rep.append(m)
        print(m, flush=True)

    # ---------------- reconciliation vs r/122 ----------------
    say("=== RECONCILIATION vs r/122 (must match to the rupee) ===")
    r122 = {}
    with open(os.path.join(R122, "stage_a_alldays.csv")) as f:
        for r in csv.DictReader(f):
            r122[(r["venue"], r["day"], r["cell"], r["arm"])] = r
    checks = [("NIFTY", "DEP_1300_1400", "SL20", "W_1300_1400", "SLP20"),
              ("NIFTY", "DEP_1000_1200", "SL20", "W_1000_1200", "SLP20"),
              ("SENSEX", "DEP_1030_1200", "SL20", "W_1030_1200", "SLP20")]
    ok_all = True
    for venue, c122, a122, c124, a124 in checks:
        mine = {r["day"]: r for r in A if r["venue"] == venue and r["cell"] == c124
                and r["arm"] == a124}
        n_cmp, n_ok, diffs = 0, 0, []
        for day, r in mine.items():
            k = (venue, day, c122, a122)
            if k not in r122:
                continue
            n_cmp += 1
            d = float(r122[k]["net"]) - r["net"]
            if abs(d) < 0.51:
                n_ok += 1
            else:
                diffs.append((day, d))
        ok = n_cmp > 0 and n_ok == n_cmp
        ok_all = ok_all and ok
        say("%s %s %s: %d/%d days match to the rupee %s %s"
            % (venue, c122, a122, n_ok, n_cmp, "PASS" if ok else "FAIL",
               ("" if ok else str(diffs[:4]))))
    if not ok_all:
        say("RECONCILIATION FAILED - STOP. Atlas not built.")
        with open(os.path.join(RES, "gates_report.txt"), "w") as f:
            f.write("\n".join(rep) + "\n")
        return

    # ---------------- bridge slopes (r/122 method) ----------------
    slope_pool = defaultdict(list)
    slope_cell = defaultdict(list)
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

    # stop overshoot (honesty column): actual loss beyond theoretical cap on SL days
    overshoot = defaultdict(list)   # (venue, arm) -> Rs/lot beyond cap
    for r in A:
        if r["reason"] != "SL":
            continue
        lot = LOT[r["venue"]]
        if r["arm"].startswith("SLP"):
            cap_rs = r["credit"] * (int(r["arm"][3:]) / 100.0) * lot
        elif r["arm"].startswith("R"):
            cap_rs = float(r["arm"][1:])
        else:
            continue
        actual = (r["exit_comb"] - r["credit"]) * lot
        overshoot[(r["venue"], r["arm"])].append(actual - cap_rs)

    # ---------------- stage B scopes ----------------
    b_mon = defaultdict(list)       # (venue,cell) all-era Mondays
    b_mon_cur = defaultdict(list)   # (venue,cell) current-era Mondays
    b_dte = defaultdict(list)       # (venue,cell) dte-matched (any weekday)
    b_wd = defaultdict(list)        # (venue,cell,weekday) for the calmness clock
    for r in B:
        k = (r["venue"], r["cell"])
        b_wd[(r["venue"], r["cell"], r["weekday"])].append(r)
        if r["weekday"] == "Mon":
            b_mon[k].append(r)
            if r["era"] == CUR_ERA[r["venue"]]:
                b_mon_cur[k].append(r)
        if r["dte_trd"] == MON_DTE[r["venue"]]:
            b_dte[k].append(r)

    # ---------------- calmness clock ----------------
    with open(os.path.join(RES, "calmness_clock.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["venue", "cell", "weekday", "n", "exc_p50", "exc_p90", "exc_p95",
                    "exc_max", "mon_cur_era_n", "mon_cur_p50", "mon_cur_p95"])
        for (venue, cell, wd), rows in sorted(b_wd.items()):
            ex = [r["exc_bp"] for r in rows]
            cur = [r["exc_bp"] for r in b_mon_cur.get((venue, cell), [])] if wd == "Mon" else []
            w.writerow([venue, cell, wd, len(ex),
                        round(pct(ex, .5), 1), round(pct(ex, .9), 1),
                        round(pct(ex, .95), 1), round(max(ex), 1)]
                       + ([len(cur), round(pct(cur, .5), 1), round(pct(cur, .95), 1)]
                          if cur else ["", "", ""]))

    # ---------------- percentiles per scope ----------------
    fpc = open(os.path.join(RES, "percentiles_long.csv"), "w", newline="")
    wpc = csv.writer(fpc)
    wpc.writerow(["venue", "cell", "scope", "n", "exc_p50", "exc_p90", "exc_p95",
                  "exc_p99", "exc_max", "term_p50", "term_p95", "term_max"])

    def scope_pct(venue, cell, scope, rows):
        if not rows:
            return None
        ex = [r["exc_bp"] for r in rows]
        tm = [r["term_bp"] for r in rows]
        wpc.writerow([venue, cell, scope, len(ex),
                      round(pct(ex, .5), 1), round(pct(ex, .9), 1), round(pct(ex, .95), 1),
                      round(pct(ex, .99), 1), round(max(ex), 1),
                      round(pct(tm, .5), 1), round(pct(tm, .95), 1), round(max(tm), 1)])
        return ex, tm

    # ---------------- Monday aggregates + gate columns ----------------
    mon_rows = defaultdict(list)
    all_rows = defaultdict(list)
    for r in A:
        k = (r["venue"], r["cell"], r["arm"])
        all_rows[k].append(r)
        if r["weekday"] == "Mon":
            mon_rows[k].append(r)

    atlas = {}
    for (venue, cell, arm), rows in sorted(mon_rows.items()):
        lot, cost = LOT[venue], COST[venue]
        nets8 = [r["net"] * NLOTS for r in rows]
        n = len(nets8)
        if n < 10:
            continue
        credits = [r["credit"] for r in rows]
        cmed = st.median(credits)
        med8 = st.median(nets8)
        mean8 = st.mean(nets8)
        win = 100.0 * sum(1 for x in nets8 if x > 0) / n
        worst8 = min(nets8)
        sd = st.stdev(nets8) if n > 1 else 0.0
        tstat = mean8 / (sd / math.sqrt(n)) if sd > 0 else 0.0
        stop_freq = 100.0 * sum(1 for r in rows if r["reason"] == "SL") / n
        dte = MON_DTE[venue]
        b, bsrc = get_b(venue, dte, cell)

        # stage B percentile scopes -> G7 worst-of
        s_dte = b_dte.get((venue, cell), [])
        s_mon = b_mon.get((venue, cell), [])
        s_cur = b_mon_cur.get((venue, cell), [])
        r_dte = scope_pct(venue, cell, "dte_matched", s_dte)
        r_mon = scope_pct(venue, cell, "monday_all_eras", s_mon)
        r_cur = scope_pct(venue, cell, "monday_cur_era", s_cur)
        if not r_dte or b is None:
            continue
        p95s = {"dte": pct(r_dte[0], .95)}
        if r_mon:
            p95s["mon"] = pct(r_mon[0], .95)
        if r_cur and len(s_cur) >= 20:
            p95s["cur"] = pct(r_cur[0], .95)
        e95_gate = max(p95s.values())
        e95_src = max(p95s, key=p95s.get)

        # SL cap in Rs @8L (+RT cost), with observed overshoot p95 honesty add-on
        if arm.startswith("SLP"):
            sl_cap = cmed * (int(arm[3:]) / 100.0) * lot * NLOTS + cost * NLOTS
        elif arm.startswith("R"):
            sl_cap = float(arm[1:]) * NLOTS + cost * NLOTS
        else:
            sl_cap = None
        ov = overshoot.get((venue, arm), [])
        ov95 = (pct(ov, .95) or 0.0) if ov else 0.0
        sl_cap_ov = sl_cap + max(0.0, ov95) * NLOTS if sl_cap is not None else None

        adv95 = cmed * b * e95_gate * lot * NLOTS + cost * NLOTS
        adv95_cap = min(adv95, sl_cap) if sl_cap else adv95
        adv95_cap_ov = min(adv95, sl_cap_ov) if sl_cap_ov else adv95
        rr = (adv95_cap / med8) if med8 > 0 else None            # 1:rr
        rr_ov = (adv95_cap_ov / med8) if med8 > 0 else None

        # modelled P(loss) (r/122 Theil-Sen on the cell's own Monday rows)
        xs = [abs(r["term_move_bp"]) for r in rows]
        ys = [r["net"] / (r["credit"] * lot) for r in rows]
        slopes = []
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                if abs(xs[i] - xs[j]) > 5.0:
                    slopes.append((ys[i] - ys[j]) / (xs[i] - xs[j]))
        be = None
        if slopes:
            cts = st.median(slopes)
            ats = st.median([ys[i] - cts * xs[i] for i in range(len(xs))])
            if cts < 0 and ats > 0:
                be = ats / (-cts)
        if be is None:
            mg = st.median([r["gross"] for r in rows]) / lot
            be = mg / (b * cmed) if b * cmed > 0 else 1e9
        if arm.startswith("SLP"):
            sl_bp = (int(arm[3:]) / 100.0) / b
        elif arm.startswith("R"):
            sl_bp = (float(arm[1:]) / lot / cmed) / b
        else:
            sl_bp = None
        n_loss = 0
        for r2 in s_dte:
            hit = r2["term_bp"] > be
            if sl_bp is not None:
                hit = hit or (r2["exc_bp"] > sl_bp)
            if hit:
                n_loss += 1
        plm = 100.0 * n_loss / len(s_dte)
        p_stop_long = (100.0 * sum(1 for r2 in s_dte if r2["exc_bp"] > sl_bp)
                       / len(s_dte)) if sl_bp else 0.0

        atlas[(venue, cell, arm)] = dict(
            venue=venue, cell=cell, arm=arm, n=n, credit_med=round(cmed, 1),
            med_net_8L=round(med8), mean_net_8L=round(mean8), win_pct=round(win, 1),
            worst_8L=round(worst8), tstat=round(tstat, 2), stop_freq_pct=round(stop_freq, 1),
            b_slope=round(b, 6), b_src=bsrc, e95_gate_bp=round(e95_gate, 1), e95_src=e95_src,
            n_dte=len(s_dte), n_mon=len(s_mon), n_cur=len(s_cur),
            sl_cap_8L=round(sl_cap) if sl_cap else "",
            ov95_perlot=round(ov95) if ov else "",
            adv95_cap_8L=round(adv95_cap), adv95_cap_ov_8L=round(adv95_cap_ov),
            rr_p95=round(rr, 2) if rr else "", rr_p95_ov=round(rr_ov, 2) if rr_ov else "",
            p_loss_model=round(plm, 1), p_stop_long=round(p_stop_long, 1),
            be_term_bp=round(be, 1))
    fpc.close()

    # ---------------- G1-G3 flags ----------------
    for k, a in atlas.items():
        a["G1"] = int(a["med_net_8L"] > 0 and a["win_pct"] >= 60.0)
        a["G2"] = int(a["rr_p95"] != "" and float(a["rr_p95"]) < 3.0)
        a["G3"] = int(a["p_loss_model"] <= 40.0)

    # ---------------- G4 plateau ----------------
    def wparse(cell):
        _, s, e = cell.split("_")
        return int(s[:2]) * 60 + int(s[2:]), int(e[:2]) * 60 + int(e[2:])

    ARMLADDER = ["SLP10", "SLP15", "SLP20", "SLP25", "SLP30", "SLP40", "NOSTOP"]
    RLADDER = ["R500", "R1000", "R1500", "R2500"]
    for (venue, cell, arm), a in atlas.items():
        s0, e0 = wparse(cell)
        nb_pass = nb_tot = 0
        for (v2, c2, a2), b2 in atlas.items():
            if v2 != venue or a2 != arm or c2 == cell:
                continue
            s1, e1 = wparse(c2)
            if abs(s1 - s0) <= 15 and abs(e1 - e0) <= 15:
                nb_tot += 1
                if b2["G1"] and b2["G2"]:
                    nb_pass += 1
        lad = ARMLADDER if arm in ARMLADDER else RLADDER
        i = lad.index(arm)
        adj_ok = True
        for j in (i - 1, i + 1):
            if 0 <= j < len(lad):
                nbb = atlas.get((venue, cell, lad[j]))
                if nbb and nbb["med_net_8L"] <= 0:
                    adj_ok = False
        a["nb_pass"], a["nb_tot"] = nb_pass, nb_tot
        a["G4"] = int(nb_pass >= 3 and adj_ok)

    # ---------------- G5 Westfall-Young sign-flip (per venue) ----------------
    say("\n=== G5 sign-flip family-wise null (%d draws) ===" % NDRAW)
    for venue in ("NIFTY", "SENSEX"):
        keys = [k for k in sorted(atlas) if k[0] == venue]
        daysets = [sorted({r["day"] for r in mon_rows[k]}) for k in keys]
        days = sorted(set().union(*[set(d) for d in daysets]))
        di = {d: i for i, d in enumerate(days)}
        M = np.full((len(keys), len(days)), np.nan)
        for i, k in enumerate(keys):
            for r in mon_rows[k]:
                M[i, di[r["day"]]] = r["net"] * NLOTS
        mask = ~np.isnan(M)
        M0 = np.where(mask, M, 0.0)
        nn = mask.sum(1)

        def tvec(sign):
            X = M0 * sign
            mu = X.sum(1) / nn
            var = (np.where(mask, (X - mu[:, None]) ** 2, 0.0)).sum(1) / (nn - 1)
            sd = np.sqrt(var)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(sd > 0, mu / (sd / np.sqrt(nn)), 0.0)

        t_obs = tvec(np.ones(len(days)))
        maxt = np.empty(NDRAW)
        for d in range(NDRAW):
            maxt[d] = np.abs(tvec(RNG.choice([-1.0, 1.0], size=len(days)))).max()
        thr = np.quantile(maxt, 0.95)
        for i, k in enumerate(keys):
            atlas[k]["t_obs"] = round(float(t_obs[i]), 2)
            atlas[k]["G5"] = int(abs(t_obs[i]) >= thr)
        say("%s: family=%d cells, max|t| observed=%.2f, null-95 of max|t|=%.2f, "
            "cells clearing=%d" % (venue, len(keys), np.abs(t_obs).max(), thr,
                                   int((np.abs(t_obs) >= thr).sum())))

    # ---------------- G6 label-shuffle best-cell null ----------------
    say("\n=== G6 label-shuffle null: best Monday cell vs best cell of random 17-day draws ===")
    for venue in ("NIFTY", "SENSEX"):
        keys = [k for k in sorted(all_rows) if k[0] == venue]
        days = sorted({r["day"] for k in keys for r in all_rows[k]})
        di = {d: i for i, d in enumerate(days)}
        M = np.full((len(keys), len(days)), np.nan)
        for i, k in enumerate(keys):
            for r in all_rows[k]:
                M[i, di[r["day"]]] = r["net"] * NLOTS
        wdmap = {r["day"]: r["weekday"] for k in keys for r in all_rows[k]}
        mon_idx = np.array([i for i, d in enumerate(days) if wdmap.get(d) == "Mon"])
        nmon = len(mon_idx)

        def best_metric(cols):
            X = M[:, cols]
            med = np.nanmedian(X, axis=1)
            win = np.nanmean(X > 0, axis=1) * 100.0
            ok = win >= 60.0
            return np.nanmax(np.where(ok, med, -1e18))

        obs = best_metric(mon_idx)
        null = np.empty(NDRAW)
        allidx = np.arange(len(days))
        for d in range(NDRAW):
            null[d] = best_metric(RNG.choice(allidx, size=nmon, replace=False))
        q95 = np.quantile(null, 0.95)
        pval = float(np.mean(null >= obs))
        say("%s: Monday best med@8L (win>=60%%) = %+.0f | shuffled-best null: med %+.0f, "
            "q95 %+.0f -> empirical p=%.3f %s"
            % (venue, obs, np.median(null), q95, pval,
               "(Monday best BEATS null-95)" if obs >= q95 else "(does NOT beat null-95)"))
        for k in [k for k in sorted(atlas) if k[0] == venue]:
            atlas[k]["G6_venue"] = int(obs >= q95)

    # ---------------- write atlas ----------------
    cols = ["venue", "cell", "arm", "n", "credit_med", "med_net_8L", "mean_net_8L",
            "win_pct", "worst_8L", "tstat", "t_obs", "stop_freq_pct", "b_slope", "b_src",
            "e95_gate_bp", "e95_src", "n_dte", "n_mon", "n_cur", "sl_cap_8L", "ov95_perlot",
            "adv95_cap_8L", "adv95_cap_ov_8L", "rr_p95", "rr_p95_ov", "p_loss_model",
            "p_stop_long", "be_term_bp", "nb_pass", "nb_tot",
            "G1", "G2", "G3", "G4", "G5", "G6_venue"]
    with open(os.path.join(RES, "monday_atlas.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in sorted(atlas):
            w.writerow({c: atlas[k].get(c, "") for c in cols})

    # ---------------- summary ----------------
    say("\n=== GATE SUMMARY ===")
    for venue in ("NIFTY", "SENSEX"):
        sub = [a for a in atlas.values() if a["venue"] == venue]
        g12 = [a for a in sub if a["G1"] and a["G2"]]
        g123 = [a for a in g12 if a["G3"]]
        g1234 = [a for a in g123 if a["G4"]]
        full = [a for a in g1234 if a["G5"] and a["G6_venue"]]
        say("%s: cells=%d  G1&G2=%d  +G3=%d  +G4=%d  +G5&G6=%d"
            % (venue, len(sub), len(g12), len(g123), len(g1234), len(full)))
        ranked = sorted(g12, key=lambda a: float(a["rr_p95"]))[:15]
        say("  top G1&G2 cells by rr_p95 (1:X):")
        for a in ranked:
            say("   %s %-6s n=%d med8=%+6d win=%4.1f rr=1:%s rr_ov=1:%s plm=%4.1f "
                "t=%.2f nb=%d/%d G3%d G4%d G5%d G6%d worst=%+d stop%%=%s"
                % (a["cell"], a["arm"], a["n"], a["med_net_8L"], a["win_pct"],
                   a["rr_p95"], a["rr_p95_ov"], a["p_loss_model"], a["t_obs"],
                   a["nb_pass"], a["nb_tot"], a["G3"], a["G4"], a["G5"], a["G6_venue"],
                   a["worst_8L"], a["stop_freq_pct"]))
        # the dropped cell for reference
        for arm in ("SLP20",):
            a = atlas.get((venue, "W_1300_1400", arm))
            if a:
                say("  [dropped-cell ref] %s W_1300_1400 %s: med8=%+d win=%.1f rr=1:%s "
                    "plm=%.1f" % (venue, arm, a["med_net_8L"], a["win_pct"], a["rr_p95"],
                                  a["p_loss_model"]))
    with open(os.path.join(RES, "gates_report.txt"), "w") as f:
        f.write("\n".join(rep) + "\n")
    print("atlas written: %d rows" % len(atlas))


if __name__ == "__main__":
    main()
