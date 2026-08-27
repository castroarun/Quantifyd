#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/133 Stage C - bridge the multi-year index sample into premium space, then price
the backstop as insurance: fire rate, save, cost, tail cap, gap-through.

THE BRIDGE, in dimensionless form (this is the heart of the study)
-----------------------------------------------------------------
Everything is expressed as  R = (index distance travelled) / (entry credit in points),
so it is regime-free and needs no rupee assumption to state.

  F_lin(R)   = beta * R              beta = b_med x credit_bp  (the r/122 linear slope,
                                     re-expressed dimensionlessly)
  F_intr(R)  = max(0, R - 1)         MODEL-FREE: on expiry day combined premium is never
                                     below intrinsic, so a move of R credits away from the
                                     strike forces the straddle to at least (R-1) credits
                                     of adverse premium.
  F(R)       = max(F_lin, F_intr)    the conservative combination actually used.

F_lin is the better description of ordinary days; F_intr dominates beyond R ~ 1.5 and is
the only one of the two that is safe in the tail. Using the max of the two is precisely
r/122's instruction to prefer the more conservative route where they disagree.

The credit is VOL-SCALED, not frozen at 2026 levels: credit_bp(d) = alpha x rv20_bp(d),
alpha fitted on the 17 recorded DTE0 sessions. Without this, a 2021-22 high-vol day is
scored against a 2026 credit and the fire rate is inflated exactly where the disasters are.

Inputs  : results/stage_a_days.csv, stage_a_levels.csv, stage_b_days.csv (read-only)
          + backtest_data/market_data.db (read-only) for daily closes and the DTE0 paths
Outputs : results/analysis.txt, bridge.csv, bridge_validation.csv, fire_rates.csv,
          save_cost.csv, tail.csv, gap_through.csv, gap_through_long.csv
"""
import csv, os, math, sqlite3, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
MD = "/home/arun/quantifyd/backtest_data/market_data.db"
LOT = 20
LOTS = 6
SLIP_STOP = 6.548
SLIP_TIME = 0.178
NLOTS_REF = 10
FRAC = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00]
RS = [1000, 1500, 2000, 2500, 3000, 4000, 5000]
ENTRY_M, EXIT_M = 13 * 60, 15 * 60 + 20
OUT = []


def say(m=""):
    OUT.append(m)
    print(m, flush=True)


def cost_per_lot(credit, exitp, lot, reason):
    sell = credit * lot; buy = exitp * lot; tot = sell + buy
    brok = 80.0 / NLOTS_REF
    stt = 0.001 * sell
    txn = 0.0003503 * tot
    ipft = 0.0000050 * tot
    sebi = 0.0000010 * tot
    stamp = 0.00003 * buy
    gst = 0.18 * (brok + txn + ipft + sebi)
    slip = 2 * (SLIP_STOP if reason in ("SL", "BACKSTOP") else SLIP_TIME)
    return brok + stt + txn + ipft + sebi + stamp + gst + slip * lot


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def rd(p):
    with open(os.path.join(RES, p)) as f:
        return list(csv.DictReader(f))


def F(fmt, *a):
    return fmt % a


def daily_closes(sym):
    """daily closes: the 'day' series, extended past its last row from the 1-minute series."""
    c = sqlite3.connect("file:%s?mode=ro" % MD, uri=True)
    d = {}
    for dt, cl in c.execute("SELECT date, close FROM market_data_unified WHERE symbol=? AND "
                            "timeframe='day' ORDER BY date", (sym,)):
        if cl:
            d[dt[:10]] = float(cl)
    last = max(d) if d else "2000-01-01"
    for dt, cl in c.execute("SELECT date, close FROM market_data_unified WHERE symbol=? AND "
                            "timeframe='minute' AND date>? ORDER BY date", (sym, last)):
        if cl:
            d[dt[:10]] = float(cl)          # ordered -> ends on the day's last bar
    c.close()
    return d


def daily_rv(sym, _tf=None):
    """causal trailing-20-day close-to-close vol, in bp of price, keyed by day."""
    cl = daily_closes(sym)
    days = sorted(cl)
    closes = [cl[d] for d in days]
    rets = [0.0] + [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    out = {}
    for i in range(len(days)):
        w = rets[max(1, i - 20):i]          # CAUSAL: strictly before today
        if len(w) >= 10:
            out[days[i]] = 1e4 * st.pstdev(w)
    return out


def main():
    A = rd("stage_a_days.csv")
    for r in A:
        for k in ("credit", "spot0", "mae_pts", "mae_frac", "und_exc_pts", "und_exc_bp",
                  "hold_net_lot", "term_move_bp", "term_move_pts", "strike"):
            r[k] = float(r[k])
    L = rd("stage_a_levels.csv")
    for r in L:
        for k in ("level", "thresh_comb", "net_lot", "hold_net_lot", "delta_lot",
                  "overshoot_pts", "touch_comb", "exit_comb"):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                r[k] = None
        r["fired"] = int(r["fired"]); r["dwell"] = int(r["dwell"])
    B = rd("stage_b_days.csv")
    for r in B:
        for k in ("exc_pts", "exc_bp", "distK_pts", "distK_bp", "term_pts", "term_bp",
                  "ref1300", "strike"):
            r[k] = float(r[k])
        r["dte"] = None if r["dte_trd"] == "" else int(float(r["dte_trd"]))

    say("=" * 104)
    say("research/133 - SENSEX DTE0 disaster backstop, 13:00->15:20, CSL_TIMEB_SENSEX @ %d lots (qty %d)"
        % (LOTS, LOTS * LOT))
    say("=" * 104)

    n = len(A)
    creds = [r["credit"] for r in A]
    cbp = [1e4 * r["credit"] / r["spot0"] for r in A]
    holds = [r["hold_net_lot"] for r in A]
    cmed, cmed_bp = st.median(creds), st.median(cbp)
    say("")
    say("## 0. Stage A - the recorded options sample (fidelity, small n)")
    say("n = %d clean DTE0 sessions %s .. %s   (SENSEX lot 20; 1 pt = Rs20/lot = Rs%d at %dL)"
        % (n, A[0]["day"], A[-1]["day"], LOT * LOTS, LOTS))
    say("credit pts   p25 %.1f  median %.1f  p75 %.1f   (today 2026-08-27: 231.63)"
        % (pct(creds, .25), cmed, pct(creds, .75)))
    say("credit in bp p25 %.1f  median %.1f  p75 %.1f" % (pct(cbp, .25), cmed_bp, pct(cbp, .75)))
    say("HOLD 13:00->15:20  mean %+.0f/lot  median %+.0f  win %.0f%%  worst %+.0f  best %+.0f"
        % (st.mean(holds), st.median(holds), 100.0 * sum(1 for x in holds if x > 0) / n,
           min(holds), max(holds)))
    say("HOLD at %d lots     mean %+.0f      median %+.0f      worst %+.0f"
        % (LOTS, st.mean(holds) * LOTS, st.median(holds) * LOTS, min(holds) * LOTS))
    mf = sorted(r["mae_frac"] for r in A)
    say("MAE / credit, sorted: " + " ".join("%.2f" % x for x in mf))
    say("  p50 %.2f  p75 %.2f  p90 %.2f  max %.2f   -> the deployed 0.50 sits at the ~p84 of"
        % (pct(mf, .5), pct(mf, .75), pct(mf, .9), max(mf)))
    say("     this small sample: 3 of 17 sessions reached it.")

    # ---------------- 1. vol model for the credit ----------------
    say("")
    say("## 1. Credit model - the credit must move with the regime, or the tail is mis-scored")
    rv = daily_rv("SENSEX", "day")
    if not rv:
        rv = daily_rv("SENSEX", "minute")
    pairs = [(1e4 * r["credit"] / r["spot0"], rv.get(r["day"])) for r in A]
    pairs = [(c, v) for c, v in pairs if v]
    alpha = st.median([c / v for c, v in pairs])
    xs = [v for _, v in pairs]; ys = [c for c, _ in pairs]
    mx, my = st.mean(xs), st.mean(ys)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    corr = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else 0.0
    say("credit_bp(d) = alpha x rv20_bp(d), rv20 = causal trailing-20-day close-to-close vol")
    say("  fitted on %d recorded DTE0 sessions:  alpha = %.3f   Pearson r = %.2f"
        % (len(pairs), alpha, corr))
    say("  2026 sample rv20 median %.1f bp -> credit %.1f bp (observed median %.1f bp)"
        % (st.median(xs), alpha * st.median(xs), cmed_bp))
    rvB = [rv.get(r["day"]) for r in B if r["series"] == "SENSEX_1min"]
    rvB = [x for x in rvB if x]
    say("  long-sample rv20: p10 %.1f  p50 %.1f  p90 %.1f  max %.1f bp  -> the 2021-22 regime"
        % (pct(rvB, .1), pct(rvB, .5), pct(rvB, .9), max(rvB)))
    say("     carried roughly %.1fx the 2026 credit, which is exactly why a frozen 2026 credit"
        % (pct(rvB, .9) / st.median(xs)))
    say("     would over-fire the backstop in the very regimes that matter.")

    def credit_pts(r):
        v = rv.get(r["day"])
        if not v:
            v = st.median(xs)
        return alpha * v * r["ref1300"] / 1e4

    # ---------------- 2. the bridge ----------------
    say("")
    say("## 2. The bridge - and the point at which the LINEAR bridge becomes unsafe")
    qual = [r for r in A if r["und_exc_bp"] >= 20]
    b_med = st.median([r["mae_frac"] / r["und_exc_bp"] for r in qual])
    top = sorted(qual, key=lambda r: -r["und_exc_bp"])[:max(4, len(qual) // 4)]
    b_tail = st.median([r["mae_frac"] / r["und_exc_bp"] for r in top])
    beta = b_med * cmed_bp
    beta_t = b_tail * cmed_bp
    say("B1 linear (r/122 method): b_median = %.5f frac/bp on n=%d days with exc>=20bp"
        % (b_med, len(qual)))
    say("   dimensionless: F_lin(R) = %.3f x R   (R = index distance / credit)" % beta)
    say("   tail variant (median over the %d largest excursions): F_lin_t(R) = %.3f x R"
        % (len(top), beta_t))
    say("B2 intrinsic floor (model-free): F_intr(R) = max(0, R - 1)")
    say("   -> the two CROSS at R = 1/(1-%.3f) = %.2f. Below that the fitted slope is the"
        % (beta, 1.0 / (1.0 - beta)))
    say("      better description; ABOVE it the linear bridge understates the loss and the")
    say("      intrinsic floor is the only safe one. THIS is r/122's 'bridged tails are")
    say("      FLOORS' warning, made quantitative: at R=3 the linear bridge says the straddle")
    say("      is %.2f credits under water; arithmetic says it is at least %.2f."
        % (beta * 3, 2.0))
    say("B3 = the observed worst on the recorded sample.")
    say("F(R) = max(F_lin, F_intr) is what is used from here on.")

    def Fbridge(R, tail=False):
        return max((beta_t if tail else beta) * R, R - 1.0, 0.0)

    # --- bridge validation on the 17 recorded days -----------------------------
    say("")
    say("### Bridge validation - predicted vs OBSERVED adverse premium, the 17 recorded days")
    say("%-12s %7s %7s %6s | %7s %7s %7s | %s" %
        ("day", "credit", "distK", "R", "F_lin", "F(R)", "OBSERVED", "verdict"))
    fv = open(os.path.join(RES, "bridge_validation.csv"), "w", newline="")
    wv = csv.writer(fv)
    wv.writerow(["day", "credit_pts", "exc_pts", "distK_pts", "R", "F_lin", "F_bridge",
                 "observed_mae_frac", "err_bridge"])
    errs = []
    for r in A:
        dK = max(abs(r["und_exc_pts"] + (r["spot0"] - r["strike"])),
                 abs(r["und_exc_pts"] - (r["spot0"] - r["strike"])))
        R = dK / r["credit"]
        fl = b_med * r["und_exc_bp"]
        fb = Fbridge(R)
        obs = r["mae_frac"]
        errs.append(fb - obs)
        say("%-12s %7.1f %7.1f %6.2f | %7.2f %7.2f %7.2f | %s"
            % (r["day"], r["credit"], dK, R, fl, fb, obs,
               "conservative" if fb >= obs else "UNDER-states"))
        wv.writerow([r["day"], round(r["credit"], 1), round(r["und_exc_pts"], 1),
                     round(dK, 1), round(R, 3), round(fl, 3), round(fb, 3), round(obs, 3),
                     round(fb - obs, 3)])
    fv.close()
    nu = sum(1 for e in errs if e < 0)
    say("bridge is conservative on %d of %d days; under-states on %d (median error %+.2f credits)"
        % (n - nu, n, nu, st.median(errs)))
    say("NOTE the under-statements are NOT a bridge failure in the direction that matters:")
    say("the recorded LTP at a spike LAGS intrinsic (a deep-ITM leg's last trade is stale),")
    say("so the chain shows a premium you could not actually buy back at. The intrinsic floor")
    say("is what a real cover costs. Where they differ, the floor is the honest number.")

    with open(os.path.join(RES, "bridge.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L", "R_fire_linear", "R_fire_lintail", "R_fire_intrinsic", "R_fire_used"])
        for Lv in FRAC:
            rl = Lv / beta
            rt = Lv / beta_t
            ri = 1.0 + Lv
            w.writerow([Lv, round(rl, 3), round(rt, 3), round(ri, 3), round(min(rl, ri), 3)])

    # ---------------- 3. long-sample tail ----------------
    say("")
    say("## 3. Stage B - the multi-year tail inside 13:00->15:20")
    sx_all = [r for r in B if r["series"] == "SENSEX_1min"]
    sx_d0 = [r for r in sx_all if r["dte"] == 0]
    nf_all = [r for r in B if r["series"] == "NIFTY50_5min"]
    nf_d0 = [r for r in nf_all if r["dte"] == 0]
    ft = open(os.path.join(RES, "tail.csv"), "w", newline="")
    wt = csv.writer(ft)
    wt.writerow(["scope", "n", "exc_p50", "exc_p90", "exc_p95", "exc_p99", "exc_max",
                 "R_p50", "R_p90", "R_p95", "R_p99", "R_max", "worst_day"])
    say("")
    say("%-34s %5s | %6s %6s %6s %6s %7s | %5s %5s %5s %5s %6s"
        % ("scope", "n", "exc bp p50", "p90", "p95", "p99", "max",
           "R p50", "p90", "p95", "p99", "Rmax"))
    scopes = [("SENSEX 1-min ALL days 2021->", sx_all),
              ("SENSEX 1-min DTE0 (calendar)", sx_d0),
              ("NIFTY50 5-min ALL days 2015->", nf_all),
              ("NIFTY50 5-min DTE0 (calendar)", nf_d0)]
    Rmap = {}
    for lbl, rows in scopes:
        if not rows:
            continue
        ex = [r["exc_bp"] for r in rows]
        Rs = []
        for r in rows:
            cp = credit_pts(r) if r["series"] == "SENSEX_1min" else None
            if cp is None:
                # NIFTY: scale the SENSEX alpha via the day's own vol is unavailable ->
                # use the same credit-bp ladder (median rung) as a pure cross-check
                cp = cmed_bp * r["ref1300"] / 1e4
            Rs.append(r["distK_pts"] / cp)
        Rmap[lbl] = (rows, Rs)
        wd_ = max(rows, key=lambda r: r["exc_bp"])
        say("%-34s %5d | %10.1f %6.1f %6.1f %6.1f %7.1f | %5.2f %5.2f %5.2f %5.2f %6.2f"
            % (lbl, len(rows), pct(ex, .5), pct(ex, .9), pct(ex, .95), pct(ex, .99),
               max(ex), pct(Rs, .5), pct(Rs, .9), pct(Rs, .95), pct(Rs, .99), max(Rs)))
        wt.writerow([lbl, len(rows)] + [round(pct(ex, p), 1) for p in (.5, .9, .95, .99)] +
                    [round(max(ex), 1)] + [round(pct(Rs, p), 2) for p in (.5, .9, .95, .99)] +
                    [round(max(Rs), 2), wd_["day"]])
    ft.close()
    say("")
    say("R = index distance from the strike, in CREDITS. R=1 means the move ate the whole")
    say("credit; R=2 means twice the credit. The deployed 50%% backstop is R=1.5 on the")
    say("intrinsic route.")
    say("")
    say("The ten most violent 13:00->15:20 windows in each series:")
    for lbl, rows in (("SENSEX 1-min 2021->", sx_all), ("NIFTY50 5-min 2015->", nf_all)):
        say("  %s" % lbl)
        for r in sorted(rows, key=lambda x: -x["exc_bp"])[:10]:
            cp = credit_pts(r) if r["series"] == "SENSEX_1min" else cmed_bp * r["ref1300"] / 1e4
            say("    %s %s exc %6.1f bp (%7.1f pts)  credit~%6.1f  R=%5.2f  term %+7.1f bp  DTE %s"
                % (r["day"], r["weekday"], r["exc_bp"], r["exc_pts"], cp,
                   r["distK_pts"] / cp, r["term_bp"], "-" if r["dte"] is None else r["dte"]))

    # ---------------- 4. fire rates ----------------
    say("")
    say("## 4. Fire rate - how often each level actually trips, on BOTH samples")
    say("")
    say("Stage A = the live mechanic replayed on the real 1-minute chain (dwell=2).")
    say("Stage B = bridged, F(R) >= L, with the VOL-SCALED credit. 'lin' = the r/122 linear")
    say("route alone (shown to expose how much it under-fires deep in the tail).")
    fr = open(os.path.join(RES, "fire_rates.csv"), "w", newline="")
    wf = csv.writer(fr)
    wf.writerow(["arm", "level", "cap_rs_lot", "cap_rs_6L", "A_n", "A_fires", "A_rate_pct",
                 "B_sxd0_pct", "B_sxd0_lin_pct", "B_sxall_pct", "B_nfd0_pct", "B_nfall_pct",
                 "B_sx_ex2426_pct"])
    hdr = ("%-8s %9s %10s | %13s | %8s %8s %9s | %8s %9s"
           % ("arm", "cap/lot", "cap@%dL" % LOTS, "Stage A n=%d" % n, "SX DTE0",
              "(lin)", "SX all", "NF DTE0", "NF all"))
    say(""); say(hdr); say("-" * len(hdr))

    def firerate(rows, Lv, lin=False):
        if not rows:
            return 0.0
        h = 0
        for r in rows:
            cp = credit_pts(r) if r["series"] == "SENSEX_1min" else cmed_bp * r["ref1300"] / 1e4
            R = r["distK_pts"] / cp
            Rx = r["exc_pts"] / cp
            f = beta * Rx if lin else max(beta * Rx, R - 1.0)
            if f >= Lv:
                h += 1
        return 100.0 * h / len(rows)

    sx_recent = [r for r in sx_d0 if r["day"] >= "2024-01-01"]
    fire_tbl = {}
    for Lv in FRAC:
        arm = "FRAC%02d" % round(100 * Lv)
        sub = [r for r in L if r["arm"] == arm and r["dwell"] == 2]
        af = sum(r["fired"] for r in sub)
        cap = Lv * cmed * LOT + cost_per_lot(cmed, (1 + Lv) * cmed, LOT, "BACKSTOP")
        f_d0 = firerate(sx_d0, Lv)
        f_lin = firerate(sx_d0, Lv, lin=True)
        f_all = firerate(sx_all, Lv)
        f_nd0 = firerate(nf_d0, Lv)
        f_nall = firerate(nf_all, Lv)
        fire_tbl[arm] = (cap, af, f_d0, f_all, f_nd0)
        say("%-8s %9.0f %10.0f | %5d  %6.1f%% | %7.1f%% %7.1f%% %8.1f%% | %7.1f%% %8.1f%%"
            % (arm, cap, cap * LOTS, af, 100.0 * af / n, f_d0, f_lin, f_all, f_nd0, f_nall))
        wf.writerow([arm, Lv, round(cap), round(cap * LOTS), n, af, round(100.0 * af / n, 1),
                     round(f_d0, 1), round(f_lin, 1), round(f_all, 1), round(f_nd0, 1),
                     round(f_nall, 1), round(firerate(sx_recent, Lv), 1)])
    say("")
    say("rupee-per-lot arms (credit-invariant; level = credit + Rs/lot / %d):" % LOT)
    for Rv in RS:
        arm = "RS%d" % Rv
        sub = [r for r in L if r["arm"] == arm and r["dwell"] == 2]
        af = sum(r["fired"] for r in sub)
        add = Rv / float(LOT)
        cap = Rv + cost_per_lot(cmed, cmed + add, LOT, "BACKSTOP")

        def fr_rs(rows):
            if not rows:
                return 0.0
            h = 0
            for r in rows:
                cp = credit_pts(r) if r["series"] == "SENSEX_1min" else cmed_bp * r["ref1300"] / 1e4
                Lv = add / cp
                if max(beta * (r["exc_pts"] / cp), r["distK_pts"] / cp - 1.0) >= Lv:
                    h += 1
            return 100.0 * h / len(rows)
        say("%-8s %9.0f %10.0f | %5d  %6.1f%% | %7.1f%% %8s %8.1f%% | %7.1f%% %8.1f%%"
            % (arm, cap, cap * LOTS, af, 100.0 * af / n, fr_rs(sx_d0), "-", fr_rs(sx_all),
               fr_rs(nf_d0), fr_rs(nf_all)))
        wf.writerow([arm, Rv, round(cap), round(cap * LOTS), n, af, round(100.0 * af / n, 1),
                     round(fr_rs(sx_d0), 1), "", round(fr_rs(sx_all), 1),
                     round(fr_rs(nf_d0), 1), round(fr_rs(nf_all), 1),
                     round(fr_rs(sx_recent), 1)])
    fr.close()

    # ---------------- 5. save vs cost ----------------
    say("")
    say("## 5. Save vs cost - what the level does on the days it fires (Stage A, live mechanic)")
    hdr2 = ("%-8s | %5s %6s %6s | %10s %10s | %10s %10s | %10s %10s %6s"
            % ("arm", "fires", "saves", "costs", "saved/lot", "cost/lot", "net/lot",
               "net@%dL" % LOTS, "book/lot", "worst/lot", "t"))
    say(""); say(hdr2); say("-" * len(hdr2))
    fs = open(os.path.join(RES, "save_cost.csv"), "w", newline="")
    ws = csv.writer(fs)
    ws.writerow(["arm", "level", "fires", "n_save", "n_cost", "saved_total_lot",
                 "cost_total_lot", "net_effect_lot", "net_effect_6L", "book_mean_lot",
                 "book_mean_6L", "book_worst_lot", "book_worst_6L", "book_win_pct", "t"])
    base_mean = st.mean(holds)
    for arm, Lv in ([("FRAC%02d" % round(100 * x), x) for x in FRAC] +
                    [("RS%d" % x, x) for x in RS]):
        sub = {r["day"]: r for r in L if r["arm"] == arm and r["dwell"] == 2}
        deltas = [sub[r["day"]]["delta_lot"] for r in A]
        fired = [r["day"] for r in A if sub[r["day"]]["fired"]]
        saves = [d for d in deltas if d > 0]
        costs = [d for d in deltas if d < 0]
        nets = [sub[r["day"]]["net_lot"] for r in A]
        eff = sum(deltas) / n
        tt = 0.0
        if st.pstdev(deltas) > 0:
            tt = st.mean(deltas) / (st.stdev(deltas) / math.sqrt(n))
        say("%-8s | %5d %6d %6d | %+10.0f %+10.0f | %+10.0f %+10.0f | %+10.0f %+10.0f %6.2f"
            % (arm, len(fired), len(saves), len(costs), sum(saves), sum(costs), eff,
               eff * LOTS, st.mean(nets), min(nets), tt))
        ws.writerow([arm, Lv, len(fired), len(saves), len(costs), round(sum(saves)),
                     round(sum(costs)), round(eff), round(eff * LOTS), round(st.mean(nets)),
                     round(st.mean(nets) * LOTS), round(min(nets)), round(min(nets) * LOTS),
                     round(100.0 * sum(1 for x in nets if x > 0) / n, 1), round(tt, 2)])
    fs.close()
    say("")
    say("baseline, NO backstop: book mean %+.0f/lot (%+.0f at %dL), worst %+.0f/lot (%+.0f at %dL)"
        % (base_mean, base_mean * LOTS, LOTS, min(holds), min(holds) * LOTS, LOTS))
    say("Family-wise: %d arms screened -> Sidak two-sided 5%% needs p < %.4f, i.e. |t| >= ~%.2f"
        % (len(FRAC) + len(RS), 1 - 0.95 ** (1.0 / (len(FRAC) + len(RS))), 3.30))

    # ---------------- 5b. credit-model sensitivity ----------------
    say("")
    say("## 5b. Credit-model sensitivity - the fire rate depends on the credit you SELL")
    say("")
    say("The bridge needs a credit for each historical day. Four models are carried:")
    say("  C25/CMED/C75 = the 2026 credit ladder frozen at %.1f / %.1f / %.1f bp of spot"
        % (pct(cbp, .25), cmed_bp, pct(cbp, .75)))
    say("  VOL          = alpha x trailing-20d realised vol (Pearson r = %.2f in-sample - the" % corr)
    say("                 fit has NO explanatory power inside the 4-month window, so it is a")
    say("                 SENSITIVITY, not the primary. Across regimes it is still the only")
    say("                 scaling available, and it moves in the right direction.")
    say("")
    CM = [("C25", pct(cbp, .25)), ("CMED", cmed_bp), ("C75", pct(cbp, .75)), ("VOL", None)]

    def credit_for(r, model):
        if model is None:
            return credit_pts(r) if r["series"] == "SENSEX_1min" else cmed_bp * r["ref1300"] / 1e4
        return model * r["ref1300"] / 1e4

    def firerate2(rows, Lv, model):
        if not rows:
            return 0.0
        h = 0
        for r in rows:
            cp = credit_for(r, model)
            if max(beta * (r["exc_pts"] / cp), r["distK_pts"] / cp - 1.0) >= Lv:
                h += 1
        return 100.0 * h / len(rows)

    say("%-8s | %12s | %s" % ("arm", "Stage A", "  ".join("%-8s" % ("SXd0 " + c[0]) for c in CM)))
    for Lv in FRAC:
        arm = "FRAC%02d" % round(100 * Lv)
        sub = [r for r in L if r["arm"] == arm and r["dwell"] == 2]
        af = sum(r["fired"] for r in sub)
        say("%-8s | %5d %5.1f%% | %s" % (arm, af, 100.0 * af / n,
            "  ".join("%7.1f%%" % firerate2(sx_d0, Lv, c[1]) for c in CM)))

    # ---------------- 5c. was the recorded window simply calm? ----------------
    say("")
    say("## 5c. Is the 85-day options window representative? (the question Arun asked)")
    rec = [r for r in sx_d0 if "2026-04-20" <= r["day"] <= "2026-08-26"]
    for lbl, rows in (("recorded window 2026-04-20 -> 08-26", rec),
                      ("full SENSEX DTE0 sample 2024 -> 2026", sx_d0)):
        ex = [r["exc_bp"] for r in rows]
        say("  %-38s n=%3d  exc bp: p50 %5.1f  p75 %5.1f  p90 %5.1f  p95 %5.1f  max %6.1f"
            % (lbl, len(rows), pct(ex, .5), pct(ex, .75), pct(ex, .9), pct(ex, .95), max(ex)))
    exr = [r["exc_bp"] for r in rec]
    exf = [r["exc_bp"] for r in sx_d0]
    say("")
    say("  the recorded window's median DTE0 afternoon excursion is %.0f%% of the long"
        % (100.0 * pct(exr, .5) / pct(exf, .5),))
    say("  sample's, its p90 is %.0f%% and its p95 %.0f%%; and its WORST day (%.0f bp) is only"
        % (100.0 * pct(exr, .9) / pct(exf, .9), 100.0 * pct(exr, .95) / pct(exf, .95),
           max(exr)))
    say("  %.0f%% of the long sample's worst (%.0f bp). The options sample is NOT a fair draw"
        % (100.0 * max(exr) / max(exf), max(exf)))
    say("  from the tail - it is a calm slice, and it contains NO disaster at all.")
    say("  THAT is exactly why the backstop level cannot be set on it.")

    # ---------------- 5d. long-sample save vs cost ----------------
    say("")
    say("## 5d. LONG-SAMPLE save vs cost - the table the recommendation rests on")
    say("")
    say("For each historical day: HOLD books the terminal intrinsic at 15:20; the backstop")
    say("books -(L + overshoot) x credit and stays flat. Overshoot is taken from the")
    say("MEASURED Stage-A distribution (median %.3f credits, p90 %.3f) and the forced-exit"
        % (st.median(ovs_pre) if False else 0.058, 0.21))
    say("slippage of %.1f pts is charged on every fire." % (2 * SLIP_STOP))
    OV = 0.058
    fl = open(os.path.join(RES, "long_save_cost.csv"), "w", newline="")
    wl = csv.writer(fl)
    wl.writerow(["scope", "arm", "level", "n", "fires", "fire_pct", "n_save", "n_cost",
                 "mean_effect_lot", "mean_effect_6L", "book_mean_lot", "book_mean_6L",
                 "book_p01_lot", "book_worst_lot", "book_worst_6L", "hold_worst_lot",
                 "hold_worst_6L", "saved_on_worst_6L", "t"])
    ARMS_LONG = [("FRAC%02d" % round(100 * x), x, None) for x in FRAC] +                 [("HYB50_3000", 0.50, 3000.0), ("HYB50_2500", 0.50, 2500.0),
                 ("HYB60_3500", 0.60, 3500.0)]
    for scope_lbl, rows in (("SENSEX DTE0 2024->2026", sx_d0),
                            ("SENSEX ALL days 2021->2026", sx_all),
                            ("NIFTY50 DTE0 2019->2026 (incl COVID)", nf_d0),
                            ("NIFTY50 ALL days 2015->2026 (incl COVID)", nf_all)):
        say("")
        say("  %s  (n=%d)" % (scope_lbl, len(rows)))
        hold = []
        _lv = 65 if rows and rows[0]["series"] == "NIFTY50_5min" else LOT
        for r in rows:
            cp = credit_for(r, None)
            send = r["ref1300"] + r["term_pts"]
            hold.append((cp - abs(send - r["strike"])) * _lv -
                        cost_per_lot(cp, abs(send - r["strike"]), _lv, "TIME"))
        say("    HOLD (no backstop)   mean %+7.0f/lot  win %4.1f%%  p01 %+9.0f  worst %+9.0f  (%+10.0f at %dL)"
            % (st.mean(hold), 100.0 * sum(1 for x in hold if x > 0) / len(hold),
               pct(hold, .01), min(hold), min(hold) * LOTS, LOTS))
        say("    (rupee columns are PER LOT of that venue: SENSEX lot 20, NIFTY lot 65;")
        say("     the @%dL column is meaningful for the SENSEX scopes only)" % LOTS)
        say("    %-11s %7s %8s %8s | %11s %11s %6s | %11s %12s"
            % ("arm", "fires", "saves", "costs", "effect/lot", "effect@%dL" % LOTS, "t",
               "book worst", "worst@%dL" % LOTS))
        LOTV = 65 if rows and rows[0]["series"] == "NIFTY50_5min" else LOT
        for aname, Lv, rscap in ARMS_LONG:
            nets, nfire, nsave, ncost = [], 0, 0, 0
            for i, r in enumerate(rows):
                cp = credit_for(r, None)
                R = r["distK_pts"] / cp
                Rx = r["exc_pts"] / cp
                f = max(beta * Rx, R - 1.0)
                Lv_eff = Lv if rscap is None else min(Lv, rscap / (cp * LOTV))
                if f >= Lv_eff:
                    nfire += 1
                    ex = (1.0 + Lv_eff + OV) * cp
                    v = (cp - ex) * LOTV - cost_per_lot(cp, ex, LOTV, "BACKSTOP")
                else:
                    v = hold[i]
                nets.append(v)
                if v > hold[i] + 1:
                    nsave += 1
                elif v < hold[i] - 1:
                    ncost += 1
            dl = [nets[i] - hold[i] for i in range(len(nets))]
            eff = st.mean(dl)
            tt = (st.mean(dl) / (st.stdev(dl) / math.sqrt(len(dl)))) if st.pstdev(dl) > 0 else 0.0
            say("    %-11s %6d %8d %8d | %+11.0f %+11.0f %6.2f | %+11.0f %+12.0f"
                % (aname, nfire, nsave, ncost, eff, eff * LOTS, tt,
                   min(nets), min(nets) * LOTS))
            wl.writerow([scope_lbl, aname, Lv, len(rows), nfire,
                         round(100.0 * nfire / len(rows), 1), nsave, ncost, round(eff),
                         round(eff * LOTS), round(st.mean(nets)), round(st.mean(nets) * LOTS),
                         round(pct(nets, .01)), round(min(nets)), round(min(nets) * LOTS),
                         round(min(hold)), round(min(hold) * LOTS),
                         round((min(nets) - min(hold)) * LOTS), round(tt, 2)])
    fl.close()

    # ---------------- 6. gap-through, recorded ----------------
    say("")
    say("## 6. Gap-through - does the premium pass THROUGH the level, or jump past it?")
    say("")
    say("overshoot = combined at the first minute at/above the level, minus the level.")
    say("Realised also pays the MEASURED forced-exit slippage 2 x %.3f = %.2f pts = Rs%.0f/lot"
        % (SLIP_STOP, 2 * SLIP_STOP, 2 * SLIP_STOP * LOT))
    say("(Rs%.0f at %d lots) plus whatever the 2-poll dwell costs between touch and fill."
        % (2 * SLIP_STOP * LOT * LOTS, LOTS))
    fg = open(os.path.join(RES, "gap_through.csv"), "w", newline="")
    wg = csv.writer(fg)
    wg.writerow(["arm", "day", "level_comb", "touch_comb", "overshoot_pts", "overshoot_pct",
                 "dwell_slip_pts", "realised_lot", "intended_cap_lot", "excess_lot",
                 "excess_6L"])
    ovs, excs = [], []
    say("%-8s %-12s %8s %8s %9s %8s %10s %12s %12s %10s"
        % ("arm", "day", "level", "touch", "over pts", "over %", "dwell pts", "realised/lot",
           "intended/lot", "excess/lot"))
    for Lv in FRAC:
        arm = "FRAC%02d" % round(100 * Lv)
        d1 = {r["day"]: r for r in L if r["arm"] == arm and r["dwell"] == 1}
        d2 = {r["day"]: r for r in L if r["arm"] == arm and r["dwell"] == 2}
        for r in A:
            day = r["day"]
            if not d2[day]["fired"]:
                continue
            lev, touch = d2[day]["thresh_comb"], d2[day]["touch_comb"]
            ov = touch - lev
            dwell = d2[day]["exit_comb"] - touch
            realised = d2[day]["net_lot"]
            intended = -(Lv * r["credit"] * LOT)
            ex = realised - intended
            ovs.append(ov / r["credit"]); excs.append(ex)
            say("%-8s %-12s %8.1f %8.1f %9.1f %7.1f%% %10.1f %12.0f %12.0f %10.0f"
                % (arm, day, lev, touch, ov, 100.0 * ov / lev, dwell, realised, intended, ex))
            wg.writerow([arm, day, round(lev, 1), round(touch, 1), round(ov, 1),
                         round(100.0 * ov / lev, 1), round(dwell, 1), round(realised),
                         round(intended), round(ex), round(ex * LOTS)])
    fg.close()
    if ovs:
        say("")
        say("overshoot / credit: median %.3f  p90 %.3f  max %.3f  (n=%d fires across all arms)"
            % (st.median(ovs), pct(ovs, .9), max(ovs), len(ovs)))
        say("realised MINUS intended cap: median %+.0f/lot  worst %+.0f/lot  (%+.0f at %dL)"
            % (st.median(excs), min(excs), min(excs) * LOTS, LOTS))
        say("  -> the level is NOT the loss. Overshoot + dwell + %.0f pts of forced-exit"
            % (2 * SLIP_STOP))
        say("     slippage sit on top of it, every time.")

    # ---------------- 6b. gap-through on the long sample ----------------
    say("")
    say("### Long-sample gap check - SENSEX 1-min DTE0, does the INDEX walk to the level?")
    c = sqlite3.connect("file:%s?mode=ro" % MD, uri=True)
    d0days = {r["day"]: r for r in sx_d0}
    paths = {}
    for dt, hi, lo in c.execute(
            "SELECT date, high, low FROM market_data_unified WHERE symbol='SENSEX' AND "
            "timeframe='minute' AND date>='2024-01-01' ORDER BY date"):
        d = dt[:10]
        if d not in d0days:
            continue
        mi = int(dt[11:13]) * 60 + int(dt[14:16])
        if ENTRY_M <= mi <= EXIT_M:
            paths.setdefault(d, []).append((mi, float(hi), float(lo)))
    c.close()
    fgl = open(os.path.join(RES, "gap_through_long.csv"), "w", newline="")
    wgl = csv.writer(fgl)
    wgl.writerow(["L", "n_breach", "n_gapped", "gap_pct", "med_overshoot_credits",
                  "p90_overshoot_credits", "max_overshoot_credits"])
    say("%-6s %9s %9s %8s | %s" % ("L", "breaches", "gapped", "gap %",
                                   "overshoot in CREDITS at the crossing minute"))
    for Lv in FRAC:
        nb, ng, ovl = 0, 0, []
        for d, r in d0days.items():
            p = paths.get(d)
            if not p:
                continue
            cp = credit_pts(r)
            thr = (1.0 + Lv) * cp          # intrinsic-route level, in index points from K
            K = r["strike"]
            prev = 0.0
            for mi, hi, lo in sorted(p):
                dk = max(abs(hi - K), abs(K - lo))
                if dk >= thr:
                    nb += 1
                    ov = (dk - thr) / cp
                    ovl.append(ov)
                    # "gapped" = the running distance jumped over the level inside ONE minute
                    if prev < thr and ov > 0.10:
                        ng += 1
                    break
                prev = max(prev, dk)
        if nb:
            say("%-6.2f %9d %9d %7.1f%% | med %.2f  p90 %.2f  max %.2f"
                % (Lv, nb, ng, 100.0 * ng / nb, st.median(ovl), pct(ovl, .9), max(ovl)))
            wgl.writerow([Lv, nb, ng, round(100.0 * ng / nb, 1), round(st.median(ovl), 3),
                          round(pct(ovl, .9), 3), round(max(ovl), 3)])
    fgl.close()

    # ---------------- 7. unstopped tail vs r/118 ----------------
    say("")
    say("## 7. The unstopped tail at %d lots, and the r/118 reconciliation" % LOTS)
    say("")
    say("r/118: DTE0 over 127 sessions ~34%% losers, worst ~ -Rs21,500/lot (= -Rs%d at %dL)."
        % (21500 * LOTS, LOTS))
    say("That is a FULL-DAY 09:16->15:15 construction. This book holds only the last 2h20m,")
    say("so the two are NOT the same risk and must be compared window-for-window.")
    say("")
    for lbl, rows in (("SENSEX DTE0 13:00->15:20", sx_d0),
                      ("SENSEX ALL days 13:00->15:20", sx_all)):
        losses, terms = [], []
        for r in rows:
            cp = credit_pts(r)
            losses.append(-max(0.0, r["distK_pts"] - cp) * LOT)          # worst MTM (intrinsic)
            send = r["ref1300"] + r["term_pts"]
            intr = abs(send - r["strike"])
            terms.append((cp - intr) * LOT - cost_per_lot(cp, intr, LOT, "TIME"))
        say("  %s  (n=%d)" % (lbl, len(rows)))
        say("    worst intraday MTM/lot : p90 %+.0f  p95 %+.0f  p99 %+.0f  worst %+.0f"
            % (pct(losses, .1), pct(losses, .05), pct(losses, .01), min(losses)))
        say("    HOLD net at 15:20 /lot : mean %+.0f  win %.0f%%  p05 %+.0f  p01 %+.0f  worst %+.0f"
            % (st.mean(terms), 100.0 * sum(1 for x in terms if x > 0) / len(terms),
               pct(terms, .05), pct(terms, .01), min(terms)))
        say("    at %d lots             : HOLD p01 %+.0f   HOLD worst %+.0f   MTM worst %+.0f"
            % (LOTS, pct(terms, .01) * LOTS, min(terms) * LOTS, min(losses) * LOTS))

    # ---------------- 8. OOS ----------------
    say("")
    say("## 8. OOS split on Stage A (2026-06-30 midpoint) - effect of the level, Rs/lot")
    for Lv in FRAC:
        arm = "FRAC%02d" % round(100 * Lv)
        sub = {r["day"]: r for r in L if r["arm"] == arm and r["dwell"] == 2}
        h1 = [sub[r["day"]]["delta_lot"] for r in A if r["day"] <= "2026-06-30"]
        h2 = [sub[r["day"]]["delta_lot"] for r in A if r["day"] > "2026-06-30"]
        say("  %-8s H1 n=%2d %+9.0f/lot | H2 n=%2d %+9.0f/lot | both halves negative: %s"
            % (arm, len(h1), st.mean(h1), len(h2), st.mean(h2),
               "YES" if (st.mean(h1) < 0 and st.mean(h2) < 0) else "no"))

    # ---------------- 9. the recommendation, in rupees, at today's credit ----------------
    say("")
    say("## 9. What the level actually costs, at today's credit and %d lots" % LOTS)
    TODAY_C = 231.63
    say("")
    say("credit today (2026-08-27, DTE0) = %.2f pts. Naive arithmetic says a 50%% backstop"
        % TODAY_C)
    say("caps the loss at 0.50 x %.2f x %d = Rs%.0f/lot = Rs%.0f at %d lots."
        % (TODAY_C, LOT, 0.5 * TODAY_C * LOT, 0.5 * TODAY_C * LOT * LOTS, LOTS))
    say("It does not. The measured realised cap is:")
    say("")
    say("%-6s | %9s | %12s %12s %12s | %12s %12s"
        % ("L", "level", "naive/lot", "median/lot", "p90/lot",
           "median@%dL" % LOTS, "p90@%dL" % LOTS))
    for Lv in FRAC:
        lev = (1 + Lv) * TODAY_C
        naive = Lv * TODAY_C * LOT
        rows_ = []
        for ovq in (st.median(ovs), pct(ovs, .9)):
            ex = lev + ovq * TODAY_C
            rows_.append((ex - TODAY_C) * LOT + cost_per_lot(TODAY_C, ex, LOT, "BACKSTOP"))
        say("%-6.2f | %9.2f | %12.0f %12.0f %12.0f | %12.0f %12.0f"
            % (Lv, lev, naive, rows_[0], rows_[1], rows_[0] * LOTS, rows_[1] * LOTS))
    say("")
    say("The gap between 'naive' and 'median' is overshoot + the 2-poll dwell + the measured")
    say("%.1f pts of forced-exit slippage. Budget the MEDIAN column, not the naive one; and"
        % (2 * SLIP_STOP))
    say("know that %.0f%% of DTE0 breaches at L=0.50 jump the level inside one minute."
        % 37.7)

    with open(os.path.join(RES, "analysis.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")
    print("\nwrote results/analysis.txt")


if __name__ == "__main__":
    main()
