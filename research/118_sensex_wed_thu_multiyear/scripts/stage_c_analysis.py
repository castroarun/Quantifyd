#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/118 Stage C — put the three datasets side by side and look for a consensus.

Inputs (all produced by the other scripts in this folder):
  stage_a_day_characterisation.csv   1-min real chain, 53 days   (rupee truth, tiny sample)
  stage_a2_bhav_straddle_daily.csv   real BSE bhavcopy, ~600 days (real options, 2.6 years)
  stage_b_daily_price_action.csv     1-min index, 1354 days       (underlying only, 5.6 years)

The point of the exercise: the SENSEX weekly expiry day MOVED twice inside our option
history (Friday in 2024, Tuesday for most of 2025, Thursday from Sept 2025). That natural
experiment separates two hypotheses that are otherwise hopelessly confounded:

    H_calendar : Wednesday is dangerous because it is Wednesday
    H_dte      : Wednesday is dangerous because in 2026 Wednesday happens to be DTE1

Everything below is written to results/ as CSV, and a markdown digest is printed.
"""
import csv
import math
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
MD = os.path.join(RES, "stage_c_tables.md")
WD_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri"]
LOT = 20
COST_RS = 200.0

_buf = []


def w(line=""):
    _buf.append(line)
    print(line, flush=True)


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        w("!! missing %s" % p)
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def desc(xs):
    if not xs:
        return dict(n=0)
    return dict(n=len(xs), mean=st.mean(xs), median=st.median(xs),
                sd=(st.pstdev(xs) if len(xs) < 2 else st.stdev(xs)),
                p01=pct(xs, 0.01), p05=pct(xs, 0.05), p25=pct(xs, 0.25),
                p75=pct(xs, 0.75), p90=pct(xs, 0.90), p95=pct(xs, 0.95),
                p99=pct(xs, 0.99), min=min(xs), max=max(xs),
                win=100.0 * sum(1 for x in xs if x > 0) / len(xs))


def perm_test(a, b, iters=20000, seed=7):
    """Two-sided permutation test on the difference of means. No scipy on the VPS."""
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    rnd = random.Random(seed)
    obs = abs(st.mean(a) - st.mean(b))
    pool = a + b
    na = len(a)
    hits = 0
    for _ in range(iters):
        rnd.shuffle(pool)
        if abs(st.mean(pool[:na]) - st.mean(pool[na:])) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (iters + 1)


def table(rows, cols, title):
    w("")
    w("**%s**" % title)
    w("")
    w("| " + " | ".join(cols) + " |")
    w("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        w("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")


def r0(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else "%.0f" % x


def r1(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else "%.1f" % x


def r2(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else "%.2f" % x


# ======================================================================================
# 1. STAGE A — the 53 recorded 1-minute days, every weekday
# ======================================================================================
def stage_a_section():
    A = load("stage_a_day_characterisation.csv")
    if not A:
        return
    w("## 1. Stage A — real 1-minute chain, %d recorded days (%s .. %s)"
      % (len(A), A[0]["day"], A[-1]["day"]))
    w("")
    w("09:16 ATM straddle held to 15:15. Net = gross - Rs200/lot. Lot 20.")

    rows = []
    for d in WD_ORDER:
        g = [r for r in A if r["weekday"] == d]
        nets = [f(r["hold_net"]) for r in g]
        if not nets:
            continue
        s = desc(nets)
        rows.append(dict(weekday=d, n=s["n"], dte=",".join(sorted({r["dte"] for r in g})),
                         total=r0(sum(nets)), mean=r0(s["mean"]), median=r0(s["median"]),
                         win=r0(s["win"]), worst=r0(s["min"]), best=r0(s["max"]),
                         mean_credit=r0(st.mean([f(r["credit"]) for r in g])),
                         mean_mae=r1(st.mean([f(r["mae_spot_pts"]) for r in g])),
                         mean_absmove=r1(st.mean([f(r["abs_move_term"]) for r in g]))))
    table(rows, ["weekday", "n", "dte", "total", "mean", "median", "win", "worst", "best",
                 "mean_credit", "mean_mae", "mean_absmove"],
          "Stage A — HOLD P&L (Rs/lot, net) by weekday, all five weekdays as control")

    for day_name in ("Wed", "Thu"):
        g = sorted([r for r in A if r["weekday"] == day_name], key=lambda r: r["day"])
        rows = [dict(day=r["day"], dte=r["dte"], credit=r1(f(r["credit"])),
                     move=r1(f(r["move_term"])), mae=r1(f(r["mae_spot_pts"])),
                     mae_at=r["mae_spot_hm"], dir=r["mae_spot_dir"],
                     m_c=r2(f(r["move_over_credit"])), mae_c=r2(f(r["mae_over_credit"])),
                     shape=r["shape"], worst_mtm=r0(f(r["worst_mtm_net"])),
                     hold_net=r0(f(r["hold_net"]))) for r in g]
        table(rows, ["day", "dte", "credit", "move", "mae", "mae_at", "dir", "m_c", "mae_c",
                     "shape", "worst_mtm", "hold_net"],
              "Stage A — every recorded %s, characterised" % day_name)

    # near-misses: winners that were deeply under water at some point
    nm = [r for r in A if r["near_miss"] == "1"]
    w("")
    w("Winning days that were >= Rs1,500/lot under water at their worst mark: **%d of %d** "
      "(%s)" % (len(nm), len(A), ", ".join("%s %s %s" % (r["day"], r["weekday"], r0(f(r["worst_mtm_net"]))) for r in nm) or "none"))

    # jackknife the single worst day out of each weekday
    w("")
    rows = []
    for d in WD_ORDER:
        nets = [f(r["hold_net"]) for r in A if r["weekday"] == d]
        if len(nets) < 3:
            continue
        wo = min(nets)
        rest = sorted(nets)[1:]
        rows.append(dict(weekday=d, n=len(nets), mean=r0(st.mean(nets)),
                         worst_day=r0(wo), mean_ex_worst=r0(st.mean(rest)),
                         flips="YES" if (st.mean(nets) < 0) != (st.mean(rest) < 0) else "no"))
    table(rows, ["weekday", "n", "mean", "worst_day", "mean_ex_worst", "flips"],
          "Stage A — leave-one-out: does dropping the single worst day flip the sign?")

    A_wed = [f(r["hold_net"]) for r in A if r["weekday"] == "Wed"]
    A_thu = [f(r["hold_net"]) for r in A if r["weekday"] == "Thu"]
    w("")
    w("Permutation test, Stage A Wed vs Thu mean HOLD P&L: p = **%.3f** (n=%d vs %d)"
      % (perm_test(A_wed, A_thu), len(A_wed), len(A_thu)))
    return A


# ======================================================================================
# 2. STAGE A2 — real BSE bhavcopy options, 2024-2026, the natural experiment
# ======================================================================================
def stage_a2_section():
    B = load("stage_a2_bhav_straddle_daily.csv")
    if not B:
        return
    w("")
    w("## 2. Stage A2 — real BSE option bhavcopy, %d days (%s .. %s)"
      % (len(B), B[0]["day"], B[-1]["day"]))
    w("")
    w("Open-to-close ATM straddle, front weekly expiry, ATM chosen from the causal 09:16 "
      "index level, both legs required to have real volume and OI. Unit is **points** "
      "(the SENSEX lot size changed inside the window); rupees shown at today's lot of 20.")

    eras = load("stage_a2_expiry_eras.csv")
    if eras:
        seq, cur, start = [], None, None
        for r in eras:
            if r["modal_expiry_weekday"] != cur:
                if cur:
                    seq.append((start, prev, cur))
                cur, start = r["modal_expiry_weekday"], r["month"]
            prev = r["month"]
        seq.append((start, prev, cur))
        table([dict(era_from=a, era_to=b, expiry_weekday=c) for a, b, c in seq],
              ["era_from", "era_to", "expiry_weekday"],
              "Weekly SENSEX expiry day, derived from our own bhavcopy data")

    def block(key, order, title, subset=None):
        src = subset if subset is not None else B
        rows = []
        for k in order:
            g = [r for r in src if r[key] == k]
            pts = [f(r["pnl_pts"]) for r in g]
            if len(pts) < 3:
                continue
            s = desc(pts)
            rows.append(dict(**{key: k}, n=s["n"],
                             mean_pts=r1(s["mean"]), med_pts=r1(s["median"]),
                             win=r0(s["win"]), sd=r1(s["sd"]),
                             p05=r1(s["p05"]), p01=r1(s["p01"]), worst=r1(s["min"]),
                             mean_credit=r1(st.mean([f(r["credit"]) for r in g])),
                             net_rs_lot20=r0(s["mean"] * LOT - COST_RS),
                             total_rs=r0(sum(pts) * LOT - COST_RS * len(pts))))
        table(rows, [key, "n", "mean_pts", "med_pts", "win", "sd", "p05", "p01", "worst",
                     "mean_credit", "net_rs_lot20", "total_rs"], title)
        return rows

    block("weekday", WD_ORDER, "A2 — by CALENDAR WEEKDAY, whole 2024-2026 window (the control view)")
    dtes = sorted({r["dte"] for r in B}, key=lambda x: int(x))
    block("dte", [d for d in dtes if int(d) <= 8],
          "A2 — by DTE (days to front weekly expiry), whole window")

    # --- the natural experiment: weekday behaviour inside each expiry era ---
    w("")
    w("### The natural experiment")
    w("")
    w("If Wednesday were intrinsically dangerous, it would be the worst weekday in all "
      "three eras. If the danger belongs to DTE1, it should follow the expiry day as it "
      "moves: DTE1 was **Thursday** in the Friday-expiry era, **Monday** in the "
      "Tuesday-expiry era, and **Wednesday** in the Thursday-expiry era.")
    for era, label in (("Fri", "Friday-expiry era (2024)"),
                       ("Tue", "Tuesday-expiry era (2025 Jan-Aug)"),
                       ("Thu", "Thursday-expiry era (2025 Sep - 2026 Aug)")):
        sub = [r for r in B if r["era"] == era]
        if len(sub) < 20:
            continue
        block("weekday", WD_ORDER, "A2 — by weekday inside the %s" % label, subset=sub)
        block("dte", [d for d in dtes if int(d) <= 8],
              "A2 — by DTE inside the %s" % label, subset=sub)

    # --- per-year stability for the two headline buckets ---
    rows = []
    for y in sorted({r["day"][:4] for r in B}):
        for lab, sel in (("Wed(cal)", lambda r: r["weekday"] == "Wed"),
                         ("Thu(cal)", lambda r: r["weekday"] == "Thu"),
                         ("DTE0", lambda r: r["dte"] == "0"),
                         ("DTE1", lambda r: r["dte"] == "1")):
            g = [r for r in B if r["day"][:4] == y and sel(r)]
            pts = [f(r["pnl_pts"]) for r in g]
            if len(pts) < 3:
                continue
            rows.append(dict(year=y, bucket=lab, n=len(pts), mean_pts=r1(st.mean(pts)),
                             win=r0(100.0 * sum(1 for x in pts if x > 0) / len(pts)),
                             worst=r1(min(pts))))
    table(rows, ["year", "bucket", "n", "mean_pts", "win", "worst"],
          "A2 — per-year stability (mean points per day, and the worst single day)")

    # --- tails: how often does the day blow through the credit ---
    rows = []
    for key, order in (("weekday", WD_ORDER), ("dte", [d for d in dtes if int(d) <= 4])):
        for k in order:
            g = [r for r in B if r[key] == k]
            if len(g) < 10:
                continue
            n = len(g)
            over = lambda mult: 100.0 * sum(
                1 for r in g if (f(r["terminal"]) > mult * f(r["credit"]))) / n
            rows.append(dict(bucket="%s=%s" % (key, k), n=n,
                             loss_pct=r0(100.0 * sum(1 for r in g if f(r["pnl_pts"]) < 0) / n),
                             over1x=r1(over(2.0)), over1_5x=r1(over(2.5)), over2x=r1(over(3.0)),
                             worst_pts=r1(min(f(r["pnl_pts"]) for r in g))))
    table(rows, ["bucket", "n", "loss_pct", "over1x", "over1_5x", "over2x", "worst_pts"],
          "A2 — tail frequency: % of days the straddle closed at >1x / >1.5x / >2x the credit "
          "(i.e. a loss of 1x, 1.5x, 2x the credit)")

    # --- leave-one-out on the big buckets ---
    rows = []
    for lab, sel in (("Wed(cal)", lambda r: r["weekday"] == "Wed"),
                     ("Thu(cal)", lambda r: r["weekday"] == "Thu"),
                     ("DTE0", lambda r: r["dte"] == "0"),
                     ("DTE1", lambda r: r["dte"] == "1"),
                     ("Wed & DTE1", lambda r: r["weekday"] == "Wed" and r["dte"] == "1"),
                     ("Thu & DTE0", lambda r: r["weekday"] == "Thu" and r["dte"] == "0")):
        pts = [f(r["pnl_pts"]) for r in B if sel(r)]
        if len(pts) < 5:
            continue
        rest = sorted(pts)[1:]
        rows.append(dict(bucket=lab, n=len(pts), mean_pts=r1(st.mean(pts)),
                         worst=r1(min(pts)), mean_ex_worst=r1(st.mean(rest)),
                         flips="YES" if (st.mean(pts) < 0) != (st.mean(rest) < 0) else "no"))
    table(rows, ["bucket", "n", "mean_pts", "worst", "mean_ex_worst", "flips"],
          "A2 — leave-one-out robustness: is the verdict hostage to one day?")

    wed = [f(r["pnl_pts"]) for r in B if r["weekday"] == "Wed"]
    thu = [f(r["pnl_pts"]) for r in B if r["weekday"] == "Thu"]
    d0 = [f(r["pnl_pts"]) for r in B if r["dte"] == "0"]
    d1 = [f(r["pnl_pts"]) for r in B if r["dte"] == "1"]
    w("")
    w("Permutation tests on mean P&L (points/day), 2024-2026 real options:")
    w("")
    w("- calendar **Wed vs Thu**: p = **%.3f** (n=%d vs %d, means %.1f vs %.1f pts)"
      % (perm_test(wed, thu), len(wed), len(thu), st.mean(wed), st.mean(thu)))
    w("- **DTE0 vs DTE1**: p = **%.3f** (n=%d vs %d, means %.1f vs %.1f pts)"
      % (perm_test(d0, d1), len(d0), len(d1), st.mean(d0), st.mean(d1)))
    return B


# ======================================================================================
# 3. STAGE B — 5.6 years of underlying price action
# ======================================================================================
def stage_b_section(era_map):
    P = load("stage_b_daily_price_action.csv")
    if not P:
        return
    w("")
    w("## 3. Stage B — SENSEX 1-minute price action, %d days (%s .. %s)"
      % (len(P), P[0]["day"], P[-1]["day"]))
    w("")
    w("Options do not exist for most of this window, so we measure what actually decides a "
      "short straddle: how far the index travels from 09:16, and how far it goes against "
      "you before it gets there.")

    def wd_block(src, title, field="abs_move_term", maefield="mae_pts"):
        rows = []
        for d in WD_ORDER:
            g = [r for r in src if r["weekday"] == d]
            xs = [f(r[field]) for r in g]
            ms = [f(r[maefield]) for r in g]
            if len(xs) < 10:
                continue
            s, sm = desc(xs), desc(ms)
            rows.append(dict(weekday=d, n=s["n"], mean=r1(s["mean"]), med=r1(s["median"]),
                             p75=r1(s["p75"]), p90=r1(s["p90"]), p95=r1(s["p95"]),
                             p99=r1(s["p99"]), max=r1(s["max"]),
                             mae_mean=r1(sm["mean"]), mae_p95=r1(sm["p95"]), mae_max=r1(sm["max"])))
        table(rows, ["weekday", "n", "mean", "med", "p75", "p90", "p95", "p99", "max",
                     "mae_mean", "mae_p95", "mae_max"], title)

    wd_block(P, "B — |terminal move from 09:16| and max adverse excursion, POINTS, full 2021-2026")
    wd_block([r for r in P if r["year"] in ("2021", "2022")],
             "B — same, 2021-2022 only (the high-vol regime)")
    wd_block([r for r in P if r["year"] >= "2023"],
             "B — same, 2023-2026 only (the modern regime)")
    wd_block(P, "B — from 13:00 instead of 09:16 (the afternoon construction), full window",
             field="abs_move_term13", maefield="mae13_pts")

    # --- is "Wednesday is the calmest day" stable year by year? ---
    rows = []
    for y in sorted({r["year"] for r in P}):
        rec = dict(year=y)
        means = {}
        for d in WD_ORDER:
            g = [f(r["abs_move_term"]) for r in P if r["year"] == y and r["weekday"] == d]
            if g:
                means[d] = st.mean(g)
                rec[d] = r1(means[d])
        if means:
            rec["calmest"] = min(means, key=means.get)
            rec["wildest"] = max(means, key=means.get)
            rec["wed_rank"] = str(sorted(means, key=means.get).index("Wed") + 1) + "/5" \
                if "Wed" in means else ""
        rows.append(rec)
    table(rows, ["year"] + WD_ORDER + ["calmest", "wildest", "wed_rank"],
          "B — mean |terminal move| by weekday, YEAR BY YEAR "
          "(wed_rank 1/5 = Wednesday was the calmest weekday that year)")

    # --- credit ladder: how often would a given credit have been blown through ---
    w("")
    w("### Credit ladder — how often the day's move exceeded the premium")
    w("")
    w("A short straddle loses on the day when |terminal move| > credit. We do not know the "
      "2021-2023 credits, so we sweep the credit range actually observed in 2026 rather "
      "than picking one number.")
    rows = []
    for C in (300, 400, 465, 550, 650, 720, 850):
        for d in WD_ORDER:
            g = [f(r["abs_move_term"]) for r in P if r["weekday"] == d]
            if not g:
                continue
            n = len(g)
            rows.append(dict(credit=C, weekday=d, n=n,
                             loss_pct=r1(100.0 * sum(1 for x in g if x > C) / n),
                             over1_5=r1(100.0 * sum(1 for x in g if x > 1.5 * C) / n),
                             over2=r1(100.0 * sum(1 for x in g if x > 2 * C) / n),
                             mean_pnl_pts=r1(st.mean([C - x for x in g])),
                             p05_pnl=r1(pct([C - x for x in g], 0.05)),
                             worst_pnl=r1(min(C - x for x in g))))
    table(rows, ["credit", "weekday", "n", "loss_pct", "over1_5", "over2", "mean_pnl_pts",
                 "p05_pnl", "worst_pnl"],
          "B — credit-sensitivity ladder by weekday (proxy P&L in points, 2021-2026)")

    # --- shape decomposition ---
    rows = []
    for d in WD_ORDER:
        g = [r for r in P if r["weekday"] == d]
        if not g:
            continue
        n = len(g)
        rr = [f(r["revert_ratio"]) for r in g]
        sp = [f(r["spike_share"]) for r in g]
        rows.append(dict(weekday=d, n=n,
                         trend_pct=r1(100.0 * sum(1 for x in rr if x >= 0.70) / n),
                         revert_pct=r1(100.0 * sum(1 for x in rr if x <= 0.40) / n),
                         mean_revert=r2(st.mean(rr)),
                         mean_spike_share=r2(st.mean(sp)),
                         mean_range=r1(st.mean([f(r["range_pts"]) for r in g])),
                         mean_rvol=r2(st.mean([f(r["rvol_pct"]) for r in g])),
                         mean_absgap=r2(st.mean([abs(f(r["gap_pct"])) for r in g]))))
    table(rows, ["weekday", "n", "trend_pct", "revert_pct", "mean_revert", "mean_spike_share",
                 "mean_range", "mean_rvol", "mean_absgap"],
          "B — day shape: does the move stick (trend) or come back (spike-and-revert)?")

    # --- regime buckets by causal trailing vol ---
    withtrail = [r for r in P if r["trail20_rvol_pct"] not in ("", None)]
    tv = sorted(f(r["trail20_rvol_pct"]) for r in withtrail)
    lo, hi = pct(tv, 1 / 3.0), pct(tv, 2 / 3.0)
    rows = []
    for name, sel in (("CALM (low tercile)", lambda x: x <= lo),
                      ("NORMAL", lambda x: lo < x <= hi),
                      ("STRESSED (high tercile)", lambda x: x > hi)):
        for d in WD_ORDER:
            g = [f(r["abs_move_term"]) for r in withtrail
                 if r["weekday"] == d and sel(f(r["trail20_rvol_pct"]))]
            if len(g) < 10:
                continue
            rows.append(dict(regime=name, weekday=d, n=len(g), mean=r1(st.mean(g)),
                             p95=r1(pct(g, 0.95)), max=r1(max(g)),
                             loss_at_550=r1(100.0 * sum(1 for x in g if x > 550) / len(g))))
    table(rows, ["regime", "weekday", "n", "mean", "p95", "max", "loss_at_550"],
          "B — regime split by CAUSAL trailing-20-day realised vol (terciles %.2f%% / %.2f%%)"
          % (lo, hi))

    # --- weekday significance on the long sample ---
    w("")
    w("Permutation tests on mean |terminal move| (points), 2021-2026 underlying:")
    w("")
    for a, b in (("Wed", "Thu"), ("Wed", "Mon"), ("Wed", "Tue"), ("Wed", "Fri")):
        xa = [f(r["abs_move_term"]) for r in P if r["weekday"] == a]
        xb = [f(r["abs_move_term"]) for r in P if r["weekday"] == b]
        w("- %s vs %s: p = **%.3f** (means %.1f vs %.1f pts, n=%d vs %d)"
          % (a, b, perm_test(xa, xb), st.mean(xa), st.mean(xb), len(xa), len(xb)))
    w("")
    w("(Five weekdays compared on several metrics — treat any single p near 0.05 as noise; "
      "a Bonferroni-style threshold here is roughly 0.05/10 = 0.005.)")
    return P


# ======================================================================================
# 4. Cross-validation of the bhavcopy proxy against the 1-minute truth
# ======================================================================================
def validate(A, B):
    if not A or not B:
        return
    w("")
    w("## 4. Is the bhavcopy proxy trustworthy?")
    w("")
    w("The two datasets overlap on the days where we have both the 1-minute chain and the "
      "bhavcopy. If the daily open-to-close replay tracks the true 09:16-to-15:15 replay, "
      "the 2.6-year bhavcopy sample can be believed.")
    bm = {r["day"]: r for r in B}
    pairs = [(f(r["hold_net"]), f(bm[r["day"]]["net_rs_lot20"]), r["day"], r["weekday"])
             for r in A if r["day"] in bm]
    if len(pairs) < 5:
        w("Only %d overlapping days - not enough to validate." % len(pairs))
        return
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    corr = num / den if den else float("nan")
    w("")
    w("- overlapping days: **%d**" % len(pairs))
    w("- correlation of daily P&L (1-min truth vs bhavcopy proxy): **%.3f**" % corr)
    w("- mean 1-min P&L Rs%.0f/lot vs mean proxy P&L Rs%.0f/lot (bias Rs%.0f)"
      % (mx, my, my - mx))
    w("- mean absolute difference: Rs%.0f/lot; sign agreement on %.0f%% of days"
      % (st.mean([abs(a - b) for a, b in zip(xs, ys)]),
         100.0 * sum(1 for a, b in zip(xs, ys) if (a > 0) == (b > 0)) / len(pairs)))
    rows = [dict(day=d, weekday=wd, onemin=r0(a), bhav=r0(b), diff=r0(b - a))
            for a, b, d, wd in sorted(pairs, key=lambda p: p[2])]
    table(rows[:60], ["day", "weekday", "onemin", "bhav", "diff"],
          "Overlap check, Rs/lot net (1-min chain vs bhavcopy open-to-close)")

    # --- is the bias uniform across weekdays, or would it distort the ranking? ---
    brows = []
    for d in WD_ORDER:
        g = [(a, b) for a, b, _, wd in pairs if wd == d]
        if len(g) < 3:
            continue
        brows.append(dict(weekday=d, n=len(g),
                          mean_bias_rs=r0(st.mean([b - a for a, b in g])),
                          median_bias_rs=r0(st.median([b - a for a, b in g]))))
    table(brows, ["weekday", "n", "mean_bias_rs", "median_bias_rs"],
          "Proxy bias by weekday (positive = bhavcopy proxy flatters the trade)")
    bias_pts = (my - mx) / LOT
    w("")
    w("The proxy sells at the day's very first trade (09:15, when the premium is at its "
      "richest) and buys back at the 15:30 close rather than 15:15, so it collects about "
      "**%.0f points (Rs%.0f/lot) per day of decay the live 09:16-to-15:15 trade never "
      "sees**. Every Stage A2 mean below is therefore quoted again with that haircut "
      "applied. The bias is a level shift, not a re-ranking: it is present on every "
      "weekday." % (bias_pts, my - mx))
    return bias_pts


def adjusted_section(bias_pts):
    """A2 means restated on the live construction's terms."""
    B = load("stage_a2_bhav_straddle_daily.csv")
    if not B or bias_pts is None:
        return
    w("")
    w("## 5. Stage A2 restated on the live construction's terms")
    w("")
    w("Stage A2 mean minus the %.0f-point proxy bias measured above, i.e. the best estimate "
      "of what a 09:16-to-15:15 held ATM straddle actually earned per day over 2024-2026." % bias_pts)
    rows = []
    for lab, sel in ([(d, (lambda dd: (lambda r: r["weekday"] == dd))(d)) for d in WD_ORDER] +
                     [("DTE0", lambda r: r["dte"] == "0"), ("DTE1", lambda r: r["dte"] == "1"),
                      ("Wed & DTE1 (today's rule)", lambda r: r["weekday"] == "Wed" and r["dte"] == "1"),
                      ("Thu & DTE0 (today's rule)", lambda r: r["weekday"] == "Thu" and r["dte"] == "0")]):
        pts = [f(r["pnl_pts"]) for r in B if sel(r)]
        if len(pts) < 5:
            continue
        adj = st.mean(pts) - bias_pts
        rows.append(dict(bucket=lab, n=len(pts), raw_mean_pts=r1(st.mean(pts)),
                         adj_mean_pts=r1(adj), adj_net_rs_lot20=r0(adj * LOT - COST_RS),
                         win=r0(100.0 * sum(1 for x in pts if x > 0) / len(pts)),
                         p05_pts=r1(pct(pts, 0.05)), worst_pts=r1(min(pts))))
    table(rows, ["bucket", "n", "raw_mean_pts", "adj_mean_pts", "adj_net_rs_lot20", "win",
                 "p05_pts", "worst_pts"],
          "A2 bias-adjusted — what HOLD really pays, 2024-2026 real options")

    # --- what research/114 saw vs what the long sample says ---
    A = load("stage_a_day_characterisation.csv")
    if A:
        rows = []
        for lab, wd, dte in (("Wednesday HOLD", "Wed", "1"), ("Thursday HOLD", "Thu", "0")):
            a = [f(r["hold_net"]) for r in A if r["weekday"] == wd]
            b = [f(r["pnl_pts"]) - bias_pts for r in B if r["weekday"] == wd and r["dte"] == dte]
            b_all = [f(r["pnl_pts"]) - bias_pts for r in B if r["dte"] == dte]
            rows.append(dict(rule=lab,
                             r114_n=len(a), r114_mean_rs=r0(st.mean(a)),
                             r114_win=r0(100.0 * sum(1 for x in a if x > 0) / len(a)),
                             r114_worst_rs=r0(min(a)),
                             r118_n=len(b), r118_mean_rs=r0(st.mean(b) * LOT - COST_RS),
                             r118_win=r0(100.0 * sum(1 for x in b if x > 0) / len(b)),
                             r118_worst_rs=r0(min(b) * LOT - COST_RS),
                             r118_dte_n=len(b_all),
                             r118_dte_mean_rs=r0(st.mean(b_all) * LOT - COST_RS)))
        table(rows, ["rule", "r114_n", "r114_mean_rs", "r114_win", "r114_worst_rs",
                     "r118_n", "r118_mean_rs", "r118_win", "r118_worst_rs",
                     "r118_dte_n", "r118_dte_mean_rs"],
              "research/114 (n=12, one quarter) vs research/118 (2.6 years of real options)")

        # Stage A permutation with the single outlier removed
        wed = [f(r["hold_net"]) for r in A if r["weekday"] == "Wed"]
        thu = [f(r["hold_net"]) for r in A if r["weekday"] == "Thu"]
        wed_ex = [x for x in wed if x > min(wed)]
        w("")
        w("Stage A Wed vs Thu permutation test **with 2026-07-08 removed**: p = **%.3f** "
          "(Wed mean becomes Rs%.0f/lot, from Rs%.0f). The whole significance of the "
          "Wednesday result lives in that one day."
          % (perm_test(wed_ex, thu), st.mean(wed_ex), st.mean(wed)))


def main():
    w("# research/118 — Stage C tables")
    w("")
    w("_Generated by `scripts/stage_c_analysis.py`. All figures net of 1.0 pt/leg-side "
      "slippage + Rs30/leg-side/lot charges (Rs200/lot round trip on both legs) unless "
      "the column says points._")
    A = stage_a_section()
    B = stage_a2_section()
    stage_b_section(None)
    bias = validate(A, B)
    adjusted_section(bias)
    with open(MD, "w") as fh:
        fh.write("\n".join(_buf) + "\n")
    print("\nwrote %s" % MD)


if __name__ == "__main__":
    main()
