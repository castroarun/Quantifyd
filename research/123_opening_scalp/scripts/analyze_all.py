#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/123 - analysis: Stage-0 premise tables, the (T x system) sweet-spot table,
the Stage-B tail bridge, the CPR/gap filter tests (random-skip null + placebos), the
margin arithmetic and the null-alternative comparison.

Writes: results/stage0_premise.csv, results/sweetspot.csv, results/tail_bridge.csv,
        results/filters_report.txt, results/margin_null.txt
"""
import csv, os, math, random, statistics as st, sqlite3
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
MD = "/home/arun/quantifyd/backtest_data/market_data.db"
random.seed(123)

LOT = {"NIFTY": 65, "SENSEX": 20}
COST = {"NIFTY": 250.0, "SENSEX": 200.0}


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    f = i - lo
    return xs[lo] * (1 - f) + xs[hi] * f


def tstat(xs):
    n = len(xs)
    if n < 3:
        return 0.0
    sd = st.pstdev(xs)
    return st.mean(xs) / (sd / math.sqrt(n)) if sd > 0 else 0.0


def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k2 in range(i, j + 1):
                r[s[k2]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


# ---------------- load stage A ----------------
A = list(csv.DictReader(open(os.path.join(RES, "stage_a_scalp.csv"))))
SC = [r for r in A if r["cell"] == "SCALP"]
S0 = [r for r in A if r["cell"] == "STAGE0"]

# ---------------- Stage 0 premise ----------------
with open(os.path.join(RES, "stage0_premise.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scope", "weekday", "wallclock", "n", "median", "p25", "p10",
                "pct_ge_4k", "pct_gt_0"])
    agg = defaultdict(float)
    wdmap = {}
    for r in S0:
        lots = int(r["book"].split("|lots")[1])
        agg[(r["day"], r["exit_hm"])] += lots * int(r["gross"])
        wdmap[r["day"]] = r["weekday"]
    bywd = defaultdict(lambda: defaultdict(list))
    for (d, hm), v in agg.items():
        bywd[wdmap[d]][hm].append(v)
    for wd in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        for hm in sorted(bywd[wd]):
            xs = bywd[wd][hm]
            w.writerow(["AGG_BOOK", wd, hm, len(xs), round(st.median(xs)),
                        round(q(xs, .25)), round(q(xs, .10)),
                        round(100 * sum(x >= 4000 for x in xs) / len(xs)),
                        round(100 * sum(x > 0 for x in xs) / len(xs))])
    # per book
    bb = defaultdict(lambda: defaultdict(list))
    for r in S0:
        lots = int(r["book"].split("|lots")[1])
        bb[r["book"]][r["exit_hm"]].append(lots * int(r["gross"]))
    for book in sorted(bb):
        for hm in sorted(bb[book]):
            xs = bb[book][hm]
            w.writerow([book, "", hm, len(xs), round(st.median(xs)),
                        round(q(xs, .25)), round(q(xs, .10)),
                        round(100 * sum(x >= 4000 for x in xs) / len(xs)),
                        round(100 * sum(x > 0 for x in xs) / len(xs))])

# ---------------- sweet-spot table (all cells + live-DTE) ----------------
g = defaultdict(list)
for r in SC:
    g[(r["venue"], r["entry_target"], int(r["T"]), r["arm"])].append(
        (int(r["net"]), int(r["dte_trd"]), r["day"]))
LIVE = {
    ("NIFTY", "09:16"): {0, 1, 2}, ("NIFTY", "09:30"): {0}, ("NIFTY", "10:00"): {2},
    ("SENSEX", "09:16"): {0, 1}, ("SENSEX", "10:30"): {1},
}
with open(os.path.join(RES, "sweetspot.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["venue", "entry", "T", "arm", "scope", "n", "mean", "median",
                "win_pct", "t", "worst", "total"])
    for k in sorted(g):
        for scope, dset in (("ALLDTE", None), ("LIVEDTE", LIVE.get((k[0], k[1])))):
            v = g[k] if dset is None else [x for x in g[k] if x[1] in dset]
            nets = [x[0] for x in v]
            if len(nets) < 3:
                continue
            w.writerow([k[0], k[1], k[2], k[3], scope, len(nets),
                        round(st.mean(nets)), round(st.median(nets)),
                        round(100 * sum(x > 0 for x in nets) / len(nets)),
                        round(tstat(nets), 2), min(nets), sum(nets)])

# ---------------- Stage B: tails + bridge ----------------
B = list(csv.DictReader(open(os.path.join(RES, "stage_b_scalp_days.csv"))))
# slope b per (venue, entry): median over NOSTOP scalp rows with exc>=20bp of
# (mae_pts/credit)/exc_bp  (pooled over T, r/122 style)
bslope = {}
for (ven, ent) in {(r["venue"], r["entry_target"]) for r in SC}:
    vals = []
    for r in SC:
        if r["venue"] != ven or r["entry_target"] != ent or r["arm"] != "NOSTOP":
            continue
        exc = float(r["und_exc_bp"])
        cred = float(r["credit"])
        if exc >= 20 and cred > 0:
            vals.append((float(r["mae_pts"]) / cred) / exc)
    bslope[(ven, ent)] = st.median(vals) if vals else float("nan")
# credit ladder per (venue, entry)
ladder = {}
for (ven, ent) in bslope:
    cs = sorted({(r["day"], float(r["credit"])) for r in SC
                 if r["venue"] == ven and r["entry_target"] == ent})
    cr = [c for _, c in cs]
    ladder[(ven, ent)] = (q(cr, .25), st.median(cr), q(cr, .75))
# long-sample exc percentiles per (venue, entry, T, dte-set of the live cell)
bx = defaultdict(list)
for r in B:
    key = (r["venue"], r["entry"], int(r["T"]))
    dte = r["dte_trd"]
    bx[key].append((float(r["exc_bp"]), None if dte == "" else int(dte), r["day"]))
with open(os.path.join(RES, "tail_bridge.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["venue", "entry", "T", "dte_scope", "n_px", "exc_p90", "exc_p95",
                "exc_p99", "b_slope", "credit_med", "tail_p95_rs_lot",
                "tail_p99_rs_lot", "p_csl20_hit", "p_csl25_hit"])
    for (ven, ent, T) in sorted(bx):
        live = LIVE.get((ven, ent), set())
        for scope_name, dset in (("ALL", None), ("LIVEDTE", live)):
            v = [e for e, d, _ in bx[(ven, ent, T)] if dset is None or d in dset]
            if len(v) < 30:
                continue
            b = bslope.get((ven, ent), float("nan"))
            cmed = ladder.get((ven, ent), (0, 0, 0))[1]
            lot = LOT[ven]
            p95, p99 = q(v, .95), q(v, .99)
            tail95 = cmed * b * p95 * lot + COST[ven]
            tail99 = cmed * b * p99 * lot + COST[ven]
            trip20 = 0.20 / b if b > 0 else 9e9
            trip25 = 0.25 / b if b > 0 else 9e9
            w.writerow([ven, ent, T, scope_name, len(v), round(q(v, .90), 1),
                        round(p95, 1), round(p99, 1), round(b, 5), round(cmed, 1),
                        round(tail95), round(tail99),
                        round(100 * sum(x >= trip20 for x in v) / len(v), 1),
                        round(100 * sum(x >= trip25 for x in v) / len(v), 1)])

# ---------------- Filters ----------------
rep = open(os.path.join(RES, "filters_report.txt"), "w")


def w_(s=""):
    rep.write(s + "\n")
    print(s, flush=True)


# features for options days: intraday-derived (stage B) primary, 'day' bars fill
def load_daily(sym):
    c = sqlite3.connect("file:%s?mode=ro" % MD, uri=True)
    out = {}
    for dt, o, h, l, cl in c.execute(
            "SELECT date,open,high,low,close FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' ORDER BY date", (sym,)):
        out[dt[:10]] = (o, h, l, cl)
    c.close()
    return out


def cprw(H, L, C):
    P = (H + L + C) / 3.0
    return 1e4 * abs((2 * P - (H + L) / 2.0) - (H + L) / 2.0) / C


def feats_from_daily(daily):
    ds = sorted(daily)
    from datetime import date as _date
    weeks = {}
    for d in ds:
        o, h, l, c2 = daily[d]
        y, wn, _ = _date.fromisoformat(d).isocalendar()
        k = (y, wn)
        if k not in weeks:
            weeks[k] = [h, l, c2]
        else:
            weeks[k][0] = max(weeks[k][0], h)
            weeks[k][1] = min(weeks[k][1], l)
            weeks[k][2] = c2
    wk = sorted(weeks)
    prevw = {k: (weeks[wk[i - 1]] if i > 0 else None) for i, k in enumerate(wk)}
    F = {}
    for i, d in enumerate(ds):
        if i < 2:
            continue
        d1, d2 = ds[i - 1], ds[i - 2]
        o, _, _, _ = daily[d]
        _, h1, l1, c1 = daily[d1]
        _, h2, l2, c2 = daily[d2]
        y, wn, _ = _date.fromisoformat(d).isocalendar()
        pw = prevw.get((y, wn))
        F[d] = dict(gap_bp=1e4 * (o - c1) / c1, cpr_t_bp=cprw(h1, l1, c1),
                    cpr_y_bp=cprw(h2, l2, c2),
                    cpr_w_bp=cprw(pw[0], pw[1], pw[2]) if pw else None)
    return F


DAYF = {"NIFTY": feats_from_daily(load_daily("NIFTY50")),
        "SENSEX": feats_from_daily(load_daily("SENSEX"))}
# long-sample (stage B) features primary
LONGF = defaultdict(dict)
for r in B:
    if r["T"] != "60":
        continue
    ft = {}
    for k in ("gap_bp", "cpr_t_bp", "cpr_y_bp", "cpr_w_bp"):
        ft[k] = float(r[k]) if r[k] not in ("", None) else None
    LONGF[r["venue"]][r["day"]] = ft


def feat_of(ven, day):
    f = LONGF[ven].get(day)
    if f and all(f.get(k) is not None for k in ("gap_bp", "cpr_t_bp", "cpr_y_bp", "cpr_w_bp")):
        return f
    return DAYF[ven].get(day)


FEATURES = ["gap_bp", "abs_gap_bp", "cpr_t_bp", "cpr_y_bp", "cpr_w_bp"]

w_("=" * 78)
w_("FILTERS - pre-registered: gap (signed / |gap| / direction), CPR today/yday/weekly")
w_("Discipline per r/121: long-sample fit first; options-day confirmation second;")
w_("every rule vs a 2,000-draw random-skip null of equal frequency; placebos included.")
w_("=" * 78)

# --- long-sample fit: Spearman(feature, term/exc at T=60), live-DTE days
w_("")
w_("A. LONG-SAMPLE FIT (T=60 windows, live-DTE days, Spearman)")
w_("%-8s %-6s %-10s %6s %10s %10s" % ("venue", "entry", "feature", "n", "rho_term", "rho_exc"))
for (ven, ent), dset in sorted(LIVE.items()):
    rows = [r for r in B if r["venue"] == ven and r["entry"] == ent and r["T"] == "60"
            and r["dte_trd"] != "" and int(r["dte_trd"]) in dset]
    for ftn in FEATURES:
        xs, yt, ye = [], [], []
        for r in rows:
            f = LONGF[ven].get(r["day"])
            if not f:
                continue
            base = "gap_bp" if ftn == "abs_gap_bp" else ftn
            v = f.get(base)
            if v is None:
                continue
            xs.append(abs(v) if ftn == "abs_gap_bp" else v)
            yt.append(float(r["term_bp"]))
            ye.append(float(r["exc_bp"]))
        if len(xs) >= 50:
            w_("%-8s %-6s %-10s %6d %+10.3f %+10.3f" %
               (ven, ent, ftn, len(xs), spearman(xs, yt), spearman(xs, ye)))

# --- options-day confirmation on 3 pre-registered cells
CELLS = [
    ("C1_TUE_TIMEB_2X", "NIFTY", "09:30", 65, "CSL25", {0}),
    ("C2_WED_STIMEB_2X", "SENSEX", "10:30", 65, "CSL20", {1}),
    ("C3_ATM2_2X_T60", "NIFTY", "09:16", 60, "RUP2500", {0, 1, 2}),
]
NDRAW = 2000
w_("")
w_("B. OPTIONS-DAY CONFIRMATION - tercile skip rules vs random-skip null")
w_("   (thresholds = long-sample live-DTE terciles; rule wins if kept-mean beats")
w_("    >=95% of equal-frequency random skips AND retains >=100% of total P&L)")
winners = 0
tested = 0
for cname, ven, ent, T, arm, dset in CELLS:
    v = [(int(x[0]), x[2]) for x in g[(ven, ent, T, arm)] if x[1] in dset]
    days = [d for _, d in v]
    nets = {d: n for n, d in v}
    # attach features (+placebos)
    fv = {}
    for d in days:
        f = feat_of(ven, d)
        if not f:
            continue
        fv[d] = dict(gap_bp=f["gap_bp"], abs_gap_bp=abs(f["gap_bp"]),
                     cpr_t_bp=f["cpr_t_bp"], cpr_y_bp=f["cpr_y_bp"],
                     cpr_w_bp=f["cpr_w_bp"],
                     placebo_noise=random.gauss(0, 1),
                     placebo_dom=int(d[8:10]) % 2)
    usable = sorted(fv)
    tot = sum(nets[d] for d in usable)
    w_("")
    w_("-- %s (%s %s T=%d %s DTE%s): n=%d usable=%d total=%+d mean=%+.0f" %
       (cname, ven, ent, T, arm, sorted(dset), len(days), len(usable), tot,
        st.mean([nets[d] for d in usable]) if usable else 0))
    # long-sample tercile thresholds
    longrows = [r for r in B if r["venue"] == ven and r["entry"] == ent and r["T"] == "60"
                and r["dte_trd"] != "" and int(r["dte_trd"]) in dset]
    thr = {}
    for ftn in FEATURES:
        base = "gap_bp" if ftn == "abs_gap_bp" else ftn
        xs = []
        for r in longrows:
            f = LONGF[ven].get(r["day"])
            if f and f.get(base) is not None:
                xs.append(abs(f[base]) if ftn == "abs_gap_bp" else f[base])
        if xs:
            thr[ftn] = (q(xs, 1 / 3), q(xs, 2 / 3))
    rules = []
    for ftn in FEATURES:
        if ftn not in thr or any(fv[d].get(ftn) is None for d in usable):
            continue
        lo, hi = thr[ftn]
        rules.append(("skip_hi_" + ftn, [d for d in usable if fv[d][ftn] >= hi]))
        rules.append(("skip_lo_" + ftn, [d for d in usable if fv[d][ftn] <= lo]))
    rules.append(("skip_gapup", [d for d in usable if fv[d]["gap_bp"] > 0]))
    rules.append(("skip_gapdn", [d for d in usable if fv[d]["gap_bp"] < 0]))
    # placebos (terciles on the sample itself)
    pn = sorted(fv[d]["placebo_noise"] for d in usable)
    rules.append(("placebo_noise_hi", [d for d in usable
                                       if fv[d]["placebo_noise"] >= q(pn, 2 / 3)]))
    rules.append(("placebo_dom_odd", [d for d in usable if fv[d]["placebo_dom"] == 1]))
    for rname, skip in rules:
        k = len(skip)
        keep = [d for d in usable if d not in set(skip)]
        if k == 0 or len(keep) < 5:
            continue
        kept_mean = st.mean([nets[d] for d in keep])
        kept_tot = sum(nets[d] for d in keep)
        # null
        beat = 0
        for _ in range(NDRAW):
            rs = random.sample(usable, k)
            km = st.mean([nets[d] for d in usable if d not in set(rs)])
            if kept_mean > km:
                beat += 1
        pct = 100.0 * beat / NDRAW
        is_p = rname.startswith("placebo")
        if not is_p:
            tested += 1
        win = pct >= 95 and (kept_tot >= tot if tot > 0 else kept_tot > tot)
        if win and not is_p:
            winners += 1
        w_("   %-22s skip=%2d keptmean=%+7.0f kepttot=%+8.0f beats_null=%5.1f%% %s%s" %
           (rname, k, kept_mean, kept_tot, pct,
            "WIN" if win else "", " [PLACEBO]" if is_p else ""))
    # monotonicity of terciles (options days) for the record
    for ftn in FEATURES:
        if ftn not in thr or any(fv[d].get(ftn) is None for d in usable):
            continue
        lo, hi = thr[ftn]
        t1 = [nets[d] for d in usable if fv[d][ftn] <= lo]
        t2 = [nets[d] for d in usable if lo < fv[d][ftn] < hi]
        t3 = [nets[d] for d in usable if fv[d][ftn] >= hi]
        if min(len(t1), len(t3)) >= 3:
            m1 = st.mean(t1)
            m2 = st.mean(t2) if len(t2) >= 1 else float("nan")
            m3 = st.mean(t3)
            mono = (m1 <= m2 <= m3) or (m1 >= m2 >= m3) if t2 else True
            w_("   tercile-means %-10s: %+7.0f | %+7.0f | %+7.0f  %s" %
               (ftn, m1, m2, m3, "monotone" if mono else "NOT monotone"))
w_("")
w_("SUMMARY: %d non-placebo rules tested, %d winners; ~%.1f expected by chance at 5%%."
   % (tested, winners, 0.05 * tested))
rep.close()

# ---------------- margin + null alternative ----------------
NM, SM, CAP = 1.65, 2.04, 44.7  # lakh per lot / capital
BOOK_LOTS = {  # weekday -> [(venue, lots, when)]
    "Mon": [("NIFTY", 12, "09:16")],
    "Tue": [("NIFTY", 12, "09:16"), ("NIFTY", 8, "09:30")],
    "Wed": [("SENSEX", 12, "09:16"), ("SENSEX", 8, "10:30")],
    "Thu": [("NIFTY", 5, "09:16"), ("SENSEX", 9, "09:16")],
    "Fri": [("NIFTY", 12, "09:16"), ("NIFTY", 8, "10:00")],
}
with open(os.path.join(RES, "margin_null.txt"), "w") as f:
    f.write("Margin arithmetic (NIFTY %.2fL/lot, SENSEX %.2fL/lot, capital %.1fL)\n"
            "Assumed lots: NAS sleeves 3 each (ATM/ATM2/ATM4/COMB), COMB Thu 5, TimeB 8,\n"
            "CSL30F_WED 3. Doubling = +1x of every morning entry, concurrent.\n\n" % (NM, SM, CAP))
    for wd, items in BOOK_LOTS.items():
        base = sum(l * (NM if v == "NIFTY" else SM) for v, l, _ in items)
        f.write("%s: base peak %5.1fL  doubled %5.1fL  vs capital %.1fL -> %s\n" %
                (wd, base, 2 * base, CAP, "FUNDABLE" if 2 * base <= CAP else "NOT FUNDABLE"))
    f.write("\nNull alternative: +1 lot on TUE TimeB (09:30-11:00 SL25) earns mean +Rs755/lot\n"
            "per Tuesday (r/122 atlas, n=16) at 1.65L margin held ~90 min, R:R@p95 1:1.5.\n"
            "Best scalp candidate (same day, same margin, 65 min) earns mean +Rs515/lot\n"
            "(t=2.07, n=16) - strictly dominated by the existing cell.\n")
print("margin_null.txt + all outputs written")
