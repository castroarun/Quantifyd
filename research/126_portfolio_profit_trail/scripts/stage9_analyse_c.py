#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/126 Stage 9 - ARM C ANALYSIS: does spreading the strikes beat all three nulls?

Nulls, in order of how hard they are to beat:
  (a) ALL_ATM              - the deployed shape
  (b) RANDOM-LEG PLACEBO   - 500 random offset triples from the same menu
  (c) DOWNSIZED ATM        - the trivial alternative. If a spread cuts the tail, so does
                             cutting size. Compared AT EQUAL WORST-DAY: scale ALL_ATM by
                             f = worst(P)/worst(ATM) and ask which keeps more mean.
                             If (c) wins, Arm C is downsizing wearing a costume.

Mechanism test: if mean(P)/mean(ATM) ~= credit(P)/credit(ATM) ~= f, the "tail improvement"
is just less premium at risk, and must be labelled as such.

Reads results/armc_cells.csv. Writes results/armc_summary.txt, results/armc_grid.csv
"""
import csv
import os
import random
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
REP = []
SYS = ["HOLD", "COMB", "RUPEE2500"]
MARGIN = {"NIFTY": 165000.0, "SENSEX": 204000.0}
LOTS = 2
CAPITAL = 4470000.0


def log(m):
    REP.append(str(m))
    print(m, flush=True)


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def tstat(xs):
    if len(xs) < 3:
        return 0.0
    sd = st.pstdev(xs)
    return st.mean(xs) / (sd / len(xs) ** 0.5) if sd > 0 else 0.0


# ------------------------------------------------------------------ load
cell = defaultdict(dict)      # (venue, day) -> (system, offset) -> row
meta = {}
for r in csv.DictReader(open(os.path.join(RES, "armc_cells.csv"))):
    cell[(r["venue"], r["day"])][(r["system"], int(r["offset"]))] = r
    meta[(r["venue"], r["day"])] = (r["weekday"], int(r["dte"]))
log("armc cells: %d venue-days" % len(cell))

OFFS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

# ============================================================ 1. INTERACTION
log("")
log("=" * 104)
log("1. THE INTERACTION THAT COULD NOT BE ASSUMED AWAY - exit rules were calibrated AT the money")
log("=" * 104)
for ven in ("NIFTY", "SENSEX"):
    days = sorted(d for (v, d) in cell if v == ven)
    log("")
    log("--- %s (n=%d days) ---" % (ven, len(days)))
    log("%-9s %6s %9s %9s %8s %8s %9s %10s" % (
        "system", "offset", "credit", "net_total", "mean", "win%", "stop_rate", "worst"))
    for sysname in SYS:
        for off in OFFS:
            rows = [cell[(ven, d)][(sysname, off)] for d in days
                    if (sysname, off) in cell[(ven, d)]]
            if len(rows) < 20:
                continue
            nets = [float(x["net_rs"]) for x in rows]
            cred = [float(x["credit"]) for x in rows]
            fires = [int(x["fires"]) for x in rows]
            log("%-9s %6d %9.1f %9d %8d %8.1f %9.1f %10d" % (
                sysname, off, st.mean(cred), sum(nets), st.mean(nets),
                100.0 * sum(1 for x in nets if x > 0) / len(nets),
                100.0 * sum(1 for x in fires if x > 0) / len(rows), min(nets)))
log("")
log("READ: SYS_ATM2's stop is a FIXED Rs2,500/lot. Watch its stop_rate as |offset| grows -")
log("      the same rupee stop is a larger %-of-credit move off-ATM, so it should fire LESS.")

# ============================================================ 2. PORTFOLIOS
PORT = {"ALL_ATM": (0, 0, 0)}
for k in (1, 2, 3, 4):
    PORT["SYM_%d" % k] = (-k, 0, k)
    PORT["ALLUP_%d" % k] = (k, k, k)
    PORT["ALLDOWN_%d" % k] = (-k, -k, -k)
    PORT["LADDER_UP_%d" % k] = (0, k, 2 * k)
    PORT["LADDER_DN_%d" % k] = (0, -k, -2 * k)


def port_days(ven, legs):
    """-> [(day, net, credit)] for the 3-system portfolio at offsets legs[i]"""
    out = []
    for d in sorted(x for (v, x) in cell if v == ven):
        cc = cell[(ven, d)]
        keys = [(SYS[i], legs[i]) for i in range(3)]
        if not all(k in cc for k in keys):
            continue
        out.append((d, sum(float(cc[k]["net_rs"]) for k in keys),
                    sum(float(cc[k]["credit"]) for k in keys)))
    return out


grid = []
log("")
log("=" * 104)
log("2. EQUAL-LOTS PORTFOLIOS (3 systems x 2 lots) - and the three nulls")
log("=" * 104)
results = {}
for ven in ("NIFTY", "SENSEX"):
    base = port_days(ven, PORT["ALL_ATM"])
    bmap = {d: n for d, n, _ in base}
    bworst, bmean, btot = min(n for _, n, _ in base), st.mean([n for _, n, _ in base]), sum(n for _, n, _ in base)
    bcred = st.mean([c for _, _, c in base])
    log("")
    log("--- %s   ALL_ATM null: n=%d total=%d mean=%d worst=%d credit=%.0f ---"
        % (ven, len(base), btot, bmean, bworst, bcred))
    log("%-14s %5s %9s %8s %8s %6s %9s %9s %8s %9s %9s %7s" % (
        "portfolio", "n", "total", "mean", "median", "win%", "worst", "p10", "credit",
        "d_mean", "t", "DOWNSZ"))
    for name, legs in PORT.items():
        pd_ = port_days(ven, legs)
        if len(pd_) < 30:
            continue
        common = [(d, n, c) for d, n, c in pd_ if d in bmap]
        nets = [n for _, n, _ in common]
        creds = [c for _, _, c in common]
        dl = [n - bmap[d] for d, n, _ in common]
        worst = min(nets)
        # null (c): downsize ATM to the SAME worst day, compare mean
        f = worst / bworst if bworst < 0 else 1.0
        dsz_mean = f * bmean
        verdict = "WINS" if dsz_mean >= st.mean(nets) else "loses"
        results[(ven, name)] = dict(nets=nets, days=[d for d, _, _ in common],
                                   worst=worst, mean=st.mean(nets), credit=st.mean(creds),
                                   dl=dl, f=f, dsz_mean=dsz_mean)
        grid.append(dict(venue=ven, portfolio=name, n=len(nets), total=round(sum(nets)),
                         mean=round(st.mean(nets)), median=round(st.median(nets)),
                         win=round(100.0 * sum(1 for x in nets if x > 0) / len(nets), 1),
                         worst=round(worst), p10=round(pct(nets, 10)),
                         credit=round(st.mean(creds)), d_mean=round(st.mean(dl)),
                         t=round(tstat(dl), 2), downsize_f=round(f, 3),
                         downsized_atm_mean=round(dsz_mean), downsize_verdict=verdict))
        log("%-14s %5d %9d %8d %8d %6.1f %9d %9d %8.0f %9d %9.2f %7s" % (
            name, len(nets), sum(nets), st.mean(nets), st.median(nets),
            100.0 * sum(1 for x in nets if x > 0) / len(nets), worst, pct(nets, 10),
            st.mean(creds), st.mean(dl), tstat(dl), verdict))

log("")
log("DOWNSZ column = null (c). 'WINS' means simply trading ALL_ATM smaller achieves the")
log("same worst day while KEEPING MORE MEAN than the strike spread does. If that column is")
log("mostly WINS, Arm C is downsizing in a costume.")

# ============================================================ 3. PLATEAU
log("")
log("=" * 104)
log("3. PLATEAU over offset magnitude (symmetric family) - monotone/broad = signal, spike = noise")
log("=" * 104)
log("%-8s %-10s %9s %9s %9s %9s %9s" % ("venue", "metric", "k=0", "k=1", "k=2", "k=3", "k=4"))
for ven in ("NIFTY", "SENSEX"):
    row_m, row_w, row_c = [], [], []
    for k in (0, 1, 2, 3, 4):
        nm = "ALL_ATM" if k == 0 else "SYM_%d" % k
        r = results.get((ven, nm))
        if not r:
            row_m.append(None); row_w.append(None); row_c.append(None); continue
        row_m.append(r["mean"]); row_w.append(r["worst"]); row_c.append(r["credit"])
    fmt = lambda xs: "".join("%9s" % (round(x) if x is not None else "-") for x in xs)
    log("%-8s %-10s%s" % (ven, "mean", fmt(row_m)))
    log("%-8s %-10s%s" % (ven, "worst", fmt(row_w)))
    log("%-8s %-10s%s" % (ven, "credit", fmt(row_c)))

# ============================================================ 4. SKEW
log("")
log("=" * 104)
log("4. SKEW ASYMMETRY - up-offsets vs down-offsets are NOT mirror images")
log("=" * 104)
log("%-8s %-12s %9s %9s %9s %9s" % ("venue", "family", "mean", "worst", "credit", "t_vs_ATM"))
for ven in ("NIFTY", "SENSEX"):
    for k in (1, 2, 3, 4):
        for fam in ("ALLUP_%d" % k, "ALLDOWN_%d" % k):
            r = results.get((ven, fam))
            if not r:
                continue
            log("%-8s %-12s %9d %9d %9.0f %9.2f" % (
                ven, fam, r["mean"], r["worst"], r["credit"], tstat(r["dl"])))

# ============================================================ 5. PLACEBO
log("")
log("=" * 104)
log("5. NULL (b) - RANDOM-LEG PLACEBO (500 random offset triples from the same menu)")
log("=" * 104)
rnd = random.Random(20260825)
for ven in ("NIFTY", "SENSEX"):
    tots, worsts = [], []
    for _ in range(500):
        legs = tuple(rnd.choice(OFFS) for _ in range(3))
        pd_ = port_days(ven, legs)
        if len(pd_) < 30:
            continue
        nets = [n for _, n, _ in pd_]
        tots.append(sum(nets)); worsts.append(min(nets))
    if not tots:
        continue
    log("")
    log("--- %s ---" % ven)
    log("  placebo total : p05=%d median=%d p95=%d" % (pct(tots, 5), st.median(tots), pct(tots, 95)))
    log("  placebo worst : p05=%d median=%d p95=%d" % (pct(worsts, 5), st.median(worsts), pct(worsts, 95)))
    for nm in ("ALL_ATM", "SYM_1", "SYM_2", "SYM_3", "SYM_4"):
        r = results.get((ven, nm))
        if not r:
            continue
        tot = sum(r["nets"])
        pr_t = 100.0 * sum(1 for x in tots if x < tot) / len(tots)
        pr_w = 100.0 * sum(1 for x in worsts if x < r["worst"]) / len(worsts)
        log("  %-8s total=%8d (pctile %5.1f)   worst=%8d (pctile %5.1f)"
            % (nm, tot, pr_t, r["worst"], pr_w))

# ============================================================ 6. OOS + FWER
log("")
log("=" * 104)
log("6. OUT-OF-SAMPLE SPLIT (fit on the earlier half, confirm on the later) + FAMILY-WISE HAIRCUT")
log("=" * 104)
for ven in ("NIFTY", "SENSEX"):
    base = results.get((ven, "ALL_ATM"))
    if not base:
        continue
    days = base["days"]
    mid = len(days) // 2
    d_is, d_oos = set(days[:mid]), set(days[mid:])
    log("")
    log("--- %s  IS: %s..%s (n=%d)   OOS: %s..%s (n=%d) ---"
        % (ven, days[0], days[mid - 1], mid, days[mid], days[-1], len(days) - mid))
    log("%-14s %10s %10s %10s %10s" % ("portfolio", "IS d_mean", "IS worst", "OOS d_mean", "OOS worst"))
    cands = []
    for nm in PORT:
        r = results.get((ven, nm))
        if not r or nm == "ALL_ATM":
            continue
        dmap = dict(zip(r["days"], r["dl"]))
        nmap = dict(zip(r["days"], r["nets"]))
        is_d = [dmap[d] for d in r["days"] if d in d_is]
        oo_d = [dmap[d] for d in r["days"] if d in d_oos]
        is_n = [nmap[d] for d in r["days"] if d in d_is]
        oo_n = [nmap[d] for d in r["days"] if d in d_oos]
        if len(is_d) < 15 or len(oo_d) < 15:
            continue
        cands.append((nm, st.mean(is_d), min(is_n), st.mean(oo_d), min(oo_n), tstat(r["dl"])))
    cands.sort(key=lambda x: -x[1])
    for nm, a, b, cc, dd, _ in cands[:8]:
        log("%-14s %10d %10d %10d %10d" % (nm, a, b, cc, dd))
    # Westfall-Young style max-t null over the family
    if cands:
        allt = [abs(x[5]) for x in cands]
        obs = max(allt)
        rnd2 = random.Random(7)
        nulls = []
        ref = results[(ven, cands[0][0])]["dl"]
        for _ in range(2000):
            mx = 0.0
            for nm, _a, _b, _c, _d, _t in cands:
                dl = results[(ven, nm)]["dl"]
                fl = [x if rnd2.random() < 0.5 else -x for x in dl]
                mx = max(mx, abs(tstat(fl)))
            nulls.append(mx)
        log("  family size=%d  max|t| observed=%.2f  null-95th=%.2f  -> %s"
            % (len(cands), obs, pct(nulls, 95),
               "SURVIVES" if obs > pct(nulls, 95) else "FAILS the family-wise haircut"))

# ============================================================ 7. MECHANISM + MARGIN
log("")
log("=" * 104)
log("7. MECHANISM - is the tail gain real risk reduction, or just less premium sold?")
log("=" * 104)
log("%-8s %-14s %8s %8s %8s %10s" % ("venue", "portfolio", "cred_r", "mean_r", "worst_r", "reading"))
for ven in ("NIFTY", "SENSEX"):
    b = results.get((ven, "ALL_ATM"))
    if not b:
        continue
    for nm in ("SYM_1", "SYM_2", "SYM_3", "SYM_4"):
        r = results.get((ven, nm))
        if not r:
            continue
        cr = r["credit"] / b["credit"]
        mr = r["mean"] / b["mean"] if b["mean"] else 0
        wr = r["worst"] / b["worst"] if b["worst"] else 0
        reading = "downsizing" if abs(cr - wr) < 0.10 and abs(cr - mr) < 0.15 else "structural"
        log("%-8s %-14s %8.3f %8.3f %8.3f %10s" % (ven, nm, cr, mr, wr, reading))
log("")
log("cred_r/mean_r/worst_r are ratios to ALL_ATM. If all three move together, the spread is")
log("simply selling less premium -> the honest label is DOWNSIZING, not diversification.")
log("")
log("MARGIN (peak concurrent, 3 systems x 2 lots): NIFTY 3x2x1.65L = %.1fL, SENSEX 3x2x2.04L = %.1fL"
    % (3 * LOTS * MARGIN["NIFTY"] / 1e5, 3 * LOTS * MARGIN["SENSEX"] / 1e5))
log("Capital ~%.1fL. Strike spreading does NOT change lot count, so margin is unchanged vs")
log("ALL_ATM (an OTM short is marginally cheaper on SPAN, which only helps). Credit-matching")
log("by RAISING lots would raise margin proportionally - reported but not recommended.")

with open(os.path.join(RES, "armc_grid.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
    w.writeheader()
    for r in grid:
        w.writerow(r)
open(os.path.join(RES, "armc_summary.txt"), "w").write("\n".join(REP) + "\n")
print("\nwrote results/armc_summary.txt, results/armc_grid.csv")
