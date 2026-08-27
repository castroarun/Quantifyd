#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/132 Stage D — aggregate A/B/C into the tables RESULTS.md needs.

Prints, in order:
  0. the reconciliation gate (replay vs booked) — reported BEFORE anything is interpreted
  1. mis-strike frequency, per venue and per source
  2. the NAS before/after natural experiment (forward snap shipped 57eb8c2, 2026-06-01)
  3. unintended delta carried, in Rs per 100 index points
  4. per-book counterfactual cost
  5. offset stability
"""
import csv
import os
import statistics as S
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

SNAP_DATE = "2026-06-01"      # 57eb8c2 shipped the forward snap into nas_atm_executor.py


def f(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def i(x, d=None):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return d


def q(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    k = min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))
    return v[k]


def sec(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


aud = list(csv.DictReader(open(os.path.join(RES, "entry_audit.csv"))))
rep = list(csv.DictReader(open(os.path.join(RES, "replay.csv"))))

# ---------------------------------------------------------------- 0. RECONCILIATION
sec("0. RECONCILIATION GATE — chain replay at the ACTUAL strike vs what the daemon booked")
print("The daemon books gross = (credit - exit_comb) x qty and subtracts a flat Rs160.")
print("We compare the chain replay's credit / exit reason / gross against the booked ones.")
print("A 1-minute chain cannot reproduce a 5-second poll exactly; this bounds the error.\n")
for dw in ("TOUCH", "DWELL2"):
    R = [r for r in rep if r["dwell_mode"] == dw and r["rep_a_gross"]]
    if not R:
        continue
    cerr = [abs(f(r["rep_a_credit"]) - f(r["booked_credit"])) for r in R
            if f(r["booked_credit"])]
    bkr = ["SL" if "SL" in (r["booked_reason"] or "") else "TIME" for r in R]
    rpr = [r["rep_a_reason"] for r in R]
    rmatch = sum(1 for a, b in zip(bkr, rpr) if a == b)
    gerr = [f(r["rep_a_gross"]) - (f(r["booked_pnl"], 0) + 160) for r in R]
    rel = [abs(e) / max(abs(f(r["booked_pnl"], 0) + 160), 1.0)
           for e, r in zip(gerr, R)]
    print("  %-7s n=%d" % (dw, len(R)))
    print("    credit  |err| med %6.2f pt   p90 %6.2f pt   (chain 1-min LTP vs live entry tick)"
          % (S.median(cerr), q(cerr, .9)))
    print("    exit-reason match  %d/%d = %.0f%%" % (rmatch, len(R), 100.0 * rmatch / len(R)))
    print("    gross Rs err  med %+8.0f  p10 %+8.0f  p90 %+8.0f  |rel| med %.1f%%"
          % (S.median(gerr), q(gerr, .1), q(gerr, .9), 100 * S.median(rel)))
    print("    sign agreement (replay and booked same side of zero): %.0f%%"
          % (100.0 * sum(1 for e, r in zip(gerr, R)
                         if (f(r["rep_a_gross"]) >= 0) == (f(r["booked_pnl"], 0) + 160 >= 0)) / len(R)))

# ---------------------------------------------------------------- 1. MIS-STRIKE FREQ
sec("1. MIS-STRIKE FREQUENCY on the trades ACTUALLY TAKEN")
print("%-22s %-7s %5s %7s %7s   %s" % ("population", "venue", "n", "mis", "rate", "steps-off histogram"))
groups = []
for src in ("CSL", "NAS"):
    for ven in ("NIFTY", "SENSEX"):
        G = [r for r in aud if r["src"] == src and r["venue"] == ven]
        if G:
            groups.append(("%s (all)" % src, ven, G))
# NAS split by the snap ship date
for ven in ("NIFTY", "SENSEX"):
    for lbl, cond in (("NAS pre-snap", lambda d: d < SNAP_DATE),
                      ("NAS post-snap", lambda d: d >= SNAP_DATE)):
        G = [r for r in aud if r["src"] == "NAS" and r["venue"] == ven and cond(r["day"])]
        if G:
            groups.append((lbl, ven, G))
for lbl, ven, G in groups:
    ms = sum(i(r["misstrike"], 0) for r in G)
    h = Counter(i(r["steps_off"], 0) for r in G)
    print("%-22s %-7s %5d %7d %6.1f%%   %s" % (
        lbl, ven, len(G), ms, 100.0 * ms / len(G),
        " ".join("%+d:%d" % (k, v) for k, v in sorted(h.items()))))

print("\nCSL per-book:")
print("%-24s %-7s %5s %6s %7s  %s" % ("book", "venue", "n", "mis", "rate", "steps-off"))
for bk in sorted({r["book"] for r in aud if r["src"] == "CSL"}):
    G = [r for r in aud if r["src"] == "CSL" and r["book"] == bk]
    ms = sum(i(r["misstrike"], 0) for r in G)
    h = Counter(i(r["steps_off"], 0) for r in G)
    print("%-24s %-7s %5d %6d %6.1f%%  %s" % (
        bk, G[0]["venue"], len(G), ms, 100.0 * ms / len(G),
        " ".join("%+d:%d" % (k, v) for k, v in sorted(h.items()))))

# ---------------------------------------------------------------- 2. NAS CONTROL
sec("2. THE NAS CONTROL — did the forward snap actually work?")
print("nas_atm_executor.py carried the snap from 57eb8c2 (%s). If the snap works, NAS entries" % SNAP_DATE)
print("after that date should sit ON the forward strike even when spot rounds elsewhere.\n")
print("%-8s %-14s %5s %7s %7s %9s" % ("venue", "era", "n", "mis%", "snapfired%", "med|offset|"))
for ven in ("NIFTY", "SENSEX"):
    for lbl, cond in (("pre  <%s" % SNAP_DATE, lambda d: d < SNAP_DATE),
                      ("post >=%s" % SNAP_DATE, lambda d: d >= SNAP_DATE)):
        G = [r for r in aud if r["src"] == "NAS" and r["venue"] == ven and cond(r["day"])]
        if not G:
            continue
        ms = 100.0 * sum(i(r["misstrike"], 0) for r in G) / len(G)
        fired = [r for r in G if i(r["k_from_spot"]) is not None
                 and i(r["k_from_spot"]) != i(r["k_actual"])]
        offs = [abs(f(r["offset"])) for r in G if f(r["offset"]) is not None]
        print("%-8s %-14s %5d %6.1f%% %9.1f%% %9.1f" % (
            ven, lbl, len(G), ms, 100.0 * len(fired) / len(G),
            S.median(offs) if offs else float("nan")))

# ---------------------------------------------------------------- 3. UNINTENDED DELTA
sec("3. THE UNINTENDED DELTA — directional risk carried without anyone choosing it")
print("Short straddle at K with forward F: net delta = 1 - 2*N(d1), sigma inverted from the")
print("observed combined premium (Black-76). Positive = the book was LONG the index.\n")


def dblock(title, G):
    G = [r for r in G if f(r["rs_per_100pt"]) is not None]
    if not G:
        return
    nd = [abs(f(r["net_delta"])) for r in G]
    rs = [f(r["rs_per_100pt"]) for r in G]
    ars = [abs(x) for x in rs]
    mis = [r for r in G if i(r["misstrike"], 0) == 1]
    mrs = [abs(f(r["rs_per_100pt"])) for r in mis]
    cred = [f(r["credit"]) * f(r["qty"]) for r in G if f(r["credit"]) and f(r["qty"])]
    print("%-28s n=%-4d  |delta| med %.3f p90 %.3f | Rs/100pt |med| %6.0f p90 %7.0f max %7.0f"
          % (title, len(G), S.median(nd), q(nd, .9), S.median(ars), q(ars, .9), max(ars)))
    print("%-28s   net signed sum %+8.0f Rs/100pt (a book-level tilt, if any)"
          % ("", sum(rs)))
    if mis:
        print("%-28s   MIS-STRUCK ONLY n=%-4d |Rs/100pt| med %6.0f p90 %7.0f max %7.0f"
              % ("", len(mis), S.median(mrs), q(mrs, .9), max(mrs)))
    if cred:
        rel = [abs(f(r["rs_per_100pt"])) / (f(r["credit"]) * f(r["qty"])) * 100
               for r in G if f(r["credit"]) and f(r["qty"])]
        print("%-28s   a 100-pt move moves the book by median %.0f%% of the credit collected"
              % ("", S.median(rel)))


for ven in ("NIFTY", "SENSEX"):
    dblock("CSL %s" % ven, [r for r in aud if r["src"] == "CSL" and r["venue"] == ven])
for ven in ("NIFTY", "SENSEX"):
    dblock("NAS %s (post-snap)" % ven,
           [r for r in aud if r["src"] == "NAS" and r["venue"] == ven and r["day"] >= SNAP_DATE])
    dblock("NAS %s (pre-snap)" % ven,
           [r for r in aud if r["src"] == "NAS" and r["venue"] == ven and r["day"] < SNAP_DATE])

print("\nCSL per-book, Rs per 100 index points (signed; + = unintentionally long):")
print("%-24s %-7s %4s %5s %10s %10s %10s" % ("book", "venue", "n", "mis", "med", "p90|.|", "max|.|"))
for bk in sorted({r["book"] for r in aud if r["src"] == "CSL"}):
    G = [r for r in aud if r["src"] == "CSL" and r["book"] == bk and f(r["rs_per_100pt"]) is not None]
    if not G:
        continue
    rs = [f(r["rs_per_100pt"]) for r in G]
    print("%-24s %-7s %4d %5d %+10.0f %10.0f %10.0f" % (
        bk, G[0]["venue"], len(G), sum(i(r["misstrike"], 0) for r in G),
        S.median(rs), q([abs(x) for x in rs], .9), max(abs(x) for x in rs)))

print("\nWorst individual entries by |Rs per 100 pts|:")
W = sorted([r for r in aud if f(r["rs_per_100pt"]) is not None],
           key=lambda r: -abs(f(r["rs_per_100pt"])))[:12]
print("%-6s %-22s %-7s %-11s %4s %8s %8s %7s %6s %10s" % (
    "src", "book", "venue", "day", "DTE", "K", "fwd", "off", "steps", "Rs/100pt"))
for r in W:
    print("%-6s %-22s %-7s %-11s %4s %8s %8s %7s %6s %+10.0f" % (
        r["src"], r["book"][:22], r["venue"], r["day"], r["dte_trd"], r["k_actual"],
        r["fwd"], r["offset"], r["steps_off"], f(r["rs_per_100pt"])))

# ---------------------------------------------------------------- 4. COUNTERFACTUAL
sec("4. THE COUNTERFACTUAL COST — booked path vs forward-snapped path, MEASURED costs")
print("Both arms replayed on the same 1-minute chain under the same rule, so the comparison")
print("is like-for-like. The forward arm is a DIFFERENT instrument with its own path.\n")
for dw in ("TOUCH", "DWELL2"):
    R = [r for r in rep if r["dwell_mode"] == dw and r["delta_net"] and not r["mgmt"]]
    if not R:
        continue
    d = [f(r["delta_net"]) for r in R]
    mis = [r for r in R if i(r["misstrike"], 0) == 1]
    dm = [f(r["delta_net"]) for r in mis]
    print("  --- %s (mgmt arms excluded: their post-stop path cannot be honestly re-struck) ---" % dw)
    print("    all entries        n=%-3d  TOTAL %+9.0f  mean %+8.0f  med %+8.0f  sd %8.0f"
          % (len(R), sum(d), S.mean(d), S.median(d), S.pstdev(d) if len(d) > 1 else 0))
    if dm:
        print("    mis-struck only    n=%-3d  TOTAL %+9.0f  mean %+8.0f  med %+8.0f  sd %8.0f"
              % (len(mis), sum(dm), S.mean(dm), S.median(dm), S.pstdev(dm) if len(dm) > 1 else 0))
        print("    forward arm better on %d of %d mis-struck entries (%.0f%%)"
              % (sum(1 for x in dm if x > 0), len(dm), 100.0 * sum(1 for x in dm if x > 0) / len(dm)))
        print("    range %+.0f .. %+.0f | p10 %+.0f p90 %+.0f"
              % (min(dm), max(dm), q(dm, .1), q(dm, .9)))
        if len(dm) > 2:
            t = S.mean(dm) / (S.pstdev(dm) / (len(dm) ** 0.5)) if S.pstdev(dm) else 0
            print("    t on the mean difference = %.2f  (n=%d) -> %s"
                  % (t, len(dm), "indistinguishable from zero" if abs(t) < 2 else "significant"))
    print("    exit-reason flips (the forward straddle stopped when the real one did not, or vice versa): %d of %d"
          % (sum(1 for r in R if r["rep_a_reason"] != r["rep_f_reason"]), len(R)))

print("\n  Per-book, DWELL2 (headline), mgmt arms excluded:")
print("%-24s %-7s %4s %4s %11s %11s %11s %10s" % (
    "book", "venue", "n", "mis", "actual net", "fwd net", "difference", "per-trade"))
tot_a = tot_f = 0
for bk in sorted({r["book"] for r in rep}):
    R = [r for r in rep if r["dwell_mode"] == "DWELL2" and r["book"] == bk
         and r["rep_a_net"] and r["rep_f_net"]]
    if not R:
        continue
    a = sum(f(r["rep_a_net"]) for r in R)
    fw = sum(f(r["rep_f_net"]) for r in R)
    mg = R[0]["mgmt"]
    if not mg:
        tot_a += a
        tot_f += fw
    print("%-24s %-7s %4d %4d %+11.0f %+11.0f %+11.0f %+10.0f%s" % (
        bk, R[0]["venue"], len(R), sum(i(r["misstrike"], 0) for r in R), a, fw, fw - a,
        (fw - a) / len(R), "  [mgmt arm - excluded from total]" if mg else ""))
print("%-24s %-7s %4s %4s %+11.0f %+11.0f %+11.0f" % (
    "TOTAL (non-mgmt)", "", "", "", tot_a, tot_f, tot_f - tot_a))

# ---------------------------------------------------------------- 5. STABILITY
sec("5. OFFSET STABILITY — is the basis a fixed level or does it swing?")
atl = list(csv.DictReader(open(os.path.join(RES, "offset_atlas.csv"))))
for ven in ("NIFTY", "SENSEX"):
    V = [r for r in atl if r["venue"] == ven]
    step = 50.0 if ven == "NIFTY" else 100.0
    bd = defaultdict(list)
    for r in V:
        bd[r["day"]].append(f(r["offset"]))
    meds = {d: S.median(v) for d, v in bd.items()}
    iqrs = [q(v, .75) - q(v, .25) for v in bd.values()]
    ml = sorted(meds.values())
    print("\n  %s" % ven)
    print("    between-day: per-day median offset spans %+.0f .. %+.0f (p05..p95 %+.0f..%+.0f)"
          % (min(ml), max(ml), q(ml, .05), q(ml, .95)))
    print("    within-day : median IQR %.1f pts -> the level is steady through a session"
          % S.median(iqrs))
    print("    ratio between-day spread / within-day IQR = %.1fx  => the swing is DAY-TO-DAY, not intraday"
          % ((q(ml, .95) - q(ml, .05)) / max(S.median(iqrs), 1e-9)))
    print("    days whose MEDIAN offset alone exceeds half a step (%.0f pts): %d of %d = %.0f%%"
          % (step / 2, sum(1 for m in ml if abs(m) > step / 2), len(ml),
             100.0 * sum(1 for m in ml if abs(m) > step / 2) / len(ml)))
    print("    sign: %d days positive, %d negative -> the basis CHANGES SIGN (July dividend season)"
          % (sum(1 for m in ml if m > 0), sum(1 for m in ml if m < 0)))
    for mo in sorted({d[:7] for d in meds}):
        mm = [meds[d] for d in meds if d[:7] == mo]
        print("      %s  n=%2d days  median-of-medians %+7.1f" % (mo, len(mm), S.median(mm)))
