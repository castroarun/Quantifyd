#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/131 - gates on the leg-stop sweep. Reads results/leg_stop_detail.csv.

Emits: leg_stop_summary.csv, r114_reconciliation.csv, and a full text report to
results/analysis.txt (also stdout) carrying the per-arm table, fire rates, plateau map,
family-wise permutation test, OOS split, worst-day comparison and the venue interaction.
"""
import csv, os, math, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DETAIL = os.path.join(RES, "leg_stop_detail.csv")
SUMMARY = os.path.join(RES, "leg_stop_summary.csv")
RECON = os.path.join(RES, "r114_reconciliation.csv")
REPORT = os.path.join(RES, "analysis.txt")

LEVEL_ORDER = ["HOLD", "LEG30", "LEG40", "LEG50", "LEG60", "LEG75", "LEG100",
               "RUP1500", "RUP2500", "RUP4000", "RUP6000", "RUP8000"]
SURVS = ["SBOTH", "SHOLD", "STRAIL"]
BUF = []


def P(s=""):
    BUF.append(s)
    print(s, flush=True)


def key(r):
    return (r["entryset"], r["level"], r["surv"], r["outer"], r["costmodel"])


def main():
    rows = list(csv.DictReader(open(DETAIL)))
    for r in rows:
        r["net"] = float(r["net"])
        r["gross"] = float(r["gross"])
        r["cost"] = float(r["cost"])
        r["leg_fired"] = int(r["leg_fired"])
    cells = {}
    for r in rows:
        cells.setdefault(key(r), []).append(r)

    # ---------------- summary csv ----------------
    srows = []
    for k, v in sorted(cells.items()):
        v = sorted(v, key=lambda x: x["day"])
        nets = [x["net"] for x in v]
        fired = [x["leg_fired"] for x in v]
        ft = [x["fire_hm"] for x in v if x["fire_hm"]]
        srows.append(dict(entryset=k[0], level=k[1], surv=k[2], outer=k[3], costmodel=k[4],
                          n=len(nets), total=round(sum(nets)), mean=round(st.mean(nets)),
                          median=round(st.median(nets)),
                          win_pct=round(100 * sum(1 for x in nets if x > 0) / len(nets)),
                          worst=round(min(nets)), best=round(max(nets)),
                          fire_pct=round(100 * sum(fired) / len(fired)),
                          med_fire_hm=sorted(ft)[len(ft) // 2] if ft else "",
                          mean_cost=round(st.mean([x["cost"] for x in v]))))
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
        w.writeheader()
        for r in srows:
            w.writerow(r)

    S = {(r["entryset"], r["level"], r["surv"], r["outer"], r["costmodel"]): r for r in srows}

    # ---------------- r/114 reconciliation ----------------
    P("=" * 100)
    P("R/114 RECONCILIATION - LEG30 (SBOTH, standalone), published: mean -227/lot, 25% win, n=12")
    P("=" * 100)
    P("%-9s %-9s %-9s  %3s %8s %8s %6s %9s %9s" %
      ("entryset", "cost", "arm", "n", "total", "mean", "win%", "worst", "best"))
    rec = []
    for es in ("B0916", "A0920"):
        for cm in ("R114COST", "MEASURED"):
            for lv, sv in (("HOLD", "-"), ("LEG30", "SBOTH")):
                r = S.get((es, lv, sv, "STANDALONE", cm))
                if not r:
                    continue
                rec.append(r)
                P("%-9s %-9s %-9s  %3d %8d %8d %6d %9d %9d" %
                  (es, cm, lv, r["n"], r["total"], r["mean"], r["win_pct"], r["worst"], r["best"]))
    with open(RECON, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec[0].keys()))
        w.writeheader()
        for r in rec:
            w.writerow(r)
    P()

    # ---------------- per-arm tables + gates ----------------
    for es in ("A0920", "B0916"):
        for outer in ("STANDALONE", "VENUE"):
            fam = [(lv, sv) for lv in LEVEL_ORDER[1:] for sv in SURVS]
            hold = cells.get((es, "HOLD", "-", outer, "MEASURED"))
            if not hold:
                continue
            hold = sorted(hold, key=lambda x: x["day"])
            days = [x["day"] for x in hold]
            hnet = np.array([x["net"] for x in hold])
            n = len(days)
            P("=" * 118)
            P("ENTRY SET %s | OUTER LAYER %s | n=%d sessions (%s .. %s) | net Rs per LOT (lot=20)"
              % (es, outer, n, days[0], days[-1]))
            P("=" * 118)
            P("%-9s %-7s %4s %8s %8s %8s %6s %9s %9s %6s %7s %7s %7s" %
              ("level", "surv", "n", "total", "mean", "median", "win%", "worst", "best",
               "fire%", "medfire", "dMEAN", "t"))
            hr = S[(es, "HOLD", "-", outer, "MEASURED")]
            P("%-9s %-7s %4d %8d %8d %8d %6d %9d %9d %6d %7s %7s %7s" %
              ("HOLD", "-", hr["n"], hr["total"], hr["mean"], hr["median"], hr["win_pct"],
               hr["worst"], hr["best"], 0, "-", "-", "-"))
            diffs, tstats, labels = [], [], []
            for lv, sv in fam:
                v = cells.get((es, lv, sv, outer, "MEASURED"))
                if not v:
                    continue
                v = sorted(v, key=lambda x: x["day"])
                r = S[(es, lv, sv, outer, "MEASURED")]
                d = np.array([x["net"] for x in v]) - hnet
                sd = d.std(ddof=1)
                t = d.mean() / (sd / math.sqrt(n)) if sd > 1e-9 else 0.0
                diffs.append(d)
                tstats.append(t)
                labels.append((lv, sv))
                P("%-9s %-7s %4d %8d %8d %8d %6d %9d %9d %6d %7s %7d %7.2f" %
                  (lv, sv, r["n"], r["total"], r["mean"], r["median"], r["win_pct"],
                   r["worst"], r["best"], r["fire_pct"], r["med_fire_hm"] or "-",
                   round(d.mean()), t))

            # family-wise sign-flip permutation on max|t|
            D = np.vstack(diffs)
            rng = np.random.default_rng(20260827)
            NPERM = 10000
            maxt = np.empty(NPERM)
            for i in range(NPERM):
                s = rng.choice([-1.0, 1.0], size=n)
                Dp = D * s
                m = Dp.mean(axis=1)
                sd = Dp.std(axis=1, ddof=1)
                tt = np.where(sd > 1e-9, m / (sd / math.sqrt(n)), 0.0)
                maxt[i] = np.abs(tt).max()
            P()
            P("FAMILY-WISE (sign-flip permutation, %d draws, max|t| over the %d stop arms):"
              % (NPERM, len(labels)))
            best = sorted(zip(tstats, labels, [d.mean() for d in diffs]), reverse=True)[:5]
            for t, (lv, sv), dm in best:
                pw = float((maxt >= abs(t)).mean())
                P("   %-9s %-7s  dMEAN %+7d  t %+6.2f  family-wise p = %.3f%s"
                  % (lv, sv, round(dm), t, pw, "   <-- PASSES" if pw < 0.05 else ""))
            P("   max|t| null: p95 = %.2f, p99 = %.2f" % (np.percentile(maxt, 95),
                                                          np.percentile(maxt, 99)))

            # OOS split
            h = n // 2
            P()
            P("OOS SPLIT (first %d sessions vs last %d): dMEAN vs HOLD, both halves" % (h, n - h))
            P("   %-9s %-7s %10s %10s %10s" % ("level", "surv", "IS_dMEAN", "OOS_dMEAN", "sign_ok"))
            _ord = sorted(range(len(labels)), key=lambda i: -diffs[i].mean())
            for i in _ord:
                lv, sv = labels[i]; d = diffs[i]
                a, b = d[:h].mean(), d[h:].mean()
                P("   %-9s %-7s %10d %10d %10s"
                  % (lv, sv, round(a), round(b), "YES" if (a > 0 and b > 0) else "no"))
            P("   (all %d arms listed, best dMEAN first)" % len(labels))
            P()

            # worst-day table
            wd = days[int(np.argmin(hnet))]
            for tgt in sorted({"2026-06-11", wd}):
                if tgt not in days:
                    continue
                i = days.index(tgt)
                P("WORST-DAY %s (HOLD net %+d/lot)%s:"
                  % (tgt, round(hnet[i]), "  <- study's worst HOLD day" if tgt == wd else ""))
                line = []
                for lv in LEVEL_ORDER[1:]:
                    for sv in SURVS:
                        v = cells.get((es, lv, sv, outer, "MEASURED"))
                        if not v:
                            continue
                        v = sorted(v, key=lambda x: x["day"])
                        line.append("%s/%s %+d" % (lv, sv, round(v[i]["net"])))
                for j in range(0, len(line), 3):
                    P("   " + " | ".join("%-22s" % x for x in line[j:j + 3]))
                P()

    # ---------------- venue interaction ----------------
    P("=" * 100)
    P("VENUE INTERACTION - does the leg stop ever fire BEFORE the book stop/TP already has?")
    P("=" * 100)
    P("%-9s %-9s %-7s %14s %14s" % ("entryset", "level", "surv", "fire% STANDALONE", "fire% VENUE"))
    for es in ("A0920", "B0916"):
        for lv in LEVEL_ORDER[1:]:
            for sv in SURVS:
                a = S.get((es, lv, sv, "STANDALONE", "MEASURED"))
                b = S.get((es, lv, sv, "VENUE", "MEASURED"))
                if a and b:
                    P("%-9s %-9s %-7s %14d %14d" % (es, lv, sv, a["fire_pct"], b["fire_pct"]))
    P()
    for es in ("A0920", "B0916"):
        v = cells.get((es, "HOLD", "-", "VENUE", "MEASURED"))
        if not v:
            continue
        rs = [x["ce_reason"] for x in v] + [x["pe_reason"] for x in v]
        P("HOLD/%s VENUE exit reasons: %s" % (es, {x: rs.count(x) for x in set(rs)}))

    with open(REPORT, "w") as f:
        f.write("\n".join(BUF) + "\n")


if __name__ == "__main__":
    main()
