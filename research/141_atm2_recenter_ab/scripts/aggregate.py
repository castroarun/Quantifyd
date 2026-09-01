#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/141 aggregation — per-arm table, churn-cost decomposition, per-DTE,
plateau, family-wise haircut, OOS split. Reads results/arms_daily.csv only."""
import os
import csv
import math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
CSVP = os.path.join(RES, "arms_daily.csv")

ARM_ORDER = ["ONE_AND_DONE", "RECENTER_1", "RECENTER_2", "RECENTER_3", "RECENTER_5",
             "RECENTER_2_CD15", "RECENTER_3_CD15", "RECENTER_5_CD15",
             "RECENTER_5_NOGUARD", "MOVESTOP_ONE", "MOVESTOP_RECENTER", "MOVESTOP_RC1", "MOVESTOP_RC_CD15", "NOSTOP_HOLD"]
BASE = "ONE_AND_DONE"
IS_END = "2026-07-28"          # research/96's own day set
OUT = []


def p(m=""):
    OUT.append(m)
    print(m, flush=True)


def load():
    rows = []
    with open(CSVP) as f:
        for r in csv.DictReader(f):
            for k in ("gross", "cost", "net", "gross_c1", "cost_c1", "net_c1",
                      "gross_after", "cost_after", "net_after", "credit0"):
                r[k] = float(r[k])
            for k in ("n_cycles", "n_recenters", "stop_fired", "dte_trd", "dte_cal"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def med(a):
    b = sorted(a)
    n = len(b)
    if not n:
        return 0.0
    return b[n // 2] if n % 2 else 0.5 * (b[n // 2 - 1] + b[n // 2])


def pct(a, q):
    b = sorted(a)
    if not b:
        return 0.0
    i = max(0, min(len(b) - 1, int(round(q / 100.0 * (len(b) - 1)))))
    return b[i]


def norm_p(t):
    """two-sided p from the normal approximation"""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))


def paired(a, b):
    """paired diff a-b -> (mean, t, p, n)"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 3:
        return 0.0, 0.0, 1.0, n
    m = sum(d) / n
    v = sum((x - m) ** 2 for x in d) / (n - 1)
    se = math.sqrt(v / n) if v > 0 else 1e-9
    t = m / se
    return m, t, norm_p(t), n


def arm_table(title, rows, venue, subset=None):
    sel = [r for r in rows if r["venue"] == venue and (subset is None or subset(r))]
    if not sel:
        return
    by = defaultdict(dict)
    for r in sel:
        by[r["arm"]][r["day"]] = r
    days = sorted(set(r["day"] for r in sel if r["arm"] == BASE))
    if not days:
        return
    lot = 65 if venue == "NIFTY" else 20
    p("")
    p("### %s — %s  (n=%d days, per LOT)" % (venue, title, len(days)))
    p("")
    p("| arm | total ₹/lot | mean ₹/lot/day | median | win% | worst day | p5 | stop-fire% | avg re-centers | max rc |")
    p("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    base_net = [by[BASE][d]["net"] for d in days if d in by[BASE]]
    res = {}
    for arm in ARM_ORDER:
        if arm not in by:
            continue
        dd = [d for d in days if d in by[arm] and d in by[BASE]]
        a = [by[arm][d]["net"] for d in dd]
        if not a:
            continue
        rc = [by[arm][d]["n_recenters"] for d in dd]
        fire = [by[arm][d]["stop_fired"] for d in dd]
        res[arm] = dict(days=dd, net=a)
        p("| %s | %+.0f | %+.0f | %+.0f | %.0f%% | %+.0f | %+.0f | %.0f%% | %.2f | %d |" % (
            ("**%s**" % arm) if arm == BASE else arm,
            sum(a), sum(a) / len(a), med(a),
            100.0 * sum(1 for v in a if v > 0) / len(a),
            min(a), pct(a, 5),
            100.0 * sum(fire) / len(fire), sum(rc) / float(len(rc)), max(rc)))
    # paired vs base
    p("")
    p("**Paired vs the incumbent `%s`** (same days, net of measured cost; "
      "family-wise Holm over %d comparisons):" % (BASE, len(res) - 1))
    p("")
    p("| arm | Δ mean ₹/lot/day | t | p (raw) | Holm-adj p | beats incumbent? |")
    p("|---|---:|---:|---:|---:|---|")
    comps = []
    for arm in ARM_ORDER:
        if arm == BASE or arm not in res:
            continue
        dd = res[arm]["days"]
        a = res[arm]["net"]
        b = [by[BASE][d]["net"] for d in dd]
        m, t, pv, n = paired(a, b)
        comps.append((arm, m, t, pv))
    k = len(comps)
    order = sorted(range(k), key=lambda i: comps[i][3])
    holm = [0.0] * k
    run = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, comps[i][3] * (k - rank))
        run = max(run, adj)
        holm[i] = run
    for i, (arm, m, t, pv) in enumerate(comps):
        verdict = "YES" if (m > 0 and holm[i] < 0.05) else ("no" if m <= 0 else "not after haircut")
        p("| %s | %+.0f | %+.2f | %.3f | %.3f | %s |" % (arm, m, t, pv, holm[i], verdict))
    return by, days


def churn_cost(rows, venue):
    sel = [r for r in rows if r["venue"] == venue]
    by = defaultdict(dict)
    for r in sel:
        by[r["arm"]][r["day"]] = r
    p("")
    p("### %s — churn-cost decomposition (what the extra round trips actually cost)" % venue)
    p("")
    p("| arm | extra cycles | extra-cycle GROSS ₹/lot | extra-cycle COST ₹/lot | extra-cycle NET ₹/lot | cost as %% of extra gross |")
    p("|---|---:|---:|---:|---:|---:|")
    for arm in ARM_ORDER:
        if arm not in by:
            continue
        rs = list(by[arm].values())
        nrc = sum(r["n_recenters"] for r in rs)
        if nrc == 0:
            continue
        g = sum(r["gross_after"] for r in rs)
        c = sum(r["cost_after"] for r in rs)
        n = sum(r["net_after"] for r in rs)
        p("| %s | %d | %+.0f | %.0f | %+.0f | %s |" % (
            arm, nrc, g, c, n,
            ("%.0f%%" % (100.0 * c / abs(g))) if abs(g) > 1 else "n/a"))


def per_dte(rows, venue, arms):
    sel = [r for r in rows if r["venue"] == venue]
    by = defaultdict(dict)
    for r in sel:
        by[r["arm"]][r["day"]] = r
    dte_of = {r["day"]: r["dte_trd"] for r in sel}
    p("")
    p("### %s — per trading-DTE (net ₹/lot/day; DTE0 = expiry day)" % venue)
    p("")
    hdr = "| DTE | n days | " + " | ".join(arms) + " |"
    p(hdr)
    p("|---|---:|" + "---:|" * len(arms))
    buckets = [("0", lambda d: d == 0), ("1", lambda d: d == 1),
               ("2", lambda d: d == 2), ("3+", lambda d: d >= 3)]
    for lbl, f in buckets:
        dd = sorted(d for d, v in dte_of.items() if f(v) and d in by[BASE])
        if not dd:
            continue
        cells = []
        for arm in arms:
            a = [by[arm][d]["net"] for d in dd if d in by[arm]]
            cells.append("%+.0f" % (sum(a) / len(a)) if a else "-")
        p("| %s | %d | %s |" % (lbl, len(dd), " | ".join(cells)))


def oos(rows, venue, arms):
    sel = [r for r in rows if r["venue"] == venue]
    by = defaultdict(dict)
    for r in sel:
        by[r["arm"]][r["day"]] = r
    p("")
    p("### %s — OOS split (IS = r/96's own day set ≤ %s; OOS = after the deploy decision)" % (venue, IS_END))
    p("")
    p("| period | n days | " + " | ".join(arms) + " |")
    p("|---|---:|" + "---:|" * len(arms))
    for lbl, f in (("IS  ≤2026-07-28", lambda d: d <= IS_END),
                   ("OOS ≥2026-07-29", lambda d: d > IS_END)):
        dd = sorted(d for d in by[BASE] if f(d))
        if not dd:
            continue
        cells = []
        for arm in arms:
            a = [by[arm][d]["net"] for d in dd if d in by[arm]]
            cells.append("%+.0f" % (sum(a) / len(a)) if a else "-")
        p("| %s | %d | %s |" % (lbl, len(dd), " | ".join(cells)))


def main():
    rows = load()
    p("# research/141 — aggregation output")
    p("")
    p("Source: `results/arms_daily.csv` — %d rows, %d venue-days."
      % (len(rows), len(set((r["venue"], r["day"]) for r in rows))))
    KEY = ["ONE_AND_DONE", "RECENTER_1", "RECENTER_2", "RECENTER_3", "RECENTER_5",
           "RECENTER_3_CD15", "MOVESTOP_ONE", "MOVESTOP_RECENTER"]
    for venue in ("NIFTY", "SENSEX"):
        p("")
        p("## %s" % venue)
        arm_table("ALL DAYS", rows, venue)
        churn_cost(rows, venue)
        per_dte(rows, venue, KEY)
        oos(rows, venue, KEY)
        arm_table("NEAR-EXPIRY trading DTE<=1", rows, venue,
                  subset=lambda r: r["dte_trd"] <= 1)
    with open(os.path.join(RES, "aggregate.md"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
