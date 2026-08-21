#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/116 - aggregate the ratchet bake-off: P&L AND give-back, side by side."""
import csv, os, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
DETAIL = os.path.join(RES, "ratchet_detail.csv")

COST = {"NIFTY": 250.0, "SENSEX": 200.0}   # Rs/lot round-trip = the materiality threshold

VORDER = ["NO_DEFENCE", "STATIC", "BE_CLAMP_50", "BE_CLAMP_60", "BE_CLAMP_70",
          "RATCHET_K1.3", "RATCHET_K1.5", "RATCHET_K1.75", "RATCHET_K2", "RATCHET_K2.5",
          "GIVEBACK_30", "GIVEBACK_50", "RS_GB_1000", "RS_GB_2000", "RS_GB_3000",
          "TIME_RATCHET_MID", "HYBRID_BE60_GB50"]


def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[i]


def load():
    rows = list(csv.DictReader(open(DETAIL)))
    for r in rows:
        for k in ("gross", "net", "peak_gross", "giveback"):
            r[k] = float(r[k])
        r["credit"] = float(r["credit"])
        r["dte"] = int(r["dte"])
    return rows


def block(rows, label):
    """rows = all variants over one slice of (construction, day) keys."""
    byv = defaultdict(list)
    for r in rows:
        byv[r["variant"]].append(r)
    static = {(r["construction"], r["day"]): r for r in byv.get("STATIC", [])}
    out = []
    for v in VORDER:
        vr = byv.get(v)
        if not vr:
            continue
        nets = [r["net"] for r in vr]
        gbs = [r["giveback"] for r in vr]
        # only days that ever showed a meaningful open profit tell us about give-back
        gbs_live = [r["giveback"] for r in vr if r["peak_gross"] >= COST[r["venue"]]]
        resc = cut = 0
        diffs = []
        for r in vr:
            s = static.get((r["construction"], r["day"]))
            if not s:
                continue
            d = r["net"] - s["net"]
            diffs.append(d)
            thr = COST[r["venue"]]
            if d >= thr:
                resc += 1
            elif d <= -thr:
                cut += 1
        stops = sum(1 for r in vr if r["reason"] != "WINDOW")
        out.append(dict(
            slice=label, variant=v, n=len(vr),
            total=round(sum(nets)), mean=round(st.mean(nets)),
            median=round(st.median(nets)),
            win_pct=round(100.0 * sum(1 for x in nets if x > 0) / len(nets)),
            worst=round(min(nets)), p05=round(pct(nets, 5)),
            gb_p50=round(pct(gbs_live, 50)), gb_p90=round(pct(gbs_live, 90)),
            gb_max=round(max(gbs_live) if gbs_live else 0),
            gb_mean=round(st.mean(gbs_live)) if gbs_live else 0,
            n_gb=len(gbs_live),
            rescue=resc, cut_short=cut,
            uplift=round(sum(diffs)),
            early_exit_pct=round(100.0 * stops / len(vr))))
    return out


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", path)


def show(rows, title):
    print("\n=== %s ===" % title)
    print("%-17s %4s %8s %7s %7s %5s %8s | %7s %7s %7s %4s | %4s %4s %8s %5s"
          % ("variant", "n", "total", "mean", "median", "win%", "worst",
             "gb_p50", "gb_p90", "gb_max", "ngb", "resc", "cut", "uplift", "exit%"))
    for r in rows:
        print("%-17s %4d %8d %7d %7d %5d %8d | %7d %7d %7d %4d | %4d %4d %8d %5d"
              % (r["variant"], r["n"], r["total"], r["mean"], r["median"], r["win_pct"],
                 r["worst"], r["gb_p50"], r["gb_p90"], r["gb_max"], r["n_gb"],
                 r["rescue"], r["cut_short"], r["uplift"], r["early_exit_pct"]))


def main():
    rows = load()
    days = sorted({r["day"] for r in rows})
    print("rows %d | days %d (%s .. %s)" % (len(rows), len(days), days[0], days[-1]))
    for cn in sorted({r["construction"] for r in rows}):
        sub = [r for r in rows if r["construction"] == cn and r["variant"] == "STATIC"]
        print("  %-14s cells=%3d  dte=%s" % (cn, len(sub), sorted({r["dte"] for r in sub})))

    allrows = []
    a = block(rows, "ALL")
    allrows += a
    show(a, "ALL CONSTRUCTIONS POOLED (net Rs per lot)")

    for cn in sorted({r["construction"] for r in rows}):
        b = block([r for r in rows if r["construction"] == cn], cn)
        allrows += b
        show(b, cn)

    write(os.path.join(RES, "ratchet_summary.csv"), allrows)

    # per venue x DTE
    dterows = []
    for ven in sorted({r["venue"] for r in rows}):
        for dte in sorted({r["dte"] for r in rows if r["venue"] == ven}):
            sl = [r for r in rows if r["venue"] == ven and r["dte"] == dte]
            b = block(sl, "%s_DTE%d" % (ven, dte))
            dterows += b
    write(os.path.join(RES, "ratchet_by_dte.csv"), dterows)
    for lab in sorted({r["slice"] for r in dterows}):
        show([r for r in dterows if r["slice"] == lab], lab)


if __name__ == "__main__":
    main()
