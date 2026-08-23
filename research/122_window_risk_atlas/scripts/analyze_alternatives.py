#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/122 - alternatives scan: does any grid window DOMINATE a deployed cell
(median net >= deployed AND bridged p95 adverse <= deployed) on the same venue-DTE?
A candidate only counts if its neighbours agree (r/120 plateau rule): same duration
at start +/-30min, and adjacent durations at the same start, must also have a
positive median. Everything found is reported; the family-wise haircut is applied in
prose (the scan touches ~10 cells per deployed row x 5 rows).
Writes results/alternatives_report.txt.
"""
import csv, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

DEPLOYED = {
    "MON_NIFTY_DTE1":  ("NIFTY", "1", "DEP_1300_1400", "SL20"),
    "TUE_NIFTY_DTE0":  ("NIFTY", "0", "DEP_0930_1100", "SL25"),
    "WED_SENSEX_DTE1": ("SENSEX", "1", "DEP_1030_1200", "SL20"),
    "THU_SENSEX_DTE0": ("SENSEX", "0", "DEP_1300_1520", "NOSTOP"),
    "FRI_NIFTY_DTE2":  ("NIFTY", "2", "DEP_1000_1200", "SL20"),
}
DURS = ["60", "90", "120", "H"]


def main():
    rows = list(csv.DictReader(open(os.path.join(RES, "atlas.csv"))))
    idx = {(r["venue"], r["dte"], r["cell"], r["arm"]): r for r in rows}
    out = []

    def num(r, k):
        return float(r[k]) if r.get(k) not in (None, "") else None

    for name, (venue, dte, cell, arm) in DEPLOYED.items():
        dep = idx.get((venue, dte, cell, arm))
        if not dep:
            out.append("%s: MISSING" % name)
            continue
        dmed, dp95 = num(dep, "med_net_10L"), num(dep, "adv_p95_capped")
        out.append("\n=== %s (%s DTE%s %s-%s %s): median %+.0f  p95adv %.0f ==="
                   % (name, venue, dte, dep["start"], dep["end"], arm, dmed, dp95))
        # scan grid cells, same venue+dte, comparison at SL20 (the deployed stop family)
        cand = []
        for r in rows:
            if r["venue"] != venue or r["dte"] != dte or not r["cell"].startswith("G_"):
                continue
            if r["arm"] != ("NOSTOP" if arm == "NOSTOP" else "SL20"):
                continue
            m, p = num(r, "med_net_10L"), num(r, "adv_p95_capped")
            if m is None or p is None:
                continue
            cand.append((m, p, r))
        cand.sort(key=lambda t: -t[0])
        out.append("  top-5 grid cells by median: " + "; ".join(
            "%s %+.0f/p95 %.0f" % (r["cell"], m, p) for m, p, r in cand[:5]))
        dom = [(m, p, r) for m, p, r in cand if m >= dmed and p <= dp95]
        if not dom:
            out.append("  DOMINATORS: none")
            continue
        for m, p, r in dom:
            # plateau: neighbours
            c = r["cell"]                      # G_HH:MM_dur
            _, hm, dur = c.split("_")
            h, mi = int(hm[:2]), int(hm[3:5])
            t = h * 60 + mi
            nb = []
            for dt2 in (-30, 30):
                nb.append("G_%02d:%02d_%s" % ((t + dt2) // 60, (t + dt2) % 60, dur))
            di = DURS.index(dur)
            for j in (di - 1, di + 1):
                if 0 <= j < len(DURS):
                    nb.append("G_%s_%s" % (hm, DURS[j]))
            nmed = []
            ok = True
            for nc in nb:
                r2 = idx.get((venue, dte, nc, r["arm"]))
                if r2 is None:
                    continue
                v = num(r2, "med_net_10L")
                nmed.append("%s %+.0f" % (nc, v))
                if v is not None and v <= 0:
                    ok = False
            out.append("  DOMINATOR %s median %+.0f p95adv %.0f  neighbours[%s]  plateau=%s"
                       % (c, m, p, ", ".join(nmed), "PASS" if ok else "FAIL"))
    txt = "\n".join(out)
    open(os.path.join(RES, "alternatives_report.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
