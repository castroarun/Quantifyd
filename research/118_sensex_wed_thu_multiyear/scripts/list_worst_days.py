#!/usr/bin/env python3
"""List the worst held-straddle days in the 2024-2026 real-option sample, and check each
against the underlying move measured independently from the 1-minute index series."""
import csv, os
RES = "/home/arun/quantifyd/research/118_sensex_wed_thu_multiyear/results"
B = list(csv.DictReader(open(os.path.join(RES, "stage_a2_bhav_straddle_daily.csv"))))
P = {r["day"]: r for r in csv.DictReader(open(os.path.join(RES, "stage_b_daily_price_action.csv")))}
out = []
for r in sorted(B, key=lambda r: float(r["pnl_pts"]))[:20]:
    p = P.get(r["day"], {})
    out.append(dict(day=r["day"], wd=r["weekday"], dte=r["dte"], era=r["era"],
                    credit=r["credit"], terminal=r["terminal"], src=r["terminal_src"],
                    pnl_pts=r["pnl_pts"],
                    idx_move=p.get("move_term", ""), idx_mae=p.get("mae_pts", ""),
                    mae_at=p.get("mae_hm", ""), shape=("%s" % p.get("revert_ratio", ""))))
print("%-12s %-4s %-4s %-4s %8s %8s %-15s %9s %9s %9s %6s %6s" %
      ("day", "wd", "dte", "era", "credit", "terminal", "src", "pnl_pts", "idx_move", "idx_mae", "at", "rev"))
for r in out:
    print("%-12s %-4s %-4s %-4s %8s %8s %-15s %9s %9s %9s %6s %6s" %
          (r["day"], r["wd"], r["dte"], r["era"], r["credit"], r["terminal"], r["src"],
           r["pnl_pts"], r["idx_move"], r["idx_mae"], r["mae_at"], r["shape"]))
with open(os.path.join(RES, "worst_days_2024_2026.csv"), "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    wtr.writeheader()
    for r in out:
        wtr.writerow(r)
print("\nwrote worst_days_2024_2026.csv")
# count of days worse than -500 pts by weekday and dte
for key in ("weekday", "dte"):
    agg = {}
    for r in B:
        k = r[key]
        agg.setdefault(k, [0, 0])
        agg[k][1] += 1
        if float(r["pnl_pts"]) < -500:
            agg[k][0] += 1
    print("\ndays worse than -500 pts by", key)
    for k in sorted(agg):
        print("   %-5s %2d / %3d  (%.1f%%)" % (k, agg[k][0], agg[k][1], 100.0 * agg[k][0] / agg[k][1]))
