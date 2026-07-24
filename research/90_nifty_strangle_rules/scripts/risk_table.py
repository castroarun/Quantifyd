#!/usr/bin/env python3
"""Risk-focused view of g1_ranking.csv: what do stops cost (mean) and buy (tail)?"""
import csv
import os

RES = "/home/arun/quantifyd/research/90_nifty_strangle_rules/results"
rows = list(csv.DictReader(open(os.path.join(RES, "g1_ranking.csv"))))

hdr = ["arm", "p", "pt", "stop", "gb", "net_mean", "net_t", "net_post22_mean",
       "worst", "p5", "win_rate", "avg_hold"]
print(" | ".join(f"{h:>8}" for h in hdr))
for arm, p in (("M", "0.025"), ("W", "0.012")):
    print(f"--- {arm} p={p} " + "-" * 80)
    sel = [r for r in rows if r["arm"] == arm and r["p"] == p]
    sel.sort(key=lambda r: (r["pt"] or "z", r["stop"] or "z", r["gb"]))
    for r in sel:
        print(" | ".join(f"{(r[h] or '-'):>8}" for h in hdr))
