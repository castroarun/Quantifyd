#!/usr/bin/env python3
"""VPS nightly regen: merge replay outputs into the travel template and deploy.
Run AFTER run_replay_nsrw.py + run_replay_paths.py (see regen_travel.sh)."""
import csv
import json

BASE = "/home/arun/quantifyd"
RES = BASE + "/research/90_nifty_strangle_rules/results"
TPL = BASE + "/research/90_nifty_strangle_rules/travel_template.html"
OUTS = [BASE + "/static/app/nsrw-travel-research90.html",
        BASE + "/frontend/public/nsrw-travel-research90.html"]

paths = json.load(open(RES + "/replay_paths.json"))
rows = {(r["entry_time"][:10], int(r["T"])): r
        for r in csv.DictReader(open(RES + "/replay_nsrw_cycles.csv"))}
for c in paths:
    r = rows.get((c["week"], c["T"]))
    if r:
        c["net"] = float(r["net"])
        c["entry_time"] = r["entry_time"]
        c["exit"] = r["exit_time"]
        c["reason"] = r["exit_reason"]
    else:
        c["net"] = c["gross"]
html = open(TPL, encoding="utf-8").read().replace(
    "/*__DATA__*/[]", json.dumps(paths, separators=(",", ":")))
for o in OUTS:
    open(o, "w", encoding="utf-8").write(html)
print("deployed", len(paths), "cycles ->", OUTS[0])
