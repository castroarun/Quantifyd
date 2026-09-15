#!/usr/bin/env python3
"""research/174 — the P0b gate table from results/liquidity_v2.csv."""
import csv
import statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results" / "liquidity_v2.csv"
rows = list(csv.DictReader(open(RES)))


def f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def med(v):
    return st.median(v) if v else float("nan")


def block(rs, title):
    print("\n=== %s  (sessions covered: %d) ===" % (title, len(set(r["session"] for r in rs))))
    print("%5s %7s %8s %9s %9s %9s %10s %10s %11s" %
          ("tenor", "n_sess", "fill%", "medDTE", "medDist%", "medPrem",
           "medVolCE", "medVolPE", "medOI_CE"))
    by = {}
    for r in rs:
        by.setdefault(int(r["target_dte"]), []).append(r)
    for t in sorted(by):
        g = by[t]
        ok = [r for r in g if r["fillable"] == "1"]
        print("%5d %7d %7.0f%% %9.0f %9.2f %9.1f %10.0f %10.0f %11.0f" %
              (t, len(g), 100.0 * len(ok) / len(g),
               med([f(r["best_dte"]) for r in ok]),
               med([f(r["atm_dist_pct"]) for r in ok]),
               med([f(r["atm_prem"]) for r in ok]),
               med([f(r["ce_contracts"]) for r in ok]),
               med([f(r["pe_contracts"]) for r in ok]),
               med([f(r["ce_oi"]) for r in ok])))


block(rows, "ALL 2015-2026")
print("\n\n---- fill% by YEAR x TENOR (a tenor is usable only where this is high) ----")
yrs = sorted(set(r["session"][:4] for r in rows))
tens = sorted(set(int(r["target_dte"]) for r in rows))
print("%5s " % "year" + "".join("%7d" % t for t in tens))
for y in yrs:
    line = "%5s " % y
    for t in tens:
        g = [r for r in rows if r["session"][:4] == y and int(r["target_dte"]) == t]
        line += "%6.0f%%" % (100.0 * sum(1 for r in g if r["fillable"] == "1") / len(g)) if g else "%7s" % "-"
    print(line)

print("\n\n---- median ATM CE volume (contracts/day) by YEAR x TENOR ----")
print("%5s " % "year" + "".join("%8d" % t for t in tens))
for y in yrs:
    line = "%5s " % y
    for t in tens:
        g = [r for r in rows if r["session"][:4] == y and int(r["target_dte"]) == t
             and r["fillable"] == "1"]
        line += "%8.0f" % med([f(r["ce_contracts"]) for r in g]) if g else "%8s" % "-"
    print(line)

print("\n\n---- median ATM distance from spot (%) by YEAR x TENOR ----")
print("%5s " % "year" + "".join("%8d" % t for t in tens))
for y in yrs:
    line = "%5s " % y
    for t in tens:
        g = [r for r in rows if r["session"][:4] == y and int(r["target_dte"]) == t
             and r["fillable"] == "1"]
        line += "%8.2f" % med([f(r["atm_dist_pct"]) for r in g]) if g else "%8s" % "-"
    print(line)
