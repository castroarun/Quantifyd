#!/usr/bin/env python3
"""research/174 — summarise results/liquidity_by_dte.csv into the P0b gate table."""
import csv
import statistics as st
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results" / "liquidity_by_dte.csv"
rows = list(csv.DictReader(open(RES)))


def f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def block(rs, title):
    print("\n=== %s  (n=%d rows) ===" % (title, len(rs)))
    print("%5s %6s %8s %8s %9s %9s %10s %12s %10s" %
          ("dte", "n", "listed%", "trad%", "medDist%", "medPrem", "medVolCE",
           "medOI_CE", "nTradStrk"))
    by = {}
    for r in rs:
        by.setdefault(int(r["target_dte"]), []).append(r)
    for d in sorted(by):
        g = by[d]
        listed = [r for r in g if r["listed_strike"]]
        trad = [r for r in g if r["trad_strike"]]
        md = st.median([f(r["trad_dist_pct"]) for r in trad]) if trad else float("nan")
        mp = st.median([f(r["trad_prem"]) for r in trad]) if trad else float("nan")
        mv = st.median([f(r["trad_ce_contracts"]) for r in trad]) if trad else float("nan")
        mo = st.median([f(r["trad_ce_oi"]) for r in trad]) if trad else float("nan")
        ns = st.median([f(r["n_traded_strikes"]) for r in g])
        print("%5d %6d %7.0f%% %7.0f%% %9.2f %9.1f %10.0f %12.0f %10.0f" %
              (d, len(g), 100.0 * len(listed) / len(g), 100.0 * len(trad) / len(g),
               md, mp, mv, mo, ns))


block(rows, "ALL YEARS")
for era, lo, hi in [("2015 (long-dated present)", "2015-01-01", "2015-12-31"),
                    ("2016-2024-02 (75-DTE truncated download)", "2016-01-01", "2024-02-29"),
                    ("2024-03 -> 2026-09 (long-dated present)", "2024-03-01", "2026-12-31")]:
    sub = [r for r in rows if lo <= r["entry_date"] <= hi]
    if sub:
        block(sub, era)
