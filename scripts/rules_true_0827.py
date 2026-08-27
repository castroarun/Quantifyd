#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RULES-TRUE reconstruction for 2026-08-27 — what the rules would have booked.

Arun took manual control mid-session and adjusted the book. His standing instruction is that
manual trades are a proxy for what the system should have done, and that the app must record the
RULES-TRUE outcome so the day can be assessed on rules rather than on intervention.

Two of today's divergences are NOT manual and matter separately:
  * SX-ATM was closed at 09:16:17 / 09:16:47 by the NO_LEG_SL race (fixed today, deploys 15:40).
    The DTE0 rule is HOLD to 15:15 with no leg stop, so its booked -Rs649 reflects the defect.
  * SX-ATM2 move-stopped legitimately at 09:31 and re-entered 77400 - that part IS rules-true.

Prices come from options_data.db (1-min recorded chain), read-only. Writes a report only; it does
NOT touch any live position row. The CSL books (TimeB, COMB) self-record correctly because their
exit path reconciles at the live premium AT THE RULE EXIT TIME, so they are reported for
completeness but need no reconstruction.
"""
import sqlite3, json, os
from datetime import date

DAY = "2026-08-27"
OC = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT_MD = "/home/arun/quantifyd/docs/LIVE_RECON_LOG.md"
OUT_JSON = "/home/arun/quantifyd/static/app/rules_true_%s.json" % DAY

oc = sqlite3.connect(OC)

def px(sym, hhmm):
    """Recorded chain LTP for a symbol at a minute (nearest print in that minute)."""
    r = oc.execute(
        "SELECT ltp FROM option_chain WHERE tradingsymbol=? AND snapshot_time BETWEEN ? AND ? "
        "ORDER BY snapshot_time LIMIT 1",
        (sym, "%sT%s:00" % (DAY, hhmm), "%sT%s:59" % (DAY, hhmm))).fetchone()
    return float(r[0]) if r and r[0] is not None else None

def last_px(sym, upto="15:15"):
    r = oc.execute(
        "SELECT ltp FROM option_chain WHERE tradingsymbol=? AND snapshot_time<=? AND ltp IS NOT NULL "
        "ORDER BY snapshot_time DESC LIMIT 1", (sym, "%sT%s:59" % (DAY, upto))).fetchone()
    return float(r[0]) if r else None

# book -> (legs [(sym, entry, qty)], rule_exit, rule_text, actual_pnl, why_diverged)
BOOKS = {
 "SENSEX ATM": dict(
    legs=[("SENSEX26AUG77600CE", 149.60, 40), ("SENSEX26AUG77600PE", 112.10, 40)],
    exit="15:15", rule="DTE0: NO per-leg stop, hold to 15:15 (research/114)",
    actual=-649, why="closed 09:16 by the NO_LEG_SL race - NOT a rule exit"),
 "SENSEX ATM4": dict(
    legs=[("SENSEX26AUG77600CE", 149.60, 40), ("SENSEX26AUG77600PE", 113.60, 40)],
    exit="15:15", rule="DTE0: NO per-leg stop, roll-to-match cannot fire, hold to 15:15",
    actual=None, why="PE closed manually 12:08"),
 "SENSEX ATM2 (re-entry)": dict(
    legs=[("SENSEX26AUG77400CE", 143.50, 40), ("SENSEX26AUG77400PE", 130.20, 40)],
    exit="15:15", rule="0.4% move-stop + per-leg 30%, exit 15:15",
    actual=None, why="closed manually 12:21"),
 "TimeB SENSEX": dict(
    legs=[("SENSEX26AUG77300CE", 171.80, 160), ("SENSEX26AUG77300PE", 59.83, 160)],
    exit="15:20", rule="DTE0: no % stop, 50% disaster backstop, exit 15:20",
    actual=None, why="closed manually; CSL self-reconciles at the rule exit"),
 "NIFTY COMB": dict(
    legs=[("NIFTY2690124300CE", 115.28, 325), ("NIFTY2690124300PE", 90.41, 325)],
    exit="15:20", rule="DTE3: combined-premium SL 20%, exit 15:20",
    actual=None, why="CE closed manually; CSL self-reconciles at the rule exit"),
}

rows, total = [], 0.0
for name, b in BOOKS.items():
    credit = sum(e for _, e, _ in b["legs"])
    legs_out, pnl, missing = [], 0.0, False
    for sym, ent, qty in b["legs"]:
        x = px(sym, b["exit"]) or last_px(sym, b["exit"])
        if x is None:
            missing = True; legs_out.append((sym, ent, None, None)); continue
        p = (ent - x) * qty
        pnl += p
        legs_out.append((sym, ent, x, round(p)))
    rows.append(dict(book=name, rule=b["rule"], why=b["why"], credit=round(credit, 2),
                     exit_at=b["exit"], legs=legs_out,
                     rules_true=None if missing else round(pnl),
                     actual=b["actual"],
                     delta=None if (missing or b["actual"] is None) else round(pnl - b["actual"])))
    if not missing:
        total += pnl

lines = []
lines.append("\n## %s — RULES-TRUE reconstruction (manual control taken mid-session)\n" % DAY)
lines.append("```")
lines.append("What the deployed rules would have booked, priced from the recorded 1-min chain.")
lines.append("Arun adjusted the book manually from ~12:08; SX-ATM was closed at 09:16 by the")
lines.append("NO_LEG_SL race (fix deploys 15:40), which is a DEFECT exit, not a rule exit.")
lines.append("")
for r in rows:
    lines.append("%-24s  rule: %s" % (r["book"], r["rule"]))
    lines.append("%-24s  why diverged: %s" % ("", r["why"]))
    for sym, ent, x, p in r["legs"]:
        lines.append("      %-22s entry %8.2f  ->  %s  %s"
                     % (sym, ent, ("%8.2f" % x) if x is not None else "   n/a  ",
                        ("%+8d" % p) if p is not None else ""))
    rt = r["rules_true"]
    lines.append("      RULES-TRUE @%s: %s%s" % (
        r["exit_at"], ("%+d" % rt) if rt is not None else "no chain data",
        ("   (actual %+d, delta %+d)" % (r["actual"], r["delta"])) if r["delta"] is not None else ""))
    lines.append("")
lines.append("RULES-TRUE TOTAL across reconstructed books: %+d" % round(total))
lines.append("```")

with open(OUT_MD, "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump({"day": DAY, "rows": rows, "rules_true_total": round(total)}, open(OUT_JSON, "w"), indent=1)
print("\n".join(lines))
print("\nwritten: %s  and  %s" % (OUT_MD, OUT_JSON))
