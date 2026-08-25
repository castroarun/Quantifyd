#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/126 Stage 12 - the 2026-08-25 WORKED EXAMPLE, with TIMEB2's 8 real lots.

Rebuilds the full live portfolio curve for the day that triggered the commission, from
broker-truth sources, and prices what each Arm-B2 candidate trigger would have done.
"""
import sqlite3
import json
import os
from collections import defaultdict

Q = "/home/arun/quantifyd/"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
DAY = "2026-08-25"
REP = []


def log(m):
    REP.append(str(m))
    print(m, flush=True)


def hm(ts):
    return ts[11:16] if "T" in ts else ts[:5]


books = defaultdict(dict)
for nm, db in (("916_ATM", "nas_916_atm_trading.db"),
               ("916_ATM2", "nas_916_atm2_trading.db"),
               ("916_ATM4", "nas_916_atm4_trading.db")):
    c = sqlite3.connect("file:%sbacktest_data/%s?mode=ro" % (Q, db), uri=True)
    for ts, dp in c.execute(
            "SELECT ts, day_pnl FROM nas_mtm_snapshots WHERE snap_date=? ORDER BY ts", (DAY,)):
        if dp is not None:
            books[nm][hm(ts)] = float(dp)
    c.close()

st_ = json.load(open(Q + "backtest_data/csl_paper_state.json"))
for r in st_.get("records", []):
    if r["day"] != DAY or r.get("source") != "REAL":
        continue
    for t, p in (r.get("series") or []):
        books[r["book"]][hm(t)] = float(p)

live = json.load(open(Q + "static/app/csl_paper_live.json"))
if live.get("day") == DAY:
    for bk, v in live.get("books", {}).items():
        if bk == "NAS_COMB20":
            for t, p in (v.get("series") or []):
                books[bk][hm(t)] = float(p)

tb2 = [x for x in json.load(open(
    Q + "research/125_expiry_afternoon_straddle/results/timeb2_live_days.json"))
    if x["day"] == DAY]

log("=" * 100)
log("2026-08-25 WORKED EXAMPLE - the full live book, TIMEB2's 8 REAL LOTS INCLUDED")
log("=" * 100)
log("")
log("TIMEB2 (standalone one-shot ledger + broker fills; the daemon never recorded it):")
for x in tb2:
    log("  qty=%d lots=%d window 13:15-14:30 %s credit=%.2f debit=%.2f booked=%+d"
        % (x["qty"], x["lots"], x.get("reason"), x["credit"], x["debit"], x["pnl"]))

c = sqlite3.connect("file:%sbacktest_data/options_data.db?mode=ro" % Q, uri=True)
if tb2:
    K, credit, qty, exp = (tb2[0]["strike"], tb2[0]["credit"], tb2[0]["qty"], tb2[0]["expiry"])
    legs = defaultdict(dict)
    for ts, it, ltp in c.execute(
            "SELECT snapshot_time, instrument_type, ltp FROM option_chain WHERE symbol='NIFTY' "
            "AND snapshot_time>=? AND snapshot_time<? AND strike=? AND expiry_date=? "
            "AND ltp IS NOT NULL", (DAY, DAY + "z", K, exp)):
        legs[hm(ts)][it] = ltp
    for t in sorted(legs):
        d = legs[t]
        if "CE" in d and "PE" in d and "13:15" <= t <= "14:30":
            books["CSL_TIMEB2_NIFTY"][t] = (credit - (d["CE"] + d["PE"])) * qty

allmins = sorted({t for b in books.values() for t in b})
allmins = [t for t in allmins if "09:15" <= t <= "15:30"]
cur_v, seen = {}, {}
for b in books:
    seen[b], cur_v[b] = False, 0.0
curve = {}
for t in allmins:
    for b in books:
        if t in books[b]:
            cur_v[b], seen[b] = books[b][t], True
    curve[t] = sum(cur_v[b] for b in books if seen[b])

pk = max(allmins, key=lambda t: curve[t])
log("")
log("Portfolio curve (LIVE-money books only):")
hdr = "  %-7s %11s   %s" % ("time", "portfolio", "  ".join("%-16s" % b for b in sorted(books)))
log(hdr)
for t in ("13:15", "13:45", "14:00", "14:03", "14:15", "14:30", "14:33", "15:00", allmins[-1]):
    if t not in curve:
        continue
    parts = []
    for b in sorted(books):
        keys = [x for x in books[b] if x <= t]
        v = books[b][max(keys)] if keys else None
        parts.append("%-16s" % ("%+.0f" % v if v is not None else "-"))
    log("  %-7s %+11.0f   %s" % (t, curve[t], "  ".join(parts)))
log("")
log("  PEAK %+.0f at %s   |   FINAL %+.0f at %s   |   GIVE-BACK %.0f"
    % (curve[pk], pk, curve[allmins[-1]], allmins[-1], curve[pk] - curve[allmins[-1]]))

sp = {}
for ts, s in c.execute(
        "SELECT snapshot_time, underlying_spot FROM option_chain WHERE symbol='NIFTY' "
        "AND snapshot_time>=? AND snapshot_time<? AND underlying_spot IS NOT NULL",
        (DAY, DAY + "z")):
    t = hm(ts)
    if t not in sp:
        sp[t] = s
log("")
log("What each Arm-B2 trigger would have done on THIS day:")
log("  %-16s %6s %9s %10s %s" % ("trigger", "arms?", "arm_hm", "spot", "wing strikes (100-wide)"))
for thr in (5000, 8000, 10000, 12000, 15000, 20000):
    arm = next((t for t in allmins if curve[t] >= thr), None)
    if arm and sp.get(arm):
        atm = round(sp[arm] / 50) * 50
        log("  ABS_%-12d %6s %9s %10.1f %d CE / %d PE"
            % (thr, "YES", arm, sp[arm], atm + 100, atm - 100))
    else:
        log("  ABS_%-12d %6s %9s %10s %s" % (thr, "no", "-", "-", "never reached"))

open(os.path.join(RES, "worked_0825.txt"), "w").write("\n".join(REP) + "\n")
print("\nwrote results/worked_0825.txt")
