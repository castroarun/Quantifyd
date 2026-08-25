#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/125 Stage 0 — rebuild the REAL combined LIVE portfolio intraday curve and
reconcile 2026-08-25 (peak +14,983 @14:03 -> +7,442 @14:33) before any sweep.

Sources (READ-ONLY):
  nas_mtm_snapshots in nas_916_atm/atm2/atm4_trading.db  (per-minute day_pnl, 70 days)
  backtest_data/csl_paper_state.json  records[] (rolling ~8 days) -> per-book series
  static/app/csl_paper_live.json      still-open books (dropped once a book closes)
Writes results/stage0_live_portfolio.csv + stage0_recon.txt
"""
import sqlite3, json, os, csv
from collections import defaultdict

Q = "/home/arun/quantifyd/"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

SUITE = [("916_ATM", "nas_916_atm_trading.db"),
         ("916_ATM2", "nas_916_atm2_trading.db"),
         ("916_ATM4", "nas_916_atm4_trading.db")]
# books counted as LIVE (real money) in the csl paper-state ledger
LIVE_BOOKS = {"NAS_COMB20", "CSL_TIMEB_NIFTY", "CSL_TIMEB_SENSEX",
              "CSL30F_SENSEX_WED", "CSL_TIMEB_NIFTY_THU"}

out = []
def log(m):
    out.append(m); print(m, flush=True)


def hm(ts):
    """'2026-08-25T14:03:00' or '14:03' -> 'HH:MM'"""
    if "T" in ts: return ts[11:16]
    if len(ts) >= 5 and ts[2] == ":": return ts[:5]
    return ts[:5]


# ---------- 1. suite curves from nas_mtm_snapshots ----------
suite = defaultdict(lambda: defaultdict(dict))   # day -> book -> {hm: day_pnl}
for name, db in SUITE:
    c = sqlite3.connect("file:%sbacktest_data/%s?mode=ro" % (Q, db), uri=True)
    for d, ts, dp in c.execute(
            "SELECT snap_date, ts, day_pnl FROM nas_mtm_snapshots ORDER BY snap_date, ts"):
        if dp is None: continue
        suite[d][name][hm(ts)] = float(dp)
    c.close()
log("suite: %d days %s..%s" % (len(suite), min(suite), max(suite)))

# ---------- 2. csl books ----------
st = json.load(open(Q + "backtest_data/csl_paper_state.json"))
csl = defaultdict(lambda: defaultdict(dict))     # day -> book -> {hm: pnl}
meta = defaultdict(dict)
for r in st.get("records", []):
    bk, day, src = r["book"], r["day"], r.get("source")
    if bk not in LIVE_BOOKS or src != "REAL":
        continue
    for t, p in (r.get("series") or []):
        csl[day][bk][hm(t)] = float(p)
    meta[day][bk] = dict(pnl=r.get("pnl"), lots=r.get("lots"), qty=r.get("qty"),
                         cfg=r.get("cfg"), reason=r.get("reason"),
                         entry=r.get("entry_ts"), exit=r.get("exit_ts"), dte=r.get("dte"))
log("csl REAL book-days: %d over %d days" % (sum(len(v) for v in csl.values()), len(csl)))
for d in sorted(csl):
    log("   %s : %s" % (d, ", ".join("%s(%sL,%s)" % (b, meta[d][b]["lots"], meta[d][b]["reason"])
                                     for b in sorted(csl[d]))))

# still-open books today
try:
    live = json.load(open(Q + "static/app/csl_paper_live.json"))
    log("csl_paper_live.json books: %s" % (list(live.keys())[:12] if isinstance(live, dict) else type(live)))
except Exception as e:
    log("csl_paper_live.json unreadable: %s" % e)

# ---------- 3. combined portfolio curve, forward-filled per book ----------
rows = []
days = sorted(set(suite) | set(csl))
for day in days:
    books = {}
    for b, ser in suite.get(day, {}).items(): books["SUITE_" + b] = ser
    for b, ser in csl.get(day, {}).items():  books[b] = ser
    if not books: continue
    mins = sorted({t for s in books.values() for t in s})
    if not mins: continue
    cur = {b: 0.0 for b in books}
    started = {b: False for b in books}
    closed = {b: False for b in books}
    # a csl book's series ends at its exit -> after that its P&L is REALISED and frozen
    lastt = {b: max(s) for b, s in books.items()}
    firstt = {b: min(s) for b, s in books.items()}
    for t in mins:
        for b, s in books.items():
            if t in s:
                cur[b] = s[t]; started[b] = True
            elif started[b] and t > lastt[b]:
                closed[b] = True         # frozen at realised value
        tot = sum(cur[b] for b in books if started[b])
        rows.append(dict(day=day, t=t, total=round(tot, 2),
                         **{("bk_" + b): round(cur[b], 2) for b in books if started[b]}))

fields = ["day", "t", "total"] + sorted({k for r in rows for k in r if k.startswith("bk_")})
with open(os.path.join(RES, "stage0_live_portfolio.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in rows: w.writerow(r)
log("wrote %d curve rows over %d days" % (len(rows), len(days)))

# ---------- 4. per-day peak / final / give-back ----------
log("")
log("%-12s %8s %6s %9s %6s %9s %9s  books" % ("day", "peak", "@", "final", "@", "giveback", "trough"))
bydayrows = defaultdict(list)
for r in rows: bydayrows[r["day"]].append(r)
for day in days:
    rr = bydayrows[day]
    if not rr: continue
    pk = max(rr, key=lambda x: x["total"])
    tr = min(rr, key=lambda x: x["total"])
    fin = rr[-1]
    bl = sorted(k[3:] for k in rr[-1] if k.startswith("bk_"))
    log("%-12s %8.0f %6s %9.0f %6s %9.0f %9.0f  %s" % (
        day, pk["total"], pk["t"], fin["total"], fin["t"],
        pk["total"] - fin["total"], tr["total"], ",".join(bl)))

# ---------- 5. the 2026-08-25 reconciliation ----------
log("")
log("=== RECONCILIATION 2026-08-25 (claim: peak +14,983 @14:03 -> +7,442 @14:33) ===")
rr = bydayrows.get("2026-08-25", [])
if rr:
    idx = {r["t"]: r for r in rr}
    for t in ("14:00", "14:03", "14:10", "14:20", "14:30", "14:33", "14:40"):
        if t in idx:
            r = idx[t]
            parts = ", ".join("%s=%+.0f" % (k[3:], v) for k, v in sorted(r.items())
                              if k.startswith("bk_"))
            log("  %s total=%+9.0f   %s" % (t, r["total"], parts))
    pk = max(rr, key=lambda x: x["total"])
    log("  observed PEAK %+.0f at %s ; final %+.0f at %s ; give-back %.0f"
        % (pk["total"], pk["t"], rr[-1]["total"], rr[-1]["t"], pk["total"] - rr[-1]["total"]))
    # suite-only curve
    s_only = [(r["t"], sum(v for k, v in r.items() if k.startswith("bk_SUITE_")))
              for r in rr]
    sp = max(s_only, key=lambda x: x[1])
    log("  SUITE-ONLY peak %+.0f at %s ; final %+.0f  (venue trail arms at +12,000 -> %s)"
        % (sp[1], sp[0], s_only[-1][1], "WOULD ARM" if sp[1] >= 12000 else "NEVER ARMS"))
    log("  per-book close: %s" % json.dumps({b: meta["2026-08-25"][b]["pnl"]
                                             for b in meta.get("2026-08-25", {})}))
else:
    log("  NO ROWS for 2026-08-25")

open(os.path.join(RES, "stage0_recon.txt"), "w").write("\n".join(out) + "\n")
print("\nDONE -> results/stage0_recon.txt")
