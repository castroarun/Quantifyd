"""research/57 G1 (fast, indexed) — MANAGEMENT sweep.
One-pass day->expiries map; spots in memory; ltp via indexed snapshot_time range (no substr scans)."""
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path("/home/arun/quantifyd"); OUT = ROOT / "research/57_positional_straddle_biweekly/results"
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; BROK = 80; ROLL_DTE = 1
oc = sqlite3.connect(str(OPT))
oc.execute("CREATE INDEX IF NOT EXISTS idx_oc_lk ON option_chain(expiry_date,strike,instrument_type,snapshot_time)")

# one pass: day -> sorted future expiries
print("building day->expiries map (one scan)...", flush=True)
EXP = {}
for day, exp in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10), expiry_date FROM option_chain WHERE symbol='NIFTY'"):
    if exp >= day: EXP.setdefault(day, set()).add(exp)
EXP = {d: sorted(s) for d, s in EXP.items()}
DAYS = sorted(EXP)
print("days:", len(DAYS), flush=True)

# spots in memory
SPOTS = {}
for st, sp in oc.execute("SELECT snapshot_time, spot_price FROM underlying_spot WHERE symbol='NIFTY' AND spot_price>0 ORDER BY snapshot_time"):
    SPOTS.setdefault(st[:10], []).append((st[11:16], float(sp)))
def spot_at(day, hhmm):
    arr = SPOTS.get(day)
    if not arr: return None
    c = [s for (t, s) in arr if t <= hhmm]
    return c[-1] if c else arr[0][1]

def ltp(strike, ot, E, day, hhmm):
    lo = day + "T00:00:00"; hi = day + "T" + hhmm + ":59"
    r = oc.execute("SELECT ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND snapshot_time>=? AND snapshot_time<=? AND symbol='NIFTY' AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
                   (E, strike, ot, lo, hi)).fetchone()
    return float(r[0]) if r and r[0] else None
def dte(E, day):
    return (datetime.strptime(E, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days

print("precomputing carry paths...", flush=True)
PATHS = []
for d0 in DAYS:
    exps = EXP.get(d0, [])
    if len(exps) < 2: continue
    E = exps[1]; spot0 = spot_at(d0, "09:20")
    if not spot0: continue
    K = round(spot0 / 50) * 50
    ce0 = ltp(K, "CE", E, d0, "09:20"); pe0 = ltp(K, "PE", E, d0, "09:20")
    if not ce0 or not pe0: continue
    credit = ce0 + pe0
    carry = [d for d in DAYS if d >= d0 and dte(E, d) >= ROLL_DTE]
    if len(carry) < 2: continue
    path = []
    for d in carry:
        c = ltp(K, "CE", E, d, "15:20"); p = ltp(K, "PE", E, d, "15:20"); sp = spot_at(d, "15:20")
        if c and p and sp: path.append((d, sp, (credit - (c + p)) * LOT))
    if len(path) >= 2:
        PATHS.append(dict(spot0=spot0, credit=credit, path=path))
print("paths:", len(PATHS), flush=True)

def evaluate(move_pct=None, pt_pct=None):
    out = []
    for tr in PATHS:
        spot0, credit, path = tr["spot0"], tr["credit"], tr["path"]
        chosen = path[-1][2]
        for (d, sp, mtm) in path:
            if (move_pct and abs(sp - spot0) / spot0 * 100 >= move_pct) or (pt_pct and mtm >= pt_pct / 100.0 * credit * LOT) or (d == path[-1][0]):
                chosen = mtm; break
        out.append(chosen - 2 * BROK)
    return np.array(out)

CONFIGS = [("move 0.4%", dict(move_pct=0.4)), ("move 0.5%", dict(move_pct=0.5)), ("move 0.7%", dict(move_pct=0.7)),
           ("move 1.0%", dict(move_pct=1.0)), ("move 1.5%", dict(move_pct=1.5)),
           ("move 0.4%+PT40", dict(move_pct=0.4, pt_pct=40)), ("move 0.7%+PT40", dict(move_pct=0.7, pt_pct=40)),
           ("move 1.5%+PT40", dict(move_pct=1.5, pt_pct=40)), ("no-mgmt", {})]
L = ["# research/57 G1 — management sweep (bi-weekly short straddle, no wings, %d trades)\n" % len(PATHS),
     "Short ATM straddle 2nd-nearest weekly, daily carry, exit on move-stop / profit-target / DTE<=1. Net Rs80/leg. **30d SIGNAL, overlapping entries.**\n",
     "| management | total | mean/trade | median | win% | worst | std |", "|---|---|---|---|---|---|---|"]
print("=== G7 0.4%% RE-VALIDATION ===", flush=True)
for name, kw in CONFIGS:
    a = evaluate(**kw)
    L.append("| %s | %+d | %+d | %+d | %d | %d | %d |" % (name, a.sum(), round(a.mean()), round(np.median(a)), round(100*(a>0).mean()), a.min(), round(a.std())))
    print("  %-14s mean=%+d median=%+d win%%=%d worst=%d std=%d" % (name, round(a.mean()), round(np.median(a)), round(100*(a>0).mean()), a.min(), round(a.std())), flush=True)
L += ["\n## Read", "- Which move-stop cuts the worst-trade tail WITHOUT killing the mean (monotonic > peak)?",
      "- PT raises win% but caps the big theta runs. 30d SIGNAL, overlapping entries. Best -> G2 wings + G3 entry."]
(OUT / "RESULTS_g7_revalidate.md").write_text("\n".join(L), encoding="utf-8")
print("G1 DONE", flush=True)
oc.close()
