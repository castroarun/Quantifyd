"""research/57 G5 — crash-stop level test (user: 3% too far, try 2%) + intraday move distribution."""
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path("/home/arun/quantifyd"); OUT = ROOT / "research/57_positional_straddle_biweekly/results"
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; BROK = 80; ROLL_DTE = 1; MOVE = 1.5; PT = 40
oc = sqlite3.connect(str(OPT))
oc.execute("CREATE INDEX IF NOT EXISTS idx_oc_lk ON option_chain(expiry_date,strike,instrument_type,snapshot_time)")
EXP = {}
for day, exp in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10), expiry_date FROM option_chain WHERE symbol='NIFTY'"):
    if exp >= day: EXP.setdefault(day, set()).add(exp)
EXP = {d: sorted(s) for d, s in EXP.items()}; DAYS = sorted(EXP)
SPOTS = {}
for st, sp in oc.execute("SELECT snapshot_time, spot_price FROM underlying_spot WHERE symbol='NIFTY' AND spot_price>0"):
    SPOTS[st[:16]] = float(sp)
TL = sorted(t for t in SPOTS if "09:15" <= t[11:16] <= "15:25")
def dte(E, day): return (datetime.strptime(E, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days
def gmin(t): return datetime.strptime(t[:16], "%Y-%m-%dT%H:%M").timestamp() / 60.0
_c = {}
def pser(K, ot, E):
    k = (K, ot, E)
    if k not in _c:
        _c[k] = {st[:16]: float(v) for st, v in oc.execute("SELECT snapshot_time,ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND symbol='NIFTY' AND ltp>0", (E, K, ot))}
    return _c[k]
def prem(K, ot, E, t):
    s = pser(K, ot, E)
    if t in s: return s[t]
    cand = [(k, v) for k, v in s.items() if k[:10] == t[:10] and k <= t]
    return max(cand)[1] if cand else None

# --- 1. intraday max-move distribution (overlapping daily entries) ---
maxmoves = []
for d0 in DAYS:
    exps = EXP.get(d0, [])
    if len(exps) < 2: continue
    E = exps[1]; spot0 = SPOTS.get(d0 + "T09:20")
    if not spot0: continue
    carry = [d for d in DAYS if d >= d0 and dte(E, d) >= ROLL_DTE]
    if len(carry) < 2: continue
    mx = 0.0
    for t in TL:
        if t[:10] < d0 or t[:10] > carry[-1] or t < d0 + "T09:20": continue
        mx = max(mx, abs(SPOTS[t] - spot0) / spot0 * 100)
    maxmoves.append(mx)
mm = np.array(maxmoves)
print("=== per-trade MAX intraday move-from-entry (n=%d) ===" % len(mm), flush=True)
print("  median %.2f%%  mean %.2f%%  biggest %.2f%%" % (np.median(mm), mm.mean(), mm.max()), flush=True)
for thr in (1.5, 2.0, 2.5, 3.0):
    print("  trades that EVER reached %.1f%%: %d of %d" % (thr, (mm >= thr).sum(), len(mm)), flush=True)

# --- 2. sequential book with different crash levels ---
def simulate(crash_pct):
    realized = 0.0; pos = None; last = -1e18; curve = []
    for t in TL:
        day = t[:10]; hm = t[11:16]; sp = SPOTS[t]
        if pos is None:
            if hm >= "09:20" and hm <= "09:21":
                exps = EXP.get(day, [])
                if len(exps) < 2: continue
                E = exps[1]; K = round(sp / 50) * 50
                ce = prem(K, "CE", E, t); pe = prem(K, "PE", E, t)
                if ce and pe: pos = dict(K=K, E=E, credit=ce + pe, spot0=sp); last = gmin(t)
            continue
        K, E, credit, spot0 = pos["K"], pos["E"], pos["credit"], pos["spot0"]
        ce = prem(K, "CE", E, t); pe = prem(K, "PE", E, t)
        if ce is None or pe is None: continue
        mtm = (credit - (ce + pe)) * LOT; mv = abs(sp - spot0) / spot0 * 100
        ex = False
        if gmin(t) - last >= 5:
            last = gmin(t)
            if mv >= crash_pct: ex = True
        if not ex and hm >= "15:18" and hm <= "15:21":
            if mv >= MOVE or mtm >= PT / 100.0 * credit * LOT or dte(E, day) <= ROLL_DTE: ex = True
        if ex:
            realized += mtm - 2 * BROK; curve.append(realized); pos = None
    if pos is not None:
        t = TL[-1]; ce = prem(pos["K"], "CE", pos["E"], t); pe = prem(pos["K"], "PE", pos["E"], t)
        if ce and pe: realized += (pos["credit"] - (ce + pe)) * LOT - 160; curve.append(realized)
    eq = np.array(curve) if curve else np.array([0.0]); mdd = (eq - np.maximum.accumulate(eq)).min()
    return len(curve), round(realized), round(mdd)

print("\n=== sequential book, crash-stop level (recipe move1.5+PT40 EOD) ===", flush=True)
L = ["# research/57 G5 — crash-stop level + intraday move distribution\n",
     "## Per-trade MAX intraday move-from-entry (n=%d): median %.2f%%, biggest %.2f%%" % (len(mm), np.median(mm), mm.max()),
     "| threshold | trades that ever reached it |", "|---|---|"]
for thr in (1.5, 2.0, 2.5, 3.0): L.append("| %.1f%% | %d of %d |" % (thr, (mm >= thr).sum(), len(mm)))
L += ["\n## Sequential book by crash-stop level", "| crash stop | closes | final P&L | book max-DD |", "|---|---|---|---|"]
for cp in (1.75, 2.0, 2.5, 3.0, 99):
    n, f, m = simulate(cp)
    nm = "no crash (EOD only)" if cp == 99 else "%.2f%%" % cp
    L.append("| %s | %d | %+d | %d |" % (nm, n, f, m))
    print("  crash %-18s closes=%d final=%+d book-maxDD=%d" % (nm, n, f, m), flush=True)
L += ["\n## Read", "- If few/no trades reach 2%%, a 2%% crash stop rarely fires on this calm sample (same as 3%%).",
      "- The crash stop is INSURANCE for a fast intraday move the EOD-1.5%% check would miss until 15:20 - its value shows up in a violent month, not this one.", "- 30d SIGNAL."]
(OUT / "RESULTS_g5_crash.md").write_text("\n".join(L), encoding="utf-8")
print("G5 DONE", flush=True)
oc.close()
