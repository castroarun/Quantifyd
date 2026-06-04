"""research/57 G6 — re-entry timing: immediate at 15:20 (capture overnight theta) vs next 09:20.
Sequential book, recipe move1.5+PT40 EOD + 2% crash. On EOD close, re-enter same tick vs next morning."""
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path("/home/arun/quantifyd"); OUT = ROOT / "research/57_positional_straddle_biweekly/results"
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; BROK = 80; ROLL_DTE = 1; MOVE = 1.5; PT = 40; CRASH = 2.0
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

def newpos(sp, day, t):
    exps = EXP.get(day, [])
    if len(exps) < 2: return None
    E = exps[1]; K = round(sp / 50) * 50
    ce = prem(K, "CE", E, t); pe = prem(K, "PE", E, t)
    if ce and pe: return dict(K=K, E=E, credit=ce + pe, spot0=sp)
    return None

def simulate(reentry):  # "0920" or "1520"
    realized = 0.0; pos = None; last = -1e18; curve = []
    for t in TL:
        day = t[:10]; hm = t[11:16]; sp = SPOTS[t]
        if pos is None:
            if hm >= "09:20" and hm <= "09:21":
                pos = newpos(sp, day, t)
                if pos: last = gmin(t)
            continue
        K, E, credit, spot0 = pos["K"], pos["E"], pos["credit"], pos["spot0"]
        ce = prem(K, "CE", E, t); pe = prem(K, "PE", E, t)
        if ce is None or pe is None: continue
        mtm = (credit - (ce + pe)) * LOT; mv = abs(sp - spot0) / spot0 * 100
        ex = False; eod = ("15:18" <= hm <= "15:21")
        if gmin(t) - last >= 5:
            last = gmin(t)
            if mv >= CRASH: ex = True
        if not ex and eod:
            if mv >= MOVE or mtm >= PT / 100.0 * credit * LOT or dte(E, day) <= ROLL_DTE: ex = True
        if ex:
            realized += mtm - 2 * BROK; curve.append(realized); pos = None
            if reentry == "1520" and eod and hm <= "15:21":   # re-enter immediately at the close tick
                pos = newpos(sp, day, t)
                if pos: last = gmin(t)
            continue
    if pos is not None:
        t = TL[-1]; ce = prem(pos["K"], "CE", pos["E"], t); pe = prem(pos["K"], "PE", pos["E"], t)
        if ce and pe: realized += (pos["credit"] - (ce + pe)) * LOT - 160; curve.append(realized)
    eq = np.array(curve) if curve else np.array([0.0]); mdd = (eq - np.maximum.accumulate(eq)).min()
    return len(curve), round(realized), round(mdd)

print("=== re-entry timing (sequential book, 2% crash) ===", flush=True)
L = ["# research/57 G6 — re-entry timing: immediate-15:20 vs next-09:20\n",
     "Sequential book, recipe move1.5+PT40 + 2%% crash. On EOD close, re-enter same tick (capture overnight theta) vs next morning. 30d SIGNAL.\n",
     "| re-entry | closes | final P&L | book max-DD |", "|---|---|---|---|"]
for mode, name in [("0920", "next 09:20 (flat overnight)"), ("1520", "immediate 15:20 (stay short overnight)")]:
    n, f, m = simulate(mode)
    L.append("| %s | %d | %+d | %d |" % (name, n, f, m))
    print("  %-40s closes=%d final=%+d book-maxDD=%d" % (name, n, f, m), flush=True)
L += ["\n## Read", "- immediate-15:20 captures the overnight theta of the fresh straddle (more time short) but eats any overnight gap on it.",
      "- next-09:20 is flat overnight (misses that decay) but re-enters AFTER the gap, cleanly re-centred.",
      "- 30d SIGNAL, calm regime (small gaps) -> likely favours staying short; a gap month would favour waiting."]
(OUT / "RESULTS_g6_reentry.md").write_text("\n".join(L), encoding="utf-8")
print("G6 DONE", flush=True)
oc.close()
