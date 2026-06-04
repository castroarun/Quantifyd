"""research/57 G2 — combined recipe (PT-40 + move-1.5%) + does the overnight wing still add value?

Precompute per carry day: straddle MTM (15:20), spot, AND the overnight wing-night P&L
(buy K±W at 15:20 -> sell 09:20 next day). Then evaluate management combos with/without wings.
Indexed ltp + in-memory spots. Net Rs80/leg (straddle 2 + wings 2/night). 30d SIGNAL.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path("/home/arun/quantifyd"); OUT = ROOT / "research/57_positional_straddle_biweekly/results"
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; BROK = 80; ROLL_DTE = 1; WING = 500
oc = sqlite3.connect(str(OPT))
oc.execute("CREATE INDEX IF NOT EXISTS idx_oc_lk ON option_chain(expiry_date,strike,instrument_type,snapshot_time)")
EXP = {}
for day, exp in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10), expiry_date FROM option_chain WHERE symbol='NIFTY'"):
    if exp >= day: EXP.setdefault(day, set()).add(exp)
EXP = {d: sorted(s) for d, s in EXP.items()}; DAYS = sorted(EXP)
SPOTS = {}
for st, sp in oc.execute("SELECT snapshot_time, spot_price FROM underlying_spot WHERE symbol='NIFTY' AND spot_price>0 ORDER BY snapshot_time"):
    SPOTS.setdefault(st[:10], []).append((st[11:16], float(sp)))
def spot_at(day, hhmm):
    arr = SPOTS.get(day)
    if not arr: return None
    c = [s for (t, s) in arr if t <= hhmm]
    return c[-1] if c else arr[0][1]
def ltp(strike, ot, E, day, hhmm):
    r = oc.execute("SELECT ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND snapshot_time>=? AND snapshot_time<=? AND symbol='NIFTY' AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
                   (E, strike, ot, day + "T00:00:00", day + "T" + hhmm + ":59")).fetchone()
    return float(r[0]) if r and r[0] else None
def dte(E, day):
    return (datetime.strptime(E, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days

print("precomputing paths + wing nights...", flush=True)
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
    for i, d in enumerate(carry):
        c = ltp(K, "CE", E, d, "15:20"); p = ltp(K, "PE", E, d, "15:20"); sp = spot_at(d, "15:20")
        if not (c and p and sp): continue
        wing_night = 0.0
        if i < len(carry) - 1:
            dn = carry[i + 1]
            wcb = ltp(K + WING, "CE", E, d, "15:20"); wpb = ltp(K - WING, "PE", E, d, "15:20")
            wcs = ltp(K + WING, "CE", E, dn, "09:20"); wps = ltp(K - WING, "PE", E, dn, "09:20")
            if wcb and wcs: wing_night += (wcs - wcb) * LOT - 2 * BROK
            if wpb and wps: wing_night += (wps - wpb) * LOT - 2 * BROK
        path.append((d, sp, (credit - (c + p)) * LOT, wing_night))
    if len(path) >= 2:
        PATHS.append(dict(spot0=spot0, credit=credit, path=path))
print("paths:", len(PATHS), flush=True)

def evaluate(move_pct=None, pt_pct=None, wings=False):
    out = []
    for tr in PATHS:
        spot0, credit, path = tr["spot0"], tr["credit"], tr["path"]
        wing_acc = 0.0; chosen = path[-1][2]
        for j, (d, sp, mtm, wn) in enumerate(path):
            stop = (move_pct and abs(sp - spot0) / spot0 * 100 >= move_pct) or (pt_pct and mtm >= pt_pct / 100.0 * credit * LOT) or (d == path[-1][0])
            if stop:
                chosen = mtm; break
            if wings: wing_acc += wn   # held this night
        out.append(chosen - 2 * BROK + (wing_acc if wings else 0))
    return np.array(out)

CONFIGS = [("no-mgmt", {}), ("move1.5", dict(move_pct=1.5)), ("PT40", dict(pt_pct=40)),
           ("move1.5+PT40 (COMBO)", dict(move_pct=1.5, pt_pct=40)),
           ("COMBO + wings500", dict(move_pct=1.5, pt_pct=40, wings=True)),
           ("PT40 + wings500", dict(pt_pct=40, wings=True)),
           ("no-mgmt + wings500", dict(wings=True))]
L = ["# research/57 G2 — combined recipe + overnight-wing value (bi-weekly short straddle, %d trades)\n" % len(PATHS),
     "move-stop 1.5%% + profit-target 40%%-credit, with/without ±%dpt overnight wings. Net Rs80/leg. **30d SIGNAL.**\n" % WING,
     "| recipe | total | mean/trade | median | win% | worst | std |", "|---|---|---|---|---|---|---|"]
print("=== G2 ===", flush=True)
for name, kw in CONFIGS:
    a = evaluate(**kw)
    L.append("| %s | %+d | %+d | %+d | %d | %d | %d |" % (name, a.sum(), round(a.mean()), round(np.median(a)), round(100*(a>0).mean()), a.min(), round(a.std())))
    print("  %-22s mean=%+d median=%+d win%%=%d worst=%d std=%d" % (name, round(a.mean()), round(np.median(a)), round(100*(a>0).mean()), a.min(), round(a.std())), flush=True)
L += ["\n## Read", "- Does COMBO (capture via PT40 + cap tail via move1.5) beat each lever alone on mean AND worst?",
      "- With the tight stop already capping the tail, do overnight wings still add value or just bleed cost?",
      "- 30d SIGNAL, overlapping entries. Best recipe -> G3 entry timing + the forward paper logger."]
(OUT / "RESULTS_g2.md").write_text("\n".join(L), encoding="utf-8")
print("G2 DONE", flush=True)
oc.close()
