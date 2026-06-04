"""research/57 G2b — OVERNIGHT GAP RISK on the recipe (user's question).

The G2 move-stop was checked only at 15:20 (daily). A position sitting <1.5% at EOD carries
NAKED overnight; a gap-open can breach the stop at a worse price than 1.5%. Test honestly:
  1. distribution of overnight underlying gaps (15:20 -> next 09:20)
  2. recipe worst with 15:20-ONLY stop  vs  GAP-AWARE stop (also check 09:20 open)
  3. the specific at-risk nights (held <1.5% at EOD, then gapped)
  4. do the EOD wings RECOVER the gap losses the stop can't prevent? (re-examine 'wings redundant')
Indexed ltp + in-memory spots. Net Rs80/leg. 30d SIGNAL.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path("/home/arun/quantifyd"); OUT = ROOT / "research/57_positional_straddle_biweekly/results"
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; BROK = 80; ROLL_DTE = 1; WING = 500; MOVE = 1.5; PT = 40
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
def spot_open(day):
    arr = SPOTS.get(day)
    return arr[0][1] if arr else None
def ltp(strike, ot, E, day, hhmm):
    r = oc.execute("SELECT ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND snapshot_time>=? AND snapshot_time<=? AND symbol='NIFTY' AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
                   (E, strike, ot, day + "T00:00:00", day + "T" + hhmm + ":59")).fetchone()
    return float(r[0]) if r and r[0] else None
def dte(E, day):
    return (datetime.strptime(E, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days

# 1. overnight gap distribution (15:20 -> next 09:20)
gaps = []
for i in range(len(DAYS) - 1):
    a = spot_at(DAYS[i], "15:20"); b = spot_at(DAYS[i + 1], "09:20")
    if a and b: gaps.append((DAYS[i + 1], (b - a) / a * 100))
gv = np.array([g for _, g in gaps])
print("=== overnight gaps (15:20->09:20), n=%d ===" % len(gv), flush=True)
print("  mean|gap|=%.2f%%  median|gap|=%.2f%%  worst=%.2f%%  >0.5%%:%d  >1%%:%d" % (
    np.abs(gv).mean(), np.median(np.abs(gv)), gv[np.argmax(np.abs(gv))], (np.abs(gv) > 0.5).sum(), (np.abs(gv) > 1.0).sum()), flush=True)
big = sorted(gaps, key=lambda x: -abs(x[1]))[:5]
print("  biggest:", [(d, round(g, 2)) for d, g in big], flush=True)

print("\nprecomputing paths (09:20 + 15:20 marks + wing nights)...", flush=True)
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
        s9 = spot_open(d); c9 = ltp(K, "CE", E, d, "09:20"); p9 = ltp(K, "PE", E, d, "09:20")
        s15 = spot_at(d, "15:20"); c15 = ltp(K, "CE", E, d, "15:20"); p15 = ltp(K, "PE", E, d, "15:20")
        if not (s15 and c15 and p15): continue
        m9 = (credit - (c9 + p9)) * LOT if (c9 and p9) else None
        m15 = (credit - (c15 + p15)) * LOT
        wn = 0.0
        if i < len(carry) - 1:
            dn = carry[i + 1]
            wcb = ltp(K + WING, "CE", E, d, "15:20"); wpb = ltp(K - WING, "PE", E, d, "15:20")
            wcs = ltp(K + WING, "CE", E, dn, "09:20"); wps = ltp(K - WING, "PE", E, dn, "09:20")
            if wcb and wcs: wn += (wcs - wcb) * LOT - 2 * BROK
            if wpb and wps: wn += (wps - wpb) * LOT - 2 * BROK
        path.append(dict(d=d, first=(i == 0), last=(d == carry[-1]), s9=s9, m9=m9, s15=s15, m15=m15, wn=wn))
    if len(path) >= 2:
        PATHS.append(dict(spot0=spot0, credit=credit, path=path))
print("paths:", len(PATHS), flush=True)

def run(gap_aware=False, wings=False):
    out = []
    for tr in PATHS:
        spot0, credit, path = tr["spot0"], tr["credit"], tr["path"]
        wing_acc = 0.0; res = path[-1]["m15"]
        for nd in path:
            # gap-aware: check the 09:20 open stop first (except entry day)
            if gap_aware and not nd["first"] and nd["s9"] and nd["m9"] is not None:
                if abs(nd["s9"] - spot0) / spot0 * 100 >= MOVE:
                    res = nd["m9"]; break
            if (abs(nd["s15"] - spot0) / spot0 * 100 >= MOVE) or (nd["m15"] >= PT / 100.0 * credit * LOT) or nd["last"]:
                res = nd["m15"]; break
            if wings: wing_acc += nd["wn"]
        out.append(res - 2 * BROK + (wing_acc if wings else 0))
    return np.array(out)

def stat(a): return "mean=%+d median=%+d win%%=%d worst=%d std=%d" % (round(a.mean()), round(np.median(a)), round(100*(a>0).mean()), a.min(), round(a.std()))
A = run(gap_aware=False); B = run(gap_aware=True); C = run(gap_aware=True, wings=True)
print("\n=== recipe (move1.5+PT40) ===", flush=True)
print("  15:20-only stop (G2 model):", stat(A), flush=True)
print("  GAP-AWARE stop (09:20+15:20):", stat(B), flush=True)
print("  GAP-AWARE + EOD wings       :", stat(C), flush=True)

L = ["# research/57 G2b — overnight GAP RISK on the recipe (move1.5+PT40, %d trades)\n" % len(PATHS),
     "User Q: a position <1.5%% at EOD carries NAKED overnight; a gap-open breaches the stop at a worse price. Tested.\n",
     "## Overnight gaps (15:20->09:20, n=%d)" % len(gv),
     "- mean|gap| %.2f%%, median %.2f%%, **worst %.2f%%**; nights >0.5%%: %d, >1%%: %d" % (np.abs(gv).mean(), np.median(np.abs(gv)), gv[np.argmax(np.abs(gv))], (np.abs(gv)>0.5).sum(), (np.abs(gv)>1.0).sum()),
     "- biggest: " + ", ".join("%s %+.2f%%" % (d, g) for d, g in big),
     "\n## Recipe under each stop model", "| stop model | " + "mean | median | win% | worst | std".replace(" | "," | ") + " |", "|---|---|---|---|---|---|",
     "| 15:20-only (the G2 number) | %s |" % stat(A).replace("mean=","").replace(" median=","|").replace(" win%=","|").replace(" worst=","|").replace(" std=","|"),
     "| GAP-AWARE (09:20+15:20) | %s |" % stat(B).replace("mean=","").replace(" median=","|").replace(" win%=","|").replace(" worst=","|").replace(" std=","|"),
     "| GAP-AWARE + EOD wings | %s |" % stat(C).replace("mean=","").replace(" median=","|").replace(" win%=","|").replace(" worst=","|").replace(" std=","|"),
     "\n## Read", "- If GAP-AWARE worst >> 15:20-only worst, overnight gaps DO breach the stop -> real risk, the 15:20 number understated it.",
     "- If wings RECOVER that gap loss (worst back up), the 'wings redundant' verdict was an artifact -> wings DO earn their keep for the overnight gap the stop can't prevent.",
     "- 30d SIGNAL."]
(OUT / "RESULTS_g2b_gaprisk.md").write_text("\n".join(L), encoding="utf-8")
print("G2b DONE", flush=True)
oc.close()
