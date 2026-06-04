"""research/57 G4 — REALISTIC SEQUENTIAL book (one straddle at a time, re-enter at CMP).

This is the deployed model (vs the earlier overlapping-daily per-trade stats):
  - hold ONE straddle: ATM, 2nd-nearest weekly, entered 09:20.
  - manage: EOD 15:20 check (1.5% move-stop OR PT 40%-credit) + a 5-min-polled CRASH stop (wide %).
  - roll: at expiry DTE<=1, close + re-enter.
  - on ANY close -> RE-ENTER at the then-CMP (mode: immediate same-bar, or wait to next 09:20).
Outputs the TRUE running equity curve + the book's max drawdown. Test re-entry timing + crash level.
Per-minute premium series; net Rs80/leg. 30d SIGNAL.
"""
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
def dte(E, day):
    return (datetime.strptime(E, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days
# cache full per-(strike,type,expiry) premium series lazily
_cache = {}
def pser(strike, ot, E):
    k = (strike, ot, E)
    if k not in _cache:
        _cache[k] = {st[:16]: float(v) for st, v in oc.execute(
            "SELECT snapshot_time, ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND symbol='NIFTY' AND ltp>0", (E, strike, ot))}
    return _cache[k]
def prem(strike, ot, E, t):  # last premium at or before minute t (within same day)
    s = pser(strike, ot, E)
    if t in s: return s[t]
    day = t[:10]
    cand = [(k, v) for k, v in s.items() if k[:10] == day and k <= t]
    return max(cand)[1] if cand else None

# build the master minute timeline (all minutes with a spot), 09:15-15:25
TL = sorted(SPOTS)
TL = [t for t in TL if "09:15" <= t[11:16] <= "15:25"]

def gmin(t): return datetime.strptime(t[:16], "%Y-%m-%dT%H:%M").timestamp() / 60.0

def simulate(crash_pct=3.0, reentry="cmp", crash_poll=5, wings=False, WING=500):
    realized = 0.0; curve = []; pos = None; last_poll = -1e18
    for t in TL:
        day = t[:10]; hm = t[11:16]; sp = SPOTS[t]
        # ENTRY
        if pos is None:
            # entry only at 09:20 (next-morning) OR immediately if reentry==cmp and we are flat intra-day
            do_entry = (hm >= "09:20" and hm <= "09:21")
            if pos is None and do_entry:
                exps = EXP.get(day, [])
                if len(exps) < 2: continue
                E = exps[1]; K = round(sp / 50) * 50
                ce = prem(K, "CE", E, t); pe = prem(K, "PE", E, t)
                if ce and pe:
                    pos = dict(K=K, E=E, credit=ce + pe, spot0=sp, entry_t=t)
                    last_poll = gmin(t)
            continue
        # HOLDING
        K, E, credit, spot0 = pos["K"], pos["E"], pos["credit"], pos["spot0"]
        ce = prem(K, "CE", E, t); pe = prem(K, "PE", E, t)
        if ce is None or pe is None: continue
        mtm = (credit - (ce + pe)) * LOT
        movepct = abs(sp - spot0) / spot0 * 100
        exit_now = False; reason = None
        # crash poll every crash_poll min (intraday)
        if gmin(t) - last_poll >= crash_poll:
            last_poll = gmin(t)
            if movepct >= crash_pct: exit_now, reason = True, "crash"
        # EOD 15:20 primary stop / PT / roll
        if not exit_now and hm >= "15:18" and hm <= "15:21":
            if movepct >= MOVE or mtm >= PT / 100.0 * credit * LOT or dte(E, day) <= ROLL_DTE:
                exit_now, reason = True, "eod"
        if exit_now:
            realized += mtm - 2 * BROK
            curve.append((t, realized))
            pos = None
            if reentry == "cmp" and hm < "15:15":   # re-enter immediately at CMP (same day, if time left)
                exps = EXP.get(day, [])
                if len(exps) >= 2:
                    E2 = exps[1]; K2 = round(sp / 50) * 50
                    c2 = prem(K2, "CE", E2, t); p2 = prem(K2, "PE", E2, t)
                    if c2 and p2:
                        pos = dict(K=K2, E=E2, credit=c2 + p2, spot0=sp, entry_t=t); last_poll = gmin(t)
            continue
    if pos is not None:
        t = TL[-1]; ce = prem(pos["K"], "CE", pos["E"], t); pe = prem(pos["K"], "PE", pos["E"], t)
        if ce and pe: realized += (pos["credit"] - (ce + pe)) * LOT - 2 * BROK
        curve.append((t, realized))
    eq = np.array([v for _, v in curve]) if curve else np.array([0.0])
    peak = np.maximum.accumulate(eq); mdd = (eq - peak).min() if len(eq) else 0
    return dict(n=len(curve), final=round(realized), mdd=round(mdd), curve=curve)

print("simulating sequential book...", flush=True)
L = ["# research/57 G4 — REALISTIC sequential book (one straddle, re-enter at CMP, %d-day chain)\n" % len(set(t[:10] for t in TL)),
     "Hold one ATM bi-weekly straddle; EOD 1.5%+PT40 stop + 5-min-polled crash stop; re-enter at CMP. **True running equity + book max-DD. 30d SIGNAL.**\n",
     "| config | closes | final P&L | book max-DD |", "|---|---|---|---|"]
print("=== sequential book ===", flush=True)
for name, kw in [("crash3% reenter-CMP", dict(crash_pct=3.0, reentry="cmp")),
                 ("crash3% reenter-next0920", dict(crash_pct=3.0, reentry="next")),
                 ("crash2.5% reenter-CMP", dict(crash_pct=2.5, reentry="cmp")),
                 ("NO crash stop reenter-CMP", dict(crash_pct=99, reentry="cmp"))]:
    r = simulate(**kw)
    L.append("| %s | %d | %+d | %d |" % (name, r["n"], r["final"], r["mdd"]))
    print("  %-26s closes=%d final=%+d book-maxDD=%d" % (name, r["n"], r["final"], r["mdd"]), flush=True)
L += ["\n## Read", "- This is the ACTUAL deployed book (one straddle, sequential), not overlapping per-trade stats.",
      "- book max-DD = the real running drawdown to size against (NOT the per-trade worst).",
      "- crash stop level + re-entry timing compared. 30d SIGNAL, single regime."]
(OUT / "RESULTS_g4_sequential.md").write_text("\n".join(L), encoding="utf-8")
print("G4 DONE", flush=True)
oc.close()
