"""research/58 V1 — INTRADAY RE-CENTER straddle backtest, per-day intraday P&L curves.
09:16 sell ATM straddle (nearest weekly). When |spot - strike| >= TRIGGER% (~94pts @0.4%), CLOSE
and SELL a fresh ATM straddle at the new level (REPLACE/re-center). Flatten 15:15. 10 lots.
Emits per-day 5-min cumulative-PnL series (entry->close) + a day-over-day curve, as JSON for the app."""
import sqlite3, json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

ROOT = Path("/home/arun/quantifyd")
OUT = ROOT / "research/58_intraday_recenter_straddle/results"; OUT.mkdir(parents=True, exist_ok=True)
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; LOTS = 10; QTY = LOT * LOTS; COST = 2 * 80          # ~per-straddle close (2 legs)
TRIGGER_PCT = 0.4; ENTER = "09:20"; EXITT = "15:15"

oc = sqlite3.connect(str(OPT))
oc.execute("CREATE INDEX IF NOT EXISTS idx_oc_lk ON option_chain(expiry_date,strike,instrument_type,snapshot_time)")
EXP = {}
for day, exp in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10), expiry_date FROM option_chain WHERE symbol='NIFTY'"):
    if exp >= day: EXP.setdefault(day, set()).add(exp)
EXP = {d: sorted(s) for d, s in EXP.items()}
DAYS = sorted(EXP)
SPOTS = {}
for st, sp in oc.execute("SELECT snapshot_time, spot_price FROM underlying_spot WHERE symbol='NIFTY' AND spot_price>0 ORDER BY snapshot_time"):
    SPOTS.setdefault(st[:10], []).append((st[11:16], float(sp)))

def spot_at(day, hhmm):
    arr = SPOTS.get(day)
    if not arr: return None
    c = [s for (t, s) in arr if t <= hhmm]
    return c[-1] if c else None
def ltp(strike, ot, E, day, hhmm):
    lo = day + "T00:00:00"; hi = day + "T" + hhmm + ":59"
    r = oc.execute("SELECT ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND snapshot_time>=? AND snapshot_time<=? AND symbol='NIFTY' AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
                   (E, strike, ot, lo, hi)).fetchone()
    return float(r[0]) if r and r[0] else None

GRID = []
t = datetime.strptime("09:15", "%H:%M"); end = datetime.strptime(EXITT, "%H:%M")
while t <= end: GRID.append(t.strftime("%H:%M")); t += timedelta(minutes=5)

per_day, finals = {}, []
for day in DAYS:
    exps = EXP.get(day, [])
    if not exps: continue
    E = exps[0]                                  # nearest weekly
    s0 = spot_at(day, ENTER)
    if not s0: continue
    K = round(s0 / 50) * 50
    ce = ltp(K, "CE", E, day, ENTER); pe = ltp(K, "PE", E, day, ENTER)
    if not ce or not pe: continue
    strike, credit, realized, n_rc = K, ce + pe, 0.0, 0
    series = []
    for hhmm in GRID:
        if hhmm < ENTER: continue
        sp = spot_at(day, hhmm)
        if not sp:
            if series: series.append([hhmm, series[-1][1]])
            continue
        c = ltp(strike, "CE", E, day, hhmm); p = ltp(strike, "PE", E, day, hhmm)
        cur = (c + p) if (c and p) else credit
        series.append([hhmm, round(realized + (credit - cur) * QTY)])
        if hhmm < EXITT and abs(sp - strike) >= TRIGGER_PCT / 100.0 * strike:
            realized += (credit - cur) * QTY - COST      # close current
            n_rc += 1
            K2 = round(sp / 50) * 50
            c2 = ltp(K2, "CE", E, day, hhmm); p2 = ltp(K2, "PE", E, day, hhmm)
            if c2 and p2: strike, credit = K2, c2 + p2   # re-enter at new ATM
    cF = ltp(strike, "CE", E, day, EXITT); pF = ltp(strike, "PE", E, day, EXITT)
    if cF and pF: realized += (credit - (cF + pF)) * QTY - COST
    per_day[day] = {"series": series, "final": round(realized), "recenters": n_rc, "expiry": E}
    finals.append(round(realized))

cum, curve = 0, []
for day in sorted(per_day):
    cum += per_day[day]["final"]; curve.append([day, cum])
json.dump({"version": "V1 intraday re-center", "trigger_pct": TRIGGER_PCT, "lots": LOTS, "lot": LOT,
           "per_day": per_day, "cum_curve": curve}, open(OUT / "v1_intraday_recenter.json", "w"))
a = np.array(finals)
print("V1 INTRADAY RE-CENTER %.1f%% | days=%d total=%+d mean/day=%+d median=%+d win%%=%d worst=%d best=%d avg_recenters=%.1f" % (
    TRIGGER_PCT, len(a), a.sum(), round(a.mean()), round(np.median(a)), round(100*(a>0).mean()), a.min(), a.max(),
    np.mean([per_day[d]["recenters"] for d in per_day])))
oc.close()
