"""research/58 V1 — INTRADAY RE-CENTER straddle, multi-trigger comparison + per-day intraday curves.
09:20 sell ATM straddle (nearest weekly); on |spot-strike|>=TRIG% close+re-sell fresh ATM (replace);
flatten 15:15. 10 lots. Saves per-trigger JSON (per-day 5-min PnL series) for the app."""
import sqlite3, json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

ROOT = Path("/home/arun/quantifyd")
OUT = ROOT / "research/58_intraday_recenter_straddle/results"; OUT.mkdir(parents=True, exist_ok=True)
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; LOTS = 10; QTY = LOT * LOTS; COST = 2 * 80
ENTER, EXITT = "09:20", "15:15"

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
_t = datetime.strptime("09:15", "%H:%M"); _e = datetime.strptime(EXITT, "%H:%M")
while _t <= _e:
    GRID.append(_t.strftime("%H:%M")); _t += timedelta(minutes=5)

def run(trig):
    per_day, finals = {}, []
    for day in DAYS:
        exps = EXP.get(day, [])
        if not exps: continue
        E = exps[0]; s0 = spot_at(day, ENTER)
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
            if hhmm < EXITT and abs(sp - strike) >= trig / 100.0 * strike:
                realized += (credit - cur) * QTY - COST; n_rc += 1
                K2 = round(sp / 50) * 50
                c2 = ltp(K2, "CE", E, day, hhmm); p2 = ltp(K2, "PE", E, day, hhmm)
                if c2 and p2: strike, credit = K2, c2 + p2
        cF = ltp(strike, "CE", E, day, EXITT); pF = ltp(strike, "PE", E, day, EXITT)
        if cF and pF: realized += (credit - (cF + pF)) * QTY - COST
        per_day[day] = {"series": series, "final": round(realized), "recenters": n_rc, "expiry": E}
        finals.append(round(realized))
    cum, curve = 0, []
    for day in sorted(per_day):
        cum += per_day[day]["final"]; curve.append([day, cum])
    a = np.array(finals)
    json.dump({"version": "V1 intraday re-center", "trigger_pct": trig, "lots": LOTS, "lot": LOT,
               "per_day": per_day, "cum_curve": curve}, open(OUT / ("v1_intraday_%d.json" % round(trig * 10)), "w"))
    return dict(trig=trig, n=len(a), tot=int(a.sum()), mean=int(a.mean()), med=int(np.median(a)),
                win=int(100 * (a > 0).mean()), worst=int(a.min()), best=int(a.max()),
                rc=round(np.mean([per_day[d]["recenters"] for d in per_day]), 1))

print("=== V1 intraday re-center: trigger comparison (10 lots) ===")
print("trig%  days   total      mean/day  median   win%  worst      best     avg_rc")
for tg in [0.4, 0.7, 1.0, 1.5, 2.0]:
    r = run(tg)
    print("%4.1f  %4d  %+9d  %+8d  %+7d  %3d  %+9d  %+8d  %.1f" %
          (r["trig"], r["n"], r["tot"], r["mean"], r["med"], r["win"], r["worst"], r["best"], r["rc"]))
oc.close()
