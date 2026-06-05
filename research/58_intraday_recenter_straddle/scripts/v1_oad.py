"""research/58 V1 (CORRECTED) — intraday ONE-AND-DONE move-stop straddle + per-day intraday curves.
09:20 sell ATM straddle (nearest weekly). On |spot-strike|>=TRIG% -> EXIT and STAY FLAT for the day
(one-and-done, NO re-enter). Else hold to 15:15. 10 lots. Saves per-trigger JSON for the app."""
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
def dte(E, day):
    return (datetime.strptime(E, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days

GRID = []
_t = datetime.strptime("09:15", "%H:%M"); _e = datetime.strptime(EXITT, "%H:%M")
while _t <= _e:
    GRID.append(_t.strftime("%H:%M")); _t += timedelta(minutes=5)

def run(trig, dte_max=None):
    per_day, finals = {}, []
    for day in DAYS:
        exps = EXP.get(day, [])
        if not exps: continue
        E = exps[0]
        if dte_max is not None and dte(E, day) > dte_max: continue
        s0 = spot_at(day, ENTER)
        if not s0: continue
        K = round(s0 / 50) * 50
        ce = ltp(K, "CE", E, day, ENTER); pe = ltp(K, "PE", E, day, ENTER)
        if not ce or not pe: continue
        credit, realized, stopped = ce + pe, None, False
        series = []
        for hhmm in GRID:
            if hhmm < ENTER: continue
            sp = spot_at(day, hhmm)
            if not sp:
                if series: series.append([hhmm, series[-1][1]])
                continue
            c = ltp(K, "CE", E, day, hhmm); p = ltp(K, "PE", E, day, hhmm)
            cur = (c + p) if (c and p) else credit
            mtm = (credit - cur) * QTY - COST
            if not stopped:
                series.append([hhmm, round(mtm)])
                if hhmm < EXITT and abs(sp - K) >= trig / 100.0 * K:
                    realized = mtm; stopped = True            # one-and-done: book + flat
            else:
                series.append([hhmm, round(realized)])         # flat — frozen pnl
        if realized is None:                                   # not stopped -> EOD close
            cF = ltp(K, "CE", E, day, EXITT); pF = ltp(K, "PE", E, day, EXITT)
            realized = ((credit - (cF + pF)) * QTY - COST) if (cF and pF) else (series[-1][1] if series else 0)
        per_day[day] = {"series": series, "final": round(realized), "stopped": stopped,
                        "dte": dte(E, day), "expiry": E}
        finals.append(round(realized))
    cum, curve = 0, []
    for day in sorted(per_day):
        cum += per_day[day]["final"]; curve.append([day, cum])
    a = np.array(finals)
    tag = "%d" % round(trig * 10) + (("_dte%d" % dte_max) if dte_max is not None else "")
    json.dump({"version": "V1 intraday one-and-done", "trigger_pct": trig, "dte_max": dte_max,
               "lots": LOTS, "lot": LOT, "per_day": per_day, "cum_curve": curve},
              open(OUT / ("v1_oad_%s.json" % tag), "w"))
    return dict(trig=trig, dte=dte_max, n=len(a), tot=int(a.sum()), mean=int(a.mean()), med=int(np.median(a)),
                win=int(100 * (a > 0).mean()), worst=int(a.min()), best=int(a.max()),
                stop=int(100 * np.mean([per_day[d]["stopped"] for d in per_day])))

print("=== V1 ONE-AND-DONE move-stop (10 lots) ===")
print("cfg            days   total      mean/day  median   win%  stop%  worst      best")
for trig in [0.3, 0.4, 0.5, 0.7]:
    r = run(trig)
    print("%.1f%% all-DTE  %4d  %+9d  %+8d  %+7d  %3d   %3d  %+9d  %+8d" %
          (r["trig"], r["n"], r["tot"], r["mean"], r["med"], r["win"], r["stop"], r["worst"], r["best"]))
for trig in [0.4]:
    r = run(trig, dte_max=1)
    print("%.1f%% 0/1-DTE  %4d  %+9d  %+8d  %+7d  %3d   %3d  %+9d  %+8d" %
          (r["trig"], r["n"], r["tot"], r["mean"], r["med"], r["win"], r["stop"], r["worst"], r["best"]))
oc.close()
