"""Replay the afternoon straddle (sell ATM 12:15 -> buy back 15:15) on OUR recorded
chain (options_data.db), baseline and +-0.4% move-stop, so it can be cross-checked
against the user's 2.5-yr CSV on the overlapping dates (2026-04-20 .. 2026-06-05).
Sizing lot 65 x 10 = QTY 650, -160 round-trip cost (consistent with /app/straddles)."""
import os, json, sqlite3
from datetime import datetime, timedelta

ROOT = "/home/arun/quantifyd"
OPT = os.path.join(ROOT, "backtest_data", "options_data.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "ourdata_afternoon.json")
LOT, LOTS = 65, 10
QTY = LOT * LOTS
TRIG = 0.4
COST = 160

oc = sqlite3.connect(OPT)
ALLDAYS = [r[0] for r in oc.execute(
    "SELECT DISTINCT substr(snapshot_time,1,10) d FROM underlying_spot WHERE symbol='NIFTY' AND spot_price>0 ORDER BY d")]

GRID = []
t = datetime.strptime("12:15", "%H:%M")
while t <= datetime.strptime("15:15", "%H:%M"):
    GRID.append(t.strftime("%H:%M")); t += timedelta(minutes=5)

def expiries(day):
    return sorted({r[0] for r in oc.execute(
        "SELECT DISTINCT expiry_date FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND expiry_date>=?", (day, day))})

per_day = {}
for DAY in ALLDAYS:
    SP = [(tt[11:16], float(s)) for tt, s in oc.execute(
        "SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND spot_price>0 ORDER BY snapshot_time", (DAY,))]
    if not SP:
        continue
    def spot_at(h, SP=SP):
        c = [s for tt, s in SP if tt <= h]; return c[-1] if c else None
    exps = expiries(DAY)
    if not exps:
        continue
    E = exps[0]
    s0 = spot_at("12:15")
    if not s0:
        continue
    K = round(s0 / 50) * 50
    def ltp(strike, ot, h, E=E, DAY=DAY):
        r = oc.execute("SELECT ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND substr(snapshot_time,1,10)=? AND snapshot_time<=? AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
                       (E, strike, ot, DAY, DAY + "T" + h + ":59")).fetchone()
        return float(r[0]) if r and r[0] else None
    ce0 = ltp(K, "CE", "12:15"); pe0 = ltp(K, "PE", "12:15")
    if not (ce0 and pe0):
        continue
    credit = ce0 + pe0
    # baseline: exit 15:15
    cF = ltp(K, "CE", "15:15"); pF = ltp(K, "PE", "15:15")
    base = round((credit - (cF + pF)) * QTY - COST) if (cF and pF) else None
    # +0.4% SL: first grid point where |spot-K_entry|>=0.4% of entry spot
    sl_pnl = None; sl_time = "15:15"
    for h in GRID:
        sp = spot_at(h); c = ltp(K, "CE", h); p = ltp(K, "PE", h)
        if not (sp and c and p):
            continue
        if abs(sp - s0) >= TRIG / 100.0 * s0:
            sl_pnl = round((credit - (c + p)) * QTY - COST); sl_time = h; break
    if sl_pnl is None:
        sl_pnl = base; sl_time = "15:15"
    if base is None:
        continue
    per_day[DAY] = {"strike": K, "entry_spot": round(s0, 1), "credit": round(credit, 2),
                    "dte": (datetime.strptime(E, "%Y-%m-%d").date() - datetime.strptime(DAY, "%Y-%m-%d").date()).days,
                    "baseline": base, "stop_0p4": sl_pnl, "stop_time": sl_time}

days = sorted(per_day)
def tot(key):
    return sum(per_day[d][key] for d in days)
summ = {
    "days": len(days), "first": days[0] if days else None, "last": days[-1] if days else None,
    "baseline_total": tot("baseline"), "stop_total": tot("stop_0p4"),
    "baseline_win": sum(1 for d in days if per_day[d]["baseline"] > 0),
    "stop_win": sum(1 for d in days if per_day[d]["stop_0p4"] > 0),
    "stop_early": sum(1 for d in days if per_day[d]["stop_time"] != "15:15"),
}
out = {"version": "afternoon straddle 12:15->15:15 on options_data.db", "lots": LOTS, "lot": LOT,
       "summary": summ, "per_day": per_day}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"))
print("AFTERNOON STRADDLE on OUR recorded chain (%d days, %s..%s)" % (summ["days"], summ["first"], summ["last"]))
print("  baseline total  Rs %+d   (win %d/%d)" % (summ["baseline_total"], summ["baseline_win"], summ["days"]))
print("  +0.4%% SL total  Rs %+d   (win %d/%d, early-exit %d days)" % (summ["stop_total"], summ["stop_win"], summ["days"], summ["stop_early"]))
print("\n  date        K     dte  credit   baseline    +0.4%%SL  (stop@)")
for d in days:
    x = per_day[d]
    print("  %s %5d  %2d  %6.1f  %+9d  %+9d  %s" % (d, x["strike"], x["dte"], x["credit"], x["baseline"], x["stop_0p4"], x["stop_time"]))
oc.close()
