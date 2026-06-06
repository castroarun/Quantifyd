"""V1 intraday one-and-done DAILY JOURNEYS for ALL recorded days. Replays the V1
short-straddle (sell ATM 09:20, exit ONCE on a +-0.4% underlying move, else hold to
15:15) for every day in options_data.db and writes static/app/straddles/v1_daily.json:
each day's intraday P&L series + exit{time,pnl} + low/high/final/dte, for the per-day
journey browser on /app/straddles. Same sizing as the live card: lot 65 x 10 = QTY 650,
-160 round-trip cost baked into the running mark (so numbers reconcile with the live card)."""
import os, json, sqlite3
from datetime import datetime, timedelta

ROOT = "/home/arun/quantifyd"
OPT = os.path.join(ROOT, "backtest_data", "options_data.db")
OUT = os.path.join(ROOT, "static", "app", "straddles", "v1_daily.json")
LOT, LOTS = 65, 10
QTY = LOT * LOTS
V1_TRIG = 0.4
COST = 160

oc = sqlite3.connect(OPT)
ALLDAYS = [r[0] for r in oc.execute(
    "SELECT DISTINCT substr(snapshot_time,1,10) d FROM underlying_spot WHERE symbol='NIFTY' AND spot_price>0 ORDER BY d")]

GRID = []
t = datetime.strptime("09:20", "%H:%M")
while t <= datetime.strptime("15:15", "%H:%M"):
    GRID.append(t.strftime("%H:%M")); t += timedelta(minutes=5)

def dte(E, day):
    return (datetime.strptime(E, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days

per_day = {}
for DAY in ALLDAYS:
    SP = [(tt[11:16], float(s)) for tt, s in oc.execute(
        "SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND spot_price>0 ORDER BY snapshot_time", (DAY,))]
    if not SP:
        continue
    def spot_at(h, SP=SP):
        c = [s for tt, s in SP if tt <= h]; return c[-1] if c else None
    exps = sorted({r[0] for r in oc.execute(
        "SELECT DISTINCT expiry_date FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND expiry_date>=?", (DAY, DAY))})
    if not exps:
        continue
    E = exps[0]
    s0 = spot_at("09:20")
    if not s0:
        continue
    K = round(s0 / 50) * 50
    def ltp(strike, ot, h, E=E, DAY=DAY):
        r = oc.execute("SELECT ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND substr(snapshot_time,1,10)=? AND snapshot_time<=? AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
                       (E, strike, ot, DAY, DAY + "T" + h + ":59")).fetchone()
        return float(r[0]) if r and r[0] else None
    ce0 = ltp(K, "CE", "09:20"); pe0 = ltp(K, "PE", "09:20")
    if not (ce0 and pe0):
        continue
    credit = ce0 + pe0; stopped = False; series = []; final = None; exit_t = None
    for h in GRID:
        sp = spot_at(h); c = ltp(K, "CE", h); p = ltp(K, "PE", h)
        if not (sp and c and p):
            continue
        mtm = round((credit - (c + p)) * QTY - COST)
        if not stopped:
            series.append([h, mtm])
            if abs(sp - K) >= V1_TRIG / 100.0 * K:
                final = mtm; stopped = True; exit_t = h
        else:
            series.append([h, final])
    if len(series) < 2:
        continue
    ys = [v for _, v in series]
    per_day[DAY] = {
        "series": series,
        "exit": ({"time": exit_t, "pnl": final} if stopped else None),
        "final": final if final is not None else ys[-1],
        "stopped": stopped, "low": min(ys), "high": max(ys),
        "dte": dte(E, DAY), "expiry": E, "strike": K, "credit": round(credit, 2)}

out = {"version": "V1 intraday one-and-done daily journeys (all recorded days)",
       "trigger_pct": V1_TRIG, "lots": LOTS, "lot": LOT,
       "days": sorted(per_day), "per_day": per_day}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"))
print("wrote %d day journeys -> %s" % (len(per_day), OUT))
for d in sorted(per_day):
    x = per_day[d]; e = x["exit"]
    print("  %s dte%d %-4s low%+7d high%+7d final%+7d %s" % (
        d, x["dte"], "STOP" if x["stopped"] else "hold", x["low"], x["high"], x["final"],
        ("exit " + e["time"]) if e else ""))
oc.close()
