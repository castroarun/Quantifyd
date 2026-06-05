"""Straddle paper LIVE logger (10 lots) — every 5 min, 09:15-15:30 IST. Reads the live options
recorder, tracks today's intraday P&L for V1 (0/1-DTE intraday one-and-done) + V2 (positional
bi-weekly, carried), records each day, and writes static/app/straddles_live.json for the
/app/straddles live section. Paper only — places no orders."""
import os, json, sqlite3
from datetime import datetime, date, timedelta

ROOT = "/home/arun/quantifyd"
OPT = os.path.join(ROOT, "backtest_data", "options_data.db")
OUT = os.path.join(ROOT, "static", "app", "straddles_live.json")
HIST = os.path.join(ROOT, "backtest_data", "straddle_live_history.json")   # per-day finals
V2ST = os.path.join(ROOT, "backtest_data", "straddle_v2_state.json")        # carried V2 position
LOT, LOTS = 65, 10
QTY = LOT * LOTS
V1_TRIG = 0.4
DAY = date.today().isoformat()

oc = sqlite3.connect(OPT)
def expiries(day):
    return sorted({r[0] for r in oc.execute(
        "SELECT DISTINCT expiry_date FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND expiry_date>=?", (day, day))})
SP = [(t[11:16], float(s)) for t, s in oc.execute(
    "SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND spot_price>0 ORDER BY snapshot_time", (DAY,))]
def spot_at(hhmm):
    c = [s for t, s in SP if t <= hhmm]
    return c[-1] if c else (SP[0][1] if SP else None)
def ltp(strike, ot, E, hhmm):
    r = oc.execute("SELECT ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND substr(snapshot_time,1,10)=? AND snapshot_time<=? AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",
                   (E, strike, ot, DAY, DAY + "T" + hhmm + ":59")).fetchone()
    return float(r[0]) if r and r[0] else None
def dte(E, day=DAY):
    return (datetime.strptime(E, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days
def now_hhmm():
    return min(datetime.now().strftime("%H:%M"), "15:15")
GRID = []
t = datetime.strptime("09:20", "%H:%M")
while t <= datetime.strptime("15:15", "%H:%M"):
    GRID.append(t.strftime("%H:%M")); t += timedelta(minutes=5)
NOW = now_hhmm()
exps = expiries(DAY)

# ---- V1: intraday one-and-done, 0/1-DTE only -----------------------------
v1 = {"status": "idle", "detail": "", "pnl_now": 0, "series": []}
if exps:  # live: track V1 every day (edge is 0/1-DTE; non-0/1 shown for tracking)
    E = exps[0]; s0 = spot_at("09:20")
    if s0:
        K = round(s0 / 50) * 50
        ce0 = ltp(K, "CE", E, "09:20"); pe0 = ltp(K, "PE", E, "09:20")
        if ce0 and pe0:
            credit = ce0 + pe0; stopped = False; series = []; final = None
            for hhmm in GRID:
                if hhmm > NOW: break
                sp = spot_at(hhmm); c = ltp(K, "CE", E, hhmm); p = ltp(K, "PE", E, hhmm)
                if not (sp and c and p): continue
                mtm = round((credit - (c + p)) * QTY - 160)
                if not stopped:
                    series.append([hhmm, mtm])
                    if abs(sp - K) >= V1_TRIG / 100.0 * K:
                        final = mtm; stopped = True
                else:
                    series.append([hhmm, final])
            v1 = {"status": "stopped" if stopped else "open", "detail": "%dCE+%dPE (exp %s, %d-DTE)" % (K, K, E, dte(E)),
                  "strike": K, "expiry": E, "credit": round(credit, 1),
                  "pnl_now": (final if final is not None else (series[-1][1] if series else 0)), "series": series}
elif exps:
    v1 = {"status": "idle", "detail": "nearest %s is %d-DTE — V1 trades 0/1-DTE only (Mon/Tue)" % (exps[0], dte(exps[0])),
          "pnl_now": 0, "series": []}

# ---- V2: positional bi-weekly, carried -----------------------------------
v2state = {}
try:
    v2state = json.load(open(V2ST))
except Exception:
    v2state = {}
need_entry = (not v2state) or (v2state.get("expiry") and dte(v2state["expiry"]) < 1)
if need_entry and len(exps) > 1:
    E = exps[1]; s0 = spot_at("09:20")
    if s0:
        K = round(s0 / 50) * 50
        ce0 = ltp(K, "CE", E, "09:20"); pe0 = ltp(K, "PE", E, "09:20")
        if ce0 and pe0:
            v2state = {"strike": K, "expiry": E, "credit": round(ce0 + pe0, 2), "entry_day": DAY}
            json.dump(v2state, open(V2ST, "w"))

v2 = {"status": "flat", "detail": "no position", "pnl_now": 0, "series": []}
if v2state:
    K = v2state["strike"]; E = v2state["expiry"]; credit = v2state["credit"]
    series = []
    for hhmm in GRID:
        if hhmm > NOW: break
        c = ltp(K, "CE", E, hhmm); p = ltp(K, "PE", E, hhmm)
        if c and p:
            series.append([hhmm, round((credit - (c + p)) * QTY - 160)])
    v2 = {"status": "open", "detail": "%dCE+%dPE (exp %s, %d-DTE) · credit Rs%.1f · entered %s" % (
              K, K, E, dte(E), credit, v2state.get("entry_day")),
          "strike": K, "expiry": E, "credit": credit, "entry_day": v2state.get("entry_day"),
          "pnl_now": (series[-1][1] if series else 0), "series": series}

# ---- assemble + write ----------------------------------------------------
def stats(s):
    ys = [v for _, v in s.get("series", [])]
    return {"low": min(ys) if ys else 0, "high": max(ys) if ys else 0}
v1.update(stats(v1)); v2.update(stats(v2))
out = {"updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "day": DAY, "lots": LOTS, "v1": v1, "v2": v2}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"))

# per-day history (final P&L each day) for the running curve
hist = {}
try:
    hist = json.load(open(HIST))
except Exception:
    hist = {"v1": {}, "v2": {}}
hist.setdefault("v1", {})[DAY] = v1["pnl_now"]
hist.setdefault("v2", {})[DAY] = v2["pnl_now"]
json.dump(hist, open(HIST, "w"))
print("straddle-live: V1 %s %+d | V2 %s %+d -> %s" % (v1["status"], v1["pnl_now"], v2["status"], v2["pnl_now"], OUT))
oc.close()
