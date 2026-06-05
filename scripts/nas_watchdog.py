"""NAS Integrity Watchdog — independent read-only poller (every 5 min, 09:15-15:30 IST).
Verifies the live automation WITHOUT placing any order: DB<->broker sync, ghost/phantom legs,
rejected-order cleanup, qty/lot balance, naked-leg exit-armed, strangle balance, engine liveness.
Emits frontend/public/watchdog.json for the /app/nas "Integrity Watchdog" section (Variant A)."""
import os, json, sqlite3
from datetime import datetime
import urllib.request

ROOT = "/home/arun/quantifyd"
BASE = "http://127.0.0.1:5000"
OUT = os.path.join(ROOT, "static", "app", "watchdog.json")
VARIANTS = [("nas", "Sq-OTM"), ("nas-atm", "Sq-ATM"), ("nas-atm2", "Sq-ATM2"), ("nas-atm4", "Sq-ATM4"),
            ("nas-916-otm", "916-OTM"), ("nas-916-atm", "916-ATM"), ("nas-916-atm2", "916-ATM2"), ("nas-916-atm4", "916-ATM4")]
LOT = 65

def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=8) as r:
            return json.loads(r.read().decode())
    except Exception:
        return {}

# ---- gather --------------------------------------------------------------
states = {k: get("/api/%s/state" % k) for k, _ in VARIANTS}
ticker = get("/api/nas/ticker/status")
naked_st = {}
for fld in ("atm_naked_st", "atm4_naked_st"):
    d = ticker.get(fld)
    if isinstance(d, dict) and d.get("tradingsymbol"):
        naked_st[d["tradingsymbol"]] = d

broker, broker_ok, broker_err = {}, False, ""
try:
    import sys; sys.path.insert(0, ROOT)
    from services.kite_service import get_kite
    kite = get_kite()
    for p in (kite.positions().get("net") or []):
        if int(p.get("quantity") or 0) != 0:
            broker[p["tradingsymbol"]] = int(p["quantity"])
    rejected = {str(o["order_id"]) for o in kite.orders() if o.get("status") == "REJECTED"}
    broker_ok = True
except Exception as e:
    rejected, broker_err = set(), str(e)

# ---- per-leg pass --------------------------------------------------------
db_live_qty, naked_legs, bad_qty, dup, rej_active, imbalanced, accum = {}, [], [], [], [], [], []
for key, name in VARIANTS:
    pos = (states.get(key) or {}).get("positions", {}) or {}
    legs = (pos.get("ce") or []) + (pos.get("pe") or [])
    if len(legs) > 2:
        accum.append("%s %d legs" % (name, len(legs)))
    prem = {}
    for x in legs:
        ts = x.get("tradingsymbol"); strike = int(x.get("strike") or 0); side = x.get("instrument_type")
        qty = int(x.get("qty") or 0); sl = x.get("sl_price"); ltp = x.get("ltp")
        if (x.get("mode") or "live") == "live":
            db_live_qty[ts] = db_live_qty.get(ts, 0) + qty
        if qty == 0 or qty % LOT != 0:
            bad_qty.append("%s %d%s qty=%d" % (name, strike, side, qty))
        prem[side] = ltp
        naked = (sl or 0) >= 999999
        if naked or len(legs) == 1:
            st = naked_st.get(ts)
            if st is None:
                naked_legs.append((name, strike, side, "fail", "NO ST monitor slot — UNMONITORED"))
            elif st.get("st_value") is None:
                naked_legs.append((name, strike, side, "fail", "ST not computing (candles=%s) — no exit" % st.get("candles_completed")))
            else:
                naked_legs.append((name, strike, side, "ok", "ST armed @%.0f" % st["st_value"]))
        elif sl is None:
            naked_legs.append((name, strike, side, "fail", "no sl_price — SL not armed"))
        else:
            naked_legs.append((name, strike, side, "ok", "SL armed @%.1f" % sl))
        # rejected still active
        oid = str(x.get("kite_order_id") or "")
        if oid in rejected:
            rej_active.append("%s %d%s (order rejected)" % (name, strike, side))
    if len([1 for x in legs]) == 2 and prem.get("CE") and prem.get("PE"):
        gap = abs(prem["CE"] - prem["PE"])
        if gap > max(prem["CE"], prem["PE"]):
            imbalanced.append("%s gap %.0f" % (name, gap))

# ---- desync / ghost ------------------------------------------------------
desync, ghost = [], []
for ts, q in db_live_qty.items():
    bq = abs(broker.get(ts, 0))
    if q != bq:
        (ghost if bq == 0 else desync).append("%s: DB %d vs broker %d" % (ts, q, bq))

# ---- assemble checks (Variant A) -----------------------------------------
def chk(check, scope, status, detail):
    return {"check": check, "scope": scope, "status": status, "detail": detail}

n_naked = len(naked_legs); n_naked_fail = sum(1 for x in naked_legs if x[3] == "fail")
naked_detail = " · ".join("%s %d%s — %s" % (n, s, sd, d) for n, s, sd, st, d in naked_legs) or "no naked legs"

groups = [
    {"name": "Position integrity", "checks": [
        chk("DB ↔ broker position sync", "all live legs", "ok" if not desync else "fail",
            "all live-leg qty match broker" if not desync else "; ".join(desync)),
        chk("Ghost / phantom legs", "all variants", "ok" if not ghost else "fail",
            "no DB-active leg flat at broker" if not ghost else "; ".join(ghost)),
        chk("Rejected-order cleanup", "all variants", "ok" if not rej_active else "fail",
            "0 rejected orders left ACTIVE" if not rej_active else "; ".join(rej_active)),
        chk("Qty / lot-size balance", "all legs", "ok" if not bad_qty else "fail",
            "every leg qty is a clean lot multiple" if not bad_qty else "; ".join(bad_qty)),
    ]},
    {"name": "Exit / stop arming", "checks": [
        chk("Naked leg exit armed", "%d naked leg%s" % (n_naked, "" if n_naked == 1 else "s"),
            "ok" if n_naked_fail == 0 else "fail", naked_detail if n_naked else "no naked legs to arm"),
    ]},
    {"name": "Adjustment / structure", "checks": [
        chk("Strangle leg balance", "2-leg variants", "ok" if not imbalanced else "warn",
            "all strangles CE/PE balanced" if not imbalanced else "; ".join(imbalanced)),
        chk("Accumulation / extra legs", "all variants", "ok" if not accum else "warn",
            "no variant carries >1 strangle" if not accum else "; ".join(accum)),
    ]},
    {"name": "Engine liveness", "checks": [
        chk("Ticker / websocket", "option feed", "ok" if ticker.get("running") else "warn",
            "subscribed & ticking" if ticker.get("running") else "ticker status unavailable"),
        chk("Kite / broker connection", "broker API", "ok" if broker_ok else "fail",
            "positions & orders reachable" if broker_ok else "Kite error: " + broker_err[:60]),
    ]},
]
ok = warn = fail = 0
for g in groups:
    for c in g["checks"]:
        ok += c["status"] == "ok"; warn += c["status"] == "warn"; fail += c["status"] == "fail"
out = {"polled_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
       "summary": {"ok": ok, "warn": warn, "fail": fail}, "groups": groups}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"))
print("watchdog: %d OK / %d warn / %d fail -> %s" % (ok, warn, fail, OUT))
for g in groups:
    for c in g["checks"]:
        print("  [%s] %-26s %s" % (c["status"].upper(), c["check"], c["detail"][:70]))
