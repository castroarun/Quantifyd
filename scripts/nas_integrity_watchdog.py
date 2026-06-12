"""NAS Integrity Watchdog — independent read-only poller (every 5 min, 09:15-15:30 IST).
Verifies the live automation WITHOUT placing any order, and emits static/app/watchdog.json for the
/app/nas "Integrity Watchdog" section. Emails on a NEW fail (de-duped). Distinct from the pipeline
heartbeat in services/nas_watchdog.py."""
import os, json, hashlib
from datetime import datetime
import urllib.request

ROOT = "/home/arun/quantifyd"
BASE = "http://127.0.0.1:5000"
OUT = os.path.join(ROOT, "static", "app", "watchdog.json")
STATE = os.path.join(ROOT, "backtest_data", "watchdog_lastfail.json")
VARIANTS = [("nas", "Sq-OTM"), ("nas-atm", "Sq-ATM"), ("nas-atm2", "Sq-ATM2"), ("nas-atm4", "Sq-ATM4"),
            ("nas-916-otm", "916-OTM"), ("nas-916-atm", "916-ATM"), ("nas-916-atm2", "916-ATM2"), ("nas-916-atm4", "916-ATM4")]
LOT = 65
MAXLOSS_WARN, MAXLOSS_FAIL = -20000, -40000     # book day-P&L thresholds (10-lot scale)
MARGIN_BUFFER = 100000                            # warn if available new-margin under this
import sys; sys.path.insert(0, ROOT)

def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=8) as r:
            return json.loads(r.read().decode())
    except Exception:
        return {}

states = {k: get("/api/%s/state" % k) for k, _ in VARIANTS}
ticker = get("/api/nas/ticker/status")
naked_st = {}
for fld in ("atm_naked_st", "atm4_naked_st"):
    d = ticker.get(fld)
    if isinstance(d, dict) and d.get("tradingsymbol"):
        naked_st[d["tradingsymbol"]] = d

broker, broker_ok, broker_err, rejected, avail_margin, working = {}, False, "", set(), None, []
try:
    from services.kite_service import get_kite
    kite = get_kite()
    for p in (kite.positions().get("net") or []):
        if int(p.get("quantity") or 0) != 0:
            broker[p["tradingsymbol"]] = int(p["quantity"])
    for o in kite.orders():
        if o.get("status") == "REJECTED":
            rejected.add(str(o["order_id"]))
        if o.get("status") in ("OPEN", "TRIGGER PENDING", "PUT ORDER REQ RECEIVED", "VALIDATION PENDING"):
            working.append("%s %s %s" % (o.get("transaction_type"), o.get("tradingsymbol"), o.get("status")))
    try:
        eq = kite.margins().get("equity", {})
        avail_margin = (eq.get("available", {}) or {}).get("live_balance")
    except Exception:
        avail_margin = None
    broker_ok = True
except Exception as e:
    broker_err = str(e)

db_live_qty, naked_legs, bad_qty, rej_active, imbalanced, accum = {}, [], [], [], [], []
day_real = day_open = 0.0
for key, name in VARIANTS:
    pos = (states.get(key) or {}).get("positions", {}) or {}
    legs = (pos.get("ce") or []) + (pos.get("pe") or [])
    for x in (pos.get("closed_today") or []):
        if x.get("entry_price") is not None and x.get("exit_price") is not None:
            day_real += ((x["entry_price"] or 0) - (x["exit_price"] or 0)) * (x.get("qty") or 0)
    if len(legs) > 2:
        accum.append("%s %d legs" % (name, len(legs)))
    prem = {}
    for x in legs:
        ts = x.get("tradingsymbol"); strike = int(x.get("strike") or 0); side = x.get("instrument_type")
        qty = int(x.get("qty") or 0); sl = x.get("sl_price"); ltp = x.get("ltp")
        day_open += ((x.get("entry_price") or 0) - (ltp if ltp is not None else x.get("entry_price") or 0)) * qty
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
        if str(x.get("kite_order_id") or "") in rejected:
            rej_active.append("%s %d%s (order rejected)" % (name, strike, side))
    if len(legs) == 2 and prem.get("CE") and prem.get("PE") and abs(prem["CE"] - prem["PE"]) > max(prem["CE"], prem["PE"]):
        imbalanced.append("%s gap %.0f" % (name, abs(prem["CE"] - prem["PE"])))

desync, ghost = [], []
for ts, q in db_live_qty.items():
    bq = abs(broker.get(ts, 0))
    if q != bq:
        (ghost if bq == 0 else desync).append("%s: DB %d vs broker %d" % (ts, q, bq))

day_pnl = round(day_real + day_open)
n_open = sum(len((states.get(k) or {}).get("positions", {}).get("ce", []) or []) +
             len((states.get(k) or {}).get("positions", {}).get("pe", []) or []) for k, _ in VARIANTS)
hhmm = datetime.now().strftime("%H:%M")

def chk(check, scope, status, detail):
    return {"check": check, "scope": scope, "status": status, "detail": detail}

n_naked = len(naked_legs); n_naked_fail = sum(1 for x in naked_legs if x[3] == "fail")
naked_detail = " · ".join("%s %d%s — %s" % (n, s, sd, d) for n, s, sd, st, d in naked_legs) or "no naked legs"
pnl_status = "fail" if day_pnl <= MAXLOSS_FAIL else ("warn" if day_pnl <= MAXLOSS_WARN else "ok")
margin_status = "ok" if (avail_margin is None or avail_margin >= MARGIN_BUFFER) else "warn"
eod_pending = n_open > 0 and hhmm >= "15:10"
eod_status = "fail" if (n_open > 0 and hhmm >= "15:31") else ("warn" if eod_pending else "ok")

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
        chk("Stale / working exit order", "broker orderbook", "ok" if not working else "warn",
            "no working orders pending" if not working else "; ".join(working[:4])),
    ]},
    {"name": "Adjustment / structure", "checks": [
        chk("Strangle leg balance", "2-leg variants", "ok" if not imbalanced else "warn",
            "all strangles CE/PE balanced" if not imbalanced else "; ".join(imbalanced)),
        chk("Accumulation / extra legs", "all variants", "ok" if not accum else "warn",
            "no variant carries >1 strangle" if not accum else "; ".join(accum)),
    ]},
    {"name": "Risk / capital", "checks": [
        chk("Day P&L vs max-loss", "whole book", pnl_status,
            "day P&L Rs%+d (warn <%d, fail <%d)" % (day_pnl, MAXLOSS_WARN, MAXLOSS_FAIL)),
        chk("Margin headroom", "F&O margin", margin_status,
            ("available Rs%s" % (("%.0f" % avail_margin) if avail_margin is not None else "n/a")) +
            ("" if margin_status == "ok" else " — under Rs%d buffer" % MARGIN_BUFFER)),
        chk("EOD square-off", "live positions", eod_status,
            "no open positions" if n_open == 0 else ("%d legs open" % n_open) +
            (" — past 15:30, squareoff may have failed" if eod_status == "fail" else
             (" — squareoff window" if eod_pending else " — intraday, ok"))),
    ]},
    {"name": "Engine liveness", "checks": [
        chk("Ticker / websocket", "option feed", "ok" if ticker.get("is_running") else "warn",
            "subscribed & ticking" if ticker.get("is_running") else "ticker status unavailable"),
        chk("Kite / broker connection", "broker API", "ok" if broker_ok else "fail",
            "positions, orders & margins reachable" if broker_ok else "Kite error: " + broker_err[:60]),
    ]},
]
ok = warn = fail = 0
fails = []
for g in groups:
    for c in g["checks"]:
        ok += c["status"] == "ok"; warn += c["status"] == "warn"; fail += c["status"] == "fail"
        if c["status"] == "fail":
            fails.append("%s — %s" % (c["check"], c["detail"]))
out = {"polled_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
       "summary": {"ok": ok, "warn": warn, "fail": fail}, "groups": groups}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"))
print("integrity-watchdog: %d OK / %d warn / %d fail" % (ok, warn, fail))

# ---- email on a NEW fail (de-duped by fail signature) --------------------
if fails:
    sig = hashlib.md5("\n".join(sorted(fails)).encode()).hexdigest()
    last = ""
    try:
        last = json.load(open(STATE)).get("sig", "")
    except Exception:
        pass
    if sig != last:
        try:
            from services.nas_watchdog import _send_alert
            body = "<h3>NAS Integrity Watchdog — %d FAIL</h3><ul>%s</ul><p>polled %s</p>" % (
                fail, "".join("<li>%s</li>" % f for f in fails), out["polled_at"])
            _send_alert("[NAS Watchdog] %d integrity FAIL" % fail, body)
            print("  emailed %d fails" % fail)
        except Exception as e:
            print("  email failed:", e)
        json.dump({"sig": sig}, open(STATE, "w"))
else:
    try:
        json.dump({"sig": ""}, open(STATE, "w"))
    except Exception:
        pass
