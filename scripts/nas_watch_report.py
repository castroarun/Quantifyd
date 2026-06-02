#!/usr/bin/env python3
"""NAS live health report — one clean, scannable status pass for the watch loop.

Run:  ./venv/bin/python3 scripts/nas_watch_report.py [--since-min N]

Checks, per cycle:
  - new trades (entries / SL / SL-BOTH / rolls / re-entries / ST-exits / desync)
  - per-variant positions + leg-level BALANCE (CE vs PE premium gap)
  - UNMONITORED legs (no tick feed and not poll-covered; naked leg with no ST)
  - MONITORING ARMS intact: SL stops + adjustment monitors + naked-ST computing
  - app<->broker DESYNC (DB active vs broker net short, per symbol)
  - feed health + P&L (realized / open / NET)
Emits a STATUS line + a consolidated WARNINGS block (exit code 1 if any warning).
"""
import json
import subprocess
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime

BASE = "http://127.0.0.1:5000"
VARIANTS = [
    ("nas", "Sq-OTM", "OTM"), ("nas-atm", "Sq-ATM", "ATM"),
    ("nas-atm2", "Sq-ATM2", "ATM2"), ("nas-atm4", "Sq-ATM4", "ATM4"),
    ("nas-916-otm", "916-OTM", "OTM"), ("nas-916-atm", "916-ATM", "ATM"),
    ("nas-916-atm2", "916-ATM2", "ATM2"), ("nas-916-atm4", "916-ATM4", "ATM4"),
]
POLL_COVERED = lambda key: key.startswith("nas-916")   # 916 family has the 10s SL poll
BALANCE_GAP_WARN = 25.0     # CE/PE premium gap that flags an imbalanced strangle
NET_WARN = -15000.0
SINCE_MIN = 12
for i, a in enumerate(sys.argv):
    if a == "--since-min" and i + 1 < len(sys.argv):
        SINCE_MIN = int(sys.argv[i + 1])

warnings = []


def W(msg):
    warnings.append(msg)


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=8) as r:
            return json.load(r)
    except Exception as e:
        return {"_err": str(e)}


def fnum(v, d="-"):
    return ("%.1f" % v) if isinstance(v, (int, float)) else d


# ---- gather -----------------------------------------------------------------
ticker = get("/api/nas/ticker/status")
states = {k: get("/api/%s/state" % k) for k, _, _ in VARIANTS}

# ticker monitored symbols -> current premium (across all families)
mon = {}
for fld in ("option_legs", "atm_option_legs", "atm2_option_legs", "atm4_option_legs"):
    for leg in (ticker.get(fld) or []):
        mon[leg.get("tradingsymbol")] = leg.get("current_premium")
naked_st = {}
for fld in ("atm_naked_st", "atm4_naked_st"):
    d = ticker.get(fld) or {}
    if d.get("active") and d.get("tradingsymbol"):
        naked_st[d["tradingsymbol"]] = d  # has st_value, candles_completed


# ---- broker (for desync) ----------------------------------------------------
broker = {}
try:
    sys.path.insert(0, ".")
    from services.kite_service import get_kite
    for p in (get_kite().positions().get("net") or []):
        t = str(p.get("tradingsymbol") or "")
        if t.startswith("NIFTY") and (t.endswith("CE") or t.endswith("PE")):
            broker[t] = int(p.get("quantity") or 0)
    broker_ok = True
except Exception as e:
    broker_ok = False
    broker_err = str(e)

_kite = None
_prem_cache = {}
def live_prem(ts):
    """Live LTP fallback for legs the ticker isn't feeding (single-slot gap)."""
    global _kite
    if ts in _prem_cache:
        return _prem_cache[ts]
    try:
        if _kite is None:
            from services.kite_service import get_kite as _gk
            _kite = _gk()
        v = _kite.ltp(["NFO:" + ts]).get("NFO:" + ts, {}).get("last_price")
    except Exception:
        v = None
    _prem_cache[ts] = v
    return v


# ---- recent trades ----------------------------------------------------------
def recent_events():
    try:
        out = subprocess.run(
            ["journalctl", "-u", "quantifyd", "--since", "%d min ago" % SINCE_MIN,
             "--no-pager", "-o", "cat"],
            capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    keys = ("Strangle #", "Forward snap", "SL HIT", "SL-BOTH", "Closing BOTH",
            "Re-enter", "ROLLED", "SECOND SL", "ST EXIT", "ST_EXIT",
            "Closed", "NAS-DESYNC", "Daily order limit", "TokenException")
    skip = ("TICK ADJ", "SL TICK", "SL handler", "apscheduler", "not found",
            "reconcile", "scheduled jobs")
    ev = []
    for ln in out.splitlines():
        if any(k in ln for k in keys) and not any(s in ln for s in skip):
            ev.append(ln.split("gunicorn", 1)[-1].split(":", 2)[-1].strip()
                      if "gunicorn" in ln else ln.strip())
    return ev[-12:]


# ---- per-variant analysis ---------------------------------------------------
rows = []
tot_r = tot_o = 0.0
naked_count_by_family = defaultdict(int)
db_active_qty = defaultdict(int)

for key, name, fam in VARIANTS:
    p = (states.get(key) or {}).get("positions", {}) or {}
    legs = []
    for side in ("ce", "pe"):
        for x in p.get(side, []) or []:
            legs.append(x)
    act = len(legs)
    # realized / open
    r = sum(((x.get("entry_price") or 0) - (x.get("exit_price") or 0)) * (x.get("qty") or 0)
            for x in p.get("closed_today", []) or []
            if x.get("entry_price") is not None and x.get("exit_price") is not None)
    o = sum(((x.get("entry_price") or 0) - ((x.get("ltp") if x.get("ltp") is not None else x.get("entry_price")) or 0)) * (x.get("qty") or 0)
            for x in legs)
    tot_r += r; tot_o += o

    if act == 0:
        rows.append((name, "flat", "", ""))
        continue
    if act > 2:
        W("%s ACCUMULATION: %d active legs (>1 strangle)" % (name, act))

    # leg descriptors + monitoring + balance
    descs, mon_flags = [], []
    prem = {}
    for x in legs:
        ts = x.get("tradingsymbol"); strike = int(x.get("strike") or 0)
        side = x.get("instrument_type"); sl = x.get("sl_price")
        ltp = x.get("ltp"); naked = (sl or 0) >= 999999
        db_active_qty[ts] += (x.get("qty") or 0)
        pv = mon.get(ts)
        if pv is None:
            pv = ltp if ltp is not None else live_prem(ts)
        prem[side] = pv
        # ---- monitoring-arm check for this leg ----
        if naked:
            naked_count_by_family[(key, fam)] += 1
            st = naked_st.get(ts)
            if st is None:
                W("%s %d%s NAKED but NO ST monitor slot -> UNMONITORED adjustment" % (name, strike, side))
                mon_flags.append("ST?")
            elif st.get("st_value") is None:
                W("%s %d%s naked ST not computing (candles=%s, st_value=None) -> exit not armed" % (name, strike, side, st.get("candles_completed")))
                mon_flags.append("ST!none")
            else:
                mon_flags.append("ST=%.0f" % st["st_value"])
        else:
            if sl is None:
                W("%s %d%s has NO sl_price -> SL not armed" % (name, strike, side))
                mon_flags.append("noSL")
            tick_fed = mon.get(ts) is not None
            if not tick_fed and not POLL_COVERED(key):
                W("%s %d%s tick-feed=None AND no 10s poll (squeeze) -> SL UNMONITORED" % (name, strike, side))
                mon_flags.append("UNMON")
            elif not tick_fed:
                mon_flags.append("poll")
            else:
                mon_flags.append("tick")
        descs.append("%d%s" % (strike, side))

    # balance (2-leg strangle only)
    bal = ""
    if act == 2 and "CE" in prem and "PE" in prem and prem["CE"] and prem["PE"]:
        gap = abs(prem["CE"] - prem["PE"])
        bal = "gap %.0f%s" % (gap, " OK" if gap <= BALANCE_GAP_WARN else " IMBAL")
        if gap > BALANCE_GAP_WARN:
            W("%s imbalanced: %s CE=%.0f PE=%.0f gap %.0f" % (name, "/".join(descs), prem["CE"], prem["PE"], gap))
    elif act == 1:
        bal = "naked"
    rows.append((name, " ".join(descs) + " (%d)" % act, bal,
                 " ".join(dict.fromkeys(mon_flags))))

# single-slot exposure: more naked legs in a family than the 1 ST slot can hold
fam_naked = defaultdict(int)
for (key, fam), c in naked_count_by_family.items():
    fam_naked[fam] += c
for fam, c in fam_naked.items():
    if fam in ("ATM", "ATM4") and c > 1:
        W("%s family has %d naked legs but only ONE ST slot -> %d unmonitored (single-slot bug)" % (fam, c, c - 1))

# ---- desync -----------------------------------------------------------------
desync = []
if broker_ok:
    for ts, dbq in db_active_qty.items():
        if dbq <= 0:
            continue
        bq = broker.get(ts, 0); bshort = -bq if bq < 0 else 0
        if dbq > bshort:
            desync.append("%s: app %d short vs broker %d" % (ts, dbq, bshort))
            W("DESYNC %s: app shows %d short, broker has %d -> leg closed externally" % (ts, dbq, bshort))

# ---- feed -------------------------------------------------------------------
conn = ticker.get("is_connected"); cndl = (ticker.get("current_candle") or {}).get("date")
if not conn:
    W("FEED disconnected (ticker is_connected=False)")
net = tot_r + tot_o
if net <= NET_WARN:
    W("NET P&L %.0f <= %.0f" % (net, NET_WARN))

# ---- render -----------------------------------------------------------------
status = "HEALTHY" if not warnings else ("ATTENTION (%d)" % len(warnings))
now = datetime.now().strftime("%a %d %b %H:%M:%S IST")
L = []
L.append("=" * 64)
L.append("NAS WATCH  -  %s" % now)
L.append("STATUS: %s   |   NET %+.0f  (real %+.0f / open %+.0f)" % (status, net, tot_r, tot_o))
L.append("FEED: %s | NIFTY %s | candle %s" % ("UP" if conn else "DOWN", fnum(ticker.get("last_ltp")), cndl))
L.append("=" * 64)
ev = recent_events()
L.append("TRADES (last %dm): %s" % (SINCE_MIN, "" if ev else "none"))
for e in ev:
    L.append("   - %s" % e[:90])
L.append("")
L.append("%-10s %-22s %-12s %s" % ("VARIANT", "LEGS", "BALANCE", "MONITORING"))
L.append("-" * 64)
for name, legs, bal, monf in rows:
    L.append("%-10s %-22s %-12s %s" % (name, legs, bal, monf))
L.append("")
L.append("ARMS: SL stops + adjustment monitors  |  DESYNC: %s" %
         ("IN SYNC" if (broker_ok and not desync) else ("%d MISMATCH" % len(desync) if broker_ok else "broker N/A")))
if warnings:
    L.append("")
    L.append("WARNINGS (%d):" % len(warnings))
    for w in warnings:
        L.append("   ! %s" % w)
else:
    L.append("ALL CHECKS GREEN: legs monitored, stops armed, balanced, in sync.")
L.append("=" * 64)
print("\n".join(L))
sys.exit(1 if warnings else 0)
