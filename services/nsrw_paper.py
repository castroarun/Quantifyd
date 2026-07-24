"""NSR-W v1.2 — NIFTY rules-based weekly strangle, LIVE PAPER book (research/90 G5).

Locked spec (do NOT tune here — see research/90_nifty_strangle_rules/, backtest
card /app/backtest/nifty-strangle-rules-research90):
  ENTRY  Monday 15:14 — sell CE + PE on NEXT week's expiry (cal DTE 6-12),
         strikes nearest Rs30 mid premium each, fill at BID. Fixed 10 lots.
  STOP   GTT-style 2.0x per leg on LTP, fill at ASK. ONE roll-away per cycle:
         first stopped side re-sold at the Rs30 rule from current spot; a
         post-roll stop closes that leg only; the survivor rides its own stop.
  PT     50% of original combined credit (on combined MID incl. realized).
  RECENTER  15:16 daily: any live leg >= 1.5x its credit -> close ALL at ask,
         re-sell a fresh Rs30 strangle (once per cycle).
  TIME   cal DTE <= 1 -> close all at 15:16.
PAPER ONLY. Accounting in cash flows; 1 pt = Rs{lot_qty}. No margin model.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger("nsrw_paper")

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "backtest_data" / "nsrw_paper.json"
OPTDB = ROOT / "backtest_data" / "options_data.db"

CFG = {
    "underlying": "NIFTY",
    "spot_key": "NSE:NIFTY 50",
    "lots": 10,
    "target": 30.0,
    "stop_mult": 2.0,
    "pt_frac": 0.5,
    "recenter_mult": 1.5,
    "dte_lo": 6,
    "dte_hi": 12,
    "min_prem": 1.5,
}


def _kite():
    from services.kite_service import get_kite_with_refresh
    return get_kite_with_refresh()


def _load():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"cycle": None, "history": [], "killed": False, "started": str(date.today())}


def _save(st):
    STATE_PATH.write_text(json.dumps(st, indent=1, default=str))


def _instruments(expiry: str):
    """NIFTY option contracts for expiry from the recorder's instruments_archive."""
    con = sqlite3.connect(OPTDB)
    cur = con.cursor()
    cur.execute(
        "SELECT tradingsymbol, strike, instrument_type, lot_size FROM instruments_archive "
        "WHERE dump_date=(SELECT MAX(dump_date) FROM instruments_archive) "
        "AND exchange='NFO' AND name=? AND expiry=? AND instrument_type IN ('CE','PE')",
        (CFG["underlying"], expiry))
    rows = cur.fetchall()
    con.close()
    return rows


def _pick_expiry(today: date):
    con = sqlite3.connect(OPTDB)
    cur = con.cursor()
    cur.execute(
        "SELECT DISTINCT expiry FROM instruments_archive "
        "WHERE dump_date=(SELECT MAX(dump_date) FROM instruments_archive) "
        "AND exchange='NFO' AND name=? AND instrument_type='CE' ORDER BY expiry",
        (CFG["underlying"],))
    exps = [r[0][:10] for r in cur.fetchall()]
    con.close()
    for e in exps:
        dte = (date.fromisoformat(e) - today).days
        if CFG["dte_lo"] <= dte <= CFG["dte_hi"]:
            return e
    return None


def _bid_ask(q):
    d = q.get("depth") or {}
    bid = (d.get("buy") or [{}])[0].get("price") or 0
    ask = (d.get("sell") or [{}])[0].get("price") or 0
    ltp = q.get("last_price") or 0
    return (bid or ltp), (ask or ltp), ltp


def _pick_leg(kite, expiry, side, spot, target):
    """Strike whose MID is nearest target on the given side. Returns leg dict or None."""
    lo, hi = (spot * 0.94, spot * 0.997) if side == "PE" else (spot * 1.003, spot * 1.06)
    cands = [(t, k, ls) for t, k, it, ls in _instruments(expiry)
             if it == side and lo <= k <= hi]
    if not cands:
        return None
    keys = ["NFO:" + c[0] for c in cands]
    quotes = {}
    for i in range(0, len(keys), 180):
        quotes.update(kite.quote(keys[i:i + 180]))
    best = None
    for t, k, ls in cands:
        q = quotes.get("NFO:" + t)
        if not q:
            continue
        bid, ask, ltp = _bid_ask(q)
        mid = (bid + ask) / 2 if bid and ask else ltp
        if mid < CFG["min_prem"] or not q.get("oi"):
            continue
        if best is None or abs(mid - target) < abs(best["mid"] - target):
            best = {"tsym": t, "strike": k, "side": side, "lot_size": ls,
                    "mid": round(mid, 2), "bid": bid, "ask": ask}
    return best


def _qty(leg):
    return int(CFG["lots"] * (leg.get("lot_size") or 65))


def _sell(cycle, leg, when, tag):
    qty = _qty(leg)
    px = leg["bid"] or leg["mid"]
    cycle["legs"].append({
        "side": leg["side"], "tsym": leg["tsym"], "strike": leg["strike"],
        "qty": -qty, "entry": px, "stop": round(CFG["stop_mult"] * px, 2),
        "status": "live", "opened": when, "lot_size": leg.get("lot_size") or 65,
    })
    cycle["flows"] = cycle.get("flows", 0.0) + px * qty
    cycle["events"].append(f"{when} {tag} SELL {leg['tsym']} @{px} x{qty}")


def _close_leg(cycle, leg, px, when, reason):
    qty = abs(leg["qty"])
    cycle["flows"] -= px * qty
    leg["status"] = reason
    leg["exit"] = px
    leg["closed"] = when
    cycle["events"].append(f"{when} {reason} BUY {leg['tsym']} @{px} x{qty}")


def _live_legs(cycle):
    return [l for l in cycle["legs"] if l["status"] == "live"]


def _finish(st, cycle, when, reason):
    cycle["status"] = "CLOSED"
    cycle["close_reason"] = reason
    cycle["closed"] = when
    lot_qty = _qty(cycle["legs"][0]) if cycle["legs"] else 650
    net_rs = round(cycle["flows"], 0)
    st["history"].append({
        "week": cycle["entry_date"], "expiry": cycle["expiry"],
        "credit0": cycle["credit0"], "reason": reason,
        "net_rs": net_rs, "net_pts": round(net_rs / lot_qty, 2),
        "events": cycle["events"],
    })
    st["cycle"] = None
    logger.info(f"[NSRW] cycle closed {reason} net Rs{net_rs}")


def _quotes_for(kite, cycle):
    keys = ["NFO:" + l["tsym"] for l in _live_legs(cycle)]
    return kite.quote(keys) if keys else {}


def entry_job():
    """Monday 15:14 — open the weekly cycle (also used by recenter)."""
    st = _load()
    if st.get("killed") or st.get("cycle"):
        return
    today = date.today()
    if today.weekday() != 0:
        return
    kite = _kite()
    expiry = _pick_expiry(today)
    if not expiry:
        logger.warning("[NSRW] no expiry in DTE window")
        return
    spot = kite.ltp(CFG["spot_key"])[CFG["spot_key"]]["last_price"]
    pe = _pick_leg(kite, expiry, "PE", spot, CFG["target"])
    ce = _pick_leg(kite, expiry, "CE", spot, CFG["target"])
    if not pe or not ce:
        logger.warning("[NSRW] leg selection failed")
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cycle = {"entry_date": str(today), "expiry": expiry, "spot0": spot,
             "legs": [], "flows": 0.0, "events": [], "rolled": False,
             "recentered": False, "status": "OPEN"}
    _sell(cycle, pe, now, "ENTRY")
    _sell(cycle, ce, now, "ENTRY")
    cycle["credit0"] = round(sum(l["entry"] for l in cycle["legs"]), 2)
    st["cycle"] = cycle
    _save(st)
    logger.info(f"[NSRW] entered {cycle['legs'][0]['tsym']}+{cycle['legs'][1]['tsym']} credit {cycle['credit0']}")


def monitor_job():
    """Every minute in market hours: stops (LTP trigger, ASK fill), one roll, PT."""
    st = _load()
    cycle = st.get("cycle")
    if not cycle or cycle["status"] != "OPEN":
        return
    now_dt = datetime.now()
    if now_dt.weekday() >= 5 or not ("09:15" <= now_dt.strftime("%H:%M") <= "15:30"):
        return
    kite = _kite()
    now = now_dt.strftime("%Y-%m-%d %H:%M")
    try:
        quotes = _quotes_for(kite, cycle)
    except Exception as e:
        logger.warning(f"[NSRW] quote fail: {e}")
        return
    if st.get("killed"):
        for leg in _live_legs(cycle):
            _, ask, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
            _close_leg(cycle, leg, ask or ltp or leg["entry"], now, "KILLED")
        _finish(st, cycle, now, "KILLED")
        _save(st)
        return
    # stops
    for leg in list(_live_legs(cycle)):
        q = quotes.get("NFO:" + leg["tsym"])
        if not q:
            continue
        bid, ask, ltp = _bid_ask(q)
        if ltp and ltp >= leg["stop"]:
            _close_leg(cycle, leg, ask or ltp, now, "STOP")
            if _live_legs(cycle) and not cycle["rolled"]:
                cycle["rolled"] = True
                spot = kite.ltp(CFG["spot_key"])[CFG["spot_key"]]["last_price"]
                nl = _pick_leg(kite, cycle["expiry"], leg["side"], spot, CFG["target"])
                if nl:
                    _sell(cycle, nl, now, "ROLL")
            if not _live_legs(cycle):
                _finish(st, cycle, now, "STOP2_FLAT")
                _save(st)
                return
            quotes = _quotes_for(kite, cycle)
    # PT on combined mid (cash-flow profit vs 50% of original credit)
    lot_qty = _qty(cycle["legs"][0])
    comb_val = 0.0
    for leg in _live_legs(cycle):
        bid, ask, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
        mid = (bid + ask) / 2 if bid and ask else ltp
        comb_val += mid * abs(leg["qty"])
    profit_pts = (cycle["flows"] - comb_val) / lot_qty
    cycle["m2m_rs"] = round(cycle["flows"] - comb_val, 0)
    if _live_legs(cycle) and profit_pts >= CFG["pt_frac"] * cycle["credit0"]:
        for leg in list(_live_legs(cycle)):
            _, ask, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
            _close_leg(cycle, leg, ask or ltp or leg["entry"], now, "PT")
        _finish(st, cycle, now, "PT")
    _save(st)


def eod_job():
    """15:16 — time exit at DTE<=1, else the once-per-cycle 1.5x recenter."""
    st = _load()
    cycle = st.get("cycle")
    if not cycle or cycle["status"] != "OPEN":
        return
    kite = _kite()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    quotes = _quotes_for(kite, cycle)
    dte = (date.fromisoformat(cycle["expiry"]) - date.today()).days
    if dte <= 1:
        for leg in list(_live_legs(cycle)):
            _, ask, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
            _close_leg(cycle, leg, ask or ltp or leg["entry"], now, "TIME")
        _finish(st, cycle, now, "TIME")
        _save(st)
        return
    if not cycle["recentered"]:
        heavy = False
        for leg in _live_legs(cycle):
            _, _, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
            if ltp and ltp >= CFG["recenter_mult"] * leg["entry"]:
                heavy = True
        if heavy:
            for leg in list(_live_legs(cycle)):
                _, ask, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
                _close_leg(cycle, leg, ask or ltp or leg["entry"], now, "RECENTER_OUT")
            spot = kite.ltp(CFG["spot_key"])[CFG["spot_key"]]["last_price"]
            pe = _pick_leg(kite, cycle["expiry"], "PE", spot, CFG["target"])
            ce = _pick_leg(kite, cycle["expiry"], "CE", spot, CFG["target"])
            if pe and ce:
                cycle["recentered"] = True
                _sell(cycle, pe, now, "RECENTER")
                _sell(cycle, ce, now, "RECENTER")
                cycle["events"].append(f"{now} RECENTERED at spot {spot}")
            else:
                _finish(st, cycle, now, "RECENTER_FLAT")
    _save(st)


def get_state():
    st = _load()
    cycle = st.get("cycle")
    hist = st.get("history", [])
    return {
        "spec": "NSR-W v1.2 paper — Mon 15:14, next-wk expiry, Rs30/leg, stop 2.0x, "
                "PT 50%, 1 roll-away, EOD recenter 1.5x, out DTE<=1, 10 lots",
        "killed": bool(st.get("killed")),
        "cycle": cycle,
        "history": hist[-30:],
        "totals": {
            "weeks": len(hist),
            "net_rs": round(sum(h["net_rs"] for h in hist), 0),
            "wins": sum(1 for h in hist if h["net_rs"] > 0),
        },
    }


def register(app, scheduler):
    from flask import jsonify
    @app.route("/api/nsrw/state")
    def nsrw_state():
        return jsonify(get_state())

    @app.route("/api/nsrw/kill-switch", methods=["POST"])
    def nsrw_kill():
        st = _load()
        st["killed"] = not st.get("killed")
        _save(st)
        return jsonify({"killed": st["killed"]})

    scheduler.add_job(entry_job, "cron", day_of_week="mon", hour=15, minute=14,
                      id="nsrw_entry", replace_existing=True, misfire_grace_time=120)
    scheduler.add_job(monitor_job, "cron", day_of_week="mon-fri", hour="9-15",
                      minute="*", id="nsrw_monitor", replace_existing=True,
                      misfire_grace_time=30)
    scheduler.add_job(eod_job, "cron", day_of_week="mon-fri", hour=15, minute=16,
                      id="nsrw_eod", replace_existing=True, misfire_grace_time=300)
    logger.info("[NSRW] registered (paper-only, 10 lots)")
