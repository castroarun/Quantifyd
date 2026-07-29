"""NSR-W v1.5 — NIFTY rules-based weekly strangle, LIVE PAPER books (research/90 G5).

TWO books, both under the backtested rule set (study card
/app/backtest/nifty-strangle-rules-research90):
  t30 — ENTER Rs30/leg, adjust Rs20 (v1.4 winner: adj sweep t 6.94 plateau 20-24)
  t20 — ENTER Rs20/leg, adjust Rs20 (its own backtested optimum: t 6.12)

Shared spec: ENTRY Monday 15:14, NEXT-week expiry (DTE 6-12), sell at BID,
fixed 10 lots. STOP GTT-style 2.0x per leg on LTP, fill ASK; the FIRST stop
RE-STRANGLES (close all, sell fresh pair at the adjust target, once/cycle);
a later stop closes that leg only, survivor rides. PT 50% of original credit.
EOD 15:16: recenter at adjust target if any leg >= 1.5x (once/cycle);
time exit at cal DTE <= 1. PAPER ONLY; cash-flow accounting; no margin model.
Each monitor tick appends [HH:MM, m2m_rs] to an intraday series for the UI.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger("nsrw_paper")

ROOT = Path(__file__).resolve().parent.parent
OPTDB = ROOT / "backtest_data" / "options_data.db"

COMMON = {
    "underlying": "NIFTY",
    "spot_key": "NSE:NIFTY 50",
    "lots": 10,
    "stop_mult": 2.0,
    "pt_frac": 0.5,
    "recenter_mult": 1.5,
    "dte_lo": 6,
    "dte_hi": 12,
    "min_prem": 1.5,
}

BOOKS = [
    dict(COMMON, id="t30", label="Rs30 entry / Rs20 adjust", target=30.0, adjust=20.0,
         state=ROOT / "backtest_data" / "nsrw_paper.json"),
    dict(COMMON, id="t20", label="Rs20 entry / Rs20 adjust", target=20.0, adjust=20.0,
         state=ROOT / "backtest_data" / "nsrw_paper_t20.json"),
]


def _kite():
    from services.kite_service import get_kite_with_refresh
    return get_kite_with_refresh()


def _load(cfg):
    if cfg["state"].exists():
        return json.loads(cfg["state"].read_text())
    return {"cycle": None, "history": [], "killed": False, "started": str(date.today())}


def _save(cfg, st):
    cfg["state"].write_text(json.dumps(st, indent=1, default=str))


def _instruments(cfg, expiry: str):
    con = sqlite3.connect(OPTDB)
    cur = con.cursor()
    cur.execute(
        "SELECT tradingsymbol, strike, instrument_type, lot_size FROM instruments_archive "
        "WHERE dump_date=(SELECT MAX(dump_date) FROM instruments_archive) "
        "AND exchange='NFO' AND name=? AND expiry=? AND instrument_type IN ('CE','PE')",
        (cfg["underlying"], expiry))
    rows = cur.fetchall()
    con.close()
    return rows


def _pick_expiry(cfg, today: date):
    con = sqlite3.connect(OPTDB)
    cur = con.cursor()
    cur.execute(
        "SELECT DISTINCT expiry FROM instruments_archive "
        "WHERE dump_date=(SELECT MAX(dump_date) FROM instruments_archive) "
        "AND exchange='NFO' AND name=? AND instrument_type='CE' ORDER BY expiry",
        (cfg["underlying"],))
    exps = [r[0][:10] for r in cur.fetchall()]
    con.close()
    for e in exps:
        if cfg["dte_lo"] <= (date.fromisoformat(e) - today).days <= cfg["dte_hi"]:
            return e
    return None


def _bid_ask(q):
    d = q.get("depth") or {}
    bid = (d.get("buy") or [{}])[0].get("price") or 0
    ask = (d.get("sell") or [{}])[0].get("price") or 0
    ltp = q.get("last_price") or 0
    return (bid or ltp), (ask or ltp), ltp


def _pick_leg(cfg, kite, expiry, side, spot, target):
    lo, hi = (spot * 0.94, spot * 0.997) if side == "PE" else (spot * 1.003, spot * 1.06)
    cands = [(t, k, ls) for t, k, it, ls in _instruments(cfg, expiry)
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
        if mid < cfg["min_prem"] or not q.get("oi"):
            continue
        if best is None or abs(mid - target) < abs(best["mid"] - target):
            best = {"tsym": t, "strike": k, "side": side, "lot_size": ls,
                    "mid": round(mid, 2), "bid": bid, "ask": ask, "ltp": ltp}
    return best


def _qty(cfg, leg):
    return int(cfg["lots"] * (leg.get("lot_size") or 65))


def _sell(cfg, cycle, leg, when, tag):
    qty = _qty(cfg, leg)
    px = leg["bid"] or leg["mid"] or leg["ltp"]
    cycle["legs"].append({
        "side": leg["side"], "tsym": leg["tsym"], "strike": leg["strike"],
        "qty": -qty, "entry": px, "stop": round(cfg["stop_mult"] * px, 2),
        "status": "live", "opened": when, "ltp": px,
        "lot_size": leg.get("lot_size") or 65,
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


def _finish(cfg, st, cycle, when, reason):
    cycle["status"] = "CLOSED"
    cycle["close_reason"] = reason
    cycle["closed"] = when
    lot_qty = _qty(cfg, cycle["legs"][0]) if cycle["legs"] else 650
    net_rs = round(cycle["flows"], 0)
    st["history"].append({
        "week": cycle["entry_date"], "expiry": cycle["expiry"],
        "credit0": cycle["credit0"], "reason": reason,
        "net_rs": net_rs, "net_pts": round(net_rs / lot_qty, 2),
        "events": cycle["events"],
        "entry_time": cycle.get("entry_time"), "entry_date": cycle.get("entry_date"),
        "spot0": cycle.get("spot0"), "path_week": cycle.get("path_week", []),
        "closed": when, "m2m_rs": net_rs,
        "legs": [{k: l.get(k) for k in ("side", "strike", "entry", "tsym", "status")} for l in cycle["legs"]],
    })
    st["cycle"] = None
    logger.info(f"[NSRW:{cfg['id']}] cycle closed {reason} net Rs{net_rs}")


def _quotes_for(kite, cycle):
    keys = ["NFO:" + l["tsym"] for l in _live_legs(cycle)]
    return kite.quote(keys) if keys else {}


def _restrangle(cfg, st, cycle, kite, quotes, now, tag):
    """Close ALL live legs at ask, sell a fresh pair at the ADJUST target."""
    for leg in list(_live_legs(cycle)):
        _, ask, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
        _close_leg(cycle, leg, ask or ltp or leg["entry"], now, tag + "_OUT")
    spot = kite.ltp(cfg["spot_key"])[cfg["spot_key"]]["last_price"]
    pe = _pick_leg(cfg, kite, cycle["expiry"], "PE", spot, cfg["adjust"])
    ce = _pick_leg(cfg, kite, cycle["expiry"], "CE", spot, cfg["adjust"])
    if pe and ce:
        _sell(cfg, cycle, pe, now, tag)
        _sell(cfg, cycle, ce, now, tag)
        return True
    _finish(cfg, st, cycle, now, tag + "_FLAT")
    return False


def _entry(cfg, tag="ENTRY"):
    st = _load(cfg)
    if st.get("killed") or st.get("cycle"):
        return
    today = date.today()
    if today.weekday() != 0:
        return
    kite = _kite()
    expiry = _pick_expiry(cfg, today)
    if not expiry:
        logger.warning(f"[NSRW:{cfg['id']}] no expiry in DTE window")
        return
    spot = kite.ltp(cfg["spot_key"])[cfg["spot_key"]]["last_price"]
    pe = _pick_leg(cfg, kite, expiry, "PE", spot, cfg["target"])
    ce = _pick_leg(cfg, kite, expiry, "CE", spot, cfg["target"])
    if not pe or not ce:
        logger.warning(f"[NSRW:{cfg['id']}] leg selection failed")
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cycle = {"entry_date": str(today), "expiry": expiry, "spot0": spot,
             "legs": [], "flows": 0.0, "events": [], "rolled": False,
             "recentered": False, "status": "OPEN", "mtm_series": [],
             "mtm_date": str(today), "entry_time": now, "path_week": []}
    _sell(cfg, cycle, pe, now, tag)
    _sell(cfg, cycle, ce, now, tag)
    cycle["credit0"] = round(sum(l["entry"] for l in cycle["legs"]), 2)
    st["cycle"] = cycle
    _save(cfg, st)
    logger.info(f"[NSRW:{cfg['id']}] entered credit {cycle['credit0']}")


def _monitor(cfg):
    st = _load(cfg)
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
        logger.warning(f"[NSRW:{cfg['id']}] quote fail: {e}")
        return
    if st.get("killed"):
        for leg in _live_legs(cycle):
            _, ask, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
            _close_leg(cycle, leg, ask or ltp or leg["entry"], now, "KILLED")
        _finish(cfg, st, cycle, now, "KILLED")
        _save(cfg, st)
        return
    # refresh leg marks
    for leg in _live_legs(cycle):
        q = quotes.get("NFO:" + leg["tsym"])
        if q:
            leg["ltp"] = q.get("last_price") or leg["ltp"]
    # stops -> v1.3 re-strangle once, then per-leg
    for leg in list(_live_legs(cycle)):
        q = quotes.get("NFO:" + leg["tsym"])
        if not q:
            continue
        bid, ask, ltp = _bid_ask(q)
        if ltp and ltp >= leg["stop"]:
            _close_leg(cycle, leg, ask or ltp, now, "STOP")
            if not cycle["rolled"]:
                cycle["rolled"] = True
                if not _restrangle(cfg, st, cycle, kite, quotes, now, "RESTRANGLE"):
                    _save(cfg, st)
                    return
            if not _live_legs(cycle):
                _finish(cfg, st, cycle, now, "STOP2_FLAT")
                _save(cfg, st)
                return
            quotes = _quotes_for(kite, cycle)
    # m2m + PT
    lot_qty = _qty(cfg, cycle["legs"][0])
    comb_val = 0.0
    for leg in _live_legs(cycle):
        bid, ask, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
        mid = (bid + ask) / 2 if bid and ask else (ltp or leg["ltp"])
        comb_val += mid * abs(leg["qty"])
    cycle["m2m_rs"] = round(cycle["flows"] - comb_val, 0)
    today = str(date.today())
    if cycle.get("mtm_date") != today:
        cycle["mtm_date"] = today
        cycle["mtm_series"] = []
    cycle.setdefault("mtm_series", []).append([now_dt.strftime("%H:%M"), cycle["m2m_rs"]])
    cycle["mtm_series"] = cycle["mtm_series"][-400:]
    stamp = now_dt.strftime("%Y-%m-%d %H:%M")
    pw = cycle.setdefault("path_week", [])
    if now_dt.minute % 5 == 0 and (not pw or pw[-1][0] != stamp):
        pw.append([stamp, cycle["m2m_rs"]])
        cycle["path_week"] = pw[-2000:]
    profit_pts = (cycle["flows"] - comb_val) / lot_qty
    if _live_legs(cycle) and profit_pts >= cfg["pt_frac"] * cycle["credit0"]:
        for leg in list(_live_legs(cycle)):
            _, ask, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
            _close_leg(cycle, leg, ask or ltp or leg["entry"], now, "PT")
        _finish(cfg, st, cycle, now, "PT")
    _save(cfg, st)


def _eod(cfg):
    st = _load(cfg)
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
        _finish(cfg, st, cycle, now, "TIME")
        _save(cfg, st)
        return
    if not cycle["recentered"]:
        heavy = any(
            (_bid_ask(quotes.get("NFO:" + leg["tsym"], {}))[2] or 0)
            >= cfg["recenter_mult"] * leg["entry"]
            for leg in _live_legs(cycle))
        if heavy:
            cycle["recentered"] = True
            _restrangle(cfg, st, cycle, kite, quotes, now, "RECENTER")
    _save(cfg, st)


def entry_job():
    for cfg in BOOKS:
        try:
            _entry(cfg)
        except Exception as e:
            logger.exception(f"[NSRW:{cfg['id']}] entry error: {e}")


def monitor_job():
    for cfg in BOOKS:
        try:
            _monitor(cfg)
        except Exception as e:
            logger.exception(f"[NSRW:{cfg['id']}] monitor error: {e}")


def eod_job():
    for cfg in BOOKS:
        try:
            _eod(cfg)
        except Exception as e:
            logger.exception(f"[NSRW:{cfg['id']}] eod error: {e}")


def catchup_entry():
    """Deploy-day helper: late Monday entry using latest quotes (event-marked)."""
    for cfg in BOOKS:
        st = _load(cfg)
        if not st.get("cycle") and date.today().weekday() == 0:
            _entry(cfg, tag="ENTRY_LATE")


def get_state():
    books = {}
    for cfg in BOOKS:
        st = _load(cfg)
        hist = st.get("history", [])
        books[cfg["id"]] = {
            "label": cfg["label"],
            "target": cfg["target"],
            "adjust": cfg["adjust"],
            "killed": bool(st.get("killed")),
            "cycle": st.get("cycle"),
            "history": hist[-30:],
            "totals": {
                "weeks": len(hist),
                "net_rs": round(sum(h["net_rs"] for h in hist), 0),
                "wins": sum(1 for h in hist if h["net_rs"] > 0),
            },
        }
    return {
        "spec": "NSR-W v1.5 paper — Mon 15:14 entry, next-wk expiry, GTT stop 2.0x, "
                "PT 50%, stop->RE-STRANGLE & EOD-recenter 1.5x both at the adjust "
                "target (once each/cycle), out DTE<=1, 10 lots/book",
        "books": books,
    }


def register(app, scheduler):
    from flask import jsonify

    @app.route("/api/nsrw/state")
    def nsrw_state():
        return jsonify(get_state())

    @app.route("/api/nsrw/kill-switch", methods=["POST"])
    def nsrw_kill():
        out = {}
        for cfg in BOOKS:
            st = _load(cfg)
            st["killed"] = not st.get("killed")
            _save(cfg, st)
            out[cfg["id"]] = st["killed"]
        return jsonify(out)

    scheduler.add_job(entry_job, "cron", day_of_week="mon", hour=15, minute=14,
                      id="nsrw_entry", replace_existing=True, misfire_grace_time=120)
    scheduler.add_job(monitor_job, "cron", day_of_week="mon-fri", hour="9-15",
                      minute="*", id="nsrw_monitor", replace_existing=True,
                      misfire_grace_time=30)
    scheduler.add_job(eod_job, "cron", day_of_week="mon-fri", hour=15, minute=16,
                      id="nsrw_eod", replace_existing=True, misfire_grace_time=300)
    logger.info("[NSRW] registered v1.5 (2 paper books: t30, t20)")
