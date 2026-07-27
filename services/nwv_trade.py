"""NWV Phase-1 — pivot-anchored jade-lizard weekly book, LIVE PAPER (research/94).

Locked spec (Arun 2026-07-27 — "construct like so", all non-ignore weeks):
  TRIGGER  Monday 09:50 IST, after the Phase-0 view fires (09:46). Trade EVERY
           week whose final_view != ignore (the view's only proven value is the
           ignore filter; research/94: this template ~breakeven historically —
           this book is the honest forward test of it).
  STRUCT   Arun's pivot-anchored asymmetric condor (jade-lizard shaped), on
           NEXT week's expiry (cal DTE 6-12):
             SELL PE @ min(floor50(weekly S1), floor50(spot-100))
             BUY  PE @ short_put - 550           (disaster wing)
             SELL CE @ max(floor50(weekly R2), ceil50(spot+200))
             BUY  CE @ short_call + 200
           Fixed 10 lots. Sells fill at BID, buys at ASK (pessimistic).
  MGMT     PT at +50% of net credit; stop at -1.0x net credit (combined MTM,
           checked every minute; matches the backtested pt50_stop1x cell).
  PIVOT-X  15:25 daily: close < weekly S1 -> exit BOTH put legs; close > weekly
           R2 -> exit BOTH call legs (research/94 phase-2: exit_side on pivot
           breach = best cell, t 2.48; rolls re-widen the tail — never roll).
  TIME     Friday 15:15 flat; DTE<=1 15:16 backstop (holiday weeks).
PAPER ONLY. Accounting in cash flows; no margin model. Backtest prior entered
Mon EOD — live entry is 09:50 (Arun's actual practice); divergence noted in
research/94 RESULTS.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger("nwv_trade")

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "backtest_data" / "nwv_trade_paper.json"
OPTDB = ROOT / "backtest_data" / "options_data.db"

CFG = {
    "underlying": "NIFTY",
    "spot_key": "NSE:NIFTY 50",
    "lots": 10,
    "put_wing": 550,
    "call_wing": 200,
    "pt_frac": 0.5,        # take profit at 50% of net credit
    "stop_mult": 1.0,      # stop at -1x net credit
    "dte_lo": 6,
    "dte_hi": 12,
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


def _r50d(x):
    return int(x // 50) * 50


def _r50u(x):
    return int(-(-x // 50)) * 50


def _instruments(expiry: str):
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


def _find_contract(expiry, strike, side):
    for t, k, it, ls in _instruments(expiry):
        if it == side and int(k) == int(strike):
            return {"tsym": t, "strike": int(k), "side": side, "lot_size": ls}
    return None


def _latest_view():
    """This week's Phase-0 view row + weekly state, or (None, None)."""
    from services.nwv_engine import get_nwv_engine
    engine = get_nwv_engine()
    v = engine.db.latest_view()
    if not v:
        return None, None
    ws = engine.db.get_weekly_state(v.get("week_start"))
    return v, ws


def _open_leg(cycle, contract, qty_sign, px, when, tag):
    """qty_sign -1 short / +1 long. Fills already chosen by caller (bid/ask)."""
    qty = int(CFG["lots"] * (contract.get("lot_size") or 65)) * qty_sign
    cycle["legs"].append({
        "side": contract["side"], "tsym": contract["tsym"],
        "strike": contract["strike"], "qty": qty, "entry": px,
        "status": "live", "opened": when,
        "lot_size": contract.get("lot_size") or 65,
    })
    cycle["flows"] = cycle.get("flows", 0.0) - px * qty      # short: +cash, long: -cash
    verb = "SELL" if qty_sign < 0 else "BUY"
    cycle["events"].append(f"{when} {tag} {verb} {contract['tsym']} @{px} x{abs(qty)}")


def _close_all(st, cycle, quotes, when, reason):
    for leg in cycle["legs"]:
        if leg["status"] != "live":
            continue
        bid, ask, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
        px = (ask if leg["qty"] < 0 else bid) or ltp or leg["entry"]
        cycle["flows"] += px * leg["qty"]      # shorts buy back (-cash), longs sell (+cash)
        leg["status"] = reason
        leg["exit"] = px
        leg["closed"] = when
        cycle["events"].append(
            f"{when} {reason} {'BUY' if leg['qty'] < 0 else 'SELL'} {leg['tsym']} @{px} x{abs(leg['qty'])}")
    cycle["status"] = "CLOSED"
    cycle["close_reason"] = reason
    cycle["closed"] = when
    lot_qty = int(CFG["lots"] * (cycle["legs"][0].get("lot_size") or 65)) if cycle["legs"] else 650
    net_rs = round(cycle["flows"], 0)
    st["history"].append({
        "week": cycle["entry_date"], "expiry": cycle["expiry"], "view": cycle["view"],
        "strikes": cycle["strikes"], "credit0": cycle["credit0"], "reason": reason,
        "net_rs": net_rs, "net_pts": round(net_rs / lot_qty, 2),
        "events": cycle["events"],
    })
    st["cycle"] = None
    logger.info(f"[NWV-TRADE] cycle closed {reason} net Rs{net_rs}")


def entry_job():
    """Monday 09:50 — build the week's jade lizard if the view says trade."""
    st = _load()
    if st.get("killed") or st.get("cycle"):
        return
    today = date.today()
    if today.weekday() != 0:
        return
    try:
        v, ws = _latest_view()
    except Exception as e:
        logger.error(f"[NWV-TRADE] view fetch failed: {e}")
        return
    if not v or not ws or v.get("week_start") != today.isoformat():
        logger.warning("[NWV-TRADE] no fresh Phase-0 view for this week — skip")
        return
    view = v.get("final_view")
    if view == "ignore":
        st["history"].append({"week": today.isoformat(), "view": view,
                              "reason": "SKIP_IGNORE", "net_rs": 0, "net_pts": 0})
        _save(st)
        logger.info("[NWV-TRADE] view=ignore — no trade this week")
        return
    kite = _kite()
    expiry = _pick_expiry(today)
    if not expiry:
        logger.warning("[NWV-TRADE] no expiry in DTE window")
        return
    spot = kite.ltp(CFG["spot_key"])[CFG["spot_key"]]["last_price"]
    sp = min(_r50d(ws["pivot_s1"]), _r50d(spot - 100))
    sc = max(_r50d(ws["pivot_r2"]), _r50u(spot + 200))
    strikes = {"sell_pe": sp, "buy_pe": sp - CFG["put_wing"],
               "sell_ce": sc, "buy_ce": sc + CFG["call_wing"]}
    legs_spec = [("sell_pe", "PE", -1), ("buy_pe", "PE", 1),
                 ("sell_ce", "CE", -1), ("buy_ce", "CE", 1)]
    contracts = {}
    for key, side, sign in legs_spec:
        c = _find_contract(expiry, strikes[key], side)
        if not c:
            logger.warning(f"[NWV-TRADE] contract missing {strikes[key]}{side} {expiry} — skip week")
            return
        contracts[key] = (c, sign)
    quotes = kite.quote(["NFO:" + c[0]["tsym"] for c in contracts.values()])
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cycle = {"entry_date": str(today), "expiry": expiry, "view": view,
             "spot0": spot, "strikes": strikes,
             "pivots": {"s1": ws["pivot_s1"], "r2": ws["pivot_r2"]},
             "legs": [], "flows": 0.0, "events": [], "status": "OPEN"}
    for key, side, sign in legs_spec:
        c, _ = contracts[key]
        bid, ask, ltp = _bid_ask(quotes.get("NFO:" + c["tsym"], {}))
        px = (bid if sign < 0 else ask) or ltp
        if not px:
            logger.warning(f"[NWV-TRADE] no quote for {c['tsym']} — skip week")
            return
        _open_leg(cycle, c, sign, px, now, "ENTRY")
    lot_qty = int(CFG["lots"] * (cycle["legs"][0].get("lot_size") or 65))
    cycle["credit0"] = round(cycle["flows"] / lot_qty, 2)      # net credit in points
    if cycle["credit0"] <= 0:
        logger.warning(f"[NWV-TRADE] net debit {cycle['credit0']} — not a credit structure, skip")
        return
    st["cycle"] = cycle
    _save(st)
    logger.info(f"[NWV-TRADE] entered {view} JL {strikes} credit {cycle['credit0']}pts")


def monitor_job():
    """Every minute market hours: combined-MTM PT / stop, kill switch."""
    st = _load()
    cycle = st.get("cycle")
    if not cycle or cycle["status"] != "OPEN":
        return
    now_dt = datetime.now()
    if now_dt.weekday() >= 5 or not ("09:15" <= now_dt.strftime("%H:%M") <= "15:30"):
        return
    kite = _kite()
    now = now_dt.strftime("%Y-%m-%d %H:%M")
    keys = ["NFO:" + l["tsym"] for l in cycle["legs"] if l["status"] == "live"]
    try:
        quotes = kite.quote(keys) if keys else {}
    except Exception as e:
        logger.warning(f"[NWV-TRADE] quote fail: {e}")
        return
    if st.get("killed"):
        _close_all(st, cycle, quotes, now, "KILLED")
        _save(st)
        return
    lot_qty = int(CFG["lots"] * (cycle["legs"][0].get("lot_size") or 65))
    # cost to flatten now (shorts at ask, longs at bid) vs entry flows
    close_cash = 0.0
    for leg in cycle["legs"]:
        if leg["status"] != "live":
            continue
        bid, ask, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
        px = (ask if leg["qty"] < 0 else bid) or ltp or leg["entry"]
        close_cash += -px * (-leg["qty"])        # buyback shorts -cash, sell longs +cash
    profit_rs = cycle["flows"] + close_cash
    cycle["m2m_rs"] = round(profit_rs, 0)
    cap_rs = cycle["credit0"] * lot_qty
    if profit_rs >= CFG["pt_frac"] * cap_rs:
        _close_all(st, cycle, quotes, now, "PT")
    elif profit_rs <= -CFG["stop_mult"] * cap_rs:
        _close_all(st, cycle, quotes, now, "STOP")
    _save(st)


def pivot_exit_job():
    """15:25 daily — exit the threatened SIDE on a close beyond weekly S1/R2
    (research/94 phase-2 winner). Never roll."""
    st = _load()
    cycle = st.get("cycle")
    if not cycle or cycle["status"] != "OPEN" or not cycle.get("pivots"):
        return
    kite = _kite()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    spot = kite.ltp(CFG["spot_key"])[CFG["spot_key"]]["last_price"]
    side = None
    if spot < cycle["pivots"]["s1"]:
        side = "PE"
    elif spot > cycle["pivots"]["r2"]:
        side = "CE"
    if not side:
        return
    live = [l for l in cycle["legs"] if l["status"] == "live" and l["side"] == side]
    if not live:
        return
    quotes = kite.quote(["NFO:" + l["tsym"] for l in live])
    for leg in live:
        bid, ask, ltp = _bid_ask(quotes.get("NFO:" + leg["tsym"], {}))
        px = (ask if leg["qty"] < 0 else bid) or ltp or leg["entry"]
        cycle["flows"] += px * leg["qty"]
        leg["status"] = "PIVOT_EXIT"
        leg["exit"] = px
        leg["closed"] = now
        cycle["events"].append(
            f"{now} PIVOT_EXIT ({side} side, spot {spot}) "
            f"{'BUY' if leg['qty'] < 0 else 'SELL'} {leg['tsym']} @{px} x{abs(leg['qty'])}")
    if not [l for l in cycle["legs"] if l["status"] == "live"]:
        # both sides gone -> finish the cycle
        _close_all(st, cycle, {}, now, "PIVOT_EXIT_FLAT")
    _save(st)
    logger.info(f"[NWV-TRADE] pivot exit fired on {side} side at spot {spot}")


def time_exit_job():
    """Fri 15:15 flat; also DTE<=1 backstop for holiday-shifted weeks."""
    st = _load()
    cycle = st.get("cycle")
    if not cycle or cycle["status"] != "OPEN":
        return
    today = date.today()
    dte = (date.fromisoformat(cycle["expiry"]) - today).days
    if today.weekday() != 4 and dte > 1:
        return
    kite = _kite()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    keys = ["NFO:" + l["tsym"] for l in cycle["legs"] if l["status"] == "live"]
    quotes = kite.quote(keys) if keys else {}
    _close_all(st, cycle, quotes, now, "TIME")
    _save(st)


def get_state():
    st = _load()
    hist = [h for h in st.get("history", []) if h.get("reason") != "SKIP_IGNORE"]
    return {
        "spec": "NWV-JL paper (research/94) — Mon 09:50 on any non-ignore view: "
                "SELL PE@S1 (+550 wing) / SELL CE@R2 (+200 wing), next-wk expiry, "
                "PT 50% credit, stop -1x, 15:25 pivot-exit of threatened side on "
                "close beyond S1/R2 (never roll), Fri 15:15 out, 10 lots. PAPER ONLY.",
        "killed": bool(st.get("killed")),
        "cycle": st.get("cycle"),
        "history": st.get("history", [])[-30:],
        "totals": {
            "weeks": len(hist),
            "net_rs": round(sum(h.get("net_rs", 0) for h in hist), 0),
            "wins": sum(1 for h in hist if h.get("net_rs", 0) > 0),
        },
    }


def register(app, scheduler):
    from flask import jsonify

    @app.route("/api/nwv-trade/state")
    def nwv_trade_state():
        return jsonify(get_state())

    @app.route("/api/nwv-trade/kill-switch", methods=["POST"])
    def nwv_trade_kill():
        st = _load()
        st["killed"] = not st.get("killed")
        _save(st)
        return jsonify({"killed": st["killed"]})

    scheduler.add_job(entry_job, "cron", day_of_week="mon", hour=9, minute=50,
                      id="nwv_trade_entry", replace_existing=True, misfire_grace_time=300)
    scheduler.add_job(monitor_job, "cron", day_of_week="mon-fri", hour="9-15",
                      minute="*", id="nwv_trade_monitor", replace_existing=True,
                      misfire_grace_time=30)
    scheduler.add_job(pivot_exit_job, "cron", day_of_week="mon-thu", hour=15, minute=25,
                      id="nwv_trade_pivot_exit", replace_existing=True, misfire_grace_time=120)
    scheduler.add_job(time_exit_job, "cron", day_of_week="mon-fri", hour=15, minute=15,
                      id="nwv_trade_time_exit", replace_existing=True, misfire_grace_time=300)
    logger.info("[NWV-TRADE] registered (paper-only, 10 lots)")
