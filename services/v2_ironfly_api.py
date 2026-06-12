"""services/v2_ironfly_api.py — V2 positional iron fly + inside-week breakout sleeve (PAPER).

Self-contained, mirrors services/nas_opt.py. Wired in app.py via register(app, scheduler).

V2 fly (the book):
  - Enter 09:20 on the 2nd-nearest NIFTY weekly: SELL ATM CE+PE, BUY ATM±500 wings (2% wings).
  - Gates: India VIX >= 13 AND combo skip-filter PASSES (skip if prior-day CPR<0.10% OR inside-week).
    Every would-be-skip is shadow-logged (forward validation ledger).
  - Manage: 2% underlying move-stop, +40% credit PT, roll at DTE<=1. Re-enter next qualifying day.

Inside-week breakout sleeve (paper-only experiment, fires only on inside-week-skip weeks):
  - 15:20 check: first daily CLOSE beyond the inside week's prior-week H/L.
  - UP-break -> CALL DEBIT spread (capture runner). DOWN-break -> broken-wing fly skewed down.

PAPER ONLY (force_paper). Live spot/premium/VIX/expiry via Kite (NasScanner); daily bars for the
CPR/inside-week signals pulled fresh from Kite (NIFTY tok 256265) since market_data.db is stale.
DB: backtest_data/v2_ironfly_trading.db. API: /api/v2-ironfly/*, /api/v2-breakout/*.
"""
import json, sqlite3, logging
from pathlib import Path
from datetime import datetime, date, timedelta

import pandas as pd

from services.v2_breakout_signals import (
    combo_skip, breakout_state, prior_day_cpr_width_pct,
    v2_fly_legs, breakout_legs, atm_strike,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "backtest_data" / "v2_ironfly_trading.db"
NIFTY_TOKEN = 256265

CFG = dict(force_paper=True, live_weekdays=(0, 1, 4), vix_floor=13.0,
           lots=10, lot_size=65, stop_pct=0.02, pt_pct=0.40,
           min_entry_dte=4, roll_dte=1, brokerage=20.0,
           slippage_pct=0.0025)   # 0.25% of premium — matches the AlgoTest backtest net basis
QTY = CFG["lots"] * CFG["lot_size"]   # 650

_scanner = None
_daily_cache = {}        # date_iso -> DataFrame
_instr_cache = {"day": None, "rows": None}


# ---------- kite/data helpers ----------
def _scn():
    global _scanner
    if _scanner is None:
        from services.nas_scanner import NasScanner
        try:
            from config import NAS_ATM_DEFAULTS as _c
        except Exception:
            _c = {"strike_interval": 50}
        _scanner = NasScanner(dict(_c))
    return _scanner


def _kite():
    from services.kite_service import get_kite
    return get_kite()


def _vix():
    try:
        q = _kite().quote(["NSE:INDIA VIX"])
        return q.get("NSE:INDIA VIX", {}).get("last_price")
    except Exception as e:
        logger.warning(f"[V2] VIX fetch failed: {e}"); return None


def _spot():
    return _scn().get_live_spot()


def _last_min_bar():
    """(label, close) of the last COMPLETED 1-min NIFTY candle — matches AlgoTest's 1-min
    candle-close engine. `label` is the candle's OWN start-time 'HH:MM:00' (= AlgoTest's
    recorded exit time, e.g. '3:20'), so a live stop logs the same time as the backtest report."""
    try:
        rows = _kite().historical_data(NIFTY_TOKEN, date.today(), date.today(), "minute")
        if not rows:
            return None, _spot()
        nowmin = datetime.now().replace(second=0, microsecond=0)
        done = [r for r in rows if r["date"].replace(tzinfo=None) < nowmin]
        bar = done[-1] if done else rows[-1]
        return bar["date"].strftime("%H:%M:00"), float(bar["close"])
    except Exception:
        return None, _spot()


def _nifty_expiries():
    """Sorted NIFTY option expiries (date objects) >= today, cached per day."""
    today = date.today().isoformat()
    if _instr_cache["day"] == today and _instr_cache["rows"]:
        return _instr_cache["rows"]
    try:
        instr = _kite().instruments("NFO")
        exps = sorted({i["expiry"] for i in instr
                       if i["name"] == "NIFTY" and i["instrument_type"] in ("CE", "PE")
                       and i["expiry"] >= date.today()})
        _instr_cache.update(day=today, rows=exps)
        return exps
    except Exception as e:
        logger.warning(f"[V2] expiry list failed: {e}"); return []


def _second_weekly():
    exps = _nifty_expiries()
    if not exps:
        return None
    return exps[1] if len(exps) > 1 else exps[0]


def _daily_df():
    """Fresh NIFTY daily bars from Kite for the causal signals (DB is stale)."""
    today = date.today().isoformat()
    if today in _daily_cache:
        return _daily_cache[today]
    try:
        rows = _kite().historical_data(NIFTY_TOKEN, date.today() - timedelta(days=160), date.today(), "day")
        df = pd.DataFrame([(r["date"].strftime("%Y-%m-%d"), r["open"], r["high"], r["low"], r["close"]) for r in rows],
                          columns=["date", "open", "high", "low", "close"])
        df["date"] = pd.to_datetime(df["date"]); df = df.drop_duplicates("date").set_index("date").sort_index()
        _daily_cache.clear(); _daily_cache[today] = df
        return df
    except Exception as e:
        logger.warning(f"[V2] daily bars fetch failed: {e}"); return None


def _resolve(legs, expiry):
    """Attach tradingsymbol + live premium to each leg dict. Returns None on any miss."""
    out = []
    for lg in legs:
        sym = _scn()._build_tradingsymbol(lg["instrument_type"], lg["strike"], expiry)
        prem = _scn().get_live_option_premium(sym) if sym else None
        if not sym or prem is None or prem <= 0:
            logger.warning(f"[V2] cannot resolve/quote {lg}"); return None
        out.append({**lg, "sym": sym, "entry": float(prem), "ltp": float(prem), "exit": None})
    return out


# ---------- DB ----------
def _conn():
    c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row; return c


def init_db():
    c = _conn()
    c.executescript("""
      CREATE TABLE IF NOT EXISTS v2_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        system TEXT, day TEXT, entry_time TEXT, entry_spot REAL, entry_vix REAL,
        expiry TEXT, dte_entry INTEGER, legs_json TEXT, net_entry REAL,
        status TEXT DEFAULT 'OPEN', exit_time TEXT, exit_day TEXT, exit_reason TEXT, exit_spot REAL,
        net_exit REAL, pnl REAL, pnl_now REAL, mode TEXT DEFAULT 'paper',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP );
      CREATE TABLE IF NOT EXISTS v2_shadow_skips (
        id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT, reasons TEXT, spot REAL, vix REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP );
    """)
    try:
        c.execute("ALTER TABLE v2_positions ADD COLUMN series_json TEXT DEFAULT '[]'")
    except Exception:
        pass        # column already exists
    try:
        c.execute("ALTER TABLE v2_positions ADD COLUMN exit_day TEXT")
    except Exception:
        pass        # column already exists
    c.commit(); c.close()


def _hhmm():
    return datetime.now().strftime("%H:%M")


def _append_series(pos_id, pnl):
    c = _conn()
    r = c.execute("SELECT series_json FROM v2_positions WHERE id=?", (pos_id,)).fetchone()
    ser = json.loads((r["series_json"] if r and r["series_json"] else "[]"))
    ser.append([_hhmm(), round(pnl)])
    ser = ser[-400:]
    c.execute("UPDATE v2_positions SET series_json=? WHERE id=?", (json.dumps(ser), pos_id))
    c.commit(); c.close()


def _open(system):
    c = _conn()
    r = c.execute("SELECT * FROM v2_positions WHERE system=? AND status='OPEN' ORDER BY id DESC LIMIT 1",
                  (system,)).fetchone()
    c.close(); return dict(r) if r else None


def _net_premium(legs):
    """credit (+) / debit (-) per unit = sum(SELL entry) - sum(BUY entry)."""
    return sum((lg["entry"] if lg["side"] == "SELL" else -lg["entry"]) for lg in legs)


def _pnl_now(legs):
    per = sum(((lg["entry"] - lg["ltp"]) if lg["side"] == "SELL" else (lg["ltp"] - lg["entry"])) for lg in legs)
    brok = len(legs) * 2 * CFG["brokerage"]
    # 0.25%/leg slippage on entry + exit premium — matches the backtest's net-of-slippage basis
    slip = CFG["slippage_pct"] * sum(lg["entry"] + lg["ltp"] for lg in legs) * QTY
    return per * QTY - brok - slip


def _insert(system, spot, vix, expiry, legs):
    dte = (expiry - date.today()).days
    c = _conn()
    c.execute("INSERT INTO v2_positions (system,day,entry_time,entry_spot,entry_vix,expiry,dte_entry,"
              "legs_json,net_entry,status,mode) VALUES (?,?,?,?,?,?,?,?,?, 'OPEN','paper')",
              (system, date.today().isoformat(), datetime.now().strftime("%H:%M:%S"), spot, vix,
               expiry.isoformat(), dte, json.dumps(legs), _net_premium(legs)))
    pid = c.lastrowid
    c.execute("UPDATE v2_positions SET series_json=? WHERE id=?", (json.dumps([[_hhmm(), 0]]), pid))
    c.commit(); c.close()
    logger.info(f"[V2] PAPER ENTRY {system} spot={spot:.0f} vix={vix} net={_net_premium(legs):.1f} "
                f"legs={[(l['side'],l['instrument_type'],l['strike']) for l in legs]}")


def _refresh_marks(pos):
    legs = json.loads(pos["legs_json"]); changed = False
    for lg in legs:
        p = _scn().get_live_option_premium(lg["sym"])
        if p and p > 0:
            lg["ltp"] = float(p); changed = True
    pnl = _pnl_now(legs)
    c = _conn()
    c.execute("UPDATE v2_positions SET legs_json=?, pnl_now=? WHERE id=?",
              (json.dumps(legs), round(pnl), pos["id"]))
    c.commit(); c.close()
    return legs, pnl


def _close(pos, reason, spot, at_time=None):
    legs = json.loads(pos["legs_json"])
    for lg in legs:
        lg["exit"] = lg["ltp"]
    pnl = _pnl_now(legs)
    xt = at_time or datetime.now().strftime("%H:%M:%S")   # candle label for move-stop, else wall-clock
    c = _conn()
    c.execute("UPDATE v2_positions SET status='CLOSED',exit_time=?,exit_day=?,exit_reason=?,exit_spot=?,"
              "net_exit=?,pnl=?,pnl_now=?,legs_json=? WHERE id=?",
              (xt, date.today().isoformat(), reason, spot, _net_premium(legs),
               round(pnl), round(pnl), json.dumps(legs), pos["id"]))
    c.commit(); c.close()
    logger.info(f"[V2] PAPER EXIT {pos['system']} ({reason}) pnl={pnl:.0f}")


# ---------- scheduler jobs ----------
def entry_job():
    """09:20 — open the V2 fly if flat & gates pass; else shadow-log the skip."""
    init_db()
    if _open("v2"):
        return
    daily = _daily_df()
    if daily is None or len(daily) < 30:
        logger.info("[V2] entry: no daily bars"); return
    today = pd.Timestamp(date.today())
    spot = _spot(); vix = _vix()
    if spot is None or vix is None:
        logger.info("[V2] entry: no spot/vix"); return
    if vix < CFG["vix_floor"]:
        _shadow(today, [f"vix<{CFG['vix_floor']}({vix})"], spot, vix); return
    skip, reasons = combo_skip(daily, today)
    if skip:
        _shadow(today, reasons, spot, vix)
        logger.info(f"[V2] entry SKIP {reasons} (shadow-logged)"); return
    exp = _second_weekly()
    if not exp or (exp - date.today()).days < CFG["min_entry_dte"]:
        logger.info(f"[V2] entry: no suitable 2nd-weekly (exp={exp})"); return
    legs = _resolve(v2_fly_legs(spot), exp)
    if not legs:
        return
    _insert("v2", spot, vix, exp, legs)


def _shadow(today, reasons, spot, vix):
    c = _conn()
    c.execute("INSERT INTO v2_shadow_skips (day,reasons,spot,vix) VALUES (?,?,?,?)",
              (today.date().isoformat(), ",".join(reasons), spot, vix))
    c.commit(); c.close()


def breakout_job():
    """15:20 — on an inside-week-skip week, enter the breakout sleeve if a break has confirmed."""
    init_db()
    if _open("breakout"):
        return
    daily = _daily_df()
    if daily is None:
        return
    today = pd.Timestamp(date.today())
    skip, reasons = combo_skip(daily, today)
    if "inside_week" not in reasons:
        return
    bo = breakout_state(daily, today)
    if not bo.get("direction"):
        return
    spot = _spot()
    exp = _second_weekly()
    if spot is None or not exp:
        return
    legs = _resolve(breakout_legs(spot, bo["direction"]), exp)
    if not legs:
        return
    _insert("breakout", spot, _vix() or 0, exp, legs)
    logger.info(f"[V2-breakout] {bo['direction']}-break entered (ref {bo.get('ref_high')}/{bo.get('ref_low')})")


def monitor_job():
    """Every minute in market hours — mark + manage open V2 / breakout positions.
    Move-stop is checked against the last COMPLETED 1-min candle close (AlgoTest-faithful)."""
    ctime, spot = _last_min_bar()
    for system in ("v2", "breakout"):
        pos = _open(system)
        if not pos:
            continue
        legs, pnl = _refresh_marks(pos)
        _append_series(pos["id"], pnl)
        if spot is None:
            continue
        es = pos["entry_spot"]; net = pos["net_entry"]
        # 2% underlying move-stop — fires on the 1-min candle close, stamped at the candle time
        if es and abs(spot - es) / es >= CFG["stop_pct"]:
            _close(pos, "move2%", spot, at_time=ctime); continue
        # profit target
        if net > 0 and pnl >= CFG["pt_pct"] * net * QTY:          # credit structures
            _close(pos, "pt40", spot); continue
        if net < 0 and pnl >= 1.0 * abs(net) * QTY:               # debit spread: 2x target
            _close(pos, "target2x", spot); continue
        if net < 0 and pnl <= -0.6 * abs(net) * QTY:
            _close(pos, "stop", spot); continue
        # roll near expiry
        exp = datetime.strptime(pos["expiry"], "%Y-%m-%d").date()
        if (exp - date.today()).days <= CFG["roll_dte"]:
            _close(pos, "roll", spot); continue


# ---------- API ----------
def _state(system):
    init_db()
    pos = _open(system)
    if pos:
        pos["legs"] = json.loads(pos["legs_json"]); pos.pop("legs_json", None)
        for l in pos["legs"]:
            l["pnl"] = round(((l["entry"] - l["ltp"]) if l["side"] == "SELL" else (l["ltp"] - l["entry"])) * QTY)
        pos["series"] = json.loads(pos.get("series_json") or "[]"); pos.pop("series_json", None)
        ser = pos["series"]
        pos["low"] = min((p[1] for p in ser), default=0)
        pos["high"] = max((p[1] for p in ser), default=0)
        pos["stop_up"] = round(pos["entry_spot"] * (1 + CFG["stop_pct"]))
        pos["stop_dn"] = round(pos["entry_spot"] * (1 - CFG["stop_pct"]))
        pos["stop_pct"] = CFG["stop_pct"] * 100
    c = _conn()
    tot = c.execute("SELECT COALESCE(SUM(pnl),0), COUNT(*) FROM v2_positions WHERE system=? AND status='CLOSED'",
                    (system,)).fetchone()
    closed = [dict(r) for r in c.execute(
        "SELECT id,day,entry_time,exit_day,exit_time,exit_reason,entry_spot,net_entry,pnl FROM v2_positions "
        "WHERE system=? AND status='CLOSED' ORDER BY id DESC LIMIT 50", (system,))]
    c.close()
    return {"system": system, "mode": "paper" if CFG["force_paper"] else "live",
            "open": pos, "closed_total_pnl": round(tot[0] or 0), "closed_trades": tot[1], "closed": closed}


def get_v2_state():
    s = _state("v2")
    c = _conn()
    sk = [dict(r) for r in c.execute("SELECT day,reasons,spot,vix FROM v2_shadow_skips ORDER BY id DESC LIMIT 60")]
    c.close()
    s["shadow_skips"] = sk
    s["config"] = {"vix_floor": CFG["vix_floor"], "wings": "2% of ATM (snapped to 50)", "stop": "2% move",
                   "pt": "40%", "lots": CFG["lots"], "qty": QTY,
                   "filter": "skip if prior-day CPR<0.10% OR inside-week",
                   "margin_est": "≈₹9.6L (₹95.8k/lot)"}
    return s


def get_breakout_state():
    s = _state("breakout")
    s["config"] = {"trigger": "first daily close beyond inside-week prior-week H/L",
                   "up": "call debit spread (runner)", "down": "broken-wing fly skewed down",
                   "note": "PAPER-ONLY experiment; bear side unvalidated"}
    return s


def kill_switch():
    """Close all open paper positions immediately at current marks."""
    init_db(); spot = _spot()
    n = 0
    for system in ("v2", "breakout"):
        pos = _open(system)
        if pos:
            _refresh_marks(pos); _close(pos, "kill_switch", spot or pos["entry_spot"]); n += 1
    return {"closed": n}


def stream():
    """SSE — re-price the open V2 fly's legs from live kite.ltp() every ~3s. Mirrors
    /api/straddles/stream. Payload {type:'tick', pnl_now, legs:[{strike,instrument_type,side,ltp,pnl}], ts}."""
    import time as _t
    yield ": connected\n\n"
    last = None
    while True:
        try:
            pos = _open("v2")
            if not pos:
                if last != "__idle__":
                    last = "__idle__"; yield "data: " + json.dumps({"type": "idle"}) + "\n\n"
                _t.sleep(5); continue
            legs = json.loads(pos["legs_json"])
            q = _kite().ltp(["NFO:" + l["sym"] for l in legs] + ["NSE:NIFTY 50"])
            for l in legs:
                v = q.get("NFO:" + l["sym"])
                if v and v.get("last_price"):
                    l["ltp"] = float(v["last_price"])
            sp = q.get("NSE:NIFTY 50", {}).get("last_price")
            pnl = _pnl_now(legs)
            out = {"type": "tick", "pnl_now": round(pnl), "ts": _t.time(), "spot": sp,
                   "legs": [{"strike": l["strike"], "instrument_type": l["instrument_type"],
                             "side": l["side"], "ltp": l["ltp"],
                             "pnl": round(((l["entry"] - l["ltp"]) if l["side"] == "SELL"
                                           else (l["ltp"] - l["entry"])) * QTY)} for l in legs]}
            sig = json.dumps(out["legs"], sort_keys=True)
            if sig != last:
                last = sig; yield "data: " + json.dumps(out) + "\n\n"
        except Exception:
            pass
        _t.sleep(3.0)


def register(app, scheduler):
    from flask import jsonify, Response
    init_db()
    app.add_url_rule("/api/v2-ironfly/state", "v2_ironfly_state", lambda: jsonify(get_v2_state()))
    app.add_url_rule("/api/v2-breakout/state", "v2_breakout_state", lambda: jsonify(get_breakout_state()))
    app.add_url_rule("/api/v2-ironfly/scan", "v2_ironfly_scan", lambda: (entry_job() or jsonify({"ok": True})), methods=["POST"])
    app.add_url_rule("/api/v2-ironfly/kill-switch", "v2_ironfly_kill", lambda: jsonify(kill_switch()), methods=["POST"])
    app.add_url_rule("/api/v2-ironfly/stream", "v2_ironfly_stream",
                     lambda: Response(stream(), mimetype="text/event-stream",
                                      headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}))
    scheduler.add_job(entry_job, "cron", day_of_week="mon-fri", hour=9, minute=20,
                      id="v2_ironfly_entry", replace_existing=True)
    scheduler.add_job(monitor_job, "cron", day_of_week="mon-fri", hour="9-15", minute="*",
                      id="v2_ironfly_monitor", replace_existing=True)
    scheduler.add_job(breakout_job, "cron", day_of_week="mon-fri", hour=15, minute=20,
                      id="v2_breakout_entry", replace_existing=True)
    logger.info("[V2] registered: /api/v2-ironfly/* + /api/v2-breakout/* + entry(09:20)/monitor(3min)/breakout(15:20) PAPER jobs")
