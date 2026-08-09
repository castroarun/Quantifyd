"""O=L / O=H first-candle break — LIVE PAPER futures book (user-specified).

PAPER ONLY. User spec (2026-08-09, verbatim intent):
  5 min after open, scan F&O stocks: first 5-min candle with OPEN==LOW ->
  LONG candidate; OPEN==HIGH -> SHORT candidate. Day gap must be within
  +/-0.5%; the first candle must not be long (range <= 0.5%). Entry when
  the candle's high (low) is broken -> 1 LOT futures (paper, stock price
  proxy x lot size). Exit: SuperTrend(7,2) on 5-min as trailing stop —
  a 5-min CLOSE beyond ST exits. Positions may carry overnight.

NOTE: deployed WITHOUT prior backtest validation at the user's request —
this paper book IS the experiment. NAV = Rs10L notional margin pool + P&L.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("ohol_paper")

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "backtest_data" / "ohol_paper.db"

CFG = {
    "capital": 1_000_000.0,
    "gap_abs_max": 0.005,
    "candle_max_range": 0.005,
    "eq_tol": 0.0005,            # open==low tolerance (0.05%)
    "st_period": 7,
    "st_mult": 2.0,
    "cost_side": 0.0003,         # futures-ish per side
    "max_positions": 10,
}


def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    with _conn() as c:
        c.execute("CREATE TABLE IF NOT EXISTS ohp_kv (k TEXT PRIMARY KEY, v TEXT)")
        c.execute("""CREATE TABLE IF NOT EXISTS ohp_setups
                     (d TEXT, symbol TEXT, side TEXT, trigger REAL,
                      candle_lo REAL, candle_hi REAL, status TEXT,
                      PRIMARY KEY (d, symbol))""")
        c.execute("""CREATE TABLE IF NOT EXISTS ohp_positions
                     (symbol TEXT PRIMARY KEY, side TEXT, lots INTEGER,
                      lot_size INTEGER, entry_price REAL, entry_time TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS ohp_fills
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT,
                      symbol TEXT, side TEXT, price REAL, lots INTEGER,
                      lot_size INTEGER, reason TEXT, pnl REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS ohp_nav
                     (d TEXT PRIMARY KEY, nav REAL, n_pos INTEGER)""")


def _get(k, default=None):
    with _conn() as c:
        r = c.execute("SELECT v FROM ohp_kv WHERE k=?", (k,)).fetchone()
    return json.loads(r[0]) if r else default


def _set(k, v):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO ohp_kv VALUES (?,?)", (k, json.dumps(v)))


# liquid F&O subset only (user spec) — the research liquid list (r/89)
LIQUID_FNO = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "TCS", "SBIN",
              "AXISBANK", "KOTAKBANK", "LT", "ITC", "BHARTIARTL", "MARUTI",
              "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL",
              "ADANIENT", "ADANIPORTS", "BAJFINANCE", "BAJAJFINSV", "HCLTECH",
              "WIPRO", "TECHM", "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB",
              "TITAN", "ASIANPAINT", "ULTRACEMCO", "GRASIM", "NTPC",
              "POWERGRID", "ONGC", "COALINDIA", "BPCL", "HINDUNILVR",
              "NESTLEIND", "BRITANNIA", "EICHERMOT", "HEROMOTOCO",
              "BAJAJ-AUTO", "M&M", "APOLLOHOSP", "INDUSINDBK", "SBILIFE",
              "HDFCLIFE", "DLF", "GODREJPROP", "AMBUJACEM", "SHREECEM",
              "PIDILITIND", "SIEMENS", "ABB", "BEL", "HAL", "TRENT", "DMART",
              "NAUKRI", "ZOMATO", "PAYTM", "PNB", "BANKBARODA", "CANBK",
              "FEDERALBNK", "IDFCFIRSTB", "AUBANK", "CHOLAFIN", "LICHSGFIN",
              "MUTHOOTFIN", "SRF", "UPL", "TATAPOWER", "TORNTPOWER", "GAIL",
              "IGL"]


def _lots():
    from services.data_manager import FNO_LOT_SIZES
    return {s: z for s, z in FNO_LOT_SIZES.items() if s in LIQUID_FNO}


def _kite():
    from services.kite_service import get_kite_with_refresh
    return get_kite_with_refresh()


def _bars(kite, tok, days=4):
    data = kite.historical_data(tok, datetime.now() - timedelta(days=days),
                                datetime.now(), "5minute")
    if not data:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date").sort_index()


def seed(force=False):
    init_db()
    if _get("seeded") and not force:
        return {"ok": True, "already": True}
    with _conn() as c:
        c.execute("DELETE FROM ohp_positions")
    _set("realized", 0.0)
    _set("seeded", True)
    _set("inception", date.today().isoformat())
    logger.info("[OHP] seeded")
    return {"ok": True}


def scan_job():
    """09:21: find O=L / O=H small first candles within gap limits."""
    if not _get("seeded") or _get("killed"):
        return
    from services.nifty500_universe import get_instrument_token
    kite = _kite()
    lots = _lots()
    today = date.today().isoformat()
    n = 0
    for s in sorted(lots.keys()):
        try:
            tok = get_instrument_token(s)
            if not tok:
                continue
            df = _bars(kite, tok)
            time.sleep(0.35)
            if df is None:
                continue
            days = sorted(df.index.normalize().unique())
            if len(days) < 2 or days[-1].date().isoformat() != today:
                continue
            dd = df[df.index.normalize() == days[-1]]
            prev_close = float(df[df.index.normalize() == days[-2]]["close"].iloc[-1])
            f = dd.iloc[0]
            gap = abs(f["open"] / prev_close - 1)
            rng = (f["high"] - f["low"]) / f["open"]
            if gap > CFG["gap_abs_max"] or rng > CFG["candle_max_range"] or rng <= 0:
                continue
            side = None
            if abs(f["open"] - f["low"]) / f["open"] <= CFG["eq_tol"]:
                side, trig = "LONG", float(f["high"])
            elif abs(f["open"] - f["high"]) / f["open"] <= CFG["eq_tol"]:
                side, trig = "SHORT", float(f["low"])
            if side:
                with _conn() as c:
                    c.execute("INSERT OR REPLACE INTO ohp_setups VALUES "
                              "(?,?,?,?,?,?, 'ARMED')",
                              (today, s, side, trig, float(f["low"]),
                               float(f["high"])))
                n += 1
        except Exception as e:
            logger.warning(f"[OHP] scan {s}: {e}")
    logger.info(f"[OHP] scan: {n} setups armed")
    _set("last_scan", datetime.now().isoformat(timespec="seconds"))


def _supertrend_last(df5):
    from services.technical_indicators import calc_supertrend
    st, direction = calc_supertrend(df5, atr_period=CFG["st_period"],
                                    multiplier=CFG["st_mult"])
    return float(st.iloc[-1]), int(direction.iloc[-1])


def cycle_job():
    """Every 5 min 09:26..15:25: trigger armed setups; trail open positions."""
    if not _get("seeded") or _get("killed"):
        return
    from services.nifty500_universe import get_instrument_token
    kite = _kite()
    lots = _lots()
    today = date.today().isoformat()
    with _conn() as c:
        setups = c.execute("SELECT symbol, side, trigger FROM ohp_setups "
                           "WHERE d=? AND status='ARMED'", (today,)).fetchall()
        poss = c.execute("SELECT symbol, side FROM ohp_positions").fetchall()
    held = {p[0] for p in poss}
    n_open = len(held)
    for s, side, trig in setups:
        if s in held or n_open >= CFG["max_positions"]:
            continue
        try:
            tok = get_instrument_token(s)
            df = _bars(kite, tok, days=2)
            time.sleep(0.35)
            if df is None:
                continue
            dd = df[df.index.normalize() == df.index.normalize()[-1]]
            ltp = float(dd["close"].iloc[-1])
            hit = (dd["high"].max() > trig) if side == "LONG" else (dd["low"].min() < trig)
            if hit:
                _enter(s, side, max(ltp, trig) if side == "LONG" else min(ltp, trig),
                       lots[s])
                with _conn() as c:
                    c.execute("UPDATE ohp_setups SET status='TRIGGERED' "
                              "WHERE d=? AND symbol=?", (today, s))
                n_open += 1
        except Exception as e:
            logger.warning(f"[OHP] trigger {s}: {e}")
    # trail exits
    for s, side in poss:
        try:
            tok = get_instrument_token(s)
            df = _bars(kite, tok)
            time.sleep(0.35)
            if df is None or len(df) < 30:
                continue
            done = df.iloc[:-1]                      # last COMPLETED bar
            st_val, _ = _supertrend_last(done)
            last_close = float(done["close"].iloc[-1])
            if (side == "LONG" and last_close < st_val) or \
               (side == "SHORT" and last_close > st_val):
                _exit(s, float(df["close"].iloc[-1]), "ST_FLIP")
        except Exception as e:
            logger.warning(f"[OHP] trail {s}: {e}")
    _set("last_cycle", datetime.now().isoformat(timespec="seconds"))


def _enter(s, side, px, lot_size):
    fill = px * (1 + CFG["cost_side"]) if side == "LONG" else px * (1 - CFG["cost_side"])
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO ohp_positions VALUES (?,?,1,?,?,?)",
                  (s, side, lot_size, fill,
                   datetime.now().isoformat(timespec="seconds")))
        c.execute("INSERT INTO ohp_fills(ts,symbol,side,price,lots,lot_size,reason,pnl) "
                  "VALUES (?,?,?,?,1,?,?,NULL)",
                  (datetime.now().isoformat(timespec="seconds"), s,
                   "BUY" if side == "LONG" else "SELL", fill, lot_size, "OHOL"))
    logger.info(f"[OHP] ENTER {side} {s} 1x{lot_size} @ {fill:.2f}")


def _exit(s, px, reason):
    with _conn() as c:
        pos = c.execute("SELECT side, lot_size, entry_price FROM ohp_positions "
                        "WHERE symbol=?", (s,)).fetchone()
        if not pos:
            return
        side, lot_size, ep = pos
        fill = px * (1 - CFG["cost_side"]) if side == "LONG" else px * (1 + CFG["cost_side"])
        pnl = (fill - ep) * lot_size if side == "LONG" else (ep - fill) * lot_size
        c.execute("DELETE FROM ohp_positions WHERE symbol=?", (s,))
        c.execute("INSERT INTO ohp_fills(ts,symbol,side,price,lots,lot_size,reason,pnl) "
                  "VALUES (?,?,?,?,1,?,?,?)",
                  (datetime.now().isoformat(timespec="seconds"), s,
                   "SELL" if side == "LONG" else "BUY", fill, lot_size,
                   reason, pnl))
    _set("realized", float(_get("realized", 0.0)) + pnl)
    logger.info(f"[OHP] EXIT {s} @ {fill:.2f} ({reason}) pnl={pnl:,.0f}")


def eod_job():
    """15:29: NAV mark (positions may carry overnight)."""
    if not _get("seeded"):
        return
    kite = _kite()
    with _conn() as c:
        poss = c.execute("SELECT symbol, side, lot_size, entry_price "
                         "FROM ohp_positions").fetchall()
    unreal = 0.0
    if poss:
        try:
            quotes = kite.ltp([f"NSE:{p[0]}" for p in poss])
            for s, side, ls, ep in poss:
                px = quotes.get(f"NSE:{s}", {}).get("last_price", ep)
                unreal += (px - ep) * ls if side == "LONG" else (ep - px) * ls
        except Exception:
            pass
    nav = CFG["capital"] + float(_get("realized", 0.0)) + unreal
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO ohp_nav VALUES (?,?,?)",
                  (date.today().isoformat(), nav, len(poss)))
    logger.info(f"[OHP] EOD NAV Rs{nav:,.0f} ({len(poss)} pos)")


def _state():
    with _conn() as c:
        poss = c.execute("SELECT symbol, side, lot_size, entry_price, entry_time "
                         "FROM ohp_positions ORDER BY symbol").fetchall()
        setups = c.execute("SELECT symbol, side, trigger, status FROM ohp_setups "
                           "WHERE d=? ORDER BY symbol",
                           (date.today().isoformat(),)).fetchall()
        navs = c.execute("SELECT d, nav, n_pos FROM ohp_nav ORDER BY d").fetchall()
        fills = c.execute("SELECT ts, symbol, side, price, lot_size, reason, pnl "
                          "FROM ohp_fills ORDER BY id DESC LIMIT 50").fetchall()
    return {
        "mode": "PAPER",
        "system": "O=L/O=H first-candle break, 1-lot futures, ST(7,2) 5m trail",
        "capital": CFG["capital"], "inception": _get("inception"),
        "killed": bool(_get("killed")), "last_scan": _get("last_scan"),
        "last_cycle": _get("last_cycle"),
        "realized": round(float(_get("realized", 0.0)), 0),
        "n_positions": len(poss),
        "positions": [{"symbol": s, "side": sd, "lot_size": ls, "entry": e,
                       "since": t} for s, sd, ls, e, t in poss],
        "today_setups": [{"symbol": s, "side": sd, "trigger": tr, "status": st}
                         for s, sd, tr, st in setups],
        "navcurve": [{"d": d, "nav": round(n, 0), "n_pos": p} for d, n, p in navs],
        "recent_fills": [{"ts": t, "symbol": s, "side": sd, "price": p,
                          "lot_size": ls, "reason": r, "pnl": pl}
                         for t, s, sd, p, ls, r, pl in fills],
    }


def register(app, scheduler):
    init_db()
    from flask import jsonify, request

    @app.route("/api/ohol-paper/state")
    def ohp_state():
        return jsonify(_state())

    @app.route("/api/ohol-paper/seed", methods=["POST"])
    def ohp_seed():
        return jsonify(seed(force=bool(request.args.get("force"))))

    @app.route("/api/ohol-paper/kill-switch", methods=["POST"])
    def ohp_kill():
        _set("killed", not bool(_get("killed")))
        return jsonify({"killed": bool(_get("killed"))})

    scheduler.add_job(scan_job, "cron", day_of_week="mon-fri",
                      hour="9", minute="21", id="ohp_scan",
                      replace_existing=True, misfire_grace_time=180)
    scheduler.add_job(cycle_job, "cron", day_of_week="mon-fri",
                      hour="9", minute="26,31,36,41,46,51,56", id="ohp_cyc9",
                      replace_existing=True, misfire_grace_time=120)
    scheduler.add_job(cycle_job, "cron", day_of_week="mon-fri",
                      hour="10-14", minute="1,6,11,16,21,26,31,36,41,46,51,56",
                      id="ohp_cyc", replace_existing=True, misfire_grace_time=120)
    scheduler.add_job(cycle_job, "cron", day_of_week="mon-fri",
                      hour="15", minute="1,6,11,16,21,25", id="ohp_cyc15",
                      replace_existing=True, misfire_grace_time=120)
    scheduler.add_job(eod_job, "cron", day_of_week="mon-fri",
                      hour="15", minute="29", id="ohp_eod",
                      replace_existing=True, misfire_grace_time=600)
    logger.info("[OHP] registered (paper-only)")
