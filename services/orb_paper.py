"""Gap-ORB 4-day revival — LIVE PAPER book (research/89 candidate, Rs10L).

PAPER ONLY — no live mode exists in this module. 90-day soak vs the
+15-20bps/trade expectation; do NOT arm live before the soak verdict.

System (locked, research/89 results/RESULTS.md — OOS consumed, so this soak
IS the validation):
  Universe: F&O stocks. Gate: NIFTY 50 > 50-DMA (no new entries when off).
  Signal: day gap-up >= 0.4% AND close crosses above the 90-min opening
  range high (09:15-10:45) -> BUY at market. Stop: OR-low (checked each
  cycle). Time exit: close of the 4th session (entry day = session 1).
  Sizing: Rs10L / 20 slots = Rs50k equal-notional; idle cash earns 5.5%.
  Costs: 5bp/side paper deduction. First-trigger intake, capped at 20.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger("orb_paper")

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "backtest_data" / "orb_paper.db"

CFG = {
    "capital": 1_000_000.0,
    "slots": 20,
    "liquid_yield": 0.055,
    "cost_side": 0.0005,
    "gap_min": 0.004,
    "or_bars": 18,               # 18 x 5min = 09:15-10:45
    "hold_sessions": 4,          # exit at close of 4th session
    "gate_sma": 50,
    "nifty_token": 256265,
}


def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    with _conn() as c:
        c.execute("CREATE TABLE IF NOT EXISTS obp_kv (k TEXT PRIMARY KEY, v TEXT)")
        c.execute("""CREATE TABLE IF NOT EXISTS obp_positions
                     (symbol TEXT PRIMARY KEY, qty REAL, entry_price REAL,
                      entry_time TEXT, stop REAL, sessions INTEGER)""")
        c.execute("""CREATE TABLE IF NOT EXISTS obp_fills
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT,
                      symbol TEXT, side TEXT, price REAL, qty REAL,
                      reason TEXT, pnl REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS obp_nav
                     (d TEXT PRIMARY KEY, nav REAL, equity REAL, cash REAL,
                      n_pos INTEGER, gate INTEGER, bench REAL)""")


def _get(k, default=None):
    with _conn() as c:
        r = c.execute("SELECT v FROM obp_kv WHERE k=?", (k,)).fetchone()
    return json.loads(r[0]) if r else default


def _set(k, v):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO obp_kv VALUES (?,?)", (k, json.dumps(v)))


def _universe():
    from services.data_manager import FNO_LOT_SIZES
    return sorted(FNO_LOT_SIZES.keys())


def _kite():
    from services.kite_service import get_kite_with_refresh
    return get_kite_with_refresh()


def _gate_on(kite):
    """NIFTY 50 close > 50-DMA, cached per day."""
    today = date.today().isoformat()
    cached = _get("gate_cache")
    if cached and cached.get("d") == today:
        return cached["on"]
    data = kite.historical_data(CFG["nifty_token"],
                                datetime.now() - timedelta(days=120),
                                datetime.now(), "day")
    closes = pd.Series([b["close"] for b in data])
    on = bool(len(closes) > CFG["gate_sma"]
              and closes.iloc[-1] > closes.tail(CFG["gate_sma"]).mean())
    _set("gate_cache", {"d": today, "on": on})
    logger.info(f"[OBP] gate NIFTY>{CFG['gate_sma']}DMA: {'ON' if on else 'OFF'}")
    return on


def _get_cash():
    return float(_get("cash", 0.0))


def seed(force=False):
    init_db()
    if _get("seeded") and not force:
        return {"ok": True, "already": True}
    with _conn() as c:
        c.execute("DELETE FROM obp_positions")
    _set("cash", CFG["capital"])
    _set("seeded", True)
    _set("inception", date.today().isoformat())
    logger.info(f"[OBP] seeded Rs{CFG['capital']:,.0f}, {CFG['slots']} slots")
    return {"ok": True, "capital": CFG["capital"]}


def cycle_job():
    """Every 30 min 10:46..15:16: detect breakouts (entries) + stop checks."""
    if not _get("seeded") or _get("killed"):
        return
    from services.nifty500_universe import get_instrument_token
    kite = _kite()
    try:
        gate = _gate_on(kite)
    except Exception as e:
        logger.warning(f"[OBP] gate check failed: {e}")
        return
    uni = _universe()
    with _conn() as c:
        held = {r[0] for r in c.execute("SELECT symbol FROM obp_positions")}
    n_ent = n_ex = 0
    for s in uni:
        try:
            tok = get_instrument_token(s)
            if not tok:
                continue
            # one call: last 4 calendar days of 5-min bars covers prev session
            data = kite.historical_data(tok, datetime.now() - timedelta(days=4),
                                        datetime.now(), "5minute")
            time.sleep(0.35)
            if not data:
                continue
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df["day"] = df["date"].dt.normalize()
            days = sorted(df["day"].unique())
            if len(days) < 2:
                continue
            today, prevd = days[-1], days[-2]
            dd = df[df["day"] == today]
            prev_close = float(df[df["day"] == prevd]["close"].iloc[-1])
            ltp = float(dd["close"].iloc[-1])
            if s in held:
                # stop check
                with _conn() as c:
                    stop = c.execute("SELECT stop FROM obp_positions WHERE symbol=?",
                                     (s,)).fetchone()[0]
                if ltp <= stop:
                    _exit(s, ltp, "STOP")
                    n_ex += 1
                continue
            if not gate or len(held) + n_ent >= CFG["slots"]:
                continue
            if len(dd) <= CFG["or_bars"]:
                continue
            gap = float(dd["open"].iloc[0]) / prev_close - 1
            if gap < CFG["gap_min"]:
                continue
            orh = float(dd["high"].iloc[:CFG["or_bars"]].max())
            orl = float(dd["low"].iloc[:CFG["or_bars"]].min())
            post = dd.iloc[CFG["or_bars"]:]
            if (post["close"] > orh).any() and ltp > orl:
                _enter(s, ltp, orl)
                n_ent += 1
        except Exception as e:
            logger.warning(f"[OBP] {s}: {e}")
    if n_ent or n_ex:
        logger.info(f"[OBP] cycle: +{n_ent} entries, -{n_ex} stop exits")
    _set("last_cycle", datetime.now().isoformat(timespec="seconds"))


def _enter(s, px, stop):
    fill = px * (1 + CFG["cost_side"])
    cash = _get_cash()
    slot = CFG["capital"] / CFG["slots"]
    if cash < slot * 0.5:
        return
    alloc = min(slot, cash)
    qty = alloc / fill
    _set("cash", cash - qty * fill)
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO obp_positions VALUES (?,?,?,?,?,1)",
                  (s, qty, fill, datetime.now().isoformat(timespec="seconds"), stop))
        c.execute("INSERT INTO obp_fills(ts,symbol,side,price,qty,reason,pnl) "
                  "VALUES (?,?,?,?,?,?,NULL)",
                  (datetime.now().isoformat(timespec="seconds"), s, "BUY",
                   fill, qty, "GAPORB"))
    logger.info(f"[OBP] ENTER {s} @ {fill:.2f} stop {stop:.2f}")


def _exit(s, px, reason):
    fill = px * (1 - CFG["cost_side"])
    with _conn() as c:
        pos = c.execute("SELECT qty, entry_price FROM obp_positions WHERE symbol=?",
                        (s,)).fetchone()
        if not pos:
            return
        qty, ep = pos
        pnl = qty * (fill - ep)
        c.execute("DELETE FROM obp_positions WHERE symbol=?", (s,))
        c.execute("INSERT INTO obp_fills(ts,symbol,side,price,qty,reason,pnl) "
                  "VALUES (?,?,?,?,?,?,?)",
                  (datetime.now().isoformat(timespec="seconds"), s, "SELL",
                   fill, qty, reason, pnl))
    _set("cash", _get_cash() + qty * fill)
    logger.info(f"[OBP] EXIT {s} @ {fill:.2f} ({reason}) pnl={pnl:,.0f}")


def eod_job():
    """15:20: time-stop exits (4th session close), session increment, liquid
    accrual, NAV mark."""
    if not _get("seeded"):
        return
    kite = _kite()
    with _conn() as c:
        poss = c.execute("SELECT symbol, qty, entry_price, sessions "
                         "FROM obp_positions").fetchall()
    quotes = {}
    if poss:
        try:
            quotes = kite.ltp([f"NSE:{p[0]}" for p in poss])
        except Exception:
            pass
    for s, qty, ep, sess in poss:
        px = quotes.get(f"NSE:{s}", {}).get("last_price", ep)
        if sess >= CFG["hold_sessions"]:
            _exit(s, px, "TIME4")
        else:
            with _conn() as c:
                c.execute("UPDATE obp_positions SET sessions=sessions+1 "
                          "WHERE symbol=?", (s,))
    # liquid accrual on idle cash
    _set("cash", _get_cash() * (1 + CFG["liquid_yield"]) ** (1 / 252))
    with _conn() as c:
        poss = c.execute("SELECT symbol, qty, entry_price FROM obp_positions").fetchall()
    equity = 0.0
    if poss:
        try:
            quotes = kite.ltp([f"NSE:{p[0]}" for p in poss])
            for s, qty, ep in poss:
                equity += qty * quotes.get(f"NSE:{s}", {}).get("last_price", ep)
        except Exception:
            equity = sum(q * ep for _, q, ep in poss)
    cash = _get_cash()
    bench = None
    try:
        bench = kite.ltp(["NSE:NIFTYBEES"])["NSE:NIFTYBEES"]["last_price"]
    except Exception:
        pass
    gate = bool((_get("gate_cache") or {}).get("on"))
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO obp_nav VALUES (?,?,?,?,?,?,?)",
                  (date.today().isoformat(), cash + equity, equity, cash,
                   len(poss), int(gate), bench))
    logger.info(f"[OBP] EOD NAV Rs{cash+equity:,.0f} ({len(poss)} pos)")


def _state():
    with _conn() as c:
        poss = c.execute("SELECT symbol, qty, entry_price, entry_time, stop, "
                         "sessions FROM obp_positions ORDER BY symbol").fetchall()
        navs = c.execute("SELECT d, nav, gate, bench FROM obp_nav ORDER BY d").fetchall()
        fills = c.execute("SELECT ts, symbol, side, price, reason, pnl FROM obp_fills "
                          "ORDER BY id DESC LIMIT 50").fetchall()
    return {
        "mode": "PAPER",
        "system": "Gap-ORB 90min/0.4%/4-day long + NIFTY>50DMA gate (research/89)",
        "capital": CFG["capital"], "inception": _get("inception"),
        "killed": bool(_get("killed")), "last_cycle": _get("last_cycle"),
        "cash": round(_get_cash(), 0), "n_positions": len(poss),
        "gate_on": bool((_get("gate_cache") or {}).get("on")),
        "positions": [{"symbol": s, "qty": round(q, 2), "entry": e, "since": t,
                       "stop": st, "sessions": ss}
                      for s, q, e, t, st, ss in poss],
        "navcurve": [{"d": d, "nav": round(n, 0), "gate": g, "bench": b}
                     for d, n, g, b in navs],
        "recent_fills": [{"ts": t, "symbol": s, "side": sd, "price": p,
                          "reason": r, "pnl": pl}
                         for t, s, sd, p, r, pl in fills],
    }


def register(app, scheduler):
    init_db()
    from flask import jsonify, request

    @app.route("/api/orb-paper/state")
    def obp_state():
        return jsonify(_state())

    @app.route("/api/orb-paper/seed", methods=["POST"])
    def obp_seed():
        return jsonify(seed(force=bool(request.args.get("force"))))

    @app.route("/api/orb-paper/kill-switch", methods=["POST"])
    def obp_kill():
        _set("killed", not bool(_get("killed")))
        return jsonify({"killed": bool(_get("killed"))})

    scheduler.add_job(cycle_job, "cron", day_of_week="mon-fri",
                      hour="11-15", minute="16,46", id="obp_cycle",
                      replace_existing=True, misfire_grace_time=300)
    scheduler.add_job(cycle_job, "cron", day_of_week="mon-fri",
                      hour="10", minute="46", id="obp_cycle_1046",
                      replace_existing=True, misfire_grace_time=300)
    scheduler.add_job(eod_job, "cron", day_of_week="mon-fri",
                      hour="15", minute="20", id="obp_eod",
                      replace_existing=True, misfire_grace_time=600)
    logger.info("[OBP] registered (paper-only)")
