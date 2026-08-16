"""HA 2-Green No-Wick Break — LIVE PAPER book (research/86 G5 soak).

The program's first full IS→Val→OOS survivor (see research/86_heikin_patterns/
results/RESULTS.md). PAPER ONLY — no live mode exists in this module.

System (locked — do NOT tune here; OOS is consumed):
  30-min Heikin-Ashi bars per F&O name. ENTER long at next bar open after a
  bar CLOSES above the real high of 2 consecutive green HA candles that both
  have no lower HA wick. EXIT at next bar open after a bar closes below the
  real low of 2 consecutive red HA candles. One position per name.
  Sleeve accounting: capital/N per name; idle sleeves accrue liquid yield.

Implementation: STATELESS RECOMPUTE — each 30-min cycle refetches ~10 days of
30-min bars per symbol from Kite, replays the pattern state machine (same
logic as the research sim), and reconciles the resulting desired position
against the DB. Robust to restarts; no incremental-state drift.

Watch-item from the study: per-trade fade 47→36→25bps across IS/Val/OOS —
this soak is the drought-vs-decay experiment.
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

logger = logging.getLogger("ha_paper")

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "backtest_data" / "ha_paper.db"

CFG = {
    "capital": 2_000_000.0,
    "liquid_yield": 0.055,        # net p.a. on idle sleeve cash (matches backtest)
    "warmup_days": 10,            # refetch window for stateless recompute
    "cost_side": 0.0003,          # 3bp slippage+cost per side (paper deduction)
    "min_sleeve_frac": 0.5,       # skip entry if sleeve cash < 50% of target (safety)
}


# ------------------------------------------------------------------ storage

def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS hap_kv
                     (k TEXT PRIMARY KEY, v TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS hap_positions
                     (symbol TEXT PRIMARY KEY, qty REAL, entry_price REAL,
                      entry_time TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS hap_fills
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT,
                      symbol TEXT, side TEXT, price REAL, qty REAL,
                      reason TEXT, pnl REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS hap_sleeves
                     (symbol TEXT PRIMARY KEY, cash REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS hap_nav
                     (d TEXT PRIMARY KEY, nav REAL, equity REAL, cash REAL,
                      n_pos INTEGER, bench REAL)""")


def _get(k, default=None):
    with _conn() as c:
        r = c.execute("SELECT v FROM hap_kv WHERE k=?", (k,)).fetchone()
    return json.loads(r[0]) if r else default


def _set(k, v):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO hap_kv VALUES (?,?)", (k, json.dumps(v)))


# ------------------------------------------------------------------ system

def _universe():
    from services.data_manager import FNO_LOT_SIZES
    return sorted(FNO_LOT_SIZES.keys())


def _kite():
    from services.kite_service import get_kite_with_refresh
    return get_kite_with_refresh()


def heikin(df: pd.DataFrame) -> pd.DataFrame:
    ha_c = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_o = np.empty(len(df))
    ha_o[0] = df["open"].iloc[0]
    hac = ha_c.to_numpy()
    for i in range(1, len(df)):
        ha_o[i] = (ha_o[i - 1] + hac[i - 1]) / 2
    out = pd.DataFrame(index=df.index)
    out["ha_open"], out["ha_close"] = ha_o, hac
    out["green"] = out["ha_close"] > out["ha_open"]
    out["ha_low"] = np.minimum.reduce([df["low"].to_numpy(), ha_o, hac])
    out["nowick_lo"] = np.isclose(out["ha_low"], np.minimum(ha_o, hac))
    return out


def replay(df: pd.DataFrame) -> dict:
    """Replay the locked state machine over the bar window.
    Returns {'in_pos': bool, 'fire_entry': bool, 'fire_exit': bool} where the
    fire_* flags mean: the LAST COMPLETED bar triggered the transition, so the
    fill belongs at the next bar open (i.e. NOW)."""
    ha = heikin(df)
    g = ha["green"].to_numpy()
    nw = ha["nowick_lo"].to_numpy()
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    in_pos = False
    trig = xtrig = None
    fire_entry = fire_exit = False
    for i in range(2, n):
        fire_entry = fire_exit = False
        if not in_pos:
            if g[i - 1] and g[i] and nw[i - 1] and nw[i]:
                trig = max(h[i - 1], h[i])
            if trig is not None and c[i] > trig:
                in_pos, trig, xtrig = True, None, None
                fire_entry = True
        else:
            if (not g[i - 1]) and (not g[i]):
                xtrig = min(l[i - 1], l[i])
            if xtrig is not None and c[i] < xtrig:
                in_pos, xtrig = False, None
                fire_exit = True
    return {"in_pos": in_pos, "fire_entry": fire_entry, "fire_exit": fire_exit}


# ------------------------------------------------------------------ engine

def _fetch_30m(kite, token, days):
    frm = datetime.now() - timedelta(days=days)
    data = kite.historical_data(token, frm, datetime.now(), "30minute")
    if not data:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    # keep only COMPLETED bars: a bar ends 30 min after its start, capped at
    # 15:30 (the last session bar 15:15 is 15 min long)
    now = datetime.now()
    ends = df.index + pd.Timedelta(minutes=30)
    close_cap = df.index.normalize() + pd.Timedelta(hours=15, minutes=30)
    ends = np.minimum(ends, close_cap)
    return df[ends <= now]


def seed(force=False):
    """Initialize sleeves; establish current pattern state WITHOUT taking
    positions mid-pattern (only NEW fires after seeding create positions)."""
    init_db()
    if _get("seeded") and not force:
        return {"ok": True, "already": True}
    uni = _universe()
    per = CFG["capital"] / len(uni)
    with _conn() as c:
        c.execute("DELETE FROM hap_sleeves")
        c.execute("DELETE FROM hap_positions")
        for s in uni:
            c.execute("INSERT INTO hap_sleeves VALUES (?,?)", (s, per))
    _set("seeded", True)
    _set("inception", date.today().isoformat())
    _set("interest_earned", 0.0)
    logger.info(f"[HAP] seeded: {len(uni)} sleeves x Rs.{per:,.0f}")
    return {"ok": True, "sleeves": len(uni), "per_sleeve": per}


def cycle_job():
    """Runs shortly after each 30-min bar close during market hours."""
    if not _get("seeded"):
        return
    if _get("killed"):
        return
    from services.nifty500_universe import get_instrument_token
    kite = _kite()
    uni = _universe()
    n_ent = n_ex = n_err = 0
    for s in uni:
        try:
            tok = get_instrument_token(s)
            if not tok:
                continue
            df = _fetch_30m(kite, tok, CFG["warmup_days"])
            time.sleep(0.35)
            if df is None or len(df) < 20:
                continue
            st = replay(df)
            ltp = float(df["close"].iloc[-1])
            with _conn() as c:
                pos = c.execute("SELECT qty, entry_price FROM hap_positions "
                                "WHERE symbol=?", (s,)).fetchone()
            if st["fire_entry"] and not pos:
                _enter(s, ltp)
                n_ent += 1
            elif st["fire_exit"] and pos:
                _exit(s, ltp, "PATTERN")
                n_ex += 1
            elif pos and not st["in_pos"]:
                # reconcile: replay says flat but DB holds (e.g. missed cycle)
                _exit(s, ltp, "RECONCILE")
                n_ex += 1
        except Exception as e:
            n_err += 1
            logger.warning(f"[HAP] {s}: {e}")
    if n_ent or n_ex or n_err:
        logger.info(f"[HAP] cycle: +{n_ent} entries, -{n_ex} exits, {n_err} errors")
    _set("last_cycle", datetime.now().isoformat(timespec="seconds"))


def _enter(s, px):
    fill = px * (1 + CFG["cost_side"])
    with _conn() as c:
        cash = c.execute("SELECT cash FROM hap_sleeves WHERE symbol=?", (s,)).fetchone()
        if not cash or cash[0] < CFG["capital"] / len(_universe()) * CFG["min_sleeve_frac"]:
            return
        qty = cash[0] / fill
        c.execute("UPDATE hap_sleeves SET cash=0 WHERE symbol=?", (s,))
        c.execute("INSERT OR REPLACE INTO hap_positions VALUES (?,?,?,?)",
                  (s, qty, fill, datetime.now().isoformat(timespec="seconds")))
        c.execute("INSERT INTO hap_fills(ts,symbol,side,price,qty,reason,pnl) "
                  "VALUES (?,?,?,?,?,?,NULL)",
                  (datetime.now().isoformat(timespec="seconds"), s, "BUY",
                   fill, qty, "PATTERN"))
    logger.info(f"[HAP] ENTER {s} @ {fill:.2f}")


def _exit(s, px, reason):
    fill = px * (1 - CFG["cost_side"])
    with _conn() as c:
        pos = c.execute("SELECT qty, entry_price FROM hap_positions "
                        "WHERE symbol=?", (s,)).fetchone()
        if not pos:
            return
        qty, ep = pos
        proceeds = qty * fill
        pnl = qty * (fill - ep)
        c.execute("DELETE FROM hap_positions WHERE symbol=?", (s,))
        c.execute("UPDATE hap_sleeves SET cash=? WHERE symbol=?", (proceeds, s))
        c.execute("INSERT INTO hap_fills(ts,symbol,side,price,qty,reason,pnl) "
                  "VALUES (?,?,?,?,?,?,?)",
                  (datetime.now().isoformat(timespec="seconds"), s, "SELL",
                   fill, qty, reason, pnl))
    logger.info(f"[HAP] EXIT {s} @ {fill:.2f} ({reason}) pnl={pnl:,.0f}")


def eod_job():
    """After close: accrue liquid yield on idle sleeves, mark NAV."""
    if not _get("seeded"):
        return
    daily = (1 + CFG["liquid_yield"]) ** (1 / 252)
    with _conn() as c:
        c.execute("UPDATE hap_sleeves SET cash = cash * ?", (daily,))
        cash = c.execute("SELECT COALESCE(SUM(cash),0) FROM hap_sleeves").fetchone()[0]
        poss = c.execute("SELECT symbol, qty, entry_price FROM hap_positions").fetchall()
    # mark positions at last close via Kite LTP (fallback entry price)
    equity = 0.0
    if poss:
        try:
            kite = _kite()
            quotes = kite.ltp([f"NSE:{p[0]}" for p in poss])
            for sym, qty, ep in poss:
                px = quotes.get(f"NSE:{sym}", {}).get("last_price", ep)
                equity += qty * px
        except Exception:
            equity = sum(q * ep for _, q, ep in poss)
    nav = cash + equity
    bench = None
    try:
        bench = _kite().ltp(["NSE:NIFTYBEES"])["NSE:NIFTYBEES"]["last_price"]
    except Exception:
        pass
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO hap_nav VALUES (?,?,?,?,?,?)",
                  (date.today().isoformat(), nav, equity, cash, len(poss), bench))
    logger.info(f"[HAP] EOD NAV Rs.{nav:,.0f} ({len(poss)} positions)")


# ------------------------------------------------------------------ api

def _state():
    with _conn() as c:
        poss = c.execute("SELECT symbol, qty, entry_price, entry_time "
                         "FROM hap_positions ORDER BY symbol").fetchall()
        cash = c.execute("SELECT COALESCE(SUM(cash),0) FROM hap_sleeves").fetchone()[0]
        navs = c.execute("SELECT d, nav, bench FROM hap_nav ORDER BY d").fetchall()
        fills = c.execute("SELECT ts, symbol, side, price, reason, pnl FROM hap_fills "
                          "ORDER BY id DESC LIMIT 50").fetchall()
    ltps = {}
    if poss:
        try:
            q = _kite().ltp([f"NSE:{p[0]}" for p in poss])
            ltps = {k.split(":")[1]: v["last_price"] for k, v in q.items()}
        except Exception:
            pass
    return {
        "mode": "PAPER", "system": "HA 2-green no-wick break 30m long (research/86)",
        "capital": CFG["capital"], "inception": _get("inception"),
        "killed": bool(_get("killed")), "last_cycle": _get("last_cycle"),
        "cash": round(cash, 0), "n_positions": len(poss),
        "positions": [{"symbol": s, "qty": round(q, 2), "entry": e, "since": t,
                       "last": ltps.get(s),
                       "mtm": round(q * (ltps[s] - e), 0) if s in ltps else None,
                       "mtm_pct": round((ltps[s] / e - 1) * 100, 2) if s in ltps else None}
                      for s, q, e, t in poss],
        "navcurve": [{"d": d, "nav": round(n, 0), "bench": b} for d, n, b in navs],
        "recent_fills": [{"ts": t, "symbol": s, "side": sd, "price": p,
                          "reason": r, "pnl": pl} for t, s, sd, p, r, pl in fills],
    }


def register(app, scheduler):
    init_db()
    from flask import jsonify, request

    @app.route("/api/ha-paper/state")
    def hap_state():
        return jsonify(_state())

    @app.route("/api/ha-paper/seed", methods=["POST"])
    def hap_seed():
        return jsonify(seed(force=bool(request.args.get("force"))))

    @app.route("/api/ha-paper/kill-switch", methods=["POST"])
    def hap_kill():
        _set("killed", not bool(_get("killed")))
        return jsonify({"killed": bool(_get("killed"))})

    # 30-min cycles: 1 min after each bar close 09:45..15:15, + 15:31 final
    scheduler.add_job(cycle_job, "cron", day_of_week="mon-fri",
                      hour="10-15", minute="16,46", id="hap_cycle",
                      replace_existing=True, misfire_grace_time=300)
    scheduler.add_job(cycle_job, "cron", day_of_week="mon-fri",
                      hour="9", minute="46", id="hap_cycle_0946",
                      replace_existing=True, misfire_grace_time=300)
    scheduler.add_job(cycle_job, "cron", day_of_week="mon-fri",
                      hour="15", minute="31", id="hap_cycle_eod",
                      replace_existing=True, misfire_grace_time=300)
    scheduler.add_job(eod_job, "cron", day_of_week="mon-fri",
                      hour="15", minute="50", id="hap_eod",
                      replace_existing=True, misfire_grace_time=600)
    logger.info("[HAP] registered (paper-only)")
