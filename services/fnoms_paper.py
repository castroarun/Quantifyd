"""F&O Multi-Signal — LIVE PAPER book (₹20L), 3 walk-forward-validated LONG systems.

PAPER ONLY. Deploys the three winners that PASSED walk-forward (OOS
2023-01-01..2025-12-31, gates PF>=1.20, Sharpe>=0.8, MaxDD<=30%):

  DON55  research/23 `55_vol_3x_atr_floor`  — close > 55d high AND volume >= 3x
         50d avg AND ATR/close >= 1%.   OOS PF 1.97 Sharpe 1.25 DD 7.7% CAGR 13.5%
  ADX30  research/26 `adx_30`             — close > 55d high AND volume >= 3x
         AND ADX(14) > 30.                OOS PF 1.77 Sharpe 1.30 CAGR 15.5%
  PBRSI  research/27 `pb_rsi40`           — uptrend (close > 200SMA) pullback:
         RSI(14) < 40 then close > prior high.  OOS PF 1.55 Sharpe 1.01 CAGR 13.0%

Shared exit engine (as backtested, research/21):
  stop = max(entry - 2*ATR14, entry*0.92) · target = entry * 1.25 ·
  max hold 60 sessions · daily EOD evaluation on completed bars.
Sizing: 1% equity risk per trade, capped at equity/10 notional; max 10
concurrent per system (30 book-wide). Costs 20bps round-trip.
Signals overlap by design (all trade the F&O universe) — a symbol already
held by ANY system is skipped, so the book never doubles up on one name.
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

logger = logging.getLogger("fnoms_paper")

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "backtest_data" / "fnoms_paper.db"

CFG = {
    "capital": 2_000_000.0,
    "risk_per_trade": 0.01,
    "max_concurrent_per_system": 10,
    "hard_stop_pct": 0.08,
    "atr_mult": 2.0,
    "target_pct": 0.25,
    "max_hold_days": 60,
    "cost_side": 0.0010,
    "liquid_yield": 0.055,
    "systems": ["DON55", "ADX30", "PBRSI"],
}


def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    with _conn() as c:
        c.execute("CREATE TABLE IF NOT EXISTS fms_kv (k TEXT PRIMARY KEY, v TEXT)")
        c.execute("""CREATE TABLE IF NOT EXISTS fms_positions
                     (symbol TEXT PRIMARY KEY, system TEXT, qty INTEGER,
                      entry_price REAL, entry_date TEXT, stop REAL,
                      target REAL, atr REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS fms_fills
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT,
                      system TEXT, side TEXT, price REAL, qty INTEGER,
                      reason TEXT, pnl REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS fms_nav
                     (d TEXT PRIMARY KEY, nav REAL, equity REAL, cash REAL,
                      n_pos INTEGER, bench REAL)""")


def _get(k, default=None):
    with _conn() as c:
        r = c.execute("SELECT v FROM fms_kv WHERE k=?", (k,)).fetchone()
    return json.loads(r[0]) if r else default


def _set(k, v):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO fms_kv VALUES (?,?)", (k, json.dumps(v)))


def _get_cash():
    return float(_get("cash", 0.0))


def _universe():
    from services.data_manager import FNO_LOT_SIZES
    return sorted(FNO_LOT_SIZES.keys())


def _kite():
    from services.kite_service import get_kite_with_refresh
    return get_kite_with_refresh()


def seed(force=False):
    init_db()
    if _get("seeded") and not force:
        return {"ok": True, "already": True}
    with _conn() as c:
        c.execute("DELETE FROM fms_positions")
    _set("cash", CFG["capital"])
    _set("seeded", True)
    _set("inception", date.today().isoformat())
    logger.info(f"[FMS] seeded Rs{CFG['capital']:,.0f}")
    return {"ok": True, "capital": CFG["capital"]}


# ------------------------------------------------------------- indicators

def _prep(df):
    d = df.copy()
    prev_close = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev_close).abs(),
                    (d["low"] - prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["atr_pct"] = d["atr"] / d["close"]
    d["high_55"] = d["high"].shift(1).rolling(55).max()
    d["prev_high"] = d["high"].shift(1)
    d["vol_avg"] = d["volume"].shift(1).rolling(50).mean()
    d["sma200"] = d["close"].rolling(200).mean()
    delta = d["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / (dn + 1e-12))
    # ADX(14)
    up_m = d["high"].diff()
    dn_m = -d["low"].diff()
    plus_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    minus_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / (atr_w + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / (atr_w + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    return d


def _signals(row, prev_rsi_min):
    """Return list of system names firing on this completed bar."""
    out = []
    if any(pd.isna(row.get(k)) for k in ("high_55", "vol_avg", "atr_pct")):
        return out
    vol_ok = row["volume"] >= 3.0 * row["vol_avg"]
    brk55 = row["close"] > row["high_55"]
    if brk55 and vol_ok and row["atr_pct"] >= 0.01:
        out.append("DON55")
    if brk55 and vol_ok and not pd.isna(row.get("adx")) and row["adx"] > 30:
        out.append("ADX30")
    if (not pd.isna(row.get("sma200")) and row["close"] > row["sma200"]
            and prev_rsi_min is not None and prev_rsi_min < 40
            and not pd.isna(row.get("prev_high")) and row["close"] > row["prev_high"]):
        out.append("PBRSI")
    return out


def _bars(kite, tok, days=420):
    data = kite.historical_data(tok, datetime.now() - timedelta(days=days),
                                datetime.now(), "day")
    if not data:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.set_index("date").sort_index()


# ------------------------------------------------------------- daily job

def daily_job():
    """15:40 Mon-Fri: evaluate exits on completed bars, then new entries."""
    if not _get("seeded") or _get("killed"):
        return
    from services.nifty500_universe import get_instrument_token
    kite = _kite()
    today = date.today().isoformat()
    with _conn() as c:
        poss = {r[0]: r for r in c.execute(
            "SELECT symbol, system, qty, entry_price, entry_date, stop, target, atr "
            "FROM fms_positions")}
    n_ent = n_ex = 0
    per_system = {s: 0 for s in CFG["systems"]}
    for _, (sym, sysname, *_rest) in poss.items():
        per_system[sysname] = per_system.get(sysname, 0) + 1

    candidates = []
    for s in _universe():
        try:
            tok = get_instrument_token(s)
            if not tok:
                continue
            df = _bars(kite, tok)
            time.sleep(0.35)
            if df is None or len(df) < 220:
                continue
            d = _prep(df)
            last = d.iloc[-1]
            ltp = float(last["close"])
            if s in poss:
                _, sysname, qty, ep, ed, stop, target, atr = poss[s]
                held = (date.today() - date.fromisoformat(ed[:10])).days
                reason = None
                if float(last["low"]) <= stop:
                    px, reason = stop, "STOP"
                elif float(last["high"]) >= target:
                    px, reason = target, "TARGET"
                elif held >= CFG["max_hold_days"]:
                    px, reason = ltp, "MAX_HOLD"
                if reason:
                    _exit(s, px, reason)
                    n_ex += 1
                continue
            prev_rsi_min = float(d["rsi"].iloc[-6:-1].min())
            fired = _signals(last, prev_rsi_min)
            if fired:
                candidates.append((s, fired[0], ltp, float(last["atr"])))
        except Exception as e:
            logger.warning(f"[FMS] {s}: {e}")

    for sym, sysname, px, atr in candidates:
        if per_system.get(sysname, 0) >= CFG["max_concurrent_per_system"]:
            continue
        if _enter(sym, sysname, px, atr):
            per_system[sysname] = per_system.get(sysname, 0) + 1
            n_ent += 1

    # liquid accrual + NAV
    _set("cash", _get_cash() * (1 + CFG["liquid_yield"]) ** (1 / 252))
    _mark_nav(kite)
    logger.info(f"[FMS] daily: +{n_ent} entries, -{n_ex} exits, "
                f"{len(candidates)} candidates")
    _set("last_run", datetime.now().isoformat(timespec="seconds"))


def _enter(sym, sysname, px, atr):
    fill = px * (1 + CFG["cost_side"])
    stop = max(fill - CFG["atr_mult"] * atr, fill * (1 - CFG["hard_stop_pct"]))
    risk_ps = fill - stop
    if risk_ps <= 0:
        return False
    cash = _get_cash()
    equity_now = cash + _equity_value()
    qty = int((equity_now * CFG["risk_per_trade"]) // risk_ps)
    cap = equity_now / (CFG["max_concurrent_per_system"] * len(CFG["systems"]))
    if qty * fill > cap:
        qty = int(cap // fill)
    if qty <= 0 or qty * fill > cash:
        return False
    _set("cash", cash - qty * fill)
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO fms_positions VALUES (?,?,?,?,?,?,?,?)",
                  (sym, sysname, qty, fill, date.today().isoformat(), stop,
                   fill * (1 + CFG["target_pct"]), atr))
        c.execute("INSERT INTO fms_fills(ts,symbol,system,side,price,qty,reason,pnl) "
                  "VALUES (?,?,?,?,?,?,?,NULL)",
                  (datetime.now().isoformat(timespec="seconds"), sym, sysname,
                   "BUY", fill, qty, sysname))
    logger.info(f"[FMS] ENTER {sysname} {sym} {qty} @ {fill:.2f} stop {stop:.2f}")
    return True


def _exit(sym, px, reason):
    fill = px * (1 - CFG["cost_side"])
    with _conn() as c:
        pos = c.execute("SELECT system, qty, entry_price FROM fms_positions "
                        "WHERE symbol=?", (sym,)).fetchone()
        if not pos:
            return
        sysname, qty, ep = pos
        pnl = qty * (fill - ep)
        c.execute("DELETE FROM fms_positions WHERE symbol=?", (sym,))
        c.execute("INSERT INTO fms_fills(ts,symbol,system,side,price,qty,reason,pnl) "
                  "VALUES (?,?,?,?,?,?,?,?)",
                  (datetime.now().isoformat(timespec="seconds"), sym, sysname,
                   "SELL", fill, qty, reason, pnl))
    _set("cash", _get_cash() + qty * fill)
    logger.info(f"[FMS] EXIT {sym} @ {fill:.2f} ({reason}) pnl={pnl:,.0f}")


def _equity_value(ltps=None):
    with _conn() as c:
        poss = c.execute("SELECT symbol, qty, entry_price FROM fms_positions").fetchall()
    if not poss:
        return 0.0
    if ltps is None:
        return sum(q * ep for _, q, ep in poss)
    return sum(q * ltps.get(s, ep) for s, q, ep in poss)


def _ltps(kite=None):
    with _conn() as c:
        syms = [r[0] for r in c.execute("SELECT symbol FROM fms_positions")]
    if not syms:
        return {}
    try:
        q = (kite or _kite()).ltp([f"NSE:{s}" for s in syms])
        return {k.split(":")[1]: v["last_price"] for k, v in q.items()}
    except Exception:
        return {}


def _mark_nav(kite=None):
    ltps = _ltps(kite)
    equity = _equity_value(ltps)
    cash = _get_cash()
    bench = None
    try:
        bench = (kite or _kite()).ltp(["NSE:NIFTYBEES"])["NSE:NIFTYBEES"]["last_price"]
    except Exception:
        pass
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM fms_positions").fetchone()[0]
        c.execute("INSERT OR REPLACE INTO fms_nav VALUES (?,?,?,?,?,?)",
                  (date.today().isoformat(), cash + equity, equity, cash, n, bench))
    logger.info(f"[FMS] NAV Rs{cash+equity:,.0f} ({n} positions)")


def _state():
    ltps = _ltps()
    with _conn() as c:
        poss = c.execute("SELECT symbol, system, qty, entry_price, entry_date, "
                         "stop, target FROM fms_positions ORDER BY system, symbol").fetchall()
        navs = c.execute("SELECT d, nav, bench FROM fms_nav ORDER BY d").fetchall()
        fills = c.execute("SELECT ts, symbol, system, side, price, qty, reason, pnl "
                          "FROM fms_fills ORDER BY id DESC LIMIT 60").fetchall()
    cash = _get_cash()
    positions = []
    for s, sysname, q, ep, ed, stop, target in poss:
        last = ltps.get(s)
        positions.append({
            "symbol": s, "system": sysname, "qty": q, "entry": ep,
            "since": ed, "stop": stop, "target": target, "last": last,
            "mtm": round(q * (last - ep), 0) if last else None,
            "mtm_pct": round((last / ep - 1) * 100, 2) if last else None})
    return {
        "mode": "PAPER",
        "system": "F&O Multi-Signal — Donchian-55 / ADX-30 / RSI-pullback (long)",
        "capital": CFG["capital"], "inception": _get("inception"),
        "killed": bool(_get("killed")), "last_run": _get("last_run"),
        "cash": round(cash, 0), "n_positions": len(poss),
        "positions": positions,
        "navcurve": [{"d": d, "nav": round(n, 0), "bench": b} for d, n, b in navs],
        "recent_fills": [{"ts": t, "symbol": s, "system": sy, "side": sd,
                          "price": p, "qty": q, "reason": r, "pnl": pl}
                         for t, s, sy, sd, p, q, r, pl in fills],
    }


def register(app, scheduler):
    init_db()
    from flask import jsonify, request

    @app.route("/api/fnoms-paper/state")
    def fms_state():
        return jsonify(_state())

    @app.route("/api/fnoms-paper/seed", methods=["POST"])
    def fms_seed():
        return jsonify(seed(force=bool(request.args.get("force"))))

    @app.route("/api/fnoms-paper/run", methods=["POST"])
    def fms_run():
        daily_job()
        return jsonify({"ok": True, "last_run": _get("last_run")})

    @app.route("/api/fnoms-paper/kill-switch", methods=["POST"])
    def fms_kill():
        _set("killed", not bool(_get("killed")))
        return jsonify({"killed": bool(_get("killed"))})

    scheduler.add_job(daily_job, "cron", day_of_week="mon-fri",
                      hour="15", minute="40", id="fms_daily",
                      replace_existing=True, misfire_grace_time=900)
    logger.info("[FMS] registered (paper-only)")
