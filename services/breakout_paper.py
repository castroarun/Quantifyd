"""services/breakout_paper.py — MTF-Bullish Volume-Breakout PAPER book (research/71 winner).

Live-reflective ₹10L paper deployment of the validated short-term breakout swing system.
FULLY AUTOMATED daily by APScheduler. This module NEVER places a real order — it only records
paper fills, so there is zero live-trading risk.

System rules (research/71 G5 winner):
  Entry  : daily+weekly+monthly MACD line>0 AND close≥98% of 252-day high AND volume≥2×20d-avg
           AND 20-day MEDIAN turnover≥₹5cr AND price≥₹20 (skip >15% single-day chase = unfillable)
  Select : when >slots free, rank the day's qualifiers by today's %-run, take the TOP 1 (max 1
           new entry/day) up to 8 concurrent; equal-weight
  Gate   : take NEW entries only when NIFTYBEES > its 200-day SMA (does NOT force-liquidate)
  Exit   : Donchian-20 lower-channel trailing stop (close < prior-20-day low) OR 20% catastrophe
           stop from entry, whichever first. No profit target.

DB: backtest_data/breakout_paper.db   API: /api/breakout-paper/*   Page: /app/breakout-paper
"""
from __future__ import annotations
import sqlite3, json, logging
from pathlib import Path
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "backtest_data" / "breakout_paper.db"
MARKET_DB = ROOT / "backtest_data" / "market_data.db"
BENCH = "NIFTYBEES"
EXCLUDE = {"NIFTYBEES", "NIFTY50", "BANKNIFTY", "INDIAVIX", "NIFTYJR", "NIFTYMID",
           "NIFTYIT", "FINNIFTY", "MIDCPNIFTY", "JUNIORBEES", "GOLDBEES", "SILVERBEES",
           "LIQUIDBEES", "BANKBEES", "ICICIB22", "CPSEETF", "MON100", "SETFNIF50"}


def _is_etf(sym):
    return sym.endswith("BEES") or sym.endswith("ETF") or sym.endswith("GOLD") \
        or sym.endswith("SILVER") or sym in EXCLUDE


CFG = dict(
    capital=1_000_000,        # ₹10 lakh
    n_hold=8,                 # max concurrent positions
    max_per_day=1,            # max NEW entries per day (the fine-filter winner)
    vol_mult=2.0,             # volume ≥ 2× 20-day average
    near_high=0.98,           # close ≥ 98% of 252-day high
    hi_lookback=252,
    turn_med_min=5.0,         # 20-day MEDIAN turnover ≥ ₹5cr
    price_min=20.0,
    chase_max_pct=15.0,       # skip a name up >15% today (unfillable gap/circuit chase)
    donchian=20,              # 20-day low trailing stop
    catastrophe=0.20,         # 20% hard stop from entry
    gate_sma=200,             # NIFTYBEES 200-day SMA regime gate for new entries
    cost_rt=0.002,            # 0.20% round-trip
    stcg_pct=0.20,            # short-term capital-gains tax, shown separately
    cash_yield=0.065,         # idle cash parked in a liquid fund @6.5% p.a.
    universe_size=500,        # working liquid universe refreshed daily from Kite
)

RULES = [
    ("Capital", "₹10,00,000 · up to 8 equal-weight positions (~₹1.25L each) · builds up as breakouts fire"),
    ("Entry signal", "Daily+Weekly+Monthly MACD line>0 · close ≥98% of 252-day high · volume ≥2× 20-day avg"),
    ("Liquidity", "20-day MEDIAN turnover ≥ ₹5cr · price ≥ ₹20 · skip names up >15% today (unfillable chase)"),
    ("Selection", "Rank the day's qualifiers by today's %-run → take the TOP 1 (max 1 new entry/day)"),
    ("Regime gate", "Take NEW entries only when NIFTYBEES > its 200-day SMA (does not force-liquidate)"),
    ("Exit (trailing)", "Close below the prior-20-day low (Donchian-20 trail) → exit. NO profit target."),
    ("Exit (catastrophe)", "Price ≤ 20% below entry → hard stop."),
    ("Idle cash", "Un-deployed cash parked in a liquid fund @ 6.5% p.a."),
    ("Costs", "Net of 0.20% round-trip; 20% STCG tracked & shown separately (not baked into NAV)"),
]


# ───────────────────────── DB ─────────────────────────
def _conn():
    c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row; return c


def init_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = _conn()
    c.executescript("""
      CREATE TABLE IF NOT EXISTS bp_positions(
        symbol TEXT PRIMARY KEY, qty REAL, entry_date TEXT, entry_price REAL,
        invested REAL, peak_price REAL);
      CREATE TABLE IF NOT EXISTS bp_closed(
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, entry_date TEXT, entry_price REAL,
        exit_date TEXT, exit_price REAL, qty REAL, gross_pnl REAL, gross_pct REAL,
        cost REAL, net_pnl REAL, reason TEXT, holding_days INTEGER, stcg_tax REAL);
      CREATE TABLE IF NOT EXISTS bp_fills(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, side TEXT,
        price REAL, qty REAL, value REAL, cost REAL, reason TEXT);
      CREATE TABLE IF NOT EXISTS bp_nav(
        d TEXT PRIMARY KEY, equity REAL, cash REAL, nav REAL, invested_pct REAL,
        gate TEXT, bench_close REAL, unrealized REAL);
      CREATE TABLE IF NOT EXISTS bp_state(key TEXT PRIMARY KEY, val TEXT);
    """)
    c.commit(); c.close()


def _get(key, default=None):
    c = _conn(); r = c.execute("SELECT val FROM bp_state WHERE key=?", (key,)).fetchone(); c.close()
    return json.loads(r["val"]) if r else default


def _set(key, val):
    c = _conn(); c.execute("INSERT OR REPLACE INTO bp_state(key,val) VALUES(?,?)",
                           (key, json.dumps(val))); c.commit(); c.close()


# ───────────────────── kite / data ─────────────────────
def _kite():
    from services.kite_service import get_kite_with_refresh
    return get_kite_with_refresh()


def _dm():
    from services.data_manager import get_data_manager
    return get_data_manager(_kite())


def _live_prices(symbols):
    if not symbols:
        return {}
    out = {}
    try:
        k = _kite()
        keys = [f"NSE:{s}" for s in symbols]
        for i in range(0, len(keys), 200):
            q = k.quote(keys[i:i + 200])
            for key, v in q.items():
                out[key.split(":", 1)[1]] = float(v.get("last_price") or
                                                   (v.get("ohlc") or {}).get("close") or 0)
    except Exception as e:
        logger.warning(f"[BP] live price fetch failed: {e}")
    return {s: p for s, p in out.items() if p > 0}


def _panel(start="2023-01-01"):
    """close & volume pivots (date × symbol). ~2.5y is enough for 252d-high + all MACDs."""
    con = sqlite3.connect(str(MARKET_DB))
    df = pd.read_sql("SELECT symbol,date,close,volume FROM market_data_unified "
                     "WHERE timeframe='day' AND close IS NOT NULL AND date>=? ORDER BY symbol,date",
                     con, params=(start,), parse_dates=["date"])
    con.close()
    close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
    vol = df.pivot_table(index="date", columns="symbol", values="volume").sort_index()
    return close, vol


def _working_universe(close, vol):
    """Top liquid names by trailing 20-day median turnover (the tradeable pool)."""
    tv = (close * vol)
    med = tv.tail(20).median().sort_values(ascending=False)
    pool = [s for s in med.index if not _is_etf(s) and med[s] / 1e7 >= CFG["turn_med_min"]]
    return pool[:CFG["universe_size"]]


def refresh_universe(full=False):
    """Pull fresh daily bars from Kite for the working liquid universe + held names + bench."""
    try:
        close, vol = _panel()
        pool = _working_universe(close, vol)
        held = [r["symbol"] for r in _conn().execute("SELECT symbol FROM bp_positions")]
        syms = sorted(set(pool + held + [BENCH]))
        frm = (date.today() - timedelta(days=20 if not full else 400))
        ok, fail, _ = _dm().download_data(syms, "day", datetime.combine(frm, datetime.min.time()),
                                          datetime.now())
        logger.info(f"[BP] refresh: {ok} ok / {fail} fail ({len(syms)} syms)")
        return ok
    except Exception as e:
        logger.error(f"[BP] refresh failed: {e}")
        return 0


# ───────────────────── signals ─────────────────────
def _macd(df):
    return df.ewm(span=12, adjust=False).mean() - df.ewm(span=26, adjust=False).mean()


def _scan(close, vol, asof):
    """Return today's qualifying breakouts as [(symbol, pct_run, turn_cr)] ranked by %-run desc."""
    uni = [s for s in _working_universe(close, vol) if s in close.columns]
    if not uni:
        return []
    h = close[uni].loc[:asof].ffill()
    v = vol[uni].loc[:asof]
    if len(h) < CFG["hi_lookback"]:
        return []
    md = _macd(h).iloc[-1]
    mw = _macd(h.resample("W-FRI").last()).iloc[-1]
    mm = _macd(h.resample("ME").last()).iloc[-1]
    vol20 = v.rolling(20).mean().iloc[-1]
    volspike = v.iloc[-1] / vol20
    hi252 = h.rolling(CFG["hi_lookback"], min_periods=200).max().iloc[-1]
    turn_med = (h * v).rolling(20).median().iloc[-1] / 1e7
    run = (h.iloc[-1] / h.iloc[-2] - 1) * 100
    last = h.iloc[-1]
    cands = []
    for s in uni:
        try:
            if (md[s] > 0 and mw[s] > 0 and mm[s] > 0 and volspike[s] >= CFG["vol_mult"]
                    and last[s] >= CFG["near_high"] * hi252[s] and last[s] >= CFG["price_min"]
                    and turn_med[s] >= CFG["turn_med_min"] and 0 < run[s] <= CFG["chase_max_pct"]):
                cands.append((s, round(float(run[s]), 2), round(float(turn_med[s]), 1)))
        except (KeyError, TypeError):
            continue
    cands.sort(key=lambda x: -x[1])
    return cands


def _gate_on(close, asof):
    """Risk-ON for new entries when NIFTYBEES > its 200-day SMA."""
    b = close[BENCH].loc[:asof].dropna() if BENCH in close.columns else pd.Series(dtype=float)
    return bool(len(b) >= CFG["gate_sma"] and b.iloc[-1] > b.tail(CFG["gate_sma"]).mean())


def _donchian_low(close, sym, asof):
    cs = close[sym].loc[:asof].dropna() if sym in close.columns else pd.Series(dtype=float)
    n = CFG["donchian"]
    return float(cs.iloc[-n - 1:-1].min()) if len(cs) > n else None


# ───────────────────── book ops ─────────────────────
def _positions():
    return {r["symbol"]: dict(r) for r in _conn().execute("SELECT * FROM bp_positions")}


def _cash():
    return float(_get("cash", 0.0))


def _record_fill(symbol, side, price, qty, reason):
    value = price * qty; cost = value * (CFG["cost_rt"] / 2)
    c = _conn()
    c.execute("INSERT INTO bp_fills(ts,symbol,side,price,qty,value,cost,reason) VALUES(?,?,?,?,?,?,?,?)",
              (datetime.now().isoformat(timespec="seconds"), symbol, side, price, qty, value, cost, reason))
    c.commit(); c.close()
    return cost


def _buy(symbol, price, rupees, d, reason):
    qty = rupees / price
    cost = _record_fill(symbol, "BUY", price, qty, reason)
    c = _conn()
    c.execute("INSERT OR REPLACE INTO bp_positions(symbol,qty,entry_date,entry_price,invested,peak_price) "
              "VALUES(?,?,?,?,?,?)", (symbol, qty, d, price, qty * price, price))
    c.commit(); c.close()
    _set("cash", _cash() - qty * price - cost)


def _sell(symbol, price, d, reason):
    pos = _positions().get(symbol)
    if not pos:
        return
    qty = pos["qty"]; cost = _record_fill(symbol, "SELL", price, qty, reason)
    gross = (price - pos["entry_price"]) * qty
    gpct = (price / pos["entry_price"] - 1) * 100
    hold = (date.fromisoformat(d[:10]) - date.fromisoformat(pos["entry_date"][:10])).days
    rt_cost = pos["invested"] * (CFG["cost_rt"] / 2) + cost
    stcg = gross * CFG["stcg_pct"] if (gross > 0 and hold < 365) else 0.0
    c = _conn()
    c.execute("INSERT INTO bp_closed(symbol,entry_date,entry_price,exit_date,exit_price,qty,"
              "gross_pnl,gross_pct,cost,net_pnl,reason,holding_days,stcg_tax) "
              "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (symbol, pos["entry_date"], pos["entry_price"], d, price, qty, gross, gpct,
               rt_cost, gross - rt_cost, reason, hold, stcg))
    c.execute("DELETE FROM bp_positions WHERE symbol=?", (symbol,))
    c.commit(); c.close()
    _set("cash", _cash() + qty * price - cost)


def _mark_nav(close, asof_iso, live=None):
    pos = _positions()
    syms = list(pos)
    px = live or (_live_prices(syms) if syms else {})
    equity = 0.0; unreal = 0.0
    c = _conn()
    for s, p in pos.items():
        pr = px.get(s) or p["entry_price"]
        equity += p["qty"] * pr
        unreal += (pr - p["entry_price"]) * p["qty"]
        if pr > p["peak_price"]:
            c.execute("UPDATE bp_positions SET peak_price=? WHERE symbol=?", (pr, s))
    c.commit(); c.close()
    cash = _cash(); nav = equity + cash
    bench = None
    try:
        bench = float(close[BENCH].loc[:asof_iso].dropna().iloc[-1])
    except Exception:
        pass
    c = _conn()
    c.execute("INSERT OR REPLACE INTO bp_nav(d,equity,cash,nav,invested_pct,gate,bench_close,unrealized) "
              "VALUES(?,?,?,?,?,?,?,?)",
              (asof_iso[:10], equity, cash, nav, (equity / nav * 100) if nav else 0,
               _get("gate", "ON"), bench, unreal))
    c.commit(); c.close()
    return nav


# ───────────────────── jobs ─────────────────────
def seed(force=False):
    """One-time: start all-cash ₹10L; the daily job builds the book as breakouts fire."""
    init_db()
    if _get("seeded") and not force:
        return {"ok": False, "msg": "already seeded"}
    refresh_universe(full=True)
    _set("capital", CFG["capital"]); _set("cash", CFG["capital"])
    _set("inception", date.today().isoformat()); _set("interest_earned", 0.0)
    _set("seeded", True)
    daily_job()   # take today's first entry if the market is risk-on and one qualifies
    return {"ok": True, "held": list(_positions()), "gate": _get("gate")}


def daily_job():
    """Mon-Fri EOD: accrue cash yield → exits (Donchian/catastrophe) → 1 new entry if gated-on."""
    if not _get("seeded"):
        return
    if _cash() > 0:
        new_cash = _cash() * (1 + CFG["cash_yield"]) ** (1 / 252)
        _set("interest_earned", round(_get("interest_earned", 0.0) + (new_cash - _cash()), 2))
        _set("cash", new_cash)
    refresh_universe(full=False)
    close, vol = _panel()
    asof = close.index[-1]; d = date.today().isoformat()
    gate_on = _gate_on(close, asof)
    _set("gate", "ON" if gate_on else "OFF")
    # 1) exits — Donchian-20 trail OR 20% catastrophe
    live = _live_prices(list(_positions()))
    for s in list(_positions()):
        p = _positions()[s]; pr = live.get(s)
        if pr is None:
            continue
        low = _donchian_low(close, s, asof)
        if pr <= p["entry_price"] * (1 - CFG["catastrophe"]):
            _sell(s, pr, d, "CATASTROPHE"); logger.info(f"[BP] catastrophe exit {s} @ {pr:.1f}")
        elif low is not None and pr < low:
            _sell(s, pr, d, "DONCHIAN"); logger.info(f"[BP] Donchian exit {s} @ {pr:.1f} (<20d low {low:.1f})")
    # 2) entry — max 1 new/day, only if risk-on and slots free
    if gate_on and len(_positions()) < CFG["n_hold"]:
        held = set(_positions())
        cands = [c for c in _scan(close, vol, asof) if c[0] not in held]
        added = 0
        per = (_cash() + 0) / max(1, CFG["n_hold"] - len(_positions()))  # fill remaining slots evenly
        for s, run, turn in cands:
            if added >= CFG["max_per_day"] or len(_positions()) >= CFG["n_hold"]:
                break
            pr = _live_prices([s]).get(s)
            if not pr:
                continue
            rupees = min(_cash(), CFG["capital"] / CFG["n_hold"])   # ~equal weight of full capital
            if rupees < 5000:
                break
            _buy(s, pr, rupees, d, "BREAKOUT")
            logger.info(f"[BP] entry {s} @ {pr:.1f} (run {run}% turn ₹{turn}cr)")
            added += 1
    _mark_nav(close, asof.isoformat(), live=_live_prices(list(_positions())))
    _set("last_daily", d)


# ───────────────────── API getters ─────────────────────
def get_state():
    init_db()
    pos = _positions()
    syms = list(pos)
    live = _live_prices(syms) if syms else {}
    cap = _get("capital", CFG["capital"]); cash = _cash()
    close = None
    try:
        close, _v = _panel()
    except Exception:
        pass
    holdings = []; equity = 0.0
    for s, p in pos.items():
        pr = live.get(s) or p["entry_price"]
        val = p["qty"] * pr; equity += val
        low = _donchian_low(close, s, close.index[-1]) if close is not None else None
        hold = (date.today() - date.fromisoformat(p["entry_date"][:10])).days
        cata = p["entry_price"] * (1 - CFG["catastrophe"])
        stop = max(low, cata) if low else cata
        holdings.append(dict(
            symbol=s, qty=round(p["qty"], 1), entry_date=p["entry_date"][:10],
            entry_price=round(p["entry_price"], 1), price=round(pr, 1),
            value=round(val), weight=0, pnl=round(val - p["invested"]),
            pnl_pct=round((pr / p["entry_price"] - 1) * 100, 1), days=hold,
            stop=round(stop, 1), stop_dist_pct=round((pr / stop - 1) * 100, 1) if stop else None))
    nav = equity + cash
    for h in holdings:
        h["weight"] = round(h["value"] / nav * 100, 1) if nav else 0
    holdings.sort(key=lambda x: -x["value"])
    if cash > 1000:
        _incep = _get("inception") or date.today().isoformat()
        holdings.append(dict(
            symbol="CASH", qty=None, entry_date=_incep[:10], entry_price=None, price=None,
            value=round(cash), weight=round(cash / nav * 100, 1) if nav else 0,
            pnl=round(_get("interest_earned", 0.0)), pnl_pct=round(CFG["cash_yield"] * 100, 1),
            is_cash=True, days=(date.today() - date.fromisoformat(_incep[:10])).days,
            stop=None, stop_dist_pct=None))
    navcurve = [dict(d=r["d"], nav=round(r["nav"]), bench=r["bench_close"], gate=r["gate"])
                for r in _conn().execute("SELECT * FROM bp_nav ORDER BY d")]
    closed = [dict(r) for r in _conn().execute(
        "SELECT * FROM bp_closed ORDER BY exit_date DESC, id DESC LIMIT 200")]
    stcg_open = sum(max(0.0, (live.get(s, p["entry_price"]) - p["entry_price"]) * p["qty"]) * CFG["stcg_pct"]
                    for s, p in pos.items())
    realized = sum(r["net_pnl"] for r in closed)
    stcg_booked = sum(r["stcg_tax"] for r in closed)
    # today's candidate breakouts (what the book WOULD enter next) + gate detail
    today_cands, gate_last, gate_sma, gate_gap, gate_on = [], None, None, None, True
    if close is not None:
        try:
            asof = close.index[-1]
            _c, _vv = _panel()
            gate_on = _gate_on(_c, asof)
            today_cands = [{"symbol": s, "run_pct": r, "turn_cr": t}
                           for s, r, t in _scan(_c, _vv, asof)[:10]]
            b = close[BENCH].dropna()
            gate_last = round(float(b.iloc[-1]), 2)
            gate_sma = round(float(b.tail(CFG["gate_sma"]).mean()), 2)
            gate_gap = round((gate_last / gate_sma - 1) * 100, 2)
        except Exception:
            pass
    return dict(
        seeded=bool(_get("seeded")), gate=_get("gate", "ON"), gate_on=gate_on,
        inception=_get("inception"), today_candidates=today_cands,
        gate_last=gate_last, gate_sma=gate_sma, gate_gap_pct=gate_gap,
        capital=cap, nav=round(nav), cash=round(cash), equity=round(equity),
        invested_pct=round(equity / nav * 100, 1) if nav else 0,
        total_return_pct=round((nav / cap - 1) * 100, 2) if cap else 0,
        unrealized=round(equity - sum(p["invested"] for p in pos.values())),
        realized_net=round(realized), n_holdings=len([h for h in holdings if not h.get("is_cash")]),
        interest_earned=round(_get("interest_earned", 0.0)), cash_yield_pct=CFG["cash_yield"] * 100,
        stcg_unbooked=round(stcg_open), stcg_booked=round(stcg_booked),
        last_daily=_get("last_daily"),
        data_asof=(close.index[-1].date().isoformat() if close is not None else None),
        holdings=holdings, navcurve=navcurve, closed=closed, rules=RULES)


# ───────────────────── register ─────────────────────
def register(app, scheduler):
    from flask import jsonify, request
    init_db()
    app.add_url_rule("/api/breakout-paper/state", "bp_state", lambda: jsonify(get_state()))
    app.add_url_rule("/api/breakout-paper/seed", "bp_seed",
                     lambda: jsonify(seed(bool((request.get_json(silent=True) or {}).get("force")))),
                     methods=["POST"])
    app.add_url_rule("/api/breakout-paper/run-daily", "bp_run_daily",
                     lambda: (daily_job() or jsonify({"ok": True})), methods=["POST"])
    scheduler.add_job(daily_job, "cron", day_of_week="mon-fri", hour=15, minute=45,
                      id="bp_daily", replace_existing=True)
    logger.info("[BP] registered /api/breakout-paper/* + daily(15:45 IST) PAPER job")
