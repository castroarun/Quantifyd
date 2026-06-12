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
from datetime import datetime, date, timedelta, time as dtime

import pandas as pd

from services.v2_breakout_signals import (
    combo_skip, breakout_state, prior_day_cpr_width_pct,
    v2_fly_legs, breakout_legs, atm_strike,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "backtest_data" / "v2_ironfly_trading.db"
NIFTY_TOKEN = 256265

# force_paper is the BREAK-GLASS master switch: while True, live is hard-disabled no matter
# what the app toggle says. We set it False so the PAPER/LIVE control on the app governs — but
# real trading still requires (a) persisted mode=='live' AND (b) armed=='1' (set only by an
# explicit Deploy click). Out of the box mode=paper + armed=0, so nothing trades until you act.
CFG = dict(force_paper=False, live_weekdays=(0, 1, 4), vix_floor=13.0,
           lots=10, lot_size=65, stop_pct=0.02, pt_pct=0.40,
           min_entry_dte=4, roll_dte=1, brokerage=20.0,
           slippage_pct=0.0025)   # 0.25% of premium — matches the AlgoTest backtest net basis
QTY = CFG["lots"] * CFG["lot_size"]   # 650

_scanner = None
_daily_cache = {}        # date_iso -> DataFrame
_instr_cache = {"day": None, "rows": None}
_or_cache = {}           # date_iso -> {open, high, low} opening-range (first 5 min)


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


def _opening_range():
    """Today's open + first-5-min (09:15–09:20) high/low from 1-min bars. Cached once complete.
    Used for the gap-day stop: when the session opens beyond the 2% band, we ignore the gap print,
    let this 5-min range form, then exit on a 1-min close beyond its high/low."""
    today = date.today().isoformat()
    if _or_cache.get("day") == today and _or_cache.get("val"):
        return _or_cache["val"]
    try:
        rows = _kite().historical_data(NIFTY_TOKEN, date.today(), date.today(), "minute")
        if not rows:
            return None
        first5 = [r for r in rows if r["date"].replace(tzinfo=None).time() < dtime(9, 20)]
        if not first5:
            return None
        val = {"open": float(rows[0]["open"]),
               "high": max(float(r["high"]) for r in first5),
               "low": min(float(r["low"]) for r in first5)}
        if len(first5) >= 5:                       # cache only once the 09:15–09:20 window is complete
            _or_cache.clear(); _or_cache.update(day=today, val=val)
        return val
    except Exception as e:
        logger.warning(f"[V2] opening-range fetch failed: {e}")
        return None


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


# Compression-gate thresholds (research/64; ~median = soft). Calm = VIX/ATR%/CPR LOW, Stoch HIGH.
# Compression = pure vol-structure (ATR+CPR+Stoch), soft pass = 2 of 3. VIX is a SEPARATE regime
# control: floor (premium richness) + hard-skip (disaster zone) — NOT a calm flag. research/64.
COMPRESSION = dict(atr_pct_max=1.1, cpr_d_max=0.16, stoch_min=65.0, soft_min_pass=2,
                   vix_floor=13.0, vix_skip=22.0)


def _compression():
    """SHADOW-ONLY today's reading from the last COMPLETED daily bar (causal) + live VIX.
    Compression (ATR/CPR/Stoch) = the calm/vol-structure gate (n_pass/3, comp_pass>=2).
    VIX is a separate regime: 'skip' (>22, disaster), 'below_floor' (<13, thin premium), 'ok' (13-22).
    would_enter = comp_pass AND vix_regime=='ok'. No trading impact."""
    df = _daily_df()
    if df is None or len(df) < 30:
        return None
    d = df[df.index.date < date.today()]           # drop today's partial bar -> prior completed day
    if len(d) < 20:
        d = df
    H, L, C = d["high"], d["low"], d["close"]; prevC = C.shift(1)
    tr = pd.concat([(H - L), (H - prevC).abs(), (L - prevC).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr_pct = float(atr14.iloc[-1] / C.iloc[-1] * 100)
    lo14, hi14 = L.rolling(14).min(), H.rolling(14).max()
    stoch = float(100 * (C.iloc[-1] - lo14.iloc[-1]) / (hi14.iloc[-1] - lo14.iloc[-1]))
    h, l, c = float(H.iloc[-1]), float(L.iloc[-1]), float(C.iloc[-1])
    piv = (h + l + c) / 3; bcv = (h + l) / 2; tcv = 2 * piv - bcv
    cpr_d = float(abs(tcv - bcv) / c * 100)
    vix = _vix()
    g = dict(atr=atr_pct < COMPRESSION["atr_pct_max"], cpr=cpr_d < COMPRESSION["cpr_d_max"],
             stoch=stoch > COMPRESSION["stoch_min"])
    npass = sum(g.values())
    comp_pass = npass >= COMPRESSION["soft_min_pass"]
    if vix is None:
        vix_regime = "unknown"
    elif vix > COMPRESSION["vix_skip"]:
        vix_regime = "skip"            # disaster zone — never enter
    elif vix < COMPRESSION["vix_floor"]:
        vix_regime = "below_floor"     # premium too thin (existing live floor)
    else:
        vix_regime = "ok"              # 13-22: tradeable
    return dict(vix=vix, vix_regime=vix_regime, atr_pct=round(atr_pct, 2), cpr_d=round(cpr_d, 3),
                stoch=round(stoch, 1), flags=g, n_pass=npass, comp_pass=comp_pass,
                would_enter=bool(comp_pass and vix_regime == "ok"))


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
      CREATE TABLE IF NOT EXISTS v2_settings (key TEXT PRIMARY KEY, val TEXT);
      CREATE TABLE IF NOT EXISTS v2_compression_log (
        day TEXT PRIMARY KEY, vix REAL, vix_regime TEXT, atr_pct REAL, cpr_d REAL, stoch REAL,
        n_pass INTEGER, comp_pass INTEGER, would_enter INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP );
    """)
    for _col in ("vix_regime TEXT", "comp_pass INTEGER"):
        try:
            c.execute(f"ALTER TABLE v2_compression_log ADD COLUMN {_col}")
        except Exception:
            pass
    try:
        c.execute("ALTER TABLE v2_positions ADD COLUMN series_json TEXT DEFAULT '[]'")
    except Exception:
        pass        # column already exists
    try:
        c.execute("ALTER TABLE v2_positions ADD COLUMN exit_day TEXT")
    except Exception:
        pass        # column already exists
    c.commit(); c.close()


# ---------- persisted mode / arming ----------
def _setting(key, default):
    c = _conn()
    c.execute("CREATE TABLE IF NOT EXISTS v2_settings (key TEXT PRIMARY KEY, val TEXT)")
    r = c.execute("SELECT val FROM v2_settings WHERE key=?", (key,)).fetchone()
    c.close()
    return r["val"] if r else default


def _set_setting(key, val):
    c = _conn()
    c.execute("CREATE TABLE IF NOT EXISTS v2_settings (key TEXT PRIMARY KEY, val TEXT)")
    c.execute("INSERT INTO v2_settings(key,val) VALUES(?,?) "
              "ON CONFLICT(key) DO UPDATE SET val=excluded.val", (key, str(val)))
    c.commit(); c.close()


def _mode():
    """Effective trading mode. force_paper hard-pins to 'paper' (break-glass)."""
    if CFG["force_paper"]:
        return "paper"
    return "live" if _setting("mode", "paper") == "live" else "paper"


def _armed():
    return _setting("armed", "0") == "1"


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


def _insert(system, spot, vix, expiry, legs, mode="paper"):
    dte = (expiry - date.today()).days
    c = _conn()
    cur = c.execute("INSERT INTO v2_positions (system,day,entry_time,entry_spot,entry_vix,expiry,dte_entry,"
              "legs_json,net_entry,status,mode) VALUES (?,?,?,?,?,?,?,?,?, 'OPEN',?)",
              (system, date.today().isoformat(), datetime.now().strftime("%H:%M:%S"), spot, vix,
               expiry.isoformat(), dte, json.dumps(legs), _net_premium(legs), mode))
    pid = cur.lastrowid
    c.execute("UPDATE v2_positions SET series_json=? WHERE id=?", (json.dumps([[_hhmm(), 0]]), pid))
    c.commit(); c.close()
    logger.info(f"[V2] {mode.upper()} ENTRY {system} spot={spot:.0f} vix={vix} net={_net_premium(legs):.1f} "
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
    live = pos.get("mode") == "live"
    if live:
        legs = _exit_live(legs)                # place reverse orders; lg["exit"] = real fill price
        per = sum(((l["entry"] - l["exit"]) if l["side"] == "SELL" else (l["exit"] - l["entry"])) for l in legs)
        pnl = per * QTY - len(legs) * 2 * CFG["brokerage"]   # real fills already include slippage
    else:
        for lg in legs:
            lg["exit"] = lg["ltp"]
        pnl = _pnl_now(legs)                    # paper: modeled brokerage + slippage
    xt = at_time or datetime.now().strftime("%H:%M:%S")   # candle label for move-stop, else wall-clock
    c = _conn()
    c.execute("UPDATE v2_positions SET status='CLOSED',exit_time=?,exit_day=?,exit_reason=?,exit_spot=?,"
              "net_exit=?,pnl=?,pnl_now=?,legs_json=? WHERE id=?",
              (xt, date.today().isoformat(), reason, spot, _net_premium(legs),
               round(pnl), round(pnl), json.dumps(legs), pos["id"]))
    c.commit(); c.close()
    logger.info(f"[V2] {'LIVE' if live else 'PAPER'} EXIT {pos['system']} ({reason}) pnl={pnl:.0f}")


# ---------- LIVE order execution (real money) ----------
def _place_order(sym, txn):
    """Place one MARKET NRML leg. Returns broker order_id."""
    return _kite().place_order(variety="regular", exchange="NFO", tradingsymbol=sym,
                               transaction_type=txn, quantity=QTY, product="NRML",
                               order_type="MARKET", validity="DAY")


def _fill_price(order_id, tries=20, pause=0.4):
    """Poll until the order is COMPLETE (avg price) or terminal-failed (None)."""
    import time as _t
    for _ in range(tries):
        try:
            for o in _kite().orders():
                if o["order_id"] == order_id:
                    if o["status"] == "COMPLETE":
                        return float(o["average_price"])
                    if o["status"] in ("REJECTED", "CANCELLED"):
                        logger.error(f"[V2] order {order_id} {o['status']}: {o.get('status_message')}")
                        return None
        except Exception as e:
            logger.warning(f"[V2] orders() poll failed: {e}")
        _t.sleep(pause)
    logger.error(f"[V2] order {order_id} did not confirm COMPLETE in time")
    return None


def _margin_check(legs):
    """(need, avail, ok) from Kite SPAN+exposure vs available equity margin. Fail-closed."""
    try:
        orders = [dict(exchange="NFO", tradingsymbol=lg["sym"], transaction_type=lg["side"],
                       variety="regular", product="NRML", order_type="MARKET", quantity=QTY)
                  for lg in legs]
        m = _kite().basket_order_margins(orders, consider_positions=True, mode="compact")
        need = float(m["final"]["total"])
        avail = float(_kite().margins("equity")["net"])
        return need, avail, (avail >= need)
    except Exception as e:
        logger.error(f"[V2] margin check failed (fail-closed): {e}")
        return None, None, False


def _enter_live(system, spot, vix, expiry, legs):
    """Place the fly for REAL: BUY wings first (margin benefit), then SELL the body.
    Any leg failure rolls back everything placed so far -> returns to flat, aborts."""
    need, avail, ok = _margin_check(legs)
    if not ok:
        logger.error(f"[V2] LIVE entry ABORTED: margin need={need} avail={avail}")
        return False
    ordered = sorted(legs, key=lambda l: 0 if l["side"] == "BUY" else 1)   # wings (BUY) first
    filled = []
    for lg in ordered:
        try:
            oid = _place_order(lg["sym"], lg["side"])
        except Exception as e:
            logger.error(f"[V2] LIVE place_order threw {lg['side']} {lg['sym']}: {e}")
            oid = None
        fp = _fill_price(oid) if oid else None
        if fp is None:
            logger.error(f"[V2] LIVE leg failed ({lg['side']} {lg['sym']}) — rolling back {len(filled)} fills")
            for f in filled:
                rev = "SELL" if f["side"] == "BUY" else "BUY"
                try:
                    _place_order(f["sym"], rev)
                except Exception as e:
                    logger.error(f"[V2] ROLLBACK FAILED {f['sym']}: {e} — MANUAL REVIEW")
            return False
        filled.append({**lg, "entry": fp, "ltp": fp, "exit": None, "order_id": oid})
    _insert(system, spot, vix, expiry, filled, mode="live")
    return True


def _exit_live(legs):
    """Square off each leg with the reverse MARKET order; set lg['exit'] to the real fill."""
    out = []
    for lg in legs:
        rev = "BUY" if lg["side"] == "SELL" else "SELL"
        try:
            oid = _place_order(lg["sym"], rev)
            fp = _fill_price(oid)
        except Exception as e:
            logger.error(f"[V2] LIVE exit place_order threw {lg['sym']}: {e}")
            fp = None
        out.append({**lg, "exit": fp if fp is not None else lg.get("ltp", lg["entry"])})
    return out


# ---------- scheduler jobs ----------
def _do_entry(force=False):
    """Core entry — paper or live per _mode(). Respects VIX/combo gates unless force=True.
    Returns {entered: bool, reason: str}. Shadow-logs skips."""
    if _open("v2"):
        return {"entered": False, "reason": "already in a position"}
    daily = _daily_df()
    if daily is None or len(daily) < 30:
        return {"entered": False, "reason": "no daily bars"}
    today = pd.Timestamp(date.today())
    spot = _spot(); vix = _vix()
    if spot is None or vix is None:
        return {"entered": False, "reason": "no live spot/VIX"}
    if not force:
        if vix < CFG["vix_floor"]:
            _shadow(today, [f"vix<{CFG['vix_floor']}({vix})"], spot, vix)
            return {"entered": False, "reason": f"VIX {vix} < {CFG['vix_floor']} floor"}
        skip, reasons = combo_skip(daily, today)
        if skip:
            _shadow(today, reasons, spot, vix)
            logger.info(f"[V2] entry SKIP {reasons} (shadow-logged)")
            return {"entered": False, "reason": "skip-filter: " + ",".join(reasons)}
    exp = _second_weekly()
    if not exp or (exp - date.today()).days < CFG["min_entry_dte"]:
        return {"entered": False, "reason": f"no 2nd-weekly with DTE≥{CFG['min_entry_dte']} (exp={exp})"}
    legs = _resolve(v2_fly_legs(spot), exp)
    if not legs:
        return {"entered": False, "reason": "could not resolve/quote legs"}
    if _mode() == "live":
        ok = _enter_live("v2", spot, vix, exp, legs)
        return {"entered": ok, "reason": "LIVE entry placed" if ok else "LIVE entry failed — see logs"}
    _insert("v2", spot, vix, exp, legs, mode="paper")
    return {"entered": True, "reason": "paper entry"}


def entry_job():
    """09:20 cron — auto-entry. In live mode this fires only when armed (post-Deploy)."""
    init_db()
    if _open("v2"):
        return
    if _mode() == "live" and not _armed():
        return                                   # live but disarmed: wait for an explicit Deploy
    _do_entry(force=False)


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
        # --- stop selection: normal day = 2% move-stop; gap-open day = opening-range breakout ---
        orng = _opening_range()
        gap_day = bool(orng and es and abs(orng["open"] - es) / es >= CFG["stop_pct"])
        if gap_day:
            # The session OPENED outside the 2% band. Ignore the gap print, take no stop action in
            # the first 5 min; from 09:20 exit on a 1-min close beyond the opening-range high/low.
            if datetime.now().time() > dtime(9, 20):
                if spot > orng["high"] or spot < orng["low"]:
                    _close(pos, "gap_or_break", spot, at_time=ctime); continue
            # else inside the opening range (or still in the first 5 min) -> hold
        else:
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
        # roll near expiry — pinned to the 15:15-15:30 window (avoid the volatile open / pre-open reject)
        exp = datetime.strptime(pos["expiry"], "%Y-%m-%d").date()
        if (exp - date.today()).days <= CFG["roll_dte"] and dtime(15, 15) <= datetime.now().time() <= dtime(15, 30):
            _close(pos, "roll", spot); continue


def compression_shadow_job():
    """09:25 daily — SHADOW ONLY: log today's compression reading (research/64 gate) for forward
    validation. Records what the soft compression filter WOULD do; changes no trading behaviour."""
    init_db()
    cs = _compression()
    if not cs:
        return
    c = _conn()
    c.execute("INSERT OR REPLACE INTO v2_compression_log (day,vix,vix_regime,atr_pct,cpr_d,stoch,n_pass,comp_pass,would_enter) "
              "VALUES (?,?,?,?,?,?,?,?,?)", (date.today().isoformat(), cs["vix"], cs["vix_regime"], cs["atr_pct"],
              cs["cpr_d"], cs["stoch"], cs["n_pass"], int(cs["comp_pass"]), int(cs["would_enter"])))
    c.commit(); c.close()
    logger.info(f"[V2-compression] {date.today()} comp={cs['n_pass']}/3 pass={cs['comp_pass']} "
                f"vix={cs['vix']}({cs['vix_regime']}) atr%={cs['atr_pct']} cpr={cs['cpr_d']} stoch={cs['stoch']} "
                f"would_enter={cs['would_enter']}")


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
    return {"system": system, "mode": _mode(),
            "open": pos, "closed_total_pnl": round(tot[0] or 0), "closed_trades": tot[1], "closed": closed}


def get_v2_state():
    s = _state("v2")
    c = _conn()
    sk = [dict(r) for r in c.execute("SELECT day,reasons,spot,vix FROM v2_shadow_skips ORDER BY id DESC LIMIT 60")]
    c.close()
    s["shadow_skips"] = sk
    try:
        s["compression_today"] = _compression()
        cc = _conn()
        s["compression_log"] = [dict(r) for r in cc.execute(
            "SELECT day,vix,vix_regime,atr_pct,cpr_d,stoch,n_pass,comp_pass,would_enter FROM v2_compression_log ORDER BY day DESC LIMIT 40")]
        cc.close()
    except Exception:
        s["compression_today"] = None; s["compression_log"] = []
    if s.get("open"):
        orng = _opening_range()
        es = s["open"].get("entry_spot")
        if orng and es:
            s["open"]["gap_day"] = bool(abs(orng["open"] - es) / es >= CFG["stop_pct"])
            s["open"]["or_high"] = round(orng["high"]); s["open"]["or_low"] = round(orng["low"])
    mode = _mode(); armed = _armed(); flat = s["open"] is None
    s["mode"] = mode
    s["armed"] = armed
    s["live_enabled"] = not CFG["force_paper"]
    s["deployable"] = (mode == "live" and flat)   # show Deploy/preview whenever live+flat
    s["config"] = {"vix_floor": CFG["vix_floor"], "wings": "2% of ATM (snapped to 50)", "stop": "2% move",
                   "pt": "40%", "lots": CFG["lots"], "qty": QTY,
                   "filter": "skip if prior-day CPR<0.10% OR inside-week",
                   "margin_est": "≈₹7.0L (₹70k/lot, Kite SPAN — floats with VIX)"}
    return s


def set_mode(mode):
    """Toggle PAPER <-> LIVE. Allowed only when flat. Never auto-arms."""
    init_db()
    if mode not in ("paper", "live"):
        return {"ok": False, "error": "mode must be 'paper' or 'live'"}
    if mode == "live" and CFG["force_paper"]:
        return {"ok": False, "error": "live is hard-disabled in code (force_paper=True)"}
    if _open("v2") or _open("breakout"):
        return {"ok": False, "error": "square off the open position before switching mode"}
    _set_setting("mode", mode)
    _set_setting("armed", "0")
    logger.warning(f"[V2] MODE -> {mode} (armed reset to 0)")
    return {"ok": True, "mode": mode, "armed": False}


def deploy(force=False):
    """Explicit go-live: ARM automation and place the first trade now. LIVE + flat only.
    After this, rolls and re-entries are automatic until kill-switch disarms."""
    init_db()
    if _mode() != "live":
        return {"ok": False, "error": "not in LIVE mode — toggle to LIVE first"}
    if _open("v2"):
        return {"ok": False, "error": "already in a position"}
    _set_setting("armed", "1")
    res = _do_entry(force=bool(force))
    res["ok"] = True
    res["armed"] = True
    logger.warning(f"[V2] DEPLOY(force={force}) entered={res.get('entered')} ({res.get('reason')})")
    return res


def get_preview():
    """Dry-run preview of the exact legs + margin + gate a Deploy would use right now."""
    init_db()
    if _mode() != "live":
        return {"ok": False, "error": "not in live mode"}
    if _open("v2"):
        return {"ok": False, "error": "already in a position"}
    spot = _spot(); vix = _vix(); exp = _second_weekly()
    if not spot or not exp:
        return {"ok": False, "error": "no live spot/expiry"}
    legs = _resolve(v2_fly_legs(spot), exp)
    if not legs:
        return {"ok": False, "error": "could not resolve/quote legs"}
    need, avail, ok = _margin_check(legs)
    gate = "—"
    daily = _daily_df()
    if daily is not None and len(daily) >= 30:
        if vix is not None and vix < CFG["vix_floor"]:
            gate = f"BLOCKED · VIX {vix} < {CFG['vix_floor']}"
        else:
            skip, reasons = combo_skip(daily, pd.Timestamp(date.today()))
            gate = ("BLOCKED · " + ",".join(reasons)) if skip else "PASS"
    return {"ok": True, "spot": round(spot), "vix": vix, "expiry": exp.isoformat(),
            "dte": (exp - date.today()).days, "net_credit": round(_net_premium(legs), 1),
            "legs": [{"side": l["side"], "instrument_type": l["instrument_type"],
                      "strike": l["strike"], "premium": round(l["entry"], 2)} for l in legs],
            "margin_need": round(need) if need else None,
            "margin_avail": round(avail) if avail else None, "margin_ok": ok, "gate": gate}


def get_breakout_state():
    s = _state("breakout")
    s["config"] = {"trigger": "first daily close beyond inside-week prior-week H/L",
                   "up": "call debit spread (runner)", "down": "broken-wing fly skewed down",
                   "note": "PAPER-ONLY experiment; bear side unvalidated"}
    return s


def kill_switch():
    """Close all open positions NOW (live = real reverse MARKET orders) and DISARM automation."""
    init_db(); spot = _spot()
    n = 0
    for system in ("v2", "breakout"):
        pos = _open(system)
        if pos:
            _refresh_marks(pos); _close(pos, "kill_switch", spot or pos["entry_spot"]); n += 1
    _set_setting("armed", "0")
    logger.warning(f"[V2] KILL-SWITCH: closed {n}, automation disarmed")
    return {"closed": n, "armed": False}


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
    from flask import jsonify, Response, request
    init_db()
    app.add_url_rule("/api/v2-ironfly/state", "v2_ironfly_state", lambda: jsonify(get_v2_state()))
    app.add_url_rule("/api/v2-breakout/state", "v2_breakout_state", lambda: jsonify(get_breakout_state()))
    app.add_url_rule("/api/v2-ironfly/scan", "v2_ironfly_scan", lambda: (entry_job() or jsonify({"ok": True})), methods=["POST"])
    app.add_url_rule("/api/v2-ironfly/kill-switch", "v2_ironfly_kill", lambda: jsonify(kill_switch()), methods=["POST"])
    app.add_url_rule("/api/v2-ironfly/mode", "v2_ironfly_mode",
                     lambda: jsonify(set_mode((request.get_json(silent=True) or {}).get("mode"))), methods=["POST"])
    app.add_url_rule("/api/v2-ironfly/deploy", "v2_ironfly_deploy",
                     lambda: jsonify(deploy(bool((request.get_json(silent=True) or {}).get("force")))), methods=["POST"])
    app.add_url_rule("/api/v2-ironfly/preview", "v2_ironfly_preview", lambda: jsonify(get_preview()))
    app.add_url_rule("/api/v2-ironfly/compression", "v2_ironfly_compression",
                     lambda: jsonify({"today": _compression(),
                                      "thresholds": COMPRESSION,
                                      "log": [dict(r) for r in _conn().execute(
                                          "SELECT day,vix,vix_regime,atr_pct,cpr_d,stoch,n_pass,comp_pass,would_enter "
                                          "FROM v2_compression_log ORDER BY day DESC LIMIT 120")]}))
    app.add_url_rule("/api/v2-ironfly/stream", "v2_ironfly_stream",
                     lambda: Response(stream(), mimetype="text/event-stream",
                                      headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}))
    scheduler.add_job(compression_shadow_job, "cron", day_of_week="mon-fri", hour=9, minute=25,
                      id="v2_compression_shadow", replace_existing=True)
    scheduler.add_job(entry_job, "cron", day_of_week="mon-fri", hour=9, minute=20,
                      id="v2_ironfly_entry", replace_existing=True)
    scheduler.add_job(monitor_job, "cron", day_of_week="mon-fri", hour="9-15", minute="*",
                      id="v2_ironfly_monitor", replace_existing=True)
    scheduler.add_job(breakout_job, "cron", day_of_week="mon-fri", hour=15, minute=20,
                      id="v2_breakout_entry", replace_existing=True)
    logger.info("[V2] registered: /api/v2-ironfly/* + /api/v2-breakout/* + entry(09:20)/monitor(3min)/breakout(15:20) PAPER jobs")
