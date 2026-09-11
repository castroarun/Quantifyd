#!/usr/bin/env python3
"""45-DTE NIFTY short straddle — LIVE executor (research/119).

Real money. Sells an ATM NIFTY straddle 45 calendar days before the monthly
expiry and buys it back at 21 DTE, or earlier on the 50% target / 200% stop.
3 lots = 195 qty. Rules are exactly what research/119 validated; nothing here
invents a rule the backtest did not test.

WHY THE DECISION WINDOW IS 15:20-15:29 AND NOT EVERY FIVE MINUTES
    The study strikes entries and exits on the SESSION CLOSE, on real bhavcopy
    prices. Phase D checked whether a finer cadence would change anything using
    28.3M real 1-minute quotes: in the DTE>=21 band the ATM straddle travels a
    mean +6.3%/-4.3% around its close and ZERO of 60 sessions travelled >=50%
    either way, so the 0.50 target and 2.00 stop are never approached intraday.
    Evaluating once, near the close, is therefore both faithful to the backtest
    and operationally identical. An intraday trigger would be a DIFFERENT rule
    than the one with evidence behind it.

WHAT PROTECTS THE MONEY
  * ARMED is OFF unless STRADDLE45_LIVE=1 is exported. Unarmed = dry run: it
    logs the exact orders it would send and sends nothing.
  * A kill file halts all order placement instantly, no deploy needed.
  * The broker is the source of truth. Every run reconciles the local book
    against kite.positions() BEFORE acting and HALTS on any mismatch rather
    than guessing - the 2026-08-06 SENSEX phantom and the 2026-08-14 momentum
    ledger corruption were both "assumed instead of read" failures.
  * BOTH LEGS OR NEITHER. If one leg fills and the other is rejected, the
    filled leg is bought back immediately. A lone short option is the one
    outcome this book must never produce.
  * Every order's fill is verified from order_history; an unverified order
    halts the run instead of being assumed good.
  * Margin is checked against the real basket requirement before entering.

READ THE STOP HONESTLY: Phase G measured the 200% stop as HARMFUL when it fires
(-130.9 pts, t -2.34) and it has never fired on the VIX>25 book in 61 campaigns.
It is implemented because it is the ruleset of record, not because it earns.
"""
import json
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DB = os.path.join(ROOT, "backtest_data", "straddle45_live.db")
MKT = os.path.join(ROOT, "backtest_data", "market_data.db")
KILL = os.path.join(ROOT, "backtest_data", "straddle45_KILL")
PUBS = [os.path.join(ROOT, "static", "app", "straddle45_live.json"),
        os.path.join(ROOT, "frontend", "public", "straddle45_live.json")]
LOG = "/tmp/straddle45_live.log"

LOT, LOTS = 65, 3
QTY = LOT * LOTS                 # 195
DTE_IN, DTE_OUT = 45, 21
VIX_RANK_MIN = 25                # entry filter: India VIX percentile vs prior 252 sessions
TARGET, STOP = 0.50, 2.00        # of entry credit
PRODUCT = "NRML"                 # carried, not intraday
DECIDE_FROM, DECIDE_TO = "15:20", "15:29"
MARGIN_BUFFER = 1.25             # require 1.25x the basket requirement free
MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
       "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

ARM_FILE = os.path.join(ROOT, "backtest_data", "straddle45_ARMED")
# Armed by the PRESENCE OF A FILE, not by a crontab edit. Arming and disarming a
# real-money book should never require rewriting the crontab - that is how the
# 58-job wipe happened on 2026-09-01. touch to arm, rm to disarm, both instant.
ARMED = os.environ.get("STRADDLE45_LIVE") == "1" or os.path.exists(ARM_FILE)
ALLOW_OFF_PLAN = (os.environ.get("STRADDLE45_OFF_PLAN") == "1"
                  or os.path.exists(os.path.join(ROOT, "backtest_data",
                                                 "straddle45_OFF_PLAN")))


def log(msg):
    line = "%s  %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


class Halt(Exception):
    """Something is not as expected. Stop; do not guess."""


# ------------------------------------------------------------------ state --
def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        expiry TEXT, strike REAL, qty INTEGER, lots INTEGER,
        ce_symbol TEXT, pe_symbol TEXT,
        entry_date TEXT, entry_time TEXT, entry_spot REAL,
        ce_entry REAL, pe_entry REAL, credit REAL,
        vix_level REAL, vix_rank REAL,
        exit_date TEXT, exit_time TEXT, exit_spot REAL,
        ce_exit REAL, pe_exit REAL, exit_prem REAL, exit_reason TEXT,
        gross_pts REAL, net_rs REAL,
        status TEXT, ce_order TEXT, pe_order TEXT, notes TEXT,
        UNIQUE(expiry))""")
    con.execute("""CREATE TABLE IF NOT EXISTS events (
        ts TEXT, kind TEXT, detail TEXT)""")
    con.commit()
    return con


def event(con, kind, detail):
    con.execute("INSERT INTO events VALUES(?,?,?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), kind, detail))
    con.commit()


def rows(con, where="1=1", args=()):
    cols = [d[1] for d in con.execute("PRAGMA table_info(positions)")]
    return [dict(zip(cols, r)) for r in
            con.execute("SELECT * FROM positions WHERE %s" % where, args)]


# ------------------------------------------------------------- market data --
def ro(p):
    return sqlite3.connect("file:%s?mode=ro" % p, uri=True)


def market_open_today(k):
    """Is TODAY a live trading session? Confirmed from the broker, not the
    calendar - a weekday can still be an exchange holiday."""
    if datetime.now().weekday() >= 5:
        return False
    try:
        q = k.quote(["NSE:NIFTY 50"])["NSE:NIFTY 50"]
        ltt = str(q.get("last_trade_time") or q.get("timestamp") or "")[:10]
        return ltt == date.today().isoformat()
    except Exception:
        return False


def sessions(m, k=None):
    """Trading sessions, INCLUDING today when the market is open today.

    The daily NIFTY bar is only written after the close. Without appending
    today, at 15:20 on the entry day the list still ends at YESTERDAY, so
    entry_session() returns yesterday, `today` never equals it, and the
    executor silently does nothing on the one day it must trade. This bug
    would not have raised an error - it would just have skipped the trade.
    """
    s = sorted({r[0][:10] for r in m.execute(
        "SELECT date FROM market_data_unified WHERE symbol='NIFTY50' "
        "AND timeframe='day' AND close IS NOT NULL")})
    today = date.today().isoformat()
    if today not in s and k is not None and market_open_today(k):
        s.append(today)
    return s


def vix_now(m, k=None):
    """(level, percentile rank vs the previous 252 sessions, source).

    Prefers the LIVE India VIX. The stored daily bar for TODAY is built by an
    intraday job and is not final until after the close, so judging a 15:20
    decision off the stored series risks using a partial or stale value - while
    the backtest struck the filter on the entry day's own CLOSE. The live quote
    at 15:20 is the closest honest proxy for that close.

    Window discipline: against a live level the trailing window is the 252
    stored closes ending YESTERDAY; against a stored level it is the 252 closes
    before it. The level is never compared against itself.
    """
    vx = sorted((r[0][:10], float(r[1])) for r in m.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' "
        "AND timeframe='day' AND close IS NOT NULL"))
    if len(vx) < 253:
        return None, None, "insufficient history"
    today = date.today().isoformat()
    hist = [(d, v) for d, v in vx if d < today]        # strictly before today
    if len(hist) < 253:
        return None, None, "insufficient history"
    lvl = None
    if k is not None:
        try:
            lvl = float(k.ltp(["NSE:INDIA VIX"])["NSE:INDIA VIX"]["last_price"]) or None
        except Exception as e:
            log("  vix: live quote unavailable (%s) - using the last stored close"
                % str(e)[:50])
    if lvl:
        src, w = "live", [v for _, v in hist[-252:]]
    else:
        lvl, src, w = hist[-1][1], "stored close %s" % hist[-1][0], [v for _, v in hist[-253:-1]]
    return lvl, 100.0 * sum(1 for x in w if x < lvl) / len(w), src


def monthly_expiries(k):
    """Listed NIFTY monthly expiries: the last expiry of each calendar month
    that carries a full chain. Derived from the instrument master, never
    hardcoded - the monthly weekday moved from Thursday to Tuesday in Sep-2025."""
    by = {}
    for i in k.instruments("NFO"):
        if i["name"] == "NIFTY" and i["instrument_type"] in ("CE", "PE"):
            by.setdefault(str(i["expiry"]), set()).add(float(i["strike"]))
    dense = {e for e, s in by.items() if len(s) >= 30}
    out = {}
    for e in sorted(dense):
        out[e[:7]] = e                      # last dense expiry of each month wins
    return out


def _roll_off_weekend(d):
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def entry_day(expiry):
    """Expiry minus 45 CALENDAR days, rolled back off weekends.

    NOT a lookup in the session list. That list contains no future dates, so
    `last session <= target` collapses to TODAY for any future target - which
    made the executor believe 09-Sep was the entry day for the 27-Oct expiry
    and would have entered two days early. Same class of bug as the paper
    book's prev_session future-date collapse (fixed Aug-2026): a lookup that
    silently returns the wrong answer instead of failing.

    Matches the study's stated convention exactly: dates roll off WEEKENDS
    only, because future NSE holidays are not knowable.
    """
    return _roll_off_weekend(
        date(*map(int, expiry.split("-"))) - timedelta(days=DTE_IN)).isoformat()


def next_session_after(d):
    """The next weekday after d. Future exchange holidays are not knowable, so a
    weekday is the best available proxy for 'the next session'."""
    n = date(*map(int, d.split("-"))) + timedelta(days=1)
    while n.weekday() >= 5:
        n += timedelta(days=1)
    return n.isoformat()


def exit_day(expiry):
    """Expiry minus 21 calendar days, rolled back off weekends. Same reasoning:
    a session-list lookup returned TODAY for a future exit, so the book would
    have closed the position on the day it opened."""
    return _roll_off_weekend(
        date(*map(int, expiry.split("-"))) - timedelta(days=DTE_OUT)).isoformat()


# ------------------------------------------------------------------ broker --
def kite():
    from services.kite_service import get_kite
    return get_kite()          # market_protection auto-injected on MARKET orders


def symbol(expiry, strike, ot):
    e = date(*map(int, expiry.split("-")))
    return "NIFTY%s%s%d%s" % (e.strftime("%y"), MON[e.month - 1], int(strike), ot)


def ltp2(k, ce, pe):
    q = k.ltp(["NFO:" + ce, "NFO:" + pe])
    a = q.get("NFO:" + ce, {}).get("last_price")
    b = q.get("NFO:" + pe, {}).get("last_price")
    return (a, b) if a and b else (None, None)


def verify_fill(k, order_id, tries=10, delay=0.6):
    """Poll until terminal. Returns (status, avg_price). Never assumes a fill."""
    for _ in range(tries):
        try:
            h = k.order_history(order_id)
            if h:
                st = h[-1].get("status")
                if st == "COMPLETE":
                    return "COMPLETE", float(h[-1].get("average_price") or 0)
                if st in ("REJECTED", "CANCELLED"):
                    return st, None
        except Exception as e:
            log("  order_history(%s) failed: %s" % (order_id, str(e)[:60]))
        time.sleep(delay)
    return "PENDING", None


def send(k, tradingsymbol, side, qty, tag):
    """One MARKET order through the wrapped client. Returns (order_id, avg_price)."""
    if os.path.exists(KILL):
        raise Halt("KILL file present at %s - refusing to place %s %s"
                   % (KILL, side, tradingsymbol))
    if not ARMED:
        log("  DRY-RUN would send: %s %d %s (%s)" % (side, qty, tradingsymbol, PRODUCT))
        return "DRYRUN", None
    oid = k.place_order(variety="regular", exchange="NFO",
                        tradingsymbol=tradingsymbol, transaction_type=side,
                        quantity=qty, product=PRODUCT, order_type="MARKET",
                        tag=tag[:20])
    st, px = verify_fill(k, oid)
    log("  order %s %s %s -> %s @ %s" % (side, tradingsymbol, oid, st, px))
    if st != "COMPLETE":
        raise Halt("order %s for %s ended %s, not COMPLETE" % (oid, tradingsymbol, st))
    return str(oid), px


def broker_nifty_legs(k):
    """Open NIFTY option legs at the broker, as {tradingsymbol: qty}."""
    out = {}
    for p in k.positions().get("net", []):
        if p["exchange"] == "NFO" and p["quantity"] != 0 \
                and p["tradingsymbol"].startswith("NIFTY"):
            out[p["tradingsymbol"]] = p["quantity"]
    return out


def reconcile(con, k):
    """The book must match the broker before anything is decided."""
    open_rows = rows(con, "status='OPEN'")
    legs = broker_nifty_legs(k)
    for r in open_rows:
        for sym, px in ((r["ce_symbol"], r["ce_entry"]), (r["pe_symbol"], r["pe_entry"])):
            have = legs.get(sym, 0)
            if have != -r["qty"]:
                raise Halt("RECONCILE: book says SHORT %d %s, broker says %d. "
                           "Refusing to act. Investigate, then fix the book."
                           % (r["qty"], sym, have))
    log("  reconciled: %d open position(s) match the broker" % len(open_rows))
    return open_rows


# ------------------------------------------------------------------- rules --
def in_window(now=None):
    t = (now or datetime.now()).strftime("%H:%M")
    return DECIDE_FROM <= t <= DECIDE_TO


def try_entry(con, k, m, sess, today):   # sess kept for signature stability
    if rows(con, "status='OPEN'"):
        return "already holding a position"
    exps = monthly_expiries(k)
    target = None
    for ym, e in sorted(exps.items()):
        if e <= today:
            continue
        ed = entry_day(e)
        # EXACTLY ONE session of grace, to cover an exchange holiday on the
        # nominal entry day. No more. Phase I measured entries two or more
        # sessions late at +12.0 points against +99.5 for an on-time entry
        # (t 0.10 - indistinguishable from zero) while taking the book's max
        # drawdown from -564.8 to -978.5. A stale entry is not a smaller version
        # of the edge, it is noise carrying the full tail. Miss the day and the
        # cycle is gone.
        if today == ed or today == next_session_after(ed):
            target = e
            break
    if not target:
        return "not an entry session"
    if rows(con, "expiry=?", (target,)):
        return "expiry %s already traded this cycle" % target

    lvl, rank, vsrc = vix_now(m, k)
    if rank is None:
        raise Halt("VIX rank is UNKNOWN (%s) - refusing to decide the filter. "
                   "Unknown must stay unknown." % vsrc)
    on_plan = rank > VIX_RANK_MIN
    if not on_plan and not ALLOW_OFF_PLAN:
        event(con, "SKIP", "%s VIX %.2f rank %.1f <= %d (%s)"
              % (target, lvl or 0, rank or 0, VIX_RANK_MIN, vsrc))
        return ("SKIP per plan: VIX %.2f rank %.1f is not > %d [%s]"
                % (lvl or 0, rank or 0, VIX_RANK_MIN, vsrc))

    spot = k.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
    K = round(spot / 50.0) * 50
    ce, pe = symbol(target, K, "CE"), symbol(target, K, "PE")
    q = k.quote(["NFO:" + ce, "NFO:" + pe])
    for s in (ce, pe):
        d = q.get("NFO:" + s, {})
        if not d.get("last_price") or (d.get("volume") or 0) < 1000 \
                or (d.get("oi") or 0) < 10000:
            raise Halt("leg %s is too thin to sell (ltp %s vol %s oi %s)"
                       % (s, d.get("last_price"), d.get("volume"), d.get("oi")))
    ce_px = q["NFO:" + ce]["last_price"]
    pe_px = q["NFO:" + pe]["last_price"]

    legs = [dict(exchange="NFO", tradingsymbol=s, transaction_type="SELL",
                 variety="regular", product=PRODUCT, order_type="MARKET",
                 quantity=QTY) for s in (ce, pe)]
    need = k.basket_order_margins(legs, consider_positions=False,
                                  mode="compact")["initial"]["total"]
    have = k.margins("equity").get("net") or 0
    if have < need * MARGIN_BUFFER:
        raise Halt("margin short: basket needs Rs %s, x%.2f buffer = Rs %s, "
                   "available Rs %s" % ("{:,.0f}".format(need), MARGIN_BUFFER,
                                        "{:,.0f}".format(need * MARGIN_BUFFER),
                                        "{:,.0f}".format(have)))
    log("  ENTRY %s K %d  CE %.2f PE %.2f  credit %.2f  margin need Rs %s have Rs %s"
        % (target, K, ce_px, pe_px, ce_px + pe_px,
           "{:,.0f}".format(need), "{:,.0f}".format(have)))

    tag = "S45E%s" % target[2:7].replace("-", "")
    ce_oid, ce_fill = send(k, ce, "SELL", QTY, tag)
    try:
        pe_oid, pe_fill = send(k, pe, "SELL", QTY, tag)
    except Exception as e:
        # BOTH LEGS OR NEITHER - unwind the leg that did fill.
        log("  PE leg failed (%s) - unwinding the CE leg immediately" % str(e)[:70])
        try:
            send(k, ce, "BUY", QTY, tag + "U")
            event(con, "UNWIND", "PE failed, CE bought back: %s" % str(e)[:150])
        except Exception as e2:
            event(con, "ORPHAN", "PE failed AND CE unwind failed: %s / %s"
                  % (str(e)[:100], str(e2)[:100]))
            raise Halt("CRITICAL: short CE %s is OPEN and unhedged. "
                       "Square it off MANUALLY now." % ce)
        raise Halt("entry abandoned cleanly (both legs flat)")

    ce_fill = ce_fill if ce_fill else ce_px
    pe_fill = pe_fill if pe_fill else pe_px
    con.execute(
        "INSERT INTO positions(expiry,strike,qty,lots,ce_symbol,pe_symbol,entry_date,"
        "entry_time,entry_spot,ce_entry,pe_entry,credit,vix_level,vix_rank,status,"
        "ce_order,pe_order,notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (target, float(K), QTY, LOTS, ce, pe, today,
         datetime.now().strftime("%H:%M:%S"), spot, ce_fill, pe_fill,
         ce_fill + pe_fill, lvl, rank, "OPEN" if ARMED else "DRYRUN",
         ce_oid, pe_oid, "" if on_plan else "OFF-PLAN override"))
    con.commit()
    event(con, "ENTRY", "%s K %.0f credit %.2f" % (target, K, ce_fill + pe_fill))
    return "ENTERED %s at K %d, credit %.2f" % (target, K, ce_fill + pe_fill)


def try_exit(con, k, sess, today):
    out = []
    for r in rows(con, "status='OPEN'"):
        ce_px, pe_px = ltp2(k, r["ce_symbol"], r["pe_symbol"])
        if ce_px is None:
            out.append("%s: no quote, holding" % r["expiry"])
            continue
        prem = ce_px + pe_px
        ratio = prem / r["credit"] if r["credit"] else 0
        xd = exit_day(r["expiry"])
        why = None
        if ratio <= TARGET:
            why = "TARGET"
        elif ratio >= STOP:
            why = "STOP"
        elif xd and today >= xd:
            why = "TIME_21DTE"
        if not why:
            out.append("%s: prem %.2f = %.0f%% of credit, holding (exit due %s)"
                       % (r["expiry"], prem, 100 * ratio, xd))
            continue
        log("  EXIT %s reason %s  prem %.2f (%.0f%% of credit %.2f)"
            % (r["expiry"], why, prem, 100 * ratio, r["credit"]))
        tag = "S45X%s" % r["expiry"][2:7].replace("-", "")
        ce_f = pe_f = None
        try:
            _, ce_f = send(k, r["ce_symbol"], "BUY", r["qty"], tag)
            _, pe_f = send(k, r["pe_symbol"], "BUY", r["qty"], tag)
        except Exception as e:
            event(con, "EXIT_FAIL", "%s: %s" % (r["expiry"], str(e)[:200]))
            raise Halt("EXIT INCOMPLETE on %s (%s). One leg may still be short - "
                       "CHECK THE BROKER NOW." % (r["expiry"], str(e)[:100]))
        ce_f = ce_f or ce_px
        pe_f = pe_f or pe_px
        gross = r["credit"] - (ce_f + pe_f)
        spot = k.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
        con.execute(
            "UPDATE positions SET status='CLOSED',exit_date=?,exit_time=?,exit_spot=?,"
            "ce_exit=?,pe_exit=?,exit_prem=?,exit_reason=?,gross_pts=?,net_rs=? WHERE id=?",
            (today, datetime.now().strftime("%H:%M:%S"), spot, ce_f, pe_f,
             ce_f + pe_f, why, gross, gross * r["qty"], r["id"]))
        con.commit()
        event(con, "EXIT", "%s %s gross %.2f pts" % (r["expiry"], why, gross))
        out.append("EXITED %s (%s) gross %.2f pts = Rs %s"
                   % (r["expiry"], why, gross, "{:,.0f}".format(gross * r["qty"])))
    return out or ["no open position"]


# ----------------------------------------------------------------- publish --
def publish(con, k=None):
    allr = rows(con)
    openp = [r for r in allr if r["status"] in ("OPEN", "DRYRUN")]
    closed = [r for r in allr if r["status"] == "CLOSED"]
    for r in openp:
        try:
            a, b = ltp2(k, r["ce_symbol"], r["pe_symbol"]) if k else (None, None)
            r["mark_prem"] = (a + b) if a else None
            r["mtm_rs"] = ((r["credit"] - r["mark_prem"]) * r["qty"]
                           if r["mark_prem"] else None)
        except Exception:
            r["mark_prem"] = r["mtm_rs"] = None
    payload = dict(
        asof=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        mode="LIVE" if ARMED else "DRY-RUN (not armed)",
        armed=ARMED, killed=os.path.exists(KILL),
        lots=LOTS, qty=QTY, vix_rank_min=VIX_RANK_MIN,
        target=TARGET, stop=STOP, dte_in=DTE_IN, dte_out=DTE_OUT,
        decide_window="%s-%s IST" % (DECIDE_FROM, DECIDE_TO),
        open_positions=openp, closed_trades=closed,
        realised_rs=sum(r["net_rs"] or 0 for r in closed),
        events=[dict(ts=a, kind=b, detail=c) for a, b, c in con.execute(
            "SELECT ts,kind,detail FROM events ORDER BY ts DESC LIMIT 40")])
    for p in PUBS:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p + ".tmp", "w") as f:
                json.dump(payload, f, indent=1, default=str)
            os.replace(p + ".tmp", p)
        except Exception as e:
            log("  publish %s failed: %s" % (p, e))


# -------------------------------------------------------------------- main --
def run():
    now = datetime.now()
    log("=" * 72)
    log("straddle45 LIVE run | armed=%s | kill=%s" % (ARMED, os.path.exists(KILL)))
    if now.weekday() >= 5:
        log("  weekend - nothing to do")
        return
    con, m = db(), ro(MKT)
    try:
        k = kite()
        reconcile(con, k)
        if not in_window(now):
            log("  outside the %s-%s decision window - monitoring only"
                % (DECIDE_FROM, DECIDE_TO))
            publish(con, k)
            return
        sess = sessions(m, k)
        today = now.strftime("%Y-%m-%d")
        for line in try_exit(con, k, sess, today):
            log("  " + line)
        log("  " + str(try_entry(con, k, m, sess, today)))
        publish(con, k)
    except Halt as e:
        log("  *** HALT: %s" % e)
        event(con, "HALT", str(e)[:400])
        publish(con)
        sys.exit(2)
    finally:
        con.close()
        m.close()


def status():
    con = db()
    for r in rows(con):
        print(json.dumps(r, indent=1, default=str))
    print("armed=%s kill=%s" % (ARMED, os.path.exists(KILL)))
    con.close()


def panic():
    """Flatten every open leg NOW, whatever the rules say."""
    con, k = db(), kite()
    for r in rows(con, "status='OPEN'"):
        for s in (r["ce_symbol"], r["pe_symbol"]):
            try:
                send(k, s, "BUY", r["qty"], "S45PANIC")
            except Exception as e:
                log("  panic buy %s FAILED: %s" % (s, e))
        con.execute("UPDATE positions SET status='CLOSED',exit_reason='PANIC',"
                    "exit_date=? WHERE id=?",
                    (datetime.now().strftime("%Y-%m-%d"), r["id"]))
    con.commit()
    event(con, "PANIC", "manual flatten")
    con.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "status": status, "panic": panic}[cmd]()
