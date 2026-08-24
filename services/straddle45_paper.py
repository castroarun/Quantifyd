#!/usr/bin/env python3
"""
45-DTE NIFTY short straddle — PAPER book (research/119).

Paper only. Places no orders and touches no live engine. It reads two read-only
sources and writes its own DB plus a static JSON the app page renders:

  nse_options_bhav (market_data.db)  real EOD option closes — entries, daily marks
  option_chain     (options_data.db) real 1-minute quotes, but only from ~27 DTE,
                                     so it can mark the BACK half of a hold live;
                                     it can never price a 45-DTE entry.

Rules, unchanged from the study:
  entry   expiry − 45 calendar days (roll back to a session), sell ATM CE + PE
  target  combined premium <= 50% of entry credit
  stop    combined premium >= 200% of entry credit
  time    expiry − 21 calendar days
  size    LOTS lots, fixed. No delta management, no rolling, no re-centring.

CLI:
  python3 services/straddle45_paper.py seed     # backtrace completed trades + open current
  python3 services/straddle45_paper.py mark     # mark to latest price, apply exits, publish
  python3 services/straddle45_paper.py show     # print state
"""
import json
import os
import sqlite3
import sys
from bisect import bisect_left
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MKT = os.path.join(ROOT, "backtest_data", "market_data.db")
OPT = os.path.join(ROOT, "backtest_data", "options_data.db")
DB = os.path.join(ROOT, "backtest_data", "straddle45_paper.db")
PUB = os.path.join(ROOT, "frontend", "public", "straddle45_paper.json")
PUB2 = os.path.join(ROOT, "static", "app", "straddle45_paper.json")

LOT = 65
LOTS = 3
QTY = LOT * LOTS
CAPITAL = 1_196_000.0          # 3%-adverse margin (2.69L/lot) x 3 + 2 x MaxDD x 3
TARGET, STOP = 0.50, 2.00
DTE_IN, DTE_OUT = 45, 21
SLIP = 0.0025
SEED_FROM = "2026-05-01"       # backtrace campaigns entered on/after this


# ------------------------------------------------------------------ helpers --
def ro(p):
    return sqlite3.connect("file:%s?mode=ro" % p, uri=True)


def costs_points(entry_prem, exit_prem):
    slip = SLIP * (entry_prem + exit_prem)
    stt = 0.0010 * entry_prem
    txn = 0.0005 * (entry_prem + exit_prem)
    brok = (20.0 * 4) / QTY
    return slip + stt + txn + brok + 0.18 * (txn + brok)


def dstr(d):
    return d.strftime("%Y-%m-%d")


def dparse(s):
    return datetime.strptime(s[:10], "%Y-%m-%d")


def init_db():
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS trades (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      expiry TEXT, strike REAL, entry_date TEXT, entry_spot REAL,
      credit REAL, qty INTEGER, lots INTEGER,
      exit_date TEXT, exit_prem REAL, exit_reason TEXT, exit_spot REAL,
      gross_pts REAL, cost_pts REAL, net_pts REAL, net_rs REAL,
      status TEXT,                       -- OPEN | CLOSED
      mark_prem REAL, mark_date TEXT, mark_src TEXT, mtm_rs REAL, mark_spot REAL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(expiry, entry_date));
    CREATE TABLE IF NOT EXISTS marks (
      trade_id INTEGER, d TEXT, prem REAL, src TEXT, mtm_rs REAL,
      UNIQUE(trade_id, d));
    """)
    con.commit()
    return con


# ------------------------------------------------------- market data access --
def sessions(con, start="2025-01-01"):
    return [r[0] for r in con.execute(
        "SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' "
        "AND trade_date>=? ORDER BY trade_date", (start,))]


def prev_session(days, target):
    """Latest session on/before target — or None if target is in the FUTURE.

    Returning the last known session for a future date silently collapses a
    not-yet-reached exit onto today, which closes positions early and invents
    entries for long-dated contracts. Unknown must stay unknown.
    """
    if not days or target > days[-1]:
        return None
    i = bisect_left(days, target)
    if i < len(days) and days[i] == target:
        return target
    return days[i - 1] if i > 0 else None


def monthly_expiries(con, days, start, end):
    """Monthly = last expiry of the month already listed 45 days out."""
    rows = con.execute(
        "SELECT expiry_date, MIN(trade_date) FROM nse_options_bhav WHERE symbol='NIFTY' "
        "AND expiry_date>=? AND expiry_date<=? GROUP BY expiry_date ORDER BY expiry_date",
        (start, end)).fetchall()
    out = {}
    for exp, first in rows:
        ed = prev_session(days, dstr(dparse(exp) - timedelta(days=DTE_IN)))
        if ed and first <= ed:
            out[exp[:7]] = exp
    return dict(sorted(out.items()))


def bhav_day(con, expiry, day):
    """{strike: (ce_close, pe_close, traded)} for one expiry on one session."""
    rows = con.execute(
        "SELECT strike, option_type, close, contracts FROM nse_options_bhav "
        "WHERE symbol='NIFTY' AND expiry_date=? AND trade_date=?", (expiry, day)).fetchall()
    d = {}
    for k, ot, c, ct in rows:
        d.setdefault(float(k), {})[ot] = (c or 0.0, ct or 0)
    return d


def spot_close(con, day):
    r = con.execute("SELECT close FROM market_data_unified WHERE symbol='NIFTY50' "
                    "AND timeframe='day' AND date LIKE ?||'%'", (day,)).fetchone()
    return float(r[0]) if r and r[0] else None


def pick_atm(day_chain, spot):
    best, bd = None, 1e18
    for k, legs in day_chain.items():
        if "CE" not in legs or "PE" not in legs:
            continue
        (ce, cec), (pe, pec) = legs["CE"], legs["PE"]
        if ce <= 0 or pe <= 0 or cec <= 0 or pec <= 0:
            continue
        if abs(k - spot) < bd:
            best, bd = k, abs(k - spot)
    return best


def combined_bhav(con, expiry, strike, day):
    ch = bhav_day(con, expiry, day)
    legs = ch.get(strike)
    if not legs or "CE" not in legs or "PE" not in legs:
        return None
    ce, pe = legs["CE"][0], legs["PE"][0]
    return (ce + pe) if ce > 0 and pe > 0 else None


def combined_kite_live(expiry, strike):
    """Live LTP straight from Kite for a currently-listed contract.

    The 1-minute recorder only covers a contract from ~27 DTE, so a fresh 45-DTE
    position has no recorded intraday history — but the contract IS listed and
    quoting, so the broker can still price it. Read-only: ltp(), no orders.
    """
    try:
        from kiteconnect import KiteConnect
        tokf = os.path.join(ROOT, "backtest_data", "access_token.json")
        api = os.environ.get("KITE_API_KEY")
        if not api or not os.path.exists(tokf):
            return None
        k = KiteConnect(api_key=api)
        k.set_access_token(json.load(open(tokf))["access_token"])
        mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        e = dparse(expiry)
        # monthly contracts use the YYMMM form, e.g. NIFTY26SEP24150CE
        root = "NIFTY%s%s%d" % (e.strftime("%y"), mon[e.month - 1], int(strike))
        keys = ["NFO:%sCE" % root, "NFO:%sPE" % root]
        q = k.ltp(keys)
        if len(q) != 2:
            return None
        ce = q[keys[0]]["last_price"]
        pe = q[keys[1]]["last_price"]
        if ce > 0 and pe > 0:
            return (ce + pe, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        return None
    return None


def kite_spot():
    """Live NIFTY 50 index level. Read-only."""
    try:
        from kiteconnect import KiteConnect
        tokf = os.path.join(ROOT, "backtest_data", "access_token.json")
        api = os.environ.get("KITE_API_KEY")
        if not api or not os.path.exists(tokf):
            return None
        k = KiteConnect(api_key=api)
        k.set_access_token(json.load(open(tokf))["access_token"])
        return k.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
    except Exception:
        return None


def combined_chain_live(expiry, strike):
    """Latest real 1-minute mark, if the recorder covers this contract."""
    if not os.path.exists(OPT):
        return None
    con = ro(OPT)
    try:
        rows = con.execute(
            "SELECT instrument_type, ltp, snapshot_time FROM option_chain "
            "WHERE expiry_date=? AND strike=? AND ltp IS NOT NULL AND ltp>0 "
            "ORDER BY snapshot_time DESC LIMIT 40", (expiry, strike)).fetchall()
    finally:
        con.close()
    ce = pe = None
    ts = None
    for it, ltp, t in rows:
        if it == "CE" and ce is None:
            ce, ts = float(ltp), t
        if it == "PE" and pe is None:
            pe, ts = float(ltp), t if ts is None else max(ts, t)
        if ce and pe:
            break
    return (ce + pe, ts) if ce and pe else None


# ------------------------------------------------------------------- engine --
def campaign_dates(days, expiry):
    """(entry session, exit session, exit calendar date).

    exit session is None while the 21-DTE date is still in the future — the
    position simply stays open and keeps marking.
    """
    e = dparse(expiry)
    ed = prev_session(days, dstr(e - timedelta(days=DTE_IN)))
    xcal = dstr(e - timedelta(days=DTE_OUT))
    return ed, prev_session(days, xcal), xcal


def seed():
    con = init_db()
    m = ro(MKT)
    days = sessions(m)
    today = dstr(datetime.now())
    exps = monthly_expiries(m, days, "2026-01-01", "2027-06-30")
    print("bhav sessions to %s | monthly expiries %s" % (days[-1], list(exps.values())))

    for ym, exp in exps.items():
        ed, xd, xcal = campaign_dates(days, exp)
        if not ed or ed < SEED_FROM:
            continue                      # entry not reached yet, or before the seed window
        if con.execute("SELECT 1 FROM trades WHERE expiry=? AND entry_date=?",
                       (exp, ed)).fetchone():
            continue
        ch = bhav_day(m, exp, ed)
        sp = spot_close(m, ed)
        if not ch or not sp:
            print("  skip %s entry %s — no data (bhav ends %s)" % (exp, ed, days[-1]))
            continue
        K = pick_atm(ch, sp)
        if K is None:
            print("  skip %s entry %s — no ATM with both legs traded" % (exp, ed))
            continue
        credit = combined_bhav(m, exp, K, ed)
        con.execute(
            "INSERT INTO trades(expiry,strike,entry_date,entry_spot,credit,qty,lots,status) "
            "VALUES(?,?,?,?,?,?,?,'OPEN')", (exp, K, ed, sp, credit, QTY, LOTS))
        print("  OPENED %s  entry %s  K %.0f  credit %.1f pts (Rs %s)"
              % (exp, ed, K, credit, "{:,.0f}".format(credit * QTY)))
    con.commit()
    con.close()
    m.close()


def mark():
    con = init_db()
    m = ro(MKT)
    days = sessions(m)
    last_bhav = days[-1]
    today = dstr(datetime.now())

    for t in con.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall():
        cols = [d[0] for d in con.execute("SELECT * FROM trades LIMIT 1").description]
        r = dict(zip(cols, t))
        exp, K, credit, ed = r["expiry"], r["strike"], r["credit"], r["entry_date"]
        _, xd, xcal = campaign_dates(days, exp)

        # walk every session since entry, applying the exits in order
        closed = False
        walk_to = min(xd, last_bhav) if xd else last_bhav
        for d in [x for x in days if ed < x <= walk_to]:
            prem = combined_bhav(m, exp, K, d)
            if prem is None:
                continue
            con.execute("INSERT OR REPLACE INTO marks(trade_id,d,prem,src,mtm_rs) VALUES(?,?,?,?,?)",
                        (r["id"], d, prem, "bhav", (credit - prem) * QTY))
            why = None
            if prem <= TARGET * credit:
                why = "TARGET"
            elif prem >= STOP * credit:
                why = "STOP"
            elif xd and d >= xd:
                why = "TIME_21DTE"
            if why:
                g = credit - prem
                c = costs_points(credit, prem)
                con.execute(
                    "UPDATE trades SET status='CLOSED',exit_date=?,exit_prem=?,exit_reason=?,exit_spot=?,"
                    "gross_pts=?,cost_pts=?,net_pts=?,net_rs=?,mark_prem=?,mark_date=?,mark_src=?,mtm_rs=? "
                    "WHERE id=?",
                    (d, prem, why, spot_close(m, d), g, c, g - c, (g - c) * QTY,
                     prem, d, "bhav", (g - c) * QTY, r["id"]))
                print("  CLOSED %s %s @ %.1f (%s) net %.1f pts = Rs %s"
                      % (exp, d, prem, why, g - c, "{:,.0f}".format((g - c) * QTY)))
                closed = True
                break
        if closed:
            continue

        # still open — mark as live as the data allows:
        #   1-min recorder (real, only from ~27 DTE) > live Kite LTP > last bhav close
        live = combined_chain_live(exp, K)
        if live:
            prem, ts = live
            src, mdate = "chain-1min", ts
        else:
            live = combined_kite_live(exp, K)
            if live:
                prem, mdate = live
                src = "kite-ltp"
            else:
                prem = combined_bhav(m, exp, K, last_bhav)
                src, mdate = "bhav", last_bhav
        if prem is not None:
            msp = kite_spot() if src != "bhav" else spot_close(m, last_bhav)
            con.execute(
                "UPDATE trades SET mark_prem=?,mark_date=?,mark_src=?,mtm_rs=?,mark_spot=? WHERE id=?",
                (prem, mdate, src, (credit - prem) * QTY, msp, r["id"]))
            print("  MARK %s %s %.1f (%s) MTM Rs %s"
                  % (exp, mdate, prem, src, "{:,.0f}".format((credit - prem) * QTY)))
    con.commit()
    publish(con, m, days)
    con.close()
    m.close()


def publish(con, m, days):
    cols = [d[0] for d in con.execute("SELECT * FROM trades LIMIT 1").description]
    rows = [dict(zip(cols, t)) for t in
            con.execute("SELECT * FROM trades ORDER BY entry_date").fetchall()]
    openp, closed = [], []
    for r in rows:
        _, xd, xcal = campaign_dates(days, r["expiry"])
        r["exit_due"] = xcal
        r["dte"] = (dparse(r["expiry"]) - datetime.now()).days
        if r["status"] == "OPEN":
            r["curve"] = [dict(d=d, prem=p, mtm=mt) for d, p, mt in con.execute(
                "SELECT d,prem,mtm_rs FROM marks WHERE trade_id=? ORDER BY d", (r["id"],))]
            openp.append(r)
        else:
            closed.append(r)
    realised = sum(x["net_rs"] or 0 for x in closed)
    unreal = sum(x["mtm_rs"] or 0 for x in openp)
    wins = [x for x in closed if (x["net_rs"] or 0) > 0]
    state = dict(
        asof=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        mode="PAPER", lots=LOTS, qty=QTY, capital=CAPITAL,
        bhav_through=days[-1],
        realised=realised, unrealised=unreal, nav=CAPITAL + realised + unreal,
        n_closed=len(closed), n_open=len(openp),
        win_rate=(100.0 * len(wins) / len(closed)) if closed else None,
        open_positions=openp, closed_trades=list(reversed(closed)),
    )
    for p in (PUB, PUB2):
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            json.dump(state, open(p, "w"), indent=1, default=str)
        except Exception as e:
            print("  publish %s failed: %s" % (p, e))
    print("published: %d open, %d closed, realised Rs %s, unrealised Rs %s"
          % (len(openp), len(closed), "{:,.0f}".format(realised), "{:,.0f}".format(unreal)))


def show():
    print(open(PUB).read() if os.path.exists(PUB) else "(not published yet)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "mark"
    {"seed": seed, "mark": mark, "show": show}[cmd]()
