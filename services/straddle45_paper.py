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
VIX_RANK_MIN = 25              # the PLAN: enter only when India VIX ranks above the
                               # 25th percentile of the previous 252 sessions.
                               # Campaigns below it are still traded on paper but
                               # tagged OFF-PLAN, so the filter's value is measured
                               # live instead of assumed.
IDLE_ETF = "LIQUID1"           # Kotak Nifty 1D Rate Liquid ETF - growth structure,
                               # 5.11% measured, deepest growth liquid ETF not already
                               # used (LIQUIDCASE is pledged, CASHIETF is Momentum's).


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
      vix_level REAL, vix_rank REAL, on_plan INTEGER,
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


def _monthly_weekday(cands):
    """The weekday NIFTY monthlies currently expire on, learned from the data.

    NSE moved the monthly from the last Thursday to the last Tuesday in Sep-2025,
    and may move it again — so this is derived, never hardcoded.
    """
    recent = [e for e in sorted(cands) if e >= "2025-10-01"][-8:]
    if not recent:
        return None
    wd = {}
    for e in recent:
        w = dparse(e).weekday()
        wd[w] = wd.get(w, 0) + 1
    return max(wd, key=wd.get)


def monthly_expiries(con, days, start, end):
    """Monthly = last expiry of the month already listed 45 days out, on the
    prevailing monthly weekday.

    The weekday guard matters: legacy far-dated contracts (2026-12-31, 2027-06-24,
    2027-12-30 — all Thursdays, listed years ahead) survive alongside the real
    monthlies and are LATER in the month. Without it, "last expiry of the month"
    picks the legacy contract and the book trades the wrong instrument. Verified
    against Dec-2026: the rule must pick 12-29 (Tue), not 12-31 (Thu).
    """
    rows = con.execute(
        "SELECT expiry_date, MIN(trade_date) FROM nse_options_bhav WHERE symbol='NIFTY' "
        "AND expiry_date>=? AND expiry_date<=? GROUP BY expiry_date ORDER BY expiry_date",
        (start, end)).fetchall()
    listed = {}
    for exp, first in rows:
        ed = prev_session(days, dstr(dparse(exp) - timedelta(days=DTE_IN)))
        if ed and first <= ed:
            listed.setdefault(exp[:7], []).append(exp)
    mw = _monthly_weekday([e for v in listed.values() for e in v])
    out = {}
    for ym, cands in listed.items():
        same = [e for e in cands if mw is None or dparse(e).weekday() == mw]
        out[ym] = max(same) if same else max(cands)
    return dict(sorted(out.items()))


def upcoming_entries(con, days, n=3):
    """The forward schedule: when the book next puts money to work.

    Future NSE holidays are unknown, so the planned entry only rolls off weekends;
    a holiday could pull it one session earlier. Flagged in the payload.
    """
    today = dstr(datetime.now())
    rows = con.execute(
        "SELECT expiry_date, MIN(trade_date) FROM nse_options_bhav WHERE symbol='NIFTY' "
        "AND expiry_date>? GROUP BY expiry_date ORDER BY expiry_date", (today,)).fetchall()
    bym = {}
    for exp, first in rows:
        bym.setdefault(exp[:7], []).append(exp)
    mw = _monthly_weekday([e for v in bym.values() for e in v])
    out = []
    for ym in sorted(bym):
        cands = [e for e in bym[ym] if mw is None or dparse(e).weekday() == mw]
        exp = max(cands) if cands else max(bym[ym])
        ent = dparse(exp) - timedelta(days=DTE_IN)
        rolled = False
        while ent.weekday() >= 5:                 # Sat/Sun -> previous Friday
            ent -= timedelta(days=1)
            rolled = True
        if dstr(ent) <= today:
            continue                              # entry already passed
        out.append(dict(expiry=exp, entry_date=dstr(ent),
                        entry_weekday=ent.strftime("%a"),
                        exit_due=dstr(dparse(exp) - timedelta(days=DTE_OUT)),
                        days_away=(ent - datetime.now()).days + 1,
                        weekend_rolled=rolled))
        if len(out) >= n:
            break
    return out


def bhav_day(con, expiry, day):
    """{strike: (ce_close, pe_close, traded)} for one expiry on one session."""
    rows = con.execute(
        "SELECT strike, option_type, close, contracts FROM nse_options_bhav "
        "WHERE symbol='NIFTY' AND expiry_date=? AND trade_date=?", (expiry, day)).fetchall()
    d = {}
    for k, ot, c, ct in rows:
        d.setdefault(float(k), {})[ot] = (c or 0.0, ct or 0)
    return d


def vix_series(con):
    return [(r[0][:10], float(r[1])) for r in con.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' "
        "AND timeframe='day' ORDER BY date") if r[1]]


def vix_rank_at(vx, idx, day):
    """(level, percentile rank vs the PREVIOUS 252 sessions). Causal."""
    i = idx.get(day)
    if i is None or i < 252:
        return None, None
    lvl = vx[i][1]
    w = [v for _, v in vx[i - 252:i]]
    return lvl, 100.0 * sum(1 for x in w if x < lvl) / len(w)


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


def legs_bhav(con, expiry, strike, day):
    """The two legs priced on a settlement session."""
    legs = (bhav_day(con, expiry, day) or {}).get(strike)
    if not legs or "CE" not in legs or "PE" not in legs:
        return None
    ce, pe = legs["CE"][0], legs["PE"][0]
    if ce <= 0 or pe <= 0:
        return None
    return dict(ce=ce, pe=pe, src="bhav", ts=day)


def legs_chain_live(expiry, strike):
    """The two legs from the 1-minute recorder, if it covers this contract."""
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
    ce = pe = ts = None
    for it, ltp, t in rows:
        if it == "CE" and ce is None:
            ce, ts = float(ltp), t
        if it == "PE" and pe is None:
            pe = float(ltp)
            ts = t if ts is None else max(ts, t)
        if ce and pe:
            break
    return dict(ce=ce, pe=pe, src="chain 1-min", ts=ts) if ce and pe else None


def legs_kite_live(expiry, strike):
    """The two legs straight from the broker. Read-only: quote(), no orders."""
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
        root = "NIFTY%s%s%d" % (e.strftime("%y"), mon[e.month - 1], int(strike))
        keys = ["NFO:%sCE" % root, "NFO:%sPE" % root]
        q = k.quote(keys)
        if len(q) != 2:
            return None
        c, p = q[keys[0]], q[keys[1]]
        if c["last_price"] > 0 and p["last_price"] > 0:
            return dict(ce=c["last_price"], pe=p["last_price"], src="kite ltp",
                        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ce_vol=c.get("volume") or 0, pe_vol=p.get("volume") or 0,
                        ce_oi=c.get("oi") or 0, pe_oi=p.get("oi") or 0,
                        ce_bid=((c.get("depth") or {}).get("buy") or [{}])[0].get("price"),
                        ce_ask=((c.get("depth") or {}).get("sell") or [{}])[0].get("price"),
                        pe_bid=((p.get("depth") or {}).get("buy") or [{}])[0].get("price"),
                        pe_ask=((p.get("depth") or {}).get("sell") or [{}])[0].get("price"))
    except Exception:
        return None
    return None


def leg_rows(pair, r, spot=None):
    """Shape a CE/PE pair into the two rows the UI draws. Both legs are SHORT."""
    if not pair:
        return None
    out = []
    for ot in ("CE", "PE"):
        px = pair.get(ot.lower())
        moneyness = None
        if spot:
            moneyness = (spot - r["strike"]) if ot == "CE" else (r["strike"] - spot)
        out.append(dict(
            side="SHORT", opt=ot, strike=r["strike"], price=px,
            volume=pair.get(ot.lower() + "_vol") or 0,
            oi=pair.get(ot.lower() + "_oi") or 0,
            bid=pair.get(ot.lower() + "_bid"), ask=pair.get(ot.lower() + "_ask"),
            itm_by=round(moneyness, 1) if moneyness is not None else None))
    return out


def legs_now(m, r):
    """Current two-leg prices, same source hierarchy the mark itself uses:
    the 1-minute recorder, then the broker, then the last settlement close."""
    if r["status"] != "OPEN":
        return legs_bhav(m, r["expiry"], r["strike"], r["exit_date"]) if r["exit_date"] else None
    for fn in (lambda: legs_chain_live(r["expiry"], r["strike"]),
               lambda: legs_kite_live(r["expiry"], r["strike"])):
        try:
            p = fn()
        except Exception:
            p = None
        if p:
            return p
    return None


def live_margins(rows):
    """REAL Kite margin for the two-leg short straddle. Read-only, no orders.

    A naked straddle has no hedge to net against, so unlike the winged stock book
    the leg order does not matter here. Both numbers are still kept: `final` is
    what the position blocks once on, `initial` what must be free to put it on.
    """
    out = {}
    if not rows:
        return out
    try:
        from kiteconnect import KiteConnect
        tokf = os.path.join(ROOT, "backtest_data", "access_token.json")
        api = os.environ.get("KITE_API_KEY")
        if not api or not os.path.exists(tokf):
            return out
        k = KiteConnect(api_key=api)
        k.set_access_token(json.load(open(tokf))["access_token"])
        mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    except Exception as e:
        print("  margin: kite unavailable (%s)" % str(e)[:50])
        return out
    for r in rows:
        try:
            e = dparse(r["expiry"])
            root = "NIFTY%s%s%d" % (e.strftime("%y"), mon[e.month - 1], int(r["strike"]))
            legs = [dict(exchange="NFO", tradingsymbol=root + ot, transaction_type="SELL",
                         variety="regular", product="NRML", order_type="MARKET",
                         quantity=int(r["qty"])) for ot in ("CE", "PE")]
            res = k.basket_order_margins(legs, consider_positions=False, mode="compact")
            peak = res["initial"]["total"]
            out[r["id"]] = ((res.get("final") or {}).get("total") or peak, peak)
        except Exception as ex:
            print("  margin %s: %s" % (r["expiry"], str(ex)[:40]))
    return out


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
    vx = vix_series(m)
    vidx = {d: i for i, (d, _) in enumerate(vx)}
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
        lvl, rk = vix_rank_at(vx, vidx, ed)
        on_plan = 1 if (rk is not None and rk > VIX_RANK_MIN) else 0
        con.execute(
            "INSERT INTO trades(expiry,strike,entry_date,entry_spot,credit,qty,lots,status,"
            "vix_level,vix_rank,on_plan) VALUES(?,?,?,?,?,?,?,'OPEN',?,?,?)",
            (exp, K, ed, sp, credit, QTY, LOTS, lvl, rk, on_plan))
        print("  OPENED %s  entry %s  K %.0f  credit %.1f pts (Rs %s)  VIX %.2f rank %.1f  %s"
              % (exp, ed, K, credit, "{:,.0f}".format(credit * QTY), lvl or 0, rk or 0,
                 "ON-PLAN" if on_plan else "OFF-PLAN (filter would skip)"))
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


def idle_cash(con, days, trades):
    """Real accrual on IDLE_ETF over every span where the book holds nothing.

    Uses the ETF's own close-to-close accretion — not an assumed rate — so the
    number is as real as the option marks. Cached in the paper DB.
    """
    spans = []
    closed = sorted([t for t in trades if t["exit_date"]], key=lambda t: t["exit_date"])
    entries = sorted(t["entry_date"] for t in trades)
    for t in closed:
        nxt = next((e for e in entries if e > t["exit_date"]), dstr(datetime.now()))
        if nxt > t["exit_date"]:
            spans.append((t["exit_date"], nxt))
    if not spans:
        return 0.0, [], None
    try:
        from kiteconnect import KiteConnect
        api = os.environ.get("KITE_API_KEY")
        tokf = os.path.join(ROOT, "backtest_data", "access_token.json")
        if not api or not os.path.exists(tokf):
            return 0.0, spans, None
        k = KiteConnect(api_key=api)
        k.set_access_token(json.load(open(tokf))["access_token"])
        tok = next((i["instrument_token"] for i in k.instruments("NSE")
                    if i["tradingsymbol"] == IDLE_ETF), None)
        if not tok:
            return 0.0, spans, None
        lo = min(a for a, _ in spans)
        h = k.historical_data(tok, dparse(lo) - timedelta(days=5), datetime.now(), "day")
        px = {d["date"].strftime("%Y-%m-%d"): d["close"] for d in h}
    except Exception:
        return 0.0, spans, None
    ks = sorted(px)
    def near(d):
        c = [x for x in ks if x <= d]
        return px[c[-1]] if c else None
    total, detail = 0.0, []
    for a, b in spans:
        pa, pb = near(a), near(b)
        if not pa or not pb or pb <= pa:
            continue
        gain = CAPITAL * (pb / pa - 1.0)
        total += gain
        detail.append(dict(frm=a, to=b, days=(dparse(b) - dparse(a)).days,
                           px_from=pa, px_to=pb, gain=round(gain, 2)))
    return total, detail, IDLE_ETF


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
    # the PLAN is the filtered book; everything else is tagged, not hidden
    onp = [x for x in closed if x.get("on_plan")]
    offp = [x for x in closed if not x.get("on_plan")]
    realised_plan = sum(x["net_rs"] or 0 for x in onp)
    realised_off = sum(x["net_rs"] or 0 for x in offp)
    unreal_plan = sum(x["mtm_rs"] or 0 for x in openp if x.get("on_plan"))
    unreal_off = sum(x["mtm_rs"] or 0 for x in openp if not x.get("on_plan"))
    # Per-leg breakdown + the real broker margin. Both are READ-ONLY projections
    # over state that already exists: no rule, no exit and no stored row changes.
    for r in rows:
        try:
            r["legs_entry"] = leg_rows(
                legs_bhav(m, r["expiry"], r["strike"], r["entry_date"]), r,
                r.get("entry_spot"))
            pair = legs_now(m, r)
            r["legs_now"] = leg_rows(pair, r, r.get("mark_spot") or r.get("exit_spot"))
            r["legs_asof"] = pair.get("ts") if pair else None
            r["legs_src"] = pair.get("src") if pair else None
        except Exception:
            r["legs_entry"] = r["legs_now"] = r["legs_asof"] = r["legs_src"] = None
    mg = live_margins(openp)
    for r in openp:
        hp = mg.get(r["id"])
        r["margin_real"] = hp[0] if hp else None
        r["margin_peak"] = hp[1] if hp else None
        r["mtm_pct"] = (100.0 * (r["mtm_rs"] or 0) / r["margin_real"]
                        if r["margin_real"] else None)
    deployed = sum(v[0] for v in mg.values()) if mg else None
    deployed_peak = sum(v[1] for v in mg.values()) if mg else None

    idle_total, idle_detail, idle_sym = idle_cash(con, days, rows)
    upcoming = upcoming_entries(m, days)
    state = dict(
        asof=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        upcoming=upcoming, entry_time="15:30", exit_time="15:30",
        mode="PAPER", lots=LOTS, qty=QTY, capital=CAPITAL,
        bhav_through=days[-1],
        realised=realised, unrealised=unreal,
        nav=CAPITAL + realised + unreal + idle_total,
        filter_name="India VIX percentile rank > %d (vs previous 252 sessions)" % VIX_RANK_MIN,
        vix_rank_min=VIX_RANK_MIN,
        realised_plan=realised_plan, realised_off=realised_off,
        unrealised_plan=unreal_plan, unrealised_off=unreal_off,
        nav_plan=CAPITAL + realised_plan + unreal_plan + idle_total,
        n_off_plan=len([x for x in rows if not x.get("on_plan")]),
        idle_etf=idle_sym, idle_earned=round(idle_total, 2), idle_spans=idle_detail,
        capital_deployed=deployed, capital_deployed_peak=deployed_peak,
        margin_asof=datetime.now().strftime("%Y-%m-%d %H:%M"),
        running_pnl=realised + unreal,
        running_pnl_pct=(100.0 * (realised + unreal) / deployed) if deployed else None,
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
