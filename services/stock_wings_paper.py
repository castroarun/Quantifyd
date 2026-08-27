#!/usr/bin/env python3
"""
Stock 45->21 DTE winged short strangle — PAPER book (research/127).

Paper only. Places no orders and touches no live engine. Reads read-only market
data and writes its own DB plus a static JSON the app page renders (no API
route, no backend restart):

  nse_options_bhav (market_data.db)   real EOD stock-option closes — entries,
                                      daily marks, exits (EOD cadence; stock
                                      options have no intraday recorder)
  market_data_unified                 daily stock closes (ATM anchor)

Rules — the C1 ruleset from research/127, one ruleset for every stock:
  entry     expiry - 45 calendar days (roll back to a session), on the monthly
            stock expiry: SELL CE @ nearest strike to spot+2.5% and
            PE @ spot-2.5%; BUY wing CE/PE ~7% of spot beyond each short strike
  liquidity all 4 legs traded that day; short legs' contracts >= 100 combined;
            each wing >= 10. Otherwise NO entry — this gate IS the stock filter.
  target    structure value <= 50% of net credit
  time      expiry - 21 calendar days
  stop      NONE (study: every premium stop hurts; the wings cap the risk)
  slots     10; candidates ranked by short-leg volume; capital Rs 20L paper
  sizing    notional per slot = slot margin / 10%-of-notional margin estimate
            (conservative mid of the study's stress band; real SPAN check owed)

CLI:
  python3 services/stock_wings_paper.py seed   # replay entries+exits to date
  python3 services/stock_wings_paper.py mark   # sweep exits, mark opens, publish
  python3 services/stock_wings_paper.py show   # print state
"""
import json
import os
import re
import sqlite3
import sys
import time
from bisect import bisect_left
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MKT = os.path.join(ROOT, "backtest_data", "market_data.db")
DB = os.path.join(ROOT, "backtest_data", "stock_wings_paper.db")
PUBS = [os.path.join(ROOT, "frontend", "public", "stock_wings_paper.json"),
        os.path.join(ROOT, "static", "app", "stock_wings_paper.json")]

CAPITAL = 2_000_000.0
MAX_SLOTS = 10
SLOT_MARGIN = CAPITAL / MAX_SLOTS
MARGIN_PCT_EST = 0.10          # conservative mid of the study's x1.5-x2 stress band
K_OFF = 0.025                  # short strikes at spot +/- 2.5%
WING_PCT = 0.07                # wings ~7% of spot beyond the shorts
WING_MIN = 0.02                # reject wings snapping closer than 2%
ATM_BAND = 0.06
TP = 0.50
DTE_IN, DTE_OUT = 45, 21
ATM_VOL_MIN, WING_VOL_MIN = 100, 10
SLIP = 0.005                   # 0.5% of premium per side — stock spreads
SEED_FROM = "2026-06-01"
STUDY = "/app/backtest/stock-45dte-neutral-wings"
INDEX_SYMS = ("NIFTY", "BANKNIFTY")


def ro(p):
    return sqlite3.connect("file:%s?mode=ro" % p, uri=True)


def dstr(d):
    return d.strftime("%Y-%m-%d")


def dparse(s):
    return datetime.strptime(s[:10], "%Y-%m-%d")


def lot_sizes():
    """FNO_LOT_SIZES without importing data_manager (heavy deps): parse the dict."""
    try:
        sys.path.insert(0, os.path.join(ROOT, "services"))
        from data_manager import FNO_LOT_SIZES  # type: ignore
        return dict(FNO_LOT_SIZES)
    except Exception:
        txt = open(os.path.join(ROOT, "services", "data_manager.py"), encoding="utf-8").read()
        m = re.search(r"FNO_LOT_SIZES\s*=\s*\{(.*?)\}", txt, re.S)
        out = {}
        for sym, lot in re.findall(r"['\"]([A-Z0-9&\-]+)['\"]\s*:\s*(\d+)", m.group(1)):
            out[sym] = int(lot)
        return out


def init_db():
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS positions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      symbol TEXT, expiry TEXT, entry_date TEXT, entry_spot REAL,
      kce REAL, kpe REAL, wce REAL, wpe REAL,
      credit REAL, gross_legs REAL, lots INTEGER, lot INTEGER, qty INTEGER,
      atm_vol REAL, wing_vol_min REAL, src TEXT,           -- SEED | LIVE
      exit_date TEXT, exit_val REAL, exit_reason TEXT, exit_spot REAL,
      gross_rs REAL, cost_rs REAL, net_rs REAL,
      status TEXT,                                          -- OPEN | CLOSED
      mark_val REAL, mark_date TEXT, mtm_rs REAL, mark_spot REAL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(symbol, expiry, entry_date));
    CREATE TABLE IF NOT EXISTS marks (
      pos_id INTEGER, d TEXT, val REAL, mtm_rs REAL, UNIQUE(pos_id, d));
    """)
    for ddl in ("ALTER TABLE positions ADD COLUMN margin_real REAL",
                "ALTER TABLE positions ADD COLUMN margin_peak REAL",
                "ALTER TABLE positions ADD COLUMN margin_asof TEXT",
                "ALTER TABLE marks ADD COLUMN src TEXT"):
        try:
            con.execute(ddl)
        except sqlite3.OperationalError:
            pass                          # column already exists
    con.commit()
    return con


# ------------------------------------------------------- market data access --
def sessions(m, start="2026-01-01"):
    return [r[0] for r in m.execute(
        "SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' "
        "AND trade_date>=? ORDER BY trade_date", (start,))]


def prev_session(days, target):
    if not days or target > days[-1]:
        return None
    i = bisect_left(days, target)
    if i < len(days) and days[i] == target:
        return target
    return days[i - 1] if i > 0 else None


def stock_monthly_expiries(m, days, start, end):
    """Per calendar month: the stock-option expiry date (stocks list monthlies
    only, so 'the last expiry of the month with any stock rows, already listed
    45 days out' is the monthly). Learned from data, never hardcoded."""
    rows = m.execute(
        "SELECT expiry_date, MIN(trade_date) FROM nse_options_bhav "
        "WHERE symbol NOT IN (?,?) AND expiry_date>=? AND expiry_date<=? "
        "GROUP BY expiry_date HAVING COUNT(*)>500 ORDER BY expiry_date",
        (*INDEX_SYMS, start, end)).fetchall()
    bym = {}
    for exp, first in rows:
        ed = prev_session(days, dstr(dparse(exp) - timedelta(days=DTE_IN)))
        if ed and first <= ed:
            bym.setdefault(exp[:7], []).append(exp)
    return {ym: max(v) for ym, v in sorted(bym.items())}


def spot_close(m, sym, day):
    r = m.execute("SELECT close FROM market_data_unified WHERE symbol=? "
                  "AND timeframe='day' AND date LIKE ?||'%'", (sym, day)).fetchone()
    return float(r[0]) if r and r[0] else None


def chain_day(m, sym, expiry, day):
    """{(strike, ot): (close, contracts)} for one symbol/expiry/session."""
    return {(float(k), ot): (c or 0.0, ct or 0) for k, ot, c, ct in m.execute(
        "SELECT strike, option_type, close, contracts FROM nse_options_bhav "
        "WHERE symbol=? AND expiry_date=? AND trade_date=?", (sym, expiry, day))}


def structure_value(m, r, day):
    """Cost-to-close per share on a session: (shorts) - (wings). Missing wing
    marks count 0 (pessimistic for us); missing short marks -> no mark."""
    ch = chain_day(m, r["symbol"], r["expiry"], day)
    sce = ch.get((r["kce"], "CE"), (0, 0))[0]
    spe = ch.get((r["kpe"], "PE"), (0, 0))[0]
    if sce <= 0 or spe <= 0:
        return None
    wce = ch.get((r["wce"], "CE"), (0, 0))[0]
    wpe = ch.get((r["wpe"], "PE"), (0, 0))[0]
    return (sce + spe) - (max(wce, 0.0) + max(wpe, 0.0))


def live_margins(rows):
    """REAL per-position margin from Zerodha for the 4-leg structure.

    The page previously sized on a 10%-of-notional ESTIMATE and flagged the Kite
    basket-margin check as owed — this is that check. A winged strangle is defined
    risk, so the long wings cut the requirement well below a naked strangle, and
    the estimate should be expected to differ.

    LEG ORDER MATTERS. Kite walks the basket sequentially, so sending the two
    shorts first prices a momentarily-NAKED strangle and overstates the
    requirement badly (HDFCBANK 1 lot: 81,548 shorts-first vs 54,246 wings-first,
    against a settled 43,219). We send the wings first and read BOTH numbers:

      held = final   — margin the position blocks once all four legs are on
      peak = initial — margin you must have free at the moment of entry

    Read-only: basket_order_margins with consider_positions=False. No orders.
    Throttled — this endpoint rate-limits. Returns {id: (held, peak)} and leaves
    any position it cannot price as absent rather than guessing.
    """
    import time
    out = {}
    try:
        from kiteconnect import KiteConnect
        api = os.environ.get("KITE_API_KEY")
        tokf = os.path.join(ROOT, "backtest_data", "access_token.json")
        if not api or not os.path.exists(tokf):
            return out
        k = KiteConnect(api_key=api)
        k.set_access_token(json.load(open(tokf))["access_token"])
        idx = {}
        for i in k.instruments("NFO"):
            if i["instrument_type"] in ("CE", "PE"):
                idx[(i["name"], str(i["expiry"]), float(i["strike"]), i["instrument_type"])] =                     i["tradingsymbol"]
    except Exception as e:
        print("  margin: kite unavailable (%s)" % str(e)[:50])
        return out

    for r in rows:
        legs, ok = [], True
        for side, ot, K in (("BUY", "CE", r["wce"]), ("BUY", "PE", r["wpe"]),
                            ("SELL", "CE", r["kce"]), ("SELL", "PE", r["kpe"])):
            ts = idx.get((r["symbol"], r["expiry"], float(K), ot))
            if not ts:
                ok = False
                break
            legs.append(dict(exchange="NFO", tradingsymbol=ts, transaction_type=side,
                             variety="regular", product="NRML", order_type="MARKET",
                             quantity=int(r["qty"])))
        if not ok:
            continue
        for attempt in range(3):
            try:
                res = k.basket_order_margins(
                    legs, consider_positions=False, mode="compact")
                peak = res["initial"]["total"]
                held = (res.get("final") or {}).get("total") or peak
                out[r["id"]] = (held, peak)
                break
            except Exception as e:
                if "Too many" in str(e) and attempt < 2:
                    time.sleep(1.2)
                    continue
                print("  margin %s: %s" % (r["symbol"], str(e)[:40]))
                break
        time.sleep(0.45)
    return out


def live_marks(rows):
    """Live cost-to-close per open position, from Kite quotes. READ ONLY.

    The book is struck on bhavcopy closes and there is no intraday recorder for
    stock options, so between EOD runs the page was frozen. This prices the four
    legs live so the MTM moves during the session. It NEVER evaluates the target
    or the stop and never writes to the positions table — exits stay exactly
    where they are, on the EOD close, which is what the study tested.

    A leg that has not traded TODAY is stale (the far wings are thin), so its
    live price is refused and the caller keeps the bhav mark for that leg.
    Returns {id: {"val","legs","stale","ts"}}.
    """
    setup = _live_setup(rows)
    if not setup:
        return {}
    k, spec, want = setup
    try:
        q = k.quote(list(want))
    except Exception as e:
        print("  live: quote failed (%s)" % str(e)[:60])
        return {}
    return _live_tick(q, rows, spec)


def _live_setup(rows):
    """One-time Kite client + leg->tradingsymbol map. The NFO instrument dump is
    multi-MB, so this must NOT run per tick — the daemon calls it once and then
    only quotes."""
    if not rows:
        return None
    try:
        from kiteconnect import KiteConnect
        api = os.environ.get("KITE_API_KEY")
        tokf = os.path.join(ROOT, "backtest_data", "access_token.json")
        if not api or not os.path.exists(tokf):
            return None
        k = KiteConnect(api_key=api)
        k.set_access_token(json.load(open(tokf))["access_token"])
        idx = {}
        for i in k.instruments("NFO"):
            if i["instrument_type"] in ("CE", "PE"):
                idx[(i["name"], str(i["expiry"]), float(i["strike"]),
                     i["instrument_type"])] = i["tradingsymbol"]
    except Exception as e:
        print("  live: kite unavailable (%s)" % str(e)[:60])
        return None
    want, spec = {}, {}
    for r in rows:
        legs = []
        for side, ot, K in (("SHORT", "CE", r["kce"]), ("SHORT", "PE", r["kpe"]),
                            ("LONG", "CE", r["wce"]), ("LONG", "PE", r["wpe"])):
            ts = idx.get((r["symbol"], r["expiry"], float(K), ot))
            legs.append((side, ot, float(K), ts))
            if ts:
                want["NFO:" + ts] = 1
        spec[r["id"]] = legs
        want["NSE:" + r["symbol"]] = 1
    return k, spec, want


def _live_tick(q, rows, spec):
    """Turn one batched quote() response into {id: live-mark}. Pure transform."""
    out = {}
    today = dstr(datetime.now())
    for r in rows:
        vals, legs, stale = {}, [], 0
        for side, ot, K, ts in spec[r["id"]]:
            d = q.get("NFO:" + str(ts), {}) if ts else {}
            lp = d.get("last_price")
            ltt = str(d.get("last_trade_time") or "")[:10]
            fresh = bool(lp) and ltt == today
            if not fresh:
                stale += 1
            vals[(side, ot)] = lp if fresh else None
            legs.append(dict(side=side, opt=ot, strike=K,
                             price=lp if fresh else None,
                             volume=d.get("volume") or 0, stale=not fresh))
        sce, spe = vals.get(("SHORT", "CE")), vals.get(("SHORT", "PE"))
        if sce is None or spe is None:
            continue                      # no live mark without both shorts
        wce = vals.get(("LONG", "CE")) or 0.0
        wpe = vals.get(("LONG", "PE")) or 0.0
        sp = (q.get("NSE:" + r["symbol"], {}) or {}).get("last_price")
        out[r["id"]] = dict(val=(sce + spe) - (wce + wpe), legs=legs,
                            stale=stale, spot=sp,
                            ts=datetime.now().strftime("%H:%M:%S"))
    return out


def leg_detail(m, r, day):
    """Per-leg prices on a session — a READ-ONLY projection for the UI.

    The structure's four legs are already priced inside structure_value(); this
    just returns them individually instead of netted, so a row can be expanded
    to show what is actually held. Adds no state and changes no trading rule.
    """
    ch = chain_day(m, r["symbol"], r["expiry"], day)
    spec = [("SHORT", "CE", r["kce"]), ("SHORT", "PE", r["kpe"]),
            ("LONG", "CE", r["wce"]), ("LONG", "PE", r["wpe"])]
    out = []
    for side, ot, K in spec:
        px, vol = ch.get((float(K), ot), (None, 0))
        out.append(dict(side=side, opt=ot, strike=K,
                        price=(px if px and px > 0 else None), volume=vol))
    return out


def statutory_charges(m, r):
    """EXACT-model broker+statutory charges for one closed round trip, per the
    Zerodha F&O options rate card: brokerage Rs20/executed order x8, STT 0.1%
    of SELL premium (entry shorts + wings sold back at exit), NSE txn 0.03503%
    of premium both sides, SEBI Rs10/crore, stamp 0.003% buy side, GST 18% on
    brokerage+txn+SEBI. Separate from modeled slippage, which is an execution-
    quality ASSUMPTION the paper soak exists to measure."""
    q = r["qty"]
    le = {(l["side"], l["opt"]): (l["price"] or 0.0) for l in (leg_detail(m, r, r["entry_date"]) or [])}
    lx = {(l["side"], l["opt"]): (l["price"] or 0.0) for l in (leg_detail(m, r, r["exit_date"]) or [])}
    sell_prem = (le.get(("SHORT", "CE"), 0) + le.get(("SHORT", "PE"), 0)
                 + lx.get(("LONG", "CE"), 0) + lx.get(("LONG", "PE"), 0)) * q
    buy_prem = (le.get(("LONG", "CE"), 0) + le.get(("LONG", "PE"), 0)
                + lx.get(("SHORT", "CE"), 0) + lx.get(("SHORT", "PE"), 0)) * q
    total = sell_prem + buy_prem
    brokerage = 20.0 * 8
    stt = 0.001 * sell_prem
    txn = 0.0003503 * total
    sebi = total / 1e7 * 10
    stamp = 0.00003 * buy_prem
    return brokerage + stt + txn + sebi + stamp + 0.18 * (brokerage + txn + sebi)


def intrinsic_value(r, spot):
    sv = max(0.0, spot - r["kce"]) + max(0.0, r["kpe"] - spot)
    wv = max(0.0, spot - r["wce"]) + max(0.0, r["wpe"] - spot)
    return sv - wv


def costs_rs(r, exit_val):
    """Slippage + STT + txn + brokerage, in rupees for the whole position."""
    entry_legs = r["gross_legs"]                 # sum of all 4 leg prices at entry
    exit_legs = abs(exit_val) + 0.5              # proxy: structure value ~ legs net; add floor
    short_prem = (entry_legs + r["credit"]) / 2.0   # shorts = (gross + credit) / 2
    per_share = SLIP * (entry_legs + exit_legs) + 0.0010 * short_prem \
        + 0.0005 * (entry_legs + exit_legs)
    brok = 20.0 * 8
    return per_share * r["qty"] + brok * 1.18


def candidates(m, days, exp, ed, lots_map):
    """All symbols passing the C1 gate on this entry session, best first."""
    out = []
    for sym, lot in lots_map.items():
        sp = spot_close(m, sym, ed)
        if not sp or lot * sp > SLOT_MARGIN / MARGIN_PCT_EST:
            continue                              # no spot, or 1 lot outgrows a slot
        ch = chain_day(m, sym, exp, ed)
        if not ch:
            continue
        strikes = sorted({k for (k, ot) in ch})

        def pick(side, target, lo=None, hi=None):
            best, bd = None, 1e18
            for k in strikes:
                if lo is not None and k <= lo:
                    continue
                if hi is not None and k >= hi:
                    continue
                c, ct = ch.get((k, side), (0, 0))
                if c <= 0 or ct <= 0:
                    continue
                if abs(k - target) < bd:
                    best, bd = k, abs(k - target)
            return best

        kce = pick("CE", sp * (1 + K_OFF))
        kpe = pick("PE", sp * (1 - K_OFF))
        if kce is None or kpe is None or kce < kpe:
            continue
        if abs(kce / sp - (1 + K_OFF)) > ATM_BAND or abs(kpe / sp - (1 - K_OFF)) > ATM_BAND:
            continue
        wce = pick("CE", kce + WING_PCT * sp, lo=kce)
        wpe = pick("PE", kpe - WING_PCT * sp, hi=kpe)
        if wce is None or wpe is None:
            continue
        if (wce - kce) / sp < WING_MIN or (kpe - wpe) / sp < WING_MIN:
            continue
        sce, sct = ch[(kce, "CE")]
        spe, pct_ = ch[(kpe, "PE")]
        wcec, wcct = ch[(wce, "CE")]
        wpec, wpct = ch[(wpe, "PE")]
        atm_vol = sct + pct_
        wing_min = min(wcct, wpct)
        if atm_vol < ATM_VOL_MIN or wing_min < WING_VOL_MIN:
            continue
        credit = (sce + spe) - (wcec + wpec)
        if credit <= 0:
            continue
        lots = max(1, int((SLOT_MARGIN / MARGIN_PCT_EST) // (lot * sp)))
        out.append(dict(symbol=sym, expiry=exp, entry_date=ed, entry_spot=sp,
                        kce=kce, kpe=kpe, wce=wce, wpe=wpe, credit=credit,
                        gross_legs=sce + spe + wcec + wpec, lots=lots, lot=lot,
                        qty=lots * lot, atm_vol=atm_vol, wing_vol_min=wing_min))
    return sorted(out, key=lambda c: -c["atm_vol"])


# ------------------------------------------------------------------- engine --
def rows_of(con, where="1=1"):
    cur = con.execute("SELECT * FROM positions WHERE " + where)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, t)) for t in cur.fetchall()]


def sweep(con, m, days, upto):
    """Walk open positions day by day up to `upto`; apply TP / 21-DTE time exit
    (no stop — the wings are the risk cap); record daily marks.

    The day's value of RECORD is the real 15:29:30 quote snapshot captured by
    the daemon (marks.src='kite') when one exists; bhavcopy closes only
    back-fill days the daemon missed and the seeded history. Bhav never
    overwrites a kite mark, and both exit rules evaluate on whichever value is
    the day's record."""
    bhav_last = days[-1]
    for r in rows_of(con, "status='OPEN'"):
        xcal = dstr(dparse(r["expiry"]) - timedelta(days=DTE_OUT))
        kmarks = {d: v for d, v in con.execute(
            "SELECT d, val FROM marks WHERE pos_id=? AND src='kite'", (r["id"],))}
        walk = sorted(set(days) | set(kmarks))
        closed = False
        for d in [x for x in walk if r["entry_date"] < x <= upto]:
            if d in kmarks:
                val, src = kmarks[d], "kite"
            elif d <= bhav_last:
                val, src = structure_value(m, r, d), "bhav"
            else:
                continue
            if val is None:
                continue                          # no short-leg marks; roll forward
            con.execute("INSERT INTO marks(pos_id,d,val,mtm_rs,src) VALUES(?,?,?,?,?) "
                        "ON CONFLICT(pos_id,d) DO NOTHING",
                        (r["id"], d, val, (r["credit"] - val) * r["qty"], src))
            why = None
            if val <= TP * r["credit"]:
                why = "TARGET"
            elif d >= xcal:
                why = "TIME_21DTE"
            elif d >= r["expiry"]:
                sp = spot_close(m, r["symbol"], d) or r["entry_spot"]
                val = intrinsic_value(r, sp)
                why = "EXPIRY"
            if why:
                gross = (r["credit"] - val) * r["qty"]
                cost = costs_rs(r, val)
                xsp = (spot_close(m, r["symbol"], d) if d <= bhav_last
                       else r.get("mark_spot"))
                con.execute(
                    "UPDATE positions SET status='CLOSED',exit_date=?,exit_val=?,exit_reason=?,"
                    "exit_spot=?,gross_rs=?,cost_rs=?,net_rs=?,mark_val=?,mark_date=?,mtm_rs=? WHERE id=?",
                    (d, val, why, xsp, gross, cost, gross - cost,
                     val, d, gross - cost, r["id"]))
                print("  CLOSED %-12s %s @ %.2f (%s, %s) net Rs %s"
                      % (r["symbol"], d, val, why, src, "{:,.0f}".format(gross - cost)))
                closed = True
                break
        if not closed:
            # still open — mark to the latest day of record (kite preferred)
            for d in reversed([x for x in walk if r["entry_date"] < x <= upto]):
                val = kmarks.get(d)
                if val is None and d <= bhav_last:
                    val = structure_value(m, r, d)
                if val is not None:
                    con.execute("UPDATE positions SET mark_val=?,mark_date=?,mtm_rs=?,"
                                "mark_spot=COALESCE(?,mark_spot) WHERE id=?",
                                (val, d, (r["credit"] - val) * r["qty"],
                                 spot_close(m, r["symbol"], d) if d <= bhav_last else None,
                                 r["id"]))
                    break
    con.commit()


def seed():
    con = init_db()
    m = ro(MKT)
    days = sessions(m)
    lots_map = {s: l for s, l in lot_sizes().items() if s not in INDEX_SYMS}
    today = dstr(datetime.now())
    exps = stock_monthly_expiries(m, days, "2026-01-01", "2027-06-30")
    print("bhav sessions to %s | stock monthlies %s" % (days[-1], list(exps.values())))
    for ym, exp in exps.items():
        ed = prev_session(days, dstr(dparse(exp) - timedelta(days=DTE_IN)))
        if not ed or ed < SEED_FROM:
            continue
        if (dparse(exp) - dparse(ed)).days > DTE_IN + 5:
            continue
        sweep(con, m, days, ed)                   # free slots that exited before this cycle
        if con.execute("SELECT 1 FROM positions WHERE expiry=? AND src='LIVE'",
                       (exp,)).fetchone():
            continue                              # cycle already entered LIVE by the daemon
        n_open = len(rows_of(con, "status='OPEN'"))
        free = MAX_SLOTS - n_open
        if free <= 0:
            print("  %s: book full, no entries" % ed)
            continue
        cands = candidates(m, days, exp, ed, lots_map)
        taken = 0
        for c in cands:
            if taken >= free:
                break
            if con.execute("SELECT 1 FROM positions WHERE symbol=? AND expiry=? AND entry_date=?",
                           (c["symbol"], exp, ed)).fetchone():
                continue
            src = "SEED" if ed < today else "LIVE"
            con.execute(
                "INSERT INTO positions(symbol,expiry,entry_date,entry_spot,kce,kpe,wce,wpe,"
                "credit,gross_legs,lots,lot,qty,atm_vol,wing_vol_min,src,status) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN')",
                (c["symbol"], exp, ed, c["entry_spot"], c["kce"], c["kpe"], c["wce"], c["wpe"],
                 c["credit"], c["gross_legs"], c["lots"], c["lot"], c["qty"],
                 c["atm_vol"], c["wing_vol_min"], src))
            taken += 1
            print("  OPENED %-12s %s exp %s  PE/CE %.0f/%.0f wings %.0f/%.0f  credit %.2f x %d qty "
                  "(Rs %s)  vol %d" % (c["symbol"], ed, exp, c["kpe"], c["kce"], c["wpe"], c["wce"],
                                       c["credit"], c["qty"],
                                       "{:,.0f}".format(c["credit"] * c["qty"]), int(c["atm_vol"])))
        print("  %s: %d/%d candidates taken (%d slots free)" % (ed, taken, len(cands), free))
    con.commit()
    sweep(con, m, days, max(days[-1], dstr(datetime.now())))
    publish(con, m, days)
    con.close()
    m.close()


def mark():
    con = init_db()
    m = ro(MKT)
    days = sessions(m)
    sweep(con, m, days, max(days[-1], dstr(datetime.now())))
    publish(con, m, days)
    con.close()
    m.close()


def upcoming(m, days, n=3):
    today = dstr(datetime.now())
    rows = m.execute(
        "SELECT expiry_date, MIN(trade_date) FROM nse_options_bhav "
        "WHERE symbol NOT IN (?,?) AND expiry_date>? GROUP BY expiry_date "
        "HAVING COUNT(*)>500 ORDER BY expiry_date", (*INDEX_SYMS, today)).fetchall()
    bym = {}
    for exp, _ in rows:
        bym.setdefault(exp[:7], []).append(exp)
    out = []
    for ym in sorted(bym):
        exp = max(bym[ym])
        ent = dparse(exp) - timedelta(days=DTE_IN)
        while ent.weekday() >= 5:
            ent -= timedelta(days=1)
        if dstr(ent) <= today:
            continue
        out.append(dict(expiry=exp, entry_date=dstr(ent), entry_weekday=ent.strftime("%a"),
                        exit_due=dstr(dparse(exp) - timedelta(days=DTE_OUT)),
                        days_away=(ent - datetime.now()).days + 1))
        if len(out) >= n:
            break
    return out


def publish(con, m, days, live=None, verbose=True, upc=None, mg=None):
    allr = rows_of(con)
    for r in allr:
        r["exit_due"] = dstr(dparse(r["expiry"]) - timedelta(days=DTE_OUT))
        r["dte"] = (dparse(r["expiry"]) - datetime.now()).days
    openp = [r for r in allr if r["status"] == "OPEN"]
    closed = sorted([r for r in allr if r["status"] == "CLOSED"], key=lambda r: r["exit_date"])
    for r in openp:
        r["curve"] = [dict(d=d, val=v, mtm=mt) for d, v, mt in con.execute(
            "SELECT d,val,mtm_rs FROM marks WHERE pos_id=? ORDER BY d", (r["id"],))]
    # per-leg breakdown so the UI can expand a row. Entry legs are priced on the
    # entry session, current legs on whatever session the row is marked to.
    for r in allr:
        try:
            r["legs_entry"] = leg_detail(m, r, r["entry_date"])
            md = r["mark_date"] if r["status"] == "OPEN" else r["exit_date"]
            r["legs_now"] = leg_detail(m, r, md) if md else None
            r["legs_asof"] = md
        except Exception:
            r["legs_entry"] = r["legs_now"] = None
    # LIVE overlay — display only. Exits are untouched and still fire on the
    # EOD close in sweep(); this just stops the page being frozen mid-session.
    # The day's final live marks are persisted to a sidecar and REUSED after
    # close for as long as they are fresher than the newest bhav session —
    # without this, the post-close EOD publish regresses the page to T-1
    # closes right after the user watched today's live numbers.
    sidecar = os.path.join(ROOT, "backtest_data", "stock_wings_last_live.json")
    if live:
        try:
            with open(sidecar + ".tmp", "w") as f:
                json.dump({"date": dstr(datetime.now()), "marks": live}, f, default=str)
            os.replace(sidecar + ".tmp", sidecar)
        except Exception:
            pass
    else:
        try:
            sd = json.load(open(sidecar))
            if sd.get("date", "") > days[-1]:
                live = {int(k): v for k, v in sd["marks"].items()}
        except Exception:
            pass
    live_ts = None
    for r in openp:
        lv = (live or {}).get(r["id"])
        r["live"] = False
        if lv and lv["val"] is not None:
            r["mark_val"] = round(lv["val"], 2)
            r["mtm_rs"] = (r["credit"] - lv["val"]) * r["qty"]
            r["legs_now"] = lv["legs"]
            r["legs_asof"] = "live " + lv["ts"]
            r["mark_spot"] = lv.get("spot") or r["mark_spot"]
            r["live"] = True
            r["stale_legs"] = lv["stale"]
            live_ts = lv["ts"]

    # margins move slowly; the 5s daemon passes a cached dict so the throttled
    # basket-margin endpoint is hit every ~10 min, not every tick. A successful
    # fetch is PERSISTED per position, so after-hours publishes (when the Kite
    # margin API is unavailable/throttled) fall back to the last good value
    # instead of blanking the column.
    if mg is None:
        mg = live_margins(openp)
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    for r in openp:
        hp = mg.get(r["id"])
        if hp:
            r["margin_real"], r["margin_peak"], r["margin_asof"] = hp[0], hp[1], now_ts
            con.execute("UPDATE positions SET margin_real=?,margin_peak=?,margin_asof=? "
                        "WHERE id=?", (hp[0], hp[1], now_ts, r["id"]))
        # else: keep the stored margin_real/margin_peak/margin_asof from the DB
        r["margin_est"] = r["entry_spot"] * r["qty"] * MARGIN_PCT_EST
        r["mtm_pct"] = (100.0 * (r["mtm_rs"] or 0) / r["margin_real"]
                        if r.get("margin_real") else None)
    if mg:
        con.commit()
    with_m = [r for r in openp if r.get("margin_real")]
    deployed = sum(r["margin_real"] for r in with_m) if with_m else None
    deployed_peak = (sum(r["margin_peak"] or r["margin_real"] for r in with_m)
                     if with_m else None)
    margin_asof = max((r.get("margin_asof") or "" for r in openp), default=None) or None
    deployed_est = sum(r["margin_est"] for r in openp)
    realised = sum(r["net_rs"] or 0 for r in closed)
    unreal = sum(r["mtm_rs"] or 0 for r in openp)
    costs_paid = sum(r["cost_rs"] or 0 for r in closed)          # in realised already
    gross_closed = sum(r["gross_rs"] or 0 for r in closed)
    est_open_exit_costs = sum(costs_rs(r, r["mark_val"] or 0.0) for r in openp)
    # split the model cost: exact statutory (Zerodha rate card, real leg values)
    # vs modeled slippage (the 0.5%/side fill-quality assumption)
    charges_stat = 0.0
    for r in closed:
        try:
            charges_stat += statutory_charges(m, r)
        except Exception:
            charges_stat += 20.0 * 8 * 1.18
    charges_slip = max(0.0, costs_paid - charges_stat)
    wins = [r for r in closed if (r["net_rs"] or 0) > 0]
    payload = dict(
        asof=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        mode="PAPER", capital=CAPITAL, max_slots=MAX_SLOTS,
        slot_margin=SLOT_MARGIN, margin_pct_est=MARGIN_PCT_EST,
        rules=dict(dte_in=DTE_IN, dte_out=DTE_OUT, k_off=K_OFF, wing_pct=WING_PCT,
                   tp=TP, stop=None, atm_vol_min=ATM_VOL_MIN, wing_vol_min=WING_VOL_MIN,
                   slip=SLIP),
        links=dict(study=STUDY, tearsheet="/app/stock45_wings_tearsheet.png",
                   github="https://github.com/castroarun/Quantifyd/tree/main/research/127_stock_neutral_wings"),
        bhav_through=days[-1],
        realised=realised, unrealised=unreal, nav=CAPITAL + realised + unreal,
        costs_paid=costs_paid, gross_closed=gross_closed,
        charges_stat=charges_stat, charges_slip=charges_slip,
        est_open_exit_costs=est_open_exit_costs,
        n_open=len(openp), n_closed=len(closed),
        capital_deployed=deployed, capital_deployed_est=deployed_est,
        capital_deployed_peak=deployed_peak,
        running_pnl_pct=(100.0 * (realised + unreal) / deployed) if deployed else None,
        live_ts=live_ts, live_n=sum(1 for r in openp if r.get("live")),
        margin_asof=margin_asof,
        running_pnl=realised + unreal,
        win_rate=(100.0 * len(wins) / len(closed)) if closed else None,
        upcoming=upc if upc is not None else upcoming(m, days),
        entry_time="15:30 (EOD close)", exit_time="15:30 (EOD close)",
        open_positions=openp, closed_trades=closed,
        note=("REAL-DATA book: intraday marks tick on live Kite quotes; the day's mark of "
              "record is the real 15:29:30 close snapshot, and the daily exits (50% "
              "target / 21-DTE time) evaluate on it — once per day at the close, exactly "
              "the cadence the study validated. New cycles enter LIVE at ~15:26 on real "
              "quotes with today's real traded volume as the liquidity gate. Bhavcopy is "
              "backfill only (seeded history and any day the daemon was down). Margin is "
              "the real Kite basket requirement, wings sent first. SEED = replayed "
              "history; LIVE = opened by the live engine."))
    for p in PUBS:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p + ".tmp", "w") as f:
                json.dump(payload, f, default=str)
            os.replace(p + ".tmp", p)
        except Exception as ex:
            print("  publish %s failed: %s" % (p, ex))
    if verbose:
        print("  published NAV Rs %s (realised %s, MTM %s, open %d, closed %d)"
              % ("{:,.0f}".format(payload["nav"]), "{:,.0f}".format(realised),
                 "{:,.0f}".format(unreal), len(openp), len(closed)))


def live():
    """Re-price open positions from live quotes and republish. Display only.

    Deliberately does NOT call sweep(): no exit is evaluated, no position row is
    written. Safe to run on a tight intraday cron.
    """
    con = init_db()
    m = ro(MKT)
    days = sessions(m)
    openp = [r for r in rows_of(con, "status='OPEN'")]
    lv = live_marks(openp)
    print("  live marks for %d/%d open positions" % (len(lv), len(openp)))
    publish(con, m, days, live=lv)
    con.close()
    m.close()


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def todays_cycle(m):
    """The expiry whose 45-DTE entry day (weekend-rolled) is TODAY, else None."""
    today = dstr(datetime.now())
    rows = m.execute(
        "SELECT expiry_date FROM nse_options_bhav WHERE symbol NOT IN (?,?) "
        "AND expiry_date>? GROUP BY expiry_date HAVING COUNT(*)>500 "
        "ORDER BY expiry_date", (*INDEX_SYMS, today)).fetchall()
    bym = {}
    for (exp,) in rows:
        bym.setdefault(exp[:7], []).append(exp)
    for ym in sorted(bym):
        exp = max(bym[ym])
        ent = dparse(exp) - timedelta(days=DTE_IN)
        while ent.weekday() >= 5:
            ent -= timedelta(days=1)
        if dstr(ent) == today:
            return exp
    return None


def entry_scan_live(con, m, k, exp, lots_map):
    """REAL-quote entry engine for a cycle day, run by the daemon at ~15:26.

    Live spot, live strike grid, TODAY's real traded volume for the liquidity
    gate, fills at last-traded price. This replaces the bhav-based entry for
    days the daemon is up; seed() backfills from bhav only if this never ran
    for the cycle (daemon down / holiday shift)."""
    today = dstr(datetime.now())
    if con.execute("SELECT 1 FROM positions WHERE expiry=? AND src='LIVE'", (exp,)).fetchone():
        return 0
    free = MAX_SLOTS - len(rows_of(con, "status='OPEN'"))
    if free <= 0:
        print("  entry: book full", flush=True)
        return 0
    try:
        instr = k.instruments("NFO")
    except Exception as e:
        print("  entry: instruments failed (%s)" % str(e)[:60], flush=True)
        return 0
    grid = {}
    for i in instr:
        if str(i["expiry"]) == exp and i["instrument_type"] in ("CE", "PE") \
                and i["name"] in lots_map:
            grid.setdefault(i["name"], {}).setdefault(
                i["instrument_type"], {})[float(i["strike"])] = i["tradingsymbol"]
    spots = {}
    for batch in _chunks(["NSE:" + s for s in grid], 400):
        try:
            spots.update({kk.split(":")[1]: v["last_price"]
                          for kk, v in k.ltp(batch).items()})
        except Exception:
            pass
    plan, want = {}, []
    for sym, g in grid.items():
        sp, lot = spots.get(sym), lots_map[sym]
        if not sp or lot * sp > SLOT_MARGIN / MARGIN_PCT_EST:
            continue
        ces, pes = sorted(g.get("CE", {})), sorted(g.get("PE", {}))
        if not ces or not pes:
            continue
        kce = min(ces, key=lambda x: abs(x - sp * (1 + K_OFF)))
        kpe = min(pes, key=lambda x: abs(x - sp * (1 - K_OFF)))
        if kce < kpe or abs(kce / sp - (1 + K_OFF)) > ATM_BAND \
                or abs(kpe / sp - (1 - K_OFF)) > ATM_BAND:
            continue
        wce_c = [x for x in ces if x > kce]
        wpe_c = [x for x in pes if x < kpe]
        if not wce_c or not wpe_c:
            continue
        wce = min(wce_c, key=lambda x: abs(x - (kce + WING_PCT * sp)))
        wpe = min(wpe_c, key=lambda x: abs(x - (kpe - WING_PCT * sp)))
        if (wce - kce) / sp < WING_MIN or (kpe - wpe) / sp < WING_MIN:
            continue
        legs = dict(sce=g["CE"][kce], spe=g["PE"][kpe], wce=g["CE"][wce], wpe=g["PE"][wpe])
        plan[sym] = dict(sp=sp, lot=lot, kce=kce, kpe=kpe, wce=wce, wpe=wpe, legs=legs)
        want += ["NFO:" + t for t in legs.values()]
    q = {}
    for batch in _chunks(sorted(set(want)), 400):
        try:
            q.update(k.quote(batch))
        except Exception as e:
            print("  entry: quote failed (%s)" % str(e)[:60], flush=True)
    cands = []
    for sym, p in plan.items():
        d_ = {t: q.get("NFO:" + p["legs"][t], {}) for t in p["legs"]}
        prc = {t: (d_[t].get("last_price") or 0) for t in p["legs"]}
        vol = {t: (d_[t].get("volume") or 0) for t in p["legs"]}
        ltt = {t: str(d_[t].get("last_trade_time") or "")[:10] for t in p["legs"]}
        if any(prc[t] <= 0 or ltt[t] != today for t in p["legs"]):
            continue                      # all 4 legs must have traded TODAY
        atm_vol = vol["sce"] + vol["spe"]
        wing_min = min(vol["wce"], vol["wpe"])
        if atm_vol < ATM_VOL_MIN or wing_min < WING_VOL_MIN:
            continue
        credit = (prc["sce"] + prc["spe"]) - (prc["wce"] + prc["wpe"])
        if credit <= 0:
            continue
        lots = max(1, int((SLOT_MARGIN / MARGIN_PCT_EST) // (p["lot"] * p["sp"])))
        cands.append(dict(symbol=sym, sp=p["sp"], kce=p["kce"], kpe=p["kpe"],
                          wce=p["wce"], wpe=p["wpe"], credit=credit,
                          gross=sum(prc.values()), lots=lots, lot=p["lot"],
                          qty=lots * p["lot"], atm_vol=atm_vol, wing_min=wing_min))
    cands.sort(key=lambda c: -c["atm_vol"])
    taken = 0
    for c in cands[:free]:
        con.execute(
            "INSERT INTO positions(symbol,expiry,entry_date,entry_spot,kce,kpe,wce,wpe,"
            "credit,gross_legs,lots,lot,qty,atm_vol,wing_vol_min,src,status) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'LIVE','OPEN')",
            (c["symbol"], exp, today, c["sp"], c["kce"], c["kpe"], c["wce"], c["wpe"],
             c["credit"], c["gross"], c["lots"], c["lot"], c["qty"],
             c["atm_vol"], c["wing_min"]))
        taken += 1
        print("  LIVE ENTRY %-12s exp %s PE/CE %.0f/%.0f wings %.0f/%.0f credit %.2f x %d"
              % (c["symbol"], exp, c["kpe"], c["kce"], c["wpe"], c["wce"],
                 c["credit"], c["qty"]), flush=True)
    con.commit()
    print("  entry: %d/%d candidates taken (%d slots were free)"
          % (taken, len(cands), free), flush=True)
    return taken


def eod_capture(con, m, days, openp, lv):
    """Record TODAY's real 15:29:30 snapshot as the day's mark of record
    (marks.src='kite'), then evaluate the daily exits on it."""
    today = dstr(datetime.now())
    n = 0
    for r in openp:
        v = (lv or {}).get(r["id"])
        if v and v.get("val") is not None:
            con.execute("INSERT OR REPLACE INTO marks(pos_id,d,val,mtm_rs,src) "
                        "VALUES(?,?,?,?,'kite')",
                        (r["id"], today, v["val"], (r["credit"] - v["val"]) * r["qty"]))
            if v.get("spot"):
                con.execute("UPDATE positions SET mark_spot=? WHERE id=?",
                            (v["spot"], r["id"]))
            n += 1
    con.commit()
    sweep(con, m, days, max(days[-1], today))
    print("  eod: %d real closes recorded, exits evaluated" % n, flush=True)


def livedaemon(tick=5):
    """Persistent intraday ticker: ONE batched quote() every `tick` seconds ->
    republish the JSON. Display only, same guarantees as live(): never evaluates
    an exit, never writes a position row. Exits outside 09:14-15:40 IST; the
    */5-min cron relaunches it under flock if it dies mid-session.

    Kite budget: 1 quote call (~50 instruments) per tick = 0.2 req/s, far under
    the 3 req/s limit. The instrument map is built ONCE (multi-MB dump) and the
    open-position set refreshed every 10 min (it only changes EOD anyway).
    """
    def in_hours(now):
        hm = now.hour * 60 + now.minute
        return now.weekday() < 5 and (9 * 60 + 14) <= hm <= (15 * 60 + 40)

    if not in_hours(datetime.now()):
        print("  daemon: outside market hours, not starting")
        return
    con = init_db()
    m = ro(MKT)
    days = sessions(m)
    openp = rows_of(con, "status='OPEN'")
    setup = _live_setup(openp)
    if not setup:
        print("  daemon: kite unavailable, exiting (cron will retry)")
        return
    k, spec, want = setup
    upc = upcoming(m, days)               # heavy GROUP BY — cache across ticks
    mgc = live_margins(openp)             # throttled endpoint — cache across ticks
    lots_map = {s: l for s, l in lot_sizes().items() if s not in INDEX_SYMS}
    entry_exp = todays_cycle(m)           # is TODAY a 45-DTE entry day?
    if entry_exp:
        print("  daemon: TODAY is the entry day for expiry %s — live scan at 15:26"
              % entry_exp, flush=True)
    print("  daemon: up, %d positions, %d margins, tick %ds"
          % (len(openp), len(mgc), tick), flush=True)
    fails, last_reload, last_log = 0, time.time(), 0.0
    did_entry = did_eod = False
    lv = None
    while in_hours(datetime.now()):
        try:
            q = k.quote(list(want))
            lv = _live_tick(q, openp, spec)
            publish(con, m, days, live=lv, verbose=False, upc=upc, mg=mgc)
            hms = datetime.now().strftime("%H:%M:%S")
            if entry_exp and not did_entry and hms >= "15:26:00":
                did_entry = True
                entry_scan_live(con, m, k, entry_exp, lots_map)
                openp = rows_of(con, "status='OPEN'")
                s2 = _live_setup(openp)
                if s2:
                    k, spec, want = s2
                m2 = live_margins(openp)
                if m2:
                    mgc = m2
            if not did_eod and hms >= "15:29:30":
                did_eod = True
                eod_capture(con, m, days, openp, lv)
                openp = rows_of(con, "status='OPEN'")   # exits may have closed rows
                s2 = _live_setup(openp)
                if s2:
                    k, spec, want = s2
                publish(con, m, days, verbose=False, upc=upc, mg=mgc)
            fails = 0
            if time.time() - last_log > 600:
                print("  daemon: %s live marks %d/%d" %
                      (datetime.now().strftime("%H:%M:%S"), len(lv), len(openp)),
                      flush=True)
                last_log = time.time()
        except Exception as e:
            fails += 1
            if fails >= 24:               # ~2 min of continuous failure
                print("  daemon: %d straight failures (%s), exiting for cron retry"
                      % (fails, str(e)[:60]))
                return
        time.sleep(tick)
        if time.time() - last_reload > 600:
            openp = rows_of(con, "status='OPEN'")
            s2 = _live_setup(openp)
            if s2:
                k, spec, want = s2
            m2 = live_margins(openp)
            if m2:
                mgc = m2                  # keep last good margins on a failed refresh
            last_reload = time.time()
    print("  daemon: market closed, exiting")
    con.close()
    m.close()


def show():
    con = init_db()
    for r in rows_of(con):
        print(r["status"], r["src"], r["symbol"], r["entry_date"], "->", r["exit_date"],
              r["exit_reason"], "net", r["net_rs"], "mtm", r["mtm_rs"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "mark"
    {"seed": seed, "mark": mark, "live": live, "livedaemon": livedaemon,
     "show": show}[cmd]()
