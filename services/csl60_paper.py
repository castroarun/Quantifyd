#!/usr/bin/env python3
"""
CSL-60 DTE-0 — NIFTY expiry-day 09:16 ATM short straddle, 60% per-leg stop.
PAPER book. Places no orders and touches no live engine.

The winning configuration of the AlgoTest study (research/136 §0d-viii, rank 1
of 98): DTE-0 only, entry 09:16, sell ATM CE+PE, per-leg SL at 1.60x entry,
trail-to-breakeven on the surviving leg after the first stop, square off 15:15.
Study baseline (ex-events, 10 lots): net Rs.23,69,304 / 294 tr / WR 63.6% /
MaxDD -1,51,578 / t 4.28. Deploy doc:
research/136_nifty_csl_portfolio/CSL60_DTE0_PAPER_1MIN_DEPLOY_STATUS.md

Data source: backtest_data/options_data.db option_chain (recorded 1-minute
quotes; timestamps ISO with 'T'). The study ran on AlgoTest 1-minute bars, so
1-minute replay here is resolution-consistent with what it validates.

Design: every run REPLAYS the whole day from the 09:16 entry deterministically
(minute by minute), so it is idempotent, order-correct on which leg stopped
first, and self-heals across missed cron minutes.

CLI:
  python3 services/csl60_paper.py mark              # cron verb (default)
  python3 services/csl60_paper.py replay 2026-09-01 # dry-run a past expiry day
  python3 services/csl60_paper.py seed              # backfill all recorded expiry days
  python3 services/csl60_paper.py show              # print state
"""
import fcntl
import json
import os
import sqlite3
import sys
from datetime import datetime, time as dtime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPT = os.path.join(ROOT, "backtest_data", "options_data.db")
DB = os.path.join(ROOT, "backtest_data", "csl60_paper.db")
PUBS = [os.path.join(ROOT, "static", "app", "csl60_paper.json"),
        os.path.join(ROOT, "frontend", "public", "csl60_paper.json")]
LOCK = os.path.join(ROOT, "backtest_data", ".csl60_paper.lock")

SYMBOL = "NIFTY"
LOT = 65
LOTS = 10                      # matches the study - 1 premium pt = Rs.650
QTY = LOT * LOTS
SL_MULT = 1.60                 # per-leg stop: 60% above entry premium
ENTRY_HHMM = "09:16"
LATE_HHMM = "09:30"            # one-shot: no entry after this (916 semantics)
EXIT_HHMM = "15:15"
COST_RATE = 0.0059             # study-parity: 0.59% of premium turnover
COST_FLAT = 80.0               # + Rs.80 per trade day
EVENT_SKIP = {"2027-02-01", "2027-02-02"}   # Union Budget (extend as scheduled)


def ro():
    return sqlite3.connect("file:%s?mode=ro" % OPT, uri=True)


def init_db():
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS legs (
      trade_date TEXT, leg TEXT, strike REAL, qty INTEGER,
      entry REAL, entry_ts TEXT, sl_level REAL,
      exit REAL, exit_ts TEXT, exit_reason TEXT,
      status TEXT, ltp REAL, ltp_ts TEXT, pnl_gross REAL,
      PRIMARY KEY (trade_date, leg));
    CREATE TABLE IF NOT EXISTS days (
      trade_date TEXT PRIMARY KEY, status TEXT, source TEXT,
      atm_strike REAL, entry_spot REAL,
      gross REAL, turnover REAL, net REAL, n_sl INTEGER, note TEXT,
      updated_at TEXT);
    """)
    return con


def nearest_expiry(c, day):
    r = c.execute(
        "SELECT MIN(expiry_date) FROM option_chain WHERE symbol=? "
        "AND snapshot_time >= ? AND snapshot_time < ?",
        (SYMBOL, day + "T00:00", day + "T23:59")).fetchone()
    return r[0] if r else None


def minute_rows(c, day, hhmm):
    """All chain rows for `day`'s own expiry in the given minute."""
    return c.execute(
        "SELECT strike, instrument_type, ltp, underlying_spot FROM option_chain "
        "WHERE symbol=? AND expiry_date=? AND snapshot_time >= ? AND snapshot_time < ? "
        "AND ltp > 0",
        (SYMBOL, day, "%sT%s:00" % (day, hhmm), "%sT%s:60" % (day, hhmm))).fetchall()


def leg_series(c, day, strike, opt_type):
    """(hh:mm, ltp) series for one leg from entry minute to square-off."""
    rows = c.execute(
        "SELECT snapshot_time, ltp FROM option_chain "
        "WHERE symbol=? AND expiry_date=? AND strike=? AND instrument_type=? "
        "AND snapshot_time >= ? AND snapshot_time <= ? AND ltp > 0 "
        "ORDER BY snapshot_time",
        (SYMBOL, day, strike, opt_type,
         "%sT%s:00" % (day, ENTRY_HHMM), "%sT15:30:59" % day)).fetchall()
    return [(t[11:16], p) for t, p in rows]


def pick_atm(rows):
    """From one minute's rows pick the ATM strike having BOTH legs quoted."""
    by_strike = {}
    spot = None
    for strike, typ, ltp, us in rows:
        spot = us or spot
        by_strike.setdefault(strike, {})[typ] = ltp
    cands = [s for s, d in by_strike.items() if "CE" in d and "PE" in d]
    if not cands or spot is None:
        return None, None, None
    atm = min(cands, key=lambda s: abs(s - spot))
    return atm, by_strike[atm], spot


def replay_day(c, day):
    """Deterministic replay of the day. Returns dict or None (no data/MISSED)."""
    rows = minute_rows(c, day, ENTRY_HHMM)
    atm, prices, spot = pick_atm(rows)
    if atm is None:
        return None
    legs = {}
    for typ in ("CE", "PE"):
        legs[typ] = dict(strike=atm, entry=prices[typ], sl=SL_MULT * prices[typ],
                         exit=None, exit_ts=None, reason=None, ltp=prices[typ],
                         ltp_ts=ENTRY_HHMM)
    series = {t: dict(leg_series(c, day, atm, t)) for t in ("CE", "PE")}
    minutes = sorted(set(series["CE"]) | set(series["PE"]))
    for mm in minutes:
        if mm <= ENTRY_HHMM:
            continue
        if mm >= EXIT_HHMM:
            break
        for typ in ("CE", "PE"):
            lg = legs[typ]
            ltp = series[typ].get(mm)
            if ltp is None:
                continue
            if lg["exit"] is None:
                lg["ltp"], lg["ltp_ts"] = ltp, mm
                if ltp >= lg["sl"]:
                    lg["exit"], lg["exit_ts"], lg["reason"] = ltp, mm, "SL"
                    other = legs["PE" if typ == "CE" else "CE"]
                    if other["exit"] is None:          # trail-to-breakeven
                        other["sl"] = min(other["sl"], other["entry"])
    # square-off survivors: first quote at/after 15:15 (series runs to 15:30 so
    # a missing 15:15 minute for one leg cannot leave it dangling); historical
    # days force-close on the last quote if even that window is absent
    historical = day < datetime.now().strftime("%Y-%m-%d")
    for typ in ("CE", "PE"):
        lg = legs[typ]
        if lg["exit"] is None:
            ordered = sorted(series[typ].items())
            post = [(mm, p) for mm, p in ordered if mm >= EXIT_HHMM]
            if post:
                mm, p = post[0]
                lg["exit"], lg["exit_ts"], lg["reason"] = p, mm, "TIME"
            elif ordered:
                mm, p = ordered[-1]
                if historical:
                    lg["exit"], lg["exit_ts"], lg["reason"] = p, mm, "TIME"
                else:                                  # day still running
                    lg["ltp"], lg["ltp_ts"] = p, mm
    return dict(atm=atm, spot=spot, legs=legs)


def day_totals(legs):
    gross = turnover = 0.0
    n_sl = 0
    closed = all(lg["exit"] is not None for lg in legs.values())
    for lg in legs.values():
        px_out = lg["exit"] if lg["exit"] is not None else lg["ltp"]
        gross += (lg["entry"] - px_out) * QTY
        turnover += (lg["entry"] + px_out) * QTY
        n_sl += 1 if lg["reason"] == "SL" else 0
    net = gross - COST_RATE * turnover - COST_FLAT
    return gross, turnover, net, n_sl, closed


def persist(con, day, res, source):
    gross, turnover, net, n_sl, closed = day_totals(res["legs"])
    now = datetime.now().isoformat(timespec="seconds")
    for typ, lg in res["legs"].items():
        px_out = lg["exit"] if lg["exit"] is not None else lg["ltp"]
        con.execute(
            "INSERT OR REPLACE INTO legs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (day, typ, lg["strike"], QTY, lg["entry"], ENTRY_HHMM, lg["sl"],
             lg["exit"], lg["exit_ts"], lg["reason"],
             "CLOSED" if lg["exit"] is not None else "OPEN",
             lg["ltp"], lg["ltp_ts"], (lg["entry"] - px_out) * QTY))
    con.execute(
        "INSERT OR REPLACE INTO days VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (day, "CLOSED" if closed else "OPEN", source, res["atm"], res["spot"],
         round(gross, 2), round(turnover, 2), round(net, 2), n_sl, "", now))
    con.commit()


def mark_missed(con, day, note):
    now = datetime.now().isoformat(timespec="seconds")
    con.execute("INSERT OR IGNORE INTO days VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (day, "MISSED", "live", None, None, 0, 0, 0, 0, note, now))
    con.commit()


def publish(con):
    days = [dict(zip([c[0] for c in con.execute("SELECT * FROM days LIMIT 0").description], r))
            for r in con.execute("SELECT * FROM days ORDER BY trade_date")]
    today = datetime.now().strftime("%Y-%m-%d")
    legs = [dict(zip([c[0] for c in con.execute("SELECT * FROM legs LIMIT 0").description], r))
            for r in con.execute("SELECT * FROM legs WHERE trade_date=?", (today,))]
    traded = [d for d in days if d["status"] in ("CLOSED", "OPEN")]
    cum = sum(d["net"] for d in traded)
    payload = dict(
        system="CSL 60 - DTE-0", mode="paper", symbol=SYMBOL,
        lots=LOTS, qty=QTY, sl_pct=60, entry=ENTRY_HHMM, exit=EXIT_HHMM,
        rules="expiry-day only - sell ATM straddle 09:16 - per-leg SL 1.60x - "
              "trail-to-BE after first stop - square off 15:15",
        study="/app/straddle-study",
        today=today, today_legs=legs,
        days=days[-60:], n_days=len(traded), cum_net=round(cum, 2),
        generated=datetime.now().isoformat(timespec="seconds"))
    blob = json.dumps(payload, indent=1)
    for pub in PUBS:
        try:
            os.makedirs(os.path.dirname(pub), exist_ok=True)
            tmp = pub + ".tmp"
            with open(tmp, "w") as f:
                f.write(blob)
            os.replace(tmp, pub)                       # atomic
        except OSError:
            pass


def mark():
    now = datetime.now()
    if now.weekday() >= 5:
        return
    day = now.strftime("%Y-%m-%d")
    if day in EVENT_SKIP:
        return
    with ro() as c:
        if nearest_expiry(c, day) != day:
            return                                     # not DTE-0: nothing to do
        con = init_db()
        already = con.execute("SELECT status FROM days WHERE trade_date=?",
                              (day,)).fetchone()
        if already and already[0] in ("CLOSED", "MISSED"):
            return
        if now.time() < dtime(9, 17):
            return                                     # entry minute not complete yet
        res = replay_day(c, day)
        if res is None:
            if now.strftime("%H:%M") > LATE_HHMM:
                mark_missed(con, day, "no 09:16 chain snapshot by %s" % LATE_HHMM)
                publish(con)
            return
        persist(con, day, res, "live")
        publish(con)
        con.close()


def seed():
    """Backfill every recorded NIFTY expiry day (provenance: replay)."""
    con = init_db()
    with ro() as c:
        rows = c.execute(
            "SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=? "
            "AND expiry_date < date('now','localtime') ORDER BY expiry_date",
            (SYMBOL,)).fetchall()
        for (day,) in rows:
            if day in EVENT_SKIP:
                continue
            live = con.execute("SELECT source FROM days WHERE trade_date=?",
                               (day,)).fetchone()
            if live and live[0] == "live":
                continue                               # live-first, never overwrite
            res = replay_day(c, day)
            if res is None:
                print("%s  no data - skipped" % day)
                continue
            persist(con, day, res, "replay")
            g, _t, n, n_sl, _cl = day_totals(res["legs"])
            print("%s  ATM %.0f  gross %+10.0f  net %+10.0f  SLs %d"
                  % (day, res["atm"], g, n, n_sl))
    publish(con)
    con.close()


def replay_cli(day):
    with ro() as c:
        res = replay_day(c, day)
    if res is None:
        print("no usable 09:16 snapshot for", day)
        return
    g, t, n, n_sl, closed = day_totals(res["legs"])
    print("%s  ATM %.0f (spot %.1f)  %s" %
          (day, res["atm"], res["spot"], "CLOSED" if closed else "OPEN"))
    for typ, lg in res["legs"].items():
        print("  %s %.0f  entry %.2f  sl %.2f  exit %s @ %s (%s)"
              % (typ, lg["strike"], lg["entry"], lg["sl"],
                 lg["exit"], lg["exit_ts"], lg["reason"]))
    print("  gross %+.0f  net %+.0f  (%d lots, qty %d)" % (g, n, LOTS, QTY))


def show():
    con = init_db()
    for r in con.execute("SELECT trade_date,status,source,atm_strike,net,n_sl "
                         "FROM days ORDER BY trade_date"):
        print(r)
    con.close()


if __name__ == "__main__":
    verb = sys.argv[1] if len(sys.argv) > 1 else "mark"
    with open(LOCK, "w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit(0)                                # previous run still going
        if verb == "mark":
            mark()
        elif verb == "seed":
            seed()
        elif verb == "replay":
            replay_cli(sys.argv[2])
        elif verb == "show":
            show()
