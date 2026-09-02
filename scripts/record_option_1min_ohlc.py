#!/usr/bin/env python3
"""Daily post-close capture of 1-MINUTE OHLC candles for index option contracts.

WHY THIS EXISTS
---------------
`option_chain` stores a once-a-minute LTP *poll*. A poll cannot see what happened
between polls, so it cannot see the intra-minute HIGH/LOW that a stop-loss actually
triggers on. That makes every stop-based options backtest unverifiable against our
own data, and makes maximum-adverse-excursion (the thing you need to calibrate a
stop) unmeasurable. 1-minute OHLC fixes both.

WHY IT MUST RUN DAILY
---------------------
Kite serves historical candles only for CURRENTLY LISTED contracts. Once a contract
expires its token returns `InputException: invalid token` - verified 2026-09-01 on
NIFTY26AUG21200CE. There is therefore NO way to backfill. Every trading day's
candles must be captured on that day, before the contract expires.

SCOPE
-----
Mirrors what the live chain recorder already tracked today, restricted to the
nearest EXPIRIES_TO_KEEP expiries per index. ~350-400 contracts/day across
NIFTY + BANKNIFTY + SENSEX, ~140k candles/day, ~5 GB/year.

Read-only against the broker. Writes only to options_data.db -> option_ohlc.
Touches no trading engine, holds no positions, places no orders.

Usage:
    python3 scripts/record_option_1min_ohlc.py                 # today
    python3 scripts/record_option_1min_ohlc.py --date 2026-09-01
    python3 scripts/record_option_1min_ohlc.py --expiries 4    # widen scope
    python3 scripts/record_option_1min_ohlc.py --dry-run
"""
import argparse
import datetime as dt
import logging
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "backtest_data", "options_data.db")
SYMBOLS = ("NIFTY", "BANKNIFTY", "SENSEX")
EXPIRIES_TO_KEEP = 2          # nearest N expiries per index
RATE_LIMIT_SLEEP = 0.35       # Kite historical: 3 req/sec
TIMEFRAME = "minute"
MAX_RETRIES = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("opt1min")


def ensure_schema(db):
    """option_ohlc already exists (created by backfill_options_data.py); be safe anyway."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS option_ohlc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument_token INTEGER NOT NULL,
            tradingsymbol VARCHAR(60) NOT NULL,
            symbol VARCHAR(15) NOT NULL,
            instrument_type VARCHAR(5) NOT NULL,
            strike REAL NOT NULL,
            expiry DATE NOT NULL,
            timeframe VARCHAR(10) DEFAULT '5minute',
            date TIMESTAMP NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume INTEGER, oi INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlc_unique
            ON option_ohlc(instrument_token, date);
        CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_strike_date
            ON option_ohlc(symbol, strike, instrument_type, date);
        CREATE INDEX IF NOT EXISTS idx_ohlc_expiry ON option_ohlc(expiry, symbol);
    """)
    db.commit()


def universe(db, day, n_expiries):
    """Contracts the chain recorder actually tracked on `day`, nearest N expiries,
    joined to their instrument_token from the daily instrument dump."""
    cur = db.cursor()
    out = []
    for sym in SYMBOLS:
        exps = [r[0] for r in cur.execute(
            "SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=? "
            "AND snapshot_time>=? AND snapshot_time<? ORDER BY expiry_date LIMIT ?",
            (sym, day + "T00:00", day + "T23:59", n_expiries))]
        if not exps:
            log.warning("%s: no chain rows on %s - skipping", sym, day)
            continue
        qmarks = ",".join("?" * len(exps))
        rows = cur.execute(
            f"SELECT DISTINCT oc.tradingsymbol, oc.symbol, oc.instrument_type, "
            f"       oc.strike, oc.expiry_date "
            f"FROM option_chain oc "
            f"WHERE oc.symbol=? AND oc.snapshot_time>=? AND oc.snapshot_time<? "
            f"AND oc.expiry_date IN ({qmarks})",
            (sym, day + "T00:00", day + "T23:59", *exps)).fetchall()
        for tsym, s, ikind, strike, exp in rows:
            tok = cur.execute(
                "SELECT instrument_token FROM instruments_archive WHERE tradingsymbol=? "
                "ORDER BY dump_date DESC LIMIT 1", (tsym,)).fetchone()
            if tok:
                out.append((tok[0], tsym, s, ikind, strike, exp))
        log.info("%s: %d expiries %s -> %d contracts", sym, len(exps), exps,
                 sum(1 for o in out if o[2] == sym))
    return out


def already_have(db, token, day):
    return db.execute(
        "SELECT 1 FROM option_ohlc WHERE instrument_token=? AND date>=? AND date<? LIMIT 1",
        (token, day + " 00:00:00", day + " 23:59:59")).fetchone() is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--expiries", type=int, default=EXPIRIES_TO_KEEP)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    day = a.date

    db = sqlite3.connect(DB_PATH, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    ensure_schema(db)

    contracts = universe(db, day, a.expiries)
    log.info("universe for %s: %d contracts", day, len(contracts))
    if a.dry_run:
        for c in contracts[:5]:
            log.info("  sample %s", c)
        log.info("DRY RUN - no API calls, no writes")
        return 0
    if not contracts:
        log.error("empty universe - is the chain recorder running?")
        return 1

    from services.kite_service import get_kite
    kite = get_kite()

    frm = dt.date.fromisoformat(day)
    ok = skip = fail = 0
    rows_written = 0
    failures = []

    for i, (token, tsym, sym, ikind, strike, exp) in enumerate(contracts, 1):
        if already_have(db, token, day):
            skip += 1
            continue
        candles = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                candles = kite.historical_data(token, frm, frm, TIMEFRAME, oi=True)
                break
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if "invalid token" in str(e).lower() or attempt == MAX_RETRIES:
                    failures.append((tsym, exp, msg))
                    break
                time.sleep(1.0)
        time.sleep(RATE_LIMIT_SLEEP)

        if not candles:
            fail += 1
            continue
        db.executemany(
            "INSERT OR IGNORE INTO option_ohlc "
            "(instrument_token, tradingsymbol, symbol, instrument_type, strike, expiry, "
            " timeframe, date, open, high, low, close, volume, oi) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(token, tsym, sym, ikind, strike, exp, TIMEFRAME,
              c["date"].strftime("%Y-%m-%d %H:%M:%S"),
              c["open"], c["high"], c["low"], c["close"],
              c.get("volume"), c.get("oi")) for c in candles])
        db.commit()
        rows_written += len(candles)
        ok += 1
        if i % 50 == 0:
            log.info("  %d/%d  ok=%d skip=%d fail=%d  candles=%d",
                     i, len(contracts), ok, skip, fail, rows_written)

    log.info("DONE %s  contracts ok=%d skipped(already had)=%d failed=%d  candles=%d",
             day, ok, skip, fail, rows_written)
    if failures:
        log.warning("failures (%d), first 10:", len(failures))
        for f in failures[:10]:
            log.warning("   %s exp=%s  %s", *f)
        expired = sum(1 for f in failures if "invalid token" in f[2].lower())
        if expired:
            log.error("%d failed with 'invalid token' - those contracts expired before "
                      "we captured them. Move this job EARLIER in the day.", expired)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
