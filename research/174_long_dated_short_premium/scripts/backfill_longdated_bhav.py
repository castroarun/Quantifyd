#!/usr/bin/env python3
"""
research/174 — P0a: repair the long-dated hole in nse_options_bhav.

WHY. research/89's stock downloader (`download_nse_bhav_stocks.py`) carries
`MAX_DTE = 75` and resumes by trade-date. Every session it touched was therefore
recorded as "done" while holding only the near chain, so the production downloader
(`download_nse_bhav.py`, no DTE cap) skipped those sessions for ever. Result: NIFTY
expiries beyond ~75 DTE are absent for 2016-01 -> 2024-02, 2026-04/05/06 and 2026-09,
present elsewhere. It is a download artifact, not a market fact.

WHAT THIS DOES. Re-downloads the NSE F&O bhavcopy for exactly the sessions whose
maximum listed DTE is <= MAX_DTE_SEEN, parses index options with NO DTE cap and NO
strike band, and writes them to a STAGING database first
(`results/bhav_longdated_stage.db`, identical schema). A separate `--merge` pass
INSERT-OR-IGNOREs the staging rows into `backtest_data/market_data.db`.

Staging exists so the ~2,100-session download can run during market hours without
ever taking a write lock on the DB the live executors read. Run `--merge` after
15:40 IST.

Safe to interrupt: commits per session, recomputes its to-do list on restart.

Usage:
  python backfill_longdated_bhav.py                 # download all truncated sessions
  python backfill_longdated_bhav.py 2019-06-14 ...  # download named sessions only
  python backfill_longdated_bhav.py --merge         # merge staging -> market_data.db
"""
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/arun/quantifyd")
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import download_nse_bhav as P                               # noqa: E402

MAX_DTE_SEEN = 80          # a session with nothing beyond this is truncated
RATE_LIMIT = 1.2
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
STAGE = RESULTS / "bhav_longdated_stage.db"
LOG = RESULTS / "backfill.log"

DDL = """CREATE TABLE IF NOT EXISTS nse_options_bhav (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trade_date TEXT, symbol TEXT, expiry_date TEXT,
    strike REAL, option_type TEXT, open REAL, high REAL, low REAL, close REAL,
    settle_price REAL, contracts INTEGER, value_in_lakhs REAL, open_interest INTEGER,
    change_in_oi INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, symbol, expiry_date, strike, option_type))"""

COLS = ("trade_date, symbol, expiry_date, strike, option_type, open, high, low, close, "
        "settle_price, contracts, value_in_lakhs, open_interest, change_in_oi")


def log(m):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def stage_con():
    c = sqlite3.connect(str(STAGE))
    c.execute(DDL)
    c.commit()
    return c


def truncated_sessions(limit_from="2016-01-01"):
    con = sqlite3.connect("file:%s?mode=ro" % P.DB_PATH, uri=True)
    q = """
    SELECT trade_date, MAX(CAST(julianday(expiry_date)-julianday(trade_date) AS INT)) mx
    FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date>=?
    GROUP BY trade_date HAVING mx<=? ORDER BY trade_date
    """
    out = [r[0] for r in con.execute(q, (limit_from, MAX_DTE_SEEN))]
    con.close()
    return out


def already_staged(sc):
    return {r[0] for r in sc.execute("SELECT DISTINCT trade_date FROM nse_options_bhav")}


def do_download(only=None):
    sc = stage_con()
    todo = only if only else truncated_sessions()
    have = already_staged(sc)
    todo = [d for d in todo if d not in have]
    log("sessions to repair: %d (already staged %d)" % (len(todo), len(have)))
    if not todo:
        return
    sess = P.create_session()
    ok = holiday = err = 0
    for i, ds in enumerate(todo, 1):
        d = datetime.strptime(ds, "%Y-%m-%d").date()
        try:
            rows, nn, nb, status = P.download_and_parse(sess, d)
        except Exception as e:                                  # noqa: BLE001
            log("  %s EXC %s" % (ds, e))
            status, rows = "error", []
        if status == "ok" and rows:
            sc.executemany("INSERT OR IGNORE INTO nse_options_bhav (%s) VALUES (%s)"
                           % (COLS, ",".join("?" * 14)), rows)
            sc.commit()
            ok += 1
            if i % 50 == 0:
                log("  %4d/%d %s parsed=%d  (ok=%d hol=%d err=%d)"
                    % (i, len(todo), ds, len(rows), ok, holiday, err))
        elif status == "holiday":
            holiday += 1
        else:
            err += 1
            log("  %4d/%d %s STATUS=%s" % (i, len(todo), ds, status))
            if status == "blocked":
                log("  blocked -> new session, sleep 30s")
                time.sleep(30)
                sess = P.create_session()
        time.sleep(RATE_LIMIT)
    n = sc.execute("SELECT COUNT(*) FROM nse_options_bhav").fetchone()[0]
    log("DOWNLOAD DONE ok=%d holiday=%d err=%d  staged_rows=%d" % (ok, holiday, err, n))


def do_merge():
    sc = sqlite3.connect("file:%s?mode=ro" % STAGE, uri=True)
    mc = sqlite3.connect(str(P.DB_PATH), timeout=60)
    before = mc.execute("SELECT COUNT(*) FROM nse_options_bhav").fetchone()[0]
    batch, total = [], 0
    for r in sc.execute("SELECT %s FROM nse_options_bhav" % COLS):
        batch.append(r)
        if len(batch) >= 20000:
            mc.executemany("INSERT OR IGNORE INTO nse_options_bhav (%s) VALUES (%s)"
                           % (COLS, ",".join("?" * 14)), batch)
            mc.commit()
            total += len(batch)
            batch = []
            log("  merged %d..." % total)
    if batch:
        mc.executemany("INSERT OR IGNORE INTO nse_options_bhav (%s) VALUES (%s)"
                       % (COLS, ",".join("?" * 14)), batch)
        mc.commit()
        total += len(batch)
    after = mc.execute("SELECT COUNT(*) FROM nse_options_bhav").fetchone()[0]
    log("MERGE DONE staged=%d  table %d -> %d  (+%d new)" % (total, before, after, after - before))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--merge":
        do_merge()
    else:
        do_download(args or None)
