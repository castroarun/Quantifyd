"""research/81 post-backfill repair pass. Run AFTER the history backfill ends.

Steps (all idempotent):
  1. ADJUSTMENT AUDIT: for every 5-min symbol, fetch a small FRESH Kite daily
     window (2024-06-02..2024-06-14, ~1 request/symbol) and compare against
     the stored 5-min session closes. Ratio drift > 2% => the stored series
     mixes adjustment bases (split/bonus after original download).
  2. REFETCH: for each drifted symbol: DELETE its 5-min rows, re-download
     2015-02-01 -> today fresh (single adjustment basis).
  3. BANKNIFTY: backfill 5-min 2015 -> current earliest via index token 260105
     (the NSE-EQ lookup can't resolve it).
  4. COALINDIA/ONGC daily re-store (previously lost the SQLite write-lock race).

Usage (VPS): cd /home/arun/quantifyd &&
  nohup venv/bin/python3 research/81_swing_edge_discovery/scripts/repair_pass.py \
      > /tmp/repair_pass.log 2>&1 &
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "research" / "81_swing_edge_discovery"
sys.path.insert(0, str(ROOT))
DB = ROOT / "backtest_data" / "market_data.db"
OUT = STUDY / "results" / "repair_report.json"

SAMPLE_FROM, SAMPLE_TO = datetime(2024, 6, 2), datetime(2024, 6, 14)
DRIFT_TOL = 0.02
BANKNIFTY_TOKEN = 260105


def stored_session_closes(sym: str) -> pd.Series:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    try:
        df = pd.read_sql_query(
            "SELECT date, close FROM market_data_unified WHERE symbol=? AND "
            "timeframe='5minute' AND date BETWEEN ? AND ? ORDER BY date",
            con, params=(sym, "2024-06-02", "2024-06-14 23:59:59"))
    finally:
        con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby(df["date"].dt.date)["close"].last()


def delete_5min(sym: str):
    con = sqlite3.connect(DB, timeout=120)
    try:
        n = con.execute("DELETE FROM market_data_unified WHERE symbol=? AND "
                        "timeframe='5minute'", (sym,)).rowcount
        con.commit()
        print(f"  deleted {n} rows for {sym}", flush=True)
    finally:
        con.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fno-only", action="store_true",
                    help="audit+refetch only F&O names (critical path for G5); "
                         "rest deferred to the full pass")
    args = ap.parse_args()

    from services.kite_service import get_kite
    from services.data_manager import CentralizedDataManager, FNO_LOT_SIZES
    from services.nifty500_universe import get_instrument_token

    kite = get_kite()
    kite.profile()
    dm = CentralizedDataManager(kite=kite)
    report = {"checked": 0, "drifted": [], "refetched": [], "errors": {},
              "banknifty": None, "daily_fixed": []}

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    syms = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='5minute'")]
    con.close()
    if args.fno_only:
        syms = [s for s in syms if s in FNO_LOT_SIZES]
        print(f"--fno-only: auditing {len(syms)} F&O names")

    # ---- 1. adjustment audit ----
    print(f"=== adjustment audit: {len(syms)} symbols ===", flush=True)
    for i, sym in enumerate(sorted(syms), 1):
        if sym in ("NIFTY50", "INDIAVIX", "BANKNIFTY"):
            continue
        try:
            tok = get_instrument_token(sym)
            if not tok:
                report["errors"][sym] = "no token (delisted?)"
                continue
            fresh = kite.historical_data(tok, SAMPLE_FROM, SAMPLE_TO, "day")
            time.sleep(0.35)
            if not fresh:
                report["errors"][sym] = "no fresh daily data"
                continue
            fclose = {r["date"].date(): r["close"] for r in fresh}
            stored = stored_session_closes(sym)
            common = [d for d in stored.index if d in fclose]
            if not common:
                report["errors"][sym] = "no overlapping sample days"
                continue
            ratios = [stored[d] / fclose[d] for d in common]
            drift = max(abs(r - 1) for r in ratios)
            report["checked"] += 1
            if drift > DRIFT_TOL:
                report["drifted"].append({"symbol": sym, "drift": round(drift, 4)})
                print(f"[{i}] {sym}: DRIFT {drift:.1%}", flush=True)
        except Exception as e:
            report["errors"][sym] = str(e)[:120]
        if i % 50 == 0:
            print(f"[{i}/{len(syms)}] audited", flush=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"audit done: {len(report['drifted'])} drifted, "
          f"{len(report['errors'])} errors", flush=True)

    # ---- 2. refetch drifted ----
    for d in report["drifted"]:
        sym = d["symbol"]
        try:
            delete_5min(sym)
            ok, fail, errs = dm.download_data(
                symbols=[sym], timeframe="5minute",
                from_date=datetime(2015, 2, 1), to_date=datetime.now())
            if ok:
                report["refetched"].append(sym)
            else:
                report["errors"][sym] = "refetch: " + "; ".join(errs)[:120]
        except Exception as e:
            report["errors"][sym] = "refetch: " + str(e)[:120]
        OUT.write_text(json.dumps(report, indent=1))

    # ---- 3. BANKNIFTY via index token ----
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
        cur_min = con.execute(
            "SELECT MIN(date) FROM market_data_unified WHERE symbol='BANKNIFTY' "
            "AND timeframe='5minute'").fetchone()[0]
        con.close()
        to_d = datetime.strptime(cur_min.split(" ")[0], "%Y-%m-%d") - timedelta(days=1)
        cur, chunks = datetime(2015, 2, 1), []
        while cur < to_d:
            end = min(cur + timedelta(days=7), to_d)
            data = kite.historical_data(BANKNIFTY_TOKEN, cur, end, "5minute")
            if data:
                chunks.append(pd.DataFrame(data))
            time.sleep(0.35)
            cur = end + timedelta(days=1)
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.sort_values("date").drop_duplicates(subset=["date"])
            dm._store_data("BANKNIFTY", "5minute", df)
            report["banknifty"] = f"stored {len(df)} bars from {df['date'].min()}"
            print(report["banknifty"], flush=True)
    except Exception as e:
        report["banknifty"] = "ERROR: " + str(e)[:200]

    # ---- 4. COALINDIA / ONGC daily ----
    for sym in ("COALINDIA", "ONGC"):
        try:
            ok, fail, errs = dm.download_data(
                symbols=[sym], timeframe="day",
                from_date=datetime(2005, 1, 1), to_date=datetime.now())
            if ok:
                report["daily_fixed"].append(sym)
        except Exception as e:
            report["errors"][sym] = "daily: " + str(e)[:120]

    OUT.write_text(json.dumps(report, indent=1))
    print("=== repair pass complete ===")
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 30
                          else f"{len(v)} items") for k, v in report.items()},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
