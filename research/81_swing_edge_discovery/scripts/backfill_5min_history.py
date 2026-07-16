"""research/81 — 5-min HISTORY backfill: 2015-02-01 -> each symbol's earliest bar.

Runs ON THE VPS only (data_manager's VPS-only guard enforces this).
The inverse of scripts/backfill_market_data_vps.py (which fills forward):
this fills BACKWARD from each symbol's current min(date) to 2015-02-01.

Design goals (see NIFTY500_HISTORY_BACKFILL_5MIN_RUN_STATUS.md):
  - Resumable: checkpoint JSON of completed symbols; re-run skips them.
  - Idempotent: _store_data dedups on (symbol, timeframe, date).
  - Token-expiry resilient: Kite tokens die daily ~06:00; the 08:5x
    auto-login cron refreshes backtest_data/access_token.json. We validate
    the session before AND after each symbol; only mark done if still valid
    afterwards (guards silent mid-symbol holes, since _fetch_from_kite
    swallows per-chunk errors).
  - Live STATUS-MD rewritten at every state transition.

Usage (VPS):
    cd /home/arun/quantifyd
    venv/bin/python3 research/81_swing_edge_discovery/scripts/backfill_5min_history.py
    # options:
    #   --symbols HDFCBANK,RELIANCE   restrict universe (smoke tests)
    #   --limit 5                     stop after N symbols (smoke tests)
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "backtest_data" / "market_data.db"
STUDY_DIR = ROOT / "research" / "81_swing_edge_discovery"
STATUS_MD = STUDY_DIR / "NIFTY500_HISTORY_BACKFILL_5MIN_RUN_STATUS.md"
CHECKPOINT = STUDY_DIR / "results" / "backfill_checkpoint.json"

TARGET_FROM = date(2015, 2, 1)   # Kite 5-min depth starts ~Feb 2015
TIMEFRAME = "5minute"
AUTH_WAIT_SECS = 600             # re-check token file every 10 min when auth is down

# Deep-history names first: fast wins that unblock engine dev on 2015+ data.
PRIORITY = [
    "BANKNIFTY", "HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "TCS",
    "SBIN", "ITC", "HINDUNILVR", "KOTAKBANK", "BHARTIARTL",
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("backfill81")


# ---------------------------------------------------------------- helpers

def earliest_bars() -> dict[str, str]:
    """{symbol: min(date)} for every symbol holding 5-min data."""
    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute(
            "SELECT symbol, MIN(date) FROM market_data_unified "
            "WHERE timeframe=? GROUP BY symbol", (TIMEFRAME,)).fetchall()
    finally:
        con.close()
    return {r[0]: r[1] for r in rows}


def symbol_min(sym: str) -> str | None:
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute(
            "SELECT MIN(date) FROM market_data_unified "
            "WHERE symbol=? AND timeframe=?", (sym, TIMEFRAME)).fetchone()
    finally:
        con.close()
    return row[0] if row else None


def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {"done": {}, "failed": {}, "started_at": None}


def save_checkpoint(ck: dict) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(ck, indent=1), encoding="utf-8")


def session_valid(kite) -> bool:
    try:
        kite.profile()
        return True
    except Exception:
        return False


def ensure_session():
    """Return a validated KiteConnect, waiting through the daily token gap."""
    from services.kite_service import get_kite
    waited = False
    while True:
        try:
            kite = get_kite()
            if session_valid(kite):
                if waited:
                    logger.info("Auth restored — resuming")
                return kite
        except Exception as e:
            logger.warning(f"get_kite failed: {e}")
        waited = True
        logger.warning(f"Kite session invalid — waiting {AUTH_WAIT_SECS}s "
                       "(auto-login cron refreshes the token ~08:5x IST)")
        write_status("WAITING_AUTH")
        time.sleep(AUTH_WAIT_SECS)


# ---------------------------------------------------------------- status-md

_state = {"total": 0, "done": 0, "skipped": 0, "failed": 0, "current": "",
          "t_start": None, "events": []}


def event(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    _state["events"].append(f"| {ts} | {msg} |")
    logger.info(msg)


def write_status(run_state: str) -> None:
    s = _state
    elapsed_m = (time.time() - s["t_start"]) / 60 if s["t_start"] else 0
    processed = s["done"] + s["failed"]
    eta_h = ((s["total"] - s["skipped"] - processed) * (elapsed_m / processed) / 60
             if processed else 0)
    body = f"""# Nifty-500 5-Min History Backfill — 2015→2024 for ~370 Names (Kite, VPS)

STATUS: {run_state}

Part of **research/81 Swing Edge Discovery** (`EDGE_DISCOVERY_81_STUDY_STATE.md` at repo root).
Full pre-registration (Ask/Base/Plan) is preserved in git history of this file;
this copy is rewritten live by `scripts/backfill_5min_history.py`.

## Status (live)

- **State:** {run_state}
- **Progress:** {s['done']} done · {s['skipped']} skipped (already deep) · {s['failed']} failed · of {s['total']} symbols
- **Current symbol:** {s['current']}
- **Elapsed:** {elapsed_m:.0f} min · **crude ETA:** {eta_h:.1f} h remaining
- **Last update:** {datetime.now().isoformat(timespec='seconds')} IST

## Event log (last 30)

| Time | Event |
|---|---|
{chr(10).join(s['events'][-30:])}

## Crash recovery

Resume (idempotent — skips checkpointed symbols):
```
cd /home/arun/quantifyd
nohup bash -c 'venv/bin/python3 research/81_swing_edge_discovery/scripts/backfill_5min_history.py && venv/bin/python3 scripts/backfill_market_data_vps.py --timeframe 5minute' > /tmp/backfill_5min_history.log 2>&1 &
```
Checkpoint: `research/81_swing_edge_discovery/results/backfill_checkpoint.json`
Log: `/tmp/backfill_5min_history.log` · If stuck WAITING_AUTH after 09:00 IST:
`curl -X POST http://127.0.0.1:5000/api/auth/auto-login`
"""
    STATUS_MD.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="comma-sep subset (smoke test)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N symbols")
    args = ap.parse_args()

    from services.data_manager import CentralizedDataManager

    ck = load_checkpoint()
    if not ck.get("started_at"):
        ck["started_at"] = datetime.now().isoformat(timespec="seconds")

    universe = earliest_bars()
    if args.symbols:
        wanted = [x.strip() for x in args.symbols.split(",") if x.strip()]
        universe = {s: universe[s] for s in wanted if s in universe}

    # priority: deep names, then F&O universe (critical path for the G5 book),
    # then the rest alphabetically
    prio = list(PRIORITY)
    try:
        from services.data_manager import FNO_LOT_SIZES
        prio += sorted(FNO_LOT_SIZES.keys())
    except Exception:
        pass
    prio = list(dict.fromkeys(prio))
    ordered = [s for s in prio if s in universe] + \
              sorted(s for s in universe if s not in prio)

    _state["total"] = len(ordered)
    _state["t_start"] = time.time()
    event(f"Run start — {len(ordered)} symbols, target from {TARGET_FROM}")
    write_status("RUNNING")

    dm = CentralizedDataManager(kite=None)  # kite attached per-symbol
    processed = 0

    for i, sym in enumerate(ordered, 1):
        if args.limit and processed >= args.limit:
            event(f"--limit {args.limit} reached, stopping cleanly")
            break

        if sym in ck["done"]:
            _state["skipped"] += 1
            continue

        cur_min = universe[sym].split(" ")[0]
        cur_min_d = datetime.strptime(cur_min, "%Y-%m-%d").date()
        if cur_min_d <= TARGET_FROM + timedelta(days=28):
            ck["done"][sym] = {"min": cur_min, "note": "already deep"}
            save_checkpoint(ck)
            _state["skipped"] += 1
            continue

        _state["current"] = sym
        write_status("RUNNING")
        kite = ensure_session()
        dm.kite = kite
        processed += 1

        t0 = time.time()
        try:
            from_dt = datetime.combine(TARGET_FROM, datetime.min.time())
            to_dt = datetime.combine(cur_min_d - timedelta(days=1),
                                     datetime.max.time())
            ok, fail, errs = dm.download_data(
                symbols=[sym], timeframe=TIMEFRAME,
                from_date=from_dt, to_date=to_dt)
            still_valid = session_valid(kite)
            new_min = symbol_min(sym) or cur_min
            mins = (time.time() - t0) / 60
            if ok and still_valid:
                ck["done"][sym] = {"min": new_min.split(" ")[0],
                                   "was": cur_min, "mins": round(mins, 1)}
                ck["failed"].pop(sym, None)
                _state["done"] += 1
                event(f"[{i}/{len(ordered)}] {sym} done in {mins:.1f}m — "
                      f"earliest {cur_min} → {new_min.split(' ')[0]}")
            elif ok and not still_valid:
                # token died mid-symbol: possible silent holes — do NOT mark done
                _state["failed"] += 1
                ck["failed"][sym] = "token died mid-symbol; will retry"
                event(f"[{i}/{len(ordered)}] {sym} UNVERIFIED (token died mid-run) — will retry")
            else:
                err = "; ".join(errs)[:200]
                if "No data fetched" in err:
                    # no pre-existing history at Kite (late listing) — terminal, fine
                    ck["done"][sym] = {"min": cur_min, "note": "no earlier history at Kite"}
                    _state["done"] += 1
                    event(f"[{i}/{len(ordered)}] {sym}: no earlier history at Kite (listed later?)")
                else:
                    _state["failed"] += 1
                    ck["failed"][sym] = err
                    event(f"[{i}/{len(ordered)}] {sym} FAILED: {err}")
        except KeyboardInterrupt:
            event("Interrupted — checkpoint saved, exiting cleanly")
            save_checkpoint(ck)
            write_status("INTERRUPTED")
            return 130
        except Exception as e:
            _state["failed"] += 1
            ck["failed"][sym] = str(e)[:200]
            event(f"[{i}/{len(ordered)}] {sym} EXCEPTION: {e}")

        save_checkpoint(ck)

    # summary
    state = "DONE" if _state["failed"] == 0 else "DONE_WITH_ERRORS"
    event(f"Run complete — done={_state['done']} skipped={_state['skipped']} "
          f"failed={_state['failed']}")
    write_status(state)
    save_checkpoint(ck)
    return 0 if _state["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
