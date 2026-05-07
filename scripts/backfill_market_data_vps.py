"""Catchup backfill — bring VPS market_data.db up to today.

This script runs ON THE VPS only. It refuses to run on laptop (VPS-canonical
rule from 2026-05-07). Use cases:

  1. Catchup mode (default): scans every symbol with existing 5-min data,
     finds the latest stored date, and downloads from there to "today".
     Idempotent — safe to re-run anytime.

  2. Symbol-list mode: pass a comma-separated list to backfill specific
     symbols only.

  3. Range mode: pass --from YYYY-MM-DD --to YYYY-MM-DD to override.

Usage on VPS (via ssh/paramiko):

    cd /home/arun/quantifyd
    venv/bin/python3 scripts/backfill_market_data_vps.py
    venv/bin/python3 scripts/backfill_market_data_vps.py --timeframe 5minute
    venv/bin/python3 scripts/backfill_market_data_vps.py --symbols HAL,RELIANCE
    venv/bin/python3 scripts/backfill_market_data_vps.py --from 2026-03-26 --to 2026-05-07

Designed for unattended overnight runs — writes a STATUS_MD file so you
can resume / monitor without re-deriving context.
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "backtest_data" / "market_data.db"
STATUS_MD = ROOT / "research" / "34_nifty500_expansion" / "MARKET_DATA_CATCHUP_BACKFILL_STATUS.md"

# Add project root to path so services import works
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill")


def _kite_session():
    """Lazy import; needs valid token at backtest_data/access_token.json."""
    from services.kite_service import get_kite
    k = get_kite()
    if k is None:
        raise RuntimeError(
            "Kite session unavailable. Refresh token via:\n"
            "  curl -X POST http://127.0.0.1:5000/api/auth/auto-login\n"
        )
    return k


def existing_universe(timeframe: str) -> dict[str, str]:
    """Return {symbol: max_date} for every symbol with existing data on this tf."""
    if not DB_PATH.exists():
        return {}
    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute(
            "SELECT symbol, MAX(date) FROM market_data_unified "
            "WHERE timeframe=? GROUP BY symbol",
            (timeframe,)
        ).fetchall()
    finally:
        con.close()
    return {r[0]: r[1] for r in rows}


def _next_day_after(max_date_str: str) -> date:
    """Parse 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD' → next calendar date."""
    s = max_date_str.split(" ")[0]
    d = datetime.strptime(s, "%Y-%m-%d").date()
    return d + timedelta(days=1)


def write_status(state: str, *, total: int = 0, done: int = 0,
                 failed: int = 0, current: str = "", errors: list[str] | None = None,
                 timeframe: str = "", from_date: date | None = None,
                 to_date: date | None = None):
    """Write a live STATUS_MD file (binding rule: no PROGRESS, use STATUS)."""
    STATUS_MD.parent.mkdir(parents=True, exist_ok=True)
    pct = (done / total * 100.0) if total else 0.0
    lines = [
        "# Market Data Catchup Backfill — STATUS\n\n",
        f"**State:** {state}\n",
        f"**Timeframe:** {timeframe}\n",
        f"**Range:** {from_date} → {to_date}\n",
        f"**Progress:** {done}/{total} ({pct:.1f}%)  ·  failed: {failed}\n",
        f"**Current symbol:** {current}\n",
        f"**Last update:** {datetime.now().isoformat(timespec='seconds')} IST\n",
    ]
    if errors:
        lines.append("\n## Recent errors (last 10)\n\n")
        for e in errors[-10:]:
            lines.append(f"- {e}\n")
    lines.append("\n## Resume\n\nRe-run the same command — it auto-skips up-to-date symbols.\n")
    STATUS_MD.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="5minute",
                        choices=["day", "60minute", "15minute", "5minute"])
    parser.add_argument("--symbols", default="",
                        help="comma-sep list; omit to catchup every symbol with existing data")
    parser.add_argument("--from", dest="from_date", default="",
                        help="YYYY-MM-DD; omit to start from each symbol's last stored date")
    parser.add_argument("--to", dest="to_date", default="",
                        help="YYYY-MM-DD; omit to use today")
    args = parser.parse_args()

    # Init data manager (this triggers the VPS-only guard from data_manager.py)
    from services.data_manager import CentralizedDataManager
    kite = _kite_session()
    dm = CentralizedDataManager(kite=kite)

    # Resolve target universe
    existing = existing_universe(args.timeframe)
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = sorted(existing.keys())
    if not symbols:
        logger.error(f"No symbols found with existing {args.timeframe} data — pass --symbols")
        return 2

    today = date.today()
    to_date = (datetime.strptime(args.to_date, "%Y-%m-%d").date()
               if args.to_date else today)

    explicit_from = (datetime.strptime(args.from_date, "%Y-%m-%d").date()
                     if args.from_date else None)

    logger.info(f"=== Catchup backfill ===")
    logger.info(f"timeframe={args.timeframe}, target_to={to_date}, symbols={len(symbols)}")
    write_status("RUNNING", total=len(symbols), timeframe=args.timeframe,
                 to_date=to_date)

    done = 0
    failed = 0
    errors: list[str] = []
    t_start = time.time()

    for i, sym in enumerate(symbols, 1):
        # Per-symbol from_date
        if explicit_from is not None:
            from_d = explicit_from
        else:
            max_d = existing.get(sym)
            if not max_d:
                # Symbol has no existing data — skip in catchup mode
                logger.info(f"[{i}/{len(symbols)}] {sym}: no existing data, skipping")
                continue
            from_d = _next_day_after(max_d)

        if from_d > to_date:
            logger.info(f"[{i}/{len(symbols)}] {sym}: up to date ({existing.get(sym)})")
            done += 1
            continue

        elapsed = (time.time() - t_start) / 60.0
        write_status("RUNNING", total=len(symbols), done=done, failed=failed,
                     current=sym, errors=errors, timeframe=args.timeframe,
                     from_date=from_d, to_date=to_date)
        logger.info(f"[{i}/{len(symbols)}] {sym}: {from_d} → {to_date}  (elapsed {elapsed:.1f}m)")

        try:
            from_dt = datetime.combine(from_d, datetime.min.time())
            to_dt = datetime.combine(to_date, datetime.max.time())
            # download_data takes a list, but we call it per-symbol for granular error handling
            ok, fail, errs = dm.download_data(
                symbols=[sym], timeframe=args.timeframe,
                from_date=from_dt, to_date=to_dt,
            )
            if fail:
                failed += 1
                errors.extend(errs)
            else:
                done += 1
        except KeyboardInterrupt:
            logger.warning("Interrupted by user — writing STATUS and exiting cleanly")
            write_status("INTERRUPTED", total=len(symbols), done=done, failed=failed,
                         errors=errors, timeframe=args.timeframe,
                         from_date=explicit_from or date(2000, 1, 1), to_date=to_date)
            return 130
        except Exception as e:
            failed += 1
            err = f"{sym}: {e}"
            errors.append(err)
            logger.error(err)

    elapsed = (time.time() - t_start) / 60.0
    state = "DONE" if failed == 0 else "DONE_WITH_ERRORS"
    write_status(state, total=len(symbols), done=done, failed=failed,
                 errors=errors, timeframe=args.timeframe,
                 from_date=explicit_from or date(2000, 1, 1), to_date=to_date)
    logger.info(f"=== {state} ===  done={done} failed={failed}  ({elapsed:.1f}m)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
