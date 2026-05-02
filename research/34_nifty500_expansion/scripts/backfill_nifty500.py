"""Phase A — Backfill 5-min data for Nifty 500 stocks not yet in DB.

Reads `data/nifty500_list.csv`, computes the gap vs existing
`market_data_unified` 5-min rows, and downloads missing stocks via the
existing `services.data_manager.CentralizedDataManager`.

Resumable: skips stocks that already have any 5-min rows in the DB
(if a stock failed mid-download last time, delete its rows manually
and re-run, or just let download_data refetch the full window).

Period: 2024-03-18 to 2026-03-25 (matches Cohort B coverage).

Logs progress to `research/34_nifty500_expansion/results/backfill.log`
and updates the Status section of NIFTY500_EXPANSION_SWEEP_STATUS.md
every 10 stocks.
"""
from __future__ import annotations

import csv
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from services.data_manager import CentralizedDataManager  # noqa: E402
from services.kite_service import get_kite  # noqa: E402

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
NIFTY500_CSV = ROOT / "data" / "nifty500_list.csv"
DB_PATH = ROOT / "backtest_data" / "market_data.db"
RESEARCH = ROOT / "research" / "34_nifty500_expansion"
RESULTS = RESEARCH / "results"
LOG_FILE = RESULTS / "backfill.log"
STATUS_MD = RESEARCH / "NIFTY500_EXPANSION_SWEEP_STATUS.md"

RESULTS.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Period
# -----------------------------------------------------------------------------
FROM_DATE = datetime(2024, 3, 18)
TO_DATE = datetime(2026, 3, 25)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("backfill")


def get_missing_stocks() -> list[str]:
    n500: set[str] = set()
    with NIFTY500_CSV.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            n500.add(r["Symbol"].strip())
    log.info(f"Nifty 500 list: {len(n500)} stocks")

    con = sqlite3.connect(DB_PATH)
    have = set(
        r[0]
        for r in con.execute(
            "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='5minute'"
        ).fetchall()
    )
    con.close()
    log.info(f"Already have 5-min data: {len(have)} stocks")

    missing = sorted(n500 - have)
    log.info(f"Missing → need to download: {len(missing)}")
    return missing


def update_status(stocks_done: int, total: int, last_symbol: str, started_at: float):
    """Rewrite the Section 4 'Status' running log header with current progress."""
    if not STATUS_MD.exists():
        return
    txt = STATUS_MD.read_text(encoding="utf-8")
    elapsed_min = (time.time() - started_at) / 60.0
    pct = 100.0 * stocks_done / max(total, 1)
    state_block = (
        f"**State:** PHASE A RUNNING (backfill)\n"
        f"**Last update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}\n"
        f"**Stocks downloaded:** {stocks_done} / {total}  ({pct:.1f}%)\n"
        f"**Last completed symbol:** {last_symbol}\n"
        f"**Elapsed:** {elapsed_min:.1f} min\n"
    )
    # Replace any existing State / Last update / Stocks / Last completed / Elapsed
    # block right after the "## 4. Status" heading. Use a simple regex.
    txt = re.sub(
        r"(## 4\. Status[^\n]*\n+)([\s\S]*?)(?=\n### Event log)",
        lambda m: m.group(1) + state_block + "\n",
        txt,
    )
    STATUS_MD.write_text(txt, encoding="utf-8")


def main():
    started_at = time.time()
    log.info("=" * 60)
    log.info("Phase A — Nifty 500 5-min backfill")
    log.info("=" * 60)

    missing = get_missing_stocks()
    if not missing:
        log.info("Nothing to download. Already in sync.")
        return
    log.info(f"Period: {FROM_DATE.date()} to {TO_DATE.date()}")
    log.info(f"First 10 missing: {missing[:10]}")

    # Kite connection
    log.info("Establishing Kite connection...")
    try:
        kite = get_kite()
        kite.profile()  # validate token
    except Exception as e:
        log.error(f"Kite auth failed: {e}")
        log.error("Run TOTP auto-login first (POST /api/auth/auto-login on the VPS).")
        sys.exit(1)
    log.info("Kite connection OK")

    dm = CentralizedDataManager(kite=kite)

    total = len(missing)
    successes = 0
    failures = 0
    errors = []

    def cb(idx, _total, symbol, status):
        if status == "completed":
            update_status(idx, total, symbol, started_at)

    # Run download in batches of 10 so we can update STATUS without too much overhead
    batch_size = 10
    for start in range(0, total, batch_size):
        batch = missing[start : start + batch_size]
        try:
            s, f, errs = dm.download_data(
                symbols=batch,
                timeframe="5minute",
                from_date=FROM_DATE,
                to_date=TO_DATE,
                progress_callback=cb,
            )
            successes += s
            failures += f
            errors.extend(errs)
        except KeyboardInterrupt:
            log.warning("Interrupted by user")
            break
        except Exception as e:
            log.error(f"Batch error (start={start}): {e}")
            failures += len(batch)
            errors.append(f"batch_{start}: {e}")

        update_status(start + len(batch), total, batch[-1], started_at)
        log.info(
            f"Batch {start//batch_size + 1}/{(total + batch_size - 1)//batch_size} "
            f"done. Cumulative: {successes} ok, {failures} fail."
        )

    elapsed = (time.time() - started_at) / 60.0
    log.info("=" * 60)
    log.info(f"Phase A complete in {elapsed:.1f} min")
    log.info(f"Successes: {successes}, Failures: {failures}")
    if errors:
        log.warning(f"Errors ({len(errors)}):")
        for e in errors[:20]:
            log.warning(f"  {e}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
