"""
One-shot Kite download of the GTAA ETF menu into market_data.db (VPS-only).

Run on VPS:  venv/bin/python research/63_gtaa_etf_rotation/scripts/download_etfs.py

Idempotent: download_data upserts; re-running just refreshes recent candles.
"""
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from services.kite_service import get_kite
from services.data_manager import CentralizedDataManager

# The slide's 3 core ETFs + the extended menu (safe asset = LIQUIDBEES).
SYMBOLS = [
    "NIFTYBEES",   # Nifty 50 (already present; refresh anyway)
    "GOLDBEES",    # Gold
    "MON100",      # Motilal Oswal Nasdaq 100 (the slide's "N100")
    "LIQUIDBEES",  # 1-day-rate / cash proxy (safe asset)
    "JUNIORBEES",  # Nifty Next 50
    "BANKBEES",    # Bank
    "SILVERBEES",  # Silver
]

FROM = datetime(2005, 1, 1)
TO = datetime.now()


def main():
    k = get_kite()
    if k is None:
        print("FATAL: no kite (check backtest_data/access_token.json)")
        sys.exit(1)

    dm = CentralizedDataManager()
    dm.set_kite(k)

    def cb(i, n, sym, status):
        print(f"  [{i}/{n}] {sym}: {status}", flush=True)

    ok, fail, errs = dm.download_data(
        symbols=SYMBOLS, timeframe="day",
        from_date=FROM, to_date=TO, progress_callback=cb,
    )
    print(f"\nDONE  success={ok}  failed={fail}")
    for e in errs:
        print("  ERR", e)

    # Coverage report
    import sqlite3
    c = sqlite3.connect(str(ROOT / "backtest_data" / "market_data.db"))
    print("\nCoverage:")
    for s in SYMBOLS:
        row = c.execute(
            "SELECT MIN(date),MAX(date),COUNT(*) FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day'", (s,)).fetchone()
        print(f"  {s:12s} {row}")


if __name__ == "__main__":
    main()
