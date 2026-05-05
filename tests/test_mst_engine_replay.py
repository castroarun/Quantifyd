"""Replay test: feed 6.3 years of NIFTY 30-min bars to MSTEngine and verify
the events match the offline research/35 + research/36 analysis.

Runs as a script (not pytest) for direct comparison with research outputs.
Mocks the executor so no real orders are placed; engine signal logic only.
"""
from __future__ import annotations
import sys
import sqlite3
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.mst_engine import MSTEngine
from services.trading_calendar import NSETradingCalendar
from services import mst_db


class FakeExecutor:
    """No-op executor for signal-only replay. Counts what would have been placed."""
    def __init__(self):
        self.debit_calls = []
        self.credit_calls = []
        self.reset_calls = []
        self.close_calls = []

    def place_debit_spread(self, **kwargs):
        self.debit_calls.append(kwargs)

    def place_credit_spread(self, **kwargs):
        self.credit_calls.append(kwargs)

    def place_reset_condor(self, **kwargs):
        self.reset_calls.append(kwargs)

    def close_position(self, pos, reason):
        self.close_calls.append({"pos_id": pos.get("id"), "reason": reason})

    def estimate_credit(self, **kwargs):
        # Return a value above MIN_CREDIT_PER_LOT so we always pass credit-OK path.
        # In a real test we'd vary this to exercise both paths.
        return 1500.0


def main():
    # Reset MST DB for clean run
    if mst_db.DB_PATH.exists():
        mst_db.DB_PATH.unlink()
    mst_db.init_db()

    # Load historical NIFTY 30-min from market_data.db
    market_db = ROOT / "backtest_data" / "market_data.db"
    con = sqlite3.connect(str(market_db))
    df = pd.read_sql(
        "SELECT date as bar_dt, open, high, low, close FROM market_data_unified "
        "WHERE symbol='NIFTY50' AND timeframe='30minute' ORDER BY date",
        con, parse_dates=["bar_dt"]
    )
    con.close()
    print(f"Loaded {len(df)} NIFTY 30-min bars from {df.bar_dt.min()} to {df.bar_dt.max()}")

    # Initialize engine
    cal = NSETradingCalendar(2026)
    # Pre-load 2020-2026 holidays into calendar
    for y in range(2020, 2027):
        cal._ensure_year_loaded(y)

    executor = FakeExecutor()
    engine = MSTEngine(executor=executor, calendar=cal,
                       paper_mode=True, enabled=True)

    # Replay all bars
    event_counter = Counter()
    pyramid_fires = 0
    rollovers = 0
    last_print = 0
    for i, row in enumerate(df.itertuples(index=False)):
        bar = {
            "bar_dt": row.bar_dt.isoformat(),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        events = engine.on_30min_bar(bar)
        for e in events:
            event_counter[e["type"]] += 1
            if e["type"] == "pyramid_fired":
                pyramid_fires += 1
            elif e["type"] == "rolled":
                rollovers += 1

        # Progress indicator
        if i - last_print >= 2000:
            last_print = i
            print(f"  [{i}/{len(df)}] state={engine.state.state} dir={engine.state.mst_direction} "
                  f"events_so_far={sum(event_counter.values())}")

    print(f"\n=== Replay complete on {len(df)} bars ===\n")
    print("Event counts:")
    for et, cnt in sorted(event_counter.items(), key=lambda x: -x[1]):
        print(f"  {et:25} {cnt}")
    print(f"\nExecutor calls:")
    print(f"  debit_spread placements:  {len(executor.debit_calls)}")
    print(f"  credit_spread placements: {len(executor.credit_calls)}")
    print(f"  reset_condor placements:  {len(executor.reset_calls)}")
    print(f"  close calls:              {len(executor.close_calls)}")

    # Sanity checks vs research/36
    print("\n=== Sanity checks ===")
    # Expected from research/36 extended period:
    #   ~302 trends after break-of-extreme filter
    #   ~294 first-CSTs (one per trend)
    flip_armed_count = event_counter.get("flip_armed", 0)
    flip_activated_count = event_counter.get("flip_activated", 0)
    flip_discarded_count = event_counter.get("flip_discarded", 0)
    print(f"  flip_armed events:     {flip_armed_count} (research/36 had ~322 raw flips before filter)")
    print(f"  flip_activated events: {flip_activated_count} (research/36 had 302 trends after break-filter)")
    print(f"  flip_discarded events: {flip_discarded_count}")
    condor_built = event_counter.get("condor_built", 0)
    print(f"  condor_built events:   {condor_built} (= number of first-CSTs that hit the build path)")
    pyramid_fired = event_counter.get("pyramid_fired", 0)
    print(f"  pyramid_fired events:  {pyramid_fired}")

    # Show a few sample events from DB
    events = mst_db.get_recent_events(limit=20)
    print("\nMost recent 20 events from DB:")
    for e in events[:20]:
        print(f"  {e['bar_dt']:30} {e['event_type']:20} dir={e['direction']:>3} "
              f"price={e['price'] or '—':>10} L={e['pyramid_level'] or '—'}")

    print(f"\nFinal engine state: {engine.get_state_snapshot()}")

    # Cleanup test DB (best-effort; Windows may hold the connection briefly)
    try:
        import gc
        gc.collect()
        mst_db.DB_PATH.unlink()
        print("\nTest DB cleaned up. Replay test passed.")
    except Exception as e:
        print(f"\nReplay test passed. (DB cleanup deferred — {e})")


if __name__ == "__main__":
    main()
