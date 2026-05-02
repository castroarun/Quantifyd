"""Aggregator: produce 79-stock-universe daily counts + summary.

Reads from per_stock_freq.csv and rebuilds the per-day setup counts
restricted to the 79-stock intraday universe. Writes:

    results/daily_count_79.csv      -- (date, variant, n_qualifying_stocks)
    results/variant_summary_79.csv  -- per-variant distribution stats
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESEARCH_ROOT = ROOT.parent
PROJECT_ROOT = RESEARCH_ROOT.parent

sys.path.insert(0, str(HERE))
from run_daily_freq import (  # noqa: E402
    INTRADAY_79, VARIANTS, daily_setup_vec, load_daily_for, scan_symbol,
)

DB_PATH = PROJECT_ROOT / "backtest_data" / "market_data.db"
RESULTS = ROOT / "results"
DAILY_79 = RESULTS / "daily_count_79.csv"
SUMMARY_79 = RESULTS / "variant_summary_79.csv"
RECENT = pd.Timestamp("2024-01-01")


def main() -> None:
    con = sqlite3.connect(str(DB_PATH))
    # day_count[tag][period][date] = count
    day_count = {v["tag"]: {"full": {}, "recent": {}} for v in VARIANTS}
    n79 = 0
    for sym in INTRADAY_79:
        daily = load_daily_for(con, sym)
        if daily.empty:
            continue
        setup = daily_setup_vec(daily)
        if setup.empty:
            continue
        n79 += 1
        masks = scan_symbol(setup)
        for tag, m in masks.items():
            for d in m.index[m]:
                day_count[tag]["full"][d] = day_count[tag]["full"].get(d, 0) + 1
                if d >= RECENT:
                    day_count[tag]["recent"][d] = day_count[tag]["recent"].get(d, 0) + 1
    con.close()
    print(f"Symbols loaded from 79-universe: {n79}")

    # Write daily_count_79.csv
    rows = []
    for tag, periods in day_count.items():
        for period, dmap in periods.items():
            for d, n in sorted(dmap.items()):
                rows.append({"variant": tag, "period": period,
                             "date": d.strftime("%Y-%m-%d"),
                             "n_qualifying_stocks": n})
    pd.DataFrame(rows).to_csv(DAILY_79, index=False)
    print(f"Wrote {DAILY_79}")

    # Compute per-variant distribution INCLUDING zero-days (use union of dates)
    all_dates_full = set()
    all_dates_recent = set()
    for tag in day_count:
        all_dates_full |= set(day_count[tag]["full"].keys())
        all_dates_recent |= set(day_count[tag]["recent"].keys())
    all_dates_full = sorted(all_dates_full)
    all_dates_recent = sorted(all_dates_recent)

    rows = []
    for v in VARIANTS:
        tag = v["tag"]
        for period_name, dates in (("full", all_dates_full), ("recent", all_dates_recent)):
            dmap = day_count[tag][period_name]
            arr = np.array([dmap.get(d, 0) for d in dates], dtype=float)
            if arr.size == 0:
                continue
            rows.append({
                "variant": tag,
                "today_narrow": v["today_narrow"],
                "yesterday_ctx": v["yesterday_ctx"],
                "wide_thresh": v["wide_thresh"],
                "narrow_thresh": v["narrow_thresh"],
                "period": period_name,
                "n_sessions": int(len(arr)),
                "mean_per_day": float(arr.mean()),
                "median_per_day": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
                "max_per_day": float(arr.max()),
                "days_ge_3": int((arr >= 3).sum()),
                "days_ge_5": int((arr >= 5).sum()),
                "total_setup_events": float(arr.sum()),
            })
    pd.DataFrame(rows).to_csv(SUMMARY_79, index=False)
    print(f"Wrote {SUMMARY_79}")


if __name__ == "__main__":
    main()
