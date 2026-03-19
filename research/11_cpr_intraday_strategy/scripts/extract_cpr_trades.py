#!/usr/bin/env python3
"""Extract individual trades from CPR0.5 + ST7_M4.0 config."""

import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
logging.disable(logging.WARNING)

from services.cpr_intraday_engine import CPRIntradayEngine, CPRIntradayConfig

config = CPRIntradayConfig(
    start_date='2024-01-01',
    end_date='2025-10-27',
    narrow_cpr_threshold=0.5,
    cpr_proximity_pct=3.0,
    max_wick_pct=30.0,
    st_period=7,
    st_multiplier=4.0,
)

print("Loading data...")
daily_data, five_min_data = CPRIntradayEngine.preload_data(
    symbols=config.symbols,
    start_date=config.start_date,
    end_date=config.end_date,
)

print("Running backtest...")
engine = CPRIntradayEngine(config, preloaded_daily=daily_data, preloaded_5min=five_min_data)
result = engine.run()

print(f"\n{'='*120}")
print(f"CPR0.5 + ST7_M4.0 | Trades: {result.total_trades} | WR: {result.win_rate:.1f}% | "
      f"PnL: {result.total_pnl:,.0f} ({result.total_pnl_pct:.2f}%) | PF: {result.profit_factor:.2f}")
print(f"{'='*120}")

print(f"\n{'#':>3} {'Symbol':12} {'Dir':6} {'Entry Date':12} {'Entry Time':10} "
      f"{'Entry Price':>12} {'Exit Date':12} {'Exit Time':10} {'Exit Price':>12} "
      f"{'Exit Reason':16} {'PnL':>10} {'PnL%':>8}")
print("-" * 140)

for i, t in enumerate(sorted(result.trades, key=lambda x: x.entry_date), 1):
    entry_dt = t.entry_date
    exit_dt = t.exit_date
    print(f"{i:3d} {t.symbol:12} {t.direction:6} {entry_dt.strftime('%Y-%m-%d'):12} "
          f"{entry_dt.strftime('%H:%M'):10} {t.entry_price:12.2f} "
          f"{exit_dt.strftime('%Y-%m-%d'):12} {exit_dt.strftime('%H:%M'):10} "
          f"{t.exit_price:12.2f} {t.exit_reason:16} {t.pnl:10,.0f} {t.pnl_pct:7.2f}%")

print(f"\n--- Exit Reason Breakdown ---")
for reason, count in sorted(result.exit_reason_counts.items(), key=lambda x: -x[1]):
    print(f"  {reason}: {count}")

print(f"\n--- Per-Symbol Stats ---")
for sym, stats in sorted(result.per_symbol_stats.items(), key=lambda x: -x[1].get('total_pnl', 0)):
    if stats.get('total_trades', 0) > 0:
        print(f"  {sym:12} | Trades: {stats['total_trades']:2d} | PnL: {stats.get('total_pnl', 0):>10,.0f}")
