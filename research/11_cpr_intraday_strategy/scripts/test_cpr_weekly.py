"""Quick sanity test for the rewritten weekly-bias CPR engine."""
import sys
import os
import logging
import time

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.cpr_intraday_engine import CPRIntradayEngine, CPRIntradayConfig

config = CPRIntradayConfig(
    start_date='2024-06-01',
    end_date='2024-09-30',
    narrow_cpr_threshold=2.0,
    max_wick_pct=25,
    cpr_proximity_pct=1.0,
    st_period=7,
    st_multiplier=3.0,
)

print(f"Config: CPR<{config.narrow_cpr_threshold}% WICK<{config.max_wick_pct}% PROX<{config.cpr_proximity_pct}% ST({config.st_period},{config.st_multiplier})")
print(f"Period: {config.start_date} to {config.end_date}")

t0 = time.time()
engine = CPRIntradayEngine(config)
result = engine.run()
elapsed = time.time() - t0

print(f"\nCompleted in {elapsed:.1f}s")
print(f"Trades: {result.total_trades}, WR: {result.win_rate:.1f}%, PnL: Rs {result.total_pnl:,.0f}, PF: {result.profit_factor:.2f}")
print(f"Exit reasons: {result.exit_reason_counts}")

# Show first few trades
if result.trades:
    print(f"\nFirst 5 trades:")
    for t in result.trades[:5]:
        print(f"  {t.entry_date.strftime('%Y-%m-%d %H:%M')} {t.symbol} {t.direction} "
              f"Entry:{t.entry_price:.1f} Exit:{t.exit_price:.1f} "
              f"PnL:Rs{t.pnl:,.0f} ({t.exit_reason})")
