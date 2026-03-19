"""Time full enrichment for all stocks."""
import sys, os, time, logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import preload_exploration_data, enrich_with_indicators

t0 = time.time()
universe, price_data = preload_exploration_data('2015-01-01', '2025-12-31')
print(f'Loaded {len(price_data)} stocks in {time.time()-t0:.0f}s')

t1 = time.time()
enriched = enrich_with_indicators(price_data)
print(f'Enriched {len(enriched)} stocks in {time.time()-t1:.0f}s')
print(f'Total: {time.time()-t0:.0f}s')
