"""Re-enrich for full 20-year period (2005-2025)."""
import sys, os, time, logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, enrich_with_indicators,
    save_enriched_cache,
)

START, END = '2005-01-01', '2025-12-31'

print(f'Loading data ({START} to {END})...', flush=True)
t0 = time.time()
universe, price_data = preload_exploration_data(START, END)
print(f'Loaded {len(price_data)} stocks with data ({sum(len(df) for df in price_data.values()):,} candles)', flush=True)
print(f'Data loaded: {len(price_data)} stocks in {time.time()-t0:.0f}s', flush=True)

print('Computing indicators for all stocks...', flush=True)
t1 = time.time()
enriched = enrich_with_indicators(price_data)
print(f'Enrichment complete: {len(enriched)} stocks', flush=True)
print(f'Indicators computed in {time.time()-t1:.0f}s', flush=True)

save_enriched_cache(enriched, START, END)
print(f'Total time: {time.time()-t0:.0f}s', flush=True)
