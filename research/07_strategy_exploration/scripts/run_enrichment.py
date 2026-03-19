"""Pre-compute indicator enrichment and save to pickle cache."""
import sys, os, time, logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.strategy_backtest import (
    preload_exploration_data, enrich_with_indicators,
    save_enriched_cache, load_enriched_cache,
)

START = '2015-01-01'
END = '2025-12-31'

# Check if cache already exists
cached = load_enriched_cache(START, END)
if cached is not None:
    print(f"Cache already exists with {len(cached)} stocks. Done.", flush=True)
    sys.exit(0)

t0 = time.time()
print(f'Loading data ({START} to {END})...', flush=True)
universe, price_data = preload_exploration_data(START, END)
print(f'Data loaded: {len(price_data)} stocks in {time.time()-t0:.0f}s', flush=True)

t1 = time.time()
print('Computing indicators for all stocks...', flush=True)
enriched = enrich_with_indicators(price_data)
print(f'Indicators computed in {time.time()-t1:.0f}s', flush=True)

save_enriched_cache(enriched, START, END)
print(f'Total time: {time.time()-t0:.0f}s', flush=True)
