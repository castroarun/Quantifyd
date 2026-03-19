# 01: Covered Call Baseline Strategy

**Period:** December 2025
**Status:** Superseded by MQ Portfolio Strategy

## What Was Tested

Initial project approach — automated covered call strategy optimization on Indian F&O stocks using simulated data.

### Scripts
- `strategy_optimizer.py` — Exhaustive parameter sweep across strike selection methods (ATM, OTM 1-3%), exit strategies, and indicator combinations
- `data_simulator.py` — Synthetic OHLCV data generator for backtesting without market data
- `run_optimization.py` — Runner script for the optimizer

### Approach
- Test all combinations of: strike method (ATM/OTM), exit strategy (hold-to-expiry, early-exit), technical filters
- Uses `CoveredCallEngine` from `services/covered_call_service.py`
- Simulated data (not real market data)

## Findings

- Covered call strategies showed moderate returns but limited upside capture
- The approach was abandoned in favor of the Momentum + Quality (MQ) portfolio strategy which showed significantly higher CAGR
- Simulated data proved inadequate for strategy validation — shifted to real Kite API data

## Next Steps / Recommendations

- **Superseded** — MQ portfolio strategy (folder 02) became the primary focus
- Covered call overlay was later revisited in folder 04 (technical indicator sweep) as `optimization_sweep6_covered_calls.csv`
- The concept of selling covered calls on MQ holdings remains a potential income enhancement but was not pursued further
