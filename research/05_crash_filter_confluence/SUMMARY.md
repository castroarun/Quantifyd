# 05: Crash Filter & Confluence Analysis

**Period:** February 14, 2026
**Status:** Crash filter validated for KC6; TTM Squeeze inconclusive

## What Was Tested

Three research tracks to improve strategy robustness:

### A. Crash Filters (for KC6 Mean Reversion)

Tested 6 filter types to block entries during market-wide selloffs:

1. **Signal clustering** — If N+ stocks trigger KC6 entry on same day, skip all
2. **ATR expansion** — If ATR(14) is X% above 50-day average, skip
3. **Consecutive loss circuit breaker** — After N consecutive SL hits, pause trading
4. **Market regime** — Track % of stocks above SMA(200)
5. **KC6 band width** — When bands widen beyond threshold, skip
6. **Drawdown from recent high** — If stock dropped >X% in Y days, skip

Tested across 6 major crash periods: 2008 GFC, 2011 Euro crisis, 2015-16 China crash, 2018 IL&FS, 2020 COVID, 2022 Russia/Ukraine.

### B. Confluence Analysis

Tested whether combining multiple indicator signals (confluence) improves entry quality vs single-indicator entries.

### C. TTM Squeeze Backtest

Tested TTM Squeeze (Bollinger Bands inside Keltner Channels) as a volatility compression breakout signal.

## Key Findings

1. **Universe ATR Ratio >= 1.3x** is the best crash filter — blocks entries when overall market volatility spikes. Now implemented in KC6 live system
2. **Signal clustering** is a useful secondary filter — 5+ simultaneous entries = likely market-wide event, not stock-specific
3. **Confluence** showed marginal improvement — not enough to justify complexity
4. **TTM Squeeze** produced a detailed analysis (see `TTM_Squeeze_Deep_Dive_Report.docx`) but results were inconclusive as a standalone entry signal

## Files

| File | Purpose |
|------|---------|
| `scripts/crash_filter_research.py` | Initial crash filter research (6 filter types) |
| `scripts/crash_filter_v2.py` | Refined filters with better parameter ranges |
| `scripts/crash_filter_v3.py` | Final crash filter with ATR ratio implementation |
| `scripts/confluence_analysis.py` | Multi-signal confluence testing |
| `scripts/confluence_v2.py` | Refined confluence with scoring |
| `scripts/ttm_squeeze_backtest.py` | TTM Squeeze indicator backtest |
| `results/TTM_Squeeze_Deep_Dive_Report.docx` | Detailed TTM analysis report |

## Next Steps / Recommendations

- Crash filter (ATR Ratio >= 1.3x) is **already deployed** in KC6 live trading system
- TTM Squeeze could be revisited as a regime filter rather than an entry signal
- Confluence approach could be tested with the combined swing strategy (folder 12) but likely not worth the complexity
