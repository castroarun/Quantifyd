# 10: Combined MQ + Breakout V3 Overlay

**Period:** February 20, 2026
**Status:** Complete — MQ Core + KC6 Tactical tested over 3-year and 15-year periods

## What Was Tested

Combined the two best standalone strategies into a unified portfolio:
- **MQ Core (60%)** — Momentum + Quality stock selection, semi-annual rebalance
- **KC6 Tactical (40%)** — Keltner Channel 6 mean reversion, daily swing trades

Also tested rebalance frequency optimization and max drawdown analysis.

### 15-Year Backtest (2010-2025)

| Component | Capital | Purpose |
|-----------|---------|---------|
| MQ Core | Rs.60L (60%) | Long-term momentum holdings |
| KC6 Tactical | Rs.40L (40%) | Short-term mean reversion trades |
| NIFTYBEES | Benchmark | Fair comparison |

### Rebalance Optimization

Tested rebalance frequencies: monthly, quarterly, semi-annual, annual, and various month combinations.

## Key Findings

1. **Combined system outperforms either component alone** over 15 years, with smoother equity curve
2. **Correlation between MQ and KC6 is low** (~0.15) — genuine diversification benefit
3. **Semi-annual rebalance (Jan + Jul) remains optimal** — more frequent rebalancing adds transaction costs without improving returns
4. **MQ ATH trailing exit** validated via TradingView Pine script (`mq_ath_trailing.pine`)
5. **Max drawdown of combined system** is lower than MQ alone due to KC6's uncorrelated returns

## Files

| File | Purpose |
|------|---------|
| `scripts/run_combined_optimization.py` | Combined MQ+V3 parameter optimization |
| `scripts/run_combined_15yr_backtest.py` | 15-year combined backtest with NIFTYBEES benchmark |
| `scripts/run_combined_maxdd_backtest.py` | Max drawdown focused analysis |
| `scripts/run_rebal_optimization.py` | Rebalance frequency sweep |
| `scripts/_combined_worker.py` | Worker helper for combined runs |
| `results/combined_equity_curve.csv` | 3-year daily equity curve (MQ + KC6 + combined + Nifty) |
| `results/combined_equity_curve_15yr.csv` | 15-year daily equity curve (3,971 days) |
| `results/mq_ath_trailing.pine` | TradingView Pine script for ATH trailing validation |

## Next Steps / Recommendations

- The 60/40 MQ+KC6 split is the **recommended production allocation**
- KC6 is already built as a live trading system (see `docs/KC6-SESSION-HANDOFF.md`)
- Consider adding ORB (Opening Range Breakout) as a third component if 5-min data is expanded beyond current 10 stocks
