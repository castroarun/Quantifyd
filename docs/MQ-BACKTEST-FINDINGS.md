# Momentum + Quality Strategy - Backtest Findings

## Backtest Period
**January 2, 2023 - December 30, 2025** (744 trading days, ~3 years)

## Performance Comparison

| Metric | MQ Strategy | Nifty 500 | Nifty 50 |
|--------|------------|-----------|----------|
| Initial Capital | 1,00,00,000 | - | - |
| Final Value | 3,40,84,773 | - | - |
| **Total Return** | **240.8%** | 52.5% | 42.5% |
| **CAGR** | **50.5%** | 15.1% | 12.6% |
| Sharpe Ratio | 0.87 | 1.18 | 1.07 |
| Sortino Ratio | 5.76 | - | 1.47 |
| **Max Drawdown** | **-7.9%** | -18.8% | -15.8% |
| Calmar Ratio | 6.39 | - | 0.80 |

### Alpha Generated
- **vs Nifty 500:** +188.3% total return outperformance (+35.4% CAGR alpha)
- **vs Nifty 50:** +198.3% total return outperformance (+37.9% CAGR alpha)

## Trade Summary
| Metric | Value |
|--------|-------|
| Closed Trades | 19 |
| Win Rate | 68% |
| Avg Winning Trade | +114.9% |
| Avg Losing Trade | -22.4% |
| Positions (current) | 30 |
| Semi-Annual Rebalances | 5 |

## Key Observations

### What Went Right
1. **Drawdown control** - MQ strategy max drawdown was 7.9% vs index drawdown of 18.8%. Hard stops (30%) and ATH drawdown exits (20%) at rebalance protected capital.
2. **Momentum selection** - 53 momentum candidates found on Day 1 (within 10% of 52-week high). Top performers: M&M (+192%), Astrazeneca (+169%), Bharti Airtel (+146%).
3. **Sector diversification** - Sector limits (max 6 stocks, 25% weight) prevented over-concentration. Financial Services hit the 6-stock cap early. Healthcare became the largest sector at 23.2%.
4. **Jan 2025 rebalance** - The biggest rebalance event (13 exits/entries due to ATH drawdown) correctly rotated out of weakening positions.

### What Needs Caution (Biases)

1. **Survivorship Bias (MAJOR)** - We used today's Nifty 500 composition, not the historical list from 2023. Stocks that crashed and were removed from the index are excluded. Stocks that rallied and were added are included. This is the single biggest source of upward bias in the results.

2. **No Fundamental Filter Active** - The spec requires revenue growth, debt/equity, OPM checks. Currently every stock gets a default quality score of 0.5, so screening is pure momentum only. The "Quality" part of "Momentum + Quality" is not yet contributing.

3. **Bull Market Tailwind** - 2023-2025 was a strong Indian equity bull market. Any momentum strategy (buy stocks near highs) will perform well in a trending market. Performance would look very different in a bear/sideways market (e.g., 2018-2020).

4. **Zero Topups Fired** - The consolidation/breakout mechanism (designed to add to winning positions from debt reserve) did not trigger once. The 20% debt fund reserve earned 6.5% p.a. but was otherwise idle. The topup parameters may need tuning.

5. **Look-Ahead in Universe** - Selecting from today's Nifty 500 to trade in 2023 introduces forward-looking bias.

6. **Lower Sharpe Than Index** - Despite 4.6x higher returns, the Sharpe ratio (0.87) is lower than Nifty 500 (1.18). This is because the concentrated 30-stock portfolio has higher daily volatility than the 500-stock index. However, the Sortino ratio (5.76 vs 1.47) shows the downside volatility is well controlled - the excess volatility comes from large upside moves, not drawdowns.

### Pre-2023 Testing
The current data only supports Jan 2023 onwards. To backtest earlier periods (including the 2020 COVID crash for stress testing), historical OHLCV data would need to be downloaded from Kite for all 375 stocks. The original 181 stocks have data back to ~Feb 2022, but the 194 added later start from Jan 2023.

## Configuration Used
```
Universe:        Nifty 500 (375 stocks with data, 94.9% coverage)
Allocation:      80% equity / 20% debt fund reserve
Positions:       30 equal-weight
Momentum:        Within 10% of 52-week high
Rebalance:       Semi-annual (January / July)
Hard Stop:       -30% from entry
ATH Drawdown:    Exit at rebalance if >20% below rolling ATH
Debt Fund Rate:  6.5% p.a.
Transaction Costs: Brokerage 0.03% + STT 0.1% (sell) + GST + Stamp + 0.1% slippage
```

## Next Steps
- **Phase 5: Agent System** - Screening, monitoring, rebalance, backtest, and reporting agents
- Activate fundamental quality filter (revenue growth, debt/equity, OPM)
- Tune consolidation/breakout parameters to trigger topups
- Download historical data for pre-2023 stress testing
- Test with historical index composition to remove survivorship bias
