# Covered_Calls - Glossary

Domain terminology and definitions for the Covered Calls backtesting project.

---

## Options Trading Terms

| Term | Definition |
|------|------------|
| **Covered Call** | Options strategy where you own stock and sell call options against it |
| **Call Option** | Contract giving buyer the right to purchase stock at strike price |
| **Strike Price** | The price at which the option can be exercised |
| **Premium** | The price received for selling an option |
| **DTE (Days to Expiration)** | Number of days until option expires |
| **ATM (At The Money)** | Option with strike price equal to stock price |
| **OTM (Out of The Money)** | Call with strike above current stock price |
| **ITM (In The Money)** | Call with strike below current stock price |
| **Assignment** | When option buyer exercises, seller must deliver shares |
| **Roll** | Close current option position and open a new one |

---

## Greeks

| Greek | Definition |
|-------|------------|
| **Delta** | Rate of change of option price vs stock price (0-1 for calls) |
| **Gamma** | Rate of change of delta |
| **Theta** | Time decay - how much value option loses per day |
| **Vega** | Sensitivity to implied volatility changes |
| **IV (Implied Volatility)** | Market's expectation of future volatility |

---

## Performance Metrics

| Metric | Definition |
|--------|------------|
| **Total Return** | Overall percentage gain/loss |
| **CAGR** | Compound Annual Growth Rate |
| **Sharpe Ratio** | Risk-adjusted return (return / volatility) |
| **Sortino Ratio** | Like Sharpe but only considers downside volatility |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | Percentage of profitable trades |
| **Premium Yield** | Annual premium collected / capital deployed |

---

## Backtest Terms

| Term | Definition |
|------|------------|
| **Walk-Forward** | Testing on unseen future data after parameter optimization |
| **Slippage** | Difference between expected and actual execution price |
| **Commission** | Trading fees paid to broker |
| **Equity Curve** | Graph of portfolio value over time |
| **Benchmark** | Comparison index (typically buy-and-hold) |

---

## Data Terms

| Term | Definition |
|------|------------|
| **OHLC** | Open, High, Low, Close - standard price bar data |
| **Adjusted Close** | Close price adjusted for splits and dividends |
| **Bid/Ask Spread** | Difference between buy and sell prices |
| **Open Interest** | Number of outstanding option contracts |
| **Volume** | Number of contracts traded |

---

## Strategy Variations

| Strategy | Description |
|----------|-------------|
| **Buy-Write** | Buy stock and sell call simultaneously |
| **Overwrite** | Sell calls against existing stock position |
| **Wheel** | Covered calls + cash-secured puts cycle |
| **Poor Man's Covered Call** | Deep ITM LEAPS instead of stock + short call |

---

**Last Updated:** 2025-12-30
