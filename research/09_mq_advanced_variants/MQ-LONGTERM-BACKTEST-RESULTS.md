# MQ Strategy Long-Term Backtest Results

**Date:** 2026-02-20
**Initial Capital:** Rs.1 Crore (10,000,000)
**Universe:** Nifty 500 (~375 symbols with data)
**Base Config:** EQ95, HSL50, ATH20, semi-annual rebalance (Jan/Jul)

---

## V1: No Parking (Baseline)

Idle cash sits uninvested after exits.

| Label | Period | Years | CAGR | Sharpe | Sortino | MaxDD | Calmar | Trades | WR | Final Value |
|-------|--------|-------|------|--------|---------|-------|--------|--------|----|-------------|
| PS30 | 2005-2025 | 21.0 | 19.59% | 0.23 | 2.89 | 66.05% | 0.30 | 53 | 73.6% | Rs.42.8Cr |
| PS30 | 2010-2025 | 16.0 | 17.39% | 0.68 | 0.83 | 28.49% | 0.61 | 123 | 55.3% | Rs.13.0Cr |
| PS30 | 2012-2025 | 14.0 | 15.93% | 0.64 | 0.81 | 25.48% | 0.63 | 45 | 71.1% | Rs.7.9Cr |
| PS30 | 2015-2025 | 11.0 | 18.48% | 0.65 | 0.77 | 28.77% | 0.64 | 108 | 55.6% | Rs.6.5Cr |
| **PS20** | **2005-2025** | **21.0** | **22.42%** | **0.23** | **4.22** | **66.68%** | **0.34** | **45** | **71.1%** | **Rs.70.0Cr** |
| PS20 | 2010-2025 | 16.0 | 17.62% | 0.66 | 0.84 | 29.84% | 0.59 | 94 | 55.3% | Rs.13.4Cr |
| PS20 | 2012-2025 | 14.0 | 19.06% | 0.78 | 0.99 | 28.08% | 0.68 | 44 | 72.7% | Rs.11.5Cr |
| PS20 | 2015-2025 | 11.0 | 18.75% | 0.65 | 0.78 | 30.27% | 0.62 | 77 | 55.8% | Rs.6.6Cr |

**Key Finding:** PS20 consistently beats PS30 across all periods. 2005-2025 results have ~66% MaxDD due to 2008 GFC.

---

## V2: NiftyBEES + Debt Parking (All-at-Once)

Idle cash deployed to NiftyBEES when Nifty50 below 200 SMA, else debt fund. Full deployment immediately.

| Label | Period | Years | CAGR | Sharpe | Sortino | MaxDD | Calmar | Trades | WR | Final Value |
|-------|--------|-------|------|--------|---------|-------|--------|--------|----|-------------|
| PS30 | 2005-2025 | 21.0 | 29.37% | 0.28 | 2.76 | 75.94% | 0.39 | 53 | 73.6% | Rs.222.9Cr |
| PS30 | 2010-2025 | 16.0 | 18.58% | 0.72 | 0.89 | 31.95% | 0.58 | 123 | 55.3% | Rs.15.3Cr |
| PS30 | 2012-2025 | 14.0 | 18.68% | 0.73 | 0.92 | 31.94% | 0.58 | 45 | 71.1% | Rs.11.0Cr |
| PS30 | 2015-2025 | 11.0 | 19.30% | 0.69 | 0.81 | 29.61% | 0.65 | 108 | 55.6% | Rs.7.0Cr |
| PS20 | 2005-2025 | 21.0 | 32.49% | 0.27 | 3.85 | 76.24% | 0.43 | 45 | 71.1% | Rs.367.8Cr |
| PS20 | 2010-2025 | 16.0 | 18.35% | 0.69 | 0.88 | 31.64% | 0.58 | 94 | 55.3% | Rs.14.8Cr |
| PS20 | 2012-2025 | 14.0 | 20.54% | 0.82 | 1.04 | 32.32% | 0.64 | 44 | 72.7% | Rs.13.7Cr |
| PS20 | 2015-2025 | 11.0 | 19.19% | 0.67 | 0.80 | 30.40% | 0.63 | 77 | 55.8% | Rs.6.9Cr |

**Key Finding:** Higher CAGR but MaxDD shot up to 76% (from 67%). Buying NiftyBEES when below 200 SMA = buying into a falling market during 2008 GFC.

---

## V3: NiftyBEES + Debt Parking (25% Tranche, Weekly Cooldown)

Same as V2 but deploy only 25% of idle cash at a time, with 1-week cooldown between deployments.

| Label | Period | Years | CAGR | Sharpe | Sortino | MaxDD | Calmar | Trades | WR | Final Value |
|-------|--------|-------|------|--------|---------|-------|--------|--------|----|-------------|
| PS30 | 2005-2025 | 21.0 | 29.06% | 0.28 | 2.77 | 77.16% | 0.38 | 53 | 73.6% | Rs.211.9Cr |
| PS30 | 2010-2025 | 16.0 | 18.49% | 0.72 | 0.89 | 31.83% | 0.58 | 123 | 55.3% | Rs.15.1Cr |
| PS30 | 2012-2025 | 14.0 | 18.51% | 0.73 | 0.92 | 31.85% | 0.58 | 45 | 71.1% | Rs.10.8Cr |
| PS30 | 2015-2025 | 11.0 | 19.26% | 0.69 | 0.82 | 29.20% | 0.66 | 108 | 55.6% | Rs.6.9Cr |
| PS20 | 2005-2025 | 21.0 | 32.16% | 0.26 | 3.86 | 77.48% | 0.42 | 45 | 71.1% | Rs.349.1Cr |
| PS20 | 2010-2025 | 16.0 | 18.30% | 0.69 | 0.87 | 31.53% | 0.58 | 94 | 55.3% | Rs.14.7Cr |
| PS20 | 2012-2025 | 14.0 | 20.44% | 0.82 | 1.03 | 32.27% | 0.63 | 44 | 72.7% | Rs.13.5Cr |
| PS20 | 2015-2025 | 11.0 | 19.15% | 0.67 | 0.80 | 30.31% | 0.63 | 77 | 55.8% | Rs.6.9Cr |

**Key Finding:** Nearly identical to V2. Tranche deployment didn't help - the core issue was direction (buying into crashes), not deployment speed.

---

## V4: Inverted - NiftyBEES Above 200 SMA + 2-Day Confirm (CHOSEN)

**Inverted logic:** NiftyBEES when Nifty50 is ABOVE 200 SMA (trend-following), debt fund when BELOW. 2-day consecutive confirmation before switching.

| Label | Period | Years | CAGR | Sharpe | Sortino | MaxDD | Calmar | Trades | WR | Final Value |
|-------|--------|-------|------|--------|---------|-------|--------|--------|----|-------------|
| PS30 | 2005-2025 | 21.0 | 25.84% | 0.27 | 2.46 | 65.75% | 0.39 | 53 | 73.6% | Rs.124.7Cr |
| PS30 | 2010-2025 | 16.0 | 17.67% | 0.70 | 0.87 | 27.47% | 0.64 | 123 | 55.3% | Rs.13.5Cr |
| PS30 | 2012-2025 | 14.0 | 16.91% | 0.73 | 0.93 | 22.70% | 0.74 | 45 | 71.1% | Rs.8.9Cr |
| PS30 | 2015-2025 | 11.0 | 18.76% | 0.67 | 0.79 | 28.88% | 0.65 | 108 | 55.6% | Rs.6.6Cr |
| **PS20** | **2005-2025** | **21.0** | **28.80%** | **0.25** | **3.54** | **66.56%** | **0.43** | **45** | **71.1%** | **Rs.203.1Cr** |
| PS20 | 2010-2025 | 16.0 | 17.89% | 0.68 | 0.87 | 29.13% | 0.61 | 94 | 55.3% | Rs.13.9Cr |
| PS20 | 2012-2025 | 14.0 | 19.56% | 0.83 | 1.06 | 26.23% | 0.75 | 44 | 72.7% | Rs.12.2Cr |
| PS20 | 2015-2025 | 11.0 | 18.89% | 0.66 | 0.79 | 30.27% | 0.62 | 77 | 55.8% | Rs.6.7Cr |

**CHOSEN CONFIG: PS20_ABOVE2D_2005_2025 = 28.80% CAGR**

**Key Findings:**
- Inverted logic (trend-following for parking) keeps MaxDD in check while boosting CAGR
- PS20 2012-2025 has the best risk-adjusted metrics: 19.56% CAGR, 26.23% MaxDD, Calmar 0.75
- PS20 2005-2025 (28.80% CAGR) chosen as the official MQ Core number for the tactical capital pool

---

## Summary: Best Configs Across All Versions

| Version | Best Config | CAGR | MaxDD | Calmar | Notes |
|---------|-------------|------|-------|--------|-------|
| V1 (No parking) | PS20 2005-2025 | 22.42% | 66.68% | 0.34 | Baseline |
| V2 (Below SMA parking) | PS20 2005-2025 | 32.49% | 76.24% | 0.43 | Higher CAGR but terrible MaxDD |
| V3 (25% tranche weekly) | PS20 2005-2025 | 32.16% | 77.48% | 0.42 | Same as V2 |
| **V4 (Above SMA + 2d confirm)** | **PS20 2005-2025** | **28.80%** | **66.56%** | **0.43** | **Best balance** |

**Why V4 wins:** Same Calmar as V2/V3 but with 10% lower MaxDD. The trend-following parking approach avoids buying NiftyBEES during market crashes, preserving capital when it matters most.

---

## Config Parameters (Chosen: PS20 + Above SMA Parking)

```python
MQBacktestConfig(
    start_date='2005-01-01',
    end_date='2025-12-31',
    initial_capital=10_000_000,
    portfolio_size=20,
    equity_allocation_pct=0.95,
    hard_stop_loss=0.50,
    rebalance_ath_drawdown=0.20,
    idle_cash_to_nifty_etf=True,
    idle_cash_to_debt=True,
    nifty_etf_above_sma=True,        # INVERTED: ETF when above SMA
    nifty_sma_confirm_days=2,         # 2-day confirmation
)
```
