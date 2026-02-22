# Optimization Pickup Guide

> Created: 2026-02-15 | Updated: 2026-02-16
> Best PS30: **ADX_min25 = 34.93% CAGR** (Sharpe 1.10, Calmar 1.40) | Best Sortino: Daily ATH 15% + Replace (27.08% CAGR, Sortino 1.30)
> CC Overlay: **ADX_LOW_20 + 10% OTM = +1.39% CAGR boost** (33.58%, Sharpe 1.11, +Rs.2.46L/yr income)
> Best concentrated: PS10_SEC70_POS30_TOP30_BIM at 48.66% CAGR

---

## MQ Strategy Overview

**Universe:** Nifty 500 (~375 symbols with data) | **Period:** 2023-01-01 to 2025-12-31 | **Capital:** Rs.1 Crore

### Entry Rules
1. **Momentum screen**: Close within 10% of 52-week high
2. **Ranking**: Sort by proximity to ATH + quality score
3. **Portfolio construction**: Top 30 stocks, equal weight (~Rs.3.17L each)
4. **Sector limits**: Max 6 stocks/sector, max 25% sector weight, max 10% single stock

### Exit Rules
- **ATH drawdown (20%)**: If stock drops 20% from its peak since entry → EXIT
  - Default: Checked only at semi-annual rebalance (Jan/Jul)
  - `daily_ath_drawdown_exit=True` enables daily check (see Sweep 5 results below)
  - `immediate_replacement=True` screens for and enters a replacement stock immediately after exit (see Sweep 5B)
- **Hard stop (50%)**: If stock drops 50% from entry price → EXIT (never fires in practice)

### Rebalance (Semi-Annual: Jan & Jul)
1. Check all positions for ATH drawdown → prune losers
2. Re-screen for new momentum candidates → fill empty slots

### Darvas Topups (every 5 trading days)
- Detect consolidation breakouts on held positions (Darvas Box + 1.5x volume)
- Top up 20% of original position size from debt reserve (Rs.20L pool)
- **This is the #1 CAGR driver** — 33 topups fire in baseline, adding ~6% CAGR

### Capital Allocation (current config)
- **Equity**: Rs.95L (equity_allocation_pct=0.95) → 30 stocks
- **Debt reserve**: Rs.20L (debt_reserve_pct=0.20 default) → funds topups @ 6.5% p.a.
- **Note**: Total = Rs.1.15Cr (115% of initial capital — the two params don't enforce summing to 100%)

---

## What's Already Done (Don't Re-test)

| Config | CAGR | Sharpe | MaxDD | Notes |
|--------|------|--------|-------|-------|
| PS30_HSL50_ATH20_EQ95 | 32.19% | 1.05 | 27.0% | **Baseline PS30 winner** (6-month ATH check) |
| PS30_HSL60_ATH20_EQ95 | 32.19% | 1.05 | 27.0% | Same as HSL50 (stop never triggers) |
| PS30_HSL100_ATH20_EQ95 | 32.19% | 1.05 | 27.0% | Confirms: HSL>50 has no effect |
| PS30_TOPUP30_CD3_HSL50 | 32.24% | 1.05 | 27.07% | Topups + cooldown 3d |
| PS30_TOPUP30_HSL70 | 32.24% | 1.05 | 27.07% | Same result as above |
| PS30_TOPUP30_BIM_HSL50 | 31.18% | 1.02 | 25.04% | Bi-monthly, lower DD |
| PS25_HSL50 | 34.48% | 1.10 | 25.81% | |
| PS20_HSL50 | 37.84% | 1.16 | 26.85% | |
| PS15_HSL50 | 38.45% | 1.16 | 28.07% | |
| PS35_HSL50 | 30.48% | 1.00 | 26.27% | |
| PS10_SEC70_POS30_TOP30_BIM | 48.66% | 1.30 | 26.35% | **Best concentrated** |
| Quality weight variations (PS30) | 32.19% | - | - | ALL identical to baseline |
| Fundamental filter variations (PS30) | 32.19% | - | - | ALL identical to baseline |

**Key findings:**
- HSL >= 50% all produce same result (hard stop never fires — ATH drawdown catches losers first)
- Quality weights & fundamental filters have ZERO effect on PS30
- Concentration (fewer stocks) is the #1 CAGR lever
- ATH drawdown exit is the only exit mechanism that actually fires
- Capital allocation: equity=0.95 + debt=0.20 = 115% (implicit 15% leverage)

---

## Sweep 5: Daily ATH Drawdown — COMPLETED (2026-02-16)

**Why this matters:** The baseline only checks ATH drawdown at semi-annual rebalance (Jan/Jul). A stock that drops 20% from peak in February won't be sold until July — up to 5 months of holding a deteriorating position. Enabling `daily_ath_drawdown_exit=True` checks every trading day.

**Config:** PS30, EQ95, HSL50, daily_ath_drawdown_exit=True, varying rebalance_ath_drawdown

### Results

| Config | CAGR | Sharpe | MaxDD | Calmar | WR | Trades | Topups |
|--------|------|--------|-------|--------|-----|--------|--------|
| **6-month ATH 20% (old baseline)** | **32.19%** | **1.05** | 27.0% | 1.19 | 81% | 21 | 33 |
| Daily ATH 15% | 18.60% | 0.83 | **5.50%** | **3.38** | 55% | 29 | 33 |
| Daily ATH 20% | 22.18% | 0.89 | **7.79%** | **2.85** | 71% | 28 | 33 |
| **Daily ATH 25%** | **25.77%** | **0.90** | **13.91%** | **1.85** | **85%** | 26 | 33 |
| Daily ATH 30% | 25.88% | 0.84 | 20.34% | 1.27 | 95% | 20 | 33 |

### Analysis

**Daily ATH 25% is the recommended live-trading config:**
- Gives up ~6% CAGR (32% → 26%) for cutting MaxDD in half (27% → 14%)
- Calmar jumps from 1.19 to 1.85 — much better risk-adjusted returns
- 85% win rate (highest among daily configs)
- Same 33 topups — Darvas breakout engine still fires equally

**Why 6-month check inflates CAGR:**
- Stocks that dip 20-25% from peak and bounce back within the 6-month window are never sold
- This is survivorship/hindsight bias — in live trading you can't know which dips will recover
- Daily check is the disciplined, tradeable approach

**Trade-off spectrum:**
- **Tighter (15%)**: Ultra-safe (5.5% MaxDD) but exits too many winners early (55% WR)
- **Sweet spot (25%)**: Good balance — 26% CAGR, 14% MaxDD, 85% WR
- **Looser (30%)**: Converges back toward 6-month behavior (20% MaxDD, same CAGR as 25%)

**Key gap identified:** When daily ATH exits fire, freed capital sits idle until the next semi-annual rebalance. This drags down CAGR significantly → see Sweep 5B below.

---

## Sweep 5B: Daily ATH + Immediate Replacement — COMPLETED (2026-02-16)

**Why this matters:** In Sweep 5, when daily ATH exit fires, the freed capital sits in `equity_cash` doing nothing until the next 6-month rebalance. The user asked: "we should find the next opportunity as per our 10% ATH ranking and redeploy on the new candidate."

**Implementation:** Added `immediate_replacement=True` config flag. After daily ATH exits fire, the engine immediately:
1. Screens the universe for fresh momentum candidates (within 10% of 52-week high)
2. Excludes current holdings AND just-exited stocks (avoid re-entering a falling stock)
3. Enters replacement positions using the freed capital (equal weight per exit)

**Files modified:** `services/mq_portfolio.py` (config flag), `services/mq_backtest_engine.py` (replacement logic)

### Results

| Config | CAGR | Sharpe | Sortino | MaxDD | Calmar | Trades | WR | Topups |
|--------|------|--------|---------|-------|--------|--------|-----|--------|
| **6-month only (baseline)** | **32.19%** | **1.05** | 1.14 | 27.0% | 1.19 | 21 | 81% | 33 |
| ATH15 no replace | 18.60% | 0.83 | 0.98 | 5.50% | 3.38 | 29 | 55% | 33 |
| **ATH15 + replace** | **27.08%** | **1.04** | **1.30** | 20.41% | 1.33 | 135 | 44% | 34 |
| ATH20 no replace | 22.18% | 0.89 | 0.93 | 7.79% | 2.85 | 28 | 71% | 33 |
| **ATH20 + replace** | **25.88%** | 0.90 | 1.04 | 19.19% | 1.35 | 68 | 50% | 33 |
| ATH25 no replace | 25.77% | 0.90 | 0.90 | 13.91% | 1.85 | 26 | 85% | 33 |
| **ATH25 + replace** | **27.08%** | 0.85 | 0.94 | 23.01% | 1.18 | 43 | 61% | 33 |
| ATH30 no replace | 25.88% | 0.84 | 0.86 | 20.34% | 1.27 | 20 | 95% | 33 |
| **ATH30 + replace** | **27.61%** | 0.86 | 0.95 | 26.69% | 1.03 | 25 | 76% | 33 |

### Analysis

**Immediate replacement recovers significant CAGR across all thresholds:**

| Threshold | Without Replace | With Replace | CAGR Boost | Trades Increase |
|-----------|----------------|--------------|------------|-----------------|
| ATH 15% | 18.60% | 27.08% | **+8.48%** | 29 → 135 |
| ATH 20% | 22.18% | 25.88% | **+3.70%** | 28 → 68 |
| ATH 25% | 25.77% | 27.08% | **+1.31%** | 26 → 43 |
| ATH 30% | 25.88% | 27.61% | **+1.73%** | 20 → 25 |

**Key finding:** Replacement helps, but no daily ATH config beats the 6-month baseline (32.19%). The fundamental issue is that daily ATH exits prune winning momentum stocks too early. A stock dropping 15-25% from peak is often just consolidating before the next leg up — the 6-month approach lets it ride.

**Best Sortino: ATH15 + Replace (1.30)** — better downside-adjusted returns than baseline (1.14). This means when considering only negative volatility, ATH15+Replace is actually superior. The cost is more trades (135 vs 21) and more drawdown (20.4% vs 27%).

**Recommended configs by objective:**
- **Max CAGR**: Baseline 6-month only (32.19%)
- **Best risk-adjusted (Calmar)**: ATH15 no replace (3.38 Calmar, but only 18.6% CAGR)
- **Best downside-adjusted (Sortino)**: ATH15 + replace (1.30 Sortino, 27.08% CAGR)
- **Balanced**: ATH20 + replace (25.88% CAGR, 19.2% MaxDD, 1.35 Calmar)

---

## Sweep 6: Covered Call Overlay — COMPLETE (63/63, 2026-02-16)

**Concept:** Sell covered calls on held MQ portfolio stocks when short-term bearish/sideways signals fire. Indian market: options are **cash-settled** — ITM expiry means paying (stock_price - strike) × lot-aligned shares as settlement, stock stays in portfolio.

**Premium model:** Black-Scholes using 20-day historical volatility as IV proxy.
**Lot size:** `round(500000 / stock_price)` per NSE F&O norms (~Rs.5L per lot).
**Constraint:** `cc_shares = (total_shares // lot_size) * lot_size` — must be exact lot multiples.

### WINNING SIGNAL: ADX LOW (Sideways Market) + 10% OTM

**ADX_LOW_20_10OTM: CAGR=33.58% (+1.39% vs baseline), Sharpe=1.11, Net=+Rs.7.4L (+2.46%/yr)**

Sell 10% OTM covered calls ONLY when ADX(14) < 20 (sideways/range-bound market). When momentum stocks enter low-volatility consolidation phases, their monthly options premiums are essentially free money — the stock doesn't trend enough to blow past a 10% OTM strike.

### Key Findings

1. **ADX LOW is the ONLY consistently profitable CC signal** — all 6 ADX variants beat baseline
2. **MACD bearish cross is #2** — all 3 MACD variants beat baseline (+1.35-1.47%/yr)
3. **"Always sell" is net NEGATIVE** — momentum stocks rally too hard, settlements > premiums
4. **EMA/RSI/Stoch/BB/KC/Ichimoku bearish signals all LOSE money** — these "bearish" signals fire on pullbacks that quickly reverse in momentum stocks
5. **Management strategies (roll up/out/defend) all NEGATIVE** — rolling costs more than it saves
6. **Stop loss (2x) with RSI is marginally positive** (+0.29%/yr) — buying back before settlements grow large helps

### Why ADX LOW Works

ADX < 20 identifies periods when the stock is NOT trending. During these consolidation phases:
- Stock stays in a range → CC expires OTM (83% OTM rate at 10% OTM)
- Premium is earned with minimal settlement costs
- When ADX rises (trend resumes), no new CCs are sold → upside is uncapped

This is the textbook "sell CC in sideways markets" approach, and it works perfectly with momentum stocks.

### Top 10 Configs (beat baseline 32.19%)

| Rank | Config | CAGR | Sharpe | CC Sold | OTM% | Net Income | Income%/yr |
|------|--------|------|--------|---------|------|------------|------------|
| 1 | **ADX_LOW_20_10OTM** | **33.58%** | **1.11** | 214 | 83% | +7,36,836 | **+2.46%** |
| 2 | ADX_LOW_20_5OTM | 33.28% | 1.09 | 214 | 68% | +5,78,668 | +1.93% |
| 3 | ADX_LOW_20_2OTM | 33.09% | 1.08 | 214 | 53% | +4,77,600 | +1.59% |
| 4 | ADX_LOW_25_5OTM | 33.05% | 1.07 | 328 | 68% | +4,55,464 | +1.52% |
| 5 | MACD_BEAR_12_26_5OTM | 33.03% | 1.06 | 341 | 65% | +4,41,488 | +1.47% |
| 6 | ADX_LOW_25_10OTM | 33.02% | 1.08 | 328 | 85% | +4,38,053 | +1.46% |
| 7 | MACD_BEAR_12_26_10OTM | 32.97% | 1.07 | 341 | 83% | +4,10,282 | +1.37% |
| 8 | MACD_BEAR_12_26_2OTM | 32.96% | 1.05 | 341 | 54% | +4,04,696 | +1.35% |
| 9 | ADX_LOW_25_2OTM | 32.92% | 1.05 | 328 | 56% | +3,82,623 | +1.28% |
| 10 | EMA_CROSS_5_20_10OTM | 32.37% | 1.04 | 257 | 80% | +93,153 | +0.31% |

### Bottom 5 (worst CC configs)

| Config | CAGR | Net Income | Income%/yr |
|--------|------|------------|------------|
| EMA_CROSS_10_30_2OTM | 30.09% | -10,78,971 | -3.60% |
| ICHIMOKU_BEAR_9_26_5OTM | 30.34% | -9,55,206 | -3.18% |
| ICHIMOKU_BEAR_9_26_2OTM | 30.39% | -9,28,340 | -3.09% |
| EMA_CROSS_10_20_2OTM | 30.38% | -9,31,939 | -3.11% |
| RSI_OB_ROLLOUT_5OTM | 30.26% | -9,93,887 | -3.31% |

### Management Strategy Summary

| Strategy | Best Result | vs Baseline |
|----------|-------------|-------------|
| Roll Up | RSI_OB 31.76% | -0.43% (NEGATIVE) |
| Roll Out | EMA_CROSS 31.56% | -0.63% (NEGATIVE) |
| Stop Loss 2x | RSI_OB 32.36% | +0.17% (marginal) |
| Defend | EMA_CROSS 31.59% | -0.60% (NEGATIVE) |

### Recommended CC Configuration

```python
# ADD TO BASELINE: ADX LOW sideways filter + 10% OTM monthly CC
cc_enabled=True,
cc_signal_type='adx_low',
cc_signal_fast=14,           # ADX period
cc_signal_threshold=20,      # Fire when ADX < 20 (sideways)
cc_strike_otm_pct=0.10,     # 10% OTM strike
cc_expiry_days=30,           # Monthly options
```

**Expected addition:** +1.39% CAGR, +Rs.2.46L/yr per Rs.1Cr capital, Sharpe 1.11 (up from 1.05)

---

## Sweep 1: Exit Rules (was Agent 1)

**Script:** `run_exit_optimization.py`
**Status:** 3 of 192 completed (halted at config 4)
**Output CSV:** `optimization_agent1_exits.csv` (never created - writes only at end)

### What Went Wrong
1. **No data preloading** - `MQBacktestEngine(config)` on line 76 loads data fresh each time (~190s)
2. **CSV writes only at end** - timeout loses all results
3. **Wrong parameter space** - Tests HSL 0.15-0.30, but we now know HSL50 is optimal

### How to Fix & Resume

**Replace the entire script with this approach:**

```python
# CRITICAL: Use preloaded data
universe, price_data = MQBacktestEngine.preload_data(MQBacktestConfig())

# CRITICAL: Updated parameter grid based on what we know
PARAMS = {
    'portfolio_size': [30],
    'equity_allocation_pct': [0.95],
    'hard_stop_loss': [0.50],           # Fixed at 50% (proven optimal)
    'rebalance_ath_drawdown': [0.15, 0.20, 0.25, 0.30],  # Wider range
    'trailing_stop_loss': [True, False],
    'daily_ath_drawdown_exit': [True, False],
}

# CRITICAL: Pass preloaded data to engine
engine = MQBacktestEngine(config,
    preloaded_universe=universe,
    preloaded_price_data=price_data)

# CRITICAL: Write each result immediately after completion
```

**Remaining combos:** 1 x 1 x 1 x 4 x 2 x 2 = 16 configs (~50s each = ~13 min)

---

## Sweep 2: Rebalance + Allocation (was Agent 2)

**Script:** Never created (agent stuck inspecting class interfaces)
**Status:** 0 of 324 completed
**Output CSV:** `optimization_agent2_rebal.csv` (never created)

### What Went Wrong
Agent spent all turns running `help(MQBacktestConfig)` and `help(BacktestResult)` instead of writing the script.

### How to Resume

**Parameter space (reduced from 324 to ~36 useful combos):**

```python
configs = []
for rebal in [[1,7], [1,4,7,10], [1,3,5,7,9,11]]:  # semi, quarterly, bi-monthly
    for ath_dd in [0.15, 0.20, 0.25, 0.30]:
        for eq in [0.90, 0.95]:
            configs.append((f'REBAL{len(rebal)}_ATH{int(ath_dd*100)}_EQ{int(eq*100)}', {
                'portfolio_size': 30,
                'hard_stop_loss': 0.50,
                'equity_allocation_pct': eq,
                'debt_reserve_pct': round(1.0 - eq, 4),
                'rebalance_ath_drawdown': ath_dd,
                'rebalance_months': rebal,
            }))
```

**Remaining:** 3 x 4 x 2 = 24 configs (~50s each = ~20 min)

---

## Sweep 3: Technical Indicators — COMPLETED (2026-02-16)

**Script:** `run_agent3_technical_optimization.py` (rewritten with preloading + incremental CSV)
**Status:** 27/27 COMPLETE
**Output CSV:** `optimization_sweep3_technical.csv`

### Full Results (27/27)

| Config | CAGR | Sharpe | Sortino | MaxDD | Calmar | WR | Trades | Topups |
|--------|------|--------|---------|-------|--------|-----|--------|--------|
| **ADX_min25** | **34.93%** | **1.10** | **1.17** | 24.91% | **1.40** | 72% | 18 | 33 |
| ADX_min20 | 33.31% | 1.05 | 1.13 | 26.66% | 1.25 | 76% | 21 | 33 |
| **BASELINE_NO_TECH** | **32.19%** | **1.05** | 1.14 | 27.0% | 1.19 | 81% | 21 | 33 |
| WEEKLY_EMA20 | 32.19% | 1.05 | 1.14 | 27.0% | 1.19 | 81% | 21 | 33 |
| MACD_12_26_9 | 29.63% | 0.97 | 1.01 | 23.43% | 1.26 | 77% | 13 | 33 |
| STREND_atr7_m3.0 | 27.79% | 1.02 | 1.10 | 16.65% | 1.67 | 69% | 29 | 33 |
| STREND_atr10_m3.0 | 26.14% | 0.91 | 0.97 | 15.75% | 1.66 | 73% | 30 | 33 |
| ADX_min30 | 25.92% | 0.86 | 0.93 | 22.43% | 1.16 | 73% | 15 | 34 |
| STREND_atr14_m3.0 | 24.57% | 0.77 | 0.82 | 19.95% | 1.23 | 77% | 34 | 33 |
| COMBO_MACD_RSI_WEEKLY | 15.98% | 0.35 | 0.39 | 18.31% | 0.87 | 63% | 8 | 33 |
| STREND_atr10_m2.0 | 13.48% | 0.24 | 0.33 | 5.76% | 2.34 | 56% | 32 | 33 |
| STREND_atr14_m2.0 | 12.81% | 0.13 | 0.17 | 5.56% | 2.30 | 61% | 31 | 33 |
| STREND_atr7_m2.0 | 9.52% | -0.60 | -0.72 | 4.19% | 2.27 | 57% | 30 | 33 |
| EMA_20_50_crossover | 9.63% | -0.49 | -0.47 | 4.29% | 2.25 | 23% | 30 | 25 |
| RSI_14_ob85 | 8.36% | -1.09 | -1.05 | 3.87% | 2.16 | 82% | 33 | 27 |
| RSI_14_ob80 | 6.85% | -2.13 | -1.27 | 2.62% | 2.62 | 81% | 31 | 8 |
| RSI_14_ob75 | 6.24% | -2.78 | -1.54 | 2.50% | 2.50 | 68% | 31 | 3 |
| RSI_7_ob75 | 5.32% | -5.17 | -1.72 | 2.44% | 2.18 | 63% | 30 | 0 |
| RSI_7_ob80 | 5.32% | -4.99 | -1.75 | 2.46% | 2.16 | 63% | 30 | 0 |
| RSI_7_ob85 | 5.18% | -3.88 | -1.89 | 3.68% | 1.41 | 55% | 31 | 2 |
| EMA_9_21_exit_price_below | 5.06% | -7.22 | -1.77 | 2.28% | 2.22 | 10% | 30 | 4 |
| EMA_10_30_exit_price_below | 4.83% | -6.13 | -1.88 | 3.07% | 1.57 | 7% | 30 | 4 |
| COMBO_EMA20_50_RSI14 | 4.72% | -6.73 | -2.02 | 3.29% | 1.44 | 7% | 30 | 3 |
| COMBO_EMA9_21_STREND | 4.68% | -6.22 | -2.02 | 3.35% | 1.40 | 9% | 23 | 5 |
| EMA_9_21_exit_crossover | 4.51% | -5.15 | -1.79 | 3.60% | 1.25 | 10% | 30 | 5 |
| EMA_20_50_exit_price_below | 4.24% | -4.93 | -1.74 | 4.09% | 1.04 | 7% | 30 | 7 |
| EMA_10_30_exit_crossover | 3.95% | -4.53 | -1.61 | 4.91% | 0.81 | 3% | 30 | 7 |

### Key Findings

1. **ADX_min25 BEATS baseline: 34.93% CAGR vs 32.19%** — ADX(14) > 25 + DI+ > DI- filters for strong trending stocks. Fewer trades (18 vs 21) but higher quality entries. Sharpe 1.10, Calmar 1.40.
2. **ADX_min20 also beats baseline: 33.31% CAGR** — Less restrictive ADX threshold, similar to baseline but with slightly better risk metrics (Calmar 1.25 vs 1.19)
3. **WEEKLY_EMA20 has zero effect** — Identical to baseline (weekly EMA(20) filter doesn't screen out any stocks)
4. **MACD_12_26_9 is decent: 29.63% CAGR** — Positive Sharpe, but only 13 trades (too restrictive on entries)
5. **SuperTrend m3.0 > m2.0** — m3.0 configs: 25-28% CAGR; m2.0 configs: 10-13% CAGR
6. **STREND_atr7_m3.0 best risk-adjusted: 27.79% CAGR, 16.65% MaxDD** (Calmar 1.67)
7. **EMA and RSI filters are destructive** — Drop CAGR to 3-9% by blocking momentum entries and topups
8. **Topups correlate with CAGR** — Configs that block topups (EMA/RSI) lose ~26% CAGR
9. **Combos are the worst** — Stacking multiple filters is too restrictive (4-16% CAGR)

### Bugs Found & Fixed During This Sweep

1. **debt_reserve_pct=0.05 kills topups** — Must NOT set explicitly; default 0.20 funds Darvas topups (33 vs 8 topups, 32% vs 26% CAGR)
2. **use_technical_filter=True master switch** — Without it, individual indicator flags are ignored by the engine

---

## Sweep 4: Combined MQ+V3 (was Agent 4)

**Script:** `run_combined_optimization.py`
**Status:** 0 of 15 completed (halted loading first config)
**Output CSV:** None

### What Went Wrong
Combined engine (MQ + V3 overlay) is heavier than MQ alone - needs V3 indicator computation on top. One config takes 100-200s. Script uses subprocess to call combined engine which adds overhead.

### How to Resume

This sweep depends on `services/combined_mq_v3_engine.py` being complete and working. Lower priority than Sweeps 1-3 since it's an overlay system.

**Configs:** 3 MQ bases x 5 V3 configs = 15 (but each takes ~150s = ~38 min)

---

## Agent Instructions Template

When spawning optimization agents, use these instructions to prevent the same failures:

```
### MANDATORY RULES FOR OPTIMIZATION AGENTS

1. **ALWAYS preload data first:**
   ```python
   from services.mq_backtest_engine import MQBacktestEngine
   from services.mq_portfolio import MQBacktestConfig
   universe, price_data = MQBacktestEngine.preload_data(MQBacktestConfig())
   ```
   Then pass to each engine:
   ```python
   engine = MQBacktestEngine(config,
       preloaded_universe=universe,
       preloaded_price_data=price_data)
   ```
   This cuts per-config time from ~190s to ~50s.

2. **ALWAYS write results incrementally (after each config):**
   ```python
   import csv
   FIELDNAMES = ['label','cagr','sharpe','sortino','max_drawdown','calmar',
                 'total_trades','win_rate','final_value','total_return_pct','topups']

   # Write header once
   with open(OUTPUT_CSV, 'w', newline='') as f:
       csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

   # After each config completes:
   with open(OUTPUT_CSV, 'a', newline='') as f:
       csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
   ```

3. **Use correct base parameters (proven optimal for PS30):**
   ```python
   # NOTE: Do NOT set debt_reserve_pct explicitly — default 0.20 is correct
   # (it funds Darvas topups which drive CAGR from 26% to 32%)
   base = dict(
       portfolio_size=30,
       equity_allocation_pct=0.95,
       hard_stop_loss=0.50,          # NOT 0.20 or 0.30
       rebalance_ath_drawdown=0.20,  # NOT 0.10 or 0.15
   )
   ```

4. **Limit batch size to fit in timeout:**
   - Each PS30 config takes ~50-60s with preloading
   - Max 8 configs per 600s bash timeout (including data load time ~30s)
   - For more configs, run multiple sequential bash calls of 8 each

5. **DO NOT waste turns on setup:**
   - Do NOT run `help()` on classes - read the script files directly
   - Do NOT test individual imports - just write and run the script
   - Write the complete script using the Write tool, then run it

6. **Print progress on every config (not every 10th):**
   ```python
   print(f'[{i}/{total}] {label} ...', end='', flush=True)
   ```

7. **Handle the CSV output path:**
   ```python
   OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'optimization_SWEEPNAME.csv')
   ```

8. **Skip already-completed configs:**
   If resuming, read existing CSV and skip labels already present:
   ```python
   import os
   done = set()
   if os.path.exists(OUTPUT_CSV):
       with open(OUTPUT_CSV) as f:
           reader = csv.DictReader(f)
           done = {row['label'] for row in reader}
       print(f'Skipping {len(done)} already-completed configs')
   ```
```

---

## Recommended Pickup Order

| Priority | Sweep | Configs | Est. Time | Why |
|----------|-------|---------|-----------|-----|
| 1 | **Technical Indicators** (Sweep 3) | 27 | ~23 min | Untested dimension, may unlock CAGR beyond 32% for PS30 |
| 2 | **Exit Rules** (Sweep 1) | 16 | ~13 min | ATH drawdown % range 15-30% is untested with HSL50 |
| 3 | **Rebalance Frequency** (Sweep 2) | 24 | ~20 min | Bi-monthly showed promise (lower DD), worth exploring |
| 4 | **Combined MQ+V3** (Sweep 4) | 15 | ~38 min | Overlay system, separate concern |

**Total estimated:** ~94 min for all 82 remaining configs.

**Execution strategy:** Run in 2-3 bash calls per sweep (8 configs each, ~8 min per call). Each call writes CSV incrementally so progress is never lost.

---

## Quick Resume Command

To resume any sweep, tell the agent:

> "Run optimization Sweep [N] from docs/OPTIMIZATION-PICKUP.md. Follow the MANDATORY RULES
> in that doc exactly. Use preloaded data, incremental CSV writes, HSL50 base config,
> and max 8 configs per bash call. Read the doc first."
