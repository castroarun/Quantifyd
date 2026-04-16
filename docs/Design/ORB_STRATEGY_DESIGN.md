# ORB — Opening Range Breakout (Intraday Cash Equity)

**Status:** Backtest Complete, Live Deployment Pending (Contabo VPS)
**Date:** 2026-04-16
**Instrument:** Cash Equity (MIS orders on Zerodha)
**Universe:** 7 High-Beta F&O Stocks (NSE)
**Timeframe:** 5-minute candles (intraday)

---

## 1. Strategy Overview

Intraday Opening Range Breakout on high-beta F&O stocks. After the first 15 minutes of market open, track the high and low of the opening range. Enter long on a close above OR High, short on a close below OR Low. Filter entries with VWAP, RSI(14) on 15-min timeframe, and CPR direction/width. Exit at OR opposite (stop loss), 1.5x R-multiple (target), or 15:20 EOD squareoff.

**Edge:** High-beta stocks exhibit strong momentum follow-through after establishing a clear opening range. CPR width filter eliminates choppy/ranging days where breakouts fail. The combination of directional filters (VWAP, RSI, CPR) ensures trades align with the intraday trend.

**Universe (7 stocks):**

| Stock | Sector | Why Selected |
|-------|--------|-------------|
| ADANIENT | Infrastructure | Highest PF (3.06), best win rate (64.6%) |
| TATASTEEL | Metals | High trade count (135), reliable PF (1.74) |
| BEL | Defence | Strong PF (1.98), consistent performer |
| VEDL | Metals | Most active (169 trades), decent PF (1.34) |
| BPCL | Oil & Gas | Good win rate (61.8%), moderate frequency |
| M&M | Auto | Broad sector diversification |
| BAJFINANCE | Financial | High liquidity, captures financial sector moves |

**Dropped:** BANKBARODA — consistently unprofitable across all 126 configurations tested.

---

## 2. Entry Rules

### 2.1 Opening Range Construction (5-min candles)

| Parameter | Value | Notes |
|-----------|-------|-------|
| OR Window | 09:15 - 09:30 | First 15 minutes of market open |
| Candles used | 3 bars | Three 5-min candles within the OR window |
| OR High | Highest high of OR bars | Breakout level for longs |
| OR Low | Lowest low of OR bars | Breakout level for shorts |

### 2.2 Breakout Detection

| Rule | Condition | Notes |
|------|-----------|-------|
| **Long entry** | 5-min candle CLOSES above OR High AND previous candle was at/below OR High | Close-based confirmation, not wick |
| **Short entry** | 5-min candle CLOSES below OR Low AND previous candle was at/above OR Low | Close-based confirmation, not wick |
| **Entry price** | Bar close price | No limit orders — enter at candle close |
| **Entry window** | 09:30 - 14:00 | No entries after 14:00 (insufficient time for target) |
| **Max trades/day** | 1 per stock | Prevents overtrading on whipsaw days |

### 2.3 Position Sizing

| Parameter | Value | Notes |
|-----------|-------|-------|
| Capital per stock | ~Rs 14,286 | Equal allocation across 7 stocks from Rs 1L capital |
| Shares per trade | Max affordable within allocation | `floor(14286 / entry_price)` |
| Order type | MIS (intraday margin) | Auto-squared off by Zerodha at 15:15 if not closed |
| Max concurrent | 7 positions | One per stock |

---

## 3. Filters

All filters must pass for an entry signal to be valid. Each filter can be independently toggled in the config.

### 3.1 VWAP Filter

| Direction | Condition |
|-----------|-----------|
| Long | Price > VWAP |
| Short | Price < VWAP |

VWAP is calculated from 09:15 using cumulative (price x volume) / cumulative volume on 5-min bars.

**Finding:** VWAP is technically redundant when CPR direction is active (identical results). Kept for robustness — CPR direction subsumes VWAP in most cases, but VWAP adds a secondary confirmation layer.

### 3.2 RSI Filter (15-min Timeframe)

| Parameter | Value |
|-----------|-------|
| Period | 14 |
| Timeframe | 15-min (resampled from 5-min) |
| Long threshold | RSI > 60 |
| Short threshold | RSI < 40 |

Ensures entries align with higher-timeframe momentum. Prevents counter-trend trades in the dead zone (RSI 40-60).

### 3.3 CPR Direction Filter

| Direction | Condition |
|-----------|-----------|
| Long | Price > TC (Top Central Pivot Range) |
| Short | Price < BC (Bottom Central Pivot Range) |

CPR is calculated from the previous trading day's High, Low, Close:

```
Pivot = (High + Low + Close) / 3
BC (Bottom CPR) = (High + Low) / 2
TC (Top CPR) = (Pivot - BC) + Pivot
```

**Safe fetch pattern:** Previous day data uses a lookback window that handles weekends and holidays (skips non-trading days to find the last valid session).

### 3.4 CPR Width Filter (The #1 Filter)

| Parameter | Value |
|-----------|-------|
| Threshold | 0.5% of price |
| Condition | Skip ENTIRE day if CPR width > threshold |

```
CPR Width = |TC - BC| / Pivot * 100
If CPR Width > 0.5% → No trades for this stock today
```

**This is the single most impactful filter.** Wide CPR days indicate indecision/ranging conditions where breakouts are more likely to fail. Narrow CPR concentrates price action and leads to stronger directional moves.

### 3.5 Gap Direction Filter

On gap-up days (open > prev close by 0.3%+), **disable long breakouts — only allow shorts.**

```
gap_pct = (today_open - prev_close) / prev_close * 100
If gap_pct > 0.3% → block long entries for that stock today
Gap-down days → no restriction (both long and short allowed)
```

**Rationale (validated by data):**

| Scenario | Trades | PF | Verdict |
|----------|--------|-----|---------|
| Gap-up + Long breakout | 149 | 1.01 | Breakeven — stock already exhausted upside |
| Gap-up + Short breakdown | 65 | 1.31 | OK — reversal from stretched open |
| Gap-down + Long breakout | 22 | 3.69 | Excellent — mean reversion |
| Gap-down + Short breakdown | 213 | 2.18 | Strong — momentum cascade |

Applying this filter lifts overall PF from 1.71 to **1.86** (+9%) while removing only low-quality trades.

### 3.6 Filters Tested and Rejected

| Filter | Result | Why Rejected |
|--------|--------|-------------|
| Previous Day H/L Break | Cuts trades, no PF improvement | Removes valid setups without adding edge |
| Virgin CPR | Too rare, reduces sample size | Not enough occurrences to be useful |
| Bollinger Bands | No improvement | Redundant with other directional filters |
| Inside Day / NR4 | Mixed results | Helps some stocks, hurts others |
| Gap size cap (remove big gaps) | PF drops from 1.71 to 1.38 | Big gaps are the MOST profitable trades |

---

## 4. Exit Rules

### 4.1 Stop Loss — OR Opposite

| Direction | Stop Loss Level |
|-----------|----------------|
| Long | OR Low |
| Short | OR High |

The opposite boundary of the opening range serves as the natural invalidation level. If price reverses through the entire OR, the breakout thesis is void.

### 4.2 Target — 1.5x R-Multiple

```
Risk (R) = |Entry Price - Stop Loss|
Target = Entry Price + (1.5 x R) for longs
Target = Entry Price - (1.5 x R) for shorts
```

| Example | Long | Short |
|---------|------|-------|
| Entry | 2500 | 2500 |
| SL (OR opposite) | 2468 | 2532 |
| Risk (R) | 32 pts | 32 pts |
| Target (1.5R) | 2548 | 2452 |

### 4.3 EOD Squareoff

| Parameter | Value |
|-----------|-------|
| Time | 15:20 IST |
| Action | Close at market price |
| Reason | Mandatory — no overnight intraday positions |

### 4.4 Exit Priority

```
1. Stop Loss (OR opposite hit on 5-min bar high/low)
2. Target (1.5R hit on 5-min bar high/low)
3. EOD Squareoff (15:20 IST — close at market)
```

SL and Target are checked on each 5-min bar using bar high and bar low. If both SL and Target could be hit on the same bar, SL takes priority (conservative assumption).

---

## 5. Backtest Results

**Period:** March 2024 - March 2026 (366 trading days)
**Configs tested:** 1,008 backtests across 126 configurations x 8 stocks
**Winning config:** OR15, SL=OR Opposite, Target=1.5R, Filters: VWAP + RSI(14 on 15min) + CPR Direction + CPR Width(0.5%)

### 5.1 Per-Stock Performance

| Stock | Trades | Win Rate | PF | Avg SL% | Avg Target% | Sum PnL% | Max DD% |
|-------|--------|----------|------|---------|-------------|----------|---------|
| ADANIENT | 79 | 64.6% | 3.06 | 1.28% | 1.92% | 45.31% | -2.44% |
| TATASTEEL | 135 | 56.3% | 1.74 | 1.25% | 1.88% | 36.50% | -4.88% |
| BEL | 134 | 57.5% | 1.98 | 1.33% | 1.99% | 38.95% | -5.98% |
| VEDL | 169 | 52.1% | 1.34 | 1.43% | 2.15% | 26.18% | -13.05% |
| BPCL | 89 | 61.8% | 1.47 | 1.59% | 2.38% | 13.38% | -4.14% |
| M&M | 127 | 53.5% | 1.33 | 1.43% | 2.14% | 13.86% | -9.70% |
| BAJFINANCE | 160 | 46.9% | 1.27 | 1.22% | 1.83% | 15.58% | -7.46% |

### 5.2 Combined Portfolio Performance

**Capital:** Rs 1,00,000 | **Allocation:** Equal weight ~Rs 14,286 per stock

| Metric | Value |
|--------|-------|
| Total P&L | +Rs 26,136 (+26.1%) |
| Annualized Return | +18.0% |
| Winning Months | 23/25 (92%) |
| Max Drawdown | -Rs 1,555 (-1.6%) |
| Avg Trades/Day | 2.4 |
| Profit Factor | 1.71 |
| Avg Risk/Trade | Rs 186 (0.2% of capital) |

### 5.3 Exit Breakdown

| Exit Type | Percentage |
|-----------|-----------|
| Target (1.5R) | 18.0% |
| Stop Loss (OR opposite) | 19.1% |
| EOD Squareoff (15:20) | 62.8% |

The majority of trades exit at EOD — neither SL nor target is hit. This is consistent with intraday strategies where most days see modest moves. The edge comes from the asymmetric payoff: targets (1.5R) are larger than stops (1R), and the directional filters push win rate above breakeven.

### 5.4 Direction Analysis

| Direction | Profit Factor | Notes |
|-----------|--------------|-------|
| Shorts | 2.00 | Significantly outperform |
| Longs | 1.37 | Positive but weaker edge |

Short trades are materially stronger. Possible explanations: panic selling creates sharper downside momentum; OR Low breaks tend to be more decisive than OR High breaks in the tested period.

---

## 6. Key Research Findings

### 6.1 OR Window Size

| OR Minutes | Best For | Notes |
|------------|----------|-------|
| 15 min | High-beta stocks (ADANIENT, TATASTEEL, BEL, VEDL, M&M) | Wider range = stronger momentum signal |
| 5 min | Lower-beta stocks (BAJFINANCE, BPCL) | Tighter range gives earlier entries |

**Conclusion:** OR15 is the single config for the portfolio. Per-stock optimization (Phase 5) could use OR5 for select stocks.

### 6.2 Wide OR Days

Counter-intuitive finding: **wide opening range days are MORE profitable**, not less.

| OR Width Quartile | PF |
|-------------------|-----|
| Widest (Q4) | 2.02 |
| Q3 | 1.80 |
| Q2 | 1.65 |
| Narrowest (Q1) | 1.74 |

Wide OR = strong early momentum = higher probability of follow-through. Narrow OR can mean indecision, leading to false breakouts.

### 6.3 Filter Effectiveness Ranking

| Rank | Filter | Impact |
|------|--------|--------|
| 1 | CPR Width (skip wide CPR days) | Single biggest PF improvement |
| 2 | RSI(14) on 15-min | Aligns with higher-TF momentum |
| 3 | CPR Direction (price vs TC/BC) | Confirms daily bias |
| 4 | VWAP | Redundant with CPR direction but kept for robustness |

### 6.4 Previous Day H/L Break Filter

Tested and rejected. Requiring price to break previous day's high/low before entry cuts trades significantly without improving profit factor. The OR breakout itself is sufficient.

---

## 7. Risk Management

### 7.1 Daily Loss Limit

| Parameter | Value |
|-----------|-------|
| Daily loss cap | Rs 3,000 (3% of capital) |
| Action | Stop all new entries for the day |

If cumulative P&L for the day hits -Rs 3,000, no new trades are opened. Existing positions are held to their SL/Target/EOD exits.

### 7.2 Per-Trade Risk

| Parameter | Value |
|-----------|-------|
| Max SL per trade | OR opposite (avg ~1.3% of position) |
| Max loss per trade | ~Rs 186 (avg, based on backtest) |
| Max daily risk (all 7 stocks hit SL) | ~Rs 1,300 |

### 7.3 Correlation Risk

All 7 stocks may trigger the same direction on strong trend days — net exposure can be 100% long or 100% short. This is acceptable for Phase 1 (small capital), but Phase 4 addresses it with index options hedging.

### 7.4 Slippage and Transaction Costs

| Factor | Assumption | Notes |
|--------|-----------|-------|
| Slippage | 0.05% per trade | Not modeled in backtest |
| Brokerage | Rs 20/trade (Zerodha) | Flat fee, negligible at this size |
| STT + charges | ~0.01% | Minimal for intraday equity |
| Expected impact | Real results 5-10% lower than backtest | Conservative estimate |

---

## 8. Capital Deployment

### Phase 1 — Cash Equity

| Parameter | Value |
|-----------|-------|
| Initial Capital | Rs 1,00,000 |
| Allocation | Equal weight across 7 stocks (~Rs 14,286 each) |
| Order type | MIS (intraday) on Zerodha |
| Position sizing | Max shares affordable within allocation |
| Max concurrent positions | 7 (one per stock) |
| Max daily risk | ~Rs 1,300 (7 x Rs 186 avg SL) |

---

## 9. Infrastructure

### 9.1 Existing Components

| Component | File | Status |
|-----------|------|--------|
| Backtest engine | `services/orb_backtest_engine.py` | Built, tested |
| Config dataclass | `ORBConfig` in engine file | Full parameter set |
| Data source | `backtest_data/market_data.db` | 5-min candles available |
| Main app | `app.py` | Flask app, Bootstrap 5 dark theme |

### 9.2 To Build

| Component | Description | Priority |
|-----------|-------------|----------|
| Paper trading module | Simulate live signals using real-time data | Phase 1 |
| Live executor | MIS order placement via Kite API | Phase 2 |
| Dashboard page | ORB signals, positions, daily P&L | Phase 1 |
| Alert system | Telegram/email on entries, exits, daily summary | Phase 2 |

### 9.3 Hosting

| Environment | Where | Purpose |
|-------------|-------|---------|
| Development | Local Windows machine | Backtesting, config optimization |
| Production | Contabo VPS (94.136.185.54) | 24/7 cloud execution for live trading |
| Data feed | Kite Historical API + WebSocket | 5-min candles (historical) + live ticks |

---

## 10. Roadmap

### Phase 1 — Cash Equity Live (Current)

- Build live execution module on Contabo VPS (94.136.185.54)
- Deploy as MIS intraday orders via Kite API — direct live trading, no paper phase
- Start with all 7 stocks simultaneously (Rs 1L capital, ~14.3K per stock)
- TOTP auto-login at 8:55 AM IST, system auto-starts at 9:14 AM
- Monitor fill quality and real slippage vs backtest assumptions
- Daily P&L reconciliation against expected signals

### Phase 3 — Futures Upgrade

Migrate from cash equity to stock futures for:

| Benefit | Details |
|---------|---------|
| Leverage | Margin ~20-25% vs full capital for cash equity |
| Short execution | No uptick rule concerns, cleaner fills |
| Position sizing | Larger positions with same capital base |

Same entry/exit logic, just different order instrument. Requires F&O activation and additional margin.

### Phase 4 — Index Options Hedge (Net Delta Neutralization)

At end of entry window (~14:00), calculate net portfolio delta:

```
Example: 2 long positions + 5 short positions = 5 net short
→ Sell 5 OTM Nifty PUT options as hedge
```

| Parameter | Value |
|-----------|-------|
| Purpose | Partial hedge against net directional exposure |
| Strike selection | ~1-2% OTM from current Nifty spot |
| Expiry | Current week (maximize theta decay) |
| Square off | With futures at 15:20 EOD |

**Benefits:**
- Partial hedge if market rallies against net short bias
- Theta decay earns additional income on most days (options expire worthless)
- Acts as "insurance premium" that often becomes profit
- Expected benefit: +0.1-0.3% daily from theta on average
- Caps tail risk on extreme gap-up/gap-down days

### Phase 5 — Expansion

- Add more F&O stocks (quarterly screen for high-beta, high-volume candidates)
- Per-stock config optimization (some stocks prefer OR5, some OR15)
- Intraday scaling: add 2nd trade/day with tighter filters
- Multi-timeframe confirmation (daily trend + intraday breakout)

---

## 11. Configuration Reference

Full `ORBConfig` dataclass from `services/orb_backtest_engine.py`:

### Core ORB Parameters

| Parameter | Type | Default | Tested Values | Notes |
|-----------|------|---------|---------------|-------|
| `or_minutes` | int | 15 | 5, 15, 30, 60 | OR window duration |
| `last_entry_time` | str | '14:00' | - | No entries after this time |
| `eod_exit_time` | str | '15:20' | - | Mandatory squareoff time |
| `max_trades_per_day` | int | 1 | 1, 2, 3 | Per stock per day |
| `allow_longs` | bool | True | - | Toggle long entries |
| `allow_shorts` | bool | True | - | Toggle short entries |

### Stop Loss Parameters

| Parameter | Type | Default | Tested Values | Notes |
|-----------|------|---------|---------------|-------|
| `sl_type` | str | 'or_opposite' | 'or_opposite', 'fixed_pct', 'atr_multiple' | SL method |
| `fixed_sl_pct` | float | 1.0 | - | % from entry (if sl_type='fixed_pct') |
| `atr_sl_multiple` | float | 1.5 | - | ATR multiplier (if sl_type='atr_multiple') |
| `atr_period` | int | 14 | - | ATR lookback |

### Target Parameters

| Parameter | Type | Default | Tested Values | Notes |
|-----------|------|---------|---------------|-------|
| `target_type` | str | 'r_multiple' | 'r_multiple', 'fixed_pct', 'or_range_multiple' | Target method |
| `r_multiple` | float | 1.5 | 1.0, 1.5, 2.0 | Risk multiple (if target_type='r_multiple') |
| `fixed_target_pct` | float | 2.0 | - | % from entry (if target_type='fixed_pct') |
| `or_range_multiple` | float | 1.0 | - | OR range multiplier |

### Filter Parameters

| Parameter | Type | Default | Winner Config | Notes |
|-----------|------|---------|---------------|-------|
| `use_vwap_filter` | bool | False | **True** | Long: price > VWAP, Short: price < VWAP |
| `use_rsi_filter` | bool | False | **True** | RSI on higher timeframe |
| `rsi_period` | int | 14 | 14 | RSI lookback |
| `rsi_timeframe` | str | '15min' | '15min' | Resampled from 5-min |
| `rsi_long_threshold` | float | 60.0 | 60.0 | Long requires RSI > this |
| `rsi_short_threshold` | float | 40.0 | 40.0 | Short requires RSI < this |
| `use_cpr_dir_filter` | bool | False | **True** | Long: price > TC, Short: price < BC |
| `use_cpr_width_filter` | bool | False | **True** | Skip day if CPR width > threshold |
| `cpr_width_threshold_pct` | float | 0.5 | 0.5 | % of price |
| `use_virgin_cpr_filter` | bool | False | False | Tested, not in winner config |
| `use_prev_cpr_filter` | bool | False | False | Tested, not in winner config |
| `use_prev_hl_filter` | bool | False | False | Tested, rejected — no PF improvement |
| `use_pivot_sr_filter` | bool | False | False | Pivot S/R proximity check |
| `pivot_sr_buffer_pct` | float | 0.2 | - | Buffer around pivot levels |
| `use_bb_filter` | bool | False | False | Bollinger Bands filter |
| `bb_period` | int | 20 | - | BB lookback |
| `bb_std` | float | 2.0 | - | BB standard deviations |
| `use_mtf_filter` | bool | False | False | Multi-timeframe EMA trend |
| `mtf_ema_period` | int | 20 | - | Daily EMA period |
| `use_inside_day_filter` | bool | False | False | Inside day detection |
| `use_narrow_range_filter` | bool | False | False | NR4/NR7 detection |
| `nr_lookback` | int | 4 | - | Narrow range lookback |

### Winner Configuration

```python
ORBConfig(
    or_minutes=15,
    last_entry_time='14:00',
    eod_exit_time='15:20',
    max_trades_per_day=1,
    sl_type='or_opposite',
    target_type='r_multiple',
    r_multiple=1.5,
    use_vwap_filter=True,
    use_rsi_filter=True,
    rsi_period=14,
    rsi_timeframe='15min',
    rsi_long_threshold=60.0,
    rsi_short_threshold=40.0,
    use_cpr_dir_filter=True,
    use_cpr_width_filter=True,
    cpr_width_threshold_pct=0.5,
)
```
