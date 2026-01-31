# Momentum + Quality Portfolio Strategy

## Strategy Specification Document

**Version:** 2.0
**Last Updated:** January 31, 2026
**Strategy Type:** Long-only equity portfolio
**Investment Style:** Momentum + Fundamental Quality hybrid
**Rebalancing Frequency:** Semi-annual + Event-driven
**Design Status:** FINALIZED

---

## 0. Design Decisions (Finalized)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | **D/E Ratio Threshold** | 0.20 (non-financial stocks only) | Strict quality gate. Banks/NBFCs use separate metrics. Accepts sector bias toward IT/FMCG/Pharma |
| 2 | **Bank/NBFC Quality Gate** | ROA > 1% + ROE > 12% (MVP). Gross NPA trend added in Phase 2 | D/E is meaningless for banks. ROA/ROE available via Yahoo Finance. NPA needs Screener.in (Phase 2) |
| 3 | **Revenue/OPM "Increasing" Definition** | Each year shows positive growth (no YoY decline) | Original spec required accelerating growth which is too restrictive. 20%→18%→25% now passes |
| 4 | **Fundamental Data Source** | Yahoo Finance (MVP). Screener.in (Phase 2) | Already integrated in codebase. Covers 500 stocks with quarterly data for 3-5 years |
| 5 | **Backtest Period (MVP)** | 3 years (2023-2025) | Yahoo Finance data is reliable for this range. Full 10-year backtest deferred to Phase 2 with Screener.in |
| 6 | **Capital Allocation** | 80% equity (30 stocks) / 20% debt fund reserve | Debt fund earns ~6-7% p.a. while waiting for topup opportunities |
| 7 | **Topup Funding Source** | Debt fund reserve. If exhausted, skip topup | Fresh capital not required. Topups only deploy from the 20% reserve |
| 8 | **Nifty 500 List Source** | Static CSV in repo, refreshed quarterly | NSE blocks programmatic access. Manual update is reliable and simple |

---

## 1. Executive Summary

This strategy combines **price momentum** (stocks near all-time highs) with **fundamental quality** (revenue growth, low debt, strong operating margins) to build a concentrated 30-stock portfolio from the Nifty 500 universe. The strategy deploys **80% of capital** into equities with **20% in a debt fund reserve** for breakout topups. It includes a unique **breakout topup mechanism** to pyramid into winners and **fundamental-based exit criteria** to avoid holding deteriorating businesses.

---

## 2. Universe Definition

| Parameter | Value |
|-----------|-------|
| **Base Universe** | Nifty 500 |
| **Universe Size** | ~500 stocks |
| **Minimum Liquidity** | Included in Nifty 500 (inherently liquid) |
| **Exclusions** | None (Nifty 500 already excludes illiquid stocks) |

### Data Requirement
```python
UNIVERSE_CONFIG = {
    "index": "NIFTY500",
    "source": "NSE",
    "refresh_frequency": "daily"
}
```

---

## 3. Entry Criteria

### 3.1 Momentum Filter (Stage 1)

**Objective:** Identify stocks exhibiting strong price momentum.

| Criterion | Condition | Calculation |
|-----------|-----------|-------------|
| **ATH Proximity** | Within 10% of 52-week high | `(52W_High - Current_Price) / 52W_High <= 0.10` |

```python
def momentum_filter(price_data: pd.DataFrame) -> bool:
    """
    Filter stocks within 10% of their 52-week high.
    
    Args:
        price_data: DataFrame with columns ['date', 'close', 'high', 'low', 'volume']
    
    Returns:
        bool: True if stock passes momentum filter
    """
    high_52w = price_data['high'].rolling(window=252).max().iloc[-1]
    current_price = price_data['close'].iloc[-1]
    
    distance_from_ath = (high_52w - current_price) / high_52w
    
    return distance_from_ath <= 0.10
```

### 3.2 Fundamental Quality Filter (Stage 2)

**Objective:** Rank momentum-filtered stocks by fundamental quality.

#### 3.2.1 Non-Financial Stocks (Ranking Criteria):

| Rank | Criterion | Condition | Data Source |
|------|-----------|-----------|-------------|
| 1 | **Revenue/Sales Growth** | >15% CAGR for last 3 years AND positive growth each year (no YoY decline) | Quarterly/Annual Results |
| 2 | **Debt-to-Equity Ratio** | ≤ 0.20 | Balance Sheet |
| 3 | **Operating Profit Margin (OPM)** | >15% for last 3 years | P&L Statement |
| 4 | **OPM Trend** | Positive each year (no YoY decline in OPM) | P&L Statement (YoY comparison) |

#### 3.2.2 Financial Stocks — Banks/NBFCs (Alternative Criteria):

D/E ratio is not applicable to financial companies (leverage is their business model).

| Rank | Criterion | Condition | Data Source | Phase |
|------|-----------|-----------|-------------|-------|
| 1 | **Revenue/Sales Growth** | Same as non-financial | Annual Results | MVP |
| 2 | **ROA (Return on Assets)** | > 1.0% | Yahoo Finance | MVP |
| 3 | **ROE (Return on Equity)** | > 12% | Yahoo Finance | MVP |
| 4 | **Gross NPA Trend** | Declining for 3 years | Screener.in | Phase 2 |
| 5 | **Net NPA Ratio** | < 2% | Screener.in | Phase 2 |

**Sector classification:** Stocks with sector = "Financial Services", "Banks", or industry containing "Bank", "NBFC", "Finance" use the financial criteria.

```python
@dataclass
class FundamentalCriteria:
    """Fundamental quality thresholds."""

    # Revenue Growth (all stocks)
    min_revenue_growth_3y_cagr: float = 0.15  # 15%
    require_revenue_positive_each_year: bool = True  # Each year must show positive growth

    # Non-Financial: Debt
    max_debt_to_equity: float = 0.20

    # Non-Financial: Operating Margins
    min_opm_3y: float = 0.15  # 15%
    require_opm_positive_each_year: bool = True  # OPM must not decline YoY

    # Financial (Banks/NBFCs): Alternative quality gate
    min_roa: float = 0.01  # 1.0% Return on Assets
    min_roe: float = 0.12  # 12% Return on Equity
    # Phase 2: max_gross_npa, require_npa_declining


def calculate_revenue_growth(annual_data: pd.DataFrame) -> dict:
    """
    Calculate 3-year revenue CAGR and YoY growth trend.
    
    Args:
        annual_data: DataFrame with columns ['year', 'revenue']
    
    Returns:
        dict with keys: 'cagr_3y', 'is_increasing', 'yoy_growth_list'
    """
    if len(annual_data) < 4:
        return {'cagr_3y': None, 'is_increasing': False, 'yoy_growth_list': []}
    
    revenue = annual_data.sort_values('year')['revenue'].values
    
    # 3-year CAGR
    cagr_3y = (revenue[-1] / revenue[-4]) ** (1/3) - 1
    
    # YoY growth for last 3 years
    yoy_growth = []
    for i in range(-3, 0):
        yoy = (revenue[i] - revenue[i-1]) / revenue[i-1]
        yoy_growth.append(yoy)

    # Check if positive each year (no YoY decline — NOT accelerating)
    is_positive_each_year = all(g > 0 for g in yoy_growth)

    return {
        'cagr_3y': cagr_3y,
        'is_positive_each_year': is_positive_each_year,
        'yoy_growth_list': yoy_growth
    }


def calculate_debt_to_equity(balance_sheet: pd.DataFrame) -> float:
    """
    Calculate Debt-to-Equity ratio from latest balance sheet.
    
    Args:
        balance_sheet: DataFrame with latest balance sheet data
    
    Returns:
        float: D/E ratio
    """
    total_debt = balance_sheet['long_term_debt'] + balance_sheet['short_term_debt']
    shareholders_equity = balance_sheet['total_equity']
    
    if shareholders_equity <= 0:
        return float('inf')  # Negative equity = fail
    
    return total_debt / shareholders_equity


def calculate_opm_metrics(annual_data: pd.DataFrame) -> dict:
    """
    Calculate Operating Profit Margin metrics.
    
    Args:
        annual_data: DataFrame with columns ['year', 'operating_profit', 'revenue']
    
    Returns:
        dict with keys: 'opm_3y_list', 'all_above_threshold', 'is_increasing'
    """
    if len(annual_data) < 3:
        return {'opm_3y_list': [], 'all_above_threshold': False, 'is_increasing': False}
    
    data = annual_data.sort_values('year').tail(3)
    opm_list = (data['operating_profit'] / data['revenue']).values
    
    # Check OPM did not decline year-over-year (positive trend, not accelerating)
    opm_no_decline = all(opm_list[i] >= opm_list[i-1] for i in range(1, len(opm_list)))

    return {
        'opm_3y_list': opm_list.tolist(),
        'all_above_threshold': all(opm >= 0.15 for opm in opm_list),
        'opm_no_decline': opm_no_decline  # Each year's OPM >= prior year
    }


def rank_stocks(filtered_stocks: list, fundamental_data: dict) -> pd.DataFrame:
    """
    Rank stocks by fundamental criteria.
    
    Ranking methodology:
    1. Revenue growth CAGR (higher = better)
    2. D/E ratio (lower = better)
    3. Average OPM (higher = better)
    4. OPM growth rate (higher = better)
    
    Returns:
        DataFrame sorted by composite rank (ascending = better)
    """
    scores = []
    
    for symbol in filtered_stocks:
        data = fundamental_data[symbol]
        
        score = {
            'symbol': symbol,
            'revenue_cagr': data['revenue']['cagr_3y'],
            'debt_to_equity': data['debt_to_equity'],
            'avg_opm': np.mean(data['opm']['opm_3y_list']),
            'opm_growth': data['opm']['opm_3y_list'][-1] - data['opm']['opm_3y_list'][0]
        }
        scores.append(score)
    
    df = pd.DataFrame(scores)
    
    # Create percentile ranks (lower rank = better)
    df['rank_revenue'] = df['revenue_cagr'].rank(ascending=False)
    df['rank_de'] = df['debt_to_equity'].rank(ascending=True)
    df['rank_opm'] = df['avg_opm'].rank(ascending=False)
    df['rank_opm_growth'] = df['opm_growth'].rank(ascending=False)
    
    # Composite rank with weights
    df['composite_rank'] = (
        df['rank_revenue'] * 0.30 +
        df['rank_de'] * 0.25 +
        df['rank_opm'] * 0.25 +
        df['rank_opm_growth'] * 0.20
    )
    
    return df.sort_values('composite_rank')
```

### 3.3 Portfolio Construction

| Parameter | Value |
|-----------|-------|
| **Portfolio Size** | 30 stocks |
| **Selection Method** | Top 30 by composite rank |
| **Initial Weighting** | Equal weight |
| **Capital Split** | 80% equity / 20% debt fund reserve |
| **Initial Position Size** | `(Portfolio_Value × 0.80) / 30` |
| **Debt Fund Reserve** | 20% of capital, earns ~6-7% p.a., used for topups |

```python
PORTFOLIO_CONFIG = {
    "max_positions": 30,
    "equity_allocation_pct": 0.80,  # 80% into stocks
    "debt_reserve_pct": 0.20,  # 20% in debt fund for topups
    "debt_fund_annual_return": 0.065,  # ~6.5% assumed return
    "initial_weight": 0.80 / 30,  # ~2.67% per position
    "min_position_size": 0.01,  # 1% minimum
    "max_position_size": 0.10,  # 10% maximum (after topups)
}
```

---

## 4. Topup Mechanism (Consolidation Breakout)

### 4.1 Consolidation Definition

A stock is in **consolidation** when:

| Criterion | Condition |
|-----------|-----------|
| **Duration** | Minimum 20 trading days |
| **Range** | `(Max_High - Min_Low) / Average_Price <= 5%` |
| **Structure** | At least 2 local highs AND 2 local lows within the range |

```python
@dataclass
class ConsolidationConfig:
    """Configuration for consolidation detection."""
    
    min_duration_days: int = 20
    max_range_pct: float = 0.05  # 5%
    min_local_highs: int = 2
    min_local_lows: int = 2
    local_extrema_window: int = 3  # Days to look for local high/low


def detect_consolidation(price_data: pd.DataFrame, config: ConsolidationConfig) -> dict:
    """
    Detect if stock is in consolidation.
    
    Args:
        price_data: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
        config: ConsolidationConfig instance
    
    Returns:
        dict with keys: 'is_consolidating', 'range_high', 'range_low', 'duration'
    """
    lookback = price_data.tail(config.min_duration_days)
    
    range_high = lookback['high'].max()
    range_low = lookback['low'].min()
    avg_price = lookback['close'].mean()
    
    range_pct = (range_high - range_low) / avg_price
    
    if range_pct > config.max_range_pct:
        return {'is_consolidating': False}
    
    # Count local highs and lows
    highs = lookback['high'].values
    lows = lookback['low'].values
    
    local_highs = 0
    local_lows = 0
    window = config.local_extrema_window
    
    for i in range(window, len(highs) - window):
        # Local high: higher than surrounding days
        if highs[i] == max(highs[i-window:i+window+1]):
            local_highs += 1
        # Local low: lower than surrounding days
        if lows[i] == min(lows[i-window:i+window+1]):
            local_lows += 1
    
    is_consolidating = (
        local_highs >= config.min_local_highs and 
        local_lows >= config.min_local_lows
    )
    
    return {
        'is_consolidating': is_consolidating,
        'range_high': range_high,
        'range_low': range_low,
        'range_pct': range_pct,
        'local_highs': local_highs,
        'local_lows': local_lows,
        'duration': config.min_duration_days
    }
```

### 4.2 Breakout Definition

A **breakout** occurs when:

| Criterion | Condition |
|-----------|-----------|
| **Price Break** | Close > Consolidation Range High × 1.02 (2% buffer) |
| **Volume Confirmation** | Volume > 1.5 × 20-day Average Volume |

```python
@dataclass
class BreakoutConfig:
    """Configuration for breakout detection."""
    
    price_buffer_pct: float = 0.02  # 2% above range high
    volume_multiplier: float = 1.5  # 1.5x average volume
    volume_avg_period: int = 20


def detect_breakout(
    price_data: pd.DataFrame, 
    consolidation: dict, 
    config: BreakoutConfig
) -> dict:
    """
    Detect breakout from consolidation.
    
    Args:
        price_data: DataFrame with OHLCV data
        consolidation: Output from detect_consolidation()
        config: BreakoutConfig instance
    
    Returns:
        dict with keys: 'is_breakout', 'breakout_price', 'volume_ratio'
    """
    if not consolidation['is_consolidating']:
        return {'is_breakout': False}
    
    current = price_data.iloc[-1]
    breakout_threshold = consolidation['range_high'] * (1 + config.price_buffer_pct)
    
    avg_volume = price_data['volume'].tail(config.volume_avg_period).mean()
    volume_ratio = current['volume'] / avg_volume
    
    is_breakout = (
        current['close'] > breakout_threshold and
        volume_ratio >= config.volume_multiplier
    )
    
    return {
        'is_breakout': is_breakout,
        'breakout_price': current['close'],
        'breakout_threshold': breakout_threshold,
        'volume_ratio': volume_ratio,
        'avg_volume': avg_volume,
        'current_volume': current['volume']
    }
```

### 4.3 Topup Execution

| Parameter | Value |
|-----------|-------|
| **Topup Amount** | 20% of **original** entry amount |
| **Funding Source** | Debt fund reserve (20% of initial capital) |
| **Repeat** | Unlimited (each new breakout triggers topup) |
| **Max Position Size** | 10% of portfolio (optional cap) |
| **If Reserve Exhausted** | Skip topup — do not sell other positions |

The 20% debt fund reserve earns ~6.5% p.a. while idle. When a breakout topup is triggered, capital is redeemed from the debt fund. If the debt fund is fully deployed into equities, further topups are skipped until positions are exited and cash returns to the reserve.

```python
@dataclass
class TopupConfig:
    """Configuration for topup mechanism."""

    topup_pct_of_initial: float = 0.20  # 20% of original entry
    max_position_pct: float = 0.10  # 10% max position size
    cooldown_days: int = 5  # Min days between topups


def calculate_topup(
    position: Position,
    portfolio_value: float,
    debt_fund_balance: float,
    config: TopupConfig
) -> dict:
    """
    Calculate topup amount for a breakout, funded from debt reserve.

    Args:
        position: Current position object
        portfolio_value: Current total portfolio value
        debt_fund_balance: Available cash in debt fund reserve
        config: TopupConfig instance

    Returns:
        dict with keys: 'topup_amount', 'new_position_value', 'blocked_reason'
    """
    original_entry_value = position.initial_entry_value
    topup_amount = original_entry_value * config.topup_pct_of_initial

    # Check debt fund has enough cash
    if debt_fund_balance < topup_amount:
        if debt_fund_balance > 0:
            topup_amount = debt_fund_balance  # Deploy whatever is left
        else:
            return {
                'topup_amount': 0,
                'new_position_value': position.current_value,
                'blocked_reason': 'Debt fund reserve exhausted'
            }

    current_position_value = position.current_value
    new_position_value = current_position_value + topup_amount

    max_allowed = portfolio_value * config.max_position_pct

    if new_position_value > max_allowed:
        topup_amount = max(0, max_allowed - current_position_value)
        reason = "Position size cap reached"
    else:
        reason = None

    return {
        'topup_amount': topup_amount,
        'new_position_value': new_position_value,
        'blocked_reason': reason
    }
```

---

## 5. Exit Criteria

### 5.1 Fundamental Exit (Primary)

Exit when company shows sustained fundamental deterioration:

| Exit Trigger | Condition |
|--------------|-----------|
| **Quarterly Decline** | 3 consecutive quarters with Sales AND Profits declining >10% YoY |
| **Annual Decline** | 2 consecutive years of negative growth in Sales AND Profits |

```python
@dataclass
class FundamentalExitConfig:
    """Configuration for fundamental exit triggers."""
    
    quarterly_decline_threshold: float = -0.10  # -10%
    quarterly_consecutive_count: int = 3
    annual_decline_threshold: float = 0.0  # negative growth
    annual_consecutive_count: int = 2


def check_fundamental_exit(
    quarterly_results: pd.DataFrame,
    annual_results: pd.DataFrame,
    config: FundamentalExitConfig
) -> dict:
    """
    Check if fundamental exit criteria are met.
    
    Args:
        quarterly_results: DataFrame with columns ['quarter', 'sales', 'profit']
        annual_results: DataFrame with columns ['year', 'sales', 'profit']
        config: FundamentalExitConfig instance
    
    Returns:
        dict with keys: 'should_exit', 'exit_reason', 'details'
    """
    # Check quarterly decline
    quarterly = quarterly_results.sort_values('quarter').tail(config.quarterly_consecutive_count)
    
    quarterly_exits = []
    for _, row in quarterly.iterrows():
        yoy_sales_growth = row['sales_yoy_growth']
        yoy_profit_growth = row['profit_yoy_growth']
        
        if yoy_sales_growth < config.quarterly_decline_threshold and \
           yoy_profit_growth < config.quarterly_decline_threshold:
            quarterly_exits.append(True)
        else:
            quarterly_exits.append(False)
    
    quarterly_exit = all(quarterly_exits) and len(quarterly_exits) == config.quarterly_consecutive_count
    
    # Check annual decline
    annual = annual_results.sort_values('year').tail(config.annual_consecutive_count)
    
    annual_exits = []
    for _, row in annual.iterrows():
        if row['sales_growth'] < config.annual_decline_threshold and \
           row['profit_growth'] < config.annual_decline_threshold:
            annual_exits.append(True)
        else:
            annual_exits.append(False)
    
    annual_exit = all(annual_exits) and len(annual_exits) == config.annual_consecutive_count
    
    if quarterly_exit:
        return {
            'should_exit': True,
            'exit_reason': 'QUARTERLY_DECLINE',
            'details': f'{config.quarterly_consecutive_count} consecutive quarters of >10% decline in sales and profits'
        }
    elif annual_exit:
        return {
            'should_exit': True,
            'exit_reason': 'ANNUAL_DECLINE',
            'details': f'{config.annual_consecutive_count} consecutive years of negative growth'
        }
    
    return {'should_exit': False, 'exit_reason': None, 'details': None}
```

### 5.2 Semi-Annual Rebalance Exit

Every 6 months, exit stocks that have fallen significantly from ATH:

| Exit Trigger | Condition |
|--------------|-----------|
| **ATH Drawdown** | Stock is down >20% from its ATH (since entry or rolling) |

```python
@dataclass
class RebalanceExitConfig:
    """Configuration for semi-annual rebalance exits."""
    
    max_drawdown_from_ath: float = 0.20  # 20%
    ath_reference: str = "rolling"  # "entry" or "rolling"
    rebalance_months: list = field(default_factory=lambda: [1, 7])  # January, July


def check_rebalance_exit(
    position: Position,
    price_data: pd.DataFrame,
    config: RebalanceExitConfig
) -> dict:
    """
    Check if position should be exited during rebalance.
    
    Args:
        position: Current position object
        price_data: DataFrame with OHLCV data
        config: RebalanceExitConfig instance
    
    Returns:
        dict with keys: 'should_exit', 'drawdown', 'ath_reference_price'
    """
    if config.ath_reference == "entry":
        ath = position.ath_at_entry
    else:  # rolling
        ath = price_data['high'].max()
    
    current_price = price_data['close'].iloc[-1]
    drawdown = (ath - current_price) / ath
    
    return {
        'should_exit': drawdown > config.max_drawdown_from_ath,
        'drawdown': drawdown,
        'ath_reference_price': ath,
        'current_price': current_price
    }
```

---

## 6. Replacement Logic

When a stock is exited, replace it with the top-ranked stock from a fresh screening:

```python
def replace_exited_position(
    exited_position: Position,
    portfolio: Portfolio,
    universe: list,
    fundamental_data: dict,
    price_data: dict
) -> Position:
    """
    Replace exited position with top-ranked new stock.
    
    Args:
        exited_position: The position being exited
        portfolio: Current portfolio object
        universe: List of all symbols in universe
        fundamental_data: Dict of fundamental data by symbol
        price_data: Dict of price data by symbol
    
    Returns:
        New Position object to add
    """
    # Get current holdings
    current_symbols = [p.symbol for p in portfolio.positions]
    
    # Filter universe (exclude current holdings)
    available = [s for s in universe if s not in current_symbols]
    
    # Apply momentum filter
    momentum_passed = [
        s for s in available 
        if momentum_filter(price_data[s])
    ]
    
    # Apply fundamental filter and rank
    fundamental_passed = []
    for symbol in momentum_passed:
        if passes_fundamental_criteria(fundamental_data[symbol]):
            fundamental_passed.append(symbol)
    
    if not fundamental_passed:
        return None  # No replacement available
    
    # Rank and select top
    ranked = rank_stocks(fundamental_passed, fundamental_data)
    top_symbol = ranked.iloc[0]['symbol']
    
    # Create new position with exited position's value
    new_position = Position(
        symbol=top_symbol,
        entry_price=price_data[top_symbol]['close'].iloc[-1],
        entry_value=exited_position.exit_value,
        entry_date=datetime.now()
    )
    
    return new_position
```

---

## 7. Risk Management (Recommended Additions)

### 7.1 Market Regime Filter (Optional but Recommended)

Pause new entries during bear markets:

| Condition | Action |
|-----------|--------|
| Nifty 500 < 200 DMA | No new entries (hold existing) |
| India VIX > 25 | No new entries |

```python
@dataclass
class MarketRegimeConfig:
    """Configuration for market regime filter."""
    
    use_regime_filter: bool = True
    index_symbol: str = "NIFTY500"
    ma_period: int = 200
    vix_threshold: float = 25.0


def check_market_regime(
    index_data: pd.DataFrame,
    vix_data: pd.DataFrame,
    config: MarketRegimeConfig
) -> dict:
    """
    Check market regime for entry permission.
    
    Returns:
        dict with keys: 'allow_entries', 'regime', 'details'
    """
    if not config.use_regime_filter:
        return {'allow_entries': True, 'regime': 'NEUTRAL', 'details': 'Filter disabled'}
    
    index_close = index_data['close'].iloc[-1]
    index_ma = index_data['close'].rolling(config.ma_period).mean().iloc[-1]
    current_vix = vix_data['close'].iloc[-1]
    
    above_ma = index_close > index_ma
    vix_ok = current_vix < config.vix_threshold
    
    if above_ma and vix_ok:
        return {'allow_entries': True, 'regime': 'BULLISH', 'details': 'Above 200 DMA, VIX normal'}
    elif not above_ma:
        return {'allow_entries': False, 'regime': 'BEARISH', 'details': f'Index below {config.ma_period} DMA'}
    else:
        return {'allow_entries': False, 'regime': 'HIGH_VOLATILITY', 'details': f'VIX > {config.vix_threshold}'}
```

### 7.2 Sector Concentration Limits (Optional)

| Parameter | Value |
|-----------|-------|
| **Max per Sector** | 6 stocks (20% of portfolio) |

```python
SECTOR_LIMITS = {
    "max_stocks_per_sector": 6,
    "max_weight_per_sector": 0.25  # 25%
}
```

### 7.3 Technical Stop Loss (Optional)

| Parameter | Value |
|-----------|-------|
| **Hard Stop** | -30% from entry price |

```python
def check_stop_loss(position: Position, current_price: float, threshold: float = 0.30) -> bool:
    """Check if position has hit stop loss."""
    drawdown = (position.entry_price - current_price) / position.entry_price
    return drawdown > threshold
```

---

## 8. Data Requirements

### 8.1 Price Data (Zerodha Kite API)

| Data Type | Frequency | History Required |
|-----------|-----------|------------------|
| OHLCV | Daily | 252 days minimum |
| 52-Week High | Daily | Computed from price |
| Volume | Daily | 20+ days |

```python
PRICE_DATA_CONFIG = {
    "source": "zerodha_kite",
    "interval": "day",
    "history_days": 365,
    "fields": ["open", "high", "low", "close", "volume"]
}
```

### 8.2 Fundamental Data (External Source Required)

| Data Type | Frequency | Source Options |
|-----------|-----------|----------------|
| Quarterly Results | Quarterly | Screener.in, Trendlyne, EODHD |
| Annual Results | Annual | Screener.in, Trendlyne, EODHD |
| Balance Sheet | Annual | Screener.in, Trendlyne, EODHD |

```python
FUNDAMENTAL_DATA_CONFIG = {
    "source": "yahoo_finance",  # MVP: yfinance. Phase 2: add screener_in for NPA, deeper history
    "refresh_frequency": "daily",
    "fields": {
        "quarterly": ["revenue", "operating_profit", "net_profit", "quarter"],
        "annual": ["revenue", "operating_profit", "net_profit", "total_debt", "equity", "year"],
        "financial_sector": ["roa", "roe"]  # Banks/NBFCs: alternative to D/E
    }
}
```

### 8.3 Recommended Data Sources

| Source | Price Data | Fundamental Data | Cost |
|--------|------------|------------------|------|
| Zerodha Kite | ✅ Excellent | ❌ None | ₹500/month |
| Screener.in | ❌ None | ✅ Good (scraping) | Free (rate limited) |
| Trendlyne | ✅ Good | ✅ Excellent | Paid subscription |
| EODHD | ✅ Good | ✅ Good | ~$80/month |

---

## 9. Backtest Configuration

### 9.1 Simulation Parameters

```python
BACKTEST_CONFIG = {
    # Time period (MVP: 3 years. Phase 2: extend to 2015-2025 with Screener.in data)
    "start_date": "2023-01-01",
    "end_date": "2025-12-31",

    # Initial capital
    "initial_capital": 10_000_000,  # ₹1 Crore
    
    # Transaction costs
    "brokerage_pct": 0.0003,  # 0.03% (Zerodha)
    "stt_pct": 0.001,  # 0.1% STT on sell
    "gst_pct": 0.18,  # 18% GST on brokerage
    "stamp_duty_pct": 0.00015,  # 0.015%
    "slippage_pct": 0.001,  # 0.1% slippage assumption
    
    # Execution
    "execution_price": "next_day_open",  # or "same_day_close"
    
    # Rebalance
    "rebalance_months": [1, 7],  # January and July
    "rebalance_day": 1,  # First trading day of month
}
```

### 9.2 Performance Metrics to Track

```python
METRICS = [
    "total_return",
    "cagr",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "max_drawdown_duration",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "avg_holding_period",
    "turnover_rate",
    "total_trades",
    "avg_trade_return",
    "best_trade",
    "worst_trade",
]
```

### 9.3 Benchmark Comparison

```python
BENCHMARKS = [
    "NIFTY500",
    "NIFTY50",
    "NIFTY500_MOMENTUM_50",
    "NIFTY500_VALUE_50",
]
```

---

## 10. Technical Architecture

### 10.1 Reuse from Existing Codebase

| Component | Existing File | Reuse Level |
|-----------|--------------|-------------|
| Kite API auth & connection | `services/kite_service.py` | Fully reusable |
| OHLCV data fetch + SQLite storage | `services/data_manager.py` | Extend with Nifty 500 universe |
| Yahoo Finance fundamentals | `services/holdings_service.py` | Extend with quarterly data + bank metrics |
| Performance metrics (Sharpe, Sortino, etc.) | `services/metrics_calculator.py` | Add CAGR, Calmar, turnover |
| Background task runner | `app.py` (APScheduler pattern) | Fully reusable |
| Backtest DB pattern | `services/backtest_db.py` | New tables, same pattern |

### 10.2 New Services to Build

| Service | Description |
|---------|-------------|
| `services/nifty500_universe.py` | Nifty 500 constituent list from static CSV + instrument token cache |
| `services/fundamental_data_service.py` | Quarterly/annual data extraction, revenue CAGR, D/E, OPM, ROA/ROE |
| `services/momentum_filter.py` | ATH proximity calculation for 500 stocks |
| `services/consolidation_breakout.py` | Consolidation detection + breakout with volume confirmation |
| `services/mq_backtest_engine.py` | Event-driven backtest loop, portfolio management, topups, exits |
| `services/mq_backtest_db.py` | New DB tables: positions, topups, trades, daily_equity |

### 10.3 Existing Architectural Patterns to Follow

The existing `CoveredCallEngine` in [cpr_covered_call_service.py](services/cpr_covered_call_service.py) provides the best reference:
- Config dataclass → Position dataclass → Trade dataclass → Engine class with `run_backtest()`
- Integration with `MetricsCalculator` for performance analysis
- Integration with `BacktestDatabase` for persistence
- Background execution via APScheduler with progress polling

---

## 11. Implementation Checklist

### Phase 1: Data Infrastructure
- [ ] Create `nifty500_universe.py` with static CSV of Nifty 500 constituents
- [ ] Extend `data_manager.py` with bulk instrument token cache for 500 stocks
- [ ] Extend `holdings_service.py` to return structured quarterly data from Yahoo Finance
- [ ] Build `fundamental_data_service.py` with revenue CAGR, D/E, OPM, ROA/ROE calculations
- [ ] Add sector classification for 500 stocks (financial vs non-financial detection)

### Phase 2: Core Strategy Logic
- [ ] Implement momentum filter (ATH proximity within 10% of 52-week high)
- [ ] Implement fundamental filters (non-financial: D/E, OPM; financial: ROA, ROE)
- [ ] Implement composite ranking with weighted scores
- [ ] Build consolidation detection algorithm (20-day, 5% range, local extrema)
- [ ] Build breakout detection with volume confirmation (1.5x average)
- [ ] Implement topup calculation with debt fund reserve tracking

### Phase 3: Portfolio Management
- [ ] Build Portfolio/Position/Trade/Topup dataclasses
- [ ] Implement 80/20 capital allocation (equity + debt fund reserve)
- [ ] Implement fundamental exit triggers (3Q or 2Y decline)
- [ ] Implement semi-annual rebalance exit (>20% ATH drawdown)
- [ ] Build replacement stock selection from fresh screening
- [ ] Add sector concentration limits (max 6 stocks, 25% weight per sector)

### Phase 4: Backtest Engine
- [ ] Build event-driven backtest loop (`mq_backtest_engine.py`)
- [ ] Implement Indian transaction cost model (brokerage + STT + GST + stamp duty)
- [ ] Add slippage simulation (0.1%)
- [ ] Build new DB tables for momentum strategy results
- [ ] Extend `MetricsCalculator` with CAGR, Calmar ratio, turnover
- [ ] Generate comparison with NIFTY500, NIFTY50, Momentum 50 benchmarks

### Phase 5: Agent System
- [ ] Build `MomentumQualityAgent` orchestrator class
- [ ] Implement `MonitoringAgent` (daily: watch 30 portfolio stocks)
- [ ] Implement `ScreeningAgent` (monthly: full Nifty 500 scan + on-demand for replacements)
- [ ] Implement `RebalanceAgent` (semi-annual: uses latest screening to exit/replace)
- [ ] Implement `BacktestAgent` (on-demand: wraps mq_backtest_engine, generates full report)
- [ ] Build HTML report generator with Chart.js visualizations (equity curve, drawdown, heatmap, trade log)
- [ ] Add APScheduler jobs for daily/monthly/semi-annual schedules
- [ ] Add agent status API endpoints and log viewer
- [ ] Build agent dashboard page (single page showing latest reports and status)
- [ ] Add `[Run Backtest]` button to dashboard with config modal (date range, capital, parameters)

---

## 12. Agent Architecture (Implementation Approach: Agent-First)

### 12.1 Agent Overview

The strategy runs as an **autonomous agent system** that screens, monitors, and reports without manual intervention. A minimal dashboard provides visibility into agent activity.

```
┌─────────────────────────────────────────────────────────┐
│                 MomentumQualityAgent                     │
│                   (Orchestrator)                         │
├───────────┬───────────┬───────────┬───────────┬──────────┤
│ Monitoring│ Screening │ Rebalance │ Backtest  │ Reporting│
│   Agent   │   Agent   │   Agent   │   Agent   │   Agent  │
├───────────┼───────────┼───────────┼───────────┼──────────┤
│ DAILY     │ MONTHLY   │SEMI-ANNUAL│ ON-DEMAND │After each│
│ 30 stocks │ 500 stocks│ Jan & Jul │ Historical│agent run │
│ post-mkt  │ 1st of mo │ Uses last │ simulation│          │
│           │ + on-demnd│ screening │ + metrics │          │
├──────────────┴──────────────┴──────────────┴────────────┤
│              Shared Services Layer                       │
│  data_manager | holdings_service | kite_service          │
│  fundamental_data_service | metrics_calculator           │
└─────────────────────────────────────────────────────────┘
```

**Why this schedule:**
- **Daily scanning of 500 stocks is wasteful** -- the portfolio rebalances only twice a year
- Daily work is limited to **monitoring the 30 stocks we already own** (consolidation, breakouts, fundamental warnings)
- The **monthly full scan** keeps a pipeline of replacement candidates ready
- Semi-annual rebalance uses the **latest monthly scan** to identify exits and replacements
- If a position exits mid-cycle (fundamental trigger), the screening runs **on-demand** to find a replacement

### 12.2 Agent Components

#### A. ScreeningAgent (Monthly, 1st of Each Month + On-Demand for Replacements)

**Runs:** 1st trading day of each month (keeps candidate pipeline fresh). Also triggered on-demand when a position exits and needs replacement.

**What it does:**

| Step | Action | Output |
|------|--------|--------|
| 1 | **Market Regime Check** | Nifty 500 vs 200 DMA + India VIX → BULLISH / BEARISH / HIGH_VOL |
| 2 | **Momentum Scan** | Filter all 500 stocks for ATH proximity within 10% | ~50-100 momentum candidates |
| 3 | **Fundamental Filter** | Apply revenue growth, D/E (or ROA/ROE for banks), OPM criteria | ~30-50 quality stocks |
| 4 | **Composite Ranking** | Rank by weighted scores (revenue 30%, D/E 25%, OPM 25%, OPM growth 20%) | Top 30 ranked |
| 5 | **Pipeline Update** | Compare ranked list vs current portfolio → flag potential replacements | Replacement candidates ready |

```python
class ScreeningAgent:
    """Daily post-market screening of Nifty 500 universe."""

    def run(self) -> ScreeningReport:
        regime = self.check_market_regime()
        momentum_stocks = self.scan_momentum(universe='NIFTY500')
        quality_stocks = self.filter_fundamentals(momentum_stocks)
        ranked = self.rank_by_composite_score(quality_stocks)

        return ScreeningReport(
            date=datetime.now(),
            regime=regime,
            total_scanned=500,
            momentum_passed=len(momentum_stocks),
            quality_passed=len(quality_stocks),
            top_30=ranked[:30],
            new_candidates=self.find_new_candidates(ranked),
            screening_funnel=self.build_funnel_data()
        )
```

#### B. MonitoringAgent (Daily, Post-Market ~4:45 PM IST)

**What it checks for each portfolio stock:**

| Check | Condition | Action Generated |
|-------|-----------|-----------------|
| **Consolidation Detection** | Price in 5% range for 20+ days with 2+ local highs/lows | Log: "STOCK_X entering consolidation (Day N/20)" |
| **Breakout Alert** | Close > Range High × 1.02 AND Volume > 1.5x avg | **TOPUP SIGNAL**: Deploy from debt fund |
| **Fundamental Exit Watch** | Quarterly results declining >10% YoY for 2+ quarters | WARNING: "2/3 quarterly declines for XYZ" |
| **Fundamental Exit Trigger** | 3 consecutive quarters declining >10% | **EXIT SIGNAL**: Sell position, find replacement |
| **ATH Drawdown Track** | Current price vs rolling ATH | Log: "STOCK_X down 15% from ATH" (alert at >20%) |
| **Position Size Check** | Any position > 10% of portfolio | WARNING: "STOCK_X at 11.2% - consider trimming" |
| **Sector Concentration** | Any sector > 25% weight or > 6 stocks | WARNING: "IT sector at 27% (6 stocks)" |
| **Debt Fund Balance** | Track reserve available for topups | Log: "Debt fund: ₹18.5L available (92.5% of original)" |

```python
class MonitoringAgent:
    """Daily portfolio health monitoring."""

    def run(self, portfolio: Portfolio) -> MonitoringReport:
        signals = []

        for position in portfolio.positions:
            # Consolidation check
            consol = self.detect_consolidation(position.symbol)
            if consol['is_consolidating']:
                signals.append(ConsolidationSignal(position.symbol, consol))

                # Breakout check (only for stocks in consolidation)
                breakout = self.detect_breakout(position.symbol, consol)
                if breakout['is_breakout']:
                    topup = self.calculate_topup(position, portfolio)
                    signals.append(BreakoutTopupSignal(position.symbol, breakout, topup))

            # Fundamental exit check
            exit_check = self.check_fundamental_exit(position.symbol)
            if exit_check['should_exit']:
                signals.append(ExitSignal(position.symbol, exit_check))
            elif exit_check.get('warning'):
                signals.append(WarningSignal(position.symbol, exit_check))

            # ATH drawdown track
            drawdown = self.calculate_ath_drawdown(position)
            if drawdown > 0.15:
                signals.append(DrawdownWarning(position.symbol, drawdown))

        return MonitoringReport(
            date=datetime.now(),
            portfolio_value=portfolio.total_value,
            debt_fund_balance=portfolio.debt_fund_balance,
            signals=signals,
            position_sizes=self.check_position_limits(portfolio),
            sector_weights=self.check_sector_limits(portfolio)
        )
```

#### C. RebalanceAgent (Semi-Annual: January 1 + July 1)

**What it does:**

| Step | Action |
|------|--------|
| 1 | Check ATH drawdown for all 30 positions |
| 2 | Exit positions with >20% drawdown from rolling ATH |
| 3 | Run fresh ScreeningAgent to get latest top-ranked stocks |
| 4 | Replace exited positions with top-ranked new stocks (not already in portfolio) |
| 5 | Re-equal-weight remaining positions (optional) |
| 6 | Generate rebalance report with all trades |

```python
class RebalanceAgent:
    """Semi-annual portfolio rebalance."""

    def run(self, portfolio: Portfolio) -> RebalanceReport:
        exits = []
        entries = []

        # Step 1-2: Exit stocks with excessive drawdown
        for position in portfolio.positions:
            drawdown = self.calculate_ath_drawdown(position)
            if drawdown > 0.20:
                exits.append(ExitTrade(
                    symbol=position.symbol,
                    reason=f'ATH drawdown {drawdown:.1%}',
                    value=position.current_value
                ))

        # Step 3: Fresh screening
        screening = ScreeningAgent().run()
        current_symbols = [p.symbol for p in portfolio.positions if p.symbol not in [e.symbol for e in exits]]

        # Step 4: Replace with top-ranked new stocks
        for candidate in screening.top_30:
            if len(entries) >= len(exits):
                break
            if candidate.symbol not in current_symbols:
                entries.append(EntryTrade(
                    symbol=candidate.symbol,
                    value=exits[len(entries)].value  # Use exited position's value
                ))

        return RebalanceReport(
            date=datetime.now(),
            exits=exits,
            entries=entries,
            portfolio_before=portfolio.snapshot(),
            portfolio_after=self.simulate_rebalance(portfolio, exits, entries)
        )
```

#### D. BacktestAgent (On-Demand or Initial Setup)

**Runs:** Triggered manually from the dashboard (`[Run Backtest]` button) or automatically during initial portfolio setup to validate the strategy before deploying capital.

**What it does:**

| Step | Action | Output |
|------|--------|--------|
| 1 | **Load Config** | Read backtest parameters (date range, capital, costs) from UI or defaults | `BacktestConfig` |
| 2 | **Fetch Historical Data** | Pull OHLCV + fundamentals for the backtest period | Price + fundamental DataFrames |
| 3 | **Simulate Strategy** | Run full event-driven loop: screening → entry → monitoring → topups → exits → rebalance | Trade log, daily equity series |
| 4 | **Calculate Metrics** | Compute CAGR, Sharpe, Sortino, max drawdown, Calmar, win rate, turnover | `BacktestMetrics` |
| 5 | **Benchmark Comparison** | Compare vs Nifty 50, Nifty 500, Momentum 50 over same period | Relative performance |
| 6 | **Generate Report** | Build HTML report with all visualizations | Stored in `reports/` + DB |

```python
class BacktestAgent:
    """On-demand historical backtest runner."""

    def run(self, config: BacktestConfig = None) -> BacktestReport:
        config = config or BacktestConfig()  # Use defaults from Section 9

        # Phase 1: Data preparation
        price_data = self.load_historical_prices(
            config.start_date, config.end_date, universe='NIFTY500'
        )
        fundamental_data = self.load_historical_fundamentals(
            config.start_date, config.end_date
        )

        # Phase 2: Run backtest engine (wraps mq_backtest_engine.py)
        engine = MQBacktestEngine(config)
        result = engine.run(price_data, fundamental_data)

        # Phase 3: Compute metrics
        metrics = MetricsCalculator().compute_all(
            equity_curve=result.daily_equity,
            trades=result.trade_log,
            benchmark_data=self.load_benchmarks(config)
        )

        # Phase 4: Build report
        return BacktestReport(
            config=config,
            metrics=metrics,
            equity_curve=result.daily_equity,
            trade_log=result.trade_log,
            topup_log=result.topup_log,
            monthly_returns=self.compute_monthly_returns(result),
            drawdown_series=self.compute_drawdowns(result),
            screening_snapshots=result.screening_history,
            benchmark_comparison=metrics.benchmark_comparison,
            capital_allocation={
                'equity_deployed': result.equity_deployed_history,
                'debt_fund_balance': result.debt_fund_history
            }
        )
```

**Backtest Report Contents:**

| Section | What It Shows |
|---------|---------------|
| **Summary Card** | CAGR, Sharpe, max drawdown, win rate, total trades |
| **Equity Curve** | Strategy vs 3 benchmarks (Nifty 50, 500, Momentum 50) |
| **Drawdown Chart** | Peak-to-trough periods with recovery duration |
| **Monthly Returns Heatmap** | Year × Month grid, green/red intensity |
| **Trade Log** | Every entry, exit, topup with dates, prices, P&L |
| **Capital Allocation Over Time** | Equity vs debt fund balance through the backtest |
| **Screening History** | How many stocks passed each filter at each screening cycle |
| **Sector Rotation** | How sector allocation shifted over time |
| **Parameter Sensitivity** | Optional: grid search results if user runs optimization |

#### E. ReportingAgent (After every agent run)

**Reports generated:**

| Report | Trigger | Content | Format |
|--------|---------|---------|--------|
| **Daily Brief** | After MonitoringAgent | Portfolio P&L, active signals (consolidation/breakout/warnings), drawdown status | HTML page + JSON |
| **Weekly Digest** | Every Sunday | Aggregated weekly signals, sector breakdown, week-over-week performance | HTML page |
| **Monthly Screening Report** | After ScreeningAgent (1st of month) | Full 500-stock funnel, top 30 ranked, pipeline candidates, market regime | HTML page |
| **Rebalance Report** | After RebalanceAgent (Jan/Jul) | Exits, entries, before/after comparison, transaction costs | HTML page |
| **Backtest Report** | On demand | Full historical simulation results | HTML page |

### 12.3 Visualizations

Each report includes embedded Chart.js visualizations rendered as HTML:

| Visualization | Chart Type | Description |
|---------------|------------|-------------|
| **Equity Curve** | Line chart | Portfolio value vs Nifty 50, Nifty 500, Momentum 50 benchmarks |
| **Drawdown Chart** | Area chart | Peak-to-trough drawdowns with recovery periods highlighted |
| **Monthly Returns Heatmap** | Table with color coding | Rows = years, Cols = months, Green/Red intensity = return % |
| **Screening Funnel** | Horizontal bar | 500 → Momentum (80) → Quality (40) → Top 30 |
| **Sector Allocation** | Doughnut chart | Current sector weights with 25% limit indicator |
| **Position Sizes** | Horizontal bar | All 30 positions by weight, with 10% cap line |
| **Consolidation Tracker** | Mini price charts | For each stock in consolidation: price range + breakout level |
| **Debt Fund Balance** | Gauge/progress bar | % of original reserve remaining |
| **Topup Timeline** | Timeline chart | Historical topups with amounts and symbols |
| **Fundamental Scorecard** | Table | Revenue CAGR, D/E, OPM, ROA/ROE for each position |

### 12.4 Scheduling

```python
AGENT_SCHEDULE = {
    # DAILY: Monitor 30 portfolio stocks only (lightweight)
    "monitoring": {
        "trigger": "cron",
        "hour": 16, "minute": 30,  # 4:30 PM IST (post-market)
        "day_of_week": "mon-fri",
        "scope": "30 portfolio stocks"
    },

    # MONTHLY: Full 500-stock screening (keeps candidate pipeline fresh)
    "screening": {
        "trigger": "cron",
        "day": 1,  # 1st trading day of month
        "hour": 17, "minute": 0,  # 5 PM IST
        "scope": "Nifty 500 universe"
    },

    # WEEKLY: Digest report (no new scanning, just aggregates daily signals)
    "weekly_digest": {
        "trigger": "cron",
        "day_of_week": "sun",
        "hour": 10, "minute": 0
    },

    # SEMI-ANNUAL: Portfolio rebalance (uses latest monthly screening)
    "rebalance": {
        "trigger": "cron",
        "month": "1,7",  # January and July
        "day": 1,
        "hour": 9, "minute": 0
    },

    # ON-DEMAND: Triggered when a fundamental exit fires and needs replacement
    "replacement_screening": {
        "trigger": "event",  # Fired by MonitoringAgent when EXIT signal detected
        "scope": "Nifty 500 universe"
    }
}
```

### 12.5 Agent State Management

```python
# Agent state persisted in SQLite
AGENT_DB_SCHEMA = """
-- Current portfolio state
CREATE TABLE mq_portfolio (
    id INTEGER PRIMARY KEY,
    created_date DATE,
    initial_capital DECIMAL(15,2),
    equity_deployed DECIMAL(15,2),
    debt_fund_balance DECIMAL(15,2),
    total_value DECIMAL(15,2),
    status VARCHAR(20)  -- 'active', 'paused', 'backtest'
);

-- Agent execution log
CREATE TABLE mq_agent_runs (
    id INTEGER PRIMARY KEY,
    agent_type VARCHAR(30),  -- 'screening', 'monitoring', 'rebalance', 'backtest'
    run_date TIMESTAMP,
    status VARCHAR(20),
    signals_count INTEGER,
    report_path VARCHAR(255),
    summary TEXT
);

-- Active signals queue
CREATE TABLE mq_signals (
    id INTEGER PRIMARY KEY,
    agent_run_id INTEGER,
    signal_type VARCHAR(30),  -- 'TOPUP', 'EXIT', 'WARNING', 'ENTRY'
    symbol VARCHAR(20),
    priority VARCHAR(10),  -- 'HIGH', 'MEDIUM', 'LOW'
    details TEXT,
    status VARCHAR(20),  -- 'pending', 'executed', 'dismissed'
    created_date TIMESTAMP
);
"""
```

### 12.6 Agent Dashboard (Minimal UI)

A single Flask page `/agent` showing:

```
┌─────────────────────────────────────────────────────────┐
│  Momentum + Quality Agent Dashboard                      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ REGIME       │  │ PORTFOLIO    │  │ DEBT FUND    │  │
│  │ 🟢 BULLISH   │  │ ₹82.4L      │  │ ₹18.2L       │  │
│  │ Above 200DMA │  │ +12.3% YTD  │  │ 91% reserve  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ACTIVE SIGNALS                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 🔴 EXIT: INFOSYS - 3Q consecutive decline       │   │
│  │ 🟡 TOPUP: TATA MOTORS - Breakout (Vol 2.1x)    │   │
│  │ 🔵 WATCH: HDFC BANK - Consolidation Day 15/20  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  RECENT REPORTS                                         │
│  ├─ Daily Brief (Jan 31) ................ [View]       │
│  ├─ Weekly Digest (Jan 26) .............. [View]       │
│  └─ Monthly Report (Jan 1) .............. [View]       │
│                                                         │
│  AGENT RUNS                                             │
│  ├─ MonitoringAgent Today 4:30 PM  ✅ 3 signals       │
│  ├─ ScreeningAgent  Feb 1 5:00 PM  ✅ 47 momentum     │
│  └─ RebalanceAgent  Jul 1, 2026   ⏰ Scheduled        │
│                                                         │
│  [Run Backtest]  [Force Screen Now]  [View All Reports] │
└─────────────────────────────────────────────────────────┘
```

---

## 13. Appendix

### A. Key Constants

```python
# Strategy parameters
PORTFOLIO_SIZE = 30
ATH_PROXIMITY_THRESHOLD = 0.10  # 10%
MIN_REVENUE_GROWTH = 0.15  # 15%
MAX_DEBT_TO_EQUITY = 0.20
MIN_OPM = 0.15  # 15%
CONSOLIDATION_DAYS = 20
CONSOLIDATION_RANGE_PCT = 0.05  # 5%
BREAKOUT_VOLUME_MULTIPLIER = 1.5
TOPUP_PCT = 0.20  # 20%
REBALANCE_ATH_DRAWDOWN = 0.20  # 20%
QUARTERLY_DECLINE_THRESHOLD = 0.10  # 10%
```

### B. Database Schema (Suggested)

```sql
-- Positions table
CREATE TABLE positions (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR(20),
    entry_date DATE,
    entry_price DECIMAL(10,2),
    initial_value DECIMAL(15,2),
    current_shares DECIMAL(10,2),
    exit_date DATE,
    exit_price DECIMAL(10,2),
    exit_reason VARCHAR(50),
    total_topups INTEGER DEFAULT 0
);

-- Topups table
CREATE TABLE topups (
    id INTEGER PRIMARY KEY,
    position_id INTEGER,
    topup_date DATE,
    topup_price DECIMAL(10,2),
    topup_value DECIMAL(15,2),
    shares_added DECIMAL(10,2),
    breakout_volume_ratio DECIMAL(5,2)
);

-- Trades table
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    position_id INTEGER,
    trade_type VARCHAR(10),  -- BUY, SELL, TOPUP
    trade_date DATE,
    price DECIMAL(10,2),
    quantity DECIMAL(10,2),
    value DECIMAL(15,2),
    transaction_costs DECIMAL(10,2)
);
```

### C. References

1. Nifty 500 Momentum 50 Index Methodology - NSE India
2. "Momentum Strategies in Indian Markets" - MomentumLAB Research
3. "Quality Value Momentum Investment Strategy" - Quant Investing
4. Zerodha Kite Connect API Documentation
5. "Does Momentum Investing work in India?" - Capitalmind

---

**Document Version History**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-31 | Initial specification |
| 2.0 | 2026-01-31 | Finalized design: 80/20 capital split, bank-specific metrics (ROA/ROE), fixed "increasing" to "positive each year", Yahoo Finance as MVP data source, 3-year backtest period, debt fund reserve for topups |

---

*This specification is designed for integration with existing backtest frameworks. Adjust parameters as needed based on backtest results.*
