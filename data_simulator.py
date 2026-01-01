"""
Data Simulator for Covered Calls Backtester
============================================

Generates realistic stock price data for backtesting when real data is unavailable.
Uses geometric Brownian motion with mean reversion and realistic Indian market characteristics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path


# Indian stock characteristics (approximate)
STOCK_PARAMS = {
    "RELIANCE": {"price": 2500, "volatility": 0.25, "drift": 0.12},
    "TCS": {"price": 3800, "volatility": 0.22, "drift": 0.10},
    "HDFCBANK": {"price": 1650, "volatility": 0.20, "drift": 0.08},
    "INFY": {"price": 1500, "volatility": 0.24, "drift": 0.11},
    "ICICIBANK": {"price": 1000, "volatility": 0.22, "drift": 0.09},
    "HINDUNILVR": {"price": 2400, "volatility": 0.18, "drift": 0.08},
    "SBIN": {"price": 600, "volatility": 0.28, "drift": 0.10},
    "BHARTIARTL": {"price": 950, "volatility": 0.26, "drift": 0.14},
    "ITC": {"price": 450, "volatility": 0.20, "drift": 0.07},
    "AXISBANK": {"price": 1100, "volatility": 0.24, "drift": 0.08},
    "KOTAKBANK": {"price": 1800, "volatility": 0.21, "drift": 0.09},
    "LT": {"price": 3000, "volatility": 0.23, "drift": 0.11},
    "MARUTI": {"price": 10000, "volatility": 0.24, "drift": 0.10},
    "TITAN": {"price": 3200, "volatility": 0.26, "drift": 0.15},
    "BAJFINANCE": {"price": 7000, "volatility": 0.30, "drift": 0.12},
}


def generate_stock_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data for a stock using geometric Brownian motion
    with mean reversion and regime changes.
    """
    if seed is not None:
        np.random.seed(seed)

    # Get stock params or use defaults
    params = STOCK_PARAMS.get(symbol, {
        "price": 1000,
        "volatility": 0.25,
        "drift": 0.10
    })

    base_price = params["price"]
    annual_volatility = params["volatility"]
    annual_drift = params["drift"]

    # Trading days (exclude weekends)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    # Daily parameters
    daily_volatility = annual_volatility / np.sqrt(252)
    daily_drift = annual_drift / 252

    # Generate price path with regime changes
    prices = [base_price]
    current_regime = "normal"
    regime_duration = 0

    for i in range(1, n_days):
        # Regime switching (bull, bear, normal, volatile)
        regime_duration += 1
        if regime_duration > np.random.randint(20, 60):
            current_regime = np.random.choice(
                ["bull", "bear", "normal", "volatile"],
                p=[0.25, 0.20, 0.40, 0.15]
            )
            regime_duration = 0

        # Adjust drift and vol based on regime
        if current_regime == "bull":
            drift_adj = daily_drift * 2
            vol_adj = daily_volatility * 0.8
        elif current_regime == "bear":
            drift_adj = -daily_drift * 1.5
            vol_adj = daily_volatility * 1.2
        elif current_regime == "volatile":
            drift_adj = daily_drift * 0.5
            vol_adj = daily_volatility * 1.8
        else:  # normal
            drift_adj = daily_drift
            vol_adj = daily_volatility

        # Add mean reversion
        mean_reversion = 0.02 * (base_price - prices[-1]) / base_price

        # GBM with mean reversion
        random_return = np.random.normal(
            drift_adj + mean_reversion,
            vol_adj
        )

        new_price = prices[-1] * (1 + random_return)
        # Prevent negative prices
        new_price = max(new_price, base_price * 0.3)
        prices.append(new_price)

    # Generate OHLCV from close prices
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Intraday volatility (create realistic OHLC)
        intraday_range = close * np.random.uniform(0.01, 0.025)

        # Determine direction bias for the day
        if i > 0:
            direction = 1 if close > prices[i - 1] else -1
        else:
            direction = np.random.choice([1, -1])

        if direction > 0:  # Bullish day
            low = close - intraday_range * np.random.uniform(0.6, 1.0)
            high = close + intraday_range * np.random.uniform(0.2, 0.5)
            open_price = low + (close - low) * np.random.uniform(0.1, 0.4)
        else:  # Bearish day
            high = close + intraday_range * np.random.uniform(0.6, 1.0)
            low = close - intraday_range * np.random.uniform(0.2, 0.5)
            open_price = high - (high - close) * np.random.uniform(0.1, 0.4)

        # Volume (inversely correlated with price stability)
        base_volume = 1000000  # Base volume
        vol_multiplier = 1 + abs(close - prices[max(0, i - 1)]) / close * 10
        volume = int(base_volume * vol_multiplier * np.random.uniform(0.5, 1.5))

        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)

    return df


def generate_all_stock_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    base_seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """Generate data for multiple stocks"""
    all_data = {}

    for i, symbol in enumerate(symbols):
        seed = base_seed + i
        df = generate_stock_data(symbol, start_date, end_date, seed)
        all_data[symbol] = df
        print(f"Generated {len(df)} bars for {symbol}")

    return all_data


def save_to_csv(data: Dict[str, pd.DataFrame], output_dir: str = "backtest_data"):
    """Save generated data to CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for symbol, df in data.items():
        filepath = output_path / f"{symbol}_daily.csv"
        df.to_csv(filepath)
        print(f"Saved {filepath}")


class SimulatedDataManager:
    """
    Drop-in replacement for DataManager that uses simulated data.
    Compatible with the strategy optimizer.
    """

    def __init__(self, start_date: datetime = None, end_date: datetime = None):
        self.start_date = start_date or datetime(2023, 1, 1)
        self.end_date = end_date or datetime(2024, 12, 31)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._base_seed = 42

    def get_stock_data(
        self,
        symbol: str,
        from_date: datetime = None,
        to_date: datetime = None
    ) -> Optional[pd.DataFrame]:
        """Get stock data, generating if not in cache"""
        cache_key = symbol

        if cache_key not in self._cache:
            # Generate data for the full period
            seed = self._base_seed + hash(symbol) % 1000
            self._cache[cache_key] = generate_stock_data(
                symbol,
                self.start_date - timedelta(days=300),  # Extra for indicators
                self.end_date,
                seed
            )

        df = self._cache[cache_key]

        # Filter by date range
        if from_date:
            df = df[df.index >= pd.Timestamp(from_date)]
        if to_date:
            df = df[df.index <= pd.Timestamp(to_date)]

        return df.copy()


if __name__ == "__main__":
    # Test data generation
    print("Generating test data...")

    symbols = list(STOCK_PARAMS.keys())
    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)

    # Generate and save
    data = generate_all_stock_data(symbols, start, end)
    save_to_csv(data)

    # Test simulated data manager
    print("\nTesting SimulatedDataManager...")
    dm = SimulatedDataManager(start, end)
    df = dm.get_stock_data("RELIANCE")
    print(f"RELIANCE data: {len(df)} bars")
    print(df.tail())
