"""
Options Greeks Calculator
=========================

Calculate Delta, Gamma, Theta, Vega for option positions.
Uses Black-Scholes model for pricing and Greeks calculation.

Adapted from quantflow/kiteconnect01 for Covered Calls Backtester.
"""

import math
from scipy.stats import norm
from datetime import datetime
from typing import Dict, Tuple, Optional, List


class GreeksCalculator:
    """Calculate option Greeks using Black-Scholes model"""

    def __init__(self, risk_free_rate: float = 0.07):
        """
        Initialize calculator

        Args:
            risk_free_rate: Annual risk-free rate (default 7% for India)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_time_to_expiry(self, current_date: str, expiry_date: str) -> float:
        """
        Calculate time to expiry in years

        Args:
            current_date: Current date (YYYY-MM-DD)
            expiry_date: Expiry date (YYYY-MM-DD)

        Returns:
            Time to expiry in years
        """
        current = datetime.strptime(current_date, "%Y-%m-%d")
        expiry = datetime.strptime(expiry_date, "%Y-%m-%d")

        days_to_expiry = (expiry - current).days

        # Convert to years (using calendar days)
        return max(days_to_expiry / 365.0, 0.0001)  # Minimum to avoid division by zero

    def calculate_d1_d2(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = None
    ) -> Tuple[float, float]:
        """
        Calculate d1 and d2 for Black-Scholes formula

        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility (annual, e.g., 0.15 for 15%)
            risk_free_rate: Risk-free rate (optional, uses instance default)

        Returns:
            Tuple of (d1, d2)
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        # Black-Scholes d1 formula
        d1 = (math.log(spot / strike) +
              (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * math.sqrt(time_to_expiry))

        # d2 = d1 - sigma * sqrt(T)
        d2 = d1 - volatility * math.sqrt(time_to_expiry)

        return d1, d2

    def calculate_option_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: str = 'CE',
        risk_free_rate: float = None
    ) -> float:
        """
        Calculate theoretical option price using Black-Scholes

        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            option_type: 'CE' for call or 'PE' for put
            risk_free_rate: Risk-free rate (optional)

        Returns:
            Theoretical option price
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        d1, d2 = self.calculate_d1_d2(spot, strike, time_to_expiry, volatility, risk_free_rate)

        if option_type == 'CE':
            # Call option
            price = (spot * norm.cdf(d1) -
                    strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        else:  # PE
            # Put option
            price = (strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
                    spot * norm.cdf(-d1))

        return price

    def calculate_delta(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: str = 'CE',
        risk_free_rate: float = None
    ) -> float:
        """
        Calculate Delta (rate of change of option price with respect to spot)

        Delta interpretation:
        - Call: 0 to 1 (typically 0.5 for ATM)
        - Put: -1 to 0 (typically -0.5 for ATM)

        For covered calls, we typically sell calls with delta 0.30-0.40

        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            option_type: 'CE' for call or 'PE' for put
            risk_free_rate: Risk-free rate (optional)

        Returns:
            Delta value
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        d1, _ = self.calculate_d1_d2(spot, strike, time_to_expiry, volatility, risk_free_rate)

        if option_type == 'CE':
            return norm.cdf(d1)
        else:  # PE
            return norm.cdf(d1) - 1

    def calculate_gamma(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = None
    ) -> float:
        """
        Calculate Gamma (rate of change of Delta with respect to spot)

        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            risk_free_rate: Risk-free rate (optional)

        Returns:
            Gamma value
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        d1, _ = self.calculate_d1_d2(spot, strike, time_to_expiry, volatility, risk_free_rate)

        gamma = norm.pdf(d1) / (spot * volatility * math.sqrt(time_to_expiry))

        return gamma

    def calculate_theta(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: str = 'CE',
        risk_free_rate: float = None
    ) -> float:
        """
        Calculate Theta (time decay - change in option price per day)

        For covered calls, positive theta means we profit from time decay.

        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            option_type: 'CE' for call or 'PE' for put
            risk_free_rate: Risk-free rate (optional)

        Returns:
            Daily theta value
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        d1, d2 = self.calculate_d1_d2(spot, strike, time_to_expiry, volatility, risk_free_rate)

        # Common term
        term1 = -(spot * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry))

        if option_type == 'CE':
            term2 = -risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            theta = term1 + term2
        else:  # PE
            term2 = risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
            theta = term1 + term2

        # Convert to daily theta (divide by 365)
        return theta / 365

    def calculate_vega(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = None
    ) -> float:
        """
        Calculate Vega (sensitivity to volatility changes)

        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            risk_free_rate: Risk-free rate (optional)

        Returns:
            Vega value (for 1% change in IV)
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        d1, _ = self.calculate_d1_d2(spot, strike, time_to_expiry, volatility, risk_free_rate)

        vega = spot * norm.pdf(d1) * math.sqrt(time_to_expiry)

        # Return vega for 1% change in IV (divide by 100)
        return vega / 100

    def calculate_all_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: str = 'CE',
        risk_free_rate: float = None
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option

        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            option_type: 'CE' for call or 'PE' for put
            risk_free_rate: Risk-free rate (optional)

        Returns:
            Dictionary with all Greeks
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        return {
            'price': self.calculate_option_price(spot, strike, time_to_expiry,
                                                 volatility, option_type, risk_free_rate),
            'delta': self.calculate_delta(spot, strike, time_to_expiry,
                                          volatility, option_type, risk_free_rate),
            'gamma': self.calculate_gamma(spot, strike, time_to_expiry,
                                          volatility, risk_free_rate),
            'theta': self.calculate_theta(spot, strike, time_to_expiry,
                                          volatility, option_type, risk_free_rate),
            'vega': self.calculate_vega(spot, strike, time_to_expiry,
                                        volatility, risk_free_rate)
        }

    def find_strike_by_delta(
        self,
        spot: float,
        time_to_expiry: float,
        volatility: float,
        target_delta: float,
        available_strikes: List[float],
        option_type: str = 'CE'
    ) -> Tuple[float, float]:
        """
        Find the strike price closest to target delta from available strikes.

        This is key for covered call strategy where we select strikes by delta
        (e.g., sell calls with delta 0.30 or 0.40)

        Args:
            spot: Current spot price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            target_delta: Target delta (e.g., 0.30 for 30 delta call)
            available_strikes: List of available strike prices
            option_type: 'CE' for call or 'PE' for put

        Returns:
            Tuple of (best_strike, actual_delta)
        """
        best_strike = None
        best_delta = None
        min_diff = float('inf')

        for strike in available_strikes:
            delta = self.calculate_delta(
                spot, strike, time_to_expiry, volatility, option_type
            )

            # For calls, delta is positive (0 to 1)
            # For puts, delta is negative (-1 to 0), so we compare absolute values
            if option_type == 'PE':
                diff = abs(abs(delta) - abs(target_delta))
            else:
                diff = abs(delta - target_delta)

            if diff < min_diff:
                min_diff = diff
                best_strike = strike
                best_delta = delta

        return best_strike, best_delta

    def find_strike_by_otm_percent(
        self,
        spot: float,
        otm_percent: float,
        available_strikes: List[float],
        option_type: str = 'CE'
    ) -> float:
        """
        Find the strike price closest to a target OTM percentage.

        Args:
            spot: Current spot price
            otm_percent: Target OTM percentage (e.g., 0.02 for 2% OTM)
            available_strikes: List of available strike prices
            option_type: 'CE' for call (OTM = above spot) or 'PE' for put (OTM = below spot)

        Returns:
            Best matching strike price
        """
        if option_type == 'CE':
            target_strike = spot * (1 + otm_percent)
        else:  # PE
            target_strike = spot * (1 - otm_percent)

        # Find closest available strike
        best_strike = min(available_strikes, key=lambda x: abs(x - target_strike))
        return best_strike

    def calculate_covered_call_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        lot_size: int = 1
    ) -> Dict[str, float]:
        """
        Calculate net Greeks for a covered call position.

        Covered call = Long stock + Short call

        Args:
            spot: Current spot price
            strike: Call strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            lot_size: Number of shares/contracts

        Returns:
            Dictionary with position Greeks
        """
        # Get call Greeks
        call_greeks = self.calculate_all_greeks(
            spot, strike, time_to_expiry, volatility, 'CE'
        )

        # Stock has delta = 1, all other Greeks = 0
        # Short call means we subtract (negative position)
        return {
            'delta': (1 - call_greeks['delta']) * lot_size,  # Stock delta - call delta
            'gamma': -call_greeks['gamma'] * lot_size,  # Short gamma
            'theta': -call_greeks['theta'] * lot_size,  # Positive theta (we collect time decay)
            'vega': -call_greeks['vega'] * lot_size,  # Short vega
            'call_price': call_greeks['price'],
            'premium_collected': call_greeks['price'] * lot_size
        }


def example_covered_call():
    """Example of calculating Greeks for a covered call position"""

    print("=" * 70)
    print("COVERED CALL GREEKS EXAMPLE".center(70))
    print("=" * 70)
    print()

    # Initialize calculator
    calc = GreeksCalculator(risk_free_rate=0.07)

    # Position setup - typical covered call on RELIANCE
    spot = 2500.0  # RELIANCE spot price
    volatility = 0.20  # 20% IV
    time_to_expiry = 30 / 365  # 30 days
    lot_size = 250  # RELIANCE lot size

    # Available strikes (typical NSE strikes)
    available_strikes = [2400, 2450, 2500, 2550, 2600, 2650, 2700]

    print("Position Setup:")
    print(f"  Underlying: RELIANCE")
    print(f"  Spot Price: Rs {spot:.2f}")
    print(f"  Volatility: {volatility * 100:.1f}%")
    print(f"  Days to Expiry: {time_to_expiry * 365:.0f}")
    print(f"  Lot Size: {lot_size}")
    print()

    # Find strike by delta (0.30 delta call)
    target_delta = 0.30
    strike, actual_delta = calc.find_strike_by_delta(
        spot, time_to_expiry, volatility, target_delta, available_strikes
    )

    print(f"Strike Selection (Target Delta: {target_delta}):")
    print(f"  Selected Strike: {strike}")
    print(f"  Actual Delta: {actual_delta:.4f}")
    print()

    # Calculate covered call Greeks
    position_greeks = calc.calculate_covered_call_greeks(
        spot, strike, time_to_expiry, volatility, lot_size
    )

    print("=" * 70)
    print("COVERED CALL POSITION GREEKS")
    print("=" * 70)
    print()
    print(f"Net Delta:    {position_greeks['delta']:>10.2f}  (Reduced upside exposure)")
    print(f"Net Gamma:    {position_greeks['gamma']:>10.4f}  (Short gamma)")
    print(f"Net Theta:    {position_greeks['theta']:>10.2f}  (Daily time decay collected)")
    print(f"Net Vega:     {position_greeks['vega']:>10.2f}  (Short volatility)")
    print()
    print(f"Call Premium: Rs {position_greeks['call_price']:>8.2f} per share")
    print(f"Total Premium: Rs {position_greeks['premium_collected']:>8.2f}")
    print()

    # Calculate potential outcomes
    premium_percent = (position_greeks['call_price'] / spot) * 100
    max_profit = (strike - spot + position_greeks['call_price']) * lot_size

    print("=" * 70)
    print("STRATEGY ANALYSIS")
    print("=" * 70)
    print()
    print(f"Premium Yield: {premium_percent:.2f}% of stock price")
    print(f"Max Profit (if assigned): Rs {max_profit:.2f}")
    print(f"Breakeven: Rs {spot - position_greeks['call_price']:.2f}")


if __name__ == "__main__":
    example_covered_call()
