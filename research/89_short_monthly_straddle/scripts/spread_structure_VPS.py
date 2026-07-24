"""
Spread Structure Generator
============================
Given a directional signal (long or short, with entry price + ATR), generate
a debit-spread trade structure: long strike, short strike, expected debit,
max profit, breakeven.

For backtest / paper trading, we generate the LOGICAL structure without
needing live option premiums. When integrated with Kite for live, fetch
the actual chain to refine.

Strategies:
  bull_call_spread — for long signals (debit spread)
  bear_put_spread  — for short signals (debit spread)

Key inputs:
  spot_price: current stock price (signal close)
  atr: 14-day ATR (drives strike spacing for stocks where pct doesn't work)
  target_pct: profit target as % of entry (default 0.25 = +25%)
  spread_width_pct: width of the spread as % of spot (default 0.10 = 10%)
                    A "10% wide" spread on a Rs 1000 stock = Rs 100 between strikes.
                    For F&O index, would use Rs steps.
  iv_assumption: implied vol assumption for theoretical premium (default 0.30 = 30% annualized)
  dte: days to expiry (default 30)
  risk_free_rate: 0.06 (6% — Indian risk-free rate proxy)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional


# ---- Strike rounding helpers (Indian stock options use specific intervals) ----

def round_to_strike(price: float, interval: float = None) -> float:
    """Round to nearest valid strike. Default interval is auto-detected from price level."""
    if interval is None:
        # Auto-detect strike interval (NSE convention)
        if price < 50:
            interval = 1
        elif price < 100:
            interval = 2.5
        elif price < 250:
            interval = 5
        elif price < 500:
            interval = 10
        elif price < 1000:
            interval = 20
        elif price < 2500:
            interval = 50
        else:
            interval = 100
    return round(price / interval) * interval


# ---- Black-Scholes for theoretical premium (used when live chain unavailable) ----

def _norm_cdf(x: float) -> float:
    """Cumulative normal distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_call(spot: float, strike: float, dte_days: int,
                       iv: float, r: float = 0.06) -> float:
    """Theoretical call premium using Black-Scholes."""
    if dte_days <= 0:
        return max(0.0, spot - strike)
    T = dte_days / 365.0
    if iv <= 0:
        return max(0.0, spot - strike)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T
    return spot * _norm_cdf(d1) - strike * math.exp(-r * T) * _norm_cdf(d2)


def black_scholes_put(spot: float, strike: float, dte_days: int,
                      iv: float, r: float = 0.06) -> float:
    """Theoretical put premium using Black-Scholes."""
    if dte_days <= 0:
        return max(0.0, strike - spot)
    T = dte_days / 365.0
    if iv <= 0:
        return max(0.0, strike - spot)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T
    return strike * math.exp(-r * T) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


# ---- Spread structure dataclass ----

@dataclass
class SpreadStructure:
    """A debit spread trade structure (logical — strikes and theoretical economics)."""
    strategy: str               # 'bull_call_spread' | 'bear_put_spread'
    spot_at_signal: float
    long_strike: float
    short_strike: float
    spread_width: float         # short - long (for bull call) or long - short (for bear put), absolute
    debit_estimate: float       # theoretical net debit using BS
    max_profit: float           # spread_width - debit
    max_loss: float             # debit (downside is capped)
    breakeven: float            # long_strike + debit (bull call) or long_strike - debit (bear put)
    risk_reward: float          # max_profit / max_loss
    target_pct_at_short_strike: float  # how much spot needs to move to hit max profit (% from spot)
    breakeven_pct: float        # how much spot needs to move to break even (% from spot)
    dte: int
    iv_assumption: float
    notes: str = ''

    def to_dict(self) -> dict:
        d = asdict(self)
        # Round numeric fields for clean display
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 2)
        return d


# ---- Spread builders ----

def build_bull_call_spread(
    spot_price: float,
    atr: Optional[float] = None,
    target_pct: float = 0.25,
    iv: float = 0.30,
    dte: int = 30,
    long_strike_offset_pct: float = 0.0,   # 0 = ATM long; -0.02 = 2% ITM long
    short_strike_target_pct: Optional[float] = None,  # default = target_pct (matches +25% target rule)
) -> SpreadStructure:
    """
    Build a bull call spread for a long-direction signal.

    Default: long ATM call + short call at +25% strike → matches the
    research/17-21 +25% target rule. Spread width = spread between strikes;
    debit ≤ width; max profit hits when spot ≥ short strike at expiry.

    Risk: net debit (capped). Reward: width − debit.
    """
    long_strike_raw = spot_price * (1 + long_strike_offset_pct)
    long_strike = round_to_strike(long_strike_raw)

    if short_strike_target_pct is None:
        short_strike_target_pct = target_pct
    short_strike_raw = spot_price * (1 + short_strike_target_pct)
    short_strike = round_to_strike(short_strike_raw)

    # Compute theoretical premia
    long_premium = black_scholes_call(spot_price, long_strike, dte, iv)
    short_premium = black_scholes_call(spot_price, short_strike, dte, iv)
    debit = long_premium - short_premium
    spread_width = short_strike - long_strike
    max_profit = max(0.0, spread_width - debit)
    max_loss = debit
    breakeven = long_strike + debit
    rr = max_profit / max_loss if max_loss > 0 else 0
    breakeven_pct = (breakeven / spot_price) - 1
    target_pct_actual = (short_strike / spot_price) - 1

    return SpreadStructure(
        strategy='bull_call_spread',
        spot_at_signal=spot_price,
        long_strike=long_strike,
        short_strike=short_strike,
        spread_width=spread_width,
        debit_estimate=debit,
        max_profit=max_profit,
        max_loss=max_loss,
        breakeven=breakeven,
        risk_reward=rr,
        target_pct_at_short_strike=target_pct_actual,
        breakeven_pct=breakeven_pct,
        dte=dte,
        iv_assumption=iv,
        notes=f'BUY {long_strike} CE, SELL {short_strike} CE — debit ~Rs {debit:.1f}',
    )


def build_bear_put_spread(
    spot_price: float,
    atr: Optional[float] = None,
    target_pct: float = 0.25,            # Target as POSITIVE % (-25% = stock falls 25%)
    iv: float = 0.30,
    dte: int = 30,
    long_strike_offset_pct: float = 0.0,
    short_strike_target_pct: Optional[float] = None,
) -> SpreadStructure:
    """
    Build a bear put spread for a short-direction signal.

    Default: long ATM put + short put at −25% strike. Mirror of bull call.
    Risk: net debit. Reward: width − debit. Hits max when spot ≤ short
    strike at expiry.
    """
    long_strike_raw = spot_price * (1 - long_strike_offset_pct)
    long_strike = round_to_strike(long_strike_raw)

    if short_strike_target_pct is None:
        short_strike_target_pct = target_pct
    short_strike_raw = spot_price * (1 - short_strike_target_pct)
    short_strike = round_to_strike(short_strike_raw)

    long_premium = black_scholes_put(spot_price, long_strike, dte, iv)
    short_premium = black_scholes_put(spot_price, short_strike, dte, iv)
    debit = long_premium - short_premium
    spread_width = long_strike - short_strike    # long is HIGHER for bear put
    max_profit = max(0.0, spread_width - debit)
    max_loss = debit
    breakeven = long_strike - debit
    rr = max_profit / max_loss if max_loss > 0 else 0
    breakeven_pct = (breakeven / spot_price) - 1
    target_pct_actual = (short_strike / spot_price) - 1

    return SpreadStructure(
        strategy='bear_put_spread',
        spot_at_signal=spot_price,
        long_strike=long_strike,
        short_strike=short_strike,
        spread_width=spread_width,
        debit_estimate=debit,
        max_profit=max_profit,
        max_loss=max_loss,
        breakeven=breakeven,
        risk_reward=rr,
        target_pct_at_short_strike=target_pct_actual,
        breakeven_pct=breakeven_pct,
        dte=dte,
        iv_assumption=iv,
        notes=f'BUY {long_strike} PE, SELL {short_strike} PE — debit ~Rs {debit:.1f}',
    )


# ---- Convenience dispatcher ----

def build_spread_for_signal(
    direction: str,            # 'LONG' or 'SHORT'
    spot_price: float,
    atr: Optional[float] = None,
    target_pct: float = 0.25,
    iv: float = 0.30,
    dte: int = 30,
) -> SpreadStructure:
    """Auto-pick spread type from direction."""
    if direction == 'LONG':
        return build_bull_call_spread(spot_price, atr, target_pct, iv, dte)
    elif direction == 'SHORT':
        return build_bear_put_spread(spot_price, atr, target_pct, iv, dte)
    else:
        raise ValueError(f'Unknown direction: {direction}')


# ---- Adjustment playbook ----

ADJUSTMENT_PLAYBOOK = [
    {
        'situation': 'Stock stalls between breakeven and target',
        'adjustment': 'Roll short leg up',
        'detail': 'Close current short, open higher strike. Re-collects premium, widens max profit.',
        'when': 'Spot is between breakeven and short strike, but momentum stalls (e.g., RSI < 50 in uptrend)',
    },
    {
        'situation': 'Stock approaches target before expiry',
        'adjustment': 'Convert to butterfly',
        'detail': 'Sell another call/put at a higher/lower strike. Locks gains, captures pin.',
        'when': 'Spot reaches 80% of way to short strike with 14+ DTE remaining',
    },
    {
        'situation': 'Stock breaks down below long strike',
        'adjustment': 'Convert to back spread',
        'detail': 'Sell extra short at long strike to fund a far OTM long. Limits loss, retains reversal upside.',
        'when': 'Spot reaches breakeven on adverse side; trend reversal flag',
    },
    {
        'situation': 'Volatility spikes mid-trade',
        'adjustment': 'Roll out one cycle',
        'detail': 'Close existing spread, reopen 30-DTE further out. IV-crush hedge.',
        'when': 'IV percentile jumps >75 pre-event (earnings, RBI, budget)',
    },
    {
        'situation': 'Partial profit + reversal threat',
        'adjustment': 'Sell short put below long strike',
        'detail': 'Free hedge if continuation; small loss if reversal.',
        'when': '50%+ of max profit captured; trend signal weakening',
    },
]


if __name__ == '__main__':
    # Quick sanity check
    print('=== Bull Call Spread on Rs 1000 stock (ATM long, +25% short, 30 DTE, 30% IV) ===')
    bull = build_bull_call_spread(spot_price=1000, atr=20, target_pct=0.25, iv=0.30, dte=30)
    for k, v in bull.to_dict().items():
        print(f'  {k}: {v}')
    print()
    print('=== Bear Put Spread on Rs 1000 stock (ATM long, -25% short, 30 DTE, 30% IV) ===')
    bear = build_bear_put_spread(spot_price=1000, target_pct=0.25, iv=0.30, dte=30)
    for k, v in bear.to_dict().items():
        print(f'  {k}: {v}')
